"""Core types for the μ-TWO recomputation scheduler.

Step 1: ActivationMeta dataclass + Status enum (per-candidate state container).
Step 2: build_candidates — populate one ActivationMeta per activation by BFS
through forward-region inputs, separating boundary sources from interior ops.
Algorithms E/F (Steps 5–6) will mutate these in place during the greedy loop.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import torch.fx as fx

from graph_prof import OP, GraphProfiler


class Status(Enum):
    RETAINED = "retained"
    RECOMPUTE = "recompute"
    SWAP = "swap"


@dataclass(eq=False)
class ActivationMeta:
    node: fx.Node
    size: int
    last_fwd_idx: int
    first_bwd_idx: int

    recomp_srcs: Set[fx.Node] = field(default_factory=set)
    recomp_subgraph: Tuple[fx.Node, ...] = field(default_factory=tuple)
    recomp_time: float = 0.0
    recomp_cnt: int = 1
    total_recomp_time: float = 0.0

    status: Status = Status.RETAINED

    @property
    def recomp_ratio(self) -> float:
        if self.total_recomp_time <= 0.0:
            return math.inf
        return self.size / self.total_recomp_time


def build_candidates(prof: GraphProfiler) -> Dict[fx.Node, ActivationMeta]:
    """Construct one ActivationMeta per activation in *prof*.

    For each act in prof.activation_nodes, BFS backward through forward-region
    inputs. The traversal separates two boundary sets:

      * recomp_subgraph — interior fwd-region ops the rewriter will re-execute.
      * recomp_srcs     — leaves the BFS halts on (PLACEHOLDER inputs or other
                          activations that are still resident and supply the
                          inputs to the cloned subgraph).

    These are deliberately kept distinct: recomp_subgraph is what gets cloned,
    recomp_srcs is what those clones bind to. Algorithm F (Step 6) later
    propagates srcs and incrementally adjusts recomp_time when an activation
    that is itself a source is picked.
    """
    metas: Dict[fx.Node, ActivationMeta] = {}

    # Iterate activations in topological order so the resulting dict has a
    # deterministic insertion order across runs (sets don't preserve it).
    # The greedy scheduler's argmax doesn't care, but downstream consumers
    # and tests do.
    for act in sorted(prof.activation_nodes, key=lambda n: prof.order_index[n]):
        # Defensive check. The GraphProfiler classifier requires a bwd user, so this
        # should always be present for any node in activation_nodes.
        first_bwd = prof.first_backward_access.get(act)
        if first_bwd is None:
            continue

        # Skip leaf-tensor activations (no input tensor deps — e.g. aten.arange
        # with all-scalar args, or aten.zeros). recomp_srcs would end up empty,
        # the rewriter has nothing to bind a clone to, and these are tiny
        # constants (~KB) so the savings aren't worth the special case.
        if not act.all_input_nodes:
            continue

        last_fwd_idx = prof.last_forward_use_idx(act)

        # `act` itself is the final op of the recomputation: regenerating it
        # requires re-executing every interior input AND running act's own op
        # to produce the output tensor. So act is included in recomp_subgraph
        # and contributes both its runtime (to recomp_time) and its output size
        # (to the simulator's window_extra). Excluding it under-estimates both,
        # which silently corrupts greedy ranking and peak prediction.
        visited: Set[fx.Node] = {act}
        stack = [act]
        interior: list[fx.Node] = [act]
        srcs: Set[fx.Node] = set()

        while stack:
            node = stack.pop()
            for inp in node.all_input_nodes:
                # Skip if visited
                if inp in visited:
                    continue
                visited.add(inp)

                # Placeholders and other activations are sources
                if inp.op == OP.PLACEHOLDER or inp in prof.activation_nodes:
                    srcs.add(inp)
                    continue

                # Keep iterating
                interior.append(inp)
                stack.append(inp)

        subgraph = tuple(sorted(interior, key=lambda n: prof.order_index[n]))
        recomp_time = sum(prof.avg_runtimes.get(n, 0.0) for n in subgraph)

        metas[act] = ActivationMeta(
            node=act,
            size=int(prof.avg_output_sizes.get(act, 0)),
            last_fwd_idx=last_fwd_idx,
            first_bwd_idx=prof.order_index[first_bwd],
            recomp_srcs=srcs,
            recomp_subgraph=subgraph,
            recomp_time=recomp_time,
            recomp_cnt=1,
            total_recomp_time=recomp_time,
        )

    return metas


# ---------------------------------------------------------------------------
# Algorithm E — propagate sources into prior recomps when a new t is picked.
# ---------------------------------------------------------------------------


def update_existing_recomps(
    t: ActivationMeta,
    recomps: Dict[fx.Node, ActivationMeta],
) -> None:
    """When `t` is freshly chosen for recompute, update prior picks in-place.

    For each prior recompute R whose source set contains t.node, t's own sources
    flow upstream into R (R must now reach further back to regenerate itself,
    since t is no longer memory-resident at R's recompute window). R also
    absorbs t's recompute time, and t.recomp_cnt bumps because t will be
    re-executed once per dependent R as well as for its own window.

    Caller convention: this runs *before* recomps[t.node] = t. Algorithm F
    (remaining-candidate update) is a separate function, not invoked here.
    `t.total_recomp_time` is also refreshed by this function so the caller
    doesn't have to remember.

    Note (intentional approximation, per TENTATIVE_PLAN §3): we mutate the
    scalar recomp_time and the recomp_srcs set, but we do not rebuild
    recomp_subgraph. The simulator's window_extra estimate uses recomp_subgraph
    and will undercount intermediate liveness for cascaded picks. Materializing
    the full transitive subgraph is deferred to the rewriter (which runs once
    after all picks are made).
    """
    for R in recomps.values():
        if t.node in R.recomp_srcs:
            R.recomp_srcs.discard(t.node)
            R.recomp_srcs.update(t.recomp_srcs)
            R.recomp_time += t.recomp_time
            t.recomp_cnt += 1
        R.total_recomp_time = R.recomp_cnt * R.recomp_time
    t.total_recomp_time = t.recomp_cnt * t.recomp_time


# ---------------------------------------------------------------------------
# Algorithm F — propagate to remaining candidates when a new t is picked.
# ---------------------------------------------------------------------------


def update_remaining_candidates(
    t: ActivationMeta,
    candidates: Dict[fx.Node, ActivationMeta],
) -> None:
    """When `t` is freshly chosen, refresh remaining candidates' costs.

    Two arms (mutually exclusive in a DAG; no candidate satisfies both at once):

    Arm 1 — t was a source of c:
      c must now reach further back to regenerate (t is no longer resident at
      c's window). Propagate t's sources into c's, accumulate t's recomp_time,
      and refresh total_recomp_time to keep the cnt × time invariant.

    Arm 2 — c is a source of t:
      If c is later picked, c will execute once per t-window. The cost estimate
      reflects this directly: total_recomp_time = t.recomp_cnt × c.recomp_time.
      Note this overrides total_recomp_time without bumping c.recomp_cnt — the
      cnt belongs to c's own pick history, not t's. Same convention as the
      paper and the reference impl; intentionally breaks the cnt × time
      invariant for c.

    Caller convention: invoke AFTER `update_existing_recomps(t, recomps)` and
    AFTER removing t from `candidates`. By that point t.recomp_cnt has been
    bumped by Algorithm E if t was a source of any prior recomp, so the
    multiplier in arm 2 is the post-E value.
    """
    # Guard the most common caller-ordering bug: forgetting to remove t from
    # candidates before this call. Without this, arm 2 would fire on t itself
    # (since t.node ∈ t.recomp_srcs is false, but other consistency invariants
    # would silently break).
    assert t.node not in candidates, (
        "t must be removed from candidates before update_remaining_candidates"
    )
    for c in candidates.values():
        if t.node in c.recomp_srcs:
            c.recomp_srcs.discard(t.node)
            c.recomp_srcs.update(t.recomp_srcs)
            c.recomp_time += t.recomp_time
            c.total_recomp_time = c.recomp_cnt * c.recomp_time
        elif c.node in t.recomp_srcs:
            c.total_recomp_time = t.recomp_cnt * c.recomp_time


# ---------------------------------------------------------------------------
# Memory simulator (Algorithm G)
# ---------------------------------------------------------------------------


def _compute_last_users(prof: GraphProfiler) -> Dict[fx.Node, int]:
    """Order index of each node's last consumer.

    Tensors with no users (e.g., unused placeholders) are alive until end of
    execution — we return n-1 as a fallback. Mirrors the convention used by
    plot_memory_stacked_timeline in graph_prof.py.
    """
    n = len(prof.node_order)
    last: Dict[fx.Node, int] = {}
    for node in prof.node_order:
        if node.op == OP.OUTPUT:
            continue
        lui = -1
        for u in node.users:
            uid = prof.order_index.get(u, -1)
            if uid > lui:
                lui = uid
        last[node] = lui if lui >= 0 else n - 1
    return last


def simulate(
    prof: GraphProfiler,
    recomps: Dict[fx.Node, ActivationMeta],
    swaps: Optional[Dict[fx.Node, ActivationMeta]] = None,
) -> List[int]:
    """Per-node alive-bytes timeline reflecting the given recomp/swap decisions.

    Phase A — base sweep over prof.node_order, summing avg_output_sizes for
    every tensor still alive at each index. Equivalent to summing the per-type
    stacks in plot_memory_stacked_timeline.

    Phase B — for each (act, meta) in recomps, the activation is freed at
    meta.last_fwd_idx (instead of its baseline last_user) and its recompute
    window is charged immediately before meta.first_bwd_idx. We use the
    conservative whole-window approximation: charge the sum of recomp_subgraph
    output sizes against the slot just before first_bwd_idx. This over-estimates
    intermediate liveness inside the window — acceptable because it never
    under-estimates the post-rewrite peak (anti-shortcut #3 in TENTATIVE_PLAN).

    Swap support is deferred to later steps; non-empty `swaps` raises.
    """
    if swaps:
        raise NotImplementedError("")

    n = len(prof.node_order)
    alive: List[int] = [0] * n
    last_user = _compute_last_users(prof)

    # Phase A: baseline sweep. Track (last_user_idx, size) for each live tensor.
    live: List[Tuple[int, int]] = []
    for i, node in enumerate(prof.node_order):
        if i > 0:
            alive[i] = alive[i - 1]
        if node.op == OP.OUTPUT:
            continue

        size = int(prof.avg_output_sizes.get(node, 0))
        if size > 0:
            alive[i] += size
            live.append((last_user[node], size))

        # Drop tensors whose last user fired before this index.
        still_live: List[Tuple[int, int]] = []
        for lui, asize in live:
            if lui < i:
                alive[i] -= asize
            else:
                still_live.append((lui, asize))
        live = still_live

    # Phase B: apply recompute decisions.
    # An activation `act` chosen for recompute is freed at last_fwd_idx instead
    # of at its baseline last user (which sits in bwd at first_bwd_idx). So in
    # the gap (last_fwd_idx, first_bwd_idx), we subtract its size.
    for meta in recomps.values():
        lo, hi = meta.last_fwd_idx, meta.first_bwd_idx
        for k in range(lo + 1, hi):
            alive[k] -= meta.size

        # Recomputing a node requires re-executing its subgraph, which takes more
        # memory for the intermediate tensors produced along the way.
        window_extra = sum(
            int(prof.avg_output_sizes.get(n_, 0)) for n_ in meta.recomp_subgraph
        )
        if window_extra > 0 and 0 <= hi - 1 < n:
            alive[hi - 1] += window_extra

    return alive
