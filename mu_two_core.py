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

    for act in prof.activation_nodes:
        # Defensive check. The GraphProfiler classifier requires a bwd user, so this
        # should always be present for any node in activation_nodes.
        first_bwd = prof.first_backward_access.get(act)
        if first_bwd is None:
            continue

        last_fwd_idx = prof.last_forward_use_idx(act)

        visited: Set[fx.Node] = {act}
        stack = [act]
        interior: list[fx.Node] = []
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
