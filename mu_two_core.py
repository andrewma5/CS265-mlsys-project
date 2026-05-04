"""Core types for the μ-TWO recomputation scheduler.

Step 1: ActivationMeta dataclass + Status enum (per-candidate state container).
Step 2: build_candidates — populate one ActivationMeta per activation by BFS
through forward-region inputs, separating boundary sources from interior ops.
Algorithms E/F (Steps 5–6) will mutate these in place during the greedy loop.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, Tuple

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
