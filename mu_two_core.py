"""Core types for the μ-TWO recomputation scheduler.

Step 1 of Phase 2: scaffolding only. Defines the per-candidate state container
that later steps (candidate construction, simulator, Algorithms E/F, greedy
loop) will read and mutate. No profiler integration or scheduling logic yet.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Set, Tuple

import torch.fx as fx


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
