"""μ-TWO greedy scheduler — Step 7 of TENTATIVE_PLAN.md.

Composes Algorithms B (build_candidates), E (update_existing_recomps),
F (update_remaining_candidates), and G (simulate) into the recompute-only
greedy pick loop.
"""

from typing import Dict, Tuple

import torch.fx as fx

from graph_prof import GraphProfiler
from mu_two_core import (
    ActivationMeta,
    Status,
    build_candidates,
    simulate,
    update_existing_recomps,
    update_remaining_candidates,
)


def greedy_recompute(
    prof: GraphProfiler,
    budget: int,
) -> Tuple[Dict[fx.Node, ActivationMeta], bool]:
    """Greedy μ-TWO recompute scheduler.

    Iteratively picks the candidate with the highest `recomp_ratio` (= size /
    total_recomp_time), runs Algorithm E + F to refresh post-pick costs, and
    re-simulates from scratch to get the new peak. Stops when the simulated
    peak ≤ budget (success) or candidates are exhausted (infeasible).

    Returns:
        recomps: Dict[fx.Node, ActivationMeta] — picks in insertion order
            (Python 3.7+ dict preserves it). Each meta carries the post-cascade
            recomp_srcs / recomp_subgraph / recomp_cnt / recomp_time, all of
            which the rewriter (Step 8) consumes directly.
        reached_budget: bool — True iff the final simulated peak ≤ budget.

    Deviation from TENTATIVE_PLAN §6 spec: the spec returns Set[Node]. We
    return Dict[Node, ActivationMeta] because the rewriter needs each meta's
    full post-cascade state (not just the node identity) and we already
    computed it. Caller can do `recomps.keys()` for a set.

    Call-order contract (load-bearing — see mu_two_core docstrings):
      1. update_existing_recomps(t, recomps)   # E bumps t.recomp_cnt
      2. recomps[t.node] = t                    # then insert t into recomps
      3. del candidates[t.node]                 # remove t from candidates
      4. update_remaining_candidates(t, ...)    # F uses post-E t.recomp_cnt
      5. simulate(...)                           # re-derive peak

    Note: this loop has no "skip-if-doesn't-help" guard. Per the paper's
    greedy heuristic, every iteration picks argmax(ratio) regardless of
    whether the new peak is lower than before. Pathological fixtures (where
    the recompute window is bigger than the saved gap) can produce picks that
    don't lower peak; the loop will keep going until candidates are exhausted.
    The infeasible-budget test exercises this path explicitly.
    """
    candidates = build_candidates(prof)
    recomps: Dict[fx.Node, ActivationMeta] = {}

    peak = max(simulate(prof, recomps))

    while peak > budget and candidates:
        t = max(candidates.values(), key=lambda c: c.recomp_ratio)

        update_existing_recomps(t, recomps)
        recomps[t.node] = t
        t.status = Status.RECOMPUTE
        del candidates[t.node]
        update_remaining_candidates(t, candidates)

        peak = max(simulate(prof, recomps))

    return recomps, peak <= budget
