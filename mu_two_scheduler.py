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
    """Greedy μ-TWO recompute scheduler with peak-aware candidate filtering.

    Iteratively picks argmax(recomp_ratio) over the subset of candidates whose
    liveness interval (last_fwd_idx, first_bwd_idx) contains the *current*
    simulated peak index, runs Algorithm E + F, and re-simulates. Stops when
    peak ≤ budget (success), candidates are exhausted, or no remaining
    candidate intersects the current peak — picking a non-intersecting
    candidate cannot lower peak (simulate's Phase B only modifies
    `[last_fwd_idx+1, first_bwd_idx-1]`), it only adds latency.

    Returns:
        recomps: Dict[fx.Node, ActivationMeta] — picks in insertion order.
            Each meta carries post-cascade recomp_srcs / recomp_subgraph /
            recomp_cnt / recomp_time, consumed directly by the rewriter.
        reached_budget: bool — True iff the final simulated peak ≤ budget.

    Call-order contract (load-bearing — see mu_two_core docstrings):
      1. update_existing_recomps(t, recomps)   # E bumps t.recomp_cnt
      2. recomps[t.node] = t                    # then insert t into recomps
      3. del candidates[t.node]                 # remove t from candidates
      4. update_remaining_candidates(t, ...)    # F uses post-E t.recomp_cnt
      5. simulate(...)                           # re-derive peak + peak_idx

    The peak-aware gate replaces the prior "exhaust all candidates on
    infeasibility" behavior: small-bs runs (param/optimizer floor dominates)
    now stop when activations no longer sit at peak, instead of picking
    every candidate and stalling the rewriter.
    """
    candidates = build_candidates(prof)
    recomps: Dict[fx.Node, ActivationMeta] = {}

    alive = simulate(prof, recomps)
    peak = max(alive)
    peak_idx = alive.index(peak)

    while peak > budget and candidates:
        eligible = [
            c for c in candidates.values()
            if c.last_fwd_idx < peak_idx < c.first_bwd_idx
        ]
        if not eligible:
            break
        t = max(eligible, key=lambda c: c.recomp_ratio)

        update_existing_recomps(t, recomps)
        recomps[t.node] = t
        t.status = Status.RECOMPUTE
        del candidates[t.node]
        update_remaining_candidates(t, candidates)

        alive = simulate(prof, recomps)
        peak = max(alive)
        peak_idx = alive.index(peak)

    return recomps, peak <= budget
