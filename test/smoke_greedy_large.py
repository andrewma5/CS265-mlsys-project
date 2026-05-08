"""One-shot diagnostic: profile each large model once, run greedy_recompute,
print baseline/final/picks. Run with:

    python test/smoke_greedy_large.py
"""

from typing import Any

import torch

from benchmarks import Experiment
from graph_prof import GraphProfiler
from graph_tracer import compile
from mu_two_core import build_candidates, simulate
from mu_two_scheduler import greedy_recompute


# Conservative batch sizes — enough to stress the graph but small enough that
# the profile + scheduler loop stays under a couple minutes per model.
TARGETS = [
    ("Resnet50", 16),
    ("Resnet152", 8),
    ("Transformer", 64),
]


def profile_model(model_name: str, batch_size: int) -> GraphProfiler:
    torch.cuda.empty_cache()
    exp = Experiment(model_name, batch_size=batch_size)
    exp.init_opt_states()

    captured: dict = {}

    def capture(gm, args: Any):
        prof = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(2):
                prof.run(*args)
            prof.reset_stats()
            for _ in range(3):
                prof.run(*args)
            prof.aggregate_stats()
        captured["prof"] = prof
        return gm

    compiled_fn = compile(exp.train_step, capture)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
    return captured["prof"]


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    for model_name, bs in TARGETS:
        print(f"\n[{model_name} bs={bs}] profiling...", flush=True)
        prof = profile_model(model_name, bs)
        n_nodes = len(prof.node_order)
        candidates = build_candidates(prof)

        baseline_peak = max(simulate(prof, recomps={}))
        budget = int(baseline_peak * 0.7)

        print(
            f"[{model_name} bs={bs}] nodes={n_nodes} "
            f"candidates={len(candidates)} "
            f"baseline_mb={baseline_peak / 1e6:.1f} "
            f"budget_mb={budget / 1e6:.1f} (70%)",
            flush=True,
        )

        recomps, reached = greedy_recompute(prof, budget=budget)
        final_peak = max(simulate(prof, recomps))

        print(
            f"[{model_name} bs={bs}] greedy: "
            f"final_mb={final_peak / 1e6:.1f} "
            f"reached={reached} "
            f"picks={len(recomps)}/{len(candidates)}",
            flush=True,
        )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
