"""Diagnostic: time each stage of the rewrite pipeline on a chosen model.

Usage:
    PYTHONPATH=. python -u test/diag_rewrite.py Resnet18 4
    PYTHONPATH=. python -u test/diag_rewrite.py Resnet50 4
"""

import sys
import time
from typing import Any

import torch

from benchmarks import Experiment
from graph_prof import GraphProfiler
from graph_tracer import compile
from mu_two_core import simulate
from mu_two_rewrite import rewrite_recomputes
from mu_two_scheduler import greedy_recompute


def _flatten(x, out=None):
    if out is None:
        out = []
    if isinstance(x, (list, tuple)):
        for e in x:
            _flatten(e, out)
    elif isinstance(x, torch.Tensor):
        out.append(x)
    return out


def _snapshot(args):
    return [a.detach().clone() if isinstance(a, torch.Tensor) else a for a in args]


def _restore(args, snap):
    with torch.no_grad():
        for a, s in zip(args, snap):
            if isinstance(a, torch.Tensor) and isinstance(s, torch.Tensor):
                a.copy_(s)


def main():
    model_name = sys.argv[1]
    bs = int(sys.argv[2])
    print(f"[{model_name} bs={bs}] starting", flush=True)

    torch.cuda.empty_cache()
    exp = Experiment(model_name, batch_size=bs)
    exp.init_opt_states()

    captured = {}

    def xform(gm, args: Any):
        n_nodes = sum(1 for _ in gm.graph.nodes)
        print(f"  graph nodes={n_nodes}", flush=True)

        t0 = time.time()
        prof = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(2):
                prof.run(*args)
            prof.reset_stats()
            for _ in range(3):
                prof.run(*args)
        prof.aggregate_stats()
        print(f"  profile: {time.time() - t0:.1f}s", flush=True)

        snap = _snapshot(args)
        with torch.no_grad():
            baseline = gm(*args)
        baseline_flat = [t.detach().clone() for t in _flatten(baseline)]
        print(f"  baseline run done, {len(baseline_flat)} tensors", flush=True)

        t0 = time.time()
        baseline_peak = max(simulate(prof, recomps={}))
        print(
            f"  simulate baseline: {time.time() - t0:.1f}s "
            f"peak_mb={baseline_peak/1e6:.1f} "
            f"acts={len(prof.activation_nodes)}",
            flush=True,
        )

        budget = int(baseline_peak * 0.7)

        t0 = time.time()
        recomps, reached = greedy_recompute(prof, budget=budget)
        print(
            f"  greedy: {time.time() - t0:.1f}s "
            f"picks={len(recomps)} reached={reached}",
            flush=True,
        )

        final_peak = max(simulate(prof, recomps))
        print(
            f"  final_mb={final_peak/1e6:.1f} (budget {budget/1e6:.1f})",
            flush=True,
        )

        # Show top picks for sanity check
        picks_sorted = sorted(
            recomps.values(), key=lambda m: -m.size
        )[:8]
        print("  top-8 picks by size:", flush=True)
        for m in picks_sorted:
            print(
                f"    {m.node.name:>30s} size={m.size/1e6:>7.2f}MB "
                f"recomp_t={m.recomp_time:.4f} cnt={m.recomp_cnt} "
                f"ratio={m.recomp_ratio:.2e}",
                flush=True,
            )
        max_cnt = max((m.recomp_cnt for m in recomps.values()), default=0)
        print(f"  max recomp_cnt={max_cnt}", flush=True)

        t0 = time.time()
        nodes_before = sum(1 for _ in gm.graph.nodes)
        gm2 = rewrite_recomputes(gm, prof, recomps)
        nodes_after = sum(1 for _ in gm2.graph.nodes)
        print(
            f"  rewrite: {time.time() - t0:.1f}s "
            f"nodes {nodes_before} -> {nodes_after} "
            f"(+{nodes_after - nodes_before})",
            flush=True,
        )

        _restore(args, snap)
        with torch.no_grad():
            new = gm2(*args)
        new_flat = _flatten(new)

        # Numerical check
        max_abs = 0.0
        max_rel = 0.0
        for i, (b, nn) in enumerate(zip(baseline_flat, new_flat)):
            d = (b - nn).abs()
            denom = b.abs().clamp(min=1e-9)
            ma = d.max().item()
            mr = (d / denom).max().item()
            if ma > max_abs:
                max_abs = ma
            if mr > max_rel:
                max_rel = mr
        print(
            f"  numerical: max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e}",
            flush=True,
        )
        captured["ok"] = max_abs < 1e-3
        return gm2

    compiled = compile(exp.train_step, xform)
    compiled(exp.model, exp.optimizer, exp.example_inputs)
    print(f"[{model_name} bs={bs}] done ok={captured.get('ok')}", flush=True)


if __name__ == "__main__":
    main()
