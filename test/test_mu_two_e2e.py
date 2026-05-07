"""End-to-end Phase 2 smoke tests (Step 9 of TENTATIVE_PLAN.md).

Exercises the full profiler -> scheduler -> rewriter -> re-profile pipeline
through `Experiment.graph_transformation` (the same entry point benchmarks.py
hits at the CLI). Three load-bearing assertions per model:

  1. allclose(baseline, rewritten) — semantics preserved.
  2. measured_peak <= predicted_peak * 1.10 — simulator is honest about
     recompute-intermediate live bytes (the central phase2-rebuild bug).
  3. measured_peak < baseline_peak — the rewrite actually freed memory
     (would catch a "scheduler picks but rewrite doesn't free" failure).

Snapshot/restore around args mirrors test_mu_two_rewrite.py: fused Adam
mutates params in place, and graph_transformation runs gm(*args) ~10 times
during its baseline + re-profile passes, so the test must reset state at
each comparison boundary.
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers (cloned from test_mu_two_rewrite.py — small enough not to factor)
# ---------------------------------------------------------------------------


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


def _assert_allclose(baseline_flat, new_flat, *, rtol, atol, label):
    assert len(baseline_flat) == len(new_flat), (
        f"{label}: tensor count mismatch: {len(baseline_flat)} vs {len(new_flat)}"
    )
    for i, (b, n) in enumerate(zip(baseline_flat, new_flat)):
        assert b.shape == n.shape, f"{label} output {i}: shape {b.shape} vs {n.shape}"
        if not torch.allclose(b, n, rtol=rtol, atol=atol):
            diff = (b - n).abs().max().item()
            raise AssertionError(
                f"{label} output {i}: max diff {diff} exceeds rtol={rtol}, atol={atol}"
            )


# ---------------------------------------------------------------------------
# Shared driver
# ---------------------------------------------------------------------------


def _run_e2e(model_name: str, batch_size: int, budget_fraction: float):
    """Drive Experiment(model_name, batch_size) through graph_transformation
    with mu_two_budget_fraction set, capturing baseline and rewritten outputs
    plus the peak-tracking attributes the integration point populates."""
    from benchmarks import Experiment
    from graph_tracer import compile

    torch.cuda.empty_cache()
    exp = Experiment(model_name, batch_size=batch_size)
    exp.init_opt_states()
    exp.mu_two_budget_fraction = budget_fraction

    captured = {}

    def xform(gm, args):
        # Snapshot once, before any mutation. We restore twice: once before
        # graph_transformation (so its internal profiling sees fresh args),
        # and again before the rewritten-gm comparison run.
        snap = _snapshot(args)

        with torch.no_grad():
            baseline = gm(*args)
        baseline_flat = [t.detach().clone() for t in _flatten(baseline)]

        _restore(args, snap)
        gm2 = exp.graph_transformation(gm, args)

        _restore(args, snap)
        with torch.no_grad():
            new = gm2(*args)
        new_flat = _flatten(new)

        captured["baseline"] = baseline_flat
        captured["new"] = new_flat
        return gm2

    compiled = compile(exp.train_step, xform)
    compiled(exp.model, exp.optimizer, exp.example_inputs)

    return exp, captured


def _assert_peak_gates(exp, model_name: str):
    """Common assertions: schedule produced picks, simulator honest, peak fell."""
    assert exp.recomp_picks is not None and len(exp.recomp_picks) >= 1, (
        f"{model_name}: scheduler made no picks at the test budget — "
        f"either the budget is too loose or build_candidates is empty."
    )
    assert exp.predicted_peak_bytes is not None
    assert exp.measured_peak_bytes is not None
    assert exp.baseline_peak_bytes is not None

    # The load-bearing gate: measured peak must be within 10% of predicted.
    # If this trips, the simulator is under-counting recompute intermediates;
    # walk back into mu_two_core.simulate (window_extra accounting).
    assert exp.measured_peak_bytes <= exp.predicted_peak_bytes * 1.10, (
        f"{model_name}: measured peak {exp.measured_peak_bytes/1e6:.1f} MB "
        f"exceeds predicted {exp.predicted_peak_bytes/1e6:.1f} MB by >10%. "
        f"Likely cause: simulator under-counts recompute intermediates."
    )

    # Peak actually moved down vs baseline — guards against
    # "scheduler picks but rewriter doesn't free" failure modes.
    assert exp.measured_peak_bytes < exp.baseline_peak_bytes, (
        f"{model_name}: rewrite did not reduce peak "
        f"(baseline {exp.baseline_peak_bytes/1e6:.1f} MB vs "
        f"measured {exp.measured_peak_bytes/1e6:.1f} MB). "
        f"Picks={len(exp.recomp_picks)}, reached_budget={exp.reached_budget}."
    )


# ---------------------------------------------------------------------------
# Resnet18 — bs=4, 0.7 baseline, tight tolerance
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_e2e_resnet18():
    exp, captured = _run_e2e("Resnet18", batch_size=4, budget_fraction=0.7)
    print(
        f"\n[e2e resnet18] picks={len(exp.recomp_picks)} "
        f"reached={exp.reached_budget} "
        f"baseline={exp.baseline_peak_bytes/1e6:.1f}MB "
        f"predicted={exp.predicted_peak_bytes/1e6:.1f}MB "
        f"measured={exp.measured_peak_bytes/1e6:.1f}MB"
    )
    _assert_peak_gates(exp, "Resnet18")
    _assert_allclose(
        captured["baseline"], captured["new"],
        rtol=1e-3, atol=1e-3, label="resnet18 e2e",
    )

    del exp
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Resnet50 — bs=2 to stay within typical dev-GPU memory; model coverage
# matters more than batch size for this smoke step.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_e2e_resnet50():
    exp, captured = _run_e2e("Resnet50", batch_size=2, budget_fraction=0.7)
    print(
        f"\n[e2e resnet50] picks={len(exp.recomp_picks)} "
        f"reached={exp.reached_budget} "
        f"baseline={exp.baseline_peak_bytes/1e6:.1f}MB "
        f"predicted={exp.predicted_peak_bytes/1e6:.1f}MB "
        f"measured={exp.measured_peak_bytes/1e6:.1f}MB"
    )
    _assert_peak_gates(exp, "Resnet50")
    _assert_allclose(
        captured["baseline"], captured["new"],
        rtol=1e-3, atol=1e-3, label="resnet50 e2e",
    )

    del exp
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Transformer — dropout draws fresh masks in the recompute clone, so
# allclose tolerance is 2e-2 (TENTATIVE_PLAN §9 gotcha).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_e2e_transformer():
    exp, captured = _run_e2e("Transformer", batch_size=4, budget_fraction=0.7)
    print(
        f"\n[e2e transformer] picks={len(exp.recomp_picks)} "
        f"reached={exp.reached_budget} "
        f"baseline={exp.baseline_peak_bytes/1e6:.1f}MB "
        f"predicted={exp.predicted_peak_bytes/1e6:.1f}MB "
        f"measured={exp.measured_peak_bytes/1e6:.1f}MB"
    )
    _assert_peak_gates(exp, "Transformer")
    _assert_allclose(
        captured["baseline"], captured["new"],
        rtol=2e-2, atol=2e-2, label="transformer e2e",
    )

    del exp
    torch.cuda.empty_cache()
