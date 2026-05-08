"""Numerical-equivalence tests for mu_two_rewrite (Step 8).

The rewriter is correctness-tested by comparing tensors produced by the
joint graph before and after rewriting. Param/opt-state mutate in-place
during the call (fused Adam updates them), so we snapshot args before the
baseline run and restore before the rewritten run — otherwise the second
run sees drifted weights and allclose can't tell a rewrite bug from an
input-divergence artifact (pattern from phase2-rebuild's test_recompute).
"""

import pytest
import torch
import torch.nn as nn
import torch.fx as fx


# ---------------------------------------------------------------------------
# Helpers
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
# CPU-only: no-picks identity short-circuits before touching prof.
# ---------------------------------------------------------------------------


def test_rewrite_no_picks_returns_unchanged():
    from mu_two_rewrite import rewrite_recomputes

    class M(nn.Module):
        def forward(self, x):
            return x + 1

    gm = fx.symbolic_trace(M())
    nodes_before = list(gm.graph.nodes)

    out = rewrite_recomputes(gm, prof=None, recomps={})

    assert out is gm
    assert list(out.graph.nodes) == nodes_before


# ---------------------------------------------------------------------------
# TinyModel — small CUDA tests exercising single pick + cascade.
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self, dim=64, layers=4):
        super().__init__()
        mods = []
        for _ in range(layers):
            mods.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*mods)

    def forward(self, x):
        return self.mod(x)


def _run_tinymodel_rewrite(budget_fraction: float):
    """Profile TinyModel, schedule at budget_fraction × baseline, rewrite.
    Returns dict with baseline/new flattened outputs and pick count."""
    from graph_prof import GraphProfiler
    from graph_tracer import SEPFunction, compile
    from mu_two_core import simulate
    from mu_two_rewrite import rewrite_recomputes
    from mu_two_scheduler import greedy_recompute

    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    # bs=32 leaves params+Adam-state dominating peak, so peak_idx falls in the
    # opt step where no activation's (last_fwd_idx, first_bwd_idx) interval
    # reaches — the peak-aware gate would (correctly) refuse all picks. bs=512
    # makes activations the dominant memory term in the fwd→bwd gap, where
    # picks can actually lower peak.
    model = _TinyModel().to(dev)
    batch = torch.randn(512, 64, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=True, capturable=True)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.rand_like(p)
    opt.step()
    opt.zero_grad()

    def train_step(model, opt, batch):
        loss = model(batch).sum()
        loss = SEPFunction.apply(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()

    captured = {}

    def xform(gm, args):
        prof = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(2):
                prof.run(*args)
            prof.reset_stats()
            for _ in range(3):
                prof.run(*args)
        prof.aggregate_stats()

        snap = _snapshot(args)
        with torch.no_grad():
            baseline = gm(*args)
        baseline_flat = [t.detach().clone() for t in _flatten(baseline)]

        baseline_peak = max(simulate(prof, recomps={}))
        budget = int(baseline_peak * budget_fraction)
        recomps, _ = greedy_recompute(prof, budget=budget)

        gm2 = rewrite_recomputes(gm, prof, recomps)

        _restore(args, snap)
        with torch.no_grad():
            new = gm2(*args)
        new_flat = _flatten(new)

        captured["baseline"] = baseline_flat
        captured["new"] = new_flat
        captured["picks"] = len(recomps)
        captured["max_recomp_cnt"] = max(
            (m.recomp_cnt for m in recomps.values()), default=0
        )
        return gm2

    compiled = compile(train_step, xform)
    compiled(model, opt, batch)
    return captured


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rewrite_tinymodel_single_pick():
    """Loose budget (95% baseline): scheduler picks at least one activation.
    Verifies the basic rewrite path (extraction + clone + rewire) produces
    numerically equivalent outputs."""
    captured = _run_tinymodel_rewrite(budget_fraction=0.95)
    assert captured["picks"] >= 1, "Expected at least one pick at 95% budget"
    # atol=1e-3 absorbs fp32 accumulation noise from re-executing matmul+relu
    # in the recompute clone (different memory layout → different cuBLAS path).
    _assert_allclose(
        captured["baseline"], captured["new"],
        rtol=2e-3, atol=2e-3, label="tinymodel single",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rewrite_tinymodel_cascade():
    """Tight budget (50% baseline) forces multiple picks. Exercises cascade
    rewriting: when picked A's transitive subgraph contains another picked
    activation S, A's clone must materialize a fresh interior copy of S
    (not bind to original or to S's own standalone clone)."""
    captured = _run_tinymodel_rewrite(budget_fraction=0.5)
    assert captured["picks"] >= 2, (
        f"Expected ≥2 picks at 50% budget, got {captured['picks']}"
    )
    # Load-bearing: assert Algorithm E actually fired (some pick was bumped to
    # recomp_cnt > 1 because it appeared in another pick's recomp_srcs at pick
    # time). Without this, the test would pass even if E/F were disabled, as
    # long as two parallel activations got picked. recomp_cnt > 1 is observable
    # only via cascade — it requires E to mutate a prior recomp's count.
    assert captured["max_recomp_cnt"] > 1, (
        f"Expected at least one cascade (recomp_cnt > 1), got "
        f"max_recomp_cnt={captured['max_recomp_cnt']}. Picks may all be parallel."
    )
    _assert_allclose(
        captured["baseline"], captured["new"],
        rtol=2e-3, atol=2e-3, label="tinymodel cascade",
    )


# ---------------------------------------------------------------------------
# Resnet18 — full pipeline numerical equivalence at 0.7 × baseline_peak.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rewrite_resnet18():
    from benchmarks import Experiment
    from graph_prof import GraphProfiler
    from graph_tracer import compile
    from mu_two_core import simulate
    from mu_two_rewrite import rewrite_recomputes
    from mu_two_scheduler import greedy_recompute

    torch.cuda.empty_cache()
    # bs=64: at small batches the param + Adam-state floor dominates the peak,
    # so simulate(recomps={}) ≈ simulate(recomps=anything) and greedy runs to
    # exhaustion rewriting all candidates — slow and uninteresting. bs=64 makes
    # activations the dominant memory term, so a 70% budget cut is achievable
    # with a handful of picks.
    exp = Experiment("Resnet18", batch_size=64)
    exp.init_opt_states()

    captured = {}

    def xform(gm, args):
        prof = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(2):
                prof.run(*args)
            prof.reset_stats()
            for _ in range(3):
                prof.run(*args)
        prof.aggregate_stats()

        snap = _snapshot(args)
        with torch.no_grad():
            baseline = gm(*args)
        baseline_flat = [t.detach().clone() for t in _flatten(baseline)]

        baseline_peak = max(simulate(prof, recomps={}))
        budget = int(baseline_peak * 0.7)
        recomps, reached = greedy_recompute(prof, budget=budget)

        gm2 = rewrite_recomputes(gm, prof, recomps)

        _restore(args, snap)
        with torch.no_grad():
            new = gm2(*args)
        # Clone immediately: graph_tracer.compile re-runs gm2 with these same
        # args after xform returns, mutating the parameter tensors in place.
        # Without the clone, new_flat would reflect "params after 2 train steps"
        # vs baseline_flat at "after 1 step".
        new_flat = [t.detach().clone() for t in _flatten(new)]

        captured["baseline"] = baseline_flat
        captured["new"] = new_flat
        captured["picks"] = len(recomps)
        captured["reached"] = reached
        return gm2

    compiled = compile(exp.train_step, xform)
    compiled(exp.model, exp.optimizer, exp.example_inputs)

    print(
        f"\n[resnet18 rewrite] picks={captured['picks']} "
        f"reached_budget={captured['reached']}"
    )
    # Resnet conv reduction order amplifies fp32 noise across the rewrite —
    # match phase2-rebuild's already-validated tolerance.
    _assert_allclose(
        captured["baseline"], captured["new"],
        rtol=1e-3, atol=1e-3, label="resnet18",
    )

    del exp, compiled
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Resnet50 — same pipeline, bigger graph. Diagnostic at 0.7 × baseline_peak.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rewrite_resnet50():
    from benchmarks import Experiment
    from graph_prof import GraphProfiler
    from graph_tracer import compile
    from mu_two_core import simulate
    from mu_two_rewrite import rewrite_recomputes
    from mu_two_scheduler import greedy_recompute

    torch.cuda.empty_cache()
    # See test_rewrite_resnet18 for batch-size rationale. Resnet50 has 4× the
    # params of Resnet18 so we need bs≥16 to push peak above the param floor.
    exp = Experiment("Resnet50", batch_size=16)
    exp.init_opt_states()

    captured = {}

    def xform(gm, args):
        prof = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(2):
                prof.run(*args)
            prof.reset_stats()
            for _ in range(3):
                prof.run(*args)
        prof.aggregate_stats()

        snap = _snapshot(args)
        with torch.no_grad():
            baseline = gm(*args)
        baseline_flat = [t.detach().clone() for t in _flatten(baseline)]

        baseline_peak = max(simulate(prof, recomps={}))
        budget = int(baseline_peak * 0.7)
        recomps, reached = greedy_recompute(prof, budget=budget)
        final_peak = max(simulate(prof, recomps))

        n_nodes_before = sum(1 for _ in gm.graph.nodes)
        gm2 = rewrite_recomputes(gm, prof, recomps)
        n_nodes_after = sum(1 for _ in gm2.graph.nodes)

        _restore(args, snap)
        with torch.no_grad():
            new = gm2(*args)
        # See test_rewrite_resnet18 — must clone, not reference.
        new_flat = [t.detach().clone() for t in _flatten(new)]

        captured["baseline"] = baseline_flat
        captured["new"] = new_flat
        captured["picks"] = len(recomps)
        captured["reached"] = reached
        captured["baseline_mb"] = baseline_peak / 1e6
        captured["budget_mb"] = budget / 1e6
        captured["final_mb"] = final_peak / 1e6
        captured["nodes_before"] = n_nodes_before
        captured["nodes_after"] = n_nodes_after
        captured["max_recomp_cnt"] = max(
            (m.recomp_cnt for m in recomps.values()), default=0
        )
        return gm2

    compiled = compile(exp.train_step, xform)
    compiled(exp.model, exp.optimizer, exp.example_inputs)

    print(
        f"\n[resnet50 rewrite] picks={captured['picks']} "
        f"reached_budget={captured['reached']} "
        f"baseline_mb={captured['baseline_mb']:.1f} "
        f"budget_mb={captured['budget_mb']:.1f} "
        f"final_mb={captured['final_mb']:.1f} "
        f"nodes={captured['nodes_before']}->{captured['nodes_after']} "
        f"max_recomp_cnt={captured['max_recomp_cnt']}"
    )
    # Resnet conv reduction order amplifies fp32 noise across the rewrite —
    # match Resnet18 tolerance.
    _assert_allclose(
        captured["baseline"], captured["new"],
        rtol=1e-3, atol=1e-3, label="resnet50",
    )

    del exp, compiled
    torch.cuda.empty_cache()
