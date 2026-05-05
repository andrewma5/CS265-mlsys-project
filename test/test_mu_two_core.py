"""Unit tests for mu_two_core (Phase 2 Steps 1–3)."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch.fx as fx

from graph_prof import OP
from mu_two_core import ActivationMeta, Status, build_candidates, simulate


def _make_meta(**overrides):
    defaults = dict(
        node=MagicMock(spec=fx.Node),
        size=100,
        last_fwd_idx=10,
        first_bwd_idx=42,
    )
    defaults.update(overrides)
    return ActivationMeta(**defaults)


def test_activation_meta_roundtrip():
    node = MagicMock(spec=fx.Node)
    meta = ActivationMeta(node=node, size=512, last_fwd_idx=7, first_bwd_idx=21)

    assert meta.node is node
    assert meta.size == 512
    assert meta.last_fwd_idx == 7
    assert meta.first_bwd_idx == 21
    assert meta.recomp_srcs == set()
    assert meta.recomp_subgraph == ()
    assert meta.recomp_time == 0.0
    assert meta.recomp_cnt == 1
    assert meta.total_recomp_time == 0.0
    assert meta.status is Status.RETAINED


def test_recomp_ratio_initial():
    meta = _make_meta(size=100, recomp_time=10.0, total_recomp_time=10.0)
    assert meta.recomp_ratio == 10.0


def test_recomp_ratio_updates_after_mutation():
    meta = _make_meta(size=100, recomp_time=10.0, total_recomp_time=10.0)
    assert meta.recomp_ratio == 10.0

    meta.recomp_time = 20.0
    meta.total_recomp_time = 20.0
    assert meta.recomp_ratio == 5.0

    meta.recomp_cnt = 2
    meta.total_recomp_time = 40.0
    assert meta.recomp_ratio == 2.5


def test_recomp_ratio_div_by_zero():
    meta = _make_meta(size=100, total_recomp_time=0.0)
    assert meta.recomp_ratio == math.inf


def test_recomp_srcs_default_is_per_instance():
    a, b = _make_meta(), _make_meta()
    a.recomp_srcs.add(MagicMock(spec=fx.Node))
    assert b.recomp_srcs == set()


def test_status_enum_members():
    assert Status.RETAINED is not Status.RECOMPUTE
    assert Status.RECOMPUTE is not Status.SWAP
    assert Status.RETAINED is not Status.SWAP
    assert {s for s in Status} == {Status.RETAINED, Status.RECOMPUTE, Status.SWAP}


# ---------------------------------------------------------------------------
# Step 2: build_candidates
# ---------------------------------------------------------------------------


class _FakeNode:
    """Stand-in for fx.Node with just the attributes build_candidates reads."""

    def __init__(self, name: str, op: str, inputs: list = ()):
        self.name = name
        self.op = op
        self.all_input_nodes = list(inputs)
        self.users: list = []  # populated by _fake_prof from the inputs lists

    def __repr__(self) -> str:  # nicer pytest output
        return f"<FakeNode {self.name}>"


def _fake_prof(nodes, activations, regions, runtimes, sizes, last_fwd, first_bwd):
    """Build a SimpleNamespace exposing the GraphProfiler attrs build_candidates / simulate use."""
    order_index = {n: i for i, n in enumerate(nodes)}
    last_fwd_dict = dict(last_fwd)

    # Derive users from each node's input list. fx.Node exposes `users` as a
    # mapping; the simulator only iterates over it, so a list is interchangeable.
    for n in nodes:
        n.users = []
    for n in nodes:
        for inp in getattr(n, "all_input_nodes", ()):
            inp.users.append(n)

    def last_forward_use_idx(act):
        last = last_fwd_dict.get(act)
        return order_index[last if last is not None else act]

    return SimpleNamespace(
        node_order=list(nodes),
        activation_nodes=set(activations),
        node_region={n: regions[n] for n in nodes},
        order_index=order_index,
        avg_runtimes=dict(runtimes),
        avg_output_sizes=dict(sizes),
        last_forward_access=last_fwd_dict,
        first_backward_access=dict(first_bwd),
        last_forward_use_idx=last_forward_use_idx,
    )


def test_build_candidates_three_layer_mlp():
    # Forward graph: X (placeholder), then for i=1..3: matmul_i = matmul(prev, P_i),
    # Z_i = relu(matmul_i). Each Z_i is an activation (used by backward).
    #
    # Expected: every Z_i has a 1-op interior subgraph (its own matmul_i),
    # and recomp_srcs ends at {prev_activation, P_i} (or {X, P_1} for Z_1).
    X = _FakeNode("X", OP.PLACEHOLDER)
    P1 = _FakeNode("P1", OP.PLACEHOLDER)
    P2 = _FakeNode("P2", OP.PLACEHOLDER)
    P3 = _FakeNode("P3", OP.PLACEHOLDER)

    matmul1 = _FakeNode("matmul1", OP.CALL_FUNCTION, [X, P1])
    Z1 = _FakeNode("Z1", OP.CALL_FUNCTION, [matmul1])
    matmul2 = _FakeNode("matmul2", OP.CALL_FUNCTION, [Z1, P2])
    Z2 = _FakeNode("Z2", OP.CALL_FUNCTION, [matmul2])
    matmul3 = _FakeNode("matmul3", OP.CALL_FUNCTION, [Z2, P3])
    Z3 = _FakeNode("Z3", OP.CALL_FUNCTION, [matmul3])

    # bwd users (one per activation) — only used to populate first_backward_access.
    bwd1 = _FakeNode("bwd1", OP.CALL_FUNCTION, [Z1])
    bwd2 = _FakeNode("bwd2", OP.CALL_FUNCTION, [Z2])
    bwd3 = _FakeNode("bwd3", OP.CALL_FUNCTION, [Z3])

    nodes = [X, P1, P2, P3, matmul1, Z1, matmul2, Z2, matmul3, Z3, bwd3, bwd2, bwd1]
    activations = {Z1, Z2, Z3}
    regions = {
        X: 0, P1: 0, P2: 0, P3: 0,
        matmul1: 0, Z1: 0,
        matmul2: 0, Z2: 0,
        matmul3: 0, Z3: 0,
        bwd1: 2, bwd2: 2, bwd3: 2,
    }
    runtimes = {matmul1: 1.0, matmul2: 2.0, matmul3: 3.0, Z1: 0.1, Z2: 0.1, Z3: 0.1}
    sizes = {Z1: 100, Z2: 200, Z3: 300}
    last_fwd = {Z1: matmul2, Z2: matmul3, Z3: Z3}  # last fwd-region user
    first_bwd = {Z1: bwd1, Z2: bwd2, Z3: bwd3}

    prof = _fake_prof(nodes, activations, regions, runtimes, sizes, last_fwd, first_bwd)
    metas = build_candidates(prof)

    assert set(metas.keys()) == {Z1, Z2, Z3}

    m1 = metas[Z1]
    assert m1.recomp_subgraph == (matmul1,)
    assert m1.recomp_srcs == {X, P1}
    assert m1.recomp_time == 1.0
    assert m1.total_recomp_time == 1.0
    assert m1.recomp_cnt == 1
    assert m1.size == 100
    assert m1.last_fwd_idx == nodes.index(matmul2)
    assert m1.first_bwd_idx == nodes.index(bwd1)

    m2 = metas[Z2]
    assert m2.recomp_subgraph == (matmul2,)
    assert m2.recomp_srcs == {Z1, P2}     # halts at the prior activation
    assert m2.recomp_time == 2.0

    m3 = metas[Z3]
    assert m3.recomp_subgraph == (matmul3,)
    assert m3.recomp_srcs == {Z2, P3}
    assert m3.recomp_time == 3.0
    assert m3.recomp_ratio == 300 / 3.0


def test_build_candidates_act_without_fwd_user_uses_own_idx():
    # An activation produced in fwd, consumed only in bwd, has no entry in
    # last_forward_access. It should still be a candidate; last_fwd_idx falls
    # back to the act's own order_index (the tensor is freed at production).
    X = _FakeNode("X", OP.PLACEHOLDER)
    P1 = _FakeNode("P1", OP.PLACEHOLDER)
    matmul1 = _FakeNode("matmul1", OP.CALL_FUNCTION, [X, P1])
    Z1 = _FakeNode("Z1", OP.CALL_FUNCTION, [matmul1])
    bwd1 = _FakeNode("bwd1", OP.CALL_FUNCTION, [Z1])

    nodes = [X, P1, matmul1, Z1, bwd1]
    activations = {Z1}
    regions = {X: 0, P1: 0, matmul1: 0, Z1: 0, bwd1: 2}
    runtimes = {matmul1: 1.0}
    sizes = {Z1: 100}
    last_fwd = {}                # Z1 has no fwd-region user
    first_bwd = {Z1: bwd1}

    prof = _fake_prof(nodes, activations, regions, runtimes, sizes, last_fwd, first_bwd)
    metas = build_candidates(prof)

    assert set(metas.keys()) == {Z1}
    assert metas[Z1].last_fwd_idx == nodes.index(Z1)
    assert metas[Z1].first_bwd_idx == nodes.index(bwd1)


def test_build_candidates_skips_act_without_bwd_user():
    X = _FakeNode("X", OP.PLACEHOLDER)
    P1 = _FakeNode("P1", OP.PLACEHOLDER)
    matmul1 = _FakeNode("matmul1", OP.CALL_FUNCTION, [X, P1])
    Z1 = _FakeNode("Z1", OP.CALL_FUNCTION, [matmul1])
    bwd1 = _FakeNode("bwd1", OP.CALL_FUNCTION, [Z1])

    # An "activation" with no first_backward_access entry — must be skipped.
    Zorphan = _FakeNode("Zorphan", OP.CALL_FUNCTION, [matmul1])

    nodes = [X, P1, matmul1, Z1, Zorphan, bwd1]
    activations = {Z1, Zorphan}
    regions = {X: 0, P1: 0, matmul1: 0, Z1: 0, Zorphan: 0, bwd1: 2}
    runtimes = {matmul1: 1.0}
    sizes = {Z1: 100, Zorphan: 100}
    last_fwd = {Z1: Z1}
    first_bwd = {Z1: bwd1}  # Zorphan deliberately absent

    prof = _fake_prof(nodes, activations, regions, runtimes, sizes, last_fwd, first_bwd)
    metas = build_candidates(prof)

    assert set(metas.keys()) == {Z1}


# ---------------------------------------------------------------------------
# Step 3: simulate (memory simulator, Algorithm G)
# ---------------------------------------------------------------------------


def _three_layer_mlp_prof():
    """Same shape as test_build_candidates_three_layer_mlp but with explicit
    sizes for every node (simulator needs all output sizes, not just acts).

    Each Z_i is consumed both by the next matmul (in fwd) and a bwd op, so the
    activations stay alive until backward. The peak occurs at Z3 — at that
    index, X, P1..P3, matmul1..3, Z1..3 are all alive.
    """
    X = _FakeNode("X", OP.PLACEHOLDER)
    P1 = _FakeNode("P1", OP.PLACEHOLDER)
    P2 = _FakeNode("P2", OP.PLACEHOLDER)
    P3 = _FakeNode("P3", OP.PLACEHOLDER)

    matmul1 = _FakeNode("matmul1", OP.CALL_FUNCTION, [X, P1])
    Z1 = _FakeNode("Z1", OP.CALL_FUNCTION, [matmul1])
    matmul2 = _FakeNode("matmul2", OP.CALL_FUNCTION, [Z1, P2])
    Z2 = _FakeNode("Z2", OP.CALL_FUNCTION, [matmul2])
    matmul3 = _FakeNode("matmul3", OP.CALL_FUNCTION, [Z2, P3])
    Z3 = _FakeNode("Z3", OP.CALL_FUNCTION, [matmul3])

    bwd1 = _FakeNode("bwd1", OP.CALL_FUNCTION, [Z1])
    bwd2 = _FakeNode("bwd2", OP.CALL_FUNCTION, [Z2])
    bwd3 = _FakeNode("bwd3", OP.CALL_FUNCTION, [Z3])

    nodes = [X, P1, P2, P3, matmul1, Z1, matmul2, Z2, matmul3, Z3, bwd3, bwd2, bwd1]
    activations = {Z1, Z2, Z3}
    regions = {
        X: 0, P1: 0, P2: 0, P3: 0,
        matmul1: 0, Z1: 0,
        matmul2: 0, Z2: 0,
        matmul3: 0, Z3: 0,
        bwd1: 2, bwd2: 2, bwd3: 2,
    }
    runtimes = {matmul1: 1.0, matmul2: 2.0, matmul3: 3.0, Z1: 0.1, Z2: 0.1, Z3: 0.1}
    sizes = {
        X: 10, P1: 11, P2: 12, P3: 13,
        matmul1: 50, Z1: 100,
        matmul2: 60, Z2: 200,
        matmul3: 70, Z3: 300,
        bwd1: 0, bwd2: 0, bwd3: 0,
    }
    last_fwd = {Z1: matmul2, Z2: matmul3, Z3: Z3}
    first_bwd = {Z1: bwd1, Z2: bwd2, Z3: bwd3}

    prof = _fake_prof(nodes, activations, regions, runtimes, sizes, last_fwd, first_bwd)
    return prof, dict(
        nodes=nodes,
        X=X, P1=P1, P2=P2, P3=P3,
        matmul1=matmul1, matmul2=matmul2, matmul3=matmul3,
        Z1=Z1, Z2=Z2, Z3=Z3,
        bwd1=bwd1, bwd2=bwd2, bwd3=bwd3,
        sizes=sizes,
    )


def _baseline_alive_via_sweep(prof):
    """Reference sweep: same shape as plot_memory_stacked_timeline summed
    across all types. The simulator's empty-decisions output must agree with
    this exactly (both are deterministic functions of the same inputs)."""
    n = len(prof.node_order)
    last_user = {}
    for i, node in enumerate(prof.node_order):
        if node.op == OP.OUTPUT:
            continue
        lui = -1
        for u in node.users:
            uid = prof.order_index.get(u, -1)
            if uid > lui:
                lui = uid
        last_user[node] = lui if lui >= 0 else n - 1

    alive = [0] * n
    live = []
    for i, node in enumerate(prof.node_order):
        if i > 0:
            alive[i] = alive[i - 1]
        if node.op == OP.OUTPUT:
            continue
        size = int(prof.avg_output_sizes.get(node, 0))
        if size > 0:
            alive[i] += size
            live.append((last_user[node], size))
        still = []
        for lui, asize in live:
            if lui < i:
                alive[i] -= asize
            else:
                still.append((lui, asize))
        live = still
    return alive


def test_simulate_empty_baseline_matches_stacked_sum():
    prof, _ = _three_layer_mlp_prof()
    out = simulate(prof, recomps={})
    expected = _baseline_alive_via_sweep(prof)
    assert out == expected
    assert max(out) >= 300


def test_simulate_returns_per_node_timeline_length():
    prof, _ = _three_layer_mlp_prof()
    out = simulate(prof, recomps={})
    assert len(out) == len(prof.node_order)


def test_simulate_swaps_unsupported_raises():
    prof, refs = _three_layer_mlp_prof()
    Z1 = refs["Z1"]
    fake_meta = ActivationMeta(node=Z1, size=100, last_fwd_idx=0, first_bwd_idx=1)
    with pytest.raises(NotImplementedError):
        simulate(prof, recomps={}, swaps={Z1: fake_meta})


# ---------------------------------------------------------------------------
# Step 3: Real-model sanity check on Resnet18.
# ---------------------------------------------------------------------------


def test_simulate_resnet18_baseline_matches_stacked_sum():
    """Real-model gate: the simulator's empty-decisions output must equal the
    same alive-bytes sweep that plot_memory_stacked_timeline sums internally.

    Skipped on CPU-only machines. Catches bugs the 6-node fake fixtures cannot:
    OUTPUT-node skipping, last-user fallback for unconsumed tensors, and
    off-by-one carry-forward across hundreds of real ops.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for real-model profiling")

    from benchmarks import Experiment
    from graph_tracer import compile

    torch.cuda.empty_cache()
    exp = Experiment("Resnet18", batch_size=4)
    exp.init_opt_states()

    captured = {}

    def capture_then_passthrough(gm, args):
        warm_up_iters, profile_iters = 2, 3
        from graph_prof import GraphProfiler

        prof = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(warm_up_iters):
                prof.run(*args)
            prof.reset_stats()
            for _ in range(profile_iters):
                prof.run(*args)
            prof.aggregate_stats()
        captured["prof"] = prof
        return gm

    compiled_fn = compile(exp.train_step, capture_then_passthrough)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)

    prof = captured["prof"]
    sim = simulate(prof, recomps={})
    ref = _baseline_alive_via_sweep(prof)

    assert len(sim) == len(prof.node_order)
    assert len(sim) == len(ref)

    sim_peak = max(sim)
    ref_peak = max(ref)
    assert ref_peak > 0
    rel_err = abs(sim_peak - ref_peak) / ref_peak
    assert rel_err < 0.01, f"sim_peak={sim_peak} ref_peak={ref_peak} rel_err={rel_err}"

    # Report-only: ratio against measured CUDA peak. Won't be 1.0 (allocator
    # slack, out-of-band tensors); printed so Step 9's gate has prior context.
    if prof.avg_cumulative_mem:
        cuda_peak = max(prof.avg_cumulative_mem)
        if cuda_peak > 0:
            print(
                f"\n[resnet18 sanity] sim_peak={sim_peak} cuda_peak={cuda_peak} "
                f"ratio={sim_peak / cuda_peak:.3f}"
            )

    del exp, compiled_fn
    torch.cuda.empty_cache()
