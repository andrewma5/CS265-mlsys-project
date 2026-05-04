"""Unit tests for mu_two_core (Phase 2 Step 1)."""

import math
from unittest.mock import MagicMock

import torch.fx as fx

from mu_two_core import ActivationMeta, Status


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
