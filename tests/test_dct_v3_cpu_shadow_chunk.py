import pytest

from scripts.dct_v3_cpu_shadow_chunk import (
    build_lineage_fingerprint,
    initial_progress,
    prepare_head_features,
    scientific_scope,
    validate_progress,
)


def test_scope_is_shadow_only():
    scope = scientific_scope()
    assert scope["scope"] == "cpu_shadow_external_transfer_training_only"
    assert scope["official_rows_scored"] == 0
    assert scope["public_lb_tuning"] is False
    assert scope["kaggle_submission_authorized"] is False
    assert scope["frozen_cuda_v3_gate_authorized"] is False


def _common_lineage():
    return dict(
        model_revision="rev",
        training_spec={"seed": 7, "accumulation": 6},
        holdout_sources=["a", "b", "c"],
        labels=["m1", "m2"],
        triplets=[[1, 2, 3], [4]],
    )


def test_lineage_changes_when_representation_changes():
    common = _common_lineage()
    mean = build_lineage_fingerprint(representation="mean", backend="fp16", **common)
    dct = build_lineage_fingerprint(representation="dct", backend="fp16", **common)
    assert mean != dct
    assert mean == build_lineage_fingerprint(
        representation="mean", backend="fp16", **common
    )


def test_lineage_changes_when_backend_changes():
    common = _common_lineage()
    fp16 = build_lineage_fingerprint(representation="dct", backend="fp16", **common)
    bf16 = build_lineage_fingerprint(representation="dct", backend="bf16", **common)
    assert fp16 != bf16


def test_unknown_backend_fails_closed_in_lineage():
    with pytest.raises(ValueError, match="unsupported CPU shadow backend"):
        build_lineage_fingerprint(
            representation="dct", backend="int8", **_common_lineage()
        )


def test_head_features_cross_autocast_boundary_as_float32():
    sentinel = object()

    class FakeTensor:
        def __init__(self):
            self.float_called = False

        def float(self):
            self.float_called = True
            return sentinel

    tensor = FakeTensor()
    assert prepare_head_features(tensor) is sentinel
    assert tensor.float_called is True


def test_progress_validation_allows_partial_group():
    progress = initial_progress()
    progress["microbatch_index"] = 2
    validate_progress(progress, group_count=5, current_group_size=6)


def test_progress_validation_rejects_out_of_range_cursor():
    progress = initial_progress()
    progress["group_index"] = 6
    progress["optimizer_steps"] = 6
    progress["microbatch_index"] = 0
    with pytest.raises(RuntimeError, match="group_index"):
        validate_progress(progress, group_count=5, current_group_size=None)


def test_progress_starts_at_zero():
    assert initial_progress() == {
        "group_index": 0,
        "microbatch_index": 0,
        "optimizer_steps": 0,
        "loss_sum": 0.0,
        "loss_count": 0,
    }
