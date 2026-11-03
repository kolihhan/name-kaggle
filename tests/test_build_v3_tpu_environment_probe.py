from __future__ import annotations

import json
from pathlib import Path

from scripts.build_v3_tpu_environment_probe import KERNEL_REF, write_kernel_bundle


def test_bundle_requests_private_offline_tpu_and_has_no_data_sources(tmp_path: Path) -> None:
    write_kernel_bundle(tmp_path)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == KERNEL_REF
    assert meta["is_private"] is True
    assert meta["enable_internet"] is False
    assert meta["machine_shape"] == "TpuV5E8"
    assert meta["competition_sources"] == []
    assert meta["dataset_sources"] == []
    assert meta["kernel_sources"] == []
    assert meta["model_sources"] == []


def test_probe_only_checks_accelerator_identity_and_never_scores(tmp_path: Path) -> None:
    write_kernel_bundle(tmp_path)
    source = (tmp_path / "kernel_main.py").read_text()
    lowered = source.lower()
    assert "torch_xla" in source
    assert "runtime.device_type" in source
    assert "real_devices" in source
    assert "environment_probe.json" in source
    assert "qwen" not in lowered
    assert "parquet" not in lowered
    assert "balanced_accuracy" not in lowered
    assert "competitions submit" not in lowered
    assert "official_rows_scored" in source
    assert "firewall_rows_scored" in source
    assert "competition_test_rows_used" in source


def test_probe_pass_requires_actual_tpu_device(tmp_path: Path) -> None:
    write_kernel_bundle(tmp_path)
    source = (tmp_path / "kernel_main.py").read_text()
    assert "device_type == 'TPU'" in source
    assert "gate0_pass" in source
