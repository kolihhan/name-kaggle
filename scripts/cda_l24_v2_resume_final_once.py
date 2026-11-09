"""Infrastructure-only recovery: rerun unchanged C1 final fit as Kaggle kernel v2.

C0/C1 training and signed private C1 dataset v1 are immutable. The first final
kernel v1 errored before model loading because Kaggle expanded its ZIP input.
Only fix: use the expanded directory with the SAME checkpoint/source SHA gates.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess
import time

from scripts.cda_l24_v2_run_once import (
    COMPETITION, DATASET_REF, DESCRIPTION, KERNEL_REF, command, official_submissions,
    require_completed, state, verify_not_previous_submit,
)
from scripts.build_cda_l24_v2_kaggle_submission import write_kernel_bundle

STAGE_A_SHA = "0dc043dda8da70aa28f04ffed105bae7ac9af19b84a322524333b197a8619fbe"
SOURCE_SHA = "f12f5c5fce027a96ee1df84a27ca571462f9c1ec4421ed49ee61ec26040a2328"
VERSION = 2
EXPECTED_FILES = {
    "adapter/adapter_model.safetensors", "adapter/adapter_config.json",
    "heads.safetensors", "meta.json", "tokenizer/tokenizer.json",
    "checkpoint.sha256",
}


def preflight() -> None:
    verify_not_previous_submit()
    status = json.loads(command([
        "kaggle", "datasets", "status", DATASET_REF, "--format", "json"
    ]))
    if (status.get("status") != "ready" or
            int(status.get("current_version_number", -1)) != 1):
        raise RuntimeError("frozen C1 dataset not ready at version 1")
    dataset_listing = command(["kaggle", "datasets", "files", DATASET_REF, "-v"])
    rows = list(csv.DictReader(dataset_listing.splitlines()))
    filenames = {r.get("name") or r.get("Name") for r in rows}
    if not EXPECTED_FILES.issubset(filenames):
        raise RuntimeError("Kaggle auto-expanded C1 files unavailable or incomplete")
    if "cda-v2-C1-checkpoint.zip" in filenames:
        raise RuntimeError("archive still present: frozen v2 mount diagnosis changed")
    if state(KERNEL_REF) != "ERROR":
        raise RuntimeError("expected failed final Notebook version 1; refusing overwrite")

    # Verify that v1 was configured with both datasets and offline model source.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        command(["kaggle", "kernels", "pull", KERNEL_REF, "-m", "-p", tmp])
        meta = json.loads((Path(tmp) / "kernel-metadata.json").read_text())
    if (meta.get("dataset_sources") != [
            DATASET_REF, "kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2"
        ] or meta.get("competition_sources") != [COMPETITION]
        or meta.get("enable_internet") is not False):
        raise RuntimeError("original official Notebook source identity drift")
    print("FROZEN_CDA_V2_C1_DATASET_V1_AND_FAILED_KERNEL_V1_VERIFIED", flush=True)


def submit_v2() -> None:
    preflight()
    folder = Path("/tmp/ntl-cda-final-v2-recovery")
    meta = write_kernel_bundle(
        folder, stage_a_sha256=STAGE_A_SHA,
        source_sha256=SOURCE_SHA, repo_root=Path("."),
    )
    if (meta["id"] != KERNEL_REF or meta["enable_internet"] is not False or
            meta["dataset_sources"][0] != DATASET_REF):
        raise RuntimeError("signed v2 Notebook source drift")
    source = folder / "kernel_main.py"
    compile(source.read_text(), str(source), "exec")
    if "resolve_checkpoint_input" not in source.read_text():
        raise RuntimeError("confirmed Kaggle auto-expanded data fix not bundled")
    print("FROZEN_CDA_V2_FINAL_NOTEBOOK_V2_STATIC_AND_INPUTS_PASS", flush=True)

    result = command([
        "kaggle", "kernels", "push", "-p", str(folder),
        "--accelerator", "NvidiaTeslaT4", "--timeout", "32400",
    ])
    if f"Kernel version {VERSION} successfully pushed" not in result:
        raise RuntimeError("expected exactly kernel version 2 accepted: " + result[-650:])
    print("CDA_V2_FIXED_FINAL_NOTEBOOK_VERSION_2_PUSHED", flush=True)
    require_completed([KERNEL_REF], maximum_checks=150, seconds=120)
    listing = command(["kaggle", "kernels", "files", KERNEL_REF, "-v"])
    if "submission.parquet" not in listing:
        raise RuntimeError("final version2 missing submission.parquet")
    verify_not_previous_submit()
    submission = command([
        "kaggle", "competitions", "submit", COMPETITION,
        "-k", KERNEL_REF, "-v", str(VERSION),
        "-f", "submission.parquet", "-m", DESCRIPTION,
    ])
    refs = re.findall(r"Submission ref:\s*(\d+)", submission, re.I)
    if len(refs) != 1:
        raise RuntimeError("submit response missing single accepted ref: " + submission[-700:])
    ref = refs[0]
    print("CDA_V2_OFFICIAL_SUBMISSION_ACCEPTED_REF", ref, flush=True)
    for _ in range(20):
        matching = [r for r in official_submissions() if r.get("ref") == ref]
        if len(matching) == 1 and matching[0].get("description") == DESCRIPTION:
            item = matching[0]
            print("CDA_V2_OFFICIAL_LEDGER", json.dumps({
                k: item.get(k) for k in
                ("ref", "status", "publicScore", "privateScore", "date", "description")
            }, sort_keys=True), flush=True)
            return
        time.sleep(15)
    raise RuntimeError("submission accepted but ledger not consistent within verifier window")


if __name__ == "__main__":
    submit_v2()
