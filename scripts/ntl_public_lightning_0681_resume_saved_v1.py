"""Run exactly one pre-saved frozen Lightning 0.681 Kaggle Notebook with no training.

Existing v1 is an unused Quick Save; v2 will be the first actual execution.
All source assets remain the public author's pretrained weights. Fail closed
on any byte/source/provenance drift or prior competition submission.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from scripts.ntl_public_lightning_0681_one_shot import (
    DEST_REF, SOURCE_REF, DESCRIPTION, _sha, check_no_prior,
    frozen_reproduction, run, wait_for_preview, official_rows,
)

PINNED_ORIGINAL_SHA256 = "07d7349d7574a907d7d6028e0ae7abe889f04036cbd82f88239b41ecd8680037"
PINNED_OWN_SAVED_SHA256 = "03abab46a4d8bf1af73a1f811f90b18aa203408fcf86071bd1cc8fb72b944222"


def load_one_notebook(folder: Path) -> tuple[dict, dict]:
    files = list(folder.glob("*.ipynb"))
    if len(files) != 1:
        raise RuntimeError("expected exactly one downloaded notebook")
    notebook = json.loads(files[0].read_text(encoding="utf-8"))
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    return notebook, meta


def source_cells(nb: dict) -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def resume_saved_once() -> None:
    check_no_prior()
    if "KernelWorkerStatus.COMPLETE" not in run(
        ["kaggle", "kernels", "status", SOURCE_REF]
    ):
        raise RuntimeError("public original source is no longer complete")
    with tempfile.TemporaryDirectory(prefix="ntl0681-resume-") as tmp:
        folder = Path(tmp)
        frozen = frozen_reproduction(folder / "frozen")
        if frozen["original_sha256"] != PINNED_ORIGINAL_SHA256:
            raise RuntimeError("original public Notebook source changed; do not run")
        built = frozen["folder"]
        built_nb, built_meta = load_one_notebook(built)
        built_files = list(built.glob("*.ipynb"))
        if _sha(built_files[0]) != PINNED_OWN_SAVED_SHA256:
            raise RuntimeError("frozen guarded Notebook source drift")
        if "id_no" in built_meta:
            raise RuntimeError("public author numeric kernel ID persisted")
        existing = folder / "existing"
        existing.mkdir()
        run(["kaggle", "kernels", "pull", DEST_REF, "-p", str(existing), "-m"])
        saved_nb, saved_meta = load_one_notebook(existing)
        if (saved_meta.get("id") != DEST_REF or
            saved_meta.get("is_private") is not True or
            saved_meta.get("competition_sources") != ["name-that-llm"] or
            saved_meta.get("dataset_sources") != built_meta.get("dataset_sources") or
            saved_meta.get("model_sources") != built_meta.get("model_sources") or
            saved_meta.get("title") != built_meta.get("title")):
            raise RuntimeError("previous Quick Save v1 identity/inputs drifted")
        if source_cells(saved_nb) != source_cells(built_nb):
            raise RuntimeError("existing saved v1 code does not match exact frozen guarded code")
        assert len(source_cells(built_nb)) > 20
        print("PUBLIC_0681_OWN_QUICK_SAVE_V1_MATCHES_FROZEN_GUARDED_CODE", flush=True)

        # Re-push the SAME proven saved code exactly once, now with execution:
        # version 1 is no-run; version 2 is first and only inference run.
        # This is not a neural training invocation and has runtime guards.
        check_no_prior()
        output = run([
            "kaggle", "kernels", "push", "-p", str(built),
            "--accelerator", "NvidiaTeslaT4", "--timeout", "32400"
        ])
        if "Kernel version 2 successfully pushed" not in output:
            raise RuntimeError("unexpected Kaggle version response; refuse additional push: " +
                               output[-800:])
        print("PUBLIC_0681_GUARDED_PRETRAINED_INFERENCE_KERNEL_V2_PUSHED", flush=True)

    # No new model training; wait for existing v2 inference to finish and
    # produce actual competition preview output. NEVER blindly submit preview
    # for a science claim: code competition submits a hidden rerun on server.
    wait_for_preview()
    check_no_prior()
    response = run([
        "kaggle", "competitions", "submit", "name-that-llm",
        "-k", DEST_REF, "-v", "2", "-f", "submission.parquet",
        "-m", DESCRIPTION,
    ])
    import re
    import time
    refs = re.findall(r"Submission ref:\s*(\d+)", response, re.I)
    if len(refs) != 1:
        raise RuntimeError("official API did not confirm single submission ref: " +
                           response[-600:])
    ref = refs[0]
    print("PUBLIC_0681_OWN_OFFICIAL_SUBMISSION_ID", ref, flush=True)
    for _ in range(20):
        rows = [x for x in official_rows() if x.get("ref") == ref]
        if len(rows) == 1 and rows[0].get("description") == DESCRIPTION:
            row = rows[0]
            print("PUBLIC_0681_OWN_OFFICIAL_LEDGER", json.dumps({
                k: row.get(k) for k in
                ("ref", "status", "publicScore", "privateScore", "description", "date")
            }, sort_keys=True), flush=True)
            return
        time.sleep(15)
    raise RuntimeError("accepted submission not confirmed in official ledger")


if __name__ == "__main__":
    resume_saved_once()
