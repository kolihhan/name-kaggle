"""Reproduce frozen Lightning 0.694 as a NEW genuine-inference, no-fit candidate.

Public source only, all pretrained assets referenced by original slugs.
No Kaggle competition test or training data enters GitHub.
This is an attributed public-source candidate, not original research.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SOURCE = "lightningv08/0-694-glm-5-3-likelihood-qwen3-8-27b-lora"
DEST = "kolihhan/ntl-lightning0694-real-no-fit-nov15"
PIN = "9808d997185c125d92b0a67501bb763eb1813ae5223d3d06ce82a5518f205a2e"
PREVIEW = ("preview = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is None and "
           "emit_known_preview(DATA,None,OUTPUT)")
TRAIN_MARK = "if training:\n        base.gradient_checkpointing_enable"
BLOCK_GAUSS = """def _ntl_no_supervised_gauss(*args, **kwargs):
    raise RuntimeError('NO_NEW_TRAINING: Gaussian fit blocked')
"""

def cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)

def build(public_dir: Path, target: Path) -> dict:
    files = list(public_dir.glob("*.ipynb"))
    if len(files) != 1:
        raise RuntimeError("exact_one_public_source_required")
    origin = files[0].read_bytes()
    if hashlib.sha256(origin).hexdigest() != PIN:
        raise RuntimeError("upstream_source_sha_changed")
    notebook = json.loads(origin)
    metadata = json.loads((public_dir / "kernel-metadata.json").read_text())
    if (metadata.get("id") != SOURCE or metadata.get("is_private") is not False
            or metadata.get("competition_sources") != ["name-that-llm"]
            or len(metadata.get("dataset_sources") or []) != 23
            or len(metadata.get("kernel_sources") or []) != 9
            or len(metadata.get("model_sources") or []) != 6):
        raise RuntimeError("upstream_identity_or_assets_changed")
    if len([c for c in notebook["cells"] if c.get("cell_type") == "code"]) != 50:
        raise RuntimeError("upstream_code_cell_count_changed")

    changed = {"preview": 0, "gaussian": 0, "atlas": 0}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if PREVIEW in src:
            if src.count(PREVIEW) != 1:
                raise RuntimeError("preview_anchor_ambiguous")
            src = src.replace(PREVIEW, "preview = False  # always genuine model inference")
            changed["preview"] += 1
        if src.startswith("%%writefile ") and src.splitlines()[0].endswith("/core_readout.py"):
            if (src.count("gauss.fit(") != 1
                    or "fixed-no-test-redo" not in src
                    or "source-test-conditioned" not in src):
                raise RuntimeError("gaussian_branch_contract_drift")
            lines = src.splitlines(keepends=True)
            lines.insert(1, BLOCK_GAUSS)
            src = "".join(lines).replace("gauss.fit(", "_ntl_no_supervised_gauss(", 1)
            changed["gaussian"] += 1
        if TRAIN_MARK in src and "model = AtlasModel(base, labels, config)" in src:
            if src.count(TRAIN_MARK) != 1:
                raise RuntimeError("atlas_train_branch_ambiguous")
            src = src.replace(
                TRAIN_MARK,
                "if training:\n"
                "        raise RuntimeError('NO_NEW_TRAINING: Atlas training blocked')\n"
                "        base.gradient_checkpointing_enable",
                1,
            )
            changed["atlas"] += 1
        cell["source"] = src.splitlines(keepends=True)
    if changed != {"preview": 1, "gaussian": 1, "atlas": 1}:
        raise RuntimeError("expected_real_inference_and_training_guards_missing")
    code = "\n".join(cell_source(c) for c in notebook["cells"] if c.get("cell_type") == "code")
    if (PREVIEW in code or "preview = False  # always genuine model inference" not in code
            or "gauss.fit(" in code
            or code.count("NO_NEW_TRAINING: Gaussian fit blocked") != 1
            or code.count("NO_NEW_TRAINING: Atlas training blocked") != 1):
        raise RuntimeError("incomplete_or_unexpected_source_transformation")

    # Keep all frozen pretrained model and dataset mounts; never copy their bytes.
    metadata.pop("id_no", None)
    metadata.pop("docker_image", None)
    metadata.update({
        "id": DEST, "title": "Lightning 0.694 - authentic no-fit inference",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    })
    name = metadata.get("code_file")
    if not name or name != Path(name).name or not name.endswith(".ipynb"):
        raise RuntimeError("invalid_notebook_filename")
    # Markdown is credited and has no effect on inference.
    notebook["cells"].insert(0, {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["Attributed reproduction of Lightningv08 0.694 public Name That LLM notebook.\n",
                   "No additional supervised training. 0.694 is an author title, not our score.\n"],
    })
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("refuse_existing_candidate_overwrite")
    target.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(notebook, ensure_ascii=False)
    (target / name).write_text(payload, encoding="utf-8")
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, sort_keys=True, indent=2))
    return {
        "original_sha256": PIN,
        "guarded_notebook_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "source": SOURCE, "dest": DEST,
        "code_cells": 50, "dataset_sources": 23, "kernel_sources": 9,
        "model_sources": 6,
        "genuine_preview_bypass_removed": True,
        "gaussian_supervised_fit_fail_closed": True,
        "atlas_neural_train_fail_closed": True,
        "new_training": False, "gpu_run": False,
        "official_submission": False,
        "accuracy_unknown": True,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.out)
    print("NTL_LIGHTNING_0694_NEW_INFERENCE_ONLY_CANDIDATE="
          + json.dumps(result, sort_keys=True), flush=True)
