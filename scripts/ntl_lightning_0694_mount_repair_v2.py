"""Fix ONLY Kaggle's verified dataset mount namespace in frozen Lightning 0.694.

The original 23 dataset / 6 model / 9 kernel inputs are unchanged. All ten
aliases below came from a completed private CPU mount census, not guesswork.
This is infrastructure repair; no training, scoring or model retuning.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts.ntl_lightning_0694_no_fit_candidate import build, cell_source

# Exact roots confirmed by the private Kaggle CPU mount census on 2026-11-15.
OWNER_BY_DATASET = {
    "nlm-enc-v5": "dhruvtibarewal",
    "nlm-l24-adapters": "dhruvtibarewal",
    "nlm-l24-adapters-v5": "dhruvtibarewal",
    "ntl-atlas-qwen3-8b-v1": "josephadamski91",
    "ntl-handwriting-blend": "nishantdahal",
    "ntl-q14-lora": "nuridudak",
    "ntl-q14-lora-v4": "nuridudak",
    "ntl-spectral-style-heads-paired-4404": "kirilltym",
    "ntl-svc-prefit-pool-v4n": "nuridudak",
    "ntl-two-stage-encoders": "nuridudak",
}
REAL_KAGGLE_REF = "kolihhan/lightning-0-694-authentic-no-fit-inference"
TITLE = "Lightning 0.694 - authentic no-fit inference"


def build_v2(source_dir: Path, dest_dir: Path) -> dict:
    original = build(source_dir, dest_dir)
    metadata_file = dest_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_file.read_text())
    nbpath = dest_dir / metadata["code_file"]
    notebook = json.loads(nbpath.read_text())
    before = "\n".join(
        cell_source(c) for c in notebook["cells"] if c.get("cell_type") == "code"
    )
    counts = {}
    for dataset, owner in sorted(OWNER_BY_DATASET.items(), key=lambda kv: -len(kv[0])):
        old = "/kaggle/input/" + dataset
        new = "/kaggle/input/datasets/" + owner + "/" + dataset
        count = 0
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = cell_source(cell)
            n = source.count(old)
            count += n
            if n:
                cell["source"] = source.replace(old, new).splitlines(keepends=True)
        if count < 1:
            raise ValueError("expected_root_absent:" + dataset)
        counts[dataset] = count

    code = "\n".join(
        cell_source(c) for c in notebook["cells"] if c.get("cell_type") == "code"
    )
    for dataset in OWNER_BY_DATASET:
        if "/kaggle/input/" + dataset in code:
            raise ValueError("unrepaired_flat_root:" + dataset)
    if (code.count("NO_NEW_TRAINING: Gaussian fit blocked") != 1
            or code.count("NO_NEW_TRAINING: Atlas training blocked") != 1
            or code.count("preview = False  # always genuine model inference") != 1):
        raise ValueError("frozen_inference_or_no_fit_guards_modified")

    # Validate the actual generated modules before starting another GPU version.
    compiled = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell_source(cell)
        if not source.startswith("%%writefile "):
            continue
        lines = source.splitlines()
        filename = lines[0].split(" ", 1)[1]
        tree = ast.parse("\n".join(lines[1:]), filename=filename)
        compile(tree, filename, "exec")
        compiled += 1
    if compiled != 37:
        raise ValueError("generated_module_inventory_drift")

    metadata.update({"id": REAL_KAGGLE_REF, "title": TITLE})
    metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    output = json.dumps(notebook, ensure_ascii=False)
    nbpath.write_text(output, encoding="utf-8")
    return {
        "original_public_source_sha256": original["original_sha256"],
        "v1_static_builder_sha256": original["guarded_notebook_sha256"],
        "v2_mount_repair_notebook_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "dataset_flat_path_occurrences_repaired": counts,
        "compiled_generated_modules": compiled,
        "input_source_counts": {k: len(metadata[k]) for k in
            ("dataset_sources", "kernel_sources", "model_sources")},
        "kaggle_existing_kernel_ref": REAL_KAGGLE_REF,
        "pretrained_model_or_weights_changed": False,
        "real_inference_guard_preserved": True,
        "supervised_training_allowed": False,
        "gpu_launched_by_builder": False,
        "competition_submission": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("NTL0694_FROZEN_MOUNT_REPAIR_V2=" +
          json.dumps(build_v2(args.source, args.out), sort_keys=True), flush=True)
