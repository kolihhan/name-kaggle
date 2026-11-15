"""Repair ten Kaggle mount paths in OUR already-frozen Lightning0694 V1 only.

Never pull/build from the mutable upstream notebook for this repair.
CPU census proved that private Kaggle mounts external datasets under
/kaggle/input/datasets/<owner>/<slug>, not the legacy flat directory.
No changes to pretrained assets, inference or training control flow.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts.ntl_lightning_0694_no_fit_candidate import cell_source

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
FROZEN_OWN_KERNEL = "kolihhan/lightning-0-694-authentic-no-fit-inference"
FROZEN_UPSTREAM_SHA256 = "9808d997185c125d92b0a67501bb763eb1813ae5223d3d06ce82a5518f205a2e"
FROZEN_V1_INITIAL_BUILDER_SHA256 = "669976456fddc0434d4c348deb7155d2951060ace89ea3a7dd6ea15d712248b9"


def build_v2(own_v1_dir: Path, destination: Path) -> dict:
    files = list(own_v1_dir.glob("*.ipynb"))
    if len(files) != 1:
        raise RuntimeError("frozen_owned_V1_notebook_missing_or_ambiguous")
    metadata = json.loads((own_v1_dir / "kernel-metadata.json").read_text())
    notebook = json.loads(files[0].read_text())
    if (metadata.get("is_private") is not True
        or metadata.get("id") != FROZEN_OWN_KERNEL
        or metadata.get("competition_sources") != ["name-that-llm"]
        or len(metadata.get("dataset_sources") or []) != 23
        or len(metadata.get("model_sources") or []) != 6
        or len(metadata.get("kernel_sources") or []) != 9):
        raise RuntimeError("frozen_owned_V1_identity_or_mount_manifest_drift")
    code_cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    if len(code_cells) != 50:
        raise RuntimeError("frozen_owned_V1_code_cell_count_drift")
    before = "\n".join(cell_source(c) for c in code_cells)
    for marker in ("preview = False  # always genuine model inference",
                   "NO_NEW_TRAINING: Gaussian fit blocked",
                   "NO_NEW_TRAINING: Atlas training blocked"):
        if before.count(marker) != 1:
            raise RuntimeError("frozen_owned_V1_guard_or_preview_drift:" + marker)
    if ("gauss.fit(" in before
        or "/kaggle/input/datasets/dhruvtibarewal/nlm-enc-v5" in before
        or "/kaggle/input/nlm-enc-v5" not in before):
        raise RuntimeError("frozen_owned_V1_already_repaired_or_unprotected")
    counts = {}
    for dataset, owner in sorted(OWNER_BY_DATASET.items(),key=lambda x:-len(x[0])):
        old = "/kaggle/input/" + dataset
        new = "/kaggle/input/datasets/" + owner + "/" + dataset
        count = 0
        for cell in code_cells:
            source = cell_source(cell)
            n = source.count(old)
            count += n
            if n:
                cell["source"] = source.replace(old, new).splitlines(keepends=True)
        if count < 1:
            raise RuntimeError("expected_flat_dataset_root_absent:" + dataset)
        counts[dataset] = count
    after = "\n".join(cell_source(c) for c in code_cells)
    for dataset in OWNER_BY_DATASET:
        if "/kaggle/input/" + dataset in after:
            raise RuntimeError("root_was_not_fully_repaired:" + dataset)
    if (after.count("preview = False  # always genuine model inference") != 1
        or after.count("NO_NEW_TRAINING: Gaussian fit blocked") != 1
        or after.count("NO_NEW_TRAINING: Atlas training blocked") != 1
        or "gauss.fit(" in after):
        raise RuntimeError("no_fit_inference_guards_were_changed")
    compiled = 0
    for cell in code_cells:
        src = cell_source(cell)
        if not src.startswith("%%writefile "):
            continue
        lines = src.splitlines()
        filename = lines[0].split(" ", 1)[1]
        tree = ast.parse("\n".join(lines[1:]), filename=filename)
        compile(tree, filename, "exec")
        compiled += 1
    if compiled != 37:
        raise RuntimeError("generated_module_count_changed")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("refuse_overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    metadata.pop("id_no", None)
    metadata.pop("docker_image", None)
    metadata["id"] = FROZEN_OWN_KERNEL
    metadata["title"] = "Lightning 0.694 - authentic no-fit inference"
    metadata["enable_gpu"] = True
    metadata["enable_internet"] = False
    metadata["machine_shape"] = "NvidiaTeslaT4"
    output_name = metadata.get("code_file")
    if not output_name or Path(output_name).name != output_name or not output_name.endswith(".ipynb"):
        raise RuntimeError("unsafe_notebook_name")
    body = json.dumps(notebook, ensure_ascii=False)
    (destination / output_name).write_text(body)
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return {
        "source": FROZEN_OWN_KERNEL, "original_public_source_sha": FROZEN_UPSTREAM_SHA256,
        "v1_initial_builder_sha": FROZEN_V1_INITIAL_BUILDER_SHA256,
        "v1_pulled_notebook_sha": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        "v1_code_sha": hashlib.sha256(before.encode()).hexdigest(),
        "v2_code_sha": hashlib.sha256(after.encode()).hexdigest(),
        "v2_notebook_sha": hashlib.sha256(body.encode()).hexdigest(),
        "replaced_flat_root_occurrences": counts,
        "compiled_generated_modules": compiled,
        "metadata_dataset_sources": len(metadata["dataset_sources"]),
        "metadata_model_sources": len(metadata["model_sources"]),
        "metadata_kernel_sources": len(metadata["kernel_sources"]),
        "model_training": False, "gpu_launched_by_builder": False,
        "official_competition_submission": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-own-v1",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    print("NTL0694_FROZEN_OWN_V1_MOUNT_REPAIR_V2="+json.dumps(
          build_v2(args.frozen_own_v1,args.out),sort_keys=True),flush=True)
