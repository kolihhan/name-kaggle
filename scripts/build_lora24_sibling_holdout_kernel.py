"""Private Kaggle GPU *inference-only* package for the frozen Stage-1 holdout.

No competition data is mounted. No training or competition submission command is
ever generated. Source modules are taken from the current checkout, preserving
the existing Stage-1 split, checkpoint-loader and inference primitives.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_two_stage_lora24_kaggle_model_kernel import offline_pip_args

KERNEL_REF = "kolihhan/ntl-lora24-stage1-sibling-holdout-v1"
KERNEL_TITLE = "NTL LoRA24 Stage1 Sibling Holdout V1"
CODE_FILE = "kernel_main.py"
MODEL_SOURCE = "qwen-lm/qwen-3/transformers/8b-base/1"
POOL_DATASET = "nishantdahal/ntl-handwriting-blend"
CHECKPOINT_DATASET = "kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint"
RUNTIME_DATASET = "kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2"
SOURCES = (
    "scripts/__init__.py",
    "scripts/run_style_baseline.py",
    "scripts/run_strong_sparse_anchor.py",
    "scripts/run_phase14_public_sparse.py",
    "scripts/run_two_stage_lora24_gate_a.py",
    "scripts/run_two_stage_lora24_gate_b.py",
    "scripts/run_two_stage_lora24_gate_b_gpu.py",
    "scripts/audit_lora24_sibling_holdout.py",
    "scripts/lora24_sibling_holdout_inference.py",
)


def kernel_source(source_texts: Mapping[str, str]) -> str:
    if set(source_texts) != set(SOURCES):
        raise RuntimeError("exact source set drift")
    pip_args = repr(offline_pip_args("__WHEELHOUSE__"))
    return f'''# GENERATED private, no-training Kaggle script
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCES = {repr(dict(source_texts))}
PIP = {pip_args}
EXPECTED_RUNTIME = {RUNTIME_DATASET!r}
EXPECTED_MODEL = {MODEL_SOURCE!r}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique(pattern):
    matches = sorted(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {{pattern}}, found {{len(matches)}}")
    return matches[0]


def verify_runtime(lineage_path):
    record = json.loads(lineage_path.read_text(encoding="utf-8"))
    for key, want in {{
        "dataset_ref": EXPECTED_RUNTIME,
        "dataset_version_expected": 1,
        "competition_train_rows_included": 0,
        "competition_test_rows_included": 0,
    }}.items():
        if record.get(key) != want:
            raise RuntimeError(f"offline runtime {{key}} mismatch")
    observed = []
    for item in record["files"]:
        f = lineage_path.parent / item["path"]
        if not f.is_file() or f.stat().st_size != int(item["size"]):
            raise RuntimeError("offline wheelhouse file missing or size drift")
        digest = sha256_file(f)
        if digest != item["sha256"]:
            raise RuntimeError("offline wheelhouse file hash mismatch")
        observed.append({{"path": item["path"], "sha256": digest, "size": int(item["size"])}})
    blob = json.dumps(observed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(blob).hexdigest() != record["file_manifest_sha256"]:
        raise RuntimeError("offline wheelhouse manifest mismatch")
    return lineage_path.parent


def find_qwen():
    candidates = []
    for file in sorted(Path("/kaggle/input/models").rglob("config.json")):
        try:
            v = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (v.get("model_type"), v.get("hidden_size"), v.get("num_hidden_layers")) != ("qwen3", 4096, 36):
            continue
        parent = file.parent
        if (parent / "tokenizer_config.json").is_file() and list(parent.glob("*.safetensors")):
            candidates.append(parent)
    if len(candidates) != 1:
        raise RuntimeError(f"Qwen3-8B root must be unique (found {{len(candidates)}})")
    return candidates[0]


wheelhouse = verify_runtime(unique("**/runtime_lineage.json"))
args = [v if v != "__WHEELHOUSE__" else str(wheelhouse) for v in PIP]
subprocess.check_call([sys.executable, *args])
package_root = Path("/kaggle/working/ntl_lora24_sibling_package")
if package_root.exists():
    shutil.rmtree(package_root)
for name, source in SOURCES.items():
    dest = package_root / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(source, encoding="utf-8")
sys.path.insert(0, str(package_root))

checkpoint = unique("**/a1-checkpoint/meta.json").parent
pool = unique("**/pool_v4.parquet")
model = find_qwen()

from scripts.lora24_sibling_holdout_inference import run

report = run(
    pool_path=pool,
    checkpoint_dir=checkpoint,
    model_root=model,
    output_dir=Path("/kaggle/working"),
)
assert report["no_training"] is True
assert report["no_kaggle_competition_submission"] is True
assert report["not_the_final_0520_stage2_model"] is True
assert len(report["sources"]) == 3
print("INFERENCE_ONLY_SUMMARY_COMPLETE", json.dumps({{
    "rows": report["pooled"]["rows"],
    "within_family_wrong": report["pooled"]["within_family_wrong"],
    "all_wrong": report["pooled"]["wrong"],
    "stage1_only": True,
}}, sort_keys=True), flush=True)
'''


def write_kernel_bundle(
    output_dir: Path,
    *,
    repo_root: Path = Path("."),
    source_texts: Mapping[str, str] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_texts is None:
        source_texts = {
            name: (Path(repo_root) / name).read_text(encoding="utf-8")
            for name in SOURCES
        }
    (output_dir / CODE_FILE).write_text(kernel_source(source_texts), encoding="utf-8")
    metadata = {
        "id": KERNEL_REF,
        "title": KERNEL_TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "dataset_sources": [POOL_DATASET, CHECKPOINT_DATASET, RUNTIME_DATASET],
        "kernel_sources": [],
        "model_sources": [MODEL_SOURCE],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    args = parser.parse_args()
    write_kernel_bundle(args.output_dir, repo_root=args.repo_root)


if __name__ == "__main__":
    main()
