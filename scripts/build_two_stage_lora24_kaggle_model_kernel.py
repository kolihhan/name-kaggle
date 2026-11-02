from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from scripts.materialize_two_stage_lora24_offline_inputs import RUNTIME_DATASET_REF
from scripts.two_stage_lora24_submission import CHECKPOINT_DATASET_REF

KERNEL_REF = "kolihhan/name-that-llm-two-stage-lora24-v1-submission-preflight"
KERNEL_TITLE = "Name That LLM Two-stage LoRA24 v1 Submission Preflight"
COMPETITION = "name-that-llm"
MODEL_SOURCE = "qwen-lm/qwen-3/transformers/8b-base/1"
BASE_IDENTITY = f"kaggle-model-source:{MODEL_SOURCE}"
CODE_FILE = "kernel_main.py"
REQUIRED_SOURCE_PATHS = (
    "scripts/__init__.py",
    "scripts/run_style_baseline.py",
    "scripts/run_strong_sparse_anchor.py",
    "scripts/run_two_stage_lora24_gate_a.py",
    "scripts/run_two_stage_lora24_gate_b.py",
    "scripts/run_two_stage_lora24_gate_b_gpu.py",
    "scripts/two_stage_lora24_submission.py",
)


def offline_pip_args(wheelhouse: str) -> list[str]:
    return [
        "-m", "pip", "install", "-q", "--no-index", "--find-links", str(wheelhouse),
        "--no-deps", "transformers==4.57.1", "peft==0.17.1", "safetensors",
    ]


def _kernel_source(source_texts: Mapping[str, str]) -> str:
    missing = sorted(set(REQUIRED_SOURCE_PATHS) - set(source_texts))
    extra = sorted(set(source_texts) - set(REQUIRED_SOURCE_PATHS))
    if missing or extra:
        raise RuntimeError(f"runtime source set mismatch: missing={missing} extra={extra}")
    sources_literal = repr(dict(source_texts))
    pip_literal = repr(offline_pip_args("__WHEELHOUSE__"))
    return f'''from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

SOURCES = {sources_literal}
PIP_ARGS_TEMPLATE = {pip_literal}
EXPECTED_RUNTIME_REF = {RUNTIME_DATASET_REF!r}
EXPECTED_CHECKPOINT_REF = {CHECKPOINT_DATASET_REF!r}
EXPECTED_MODEL_SOURCE = {MODEL_SOURCE!r}
BASE_IDENTITY = {BASE_IDENTITY!r}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root, declared_files, expected_manifest):
    root = Path(root)
    observed = []
    for entry in declared_files:
        path = root / entry["path"]
        if not path.is_file():
            raise RuntimeError(f"missing locked input file: {{path}}")
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != entry["sha256"] or size != int(entry["size"]):
            raise RuntimeError(f"locked input byte drift: {{entry['path']}}")
        observed.append({{"path": entry["path"], "sha256": digest, "size": size}})
    encoded = json.dumps(observed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = hashlib.sha256(encoded).hexdigest()
    if manifest != expected_manifest:
        raise RuntimeError(f"locked input manifest drift: {{manifest}} != {{expected_manifest}}")
    return manifest


def exactly_one(pattern):
    matches = sorted(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one Kaggle input for {{pattern}}; found {{matches}}")
    return matches[0]


def first_existing(paths):
    for value in paths:
        path = Path(value)
        if path.is_file():
            return path
    raise FileNotFoundError("none of the expected competition input paths exist")


def find_qwen_model_root():
    candidates = []
    for config_path in sorted(Path("/kaggle/input/models").rglob("config.json")):
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("model_type") != "qwen3":
            continue
        if int(payload.get("hidden_size", -1)) != 4096:
            continue
        if int(payload.get("num_hidden_layers", -1)) != 36:
            continue
        root = config_path.parent
        if not (root / "tokenizer_config.json").is_file():
            continue
        if not any(root.glob("*.safetensors")):
            continue
        candidates.append(root)
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one Qwen3-8B Base model root; found {{candidates}}")
    return candidates[0]


runtime_lineage_path = exactly_one("**/runtime_lineage.json")
runtime_lineage = json.loads(runtime_lineage_path.read_text(encoding="utf-8"))
for key, expected in {{
    "dataset_ref": EXPECTED_RUNTIME_REF,
    "dataset_version_expected": 1,
    "competition_train_rows_included": 0,
    "competition_test_rows_included": 0,
}}.items():
    if runtime_lineage.get(key) != expected:
        raise RuntimeError(f"runtime lineage drift for {{key}}: {{runtime_lineage.get(key)!r}} != {{expected!r}}")
wheelhouse = runtime_lineage_path.parent
runtime_manifest = verify_manifest(
    wheelhouse, runtime_lineage["files"], runtime_lineage["file_manifest_sha256"]
)
pip_args = [value if value != "__WHEELHOUSE__" else str(wheelhouse) for value in PIP_ARGS_TEMPLATE]
subprocess.check_call([sys.executable, *pip_args])

package_root = Path("/kaggle/working/lora24_submission_pkg")
if package_root.exists():
    shutil.rmtree(package_root)
for rel, content in SOURCES.items():
    target = package_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
sys.path.insert(0, str(package_root))

model_root = find_qwen_model_root()
checkpoint_meta = exactly_one("**/a1-checkpoint/meta.json")
checkpoint_dir = checkpoint_meta.parent
train_path = first_existing([
    "/kaggle/input/name-that-llm/train.csv",
    "/kaggle/input/competitions/name-that-llm/train.csv",
])
test_path = first_existing([
    "/kaggle/input/name-that-llm/test.csv",
    "/kaggle/input/competitions/name-that-llm/test.csv",
])

from scripts.two_stage_lora24_submission import run_final_fit

lineage = run_final_fit(
    train_path=train_path,
    test_path=test_path,
    checkpoint_dir=checkpoint_dir,
    snapshot_path=model_root,
    output_dir=Path("/kaggle/working"),
    base_manifest_sha256=BASE_IDENTITY,
    runtime_manifest_sha256=runtime_manifest,
)
lineage_path = Path("/kaggle/working/submission_lineage.json")
submission_path = Path("/kaggle/working/submission.parquet")
if not lineage_path.is_file() or not submission_path.is_file():
    raise RuntimeError("final fit did not produce required submission outputs")
print("MODEL_ROOT", model_root)
print("MODEL_SOURCE", EXPECTED_MODEL_SOURCE)
print(json.dumps(lineage, indent=2, sort_keys=True))
'''


def write_kernel_bundle(output_dir: Path, *, source_texts: Mapping[str, str] | None = None, repo_root: Path = Path(".")) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_texts is None:
        source_texts = {rel: (Path(repo_root) / rel).read_text(encoding="utf-8") for rel in REQUIRED_SOURCE_PATHS}
    (output_dir / CODE_FILE).write_text(_kernel_source(source_texts), encoding="utf-8")
    metadata = {
        "id": KERNEL_REF,
        "title": KERNEL_TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [COMPETITION],
        "dataset_sources": [CHECKPOINT_DATASET_REF, RUNTIME_DATASET_REF],
        "kernel_sources": [],
        "model_sources": [MODEL_SOURCE],
    }
    (output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline LoRA24 v1 kernel using Kaggle Qwen3-8B Base model source")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_kernel_bundle(args.output_dir, repo_root=args.repo_root)
    print(f"built offline model-source preflight kernel: {KERNEL_REF}")


if __name__ == "__main__":
    main()
