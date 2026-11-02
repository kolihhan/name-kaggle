from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.materialize_prompt_invariant_v2_worker import materialize

KERNEL_REF = "kolihhan/name-that-llm-v2-transfer"
KERNEL_TITLE = "NTL V2 Transfer A-B"
MODEL_SOURCE = "qwen-lm/qwen-3/transformers/8b-base/1"
RUNTIME_DATASET_REF = "kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2"
POOL_DATASET_REF = "nishantdahal/ntl-handwriting-blend"
EXPECTED_POOL_SHA256 = "37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec"
EXPECTED_HOLDOUT_SOURCES = ["lucasboesen", "jadkutbi", "maniceet"]
EXPECTED_TRAIN_ROWS = 24099
EXPECTED_HELDOUT_ROWS = 10745
EXPECTED_ELIGIBLE_ROWS = 10723
EXPECTED_ELIGIBLE_LABELS = 11
CODE_FILE = "kernel_main.py"

REQUIRED_SOURCE_PATHS = (
    "scripts/__init__.py",
    "scripts/run_style_baseline.py",
    "scripts/run_strong_sparse_anchor.py",
    "scripts/run_phase14_public_sparse.py",
    "scripts/run_two_stage_lora24_gate_a.py",
    "scripts/run_two_stage_lora24_gate_a_gpu.py",
    "scripts/prompt_invariant_v2_transfer_split.py",
    "scripts/cross_prompt_prototype.py",
    "scripts/prototype_training_integration.py",
    "scripts/prompt_invariant_v2_source_eval.py",
)
V2_WORKER_PATH = "scripts/run_prompt_invariant_contrastive_v2_gate_a_gpu.py"


def offline_pip_args(wheelhouse: str) -> list[str]:
    return [
        "-m", "pip", "install", "-q", "--no-index", "--find-links", str(wheelhouse),
        "--no-deps", "transformers==4.57.1", "peft==0.17.1", "safetensors",
        "tokenizers==0.22.1", "huggingface-hub==0.36.0",
    ]


def _kernel_source(source_texts: Mapping[str, str]) -> str:
    required = set(REQUIRED_SOURCE_PATHS) | {V2_WORKER_PATH}
    missing = sorted(required - set(source_texts))
    extra = sorted(set(source_texts) - required)
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

import numpy as np
import pandas as pd

SOURCES = {sources_literal}
PIP_ARGS_TEMPLATE = {pip_literal}
EXPECTED_RUNTIME_REF = {RUNTIME_DATASET_REF!r}
EXPECTED_POOL_SHA256 = {EXPECTED_POOL_SHA256!r}
EXPECTED_HOLDOUT_SOURCES = {EXPECTED_HOLDOUT_SOURCES!r}
EXPECTED_TRAIN_ROWS = {EXPECTED_TRAIN_ROWS}
EXPECTED_HELDOUT_ROWS = {EXPECTED_HELDOUT_ROWS}
EXPECTED_ELIGIBLE_ROWS = {EXPECTED_ELIGIBLE_ROWS}
EXPECTED_ELIGIBLE_LABELS = {EXPECTED_ELIGIBLE_LABELS}


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
            raise RuntimeError(f"missing locked runtime file: {{path}}")
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != entry["sha256"] or size != int(entry["size"]):
            raise RuntimeError(f"runtime byte drift: {{entry['path']}}")
        observed.append({{"path": entry["path"], "sha256": digest, "size": size}})
    encoded = json.dumps(observed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = hashlib.sha256(encoded).hexdigest()
    if manifest != expected_manifest:
        raise RuntimeError(f"runtime manifest drift: {{manifest}} != {{expected_manifest}}")
    return manifest


def exactly_one(pattern):
    matches = sorted(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one input for {{pattern}}, found {{matches}}")
    return matches[0]


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
        if (root / "tokenizer_config.json").is_file() and any(root.glob("*.safetensors")):
            candidates.append(root)
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one Qwen3-8B root; found {{candidates}}")
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
        raise RuntimeError(f"runtime lineage drift for {{key}}")
wheelhouse = runtime_lineage_path.parent
runtime_manifest = verify_manifest(
    wheelhouse, runtime_lineage["files"], runtime_lineage["file_manifest_sha256"]
)
pip_args = [value if value != "__WHEELHOUSE__" else str(wheelhouse) for value in PIP_ARGS_TEMPLATE]
subprocess.check_call([sys.executable, *pip_args])

package_root = Path("/kaggle/working/v2_transfer_pkg")
if package_root.exists():
    shutil.rmtree(package_root)
for rel, content in SOURCES.items():
    target = package_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
sys.path.insert(0, str(package_root))

from scripts.run_phase14_public_sparse import build_conservative_subset
from scripts.prompt_invariant_v2_transfer_split import build_transfer_split
from scripts.prompt_invariant_v2_source_eval import evaluate_source_transfer
import scripts.run_two_stage_lora24_gate_a_gpu as v1_worker
import scripts.run_prompt_invariant_contrastive_v2_gate_a_gpu as v2_worker

pool_path = exactly_one("**/pool_v4.parquet")
observed_pool_sha = sha256_file(pool_path)
if observed_pool_sha != EXPECTED_POOL_SHA256:
    raise RuntimeError(f"pool SHA drift: {{observed_pool_sha}}")
raw = pd.read_parquet(pool_path)
safe = build_conservative_subset(raw, validate_full_contract=True).reset_index(drop=True)
split = build_transfer_split(safe)
if split["holdout_sources"] != EXPECTED_HOLDOUT_SOURCES:
    raise RuntimeError(f"transfer source drift: {{split['holdout_sources']}}")
if int(split["train_rows"]) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError("transfer train-row drift")
if int(split["heldout_rows"]) != EXPECTED_HELDOUT_ROWS:
    raise RuntimeError("transfer heldout-row drift")
if int(split["eligible_heldout_rows"]) != EXPECTED_ELIGIBLE_ROWS:
    raise RuntimeError("transfer eligible-row drift")
if int(split["eligible_label_count"]) != EXPECTED_ELIGIBLE_LABELS:
    raise RuntimeError("transfer eligible-label drift")
if not bool(split["valid_for_transfer_gate"]):
    raise RuntimeError("frozen transfer split is no longer valid")

safe_path = Path("/kaggle/working/phase14_safe_transfer.parquet")
safe.to_parquet(safe_path, index=False)
model_root = find_qwen_model_root()

# Instrumentation only: both workers use the same frozen alternate split.
v1_worker.build_stage1_split = build_transfer_split
v2_worker.build_stage1_split = build_transfer_split


def run_and_capture(module, name):
    captured = {{}}
    original_score = module._score_condition
    def capture(frame, heldout_idx, predictions, eligible_labels, holdout_sources):
        captured["heldout_idx"] = np.asarray(heldout_idx, dtype=np.int64).copy()
        captured["predictions"] = list(predictions)
        captured["eligible_labels"] = list(eligible_labels)
        captured["holdout_sources"] = list(holdout_sources)
        return original_score(frame, heldout_idx, predictions, eligible_labels, holdout_sources)
    module._score_condition = capture
    output = Path(f"/kaggle/working/{{name}}_worker.json")
    checkpoint = Path(f"/tmp/{{name}}_checkpoint")
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    module._run_condition_worker(
        condition="A1",
        safe_path=safe_path,
        snapshot_path=model_root,
        worker_output=output,
        checkpoint_dir=checkpoint,
    )
    worker_result = json.loads(output.read_text(encoding="utf-8"))
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    if not captured:
        raise RuntimeError(f"{{name}} worker did not expose row-level predictions")
    return captured, worker_result


v1_capture, v1_result = run_and_capture(v1_worker, "v1")
v2_capture, v2_result = run_and_capture(v2_worker, "v2")
if not np.array_equal(v1_capture["heldout_idx"], v2_capture["heldout_idx"]):
    raise RuntimeError("v1/v2 heldout row alignment differs")
if v1_capture["eligible_labels"] != v2_capture["eligible_labels"]:
    raise RuntimeError("v1/v2 eligible labels differ")

heldout_idx = v1_capture["heldout_idx"]
eval_frame = safe.iloc[heldout_idx][["source", "prompt_id", "model"]].copy().reset_index(drop=True)
eval_frame["v1_prediction"] = v1_capture["predictions"]
eval_frame["v2_prediction"] = v2_capture["predictions"]
eligible = set(v1_capture["eligible_labels"])
eval_frame = eval_frame.loc[eval_frame["model"].astype(str).isin(eligible)].reset_index(drop=True)
if len(eval_frame) != EXPECTED_ELIGIBLE_ROWS:
    raise RuntimeError(f"eligible eval row drift: {{len(eval_frame)}}")

decision, details = evaluate_source_transfer(eval_frame)
result = {{
    "candidate": "prompt-invariant-contrastive-v2",
    "scope": "paired_external_source_transfer_no_official_rows_no_public_lb",
    "pool_sha256": observed_pool_sha,
    "runtime_manifest_sha256": runtime_manifest,
    "model_source": {MODEL_SOURCE!r},
    "holdout_sources": EXPECTED_HOLDOUT_SOURCES,
    "transfer_split": {{
        k: v for k, v in split.items() if k not in {{"train_idx", "heldout_idx"}}
    }},
    "v1_worker": v1_result,
    "v2_worker": v2_result,
    "transfer_gate": details,
    "decision": decision.status,
    "official_rows_scored": 0,
    "public_lb_tuning": False,
}}
Path("/kaggle/working/paired_transfer_result.json").write_text(
    json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
)
eval_frame.to_csv("/kaggle/working/paired_transfer_predictions.csv", index=False)
print(json.dumps(result, indent=2, sort_keys=True))
'''


def write_kernel_bundle(output_dir: Path, *, repo_root: Path = Path(".")) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_texts = {
        rel: (repo_root / rel).read_text(encoding="utf-8")
        for rel in REQUIRED_SOURCE_PATHS
    }
    v1_source = source_texts["scripts/run_two_stage_lora24_gate_a_gpu.py"]
    source_texts[V2_WORKER_PATH] = materialize(v1_source)
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
        "competition_sources": [],
        "dataset_sources": [POOL_DATASET_REF, RUNTIME_DATASET_REF],
        "kernel_sources": [],
        "model_sources": [MODEL_SOURCE],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    write_kernel_bundle(args.output_dir, repo_root=args.repo_root)
    print(f"built frozen paired transfer kernel: {KERNEL_REF}")


if __name__ == "__main__":
    main()
