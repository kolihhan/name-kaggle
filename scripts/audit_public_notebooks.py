"""Read-only static audit of public Name That LLM Kaggle notebook code.

No competition dataset, checkpoints, inference or training are downloaded/executed.
Only source notebooks and public asset metadata are read.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

TARGETS = [
    ("lightning-666", [
        "lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble",
        "lightningv08/0-641-lora-llama-qwen3-8-27b",
    ]),
    ("lightning-642", [
        "lightningv08/0-636-lora-llama-qwen3-8-27b",
    ]),
    ("cosmorin-622", [
        "cosmorin/name-that-llm-public620-reproduction",
    ]),
    ("control-old-public", ["nishantdahal/two-stage-lora-part-two"]),
    ("alexchilton-667", [
        "alexchilton/name-that-llm-0-667-all-public-what-failed",
        "alexchilton/name-that-llm-0-662-all-public-what-failed",
    ]),
]
INPUT_PATTERN = re.compile(r"/kaggle/input/([A-Za-z0-9_./-]+)")
URL_PATTERN = re.compile(r"https?://[A-Za-z0-9_./?=&%#-]+")
MARKERS = {
    "trains_at_notebook_runtime": [
        r"\.fit\s*\(", r"\.train\s*\(", r"\.backward\s*\(", r"\.optimizer_step\s*\(",
        r"\bTrainer\s*\(", r"\bSFTTrainer\s*\(",
    ],
    "loads_existing_model_assets": [
        r"\btorch\.load\s*\(", r"\bjoblib\.load\s*\(", r"load_model\s*\(",
        r"from_pretrained\s*\(", r"load_adapter\s*\(",
        r"adapter_model\.safetensors", r"\.safetensors",
    ],
    "ensemble_or_voting": [
        r"ensemble", r"weighted", r"\bblend\b", r"\bvot(?:e|ing)\b",
        r"argmax", r"probab", r"softmax",
    ],
    "competition_training_rows_referenced": [
        r"train\.csv", r"train_path", r"competition_train",
    ],
    "writes_submission": [
        r"submission\.parquet", r"to_parquet\s*\(",
    ],
}


def source_cells(folder: Path):
    sources = []
    for file in sorted(folder.iterdir()):
        if file.suffix == ".ipynb":
            nb = json.loads(file.read_text(encoding="utf-8"))
            code = "\n\n".join(
                "".join(c.get("source", []))
                if isinstance(c.get("source", ""), list)
                else c.get("source", "")
                for c in nb.get("cells", [])
                if c.get("cell_type") == "code"
            )
            sources.append((file, code))
        elif file.suffix == ".py":
            sources.append((file, file.read_text(encoding="utf-8")))
    return sources


def pull(label: str, refs: list[str], folder: Path):
    attempted = []
    for ref in refs:
        result = subprocess.run(
            ["kaggle", "kernels", "pull", ref, "-p", str(folder), "-m"],
            capture_output=True, text=True, timeout=90,
        )
        err = (result.stderr or result.stdout or "").lower()
        cause = ("auth" if "401" in err or "unauthorized" in err else
                 "forbidden" if "403" in err or "forbidden" in err else
                 "not_found" if "404" in err or "not found" in err else
                 "cli_arguments" if "usage:" in err or "argument" in err else
                 "other")
        attempted.append({"ref": ref, "ok": result.returncode == 0, "error_class": cause})
        if result.returncode == 0 and source_cells(folder):
            return ref, attempted
    return None, attempted


def audit_notebook(label: str, refs: list[str], root: Path):
    folder = root / label
    folder.mkdir(parents=True, exist_ok=True)
    pulled, attempted = pull(label, refs, folder)
    if not pulled:
        return {"label": label, "status": "PULL_BLOCKED", "attempts": attempted}
    srcs = source_cells(folder)
    code = "\n".join(c for _, c in srcs)
    metadata_path = folder / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    result = {
        "label": label,
        "status": "SOURCE_READ",
        "pulled_ref": pulled,
        "attempts": attempted,
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "source_bytes": len(code.encode()),
        "source_files": [{"name": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f, _ in srcs],
        "notebook_metadata": {
            key: metadata.get(key) for key in (
                "id", "title", "language", "kernel_type", "is_private",
                "enable_gpu", "enable_internet", "dataset_sources",
                "competition_sources", "kernel_sources", "model_sources",
            ) if key in metadata
        },
        "literal_input_paths": sorted(set(m.group(1).rstrip("),;:") for m in INPUT_PATTERN.finditer(code)))[:150],
        "external_urls": sorted(set(URL_PATTERN.findall(code)))[:70],
        "markers": {
            group: {pattern: bool(re.search(pattern, code, re.IGNORECASE))
                    for pattern in expressions}
            for group, expressions in MARKERS.items()
        },
        "imports": sorted(set(
            re.findall(r"(?m)^\s*(?:from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import|import\s+([A-Za-z_][A-Za-z0-9_.]*))",
                       code)
        )),
        "static_only": True,
        "checkpoint_downloaded": False,
        "prediction_run": False,
        "official_train_downloaded": False,
        "warning": "Source inputs/markers show possible dependencies, NOT verified asset availability or live rerun behavior.",
    }
    # Extract class/function names as architecture hints; cannot prove execution.
    result["code_functions"] = sorted(set(re.findall(r"(?m)^def\s+([A-Za-z_]\w*)\s*\(", code)))[:100]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN missing")
    root = Path("/tmp/ntl-public-notebooks")
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for label, refs in TARGETS:
        print(f"AUDIT_BEGIN {label}", flush=True)
        try:
            result = audit_notebook(label, refs, root)
        except Exception as exc:
            result = {"label": label, "status": "AUDIT_ERROR", "error_type": type(exc).__name__}
        results.append(result)
        print(f"AUDIT_RESULT {label} {result['status']} {result.get('pulled_ref', '')}", flush=True)
    report = {
        "scope": "public_notebook_source_and_metadata_only",
        "version": 1,
        "training_performed": False,
        "prediction_performed": False,
        "kaggle_submission_performed": False,
        "official_data_downloaded": False,
        "target_count": len(TARGETS),
        "source_read_count": sum(x["status"] == "SOURCE_READ" for x in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"AUDIT_COMPLETE {report['source_read_count']}/{len(TARGETS)} source notebooks read")
    if not report["source_read_count"]:
        raise SystemExit("No public notebook sources could be fetched; report retained")


if __name__ == "__main__":
    main()
