#!/usr/bin/env python3
"""Build a provenance-pinned, inference-only Kaggle Notebook from public V1.

Reads public Apache-2.0 notebook source; never downloads competition rows or weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

UPSTREAM = "cosmorin/name-that-llm-public620-reproduction/1"
UPSTREAM_SHA = "e9819e87b1b60231388aff0db1d0ef312b707c6c6b9911a054603333f6bb067b"
CODE_SHA = "1595f543caf314fc766a79dca313961af4d73f2a5cbea1ec2b4724aea5c4289b"
OUTPUT_ID = "kolihhan/ntl-cosmorin0622-prefit-reuse"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output directory must be empty: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    pulled = out / "_source"
    pulled.mkdir()
    res = subprocess.run(
        ["kaggle", "kernels", "pull", UPSTREAM, "-p", str(pulled), "-m"],
        text=True, capture_output=True, timeout=90,
    )
    if res.returncode:
        raise SystemExit("Published V1 source pull failed; no push/submit permitted.")
    notebooks = list(pulled.glob("*.ipynb"))
    assert len(notebooks) == 1, "Expected one notebook"
    blob = notebooks[0].read_bytes()
    assert hashlib.sha256(blob).hexdigest() == UPSTREAM_SHA, "Scored V1 source has changed"
    nb = json.loads(blob)
    code = "\n".join(
        "".join(c.get("source", [])) if isinstance(c.get("source", ""), list)
        else c.get("source", "")
        for c in nb["cells"] if c.get("cell_type") == "code"
    )
    assert hashlib.sha256(code.encode()).hexdigest() == CODE_SHA, "Source code drift"
    for require in ("joblib.load(cp)", "submission.parquet", "EXPECTED_CACHE_SHA256"):
        assert require in code, require
    for forbidden in ("SFTTrainer(", "Trainer(", "LinearSVC(", ".backward(", "optimizer.step("):
        assert forbidden not in code, "Unexpected model training: " + forbidden

    # Preserve unmodified executable cells and upstream author's attribution.
    nb["cells"][0]["source"] += [
        "\n",
        "\n",
        "Reuse note: copied unmodified executable code from Cosmorin's scored V1, ",
        "which in turn credits Nishant Dahal. This is an independent rerun, ",
        "not a novel model or a guaranteed 0.622 result. ",
        "Upstream source: https://www.kaggle.com/code/cosmorin/name-that-llm-public620-reproduction ",
        "(V1). License: Apache-2.0 for notebook code; assets retain their own terms.\n",
    ]
    original = json.loads((pulled / "kernel-metadata.json").read_text())
    assert original["id"] == "cosmorin/name-that-llm-public620-reproduction"
    assert original["enable_gpu"] is True
    assert original["enable_internet"] is False
    assert original["competition_sources"] == ["name-that-llm"]
    assert original["dataset_sources"] == [
        "cosmorin/name-that-llm-0-622-reproduction-svc-cache",
        "nishantdahal/ntl-handwriting-blend"
    ]
    assert original["model_sources"] == [
        "qwen-lm/qwen-3/Transformers/14b-base/1",
        "qwen-lm/qwen2.5/Transformers/14b/1"
    ]
    meta = {
        k:v for k,v in original.items()
        if k in {
            "id", "title", "code_file", "language", "kernel_type", "is_private",
            "enable_gpu", "enable_tpu", "enable_internet", "dataset_sources",
            "kernel_sources", "competition_sources", "model_sources",
            "machine_shape"
        }
    }
    meta["id"] = OUTPUT_ID
    meta["title"] = "NTL 0.622 public prefit reproduction (Nishant Cosmorin)"
    meta["is_private"] = True
    # Do not pretend we can force Kaggle's historical V1 Docker SHA.
    meta.pop("docker_image", None)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    (out / meta["code_file"]).write_text(json.dumps(nb, ensure_ascii=False, indent=1)+"\n")
    print("SOURCE_VERIFIED", UPSTREAM, UPSTREAM_SHA)
    print("KERNEL_PREPARED", OUTPUT_ID, "training_performed=false")

if __name__ == "__main__":
    main()
