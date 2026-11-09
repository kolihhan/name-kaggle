"""One-shot reuse of public Lightning 0.681 Notebook with hard no-training guards.

Only Kaggle remote execution is allowed, using the author's fixed pretrained
neural weights, prefit SVC and Gaussian states. This is a reproduction, not a
new scientific contribution. No text/official competition data enters Git.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SOURCE_REF = "lightningv08/0-681-qwen3-embedding-lora-qwen3-8-27b"
DEST_REF = "kolihhan/ntl-public-lightning-0681-frozen-no-train"
COMPETITION = "name-that-llm"
DESCRIPTION = "Frozen public Lightning 0.681 reproduction no new supervised training 20261109"
EXPECTED_PREFIT = {
    "lightningv08/ntl-olmo2-native24-final-inference-assets",
    "nuridudak/ntl-svc-prefit-pool-v4n",
}
CONTRACT_PATH = Path("artifacts/candidates/public-lightning-0681-no-neural-training-repro/CONTRACT.md")


def run(args: list[str], *, checked: bool = True) -> str:
    p = subprocess.run(args, capture_output=True, text=True)
    if checked and p.returncode:
        msg = "\n".join(
            line for line in (p.stdout + "\n" + p.stderr).splitlines()
            if re.search(r"error|failed|timeout|quota|permission|invalid|not found|blocked", line, re.I)
        )
        raise RuntimeError(f"CLI failed {args[:3]}: {(msg or p.stderr)[-1300:]}")
    return (p.stdout + "\n" + p.stderr).strip()


def official_rows() -> list[dict]:
    return list(csv.DictReader(run([
        "kaggle", "competitions", "submissions", COMPETITION, "-v", "-q"
    ]).splitlines()))


def check_no_prior() -> None:
    if any(row.get("description") == DESCRIPTION for row in official_rows()):
        raise RuntimeError("duplicate competition entry would violate one-shot agreement")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_reproduction(destination: Path) -> dict:
    if not CONTRACT_PATH.is_file():
        raise RuntimeError("frozen scientific/reproduction contract missing")
    source_dir = destination / "source"
    target_dir = destination / "kernel"
    source_dir.mkdir(parents=True, exist_ok=False)
    target_dir.mkdir(parents=True, exist_ok=False)

    out = run(["kaggle", "kernels", "pull", SOURCE_REF, "-p", str(source_dir), "-m"])
    if not out:
        raise RuntimeError("empty public source pull")
    source_meta = json.loads((source_dir / "kernel-metadata.json").read_text())
    if (source_meta.get("id") != SOURCE_REF or
        source_meta.get("is_private") is not False or
        source_meta.get("enable_internet") is not False or
        source_meta.get("kernel_type") != "notebook" or
        source_meta.get("competition_sources") != [COMPETITION]):
        raise RuntimeError("public source identity or submission policy drift")
    sources = source_meta.get("dataset_sources") or []
    models = source_meta.get("model_sources") or []
    if (len(sources) != 21 or len(models) != 6 or
        not EXPECTED_PREFIT.issubset(set(sources))):
        raise RuntimeError("upstream pretrained artifact inputs changed")
    notebooks = list(source_dir.glob("*.ipynb"))
    if len(notebooks) != 1:
        raise RuntimeError("public source must be exactly one notebook")
    original = notebooks[0]
    notebook = json.loads(original.read_text(encoding="utf-8"))
    original_sha = _sha(original)

    cells = notebook.get("cells", [])
    texts = ["".join(cell.get("source", [])) if cell.get("cell_type") == "code" else ""
             for cell in cells]
    corpus = "\n".join(texts)
    if (not any("CONFIG = " in s and "'RELEASE_READY': True" in s for s in texts) or
        len(re.findall(r"\.(?:fit|fit_transform|partial_fit)\s*\(", corpus)) != 4 or
        len(re.findall(r"\.train\s*\(", corpus)) != 1 or
        re.search(r"\.(?:backward|step|zero_grad)\s*\(", corpus) or
        re.search(r"\btorch\.optim\b", corpus)):
        raise RuntimeError("upstream pretrained-vs-training call inventory drift")
    if "source-test-conditioned" not in corpus or "fixed-no-test-redo" not in corpus:
        raise RuntimeError("original Gaussian prefit fallback identity not auditable")

    guard_gaussian = (
        "import gauss\n"
        "\ndef _no_new_supervised_gaussian_fit(*args, **kwargs):\n"
        "    raise RuntimeError('NO_NEW_TRAINING: supervised Gaussian fit forbidden')\n"
        "gauss.fit = _no_new_supervised_gaussian_fit\n"
    )
    count_gauss = 0
    count_atlas = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        code = "".join(src) if isinstance(src, list) else str(src)
        if "def member_scores" in code and "import gauss\n" in code:
            if count_gauss:
                raise RuntimeError("ambiguous Gaussian fit guard target")
            code = code.replace("import gauss\n", guard_gaussian, 1)
            count_gauss += 1
        if "model = AtlasModel(base, labels, config)" in code and "if training:" in code:
            if count_atlas:
                raise RuntimeError("ambiguous Atlas train-mode guard target")
            match = "if training:\n        base.gradient_checkpointing_enable"
            if match not in code:
                raise RuntimeError("Atlas training guard exact source changed")
            code = code.replace(
                match,
                "if training:\n"
                "        raise RuntimeError('NO_NEW_TRAINING: Atlas train mode forbidden')\n"
                "        base.gradient_checkpointing_enable", 1
            )
            count_atlas += 1
        cell["source"] = code.splitlines(keepends=True)
    if (count_gauss, count_atlas) != (1, 1):
        raise RuntimeError(f"training guard targets incomplete {(count_gauss, count_atlas)}")

    # Credit original author: no generated text, weights, or competition data.
    notebook["cells"].insert(0, {
        "cell_type": "markdown",
        "id": "repro0681credit",
        "metadata": {},
        "source": [
            "Frozen reproduction of [Lightningv08's public 0.681 Name That LLM Notebook]"
            "(https://www.kaggle.com/code/lightningv08/0-681-qwen3-embedding-lora-qwen3-8-27b), "
            "by Vladislav Lavrentev. All original source notices retained. "
            "No new neural or supervised Gaussian training permitted; "
            "only original unlabeled query-text TF-IDF grouping is allowed. "
            "External author's 0.681 is not this reproduction's score.\n"
        ],
    })
    # Kaggle CLI's metadata contract prefers numeric id_no over textual id.
    # A public notebook pull may include the upstream author's numeric ID;
    # retaining it makes SaveKernel try to update THEIR notebook -> HTTP 403.
    # Always remove inherited identity when creating a new own notebook.
    upstream_id_no_present = "id_no" in source_meta
    source_meta.pop("id_no", None)
    print("PUBLIC_0681_UPSTREAM_ID_NO_PRESENT", upstream_id_no_present, flush=True)
    source_meta.update({
        "id": DEST_REF,
        "title": DEST_REF.split("/", 1)[1],
        "is_private": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
    })
    if "id_no" in source_meta or source_meta["id"] != DEST_REF:
        raise RuntimeError("original public Notebook numeric identity leaked into new kernel")
    out_name = source_meta.get("code_file")
    if not out_name or Path(out_name).name != out_name or not out_name.endswith(".ipynb"):
        raise RuntimeError("unsafe or missing notebook code_file")
    (target_dir / out_name).write_text(
        json.dumps(notebook, ensure_ascii=False), encoding="utf-8"
    )
    (target_dir / "kernel-metadata.json").write_text(
        json.dumps(source_meta, indent=2), encoding="utf-8"
    )
    built = json.loads((target_dir / out_name).read_text(encoding="utf-8"))
    full = "\n".join("".join(c.get("source", [])) for c in built["cells"]
                     if c.get("cell_type") == "code")
    if (full.count("gauss.fit = _no_new_supervised_gaussian_fit") != 1 or
        full.count("NO_NEW_TRAINING: Atlas train mode forbidden") != 1):
        raise RuntimeError("training guards not persisted")
    print("PUBLIC_0681_FROZEN_ORIGINAL_NOTEBOOK_SHA256", original_sha, flush=True)
    print("PUBLIC_0681_FROZEN_REPRO_NOTEBOOK_SHA256",
          _sha(target_dir / out_name), flush=True)
    print("PUBLIC_0681_NO_NEW_SUPERVISED_TRAINING_GUARDS_VERIFIED", flush=True)
    return {"folder": target_dir, "original_sha256": original_sha}


def kernel_status() -> str:
    raw = run(["kaggle", "kernels", "status", DEST_REF])
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", raw)
    if not m:
        raise RuntimeError("Kaggle kernel status is unknown")
    return m.group(1)


def wait_for_preview() -> None:
    for i in range(150):
        status = kernel_status()
        print("PUBLIC_0681_PREVIEW_KERNEL_STATUS", i + 1, status, flush=True)
        if status in ("ERROR", "FAILED", "CANCELLED"):
            raise RuntimeError("frozen inference preview kernel failed; do not retrain")
        if status == "COMPLETE":
            break
        if i == 149:
            raise RuntimeError("Kaggle inference preview timeout, no submission")
        time.sleep(120)
    text = run(["kaggle", "kernels", "files", DEST_REF, "-v", "--page-size", "200"])
    if ("Next Page Token =" in text or "submission.parquet" not in text):
        raise RuntimeError("preview output absent or paginated unexpectedly")
    print("PUBLIC_0681_PREVIEW_OUTPUT_READY", flush=True)


def submit_once() -> None:
    check_no_prior()
    if "KernelWorkerStatus." in run(
        ["kaggle", "kernels", "status", DEST_REF], checked=False
    ):
        raise RuntimeError("target Kaggle kernel already exists; no silent overwrite")

    status = run(["kaggle", "kernels", "status", SOURCE_REF])
    if "KernelWorkerStatus.COMPLETE" not in status:
        raise RuntimeError("published source kernel not COMPLETE")
    frozen = frozen_reproduction(Path("/tmp/ntl-public-0681"))
    check_no_prior()

    pushed = run([
        "kaggle", "kernels", "push", "-p", str(frozen["folder"]),
        "--accelerator", "NvidiaTeslaT4", "--timeout", "32400"
    ])
    if "Kernel version 1 successfully pushed" not in pushed:
        raise RuntimeError("expected one new frozen Notebook v1; no resubmit")
    print("PUBLIC_0681_OWN_KERNEL_VERSION1_PUSHED_WITH_NO_TRAINING_GUARDS", flush=True)
    wait_for_preview()
    check_no_prior()

    response = run([
        "kaggle", "competitions", "submit", COMPETITION,
        "-k", DEST_REF, "-v", "1", "-f", "submission.parquet",
        "-m", DESCRIPTION
    ])
    refs = re.findall(r"Submission ref:\s*(\d+)", response, re.I)
    if len(refs) != 1:
        raise RuntimeError("submission not provably accepted: " + response[-500:])
    accepted = refs[0]
    print("PUBLIC_0681_OWN_OFFICIAL_SUBMISSION_ID", accepted, flush=True)
    for _ in range(20):
        matches = [x for x in official_rows() if x.get("ref") == accepted]
        if len(matches) == 1 and matches[0].get("description") == DESCRIPTION:
            row = matches[0]
            print("PUBLIC_0681_OWN_OFFICIAL_LEDGER", json.dumps({
                k: row.get(k) for k in
                ("ref", "status", "publicScore", "privateScore", "date", "description")
            }, sort_keys=True), flush=True)
            return
        time.sleep(15)
    raise RuntimeError("official Kaggle accepted but ledger not yet consistent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    if args.static_only:
        # Offline checks are separately executed via tests. This mode must not
        # require Kaggle credentials, download data, or push a Notebook.
        if not CONTRACT_PATH.is_file():
            raise RuntimeError("contract missing")
        print("PUBLIC_0681_STATIC_PRECHECK_PASS")
    else:
        submit_once()
