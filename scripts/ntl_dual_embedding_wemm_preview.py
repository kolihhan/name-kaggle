"""Private preview of pinned public dual-embedding + WeMM Notebook with no-training guards.

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

SOURCE_REF = "ozertuu/llm-dual-embedding-lora-qwen3-8-27b-gsq"
DEST_REF = "kolihhan/ntl-dual-embedding-wemm-frozen-v1"
COMPETITION = "name-that-llm"
DESCRIPTION = "WEMM frozen dual embedding exploratory no-training 20261112"
EXPECTED_PREFIT = {
    "lightningv08/ntl-olmo2-native24-final-inference-assets",
    "nuridudak/ntl-svc-prefit-pool-v4n",
}
CONTRACT_PATH = Path("artifacts/candidates/dual-embedding-wemm-frozen-v1/CONTRACT.md")


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
    if (len(sources) != 21 or len(models) != 7 or len(source_meta.get('kernel_sources') or []) != 8 or
        not EXPECTED_PREFIT.issubset(set(sources)) or
        "dangkhoa2016/tencent-wemm-embedding-4b/Transformers/default/1" not in models):
        raise RuntimeError("upstream pretrained artifact inputs changed")
    notebooks = list(source_dir.glob("*.ipynb"))
    if len(notebooks) != 1:
        raise RuntimeError("public source must be exactly one notebook")
    original = notebooks[0]
    notebook = json.loads(original.read_text(encoding="utf-8"))
    original_sha = _sha(original)
    source_code_cells = [
        "".join(c.get("source", [])) if isinstance(c.get("source", []), list)
        else str(c.get("source", ""))
        for c in notebook.get("cells", []) if c.get("cell_type") == "code"
    ]
    code_digest = hashlib.sha256(json.dumps(
        source_code_cells, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    # The Notebook container/markdown can be revised; executable code must
    # be exact. A changed digest always aborts before any GPU/competition call.
    expected_code_digest = "a65e0115aaa56e3ce64391e9911fed7d3521c7ace3a1e990d92cc4f1b8a03a82"
    if code_digest != expected_code_digest or len(source_code_cells) != 43:
        raise RuntimeError("source executable code changed; no promotion")
    cells = notebook.get("cells", [])
    texts = ["".join(cell.get("source", [])) if cell.get("cell_type") == "code"
             else "" for cell in cells]
    corpus = "\n".join(texts)
    # Exact six training/fit callsites are inspected in the source audit.
    # Three TF-IDF fit_transform are unsupervised prompt-group features,
    # one gauss.fit is blocked, two supervised model.fit are removed below.
    if (not any("CONFIG = " in x and "'RELEASE_READY': True" in x for x in texts) or
        len(re.findall(r"\.(?:fit|fit_transform|partial_fit)\s*\(", corpus)) != 6 or
        len(re.findall(r"\.train\s*\(", corpus)) != 1 or
        re.search(r"\.(?:backward|step|zero_grad)\s*\(", corpus) or
        re.search(r"\btorch\.optim\b", corpus) or
        "source-test-conditioned" not in corpus or
        "fixed-no-test-redo" not in corpus):
        raise RuntimeError("unexpected source training path inventory")
    style_cell = source_code_cells[33]
    if (not style_cell.startswith("%%writefile /kaggle/temp/ntl-gsq-v27-runtime/sibling_style.py") or
        style_cell.count("model.fit(X[a], target[a])") != 1 or
        style_cell.count("model.fit(X, target)") != 1 or
        style_cell.count("def apply_grouped_style_residual(") != 1):
        raise RuntimeError("source supervised sibling-style code drift")
    print("NTL_WEMM_SOURCE_PIN_VERIFIED", json.dumps({
        "code_sha256": code_digest, "notebook_sha256": original_sha,
        "changed_noncode_container_ok": original_sha !=
            "e943174f3cfe114eeb96d9f721c675826f6ef8a91e30ae055dd86e5c48257865",
        "fit_calls_original": 6, "style_supervised_fit_sites_to_remove": 2,
        "new_training_allowed": False,
    }, sort_keys=True),flush=True)

    guard_gaussian = (
        "import gauss\n"
        "\ndef _no_new_supervised_gaussian_fit(*args, **kwargs):\n"
        "    raise RuntimeError('NO_NEW_TRAINING: supervised Gaussian fit forbidden')\n"
        "gauss.fit = _no_new_supervised_gaussian_fit\n"
    )
    count_gauss = 0
    count_atlas = 0
    count_core_guard = 0
    count_style = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        code = "".join(src) if isinstance(src, list) else str(src)
        if "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/sibling_style.py" in code:
            if count_style:
                raise RuntimeError("more than one sibling-style cell")
            start = code.index("def apply_grouped_style_residual(")
            # No new supervised fitting: remove the entire CV+final LogisticRegression
            # method. Preserve unrelated style-feature functions for imports.
            code = code[:start] + (
                "def apply_grouped_style_residual(logits, train_text, test_text, "
                "labels, y, groups, max_alpha=0.10, diagnostics_path=None):\n"
                "    # Frozen WeMM experiment: sibling-style training is disabled.\n"
                "    return np.asarray(logits, dtype=np.float64).copy()\n"
            )
            count_style += 1
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
        if "def l24_features" in code and "'--train-final'" in code:
            if count_core_guard:
                raise RuntimeError("ambiguous core_worker training guard")
            needle = "args = parser.parse_args()\n    config ="
            if needle not in code:
                raise RuntimeError("core worker CLI training guard target changed")
            code = code.replace(needle,
                "args = parser.parse_args()\n"
                "    if args.train_final: raise RuntimeError('NO_NEW_TRAINING: core worker train_final forbidden')\n"
                "    config =", 1)
            count_core_guard += 1
        cell["source"] = code.splitlines(keepends=True)
    if (count_gauss, count_atlas, count_core_guard, count_style) != (1, 1, 1, 1):
        raise RuntimeError(f"training guard targets incomplete {(count_gauss, count_atlas)}")

    # Credit original author: no generated text, weights, or competition data.
    notebook["cells"].insert(0, {
        "cell_type": "markdown",
        "id": "wemmsourcecredit",
        "metadata": {},
        "source": [
            "Guarded reproduction of [Ozertuu's public dual-embedding + WeMM Name That LLM Notebook]"
            "(https://www.kaggle.com/code/ozertuu/llm-dual-embedding-lora-qwen3-8-27b-gsq), "
            "by Ertuğrul Özer. All original source notices retained. "
            "No new neural or supervised Gaussian training permitted; "
            "only original unlabeled query-text TF-IDF grouping is allowed. "
            "Upstream title is not an authenticated score for this private copy.\n"
        ],
    })
    # Kaggle CLI's metadata contract prefers numeric id_no over textual id.
    # A public notebook pull may include the upstream author's numeric ID;
    # retaining it makes SaveKernel try to update THEIR notebook -> HTTP 403.
    # Always remove inherited identity when creating a new own notebook.
    upstream_id_no_present = "id_no" in source_meta
    source_meta.pop("id_no", None)
    print("NTL_WEMM_V1_UPSTREAM_ID_NO_PRESENT", upstream_id_no_present, flush=True)
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
    if (full.count("NO_NEW_TRAINING: core worker train_final forbidden") != 1 or
        full.count("gauss.fit = _no_new_supervised_gaussian_fit") != 1 or
        full.count("NO_NEW_TRAINING: Atlas train mode forbidden") != 1 or
        "model.fit(X[a], target[a])" in full or
        "model.fit(X, target)" in full):
        raise RuntimeError("training guards not persisted")
    print("NTL_WEMM_V1_FROZEN_ORIGINAL_NOTEBOOK_SHA256", original_sha, flush=True)
    print("NTL_WEMM_V1_FROZEN_REPRO_NOTEBOOK_SHA256",
          _sha(target_dir / out_name), flush=True)
    print("NTL_WEMM_V1_NO_NEW_SUPERVISED_TRAINING_GUARDS_VERIFIED", flush=True)
    return {"folder": target_dir, "original_sha256": original_sha}


def kernel_status() -> str:
    raw = run(["kaggle", "kernels", "status", DEST_REF])
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", raw)
    if not m:
        raise RuntimeError("Kaggle kernel status is unknown")
    return m.group(1)


def wait_for_preview() -> None:
    for i in range(60):
        status = kernel_status()
        print("NTL_WEMM_V1_PREVIEW_KERNEL_STATUS", i + 1, status, flush=True)
        if status in ("ERROR", "FAILED", "CANCELLED"):
            raise RuntimeError("frozen inference preview kernel failed; do not retrain")
        if status == "COMPLETE":
            break
        if i == 59:
            raise RuntimeError("Kaggle inference preview timeout, no submission")
        time.sleep(30)
    text = run(["kaggle", "kernels", "files", DEST_REF, "-v", "--page-size", "200"])
    if ("Next Page Token =" in text or "submission.parquet" not in text):
        raise RuntimeError("preview output absent or paginated unexpectedly")
    print("NTL_WEMM_V1_PREVIEW_OUTPUT_READY", flush=True)


def private_preview_once() -> None:
    # The only persistent Kaggle write is a NEW private notebook v1; never submit.
    if "KernelWorkerStatus." in run(
        ["kaggle", "kernels", "status", DEST_REF], checked=False
    ):
        raise RuntimeError("destination already exists; refuse overwrite or duplicate run")
    if "KernelWorkerStatus.COMPLETE" not in run(
        ["kaggle", "kernels", "status", SOURCE_REF]
    ):
        raise RuntimeError("public source preview not COMPLETE; no fork")
    frozen = frozen_reproduction(Path("/tmp/ntl-dual-wemm-preview"))
    output = run(["kaggle", "kernels", "push", "-p", str(frozen["folder"]),
                  "--accelerator", "NvidiaTeslaT4", "--timeout", "32400"])
    if "Kernel version 1 successfully pushed" not in output:
        raise RuntimeError("private preview not provably created")
    print("NTL_WEMM_V1_PRIVATE_VERSION1_CREATED_NO_COMPETITION_SUBMISSION", flush=True)
    wait_for_preview()
    print("NTL_WEMM_V1_PRIVATE_PREVIEW_OUTPUT_CONFIRMED", flush=True)



def competition_submit_once() -> None:
    """One user-authorized experimental competition entry after private preview."""
    check_no_prior()
    anchor = [r for r in official_rows() if r.get("ref") == "58210835"]
    if len(anchor) != 1 or anchor[0].get("publicScore") != "0.679" or anchor[0].get("status") != "SubmissionStatus.COMPLETE":
        raise RuntimeError("0.679 control not verified")
    private_preview_once()
    check_no_prior()
    response = run(["kaggle", "competitions", "submit", COMPETITION,
                    "-k", DEST_REF, "-v", "1", "-f", "submission.parquet",
                    "-m", DESCRIPTION])
    refs = re.findall(r"Submission ref:\s*(\d+)", response, re.I)
    if len(refs) != 1:
        raise RuntimeError("ambiguous competition result: do not retry")
    print("NTL_WEMM_OFFICIAL_ACCEPTED_REF", refs[0], flush=True)
    for i in range(8):
        rows = [r for r in official_rows() if r.get("ref") == refs[0]]
        if len(rows) == 1 and rows[0].get("description") == DESCRIPTION:
            print("NTL_WEMM_OFFICIAL_LEDGER", json.dumps({
                k: rows[0].get(k) for k in ("ref", "status", "publicScore",
                                            "privateScore", "date", "description")
            }, sort_keys=True), flush=True)
            return
        time.sleep(12)
    raise RuntimeError("accepted ref absent from the authenticated ledger")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--source-preflight", action="store_true")
    parser.add_argument("--private-preview-once", action="store_true")
    parser.add_argument("--submit-once", action="store_true")
    args = parser.parse_args()
    if sum((args.static_only, args.source_preflight,
            args.private_preview_once, args.submit_once)) != 1:
        parser.error("select exactly one explicit operation; no default side effects")
    if args.static_only:
        if not CONTRACT_PATH.is_file():
            raise RuntimeError("contract missing")
        print("NTL_WEMM_V1_STATIC_PRECHECK_PASS", flush=True)
    elif args.source_preflight:
        frozen_reproduction(Path("/tmp/ntl-dual-wemm-source-preflight"))
        print("NTL_WEMM_V1_GUARDS_PASS_NO_GPU_NO_SUBMISSION", flush=True)
    elif args.private_preview_once:
        private_preview_once()
    else:
        competition_submit_once()
