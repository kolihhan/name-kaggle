"""One authorized exploratory Kaggle Code Competition submission, no new training.

Scientific change: class-agnostic confidence-adaptive shrinkage of the
existing unlabeled Sinkhorn prior correction in the frozen 0.679 pipeline.

Never consumes public LB scores for tuning, never fits models, and never
publishes Kaggle texts or the copied private notebook to this repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

COMP = "name-that-llm"
ANCHOR_KERNEL = "kolihhan/ntl-ozertuu-0683-guarded-no-train"
DEST_KERNEL = "kolihhan/ntl-margin-gated-transport-v1"
DESCRIPTION = (
    "MGT-v1 unlabeled confidence-adaptive prior transport; "
    "frozen 0.679 prefit baseline; no new training 20261111"
)
ANCHOR_SUBMISSION = "58210835"
ANCHOR_CODE_SHA256 = (
    "28fd128b745f0d95b70f79ea9b2c546d981edf9be08207cbf71a2d9e80ee1c86"
)
CONTRACT = Path("artifacts/candidates/margin-gated-transport-v1/CONTRACT.md")
OLD = "        total = (1 - strength) * np.log(P + 1e-12) + strength * np.log(Q + 1e-12)"
NEW = """        # MGT-v1: soften correction on confident rows without labels/training.
        ordered = np.sort(P, axis=1)
        margin = np.clip(ordered[:, -1] - ordered[:, -2], 0.0, 1.0)
        row_strength = float(strength) * (1.0 - margin)
        base_log = np.log(P + 1e-12)
        total = base_log + row_strength[:, None] * (np.log(Q + 1e-12) - base_log)"""
SOURCE_GUARDS = (
    "gauss.fit = _no_new_supervised_gaussian_fit",
    "NO_NEW_TRAINING: Atlas train mode forbidden",
    "NO_NEW_TRAINING: core worker train_final forbidden",
)


def cli(args: list[str], *, timeout: int = 120, checked: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if checked and p.returncode:
        # Never leak any token or full provider response to Github logs.
        raw = (p.stdout + "\n" + p.stderr)
        codes = {
            "http_400": bool(re.search(r"(?<!\d)400(?!\d)", raw)),
            "http_401": bool(re.search(r"(?<!\d)401(?!\d)", raw)),
            "http_403": bool(re.search(r"(?<!\d)403(?!\d)", raw)),
            "http_404": bool(re.search(r"(?<!\d)404(?!\d)", raw)),
            "quota": "quota" in raw.lower(),
            "timeout": "timeout" in raw.lower(),
        }
        print("NTL_MGT_CLI_ERROR_CLASS", json.dumps({
            "command": args[:3], "rc": p.returncode, **codes
        }, sort_keys=True), flush=True)
        raise RuntimeError(f"Kaggle CLI failed {args[:3]}, rc={p.returncode}")
    return p


def code_text(cell: dict) -> str:
    s = cell.get("source", [])
    return "".join(s) if isinstance(s, list) else str(s)


def all_code(nb: dict) -> list[str]:
    return [code_text(c) for c in nb["cells"] if c.get("cell_type") == "code"]


def code_sha(nb: dict) -> str:
    return hashlib.sha256(json.dumps(all_code(nb), separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def official_rows() -> list[dict]:
    p = cli(["kaggle", "competitions", "submissions", COMP, "-v", "-q"], timeout=90)
    return list(csv.DictReader(p.stdout.lstrip("\ufeff").splitlines()))


def official_gate() -> None:
    rows = official_rows()
    anchor = [row for row in rows if row.get("ref") == ANCHOR_SUBMISSION]
    if len(anchor) != 1 or anchor[0].get("status") != "SubmissionStatus.COMPLETE":
        raise RuntimeError("frozen 0.679 anchor not confirmed complete")
    if anchor[0].get("publicScore") != "0.679":
        raise RuntimeError("anchor official publicScore differs from frozen 0.679")
    if any(r.get("description") == DESCRIPTION for r in rows):
        raise RuntimeError("this MGT version has already been officially submitted")
    print("NTL_MGT_OFFICIAL_ANCHOR", json.dumps({
        "ref": ANCHOR_SUBMISSION, "status": "COMPLETE", "publicScore": "0.679",
        "candidate_description_not_submitted": True,
    }, sort_keys=True), flush=True)


def destination_absent() -> None:
    p = cli(["kaggle", "kernels", "status", DEST_KERNEL], checked=False, timeout=55)
    raw = p.stdout + "\n" + p.stderr
    if re.search(r"KernelWorkerStatus\.[A-Z_]+", raw):
        raise RuntimeError("destination Notebook already exists; refuse resubmit")
    if p.returncode:
        # Strictly redact credentials and only report local CLI error category.
        safe = []
        for line in raw.splitlines()[-8:]:
            if re.search(r"(?i)error|exception|kernel|not|404|status|invalid|permission|access", line):
                line = re.sub(r"(?i)(?:authorization|token|api[_ -]?key|secret|password)\\s*[:=]\\s*\\S+",
                              "[redacted]", line)
                line = re.sub(r"[A-Za-z0-9_=-]{50,}", "[redacted-long-value]", line)
                safe.append(line[:220])
        print("NTL_MGT_DESTINATION_STATUS_DIAGNOSTIC",
              json.dumps({"rc": p.returncode,"safe_lines":safe[-5:]}), flush=True)
    if p.returncode == 0 or "403" in raw or "401" in raw or "429" in raw:
        raise RuntimeError("destination Notebook status ambiguous or unauthorized")
    # Expected normal 404/no kernel found is safe to create new.
    if not re.search(r"not found|404|does not exist|cannot be found", raw, re.I):
        raise RuntimeError("destination kernel status error not recognized as 404")
    print("NTL_MGT_NEW_DESTINATION_CONFIRMED", DEST_KERNEL, flush=True)


def math_synthetic_gate() -> None:
    # No user or Kaggle data. Confirm the fixed rule changes outcomes in
    # nontrivial cases and never enlarges the baseline correction strength.
    import numpy as np

    p = np.full((3, 18), 0.02 / 17.0, dtype=np.float64)
    p[0, :] = 1 / 18.0
    p[1, 0] = 0.98
    p[2, 0] = 0.45
    p[2, 1] = 0.43
    p[2, 2:] = 0.12 / 16.0
    q = np.tile(np.arange(1, 19, dtype=np.float64), (3, 1))
    q = q / q.sum(axis=1, keepdims=True)
    p /= p.sum(axis=1, keepdims=True)
    strength = 0.30
    ordered = np.sort(p, axis=1)
    margin = np.clip(ordered[:, -1] - ordered[:, -2], 0.0, 1.0)
    row_strength = strength * (1 - margin)
    assert 0 <= row_strength.min() <= row_strength.max() <= strength
    assert np.isclose(row_strength[0], strength, atol=1e-12)
    assert row_strength[1] < 0.02
    assert row_strength[2] > 0.28
    old = (1 - strength) * np.log(p + 1e-12) + strength * np.log(q + 1e-12)
    b = np.log(p + 1e-12)
    new = b + row_strength[:, None] * (np.log(q + 1e-12) - b)
    assert np.isfinite(new).all() and not np.allclose(old, new)
    assert np.allclose(old[0], new[0], atol=1e-12)
    print("NTL_MGT_SYNTHETIC_MATH_TEST_PASS", json.dumps({
        "rows": 3, "classes": 18,
        "high_confidence_downweighted": True,
        "ambiguous_prior_strength_preserved": True,
        "ground_truth_used": False,
        "model_fitting": False,
    }, sort_keys=True), flush=True)


def build(where: Path) -> dict:
    if not CONTRACT.is_file():
        raise RuntimeError("predeclared contract missing")
    src = where / "source"
    dst = where / "kernel"
    src.mkdir(parents=True, exist_ok=False)
    dst.mkdir(parents=True, exist_ok=False)
    cli(["kaggle", "kernels", "pull", ANCHOR_KERNEL, "-p", str(src), "-m"], timeout=120)
    files = list(src.glob("*.ipynb"))
    if len(files) != 1:
        raise RuntimeError("owned scored source Notebook not a unique file")
    nb = json.loads(files[0].read_text(encoding="utf-8"))
    meta = json.loads((src / "kernel-metadata.json").read_text(encoding="utf-8"))
    if (meta.get("id") != ANCHOR_KERNEL or
        meta.get("is_private") is not True or
        meta.get("enable_internet") is not False or
        meta.get("kernel_type") != "notebook" or
        meta.get("competition_sources") != [COMP] or
        len(meta.get("dataset_sources") or []) != 20 or
        len(meta.get("kernel_sources") or []) != 8 or
        len(meta.get("model_sources") or []) != 6):
        raise RuntimeError("scored champion metadata changed")
    digest = code_sha(nb)
    if digest != ANCHOR_CODE_SHA256:
        raise RuntimeError("owned scored champion code-cell digest changed")
    codes = all_code(nb)
    if any(sum(x in s for s in codes) != 1 for x in SOURCE_GUARDS):
        raise RuntimeError("all three original no-new-training guards required")
    if len(nb.get("cells", [])) != 53:
        raise RuntimeError("scored champion cell count changed")
    if any(nb["cells"][idx]["cell_type"] != "code" for idx in (5, 7, 9, 44, 51, 52)):
        raise RuntimeError("scored champion code cell layout changed")
    cell = nb["cells"][9]
    before = code_text(cell)
    if before.count(OLD) != 1 or not before.startswith("%%writefile "):
        raise RuntimeError("original transport interpolation target drift")
    after = before.replace(OLD, NEW, 1)
    # This must be a single function's *one* scientific change.
    if after.replace(NEW, OLD) != before:
        raise RuntimeError("patch is not a minimal reversible single-line delta")
    compile("\n".join(after.splitlines()[1:]), "<mgt-core-math>", "exec")
    cell["source"] = after.splitlines(keepends=True)
    changed = [ix for ix, (a, b) in enumerate(zip(codes, all_code(nb))) if a != b]
    if changed != [sum(c["cell_type"] == "code" for c in nb["cells"][:9])]:
        raise RuntimeError("unexpected modification beyond core_math cell9")
    if any(sum(x in s for s in all_code(nb)) != 1 for x in SOURCE_GUARDS):
        raise RuntimeError("no-training guards modified")
    # Add clear upstream attribution; never copy raw Kaggle passages into Git.
    nb["cells"].insert(0, {
        "cell_type": "markdown", "metadata": {},
        "source": [
            "**MGT-v1 exploratory inference, no new training** — original frozen "
            "0.679 Kaggle source is [Ozertuu's public 0.683-title Notebook]"
            "(https://www.kaggle.com/code/ozertuu/0-683-0-682-qwen3-8-27b-gsq-15-mix-0-52), "
            "with original credits and all pretrained assets unchanged. "
            "One new class-agnostic top-2 margin gate suppresses unwarranted "
            "Sinkhorn correction on confident examples, with no learned "
            "parameters, no label-specific overrides, and no LB adaptation. "
            "Official Public score of this variant is unknown until Kaggle scoring.\n",
        ]
    })
    if nb.get("nbformat") != 4 or nb.get("nbformat_minor") != 4:
        raise RuntimeError("source Notebook format has changed from known 4.4")
    for c in nb["cells"]:
        c.pop("id", None)  # Strict valid nbformat 4.4.
        if c.get("cell_type") == "code":
            c["execution_count"] = None
            c["outputs"] = []

    meta.pop("id_no", None)  # Avoid upstream ownership/403 on SaveKernel.
    meta["id"] = DEST_KERNEL
    meta["title"] = DEST_KERNEL.split("/", 1)[1]  # Kaggle slug must match.
    meta["is_private"] = True
    meta["enable_internet"] = False
    meta["machine_shape"] = "NvidiaTeslaT4"
    name = meta.get("code_file")
    if not name or Path(name).name != name or not name.endswith(".ipynb"):
        raise RuntimeError("missing or unsafe notebook code_file")
    filename = dst / name
    filename.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    try:
        import nbformat
        nbformat.validate(nb)
    except ImportError:
        raise RuntimeError("strict nbformat validation dependency missing")
    new_digest = hashlib.sha256(filename.read_bytes()).hexdigest()
    if code_sha(nb) == ANCHOR_CODE_SHA256:
        raise RuntimeError("candidate's executable code still equals champion")
    if any(sum(x in s for s in all_code(nb)) != 1 for x in SOURCE_GUARDS):
        raise RuntimeError("source training guard no longer unique")
    print("NTL_MGT_BUILD_VERIFIED", json.dumps({
        "anchor_code_sha256": ANCHOR_CODE_SHA256,
        "new_notebook_sha256": new_digest,
        "scientific_changes": 1,
        "changed_original_cell_index": 9,
        "no_new_training_guards": 3,
        "18_class_output_unchanged": True,
        "data_source_count": 20, "kernel_source_count": 8, "model_source_count": 6,
        "private": True, "internet": False,
        "file_bytes": filename.stat().st_size,
    }, sort_keys=True), flush=True)
    return {"folder": dst, "notebook_sha256": new_digest}


def status() -> str:
    p = cli(["kaggle", "kernels", "status", DEST_KERNEL], timeout=60, checked=False)
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", p.stdout + "\n" + p.stderr)
    if p.returncode and not m:
        raise RuntimeError("new Notebook no longer reachable after push")
    return m.group(1) if m else "UNKNOWN"


def wait_preview() -> None:
    for i in range(1, 115):
        v = status()
        print("NTL_MGT_PREVIEW_STATUS", i, v, flush=True)
        if v in ("ERROR", "FAILED", "CANCELLED"):
            raise RuntimeError("private preview failed, do not train/re-push")
        if v == "COMPLETE":
            break
        time.sleep(30)
    else:
        raise RuntimeError("private Kaggle preview not COMPLETE; no competition submission")
    files = cli(["kaggle", "kernels", "files", DEST_KERNEL, "-v", "--page-size", "200"])
    if "submission.parquet" not in files.stdout:
        raise RuntimeError("expected competition submission.parquet missing")
    if "Next Page Token" in files.stdout:
        raise RuntimeError("Kaggle file listing was incomplete/paginated")
    print("NTL_MGT_PRIVATE_PREVIEW_COMPLETE", json.dumps({
        "kernel": DEST_KERNEL, "submission_parquet": True,
        "warning": "3-row preview verifies packaging, not full hidden inference"
    }), flush=True)


def submit() -> None:
    official_gate()
    destination_absent()
    math_synthetic_gate()
    build_result = build(Path("/tmp/ntl-margin-gated-transport-v1"))
    official_gate()
    pushed = cli(["kaggle", "kernels", "push", "-p", str(build_result["folder"]),
                  "--accelerator", "NvidiaTeslaT4", "--timeout", "32400"], timeout=160)
    success = re.findall(r"Kernel version\s+1\s+successfully pushed", pushed.stdout + pushed.stderr, re.I)
    if len(success) != 1:
        raise RuntimeError("one-shot notebook version 1 push not confirmed")
    print("NTL_MGT_PRIVATE_KERNEL_VERSION1_ACCEPTED", DEST_KERNEL, flush=True)
    wait_preview()
    official_gate()
    p = cli(["kaggle", "competitions", "submit", COMP,
             "-k", DEST_KERNEL, "-v", "1", "-f", "submission.parquet",
             "-m", DESCRIPTION], timeout=150)
    raw = p.stdout + "\n" + p.stderr
    match = re.findall(r"Submission ref:\s*(\d+)", raw, re.I)
    if len(match) != 1:
        raise RuntimeError("official submit response lacked one clear submission ref")
    ref = match[0]
    print("NTL_MGT_NEW_OFFICIAL_SUBMISSION_ID", ref, flush=True)
    for i in range(1, 26):
        entries = [r for r in official_rows() if r.get("ref") == ref]
        if len(entries) == 1 and entries[0].get("description") == DESCRIPTION:
            row = entries[0]
            print("NTL_MGT_OFFICIAL_LEDGER", json.dumps({
                "ref": ref, "status": row.get("status"),
                "publicScore": row.get("publicScore"),
                "privateScore": row.get("privateScore"),
                "date": row.get("date"), "description": row.get("description"),
            }, sort_keys=True), flush=True)
            if row.get("status") in ("SubmissionStatus.COMPLETE",
                                     "SubmissionStatus.ERROR", "SubmissionStatus.FAILED"):
                return
        time.sleep(24)
    print("NTL_MGT_ACCEPTED_SCORE_NOT_YET_TERMINAL", ref, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    if args.static_only:
        if not CONTRACT.is_file():
            raise RuntimeError("MGT preregistered contract missing")
        math_synthetic_gate()
        print("NTL_MGT_STATIC_NO_GPU_COMPLETE", flush=True)
    elif args.preflight:
        official_gate()
        destination_absent()
        math_synthetic_gate()
        build(Path("/tmp/ntl-margin-gated-transport-preflight"))
        print("NTL_MGT_SOURCE_PREFLIGHT_NO_GPU_COMPLETE", flush=True)
    else:
        submit()


if __name__ == "__main__":
    main()
