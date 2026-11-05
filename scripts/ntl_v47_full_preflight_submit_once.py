"""One-shot Name That LLM Lightning V47 no-training full-sample preflight, then official submit.

This script is executed only by an explicitly authorized, one-shot GitHub Actions
workflow. It never uploads official data to GitHub and never trains any model.
Fail closed if the exact public source, Kaggle credentials, or execution differs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

SOURCE = "lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble"
VERSION = "47"
UPSTREAM_SHA256 = "be996d4921795156b2bc27e78171c70de02c9a43fdcfcaf7bc949c72e7dce32d"
PREFLIGHT = "kolihhan/ntl-private-permission-canary-20261105"
DESCRIPTION = "NTL Lightning V47 exact public source after full-sample inference gate 20261105"
BASELINE_REF = 57985641
LABELS = {
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6",
    "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
}
ROOT = pathlib.Path("/tmp/ntl-v47-once")


def log(*args):
    print(*args, flush=True)


def cli(*argv, check=True):
    cmd = ["kaggle", *argv]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
    if p.returncode != 0 and check:
        msg = (p.stderr + "\n" + p.stdout)[-1800:]
        raise RuntimeError("Kaggle CLI failed " + " ".join(argv[:3]) + ": " + msg)
    return p


def submissions():
    p = cli("competitions", "submissions", "name-that-llm", "-v", "-q")
    rows = list(csv.DictReader(p.stdout.splitlines()))
    if not rows:
        raise RuntimeError("Kaggle submissions returned no records; fail closed")
    return rows


def ensure_no_duplicate_or_newer(rows):
    rows_by_id = {int(r["ref"]): r for r in rows}
    if BASELINE_REF not in rows_by_id:
        raise RuntimeError("Known last submission not present; cannot establish history")
    if any(DESCRIPTION in (r.get("description") or "") for r in rows):
        raise RuntimeError("Official attempt with this exact description already exists: do not retry")
    newer = [r for r in rows if int(r["ref"]) > BASELINE_REF]
    if newer:
        log("NEWER_SUBMISSIONS_FOUND", json.dumps([
            {k: r.get(k) for k in ("ref", "status", "publicScore", "date")} for r in newer
        ]))
        raise RuntimeError("Newer Kaggle submission exists; stop rather than duplicate")
    log("PRE_SUBMISSION_CHECK_OK", "rows", len(rows), "latest", max(rows_by_id))


def prepare_private_full_sample():
    folder = ROOT / "source"
    private = ROOT / "private"
    folder.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    cli("kernels", "pull", f"{SOURCE}/{VERSION}", "-p", str(folder), "-m")
    notebooks = list(folder.glob("*.ipynb"))
    if len(notebooks) != 1:
        raise RuntimeError(f"Expected one pinned notebook; got {len(notebooks)}")
    raw = notebooks[0].read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    log("PINNED_SOURCE_SHA256", observed)
    if observed != UPSTREAM_SHA256:
        raise RuntimeError(f"V47 upstream SHA drift: {observed} != {UPSTREAM_SHA256}")
    md = json.loads((folder / "kernel-metadata.json").read_text())
    if (md.get("id") != SOURCE or md.get("enable_internet") is not False
            or md.get("enable_gpu") is not True
            or md.get("competition_sources") != ["name-that-llm"]):
        raise RuntimeError("Source metadata fails exact no-internet competition contract")

    nb = json.loads(raw)
    needle = re.compile(
        r"(?m)^\s*preview\s*=\s*emit_known_preview\(DATA\s*,\s*None\s*,\s*OUTPUT\)\s*$"
    )
    found = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        val = cell["source"]
        original = "".join(val) if isinstance(val, list) else val
        patched, count = needle.subn(
            "preview = False  # Kaggle official-sample full inference gate ONLY",
            original,
        )
        found += count
        if count:
            cell["source"] = patched.splitlines(keepends=True)
    if found != 1:
        raise RuntimeError(f"Expected exactly one known-preview patch, got {found}")
    proof_code = """
import json, hashlib, pathlib, pandas as pd
p = pathlib.Path('/kaggle/working/submission.parquet')
possible = [
    pathlib.Path('/kaggle/input/name-that-llm/test.csv'),
    pathlib.Path('/kaggle/input/competitions/name-that-llm/test.csv'),
]
test_path = next((x for x in possible if x.is_file()), None)
assert test_path is not None, 'Official competition test.csv is not mounted'
t = pd.read_csv(test_path, dtype={'id':str})
s = pd.read_parquet(p)
train_path = test_path.with_name('train.csv')
assert train_path.is_file(), 'Official competition training labels are not mounted'
allowed = set(pd.read_csv(train_path, usecols=['model'])['model'].dropna().astype(str))
assert len(allowed) == 18, 'Expected 18 exact official labels'
assert preview is False, 'known-fixture shortcut was used'
assert p.is_file(), 'missing submission.parquet'
assert list(s.columns) == ['id','model'], list(s.columns)
assert len(s) == len(t), (len(s),len(t))
assert s['id'].notna().all() and s['model'].notna().all()
assert s['id'].is_unique and t['id'].is_unique
assert s['id'].astype(str).tolist() == t['id'].astype(str).tolist(), 'ID mismatch'
assert set(s['model']).issubset(allowed), 'out-of-vocabulary predicted model'
assert not p.with_suffix('.mode.json').exists(), 'fixture preview artifact exists'
digest = hashlib.sha256(
    ('\\n'.join(t['id'].astype(str))).encode()
).hexdigest()
proof = {
    'source':'lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble',
    'upstream_version':47, 'preflight_mode':'actual_inference_preview_disabled',
    'model_weights_inference_executed':True, 'training_performed':False,
    'test_rows':len(t), 'submission_rows':len(s),
    'columns':list(s.columns), 'input_id_sha256':digest,
    'output_id_sha256':hashlib.sha256(
        ('\\n'.join(s['id'].astype(str))).encode()
    ).hexdigest(), 'missing_ids':0,'duplicate_ids':0,
}
pathlib.Path('/kaggle/working/ntl_v47_full_sample_proof.json').write_text(
    json.dumps(proof,sort_keys=True)
)
print('OFFICIAL_SAMPLE_FULL_INFERENCE_PASS',json.dumps(proof,sort_keys=True),flush=True)
"""
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": proof_code.splitlines(keepends=True),
    })
    (private / notebooks[0].name).write_text(json.dumps(nb, ensure_ascii=False))
    # Do not retain the original author's numeric Kaggle notebook identity.
    md.pop("id_no", None)
    md["id"] = PREFLIGHT
    md["title"] = "NTL Private Permission Canary 20261105"
    md["is_private"] = True
    md["enable_internet"] = False
    md["enable_gpu"] = True
    md["machine_shape"] = "NvidiaTeslaT4"
    (private / "kernel-metadata.json").write_text(json.dumps(md, indent=2))
    log("PREPARED_PRIVATE_FULL_INFERENCE", PREFLIGHT)
    return private


def run_or_reuse_preflight(folder):
    # This dedicated canary was created and advanced from V1 to V5 to
    # validate SaveKernel permission for the exact full V47 source families.
    # Upgrade it exactly once from a known existing private-kernel identity.
    existing = cli("kernels", "status", PREFLIGHT, check=False)
    if existing.returncode != 0:
        raise RuntimeError("Known private canary missing; refuse to create a fresh ambiguous kernel")
    log("KNOWN_PRIVATE_CANARY_STATUS", (existing.stdout + existing.stderr)[-350:])
    p = cli(
        "kernels", "push", "-p", str(folder),
        "--accelerator", "NvidiaTeslaT4", "--timeout", "32400",
        check=False,
    )
    log("PRIVATE_PREFLIGHT_PUSH_OUTPUT", (p.stdout + p.stderr)[-1500:])
    if p.returncode != 0:
        raise RuntimeError("FULL_SAMPLE_PREFLIGHT_UPDATE_FAILED: " + (p.stderr + p.stdout)[-1200:])
    # Kaggle may queue GPU work; a successful push is NOT a successful inference.
    deadline = time.monotonic() + 200 * 60
    prev = None
    while time.monotonic() < deadline:
        s = cli("kernels", "status", PREFLIGHT, check=False)
        status = (s.stdout + "\n" + s.stderr).strip()
        if status != prev:
            log("KAGGLE_PRIVATE_PREFLIGHT_STATUS", status[-450:])
            prev = status
        if re.search(r"KernelWorkerStatus\.COMPLETE\b", status):
            return
        if re.search(r"KernelWorkerStatus\.(ERROR|FAILED|CANCELLED)\b", status):
            diagnostics = cli("kernels", "logs", PREFLIGHT, check=False)
            raise RuntimeError(
                "FULL_SAMPLE_KAGGLE_PRIVATE_KERNEL_FAILED: "
                + (diagnostics.stdout + diagnostics.stderr)[-3000:]
            )
        time.sleep(30)
    raise RuntimeError("FULL_SAMPLE_PREFLIGHT_UNFINISHED: no submit; latest=" + (prev or "NONE"))


def verify_output():
    out = ROOT / "output"
    out.mkdir()
    cli("kernels", "output", PREFLIGHT, "-p", str(out), "-o")
    proof_files = list(out.rglob("ntl_v47_full_sample_proof.json"))
    subfiles = list(out.rglob("submission.parquet"))
    if len(proof_files) != 1 or len(subfiles) != 1:
        raise RuntimeError(f"Missing proof or parquet: proof={len(proof_files)} parquet={len(subfiles)}")
    proof = json.loads(proof_files[0].read_text())
    import pandas as pd
    sub = pd.read_parquet(subfiles[0])
    if proof.get("upstream_version") != 47 or proof.get("preflight_mode") != "actual_inference_preview_disabled":
        raise RuntimeError("Wrong preflight version or fake preview mode")
    if not proof.get("model_weights_inference_executed") or proof.get("training_performed"):
        raise RuntimeError("No independent full inference proof")
    if proof.get("test_rows") != len(sub) or proof.get("submission_rows") != len(sub):
        raise RuntimeError("row mismatch")
    if sub.columns.tolist() != ["id", "model"]:
        raise RuntimeError("Wrong columns")
    if not sub.id.is_unique or sub.id.isna().any() or sub.model.isna().any():
        raise RuntimeError("Invalid ids or labels")
    if not set(sub.model).issubset(LABELS):
        raise RuntimeError("Unknown predicted labels")
    if proof["input_id_sha256"] != proof["output_id_sha256"]:
        raise RuntimeError("IDs do not match")
    log("EXACT_V47_KAGGLE_FULL_SAMPLE_VERIFIED", json.dumps(proof, sort_keys=True))
    # Do not upload generated parquet or official data to GitHub.


def submit_exact_once():
    ensure_no_duplicate_or_newer(submissions())
    p = cli(
        "competitions", "submit", "name-that-llm",
        "-k", SOURCE, "-v", VERSION,
        "-f", "submission.parquet",
        "-m", DESCRIPTION,
        check=False,
    )
    msg = (p.stdout + "\n" + p.stderr).strip()
    log("OFFICIAL_COMPETITION_SUBMIT_RESPONSE", msg[-1700:])
    if p.returncode != 0:
        raise RuntimeError("OFFICIAL_COMPETITION_SUBMIT_REJECTED: " + msg[-1500:])
    hit = re.search(r"Submission ref:\s*(\d+)", msg)
    if not hit:
        # The HTTP success is not enough: verify by read-only actual records.
        matching = [r for r in submissions() if r.get("description") == DESCRIPTION]
        if len(matching) != 1:
            raise RuntimeError("No unique official Kaggle competition submission ID returned")
        ref = matching[0]["ref"]
    else:
        ref = hit.group(1)
    log("OFFICIAL_COMPETITION_SUBMISSION_ID", ref)
    return ref


def follow_real_score(ref):
    deadline = time.monotonic() + 55 * 60
    last = None
    while time.monotonic() < deadline:
        matches = [r for r in submissions() if r.get("ref") == str(ref)]
        if len(matches) != 1:
            raise RuntimeError(f"Competition submission {ref} missing from My Submissions")
        r = matches[0]
        tup = (r.get("status"), r.get("publicScore"), r.get("privateScore"))
        if tup != last:
            log("LIVE_OFFICIAL_SUBMISSION_STATE", json.dumps({
                k: r.get(k) for k in ("ref", "status", "publicScore", "privateScore", "description", "date")
            }, sort_keys=True))
            last = tup
        score = (r.get("publicScore") or "").strip()
        if score:
            delta = float(score) - 0.520
            log("ACTUAL_PUBLIC_SCORE", score, "DELTA_VS_0520", f"{delta:+.3f}", "REF", ref)
            return
        if "ERROR" in (r.get("status") or "").upper():
            raise RuntimeError("Kaggle platform reports ERROR; publicScore blank; root cause unknown")
        time.sleep(45)
    raise RuntimeError(
        "OFFICIAL_SUBMISSION_SCORE_NOT_RETURNED: "
        + str(last) + "; detailed private rerun traceback unavailable"
    )


def main():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN not configured")
    if ROOT.exists():
        raise RuntimeError("Refusing to reuse working directory")
    ROOT.mkdir(parents=True)
    ensure_no_duplicate_or_newer(submissions())
    private = prepare_private_full_sample()
    run_or_reuse_preflight(private)
    verify_output()
    ref = submit_exact_once()
    follow_real_score(ref)


if __name__ == "__main__":
    main()
