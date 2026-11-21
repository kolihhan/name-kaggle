#!/usr/bin/env python3
"""One-shot Kaggle reproduction of a public Notebook; never exports Kaggle credentials."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SOURCE = "estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix"
SLUG = "ntl-estopadilla-glm53-0716-repro-20261122"
COMPETITION = "name-that-llm"
MARKER = "estopadilla-glm53-0716-20261122"
REPO = "kolihhan/name-kaggle"
BRANCH = "refs/heads/ops/submit-estopadilla-glm053-20261122"
WORKDIR = Path("/tmp/ntl_estopadilla_oneshot")
PULL_DIR = WORKDIR / "notebook"


def run(args: list[str], *, timeout: int = 180, quiet: bool = False) -> str:
    command = [str(x) for x in args]
    p = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        # Error messages may include credential-bearing URLs; avoid printing raw stderr.
        safe_error = p.stderr.replace(os.environ.get("KAGGLE_API_TOKEN") or "<no-token>", "[REDACTED]")
        safe_error = re.sub(r"KGAT_[A-Za-z0-9_-]+", "[REDACTED]", safe_error)
        print(f"COMMAND_FAILED: {' '.join(command[:5])}; exit={p.returncode}; error={safe_error[-1600:]}", flush=True)
        raise RuntimeError("Kaggle CLI command failed; see code and endpoint permissions")
    if not quiet:
        print(p.stdout[-2000:], flush=True)
    return p.stdout.strip()


def load_source() -> tuple[Path, dict]:
    PULL_DIR.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "pull", SOURCE, "-p", str(PULL_DIR), "-m"], timeout=300)
    metadata_path = PULL_DIR / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("The public Notebook was not downloaded with metadata")
    meta = json.loads(metadata_path.read_text())
    available = list(PULL_DIR.glob("*.ipynb")) + list(PULL_DIR.glob("*.py"))
    named = PULL_DIR / Path(meta.get("code_file", "")).name
    code = named if named.exists() else (available[0] if len(available) == 1 else None)
    if not code:
        raise RuntimeError("Source code file is ambiguous or missing")
    if not meta.get("kernel_type") in {"notebook", "script"}:
        raise RuntimeError("Unsupported kernel type")
    if code.suffix == ".ipynb":
        notebook = json.loads(code.read_text())
        script = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
    else:
        script = code.read_text()
    # Do not transfer any source that explicitly reads or exports credentials.
    suspicious = ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "UserSecretsClient", "get_secret(",
                  "kaggle.json", ".kaggle/access_token", "os.environ['HF_TOKEN']")
    if any(fragment in script for fragment in suspicious):
        raise RuntimeError("Source requests secrets; fail closed without pushing")
    if COMPETITION not in json.dumps(meta) and COMPETITION not in script:
        print("NOTE: competition not explicitly referenced; it will be attached in clone metadata")
    print("SOURCE_READY", SOURCE)
    print("SOURCE_KIND", code.suffix, "CODE_BYTES", code.stat().st_size)
    print("SOURCES_COUNTS", {k: len(meta.get(k, [])) for k in
                             ("dataset_sources", "competition_sources", "kernel_sources", "model_sources")})
    return code, meta


def make_private_copy(code: Path, meta: dict, username: str) -> str:
    kernel = f"{username}/{SLUG}"
    meta.pop("id_no", None)  # otherwise Kaggle would update the author's notebook
    meta["id"] = kernel
    meta["title"] = "NTL Estopadilla GLM53 User Prefix 0716 Repro 20261122"
    meta["code_file"] = code.name
    meta["is_private"] = "true"
    meta["language"] = "python"
    meta["enable_gpu"] = str(meta.get("enable_gpu", "false")).lower()
    meta["enable_internet"] = str(meta.get("enable_internet", "false")).lower()
    competitions = list(meta.get("competition_sources") or [])
    if COMPETITION not in competitions:
        competitions.append(COMPETITION)
    meta["competition_sources"] = competitions
    (PULL_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("PRIVATE_NOTEBOOK_TARGET", f"https://www.kaggle.com/code/{kernel}")
    print("GPU_PRESERVED", meta["enable_gpu"])
    return kernel


def check_existing() -> bool:
    rows = run(["kaggle", "competitions", "submissions", COMPETITION, "-v"],
               timeout=120, quiet=True)
    if MARKER in rows:
        print("ONE_SHOT_ALREADY_SUBMITTED: exact marker present; no new submission")
        return True
    return False


def poll(kernel: str) -> None:
    deadline = time.monotonic() + 195 * 60
    while time.monotonic() < deadline:
        try:
            status = run(["kaggle", "kernels", "status", kernel], timeout=120, quiet=True)
        except (subprocess.TimeoutExpired, RuntimeError):
            print("KERNEL_STATUS_RETRY", flush=True)
            time.sleep(45)
            continue
        normalized = status.lower()
        print("KERNEL_STATUS", status.replace("\n", " ")[:600], flush=True)
        if re.search(r"\b(complete|completed)\b", normalized):
            return
        if re.search(r"\b(error|failed|failure|cancelled|canceled)\b", normalized):
            raise RuntimeError("Kaggle Notebook failed; no competition submission")
        time.sleep(45)
    raise TimeoutError("Notebook did not finish during the one-shot CI window; no submission")


def output_name(kernel: str) -> str:
    text = run(["kaggle", "kernels", "files", kernel, "-v"], timeout=120, quiet=True)
    filenames = re.findall(r"([A-Za-z0-9_./-]+\.(?:parquet|csv))", text, flags=re.I)
    candidates = [Path(name).name for name in filenames
                  if re.search(r"(^|[_.-])submission([_.-]|$)", Path(name).stem, re.I)
                  and not Path(name).name.lower().startswith("sample")]
    unique = list(dict.fromkeys(candidates))
    print("SUBMISSION_OUTPUT_CANDIDATES", unique)
    if unique != ["submission.parquet"]:
        raise RuntimeError("Cannot unambiguously choose an actual Notebook submission output; no submission")
    return unique[0]


def main() -> None:
    if os.getenv("GITHUB_REPOSITORY") != REPO or os.getenv("GITHUB_REF") != BRANCH:
        raise RuntimeError("Wrong repository/ref; refusing to submit")
    if not os.getenv("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN GitHub Actions Secret is missing")
    import kagglehub
    username = kagglehub.whoami(verbose=False)["username"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise RuntimeError("Unexpected Kaggle username")
    print("KAGGLE_AUTH_PASS", username)
    if check_existing():
        return
    code, meta = load_source()
    kernel = make_private_copy(code, meta, username)
    # A private reproduction of the public code; the original notebook is not modified.
    run(["kaggle", "kernels", "push", "-p", str(PULL_DIR)], timeout=300)
    print("KERNEL_PUSH_ACCEPTED")
    poll(kernel)
    filename = output_name(kernel)
    if check_existing():
        return
    result = run(["kaggle", "competitions", "submit", COMPETITION,
                  "-f", filename, "-k", kernel, "-m",
                  f"Public estopadilla GLM5.3 likelihood user-prefix reproduction {MARKER}"],
                 timeout=300)
    refs = re.findall(r"(?:Submission\s+(?:ref|id)\s*:?\s*)(\d+)", result, re.I)
    if refs:
        print("OFFICIAL_SUBMISSION_ID", refs[-1])
        try:
            print("OFFICIAL_SUBMISSION_STATUS",
                  run(["kaggle", "competitions", "submission", refs[-1]], timeout=90))
        except Exception:
            print("OFFICIAL_SUBMISSION_STATUS_PENDING")
    else:
        print("OFFICIAL_SUBMISSION_RESPONSE", result[:2000])
    print("ONE_SHOT_SUBMISSION_ATTEMPT_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Avoid printing raw HTTP errors, which could include security context.
        print("FAIL_CLOSED", type(exc).__name__, str(exc)[:250], flush=True)
        sys.exit(1)
