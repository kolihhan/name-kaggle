"""Latest NTL source intake, read-only Kaggle public catalog + file metadata.

This is a DATA DISCOVERY gate, NOT a benchmark, and never reads raw response text.
Current 18-class availability is not equivalent to independent validation.
No model training, paid endpoint, GPU or competition submission.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from datetime import datetime, timezone

SEARCHES = ("name-that-llm", "name that llm", "ntl")
PREVIOUSLY_AUDITED = {
    "liuyihuileo/ntl-external-training-candidates-18",
    "keisukefujiwara/nlm-extra-gen",
    "crtproduction/name-that-llm-open-pool",
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
    "nishantdahal/ntl-handwriting-blend",
    "lxdlxdlxd/ntl-synth-generations-v1",
    "willrice/name-that-llm-openrouter-harvest",
    "olegzholobov/name-that-llm-response-corpus",
    "seyitkaangunes/ntl-extra-api-passages",
    "toanplam/ntl-fresh-generation-pilot-nov08",
    "nimithashreea/ntl-proxy-candidate-generations-20261110",
    "maniceet/name-that-llm-openrouter-openai-generations",
}
# Name-based qualification is NOT proof of available 18 exact served models.
CUTOFF = "2026-11-18"
MAX_FOLLOWUP = 12


def run_cli(args, timeout=70):
    try:
        p = subprocess.run(["kaggle", *args], capture_output=True, text=True,
                           timeout=timeout, check=False)
        if p.returncode != 0:
            return None, ("CLI_EXIT_" + str(p.returncode))
        return p.stdout, None
    except subprocess.TimeoutExpired:
        return None, "CLI_TIMEOUT"


def csv_rows(output, required):
    if output is None:
        return None
    lines = output.replace("\ufeff", "").splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.lower().startswith(required.lower() + ",")), None)
    if start is None:
        return None
    try:
        return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    except csv.Error:
        return None


def detect_iso_date(value):
    raw = str(value or "").strip()
    m = re.search(r"20\d\d-\d\d-\d\d", raw)
    return m.group(0) if m else None


def live_catalog():
    found = {}
    failures = []
    for term in SEARCHES:
        for page in (1, 2):
            raw, err = run_cli(["datasets", "list", "--search", term,
                                "--sort-by", "updated", "--page-size", "100",
                                "--page", str(page), "-v"], timeout=75)
            data = csv_rows(raw, "ref")
            if err or data is None:
                failures.append({"query": term, "page": page,
                                 "error": err or "NO_PARSEABLE_CSV"})
                continue
            for entry in data:
                ref = str(entry.get("ref", "")).strip()
                if not ref or "/" not in ref:
                    continue
                updated = str(entry.get("lastUpdated") or
                              entry.get("last_updated") or
                              entry.get("lastUpdatedDate") or
                              entry.get("lastUpdatedTime") or "")
                record = {
                    "ref": ref,
                    "updated_text": updated[:60],
                    "updated_date": detect_iso_date(updated),
                    "license_in_search": str(entry.get("licenseName") or
                                             entry.get("license") or "")[:100],
                    "title": str(entry.get("title") or "")[:130],
                    "description": str(entry.get("description") or "")[:130],
                    "owner_ref_previously_audited": ref in PREVIOUSLY_AUDITED,
                }
                old = found.get(ref)
                if old is None or record["updated_text"] > old["updated_text"]:
                    found[ref] = record
    return found, failures


def intake():
    catalog, failures = live_catalog()
    names = list(catalog.values())
    candidate = [e for e in names if not e["owner_ref_previously_audited"]
                 and (e["updated_date"] is None or e["updated_date"] >= CUTOFF)]
    # Unknown dates are retained, not silently treated as "new".
    candidate.sort(key=lambda x: (x["updated_date"] or "",
                                   x["updated_text"], x["ref"]), reverse=True)
    selected = candidate[:MAX_FOLLOWUP]
    examined = []
    for entry in selected:
        raw, err = run_cli(["datasets", "files", entry["ref"],
                            "--page-size", "100", "-v"], timeout=55)
        files = csv_rows(raw, "name")
        file_names = [] if files is None else [
            str(x.get("name") or x.get("Name") or "")[:140]
            for x in files if (x.get("name") or x.get("Name"))
        ]
        examined.append({
            **entry,
            "file_inventory_status": err or
            ("NO_PARSEABLE_CSV" if files is None else "OK"),
            "file_count_first_page": len(file_names),
            "metadata_file_names": file_names[:35],
            "maybe_has_model_count_evidence": any(
                re.search(r"(counts.*model|model.*counts|audit.*json|readme)",
                          x, re.IGNORECASE) for x in file_names),
            "actual_18_class_support_verified": False,
            "served_model_identity_verified": False,
            "all_prompt_source_independence_verified": False,
            "data_admitted_for_benchmark": False,
        })
    return {
        "scope": "authenticated_public_kaggle_catalog_readonly",
        "utc_audit": datetime.now(timezone.utc).isoformat(),
        "search_queries": list(SEARCHES),
        "pages_per_query": 2,
        "records_distinct": len(catalog),
        "catalog_query_failures": failures,
        "prior_known_sources": sorted(PREVIOUSLY_AUDITED),
        "new_or_undated_ref_candidates": len(candidate),
        "followup_cap": MAX_FOLLOWUP,
        "metadata_followup": examined,
        "recent_catalog_examples": candidate[:30],
        "admission_rule": "none admitted from file listing; next requires true 18-label served-ID protocol evidence and contamination screening",
        "new_paid_generation": False,
        "new_foundation_model_inference": False,
        "new_training": False,
        "new_GPU": False,
        "new_competition_submission": False,
        "raw_response_or_official_competition_text_exported": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        assert detect_iso_date("2026-11-21T11:00:00Z") == "2026-11-21"
        assert csv_rows("hello\nref,title\nx/y,sample\n", "ref")[0]["ref"] == "x/y"
        assert csv_rows("header\n","ref") is None
        print("NTL_SOURCE_INTAKE_CPU_SMOKE=PASS", flush=True)
    else:
        print("NTL_FULL18_SOURCE_INTAKE=" +
              json.dumps(intake(), sort_keys=True), flush=True)
