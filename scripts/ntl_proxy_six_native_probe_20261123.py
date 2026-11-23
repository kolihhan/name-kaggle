"""Bounded *real* endpoint identity probe for six models absent from Kaggle catalog.

No training/evaluation data, no writing responses, no competition submission.
Read only returned model IDs, output-length and status. Short provider calls spend
only available Kaggle Model Proxy allowance, with a conservative per-call cap.
"""
from __future__ import annotations

import json
import subprocess
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from scripts.ntl_probe_kaggle_served_models import (
    NoRedirect, native_request, observed, origin,
)

MISSING = (
    "claude-fable-5", "gemini-3.7-flash", "glm-5.3",
    "glm-5.3-flash", "kimi-k3", "qwen3.8-max",
)
OUTPUT_TOKEN_CAP = 256
MIN_QUOTA_BEFORE_NEXT_CALL = 2.00
MAX_OBSERVED_CREDIT_USE = 0.75


def quota_remaining() -> dict[str, float]:
    run = subprocess.run(
        ["kaggle", "benchmarks", "quota", "--format", "json"],
        capture_output=True, text=True, timeout=40, check=False,
    )
    if run.returncode:
        raise RuntimeError("quota API failed closed")
    payload = json.loads(run.stdout)
    rows = payload if isinstance(payload, list) else next(
        (v for v in payload.values() if isinstance(v, list)), []
    ) if isinstance(payload, dict) else []
    result = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        period = str(item.get("period", "")).strip().lower()
        remaining = item.get("remaining", "")
        try:
            amount = float(str(remaining).replace("$", "").replace(",", "").strip())
        except ValueError:
            continue
        if period in ("daily", "monthly"):
            result[period] = amount
    if set(result) != {"daily", "monthly"}:
        raise RuntimeError("missing daily/monthly quota: fail closed")
    return result


def one(base: str, key: str, label: str) -> dict:
    route, path, body = native_request(label)
    for token_arg in ("max_tokens", "max_output_tokens", "max_completion_tokens"):
        if token_arg in body:
            body[token_arg] = min(int(body[token_arg]), OUTPUT_TOKEN_CAP)
    row = {"label": label, "route": route, "interface": path}
    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    }
    if path.startswith("/anthropic/"):
        headers = {
            "x-api-key": key, "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    req = Request(
        base + path,
        data=json.dumps({"model": route, **body}).encode("utf-8"),
        headers=headers,
    )
    try:
        with build_opener(NoRedirect()).open(req, timeout=45) as resp:
            raw = resp.read(1000000)
        data = json.loads(raw)
        result = observed(row, label, data)
        # observed deliberately returns only status, served model and word count.
        # Never include generated text, request data, or provider payload.
        return result
    except HTTPError as exc:
        return {**row, "status": f"http_{exc.code}", "raw_model_exact": False}
    except (URLError, TimeoutError, ValueError, TypeError, AttributeError) as exc:
        return {**row, "status": type(exc).__name__, "raw_model_exact": False}


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    assert len(MISSING) == len(set(MISSING)) == 6
    initial = quota_remaining()
    if min(initial.values()) < MIN_QUOTA_BEFORE_NEXT_CALL:
        raise SystemExit("NTL_SIX_NATIVE_BLOCKED_LOW_QUOTA")
    kaggle = KaggleApi()
    kaggle.authenticate()
    config = kaggle._fetch_model_proxy_env(source="auth")
    base = origin(config["MODEL_PROXY_URL"])
    key = config["MODEL_PROXY_API_KEY"]
    if not key:
        raise SystemExit("NTL_SIX_NATIVE_BLOCKED_NO_PROXY_AUTH")
    rows = []
    for label in MISSING:
        current = quota_remaining()
        consumed = max(0.0, initial["daily"] - current["daily"])
        if min(current.values()) < MIN_QUOTA_BEFORE_NEXT_CALL or consumed >= MAX_OBSERVED_CREDIT_USE:
            rows.append({"label": label, "status": "skipped_credit_guard"})
            continue
        row = one(base, key, label)
        rows.append(row)
        print("NTL_SIX_NATIVE_ONE=" + json.dumps(row, sort_keys=True), flush=True)
    final = quota_remaining()
    print("NTL_SIX_NATIVE_RESULT=" + json.dumps({
        "attempted": sum(r.get("status") != "skipped_credit_guard" for r in rows),
        "raw_model_exact_and_40_to_250_words":
            sum(bool(r.get("exact_and_40_to_250_words")) for r in rows),
        "status_by_label": {r["label"]: r.get("status") for r in rows},
        "matched_model_ids": {r["label"]: r.get("raw_model")
                             for r in rows if r.get("raw_model_exact")},
        "initial_quota_remaining_usd": initial,
        "final_quota_remaining_usd": final,
        "observed_daily_credit_spend_usd":
            round(max(0.0, initial["daily"] - final["daily"]), 4),
        "new_training": False,
        "official_competition_submissions": 0,
        "raw_responses_saved": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
