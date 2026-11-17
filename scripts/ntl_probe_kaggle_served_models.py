#!/usr/bin/env python3
"""Use the Kaggle Model Proxy quota to test which raw model identities it returns."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROUTES = {
    "gpt-5.5": "gpt-5.5-2026-04-23", "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-5.6-sol": "gpt-5.6-sol", "gpt-5.6-terra": "gpt-5.6-terra", "gpt-6-astra": "gpt-6-astra",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview", "gemini-3.6-flash": "gemini-3.6-flash",
    "gemini-3.7-flash": "gemini-3.7-flash", "gemini-3.8-flash": "gemini-3.8-flash",
    "claude-fable-5": "claude-fable-5", "claude-opus-4-6": "claude-opus-4-6-default",
    "claude-opus-4-8": "claude-opus-4-8-default", "claude-opus-5": "claude-opus-5-default",
    "claude-sonnet-5": "claude-sonnet-5-default", "qwen3.8-max": "qwen3.8-max",
    "glm-5.3": "glm-5.3", "glm-5.3-flash": "glm-5.3-flash", "kimi-k3": "kimi-k3",
}
UNCATALOGUED = {"claude-fable-5", "gemini-3.7-flash", "glm-5.3", "glm-5.3-flash", "kimi-k3", "qwen3.8-max"}
PROMPT = (
    "Write one 65 to 85 word paragraph about a paper seed envelope left in an empty "
    "community seed library just before closing. Include a practical detail. Use no title."
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def endpoint(uri: str) -> str:
    parsed = urlsplit(uri)
    allowed = parsed.hostname and parsed.hostname.endswith((".kaggle.net", ".kaggle.com"))
    if parsed.scheme != "https" or not allowed or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("Unrecognized Kaggle Model Proxy origin")
    base = uri.rstrip("/")
    if base.endswith("/openapi/v1"):
        return base + "/chat/completions"
    if base.endswith("/openapi"):
        return base + "/v1/chat/completions"
    return base + "/openapi/v1/chat/completions"


def probe(url: str, key: str, label: str) -> dict:
    route = ROUTES[label]
    row = {"label": label, "requested": route, "listed_in_frozen_catalog": label not in UNCATALOGUED}
    request_data = {"model": route, "messages": [{"role": "user", "content": PROMPT}], "stream": False}
    for cap in ("max_tokens", "max_completion_tokens"):
        request = Request(url, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                          data=json.dumps({**request_data, cap: 240}).encode())
        try:
            with build_opener(NoRedirect()).open(request, timeout=55) as response:
                data = json.loads(response.read(1_000_000))
            if not isinstance(data, dict):
                return {**row, "status": "non_object_response"}
            served = data.get("model")
            if not isinstance(served, str) or not re.fullmatch(r"[a-zA-Z0-9_.@/-]{1,160}", served):
                served = None
            choices = data.get("choices") or []
            content = choices[0].get("message", {}).get("content") if choices else None
            words = len(content.split()) if isinstance(content, str) else 0
            canonical = served.rsplit("/", 1)[-1].removesuffix("-default") if served else None
            exact = canonical == label or (label == "gpt-5.5" and canonical == "gpt-5.5-2026-04-23")
            return {**row, "status": "http_200", "raw_model": served, "words": words,
                    "raw_model_exact": exact, "exact_and_40_to_250_words": exact and 40 <= words <= 250}
        except HTTPError as error:
            payload = error.read(8192).decode(errors="replace")
            if cap == "max_tokens" and error.code in (400, 422) and "max_completion_tokens" in payload:
                continue
            return {**row, "status": f"http_{error.code}"}
        except (URLError, TimeoutError, ValueError, TypeError, AttributeError) as error:
            return {**row, "status": type(error).__name__}
    return {**row, "status": "completion_cap_rejected"}


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    balances = api.benchmarks_quota().quota_balances or []
    if not balances or any(x.quota_used >= x.total_quota_allowed for x in balances):
        raise SystemExit("KAGGLE_MODEL_PROXY_NO_REMAINING_QUOTA")
    config = api._fetch_model_proxy_env(source="auth")
    url = endpoint(config["MODEL_PROXY_URL"])
    key = config["MODEL_PROXY_API_KEY"]
    if not key:
        raise SystemExit("KAGGLE_MODEL_PROXY_NO_AUTHORIZED_KEY")
    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = list(workers.map(lambda label: probe(url, key, label), ROUTES))
    if len(rows) != 18:
        raise RuntimeError("Unexpected probe denominator")
    print("NTL_KAGGLE_PROXY_GENERATION_PROBE=" + json.dumps({
        "records": rows, "status_counts": Counter(row["status"] for row in rows),
        "exact_raw_model_and_text": sum(row.get("exact_and_40_to_250_words", False) for row in rows),
        "uncatalogued_exact_and_text": sum(row.get("exact_and_40_to_250_words", False) for row in rows
                                          if row["label"] in UNCATALOGUED),
        "new_training": False, "official_submissions": 0,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit("NTL_KAGGLE_PROXY_BLOCKED_" + type(error).__name__) from None
