#!/usr/bin/env python3
"""Use the Kaggle Model Proxy quota to test which raw model identities it returns."""
from __future__ import annotations

import argparse
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


def origin(uri: str) -> str:
    parsed = urlsplit(uri)
    allowed = parsed.hostname and parsed.hostname.endswith((".kaggle.net", ".kaggle.com"))
    if parsed.scheme != "https" or not allowed or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("Unrecognized Kaggle Model Proxy origin")
    base = uri.rstrip("/")
    return base.removesuffix("/openapi/v1").removesuffix("/openapi")


def native_request(label: str) -> tuple[str, str, dict]:
    if label.startswith("gpt-"):
        return "openai/" + ROUTES[label], "/openapi/v1/responses", {"input": PROMPT, "max_output_tokens": 640}
    if label.startswith("claude-"):
        return "anthropic/" + label, "/anthropic/v1/messages", {
            "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 640,
        }
    if label.startswith("gemini-"):
        provider = "google"
    elif label.startswith("glm-"):
        provider = "z-ai"
    else:
        provider = {"qwen3.8-max": "qwen", "kimi-k3": "moonshotai"}[label]
    return f"{provider}/{label}", "/openapi/v1/chat/completions", {
        "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 640, "stream": False,
    }


def text_from_response(data: dict) -> str:
    choices = data.get("choices") or []
    if choices:
        content = choices[0].get("message", {}).get("content")
        return content if isinstance(content, str) else ""
    if isinstance(data.get("content"), list):
        return " ".join(str(block.get("text", "")) for block in data["content"] if block.get("type") == "text")
    return " ".join(str(block.get("text", "")) for message in data.get("output", [])
                    if message.get("type") == "message" for block in message.get("content", [])
                    if block.get("type") == "output_text")


def observed(row: dict, label: str, data: dict) -> dict:
    if not isinstance(data, dict):
        return {**row, "status": "non_object_response"}
    served = data.get("model")
    if not isinstance(served, str) or not re.fullmatch(r"[a-zA-Z0-9_.@/-]{1,160}", served):
        served = None
    words = len(text_from_response(data).split())
    canonical = served.rsplit("/", 1)[-1].removesuffix("-default") if served else None
    exact = canonical == label or (label == "gpt-5.5" and canonical == "gpt-5.5-2026-04-23")
    return {**row, "status": "http_200", "raw_model": served, "words": words,
            "raw_model_exact": exact, "exact_and_40_to_250_words": exact and 40 <= words <= 250}


def probe_native(base: str, key: str, label: str) -> dict:
    route, path, body = native_request(label)
    row = {"label": label, "requested": route, "interface": path,
           "listed_in_frozen_catalog": label not in UNCATALOGUED}
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    if path.startswith("/anthropic/"):
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    request = Request(base + path, headers=headers, data=json.dumps({"model": route, **body}).encode())
    try:
        with build_opener(NoRedirect()).open(request, timeout=65) as response:
            return observed(row, label, json.loads(response.read(1_000_000)))
    except HTTPError as error:
        detail = error.read(8192).decode(errors="replace").lower()
        reason = "unclassified"
        if "model" in detail and any(word in detail for word in ("not found", "unsupported", "unknown")):
            reason = "unknown_or_unsupported_model"
        elif "not found" in detail:
            reason = "path_or_resource_not_found"
        elif "max_" in detail:
            reason = "token_limit_parameter"
        return {**row, "status": f"http_{error.code}", "error_category": reason}
    except (URLError, TimeoutError, ValueError, TypeError, AttributeError) as error:
        return {**row, "status": type(error).__name__}


def proxy_catalog(base: str, key: str) -> dict:
    request = Request(base + "/openapi/v1/models", headers={"Authorization": "Bearer " + key})
    try:
        with build_opener(NoRedirect()).open(request, timeout=20) as response:
            data = json.loads(response.read(1_000_000))
        ids = [x["id"] for x in data.get("data", []) if isinstance(x.get("id"), str)]
        exact = sorted({x for x in ids if x.rsplit("/", 1)[-1].removesuffix("-default") in ROUTES
                        and re.fullmatch(r"[a-zA-Z0-9_.@/-]{1,160}", x)})
        return {"status": "http_200", "listed": len(ids), "exact_requested_labels": exact}
    except HTTPError as error:
        return {"status": f"http_{error.code}"}
    except (URLError, TimeoutError, ValueError, TypeError, AttributeError) as error:
        return {"status": type(error).__name__}


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
            return observed(row, label, data)
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-endpoints", action="store_true")
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    balances = api.benchmarks_quota().quota_balances or []
    if not balances or any(x.quota_used >= x.total_quota_allowed for x in balances):
        raise SystemExit("KAGGLE_MODEL_PROXY_NO_REMAINING_QUOTA")
    config = api._fetch_model_proxy_env(source="auth")
    base = origin(config["MODEL_PROXY_URL"])
    key = config["MODEL_PROXY_API_KEY"]
    if not key:
        raise SystemExit("KAGGLE_MODEL_PROXY_NO_AUTHORIZED_KEY")
    catalog = proxy_catalog(base, key) if args.provider_endpoints else None
    with ThreadPoolExecutor(max_workers=4) as workers:
        if args.provider_endpoints:
            rows = list(workers.map(lambda label: probe_native(base, key, label), ROUTES))
        else:
            rows = list(workers.map(lambda label: probe(base + "/openapi/v1/chat/completions", key, label), ROUTES))
    if len(rows) != 18:
        raise RuntimeError("Unexpected probe denominator")
    print("NTL_KAGGLE_PROXY_GENERATION_PROBE=" + json.dumps({
        "records": rows, "status_counts": Counter(row["status"] for row in rows),
        "raw_proxy_catalog": catalog,
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
