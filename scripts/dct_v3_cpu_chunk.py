from __future__ import annotations

from typing import Any, Mapping

CANDIDATE = "dct-temporal-fingerprint-v3"
SCOPE = "exact_cpu_chunk_execution_no_scoring"
SCIENTIFIC_BASE_SHA = "2d0a86fffa880709ad25c8ee9153f2b2838617a8"
ACCUMULATION_SIZE = 6


def chunk_contract() -> dict[str, Any]:
    return {
        "candidate": CANDIDATE,
        "scope": SCOPE,
        "scientific_base_sha": SCIENTIFIC_BASE_SHA,
        "accumulation_size": ACCUMULATION_SIZE,
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "firewall_rows_scored": 0,
        "public_lb_tuning": False,
        "submission_authorized": False,
    }


def next_chunk_boundary(*, cursor: int, total_microbatches: int, microbatch_budget: int) -> int:
    cursor = int(cursor)
    total_microbatches = int(total_microbatches)
    microbatch_budget = int(microbatch_budget)
    if total_microbatches < 0:
        raise ValueError("total_microbatches must be non-negative")
    if not 0 <= cursor <= total_microbatches:
        raise ValueError("cursor must satisfy 0 <= cursor <= total_microbatches")
    if microbatch_budget <= 0:
        raise ValueError("microbatch_budget must be positive")
    return min(total_microbatches, cursor + microbatch_budget)


def _expected_accumulation_position(*, cursor: int, total_microbatches: int) -> int:
    if cursor == total_microbatches:
        return 0
    return cursor % ACCUMULATION_SIZE


def validate_chunk_manifest(
    payload: Mapping[str, Any],
    *,
    expected_predecessor_sha256: str | None = None,
) -> dict[str, Any]:
    contract = chunk_contract()
    for key, expected in contract.items():
        observed = payload.get(key)
        if observed != expected:
            raise RuntimeError(f"chunk manifest drift for {key}: {observed!r} != {expected!r}")

    total = int(payload.get("total_microbatches", -1))
    cursor = int(payload.get("cursor", -1))
    generation = int(payload.get("generation", -1))
    accumulation_position = int(payload.get("accumulation_position", -1))
    if total <= 0:
        raise RuntimeError("total_microbatches must be positive")
    if not 0 <= cursor <= total:
        raise RuntimeError("chunk cursor is outside total_microbatches")
    if generation < 0:
        raise RuntimeError("chunk generation must be non-negative")

    expected_position = _expected_accumulation_position(
        cursor=cursor, total_microbatches=total
    )
    if accumulation_position != expected_position:
        raise RuntimeError(
            "chunk accumulation position drift: "
            f"{accumulation_position} != {expected_position}"
        )

    predecessor = payload.get("predecessor_state_sha256")
    if expected_predecessor_sha256 is not None and predecessor != expected_predecessor_sha256:
        raise RuntimeError(
            "chunk predecessor state drift: "
            f"{predecessor!r} != {expected_predecessor_sha256!r}"
        )

    return {
        "candidate": CANDIDATE,
        "cursor": cursor,
        "accumulation_position": accumulation_position,
        "accumulation_size": ACCUMULATION_SIZE,
        "total_microbatches": total,
        "generation": generation,
        "complete": cursor == total,
        "predecessor_state_sha256": predecessor,
    }
