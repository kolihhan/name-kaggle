"""Frozen 0.715 NTL observability: sidecar-only GLM boundary trace.

This does not fit any model or change any scoring file. Intended for two extra
cells in a PRIVATE derivative of the SHA-pinned 0.715 scorer, never for a public
competition-data artifact. The sole persistent trace is aggregate JSON.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUBMISSION = Path("/kaggle/working/submission.parquet")
PRE_LOGITS = Path("/kaggle/temp/final-logits.npz")
PRE_SNAPSHOT = Path("/kaggle/temp/ntl_stage_trace_pre.parquet")
REPORT = Path("/kaggle/working/ntl_stage_trace_aggregates.json")
LABELS = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8",
    "claude-opus-5", "claude-sonnet-5", "gemini-3.1-pro-preview",
    "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "kimi-k3",
    "qwen3.8-max",
)
FAMILY = {
    "claude-": "Anthropic", "gemini-": "Google", "glm-": "Z.ai",
    "gpt-": "OpenAI", "kimi-": "Moonshot", "qwen": "Alibaba"
}


def _family(v: str) -> str:
    for pref, fam in FAMILY.items():
        if v.startswith(pref):
            return fam
    raise ValueError("unknown model label")


def _data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if set(df.columns) != {"id", "model"} or df.id.isna().any():
        raise ValueError("invalid output schema")
    ids = df.id.astype(str)
    if ids.duplicated().any() or ids.empty:
        raise ValueError("duplicate or empty IDs")
    if not df.model.astype(str).isin(LABELS).all():
        raise ValueError("unexpected model label")
    return df.assign(id=ids, model=df.model.astype(str))


def _logits_check(frame: pd.DataFrame, logits: Path) -> dict[str, Any]:
    if not logits.is_file():
        raise FileNotFoundError("frozen strong pre-GLM logits missing")
    with np.load(logits, allow_pickle=False) as z:
        if set(("ids", "labels", "logits")) - set(z.files):
            raise ValueError("required original strong logits not present")
        ids = np.asarray(z["ids"], dtype=str)
        labels = np.asarray(z["labels"], dtype=str)
        scores = np.asarray(z["logits"])
        if ids.shape != (len(frame),) or scores.shape != (len(frame), len(LABELS)):
            raise ValueError("logits row count or class dimension drift")
        if not np.array_equal(ids, frame.id.to_numpy()):
            raise ValueError("strong logits and submission IDs misaligned")
        if labels.shape != (18,) or set(labels.tolist()) != set(LABELS):
            raise ValueError("pre-GLM score classes do not match official labels")
        if np.isnan(scores).any():
            raise ValueError("NaN in strong scores")
        return {"strong_logits_rows_aligned": len(frame),
                "strong_logits_label_count": len(labels),
                "strong_logits_shape_valid": True,
                "strong_logits_are_pre_GLM_not_post_V1": True}


def snapshot_pre_glm(
    submission: Path = SUBMISSION,
    original_logits: Path = PRE_LOGITS,
    snapshot: Path = PRE_SNAPSHOT,
) -> dict[str, Any]:
    """Read-only snapshot immediately BEFORE V1 edits OUT (no truth, no inference)."""
    if snapshot.exists():
        raise FileExistsError("snapshot must be written exactly once")
    source_sha_before = hashlib.sha256(submission.read_bytes()).hexdigest()
    df = _data(submission)
    shape = _logits_check(df, original_logits)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(snapshot, index=False)
    reread = _data(snapshot)
    if not df.equals(reread):
        snapshot.unlink(missing_ok=True)
        raise AssertionError("pre-GLM labels changed in snapshot")
    if hashlib.sha256(submission.read_bytes()).hexdigest() != source_sha_before:
        snapshot.unlink(missing_ok=True)
        raise AssertionError("snapshot modified scorer's submission bytes")
    return {"pre_snapshot_rows": len(df), **shape,
            "baseline_submission_bytes_unchanged": True,
            "private_pre_predictions_preserved_in_temp_only": True}


def audit_post_glm(
    submission: Path = SUBMISSION,
    snapshot: Path = PRE_SNAPSHOT,
    original_logits: Path = PRE_LOGITS,
    v1_report: Path | None = None,
    aggregate_out: Path = REPORT,
) -> dict[str, Any]:
    """After ALL original cells: classify label transitions without any ground truth."""
    if not snapshot.exists():
        raise FileNotFoundError("no pre-V1 sidecar: trace not comparable")
    before_sha = hashlib.sha256(submission.read_bytes()).hexdigest()
    pre, post = _data(snapshot), _data(submission)
    if len(pre) != len(post) or not np.array_equal(pre.id, post.id):
        raise ValueError("pre/post stage IDs or ordering drift")
    shape = _logits_check(pre, original_logits)
    a, b = pre.model.to_numpy(dtype=str), post.model.to_numpy(dtype=str)
    fa, fb = (np.array([_family(t) for t in a]),
              np.array([_family(t) for t in b]))
    changed = a != b
    cross = (fa != fb) & changed
    sibling = (fa == fb) & changed
    assert int(cross.sum() + sibling.sum()) == int(changed.sum())
    # All counts are UNLABELLED. Never call changes rescues or regressions.
    family_pairs = {}
    for x, y in zip(fa[cross], fb[cross]):
        k = f"{x} -> {y}"
        family_pairs[k] = family_pairs.get(k, 0) + 1
    v1_exists = bool(v1_report and v1_report.is_file())
    # No report content is trusted or copied: may contain row-specific raw logs.
    result = {
        "status": "UNLABELLED_STAGE_CHANGE_TRACE_ONLY",
        "n_rows": len(pre),
        "changed_final_labels": int(changed.sum()),
        "changed_across_families": int(cross.sum()),
        "changed_within_family_siblings": int(sibling.sum()),
        "unchanged": int((~changed).sum()),
        "changed_to_family": {f: int((changed & (fb == f)).sum()) for f in sorted(set(fb))},
        "changed_from_family": {f: int((changed & (fa == f)).sum()) for f in sorted(set(fa))},
        "cross_family_transition_counts": dict(sorted(family_pairs.items())),
        "original_glm_stage_report_present": v1_exists,
        **shape,
        "out_parquet_sha_unchanged_by_audit":
            hashlib.sha256(submission.read_bytes()).hexdigest() == before_sha,
        "no_true_labels_accessed": True,
        "no_accuracy_or_BA_claim": True,
        "no_new_model_or_inference_or_training": True,
        "not_a_full18_independent_validation": True,
    }
    if not result["out_parquet_sha_unchanged_by_audit"]:
        raise AssertionError("audit changed scorer output")
    if aggregate_out.exists():
        raise FileExistsError("aggregate trace must not overwrite a prior run")
    aggregate_out.parent.mkdir(parents=True, exist_ok=True)
    aggregate_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def self_test() -> None:
    """Run with synthetic labels / fake logits, no Kaggle data."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        out, pre, logits, report, aggregate = (
            root / "submission.parquet", root / "pre.parquet",
            root / "logits.npz", root / "v1_report.json", root / "trace.json")
        ids = np.array(["mock-A", "mock-B", "mock-C"], dtype=str)
        before = [LABELS[9], LABELS[11], LABELS[1]]
        pd.DataFrame({"id": ids, "model": before}).to_parquet(out, index=False)
        np.savez(logits, ids=ids, labels=np.array(LABELS),
                 logits=np.zeros((3, 18), dtype=float))
        state = snapshot_pre_glm(out, logits, pre)
        assert state["pre_snapshot_rows"] == 3
        # Mock V1 would change one GLM sibling and one model family.
        pd.DataFrame({"id": ids, "model": [
            LABELS[10], LABELS[11], LABELS[11]]}).to_parquet(out, index=False)
        report.write_text('{"mock_only":true}')
        byte_before = out.read_bytes()
        res = audit_post_glm(out, pre, logits, report, aggregate)
        assert res["changed_final_labels"] == 2
        assert res["changed_within_family_siblings"] == 1
        assert res["changed_across_families"] == 1
        assert res["original_glm_stage_report_present"]
        assert out.read_bytes() == byte_before
        assert json.loads(aggregate.read_text())["no_accuracy_or_BA_claim"]
        try:
            snapshot_pre_glm(out, logits, pre)
        except FileExistsError:
            pass
        else:
            raise AssertionError("must refuse duplicate snapshot")
        # Re-run on a different output WITHOUT a valid saved logits row order.
        wrong = root / "wrong.npz"
        np.savez(wrong, ids=ids[::-1], labels=np.array(LABELS),
                 logits=np.zeros((3, 18)))
        try:
            _logits_check(_data(out), wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("must catch misaligned logits")
    print("NTL_0715_STAGE_TRACE_SYNTHETIC=PASS", flush=True)


if __name__ == "__main__":
    self_test()
