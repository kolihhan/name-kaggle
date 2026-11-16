"""Frozen 18-class external-data selection for authentic Lightning0691 transfer.

No fitting, no model labels in the inference CSV, no public export of source
passages/IDs. Dataset provenance remains an uncertainty even after strict gates.
This self-contained module is inlined into the private Kaggle pilot input cell.
"""
import hashlib
import pandas as pd

NTL_OLEG18_SALT = "ntl-pro-oleg18-select-12-per-class-20261116-4ed7c9"
NTL_OLEG18_MODELS = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8",
    "claude-opus-5", "claude-sonnet-5", "gemini-3.1-pro-preview",
    "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "kimi-k3",
    "qwen3.8-max",
)
NTL_OLEG18_STRONG_EVIDENCE = (
    "request_and_returned_model", "raw_api_response_model",
    "row_request_response_model_id", "platform_response_model_label",
    "row_request_returned_model_source_reported",
)
NTL_OLEG18_SHA = {
    "external_short_wide.parquet":
        "15a67d022315b2508af8edd749a7c041d42cc37e266106028c73e405659c400f",
    "external_cell_metadata.parquet":
        "397af75e5f2809107d0758a147646a68a0132479a6ea77432d86243e92ad9079",
}


def ntl_oleg18_select(wide, meta, per_class=12):
    """Deterministic balanced sample; return truth privately, never to the agent.

    Control unseen-group leakage: one response per distinct prompt group for
    ALL labels, not just within each class. All rows must have strong observed
    model IDs and public source license metadata.
    """
    if per_class != 12:
        raise ValueError("frozen_12_per_class_no_tuning")
    if list(wide.columns) != ["id", "prompt", *NTL_OLEG18_MODELS]:
        raise ValueError("original_external_wide_schema_mismatch")
    if wide.id.isna().any() or wide.id.duplicated().any():
        raise ValueError("external_id_missing_or_duplicate")
    needed = ("view", "id", "model", "prompt_group_id",
        "short_answer_candidate", "mechanically_eligible",
        "text_only_suitable", "text_shared_across_labels",
        "license_unknown", "model_evidence_class")
    if any(k not in meta.columns for k in needed):
        raise ValueError("provenance_required_fields_missing")
    matched = meta.loc[meta["id"].isin(set(wide["id"]))].copy()
    if set(matched["view"].unique()) != {"short"}:
        raise ValueError("not_unique_exact_external_short_view")
    if matched.duplicated(["id", "model"]).any():
        raise ValueError("duplicate_weak_external_response")
    mask = (
        matched["short_answer_candidate"].eq(True)
        & matched["mechanically_eligible"].eq(True)
        & matched["text_only_suitable"].eq(True)
        & ~matched["text_shared_across_labels"].eq(True)
        & ~matched["license_unknown"].eq(True)
        & matched["model_evidence_class"].isin(NTL_OLEG18_STRONG_EVIDENCE)
    )
    allowed = matched.loc[mask, ["id", "model", "prompt_group_id"]]
    long = wide.melt(id_vars=["id"], value_vars=list(NTL_OLEG18_MODELS),
                     var_name="model", value_name="text").dropna(subset=["text"])
    qualified = long.merge(allowed, on=["id", "model"],
                           how="inner", validate="one_to_one")
    qualified = qualified.loc[qualified["text"].map(
        lambda s: isinstance(s, str) and 40 <= len(s.split()) <= 250)].copy()
    qualified["prompt_group_id"] = qualified["prompt_group_id"].astype(str)
    qualified = qualified.loc[
        qualified["prompt_group_id"].str.len().gt(0)
        & qualified["prompt_group_id"].ne("nan")]
    if any(qualified.loc[qualified["model"].eq(label)].shape[0] < per_class
           for label in NTL_OLEG18_MODELS):
        raise ValueError("insufficient_all_18_evidence_supported_labels")
    qualified["sort_hash"] = [
        hashlib.sha256((NTL_OLEG18_SALT + "|" + str(uid) + "|" + str(lab))
                       .encode()).hexdigest()
        for uid, lab in zip(qualified["id"], qualified["model"])
    ]
    result = []
    used_groups = set()
    used_text = set()
    # Fixed lexicographic class order, not tuned by any model results.
    for label in NTL_OLEG18_MODELS:
        candidates = qualified.loc[qualified["model"].eq(label)]
        candidates = candidates.sort_values("sort_hash", kind="stable")
        n = 0
        for item in candidates.itertuples(index=False):
            content_hash = hashlib.sha256(item.text.strip().encode()).hexdigest()
            if item.prompt_group_id in used_groups or content_hash in used_text:
                continue
            used_groups.add(item.prompt_group_id)
            used_text.add(content_hash)
            result.append({
                "id": "e18_" + item.sort_hash[:20],
                "text": item.text,
                "model": label,
                "prompt_group_id": item.prompt_group_id,
            })
            n += 1
            if n == per_class:
                break
        if n != per_class:
            raise ValueError("not_enough_source_disjoint_" + label)
    out = pd.DataFrame(result)
    if (len(out) != per_class * 18 or out.id.duplicated().any()
        or out.prompt_group_id.duplicated().any()):
        raise ValueError("selected_18class_group_balance_contract_drift")
    if not set(out.model.unique()) == set(NTL_OLEG18_MODELS):
        raise ValueError("selected_label_set_incomplete")
    # Interleave via a second frozen stable hash to remove grouped label order
    # from serialized input; IDs contain only hashes, not semantic labels.
    out["order_key"] = out.id.map(lambda x: hashlib.sha256(
        ("ntl-oleg18-shuffle-20261116|" + str(x)).encode()).hexdigest())
    return out.sort_values("order_key", kind="stable").drop(
        columns="order_key").reset_index(drop=True)
