import pandas as pd

from scripts.audit_domain_separability import (
    HOLDOUT_SALT,
    STYLE_COLUMNS,
    assign_domain_holdout,
    build_label_domain_split,
    extract_style_features,
    summarize_scores,
)


def test_domain_holdout_is_frozen_and_prompt_group_stable():
    assert HOLDOUT_SALT == "ntl-domain-separability-v1"
    rows = []
    for domain in ["official", "external"]:
        for prompt_id in [f"p{i:03d}" for i in range(100)]:
            for replica in range(2):
                rows.append(
                    {
                        "domain": domain,
                        "prompt_id": prompt_id,
                        "model": "m1",
                        "text": f"{domain}-{prompt_id}-{replica}",
                    }
                )
    frame = pd.DataFrame(rows)
    first = assign_domain_holdout(frame, comparison="phase7", model_label="m1")
    second = assign_domain_holdout(frame.sample(frac=1.0, random_state=9), comparison="phase7", model_label="m1")

    first_map = (
        first[["domain", "prompt_id", "is_holdout"]]
        .drop_duplicates()
        .set_index(["domain", "prompt_id"])["is_holdout"]
        .sort_index()
    )
    second_map = (
        second[["domain", "prompt_id", "is_holdout"]]
        .drop_duplicates()
        .set_index(["domain", "prompt_id"])["is_holdout"]
        .sort_index()
    )
    assert first_map.equals(second_map)
    assert first.groupby(["domain", "prompt_id"])["is_holdout"].nunique().eq(1).all()
    assert first.groupby("domain")["is_holdout"].any().all()
    assert first.groupby("domain")["is_holdout"].apply(lambda s: (~s).any()).all()


def test_build_label_domain_split_keeps_prompt_groups_out_of_train_test_overlap():
    official = pd.DataFrame(
        {
            "prompt_id": [f"o{i:03d}" for i in range(100)],
            "model": ["m1"] * 100,
            "text": [f"official response {i}" for i in range(100)],
        }
    )
    external = pd.DataFrame(
        {
            "prompt_id": [f"e{i:03d}" for i in range(120)],
            "model": ["m1"] * 120,
            "text": [f"external response {i}" for i in range(120)],
        }
    )

    train, test = build_label_domain_split(
        official,
        external,
        comparison="phase7",
        model_label="m1",
    )
    assert set(train["domain"]) == {"official", "external"}
    assert set(test["domain"]) == {"official", "external"}
    for domain in ["official", "external"]:
        train_prompts = set(train.loc[train.domain == domain, "prompt_id"])
        test_prompts = set(test.loc[test.domain == domain, "prompt_id"])
        assert train_prompts.isdisjoint(test_prompts)


def test_style_features_capture_basic_format_signals():
    text = "# Heading\n- item 42!\n```py\nx = (1 + 2)\n```\nhttps://example.com"
    features = extract_style_features([text])
    assert features.columns.tolist() == STYLE_COLUMNS
    row = features.iloc[0]
    assert row["char_count"] == len(text)
    assert row["token_count"] > 0
    assert row["line_count"] == 6
    assert row["code_fence_count"] == 2
    assert row["markdown_heading_count"] == 1
    assert row["markdown_list_count"] == 1
    assert row["url_count"] == 1
    assert row["digit_per_1k"] > 0
    assert row["punct_per_1k"] > 0


def test_score_summary_uses_sample_standard_deviation():
    summary = summarize_scores([0.6, 0.7, 0.8])
    assert abs(summary["mean"] - 0.7) < 1e-12
    assert abs(summary["sample_std"] - 0.1) < 1e-12
    assert summary["min"] == 0.6
    assert summary["max"] == 0.8
