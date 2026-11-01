import pandas as pd

from scripts.make_shadow_folds import (
    SHADOW_SALTS,
    assign_prompt_folds,
    build_shadow_folds,
)


def test_shadow_salts_are_frozen():
    assert SHADOW_SALTS == {
        "shadow_a": "ntl-shadow-v1-a",
        "shadow_b": "ntl-shadow-v1-b",
        "shadow_c": "ntl-shadow-v1-c",
    }


def test_assign_prompt_folds_is_deterministic_and_balanced():
    prompts = [f"p{i:03d}" for i in range(150)]
    first = assign_prompt_folds(prompts, "ntl-shadow-v1-a")
    second = assign_prompt_folds(list(reversed(prompts)), "ntl-shadow-v1-a")
    assert first == second
    assert set(first) == set(prompts)
    counts = pd.Series(first).value_counts().sort_index().to_dict()
    assert counts == {0: 30, 1: 30, 2: 30, 3: 30, 4: 30}


def test_shadow_salts_produce_distinct_assignments():
    prompts = [f"p{i:03d}" for i in range(150)]
    mappings = [assign_prompt_folds(prompts, salt) for salt in SHADOW_SALTS.values()]
    assert mappings[0] != mappings[1]
    assert mappings[0] != mappings[2]
    assert mappings[1] != mappings[2]


def test_build_shadow_folds_keeps_prompt_groups_together():
    rows = []
    row_id = 0
    for prompt in [f"p{i:03d}" for i in range(150)]:
        for model in ["m1", "m2"]:
            rows.append({"id": row_id, "prompt_id": prompt, "model": model, "text": f"{prompt}-{model}"})
            row_id += 1
    train = pd.DataFrame(rows)
    shadow = build_shadow_folds(train)
    assert shadow.columns.tolist() == ["id", "prompt_id", "shadow_a", "shadow_b", "shadow_c"]
    assert len(shadow) == len(train)
    assert shadow["id"].is_unique
    for column in SHADOW_SALTS:
        assert set(shadow[column]) == {0, 1, 2, 3, 4}
        assert shadow.groupby("prompt_id")[column].nunique().eq(1).all()
        prompt_counts = shadow[["prompt_id", column]].drop_duplicates()[column].value_counts().sort_index()
        assert prompt_counts.to_dict() == {0: 30, 1: 30, 2: 30, 3: 30, 4: 30}
