from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest

source = Path(__file__).resolve().parents[1] / "scripts" / "audit_official_sibling_signals.py"
spec = importlib.util.spec_from_file_location("audit_official_sibling_signals", source)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def fixture(n=3):
    labels = sorted({label for a, b in audit_module.PAIRS for label in (a, b)})
    return [
        {"prompt_id": f"p{i}", "model": label,
         "text": f"Here is a long interesting explanation, with commas. {i}"}
        for i in range(n) for label in labels
    ]


def test_feature_set_is_frozen_and_only_aggregate():
    f = audit_module.featurize("A longer thoughtful phrase, however—here's 10!")
    assert len(f) == 19
    assert f["words"] == 6
    assert f["commas_per100"] > 0
    assert f["however"] == 1
    assert f["em_dash"] == 1


def test_exact_two_sided_sign_p():
    assert audit_module.paired_sign_p(0, 0) == 1
    assert audit_module.paired_sign_p(5, 0) == 2 / 32
    assert audit_module.paired_sign_p(5, 5) == 1


def test_hundred_fourteen_tests_and_zero_when_identical():
    out = audit_module.audit(fixture())
    assert out["feature_count"] == 19
    assert out["tests_total"] == 6 * 19
    assert out["bh_significant_q_below_005"] == 0
    assert out["significant_aggregate_only"] == []
    assert out["status"].startswith("DESCRIPTIVE_")


def test_missing_pair_rejected():
    rows = [r for r in fixture() if r["model"] != "glm-5.3-flash"]
    with pytest.raises(ValueError, match="Missing matched"):
        audit_module.audit(rows)


def test_duplicate_cell_rejected():
    rows = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        audit_module.audit(rows + [dict(rows[0])])


def test_bh_aggregate_adjustment():
    arr = [{"p": 0.01}, {"p": 0.02}, {"p": 0.06}]
    audit_module.bh_correct(arr)
    assert [round(x["q"], 3) for x in arr] == [0.03, 0.03, 0.06]
