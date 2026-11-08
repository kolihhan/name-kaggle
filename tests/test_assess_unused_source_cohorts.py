import pandas as pd
from scripts.assess_unused_source_cohorts import source_eligibility

def test_original_training_source_not_counted_as_fresh():
    official=pd.DataFrame({"prompt_id":["official-p"],"model":["a"],"text":["official"]})
    fit=pd.DataFrame({"source":["used"],"prompt_id":["p1"],"prompt_key":["key-a"],
                      "model":["a"],"text":["used row"]})
    new=pd.DataFrame({"source":["used"]*400,"model":["a"]*400,
                      "prompt_id":[f"np{i}" for i in range(400)],
                      "text":[f"new {i}" for i in range(400)]})
    result=source_eligibility(new,fit,official)
    assert result["count_capacity_eligible"]==0
    assert result["already_trained_source_count"]==1
    assert result["selected_by_frozen_source_hash"] is None
    assert result["independent_scientific_transfer_verified"] is False

def test_overlap_purged_and_label_coverage_fails_closed():
    labels=[f"model-{i}" for i in range(18)]
    official=pd.DataFrame({"prompt_id":["official-p"]*18,
                           "model":labels,"text":[f"official {i}" for i in range(18)]})
    fit=pd.DataFrame({"source":["used"],"prompt_id":["train-p"],
                      "prompt_key":["key-a"],"model":[labels[0]],
                      "text":["training"]})
    records=[]
    for i in range(18):
        for j in range(24):
            records.append({"source":"new-independent-name-only","model":labels[i],
                            "prompt_id":f"p{i}-{j}","prompt_key":f"n{i}-{j}",
                            "text":f"fresh {i} {j}",
                            "source_license":"CC BY-SA 4.0",
                            "flag_near_train_prompt":False,
                            "flag_train_prompt_regen":False,
                            "flag_kb_proxy":False,
                            "flag_gemini37_routing_risk":False})
    records[0]["text"]="training"
    records[1]["prompt_id"]="official-p"
    rows=pd.DataFrame(records)
    result=source_eligibility(rows,fit,official)
    assert result["count_capacity_eligible"]==1
    assert result["source_candidates"][0]["eligible_no_overlaps"]==430
    assert result["source_candidates"][0]["exact_train_or_official_text_overlap_count"]==1
    assert result["source_candidates"][0]["prompt_overlap_count"]==1
    assert result["source_integrity_verdict"]=="UNPROVEN"
