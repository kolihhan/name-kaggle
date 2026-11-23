"""NTL Public-LB selection-bias audit, no training, no scoring, no submissions.

Outputs only public leaderboard/owned-submission aggregate metadata + transparent
hypothetical paired-discordance sensitivity. Neither the latter nor Public score
alone can prove overfitting or improvement. See docs/research/ntl-0715-vs0719-overfit-audit-20261123.md.
"""
from __future__ import annotations
import csv
import io
import json
import math
import re
import subprocess
from collections import Counter

COMP="name-that-llm"
OWNED="kolihhan"
INTEREST=("sanpier","dennis6886","estopadilla","areebbb","hcluo8888","ranjeet258","kolihhan")


def emit(name,payload):
    print("NTL_OVERFIT_"+name+"="+json.dumps(payload,sort_keys=True,ensure_ascii=False,default=str),flush=True)


def cli(*args,timeout=70):
    try:
        return subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=timeout)
    except Exception as exc:
        emit("CLIENT_EXCEPTION",{"cmd":args[:3],"exception":type(exc).__name__})
        return None


def sensitivity():
    """Pure math: balanced n~1600 and symmetric independent rescue/regression signs.

    This is a toy CONDITIONAL scenario, NOT a test based on actual Kaggle labels.
    Matched-pair variance requires true changed-row count and per-label counts.
    """
    n_test=4000
    fraction=.40
    n_public_approx=round(n_test*fraction)
    delta=.719-.715
    net=delta*n_public_approx
    examples=[]
    for m in (8,10,20,40,80,160):
        # Rounded margin >= 6; six is slightly less than expected 6.4.
        q=sum(math.comb(m,k)/(2**m)
              for k in range(m+1) if abs(2*k-m)>=6)
        examples.append({"changed_rows_HYPOTHETICAL":m,
                         "two_sided_p_given_6_net_under_symmetric_null":round(q,5)})
    emit("MATH_ASSUMPTIONS",{
        "metric":"18-way Balanced Accuracy",
        "public_fraction_source":"Kaggle competition leaderboard states approximately 40%",
        "hidden_test_n_source":"Kaggle data page says about 4000",
        "public_n_approx":n_public_approx,
        "difference_BA":round(delta,6),
        "equal_class_size_approx":round(n_public_approx/18,1),
        "net_correct_equivalent_only_if_18_CLASSES_BALANCED":round(net,2),
        "not_exact_number_of_fixes":True,
        "actual_disagreement_count_unknown":True,
        "class_proportions_unknown":True,
        "prompt_correlation_not_accounted_for":True})
    emit("PAIRED_SENSITIVITY_HYPOTHETICAL",examples)


def leaderboard():
    # API surface and permissions may vary. Do not silently call missing rows 0.
    cmds=[
        ["competitions","leaderboard","-c",COMP,"--show"],
        ["competitions","leaderboard","-c",COMP,"-s"],
    ]
    worked=False
    for args in cmds:
        r=cli(*args)
        if r is None or r.returncode:
            emit("LEADERBOARD_ATTEMPT",{"command":args[-1],
                "ok":False,"error_type":"client_nonzero_or_unavailable"})
            continue
        lines=r.stdout.splitlines()
        matches=[s.strip()[:220] for s in lines
                 if any(n in s.lower() for n in INTEREST)]
        emit("PUBLIC_LEADERBOARD_METADATA",{
            "ok":True,"total_lines":len(lines),
            "matching_PUBLIC_ROWS":matches[:25],
            "match_count":len(matches),
            "does_not_show_version_score_of_every_notebook":True})
        worked=True
        break
    if not worked:
        emit("PUBLIC_LEADERBOARD_METADATA",{"ok":False,"reason":"not_returned_do_not_infer_no_entries"})


def owned_submissions():
    p=cli("competitions","submissions","-c",COMP,"-v",timeout=75)
    if p is None or p.returncode:
        emit("OWNED_SUBMISSION_AGGREGATES",{"available":False,"reason":"client_nonzero_or_unavailable"})
        return
    rows=list(csv.DictReader(io.StringIO(p.stdout.lstrip("\ufeff"))))
    if not rows:
        emit("OWNED_SUBMISSION_AGGREGATES",{"available":False,"reason":"csv_not_parseable"})
        return
    fields=list(rows[0])
    # Do not print individual submission IDs/notes/URLs or hidden labels.
    score_cols=[k for k in fields if k.lower().replace("_","") in
                ("publicscore","score")]
    scored=[]
    for row in rows:
        for k in score_cols:
            try:
                val=float(row[k])
            except (ValueError,TypeError):
                continue
            if math.isfinite(val) and 0<=val<=1:
                scored.append(val)
                break
    status_key=next((k for k in fields if "status" in k.lower()),None)
    statuses=Counter(str(x.get(status_key,"")) for x in rows) if status_key else {}
    emit("OWNED_SUBMISSION_AGGREGATES",{
        "available":True,"total_entries_returned":len(rows),
        "score_field_present":bool(score_cols),"valid_score_count":len(scored),
        "max_visible_score":max(scored) if scored else None,
        "status_counts":dict(statuses),
        "public_scores_not_independent_validation":True,
        "candidate_effect_cannot_be_estimated_from_owned_ledger_alone":True})


def main():
    sensitivity()
    leaderboard()
    owned_submissions()
    emit("END",{
        "read_only":True,
        "no_model_weights_or_competition_training_data_downloaded":True,
        "no_submission_or_training":True,
        "overfit_proven":False,
        "next_true_test":"same-row independent label/prompt/source-scoped C0/C1 paired evaluation"
    })


if __name__=="__main__":
    main()
