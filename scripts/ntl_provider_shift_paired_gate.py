#!/usr/bin/env python3
"""Read-only paired 18-class source-shift gate; never reads Kaggle test labels."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

def evaluate(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8", newline="")))
    required = {"prompt_id", "label", "source", "baseline_pred", "candidate_pred"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Nonempty CSV with columns " + ",".join(sorted(required)) + " required")
    labels = sorted({r["label"] for r in rows})
    if len(labels) != 18:
        raise ValueError(f"Require complete 18-class authentic panel; got {len(labels)}")
    if any(not r["prompt_id"] or not r["source"] or any(r[k] not in labels for k in ("baseline_pred","candidate_pred")) for r in rows):
        raise ValueError("Missing prompt/source or prediction outside 18 labels")
    if len({(r["source"],r["prompt_id"],r["label"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate source/prompt/label row: inspect source identity")
    sources = sorted({r["source"] for r in rows})
    if len(sources) < 2:
        raise ValueError("Need at least two documented, genuinely independent sources")
    def metrics(rs):
        counts = Counter(r["label"] for r in rs)
        if set(counts) != set(labels):
            return {"n":len(rs),"complete_18":False}
        recalls = {}
        for col in ("baseline_pred","candidate_pred"):
            recalls[col] = {lab:sum(r[col]==lab for r in rs if r["label"]==lab)/counts[lab] for lab in labels}
        rescue = sum(r["baseline_pred"]!=r["label"] and r["candidate_pred"]==r["label"] for r in rs)
        regress = sum(r["baseline_pred"]==r["label"] and r["candidate_pred"]!=r["label"] for r in rs)
        a=sum(recalls["baseline_pred"].values())/18
        b=sum(recalls["candidate_pred"].values())/18
        return {"n":len(rs),"complete_18":True,"baseline_BA":a,"candidate_BA":b,"delta_BA":b-a,
                "rescues":rescue,"regressions":regress,"worst_class_baseline":min(recalls["baseline_pred"].values()),
                "worst_class_candidate":min(recalls["candidate_pred"].values()),
                "per_class":{lab:{"baseline":recalls["baseline_pred"][lab],"candidate":recalls["candidate_pred"][lab]} for lab in labels}}
    report={"warning":"Development evidence only; not independent OOD unless provenance proves independence; not official Public BA.",
            "total":metrics(rows),"sources":{}}
    for source in sources:
        subset=[r for r in rows if r["source"]==source]
        report["sources"][source]=metrics(subset)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--panel",required=True,help="Private real labeled same-row panel; never commit it")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    result=evaluate(args.panel)
    Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"total":{k:v for k,v in result["total"].items() if k!="per_class"},
                      "sources":{s:{k:v for k,v in m.items() if k!="per_class"} for s,m in result["sources"].items()}},indent=2))
if __name__=="__main__":
    main()
