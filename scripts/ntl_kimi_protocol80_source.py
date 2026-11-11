"""Frozen strict Kimi K3 high/off 80-pair trial source, private research only."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib,json
from statistics import mean,median

SOURCE_SHA="a38745dd0b648773204193d2a34a9b1ccd5ce9a41cfd01b51eea49ceed56d87e"
SOURCE_REF="willrice/name-that-llm-openrouter-harvest"
FILE="setting_trials.jsonl"
TARGET="kimi-k3"
SETTINGS=("high","off")
EXPECTED_PAIRS=80
EXPECTED_ROWS=160

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def read_pairs(path:Path)->tuple[list[dict],dict]:
    raw=path.read_bytes()
    if sha(raw)!=SOURCE_SHA:raise RuntimeError("frozen Kimi trial file SHA drift")
    dataset=[json.loads(t) for t in raw.decode("utf-8").splitlines() if t.strip()]
    if len(dataset)!=1210:raise RuntimeError("Kimi trial row count changed")
    grouped=defaultdict(lambda:defaultdict(list))
    for item in dataset:
        if item.get("model")==TARGET and item.get("setting") in SETTINGS:
            grouped[str(item.get("prompt_id"))][item["setting"]].append(item)
    if len(grouped)<EXPECTED_PAIRS:
        raise RuntimeError("fewer than 80 model-prompt groups in frozen trial bank")
    records=[]
    for pid,settings in sorted(grouped.items()):
        if not all(len(settings.get(x,[]))==1 for x in SETTINGS):
            continue
        h,o=(settings[x][0] for x in SETTINGS)
        if any(h.get(k) in (None,"") or h.get(k)!=o.get(k) for k in
            ("served_model","served_provider","model_id","target_words")):
            raise RuntimeError("high/off served identity/target mismatch "+pid)
        if any(x.get("finish_reason")!="stop" for x in (h,o)):
            raise RuntimeError("high/off incomplete response "+pid)
        for setting,item in zip(SETTINGS,(h,o)):
            txt=item.get("text")
            if not isinstance(txt,str) or not txt.strip():
                raise RuntimeError("high/off missing response text "+pid)
            n=item.get("n_words")
            if not isinstance(n,int) or not 40<=n<=250:
                raise RuntimeError("high/off response outside word range "+pid)
            records.append({
                "id":f"k{len(records)//2:04d}{'h' if setting=='high' else 'o'}",
                "prompt_id":pid,"setting":setting,"model":TARGET,
                "text":txt.strip(),"n_words":n,
                "target_words":item["target_words"],
                "served_model":item["served_model"],
                "served_provider":item["served_provider"],
                "model_id":item["model_id"],
                "reasoning_tokens":item.get("reasoning_tokens")
            })
    # The trial bank also has single-setting Kimi prompts. Require exactly
    # 80 *complete* high/off groups, not 80 total Kimi prompt keys.
    if len(records)!=EXPECTED_ROWS:
        raise RuntimeError("expected exact 80 complete pairs and 160 rows")
    hashes=[sha(" ".join(x["text"].split()).casefold().encode()) for x in records]
    if len(set(hashes))!=EXPECTED_ROWS:
        raise RuntimeError("repeated Kimi text among selected pairs")
    for i in range(EXPECTED_PAIRS):
        h,o=records[2*i:2*i+2]
        if not (h["prompt_id"]==o["prompt_id"] and
                h["target_words"]==o["target_words"] and
                h["served_model"]==o["served_model"] and
                h["served_provider"]==o["served_provider"]):
            raise RuntimeError("Kimi pair mismatch")
    metrics={}
    for setting in SETTINGS:
        entries=[x for x in records if x["setting"]==setting]
        lengths=[x["n_words"] for x in entries]
        reasons=[x["reasoning_tokens"] for x in entries
                 if isinstance(x["reasoning_tokens"],int)]
        metrics[setting]={
            "rows":len(entries),
            "mean_words":round(mean(lengths),4),
            "median_words":median(lengths),
            "mean_reasoning_tokens":round(mean(reasons),3) if reasons else None,
            "reasoning_tokens_present":len(reasons),
            "em_dash_text_fraction":round(mean("—" in x["text"] for x in entries),4)
        }
    meta={
        "source_ref":SOURCE_REF,"source_sha256":sha(raw),"source_rows":len(dataset),
        "observed_model":"kimi-k3","matched_pairs":EXPECTED_PAIRS,
        "rows":EXPECTED_ROWS,"match_by_prompt_served_and_target":True,
        "selected_prompt_id_hash":sha(json.dumps(
            [records[2*i]["prompt_id"] for i in range(EXPECTED_PAIRS)],
            separators=(",",":")).encode()),
        "selected_passage_hashes_digest":sha(json.dumps(
            sorted(hashes),separators=(",",":")).encode()),
        "served_models":sorted(set(str(x["served_model"]) for x in records)),
        "model_ids":sorted(set(str(x["model_id"]) for x in records)),
        "served_providers":sorted(set(str(x["served_provider"]) for x in records)),
        "source_setting_stats":metrics,
        "mean_high_minus_off_words":round(mean(
            records[2*i]["n_words"]-records[2*i+1]["n_words"]
            for i in range(EXPECTED_PAIRS)),4),
        "no_training":True,"no_submission":True
    }
    return records,meta
