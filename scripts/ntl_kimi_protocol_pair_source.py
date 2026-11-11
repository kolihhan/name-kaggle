"""Freeze strictly paired Kimi K3 high/off responses, no training or text logs."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
from statistics import mean,median

EXPECTED_SHA="88ee2c08f4344d86d8f4a54c5ac977bc2f27f5363cd991fadf9411c121a4ea96"
SOURCE="willrice/name-that-llm-openrouter-harvest"
SETTINGS=("high","off")
PAIRS=144

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def clean(row):return row.get("text","").strip() if isinstance(row.get("text"),str) else ""
def integer(row,key):
    try:return int(row[key])
    except (KeyError,TypeError,ValueError):return None

def build(path:Path):
    raw=Path(path).read_bytes()
    digest=sha(raw)
    if digest!=EXPECTED_SHA:raise RuntimeError("Kimi public source bytes changed")
    allrows=[json.loads(s) for s in raw.decode("utf-8").splitlines() if s.strip()]
    if len(allrows)!=12308:raise RuntimeError("Kimi public calls count changed")
    grouped=defaultdict(lambda:defaultdict(list))
    for r in allrows:
        if r.get("model")=="kimi-k3" and r.get("setting") in SETTINGS:
            grouped[str(r.get("prompt_id"))][r["setting"]].append(r)
    fails=Counter();eligible=[]
    for pid,groups in grouped.items():
        if not all(len(groups.get(s,[]))==1 for s in SETTINGS):
            fails["missing_or_duplicate_condition"]+=1;continue
        high,off=(groups[x][0] for x in SETTINGS)
        if any(x.get("finish_reason")!="stop" or not clean(x) for x in (high,off)):
            fails["incomplete_or_empty"]+=1;continue
        if any(high.get(k)!=off.get(k) or high.get(k) in (None,"") for k in
                ("served_model","served_provider","model_id","target_words")):
            fails["served_model_or_target_words_mismatch"]+=1;continue
        nw=(integer(high,"n_words"),integer(off,"n_words"))
        if any(n is None or not 40<=n<=250 for n in nw):
            fails["length_outside_40_250"]+=1;continue
        if sha(clean(high).encode())==sha(clean(off).encode()):
            fails["identical_response"]+=1;continue
        eligible.append((pid,high,off))
    eligible.sort(key=lambda p:(sha(("ntl-kimi-v1:"+p[0]).encode()),p[0]))
    if len(eligible)<PAIRS:
        print("NTL_KIMI_STRICT_ELIGIBILITY_FAILURE",json.dumps({
            "raw_unique_kimi_prompt_ids":len(grouped),
            "strict_eligible_pairs":len(eligible),
            "predeclared_required_pairs":PAIRS,
            "discard_reasons":dict(fails)
        },sort_keys=True),flush=True)
        raise RuntimeError("fewer than frozen 144 eligible Kimi pairs")
    selected=eligible[:PAIRS];seen=set();records=[]
    for i,(pid,high,off) in enumerate(selected):
        for code,entry in (("h",high),("o",off)):
            txt=clean(entry)
            hh=sha(" ".join(txt.split()).casefold().encode())
            if hh in seen:raise RuntimeError("duplicated normalized response text")
            seen.add(hh)
            records.append({"id":f"k{i:04d}{code}","prompt_id":pid,
                "setting":entry["setting"],"model":"kimi-k3","text":txt,
                "n_words":integer(entry,"n_words"),
                "target_words":integer(entry,"target_words"),
                "reasoning_tokens":integer(entry,"reasoning_tokens"),
                "served_model":str(entry["served_model"]),
                "served_provider":str(entry["served_provider"]),
                "model_id":str(entry["model_id"])})
    if len(records)!=2*PAIRS:raise RuntimeError("incorrect selected row count")
    if any(records[2*i]["target_words"]!=records[2*i+1]["target_words"]
           for i in range(PAIRS)):raise RuntimeError("target words mismatch")
    stats={}
    for setting in SETTINGS:
        rr=[r for r in records if r["setting"]==setting]
        words=[r["n_words"] for r in rr]
        reason=[r["reasoning_tokens"] for r in rr if r["reasoning_tokens"] is not None]
        stats[setting]={"n":len(rr),"median_words":median(words),
           "mean_words":round(mean(words),3),
           "mean_reasoning_tokens":round(mean(reason),3) if reason else None,
           "non_null_reasoning_tokens":len(reason),
           "em_dash_rate":round(mean("—" in r["text"] for r in rr),4),
           "semicolon_rate":round(mean(";" in r["text"] for r in rr),4),
           "markdown_bold_rate":round(mean("**" in r["text"] for r in rr),4)}
    report={"source":SOURCE,"sha256":digest,"source_records":len(allrows),
        "raw_kimi_prompt_ids":len(grouped),"eligible_pairs":len(eligible),
        "selected_pairs":PAIRS,"selected_rows":len(records),
        "discard_reasons":dict(fails),
        "selected_prompt_order_sha256":sha(json.dumps(
            [p[0] for p in selected],separators=(",",":")).encode()),
        "selected_normalized_text_hashes_sha256":sha(json.dumps(
            sorted(seen),separators=(",",":")).encode()),
        "model_ids":sorted({r["model_id"] for r in records}),
        "served_models":sorted({r["served_model"] for r in records}),
        "served_providers":sorted({r["served_provider"] for r in records}),
        "target_words_match_in_each_pair":True,
        "condition_summary":stats,
        "mean_high_minus_off_words":round(mean(
            records[2*i]["n_words"]-records[2*i+1]["n_words"]
            for i in range(PAIRS)),3),
        "no_training":True,"no_submission":True,"raw_text_in_logs":False}
    return records,report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--calls",type=Path,required=True)
    p.add_argument("--private-output",type=Path)
    a=p.parse_args()
    rows,summary=build(a.calls)
    print("NTL_KIMI_PROTOCOL_ELIGIBILITY",json.dumps(summary,sort_keys=True),flush=True)
    if a.private_output:
        a.private_output.parent.mkdir(parents=True,exist_ok=True)
        with a.private_output.open("w",encoding="utf-8") as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")

if __name__=="__main__":main()
