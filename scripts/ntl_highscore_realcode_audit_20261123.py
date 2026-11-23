"""NTL high-score public-code forensic comparison; source-only, no model/data/predictions.

Intentionally does not run notebooks, download source datasets or model weights,
evaluate hidden/public test, fit anything, or submit to a competition.
The Kaggle code pull is to temporary scratch, deleted on process exit.

IMPORTANT: Kaggle notebook URL /5 or /3 is not proof of returned version.
We report the code reference and source SHA, and distinguish latest-vs-scored.
"""
from __future__ import annotations

import csv
import difflib
import hashlib
import io
import json
import re
import subprocess
import tempfile
from pathlib import Path

EXPECTED_OWNER_CODE_SHA = "202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8"
SOURCES = {
    "owner0715": "kolihhan/ntl-estopadilla-glm53-0716-repro-20261122",
    "estopadilla_scored_v5": "estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix/5",
    "sanpier_scored_v3": "sanpier/0-709-320-token-glm-likelihood-qwen-ensemble/3",
    "sanpier_latest": "sanpier/0-709-320-token-glm-likelihood-qwen-ensemble",
    "dennis_latest": "dennis6886/glm-user-prefix-guarded-fable-return",
    "evgen_latest": "evgendvorkin/name-that-llm-deep-forensic-eda-with-analogie",
}
SIGNALS = {
    "qwen_lora": r"qwen.{0,40}lora|lora.{0,40}qwen",
    "glm_likelihood": r"glm.{0,50}likelihood|likelihood.{0,50}glm|glm_logprob",
    "glm_prefix": r"user.{0,30}prefix|prefix.{0,40}glm|prefix_decisions",
    "fable_return": r"fable_return|fable.{0,30}guard|fable.{0,30}route",
    "family_residual": r"pngy_residual|family.{0,40}residual|family.{0,40}posterior",
    "style_model": r"ntl660_readout|style.{0,40}model|stylometry",
    "soft_groups": r"group.{0,40}soft|group.{0,40}covariance|gauss\.py",
    "hard_hungarian": r"hungarian|linear_sum_assignment",
    "final_logits": r"save_final_logits|final-logits\.npz",
    "model_training": r"(?<!\.)\.backward\s*\(|optimizer\.step\s*\(|peft\.get_peft_model|PeftModel.from_pretrained",
    "pretrained_load": r"from_pretrained\s*\(",
    "model_runtime_limit": r"deadline|RUN_LIMIT_S|TIME_BUDGET|320|timeout",
    "submission": r"submission\.parquet",
}
SOURCE_PAT = re.compile(
    r"(?:glm|likelihood|prefix|fable|qwen|sibling|family|residual|gauss|logit|"
    r"calibr|sinkhorn|center|weight|guard|router|inference|score|token|model)",
    re.I,
)

def print_event(tag, obj):
    print("NTL_HIGHCODE_" + tag + "=" + json.dumps(obj, sort_keys=True, ensure_ascii=False), flush=True)


def pull(name, ref, scratch):
    dest = scratch / name
    dest.mkdir()
    try:
        p = subprocess.run(
            ["kaggle", "kernels", "pull", ref, "-p", str(dest), "-m"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=125)
    except Exception as exc:
        print_event("PULL", {"key": name, "requested_ref": ref,
                             "ok": False, "error_type": type(exc).__name__})
        return None
    if p.returncode != 0:
        print_event("PULL", {"key": name, "requested_ref": ref, "ok": False,
                             "returncode": p.returncode,
                             "error_category": "version-or-access-or-rate-limit"})
        return None
    mdpath = dest / "kernel-metadata.json"
    md = json.loads(mdpath.read_text()) if mdpath.is_file() else {}
    candidates = list(dest.glob("*.ipynb"))
    if md.get("code_file") and (dest/md["code_file"]).suffix==".ipynb":
        candidates = [dest / md["code_file"]]
    if len(candidates) != 1 or not candidates[0].is_file():
        print_event("PULL", {"key":name,"requested_ref":ref,"ok":False,
                             "error_category":"no_single_ipynb", "count":len(candidates)})
        return None
    nb = json.loads(candidates[0].read_text())
    cells = [
        ''.join(c.get('source',[])) if isinstance(c.get('source',[]), list)
        else str(c.get('source',''))
        for c in nb.get("cells",[]) if c.get("cell_type")=="code"
    ]
    code = '\n'.join(cells)
    sha = hashlib.sha256(code.encode()).hexdigest()
    parts = {}
    for i,source in enumerate(cells):
        lines = source.splitlines()
        match = re.match(r"\s*%%writefile\s+(?:-a\s+)?([a-zA-Z0-9_./-]+)",lines[0]) if lines else None
        part_key = match.group(1) if match else f"CELL_{i:02d}"
        if part_key in parts:
            part_key += f"_COPY_{i:02d}"
        parts[part_key]='\n'.join(lines[1:] if match else lines)
    counts={name:int(len(re.findall(pat,code,re.I|re.S))) for name,pat in SIGNALS.items()}
    names = []
    for part,data in parts.items():
        if SOURCE_PAT.search(part) or SOURCE_PAT.search(data[:180]):
            names.append(part)
    info = {
        "key":name,"requested_ref":ref,"pulled":True,"actual_code_sha":sha,
        "code_cells":len(cells),"code_chars":len(code),
        "module_part_names":names[:55],
        "signals":counts,"model_sources_count":len(md.get("model_sources") or []),
        "dataset_sources_count":len(md.get("dataset_sources") or []),
        "kernel_sources_count":len(md.get("kernel_sources") or []),
        "metadata_kernel_id":md.get("id",""),
        "version_in_pulled_metadata":md.get("version","NOT_EXPOSED"),
        "version_pinned_response_verified":False,  # CLI pull does not prove request exact version.
        "training_run_performed":False,
    }
    if name=="owner0715":
        info["matches_prior_independent_scored_source_sha"] = sha == EXPECTED_OWNER_CODE_SHA
        if sha != EXPECTED_OWNER_CODE_SHA:
            raise ValueError("owner_scored_code_sha_drift")
    print_event("SOURCE",info)
    return {"sha":sha,"parts":parts,"code":code,"info":info}


def compare(base, name, target):
    b,t=base["parts"],target["parts"]
    both=set(b)&set(t)
    changed=[k for k in both if b[k].strip()!=t[k].strip()]
    new=sorted(set(t)-set(b))
    deleted=sorted(set(b)-set(t))
    print_event("DELTA_SUMMARY",{
        "other":name,"same_owner_code":base["sha"]==target["sha"],
        "identical_parts":len(both)-len(changed),
        "different_shared_parts":len(changed),
        "added_parts":len(new),"removed_parts":len(deleted),
        "changed_part_names":sorted(changed)[:45],
        "new_part_names":new[:45],"removed_part_names":deleted[:45],
        "score_attributable_to_delta":False,
    })
    # Ranking by actual overlapping scope: exact file-code differences are more
    # meaningful than cell-position diffs when a notebook adds EDA/demo cells.
    candidates=[]
    for key in changed:
        left,right=b[key].splitlines(),t[key].splitlines()
        matcher=difflib.SequenceMatcher(a=left,b=right,autojunk=False)
        for tag,i,j,ii,jj in matcher.get_opcodes():
            if tag=="equal":continue
            old=left[i:j]
            newlines=right[ii:jj]
            if not SOURCE_PAT.search('\n'.join(old+newlines)):
                continue
            # Do not print giant embedded base64 or code fixtures.
            text_old='\n'.join(old)[:450]
            text_new='\n'.join(newlines)[:650]
            if any(len(line)>700 for line in old+newlines):continue
            candidates.append({"part":key,"type":tag,"owner_line":i+1,
                               "other_line":ii+1,
                               "old":text_old,"new":text_new,
                               "changed_lines":len(old)+len(newlines)})
    # Bound printed excerpts; no notebook code or weight files persisted.
    candidates.sort(key=lambda z:(-int(any(k in (z["old"]+z["new"]).lower()
                     for k in ("glm","fable_return","prefix","likelihood","final_logit","guard"))),
                     z["part"],z["other_line"]))
    print_event("TOP_DECISION_EDITS",{"other":name,"edits_total":len(candidates),
                                     "evidence":candidates[:11]})


def public_ranking():
    p=subprocess.run(
        ["kaggle","kernels","list","--competition","name-that-llm",
         "--sort-by","scoreDescending","--page","1","--page-size","50","--csv"],
        capture_output=True,text=True,timeout=95)
    if p.returncode!=0:
        print_event("RANKING",{"available":False,"reason":"CLI_nonzero"})
        return
    rows=list(csv.DictReader(io.StringIO(p.stdout.lstrip('\ufeff'))))
    found=[]
    for v in rows:
        ref=str(v.get("ref") or v.get("kernelRef") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+",ref):
            continue
        score=str(v.get("score") or v.get("publicScore") or "")
        found.append({"ref":ref,"claimed_api_score":score if score else None,
                      "title":str(v.get("title") or "")[:95]})
    print_event("RANKING",{"returned":len(rows),"cols":list(rows[0]) if rows else [],
                            "top15":found[:15],
                            "blank_score_is_not_verified_score":True})

def main():
    public_ranking()
    with tempfile.TemporaryDirectory(prefix="ntl-highscore-source-audit-") as d:
        scratch=Path(d)
        result={}
        for name,ref in SOURCES.items():
            result[name]=pull(name,ref,scratch)
        anchor=result.get("owner0715")
        if anchor is None:
            raise RuntimeError("No exact authenticated owner scored code: cannot compare")
        for name in SOURCES:
            if name=="owner0715" or result.get(name) is None:continue
            compare(anchor,name,result[name])
        # Version-level integrity is never inferred from public notebook title.
        # The /3 and /5 requested refs might be served as current by an API
        # implementation: explicitly check whether SHA differs from latest.
        v3=result.get("sanpier_scored_v3")
        latest=result.get("sanpier_latest")
        if v3 is not None and latest is not None:
            print_event("SANPIER_V3_VS_LATEST",{
                "requested_score_version":3,
                "sha_equal":v3["sha"]==latest["sha"],
                "v3_sha":v3["sha"],"latest_sha":latest["sha"],
                "V3_score_0717_is_from_public_kaggle_metadata_not_source_claim":True,
                "if_sha_equal_exact_version_not_automatically_proven":True})
    print_event("END",{"done":True,"no_training":True,"no_GPU":True,
                         "no_official_test":True,"no_submission":True,
                         "source_files_not_uploaded":True})

if __name__=="__main__":
    main()
