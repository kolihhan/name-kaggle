"""NTL publicly shared high-ranked notebook mechanism census, source-only.

Retrieve Kaggle's publicly listed 'scoreDescending' (which exposes NO numeric
score in CLI v2.2.5), then inspect 22 top refs plus known 0.717 0.719
lineages. Do not conflate list rank, title score, or notebook best with official
owner-score. No execution/training/model-data download/submissions.

Summarize module lineage and feature families, NOT notebook source passages.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

REQUIRED = [
"dennis6886/glm-user-prefix-guarded-fable-return",
"areebbb/0-719-preserve-glm-to-fable-corrections",
"asteriospanoras/0-718-which-llms-look-alike-error-map-and-ideas",
"sanpier/0-709-320-token-glm-likelihood-qwen-ensemble/3",
"estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix/5",
]
OWNER="kolihhan/ntl-estopadilla-glm53-0716-repro-20261122"
LIMIT=25
MAX_TOTAL=30
MAX_MODULES=250
MARKERS = {
 "pretrained_backbone":r"from_pretrained\s*\(|(?:%%writefile\s+)(?:infer|atlas_infer|adapter|pipe)\.py",
 "multiple_gpu_experts":r"gsq_stage|native_q8|atlas_infer|qwen_groups|comp_stack",
 "glm_causal_likelihood":r"v1_stage\.py|prefix_variant_stage\.py|glm53_scores|glm53-segs|ll_model\.json",
 "glm_user_prefix":r"prefix_decisions\.py|fable_return|prefix_variant_stage\.py",
 "stylometry":r"ntl660_readout\.py|TfidfVectorizer|function.words|stylometry|punctuation",
 "soft_prompt_grouping":r"gauss\.py|group_covariance|group.*soft|s10.*group",
 "family_residual":r"pngy_residual\.py|Residual\s*\(|family_columns\s*\(",
 "hard_matching":r"linear_sum_assignment|hungarian|one.per.model",
 "model_fitting_code":r"\.fit\s*\(|\.train\s*\(|optimizer\.step\s*\(",
 "train_only_eda":r"prompt.by.model|eda|eta.squared|model_eta_squared|train\[[\"']prompt_id",
}
KEY_MODULES={
"adapter.py","atlas_infer.py","infer.py","native_q8.py","pipe.py",
"pngy_residual.py","gauss.py","ntl660_readout.py","comp_stack.py",
"prefix_decisions.py","prefix_variant_stage.py","apex_postprocess.py",
"/kaggle/temp/v1/v1_stage.py","/kaggle/temp/v1/features.py"
}
def emit(event,obj):
    print("NTL_CENSUS_"+event+"="+json.dumps(obj,sort_keys=True,ensure_ascii=False),flush=True)

def call(args,timeout=110):
    try:return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    except Exception as e:
        return type("Bad",(),{"returncode":-1,"stdout":"","stderr":type(e).__name__})()

def get_ranking():
    r=call(["kaggle","kernels","list","--competition","name-that-llm",
            "--sort-by","scoreDescending","--page","1","--page-size","50","--csv"],timeout=90)
    if r.returncode:
        emit("RANKING_UNAVAILABLE",{"returncode":r.returncode,"errtype":"CLI_nonzero"})
        return []
    rows=list(csv.DictReader(io.StringIO(r.stdout.lstrip("\ufeff"))))
    refs=[]
    for entry in rows:
        ref=(entry.get("ref") or "").strip()
        if re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+",ref):
            refs.append((ref,entry.get("title") or ""))
    emit("RANKING_METADATA",{"n":len(refs),"fields":list(rows[0]) if rows else [],
      "reported_score_column_exists": bool(rows and any("score" in c.lower() for c in rows[0])),
      "ordered_by_server_scoreDescending_request":True,
      "ordering_not_numeric_score_proof":True,
      "first_25_refs":[x[0] for x in refs[:25]]})
    return refs

def read_nb(ref,dest):
    dest.mkdir(parents=True,exist_ok=True)
    p=call(["kaggle","kernels","pull",ref,"-p",str(dest),"-m"],timeout=110)
    if p.returncode:return {"ok":False,"reason":"cannot_pull","returncode":p.returncode}
    mdfile=dest/"kernel-metadata.json"
    if not mdfile.is_file():return {"ok":False,"reason":"missing_meta"}
    md=json.loads(mdfile.read_text())
    nbs=list(dest.glob("*.ipynb"))
    if md.get("code_file") and (dest/md["code_file"]).is_file():
        nbs=[dest/md["code_file"]]
    if len(nbs)!=1:return {"ok":False,"reason":"not_one_nb","n":len(nbs)}
    nb=json.loads(nbs[0].read_text())
    cells=["".join(c["source"]) if isinstance(c.get("source"),list) else str(c.get("source",""))
           for c in nb.get("cells",[]) if c.get("cell_type")=="code"]
    code="\n".join(cells)
    modules={}
    for cell in cells:
        first=cell.splitlines()[0] if cell.splitlines() else ""
        m=re.match(r"\s*%%writefile\s+(?:-a\s+)?([a-zA-Z0-9_./-]+)",first)
        if m:
            key=m.group(1)
            body=cell.split("\n",1)[-1]
            if key in modules:key+="_DUP"
            modules[key]=hashlib.sha256(body.encode()).hexdigest()
    if len(modules)>MAX_MODULES:return {"ok":False,"reason":"too_many_modules"}
    counts={name:bool(re.search(pat,code,re.I)) for name,pat in MARKERS.items()}
    kcores={k:v for k,v in modules.items() if k in KEY_MODULES or Path(k).name in KEY_MODULES}
    return {"ok":True,"sha":hashlib.sha256(code.encode()).hexdigest(),
            "code_cells":len(cells),"code_chars":len(code),
            "model_sources":len(md.get("model_sources") or []),
            "dataset_sources":len(md.get("dataset_sources") or []),
            "kernel_sources":len(md.get("kernel_sources") or []),
            "input_fingerprint":hashlib.sha256(json.dumps(
                 {k:sorted(md.get(k) or []) for k in
                 ("model_sources","dataset_sources","kernel_sources")},
                 sort_keys=True).encode()).hexdigest(),
            "modules":kcores,"markers":counts,
            "exact_version_returned_in_metadata":bool(md.get("version")),
            "metadata_ref":md.get("id","")}

def main():
    ranking=get_ranking()
    selected=[]
    for rank,(ref,title) in enumerate(ranking[:LIMIT],1):
        selected.append((rank,ref,title))
    for ref in REQUIRED+[OWNER]:
        if not any(x[1].split("/")[0:2]==ref.split("/")[0:2] for x in selected):
            selected.append((None,ref,""))
    # Ensure true distinct ref; V3 and latest may map to same upstream slug,
    # but explicit targeted V3 is retained.
    for ref in REQUIRED:
        if not any(x[1]==ref for x in selected):selected.append((None,ref,""))
    selected=selected[:MAX_TOTAL]
    with tempfile.TemporaryDirectory(prefix="ntl-public-highrank-source-") as t:
        results=[]
        for idx,(rank,ref,title) in enumerate(selected):
            x=read_nb(ref,Path(t)/("n"+str(idx)))
            if not x["ok"]:
                emit("SOURCE_FAILED",{"rank":rank,"ref":ref,"error":x["reason"]})
                continue
            results.append({"rank":rank,"ref":ref,"title":title,**x})
        anchor=next((x for x in results if x["ref"]==OWNER),None)
        # First earlier code-audit pinned SHA, fail on unexpected drift.
        OWNER_SHA="202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8"
        if anchor is not None and anchor["sha"]!=OWNER_SHA:
            raise RuntimeError("owner_scored_code_drift_no_comparable_lineage")
        for x in results:
            sameness=0;overlap=0
            if anchor:
                for k,v in x["modules"].items():
                    if k in anchor["modules"]:
                        overlap+=1
                        sameness+=(v==anchor["modules"][k])
            emit("PUBLIC_NOTEBOOK",{"ranked_index":x["rank"],"ref":x["ref"],
                "code_sha":x["sha"],"same_exact_owner_code":bool(anchor and x["sha"]==anchor["sha"]),
                "code_cells":x["code_cells"],
                "attached_models":x["model_sources"],
                "attached_datasets":x["dataset_sources"],
                "attached_kernels":x["kernel_sources"],
                "same_input_manifest_as_owner":bool(anchor and x["input_fingerprint"]==anchor["input_fingerprint"]),
                "scoring_modules_overlap_owner":overlap,"scoring_modules_identical_owner":sameness,
                "modules":sorted(x["modules"]),
                "mechanism_markers":x["markers"],
                "score":None,"score_not_exposed_by_Kaggle_list_CLI":True})
        # Cohort sizes and patterns are restricted to pulled *selected* notebooks
        ranked=[x for x in results if x["rank"] is not None]
        unique_sha={x["sha"] for x in ranked}
        c=Counter()
        for x in ranked:
            for k,v in x["markers"].items():
                c[k]+=int(v)
        hist=Counter(x["model_sources"] for x in ranked)
        emit("AGGREGATE",{
            "ranked_requested":min(LIMIT,len(ranking)),
            "ranked_successfully_pulled":len(ranked),
            "ranked_unique_source_sha":len(unique_sha),
            "all_refs_successfully_pulled":len(results),
            "models_mount_histogram":dict(sorted(hist.items())),
            "family_marker_presence_counts":dict(sorted(c.items())),
            "ranked_attached_9_models":sum(x["model_sources"]==9 for x in ranked),
            "ranked_exact_0_new_model_mounts":sum(x["model_sources"]==0 for x in ranked),
            "ranked_strong_backbone_plus_GLM_family":sum(
                x["markers"]["multiple_gpu_experts"] and x["markers"]["glm_causal_likelihood"]
                for x in ranked),
            "ranked_strong_backbone_plus_style":sum(
                x["markers"]["multiple_gpu_experts"] and x["markers"]["stylometry"] for x in ranked),
            "sample_is_public_pulled_code_not_all_highscore_teams":True,
            "codes_can_contain_nonexecuted_marker_text":True,
            "no_verified_scoring_numbers_from_title_or_order":True})
    emit("END",{"no_training":True,"no_model_execution":True,"no_GPU":True,
                 "no_competition_submission":True,"source_not_persisted":True})
if __name__=="__main__":main()
