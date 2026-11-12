"""256 original synthetic passages: 128 grouped / 128 singleton.

Purpose: verify frozen Gaussian scores actually contribute on a valid
2..18 member topic group while measuring realistic variable-length
full six-model + GSQ + fivefold inference. No true labels, no BA and no
official Kaggle competition submission.

Distinct group keyword occurs in exactly 4 of 256 rows (1.5625%):
it passes native TfidfVectorizer min_df=2 / max_df=.05. Other generic
words appear in many rows and are discarded. Uniques occur once.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from scripts import ntl_native_gauss_prefit_override_v2_20261113 as base

DEST="kolihhan/ntl-gauss-prefit-256-activated-nov13"
TITLE="ntl-gauss-prefit-256-activated-nov13"
ANCHOR="(RUNTIME/'config.json').write_text(json.dumps(CONFIG),encoding='utf-8')"
INJECT=r'''
# Deterministic ORIGINAL synthetic benchmark. No official train/test rows or labels.
_common_sentences = [
    "The committee discusses observations, assumptions, methods, results, and alternative explanations.",
    "An example provides a concrete basis for comparing predictions with observations and checking errors.",
    "The investigators describe the procedure carefully so that other teams can reproduce the outcome.",
    "Different conditions can lead to different conclusions unless the relevant variables are controlled.",
    "A sound report distinguishes established evidence from exploratory hypotheses and remaining questions.",
    "The summary explains a plausible interpretation and notes what a broader experiment would require.",
    "Researchers can compare measurements at multiple times and examine whether patterns remain stable."
]
_original_group_benchmark=[]
for _index in range(256):
    _is_grouped=_index<128
    _gid=_index//4 if _is_grouped else _index-128
    _word=("groupcluster"+f"{_gid:03d}") if _is_grouped else ("singlecase"+f"{_gid:03d}")
    _repeat=(2,4,7,10,4,7,10,2)[_index%8]
    _prefix=("This study concerns "+_word+" and how observations can be interpreted. "
             +"We focus on the "+_word+" example to compare alternative explanations. ")
    _body=" ".join(_common_sentences*_repeat)
    _original_group_benchmark.append({"id":f"sim{_index:06d}","text":_prefix+_body})
_synthetic_input=Path("/kaggle/working/ntl_gauss_prefit_256_synthetic.csv")
pd.DataFrame(_original_group_benchmark,columns=["id","text"]).to_csv(_synthetic_input,index=False)
assert len(_original_group_benchmark)==256 and _synthetic_input.is_file()
assert len(set(r["id"] for r in _original_group_benchmark))==256
DATA=_synthetic_input
CONFIG["paths"]["test_csv"]=str(_synthetic_input)
EXPLICIT_PATHS["paths"]["test_csv"]["path"]=str(_synthetic_input)
(RUNTIME/"legacy/explicit-paths.json").write_text(json.dumps(EXPLICIT_PATHS),encoding="utf-8")
print("NTL_GAUSS256_SYNTHETIC_INPUT="+json.dumps({
 "rows":256,"grouped_design_rows":128,"candidate_groups":32,"singletons":128,
 "input_origin":"original_deterministic_synthetic","label_truth_available":False,
 "official_test_input_used":False,"no_training":True
},sort_keys=True),flush=True)
'''
POST=r'''
print("NTL_GAUSS256_ELAPSED_SECONDS="+str(round(time.monotonic()-float(os.environ["NTL_STARTED_MONOTONIC"]),3)),flush=True)
print("NTL_GAUSS256_NO_OFFICIAL_COMPETITION_SUBMISSION=TRUE",flush=True)
'''


def text(c:dict)->str:
    v=c.get("source",[])
    return "".join(v) if isinstance(v,list) else str(v)


def make(out:Path)->dict:
    prior=base.make(out)
    root=out/"kernel"
    notebooks=list(root.glob("*.ipynb"))
    if len(notebooks)!=1:raise ValueError("frozen Notebook ambiguous")
    notebook=notebooks[0]
    nb=json.loads(notebook.read_text(encoding="utf-8"))
    count=0
    for c in nb["cells"]:
        if c.get("cell_type")!="code":continue
        src=text(c)
        if src.startswith("CONFIG['gsq_share'] = 0.03"):
            if src.count(ANCHOR)!=1:raise ValueError("old config binding drift")
            if "NTL_GAUSS256_SYNTHETIC_INPUT=" in src:raise ValueError("repeated injection")
            c["source"]=(src.replace(ANCHOR,INJECT+"\n"+ANCHOR,1)+"\n"+POST).splitlines(keepends=True)
            count+=1
    if count!=1:raise ValueError("must change one final execution cell")
    src="\n".join(text(c) for c in nb["cells"] if c.get("cell_type")=="code")
    if src.count("NTL_GAUSS256_SYNTHETIC_INPUT=")!=1 or src.count("NTL_GAUSS_PREFIT_SCORES=")!=1:
        raise ValueError("benchmark or Gaussian score observability missing")
    for guard in base.base.base.base.GUARDS:
        if src.count(guard)!=1:raise ValueError("new supervised fitting guard drift")
    metadata_path=root/"kernel-metadata.json"
    meta=json.loads(metadata_path.read_text())
    if not meta["enable_gpu"] or "docker_image" in meta or len(meta["model_sources"])!=6:
        raise ValueError("gpu provisioning or assets changed")
    meta["id"]=DEST;meta["title"]=TITLE
    metadata_path.write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n")
    encoded=json.dumps(nb,ensure_ascii=False).encode("utf-8")
    notebook.write_bytes(encoded)
    return {
      "candidate":DEST,"synthetic_rows":256,"group_design":"32 x 4 plus 128 singleton",
      "expected_eligible_for_gaussian":128,"real_official_test_used":False,
      "new_supervised_training":False,"pretrained_gaussian_weights":{"q14v5":0.7,"q8v5":0.44},
      "no_official_submission":True,"source_sha256":hashlib.sha256(encoded).hexdigest(),
      "underlying_candidate":prior["slug"]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();print("NTL_GAUSS256_PREPARED="+json.dumps(make(a.out),sort_keys=True),flush=True)


if __name__=="__main__":
    main()
