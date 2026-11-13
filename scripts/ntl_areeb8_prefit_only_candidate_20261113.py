"""NTL exploratory 8-model public-source inference-only candidate.

Provenance: areebbb/0-684-i-m-better, immutable SHA, public Kaggle code.
This is a materially different 8-model+complementary-expert candidate from
our 0.679 six-model champion. It is NOT a proven 0.684 reproduction.

Two hidden-rerun hazards are explicitly corrected:
1. remove known 3-row constant-label FAST_COMMIT shortcut;
2. disable run-time supervised SVC and Gaussian fitting, preserving the
   original prefit core and neural weights. No model or labels are trained.

The original score claim in a title must never be reported as our score.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

SOURCE = "areebbb/0-684-i-m-better"
DEST = "kolihhan/ntl-areeb8-prefit-only-nov13"
EXPECTED_SHA = "5d053f49005df0b2511e07bfb154faf53f22bca130761ac4a17b61aa5939cb00"
FAST = "FAST_COMMIT = len(pd.read_csv(DATA + '/test.csv')) < 50"
GUARD = """def _ntl_no_supervised_fit(*args, **kwargs):
    raise RuntimeError('NO_NEW_TRAINING: supervised classifier fitting forbidden')
from sklearn.svm import LinearSVC as _ntl_svc_type
from sklearn.linear_model import LogisticRegression as _ntl_lr_type
_ntl_svc_type.fit = _ntl_no_supervised_fit
_ntl_lr_type.fit = _ntl_no_supervised_fit
"""
# Existing a.svc_w=0, a.distort_w=0 and restored frozen core votes are kept.
# Do not patch gauss.fit to run or refit; disallow the two original flags.


def source(cell):
    v=cell.get("source",[])
    return "".join(v) if isinstance(v,list) else str(v)


def build(source_dir:Path, target:Path)->dict:
    paths=list(source_dir.glob("*.ipynb"))
    if len(paths)!=1:raise ValueError("expected one immutable source notebook")
    old=paths[0].read_bytes()
    if hashlib.sha256(old).hexdigest()!=EXPECTED_SHA:
        raise ValueError("public source SHA drift; refuse alternate model")
    nb=json.loads(old)
    meta=json.loads((source_dir/"kernel-metadata.json").read_text())
    if (meta.get("id")!=SOURCE or meta.get("is_private") is not False or
        len(meta.get("model_sources") or [])!=8 or
        len(meta.get("dataset_sources") or [])!=17 or
        len(meta.get("kernel_sources") or [])!=8 or
        meta.get("competition_sources")!=["name-that-llm"]):
        raise ValueError("public source asset or owner drift")
    changed={"fast":0,"flags":0,"runtime_guard":0}
    for cell in nb["cells"]:
        if cell.get("cell_type")!="code":continue
        code=source(cell)
        if FAST in code:
            if code.count(FAST)!=1:raise ValueError("ambiguous FAST_COMMIT")
            code=code.replace(FAST,"FAST_COMMIT = False  # always genuine inference",1)
            changed["fast"]+=1
        if code.startswith("MODELS = ") or code.startswith("LAYERS = [2, 4"):
            # Native source has exactly two Gaussian flags in its final model list.
            for before in ('"gauss": 0.70','"gauss": 0.44'):
                if code.count(before)!=1:raise ValueError("model Gaussian flag source drift")
                code=code.replace(before,'"gauss": 0.0',1)
            if not ("'--svc_w', '0'" in code and "'--distort_w', '0'" in code):
                raise ValueError("supervised SVC/distort tuning not disabled")
            if "if FAST_COMMIT:" not in code:
                raise ValueError("expected real inference else branch missing")
            changed["flags"]+=1
        if code.startswith("%%writefile infer.py"):
            if code.count("_svc.fit(")!=1 or code.count("_ga.fit(")!=1 or code.count("LogisticRegression(")!=1:
                raise ValueError("unexpected supervised fit call inventory")
            lines=code.splitlines(keepends=True)
            if len(lines)<50:raise ValueError("inference module truncated")
            # Add guards after the %%writefile magic, so the generated module
            # blocks every sklearn supervised fit even if an upstream gate drifts.
            lines.insert(1,GUARD+"\n")
            code="".join(lines)
            # fail closed in Gaussian path even if the zero flag is overridden.
            code=code.replace("_ga.fit(","_ntl_no_supervised_fit(",1)
            if code.count("NO_NEW_TRAINING: supervised classifier fitting forbidden")!=1:
                raise ValueError("no training runtime guard lost")
            changed["runtime_guard"]+=1
        cell["source"]=code.splitlines(keepends=True)
    if changed!={"fast":1,"flags":1,"runtime_guard":1}:
        raise ValueError("one or more source patches missing: "+str(changed))
    all_code="\n".join(source(c) for c in nb["cells"] if c.get("cell_type")=="code")
    if (FAST in all_code or "FAST_COMMIT = False" not in all_code
        or all_code.count('"gauss": 0.0')!=2
        or "_ga.fit(" in all_code
        or "NTL_AREEB_NO_SUPERVISED_FIT_GUARD=" in all_code):
        raise ValueError("source mutation failed/ambiguous")
    # A fresh explicit runtime marker to audit this exact modified source.
    marker={"cell_type":"code","execution_count":None,"outputs":[],"metadata":{},
            "source":["print('NTL_AREEB_NO_SUPERVISED_FIT_GUARD=TRUE')\n"]}
    # The runtime training defense lives in the generated infer.py, not only
    # the marker. Append marker as last cell after submitted output integrity.
    nb["cells"].append(marker)
    code_name=meta.get("code_file")
    if not code_name or Path(code_name).name!=code_name or not code_name.endswith(".ipynb"):
        raise ValueError("invalid notebook code_file")
    meta.pop("id_no",None)
    meta.pop("docker_image",None)  # inherited legacy CPU runtime not allowed
    meta.update({"id":DEST,"title":DEST.rsplit("/",1)[1],"is_private":True,
                 "enable_gpu":True,"enable_tpu":False,"enable_internet":False,
                 "machine_shape":"NvidiaTeslaT4"})
    folder=target/"kernel"
    if folder.exists():raise FileExistsError("refuse overwrite")
    folder.mkdir(parents=True)
    encoded=json.dumps(nb,ensure_ascii=False).encode("utf-8")
    (folder/code_name).write_bytes(encoded)
    (folder/"kernel-metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
    return {"source":SOURCE,"new_slug":DEST,"original_sha256":EXPECTED_SHA,
            "candidate_sha256":hashlib.sha256(encoded).hexdigest(),
            "gaussian_disabled_models":["q14v5","q8v5"],
            "prefit_svc_and_distort_only":True,
            "different_eight_model_stack":True,
            "original_preview_bypass_removed":True,
            "runtime_supervised_fit_blocked":True,
            "new_training":False,"official_submitted":False,"accuracy_known":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args()
    print("NTL_AREEB_PREFIT_ONLY_CANDIDATE="+json.dumps(build(x.source,x.out),sort_keys=True),flush=True)


if __name__=="__main__":main()
