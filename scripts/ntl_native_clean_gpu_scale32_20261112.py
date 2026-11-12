"""Bounded full Native 32-row synthetic throughput experiment, no model fitting.

Build from the exact frozen, CUDA-clean, inference-only Native candidate.
Changes only the last executing Notebook cell to supply deterministic
original synthetic texts (not official data) and report wall time.
DO NOT score these artificial texts, DO NOT submit to competition.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from scripts import ntl_native_clean_gpu_frozen_prefit_20261112 as base

DEST="kolihhan/ntl-native-clean-synthetic32-nov12"
TITLE="ntl-native-clean-synthetic32-nov12"
N=32
ANCHOR="(RUNTIME/'config.json').write_text(json.dumps(CONFIG),encoding='utf-8')"
TIMER_MARKER="NTL_SCALE32_ELAPSED_SECONDS"
DATA_MARKER="NTL_SCALE32_SYNTHETIC_INPUT_CREATED"

# Create all synthetic benchmark passages inside Kaggle: no official data
# are copied, displayed or committed. No true labels, hence no BA estimate.
SYNTHETIC_CELL=r'''
# One-time synthetic scaling test. NOT VALIDATION DATA, NEVER AN OFFICIAL SUBMISSION.
_ntl_scale_topics = (
  "astronomy","geology","botany","robotics","architecture","history","ecology","music",
  "mathematics","meteorology","navigation","chemistry","photography","anthropology",
  "linguistics","geography"
)
_ntl_scale_verbs = (
  "explains","compares","questions","describes","reframes","tests","contrasts","explores"
)
_ntl_scale_rows = []
for _ntl_i in range(32):
    _ntl_topic = _ntl_scale_topics[_ntl_i % len(_ntl_scale_topics)]
    _ntl_verb = _ntl_scale_verbs[_ntl_i % len(_ntl_scale_verbs)]
    _ntl_sentences = [
      "An experienced guide " + _ntl_verb + " how " + _ntl_topic + " connects careful observation with everyday problem solving.",
      "People often start with a familiar example before asking what might change under unusual conditions.",
      "A useful account separates what the evidence establishes from the guesses that still need a fair test.",
      "Different explanations can sound persuasive until they are compared on the same concrete example.",
      "The most informative question is whether a proposed distinction continues to hold outside its original setting.",
      "For that reason, the group records assumptions explicitly and checks them using a repeatable procedure.",
      "The final observation is modest but helpful because it describes exactly what was tested and what remains unknown.",
    ]
    _ntl_text = " ".join(_ntl_sentences[_ntl_i%7:]+_ntl_sentences[:_ntl_i%7])
    _ntl_scale_rows.append({"id":f"scale{_ntl_i:06d}","text":_ntl_text})
_ntl_scale_path = Path("/kaggle/working/ntl_scale32_synthetic.csv")
pd.DataFrame(_ntl_scale_rows, columns=["id","text"]).to_csv(_ntl_scale_path,index=False)
assert len(_ntl_scale_rows)==32 and _ntl_scale_path.is_file()
DATA = _ntl_scale_path
CONFIG["paths"]["test_csv"] = str(_ntl_scale_path)
EXPLICIT_PATHS["paths"]["test_csv"]["path"] = str(_ntl_scale_path)
(RUNTIME/"legacy/explicit-paths.json").write_text(json.dumps(EXPLICIT_PATHS),encoding="utf-8")
print("NTL_SCALE32_SYNTHETIC_INPUT_CREATED="+json.dumps({
 "rows":len(_ntl_scale_rows),"input_origin":"deterministic_original_synthetic",
 "benchmark_only":True,"labels_known":False,"no_hidden_test_text_used":True
},sort_keys=True),flush=True)
'''

POST=r'''
print("NTL_SCALE32_ELAPSED_SECONDS="+str(round(time.monotonic()-float(os.environ['NTL_STARTED_MONOTONIC']),3)),flush=True)
print("NTL_SCALE32_NO_TRAINING_NO_OFFICIAL_COMPETITION_SUBMISSION=True",flush=True)
'''

def source(cell:dict)->str:
    s=cell.get("source",[])
    return "".join(s) if isinstance(s,list) else str(s)


def make(out:Path)->dict:
    initial=base.make(out)
    root=out/"kernel"
    candidates=list(root.glob("*.ipynb"))
    if len(candidates)!=1:raise ValueError("exact Native source ambiguity")
    target=candidates[0]
    nb=json.loads(target.read_text(encoding="utf-8"))
    if len(nb["cells"])<52:raise ValueError("Native notebook cell count drift")
    hits=0
    for cell in nb["cells"]:
        if cell.get("cell_type")!="code":continue
        text=source(cell)
        if text.startswith("CONFIG['gsq_share'] = 0.03"):
            if text.count(ANCHOR)!=1 or TIMER_MARKER in text or DATA_MARKER in text:
                raise ValueError("not exact frozen final executing cell")
            if "actual = pd.read_csv(DATA,dtype={'id':str})" not in text:
                raise ValueError("missing same-row output integrity gate")
            modified=text.replace(ANCHOR,SYNTHETIC_CELL+"\n"+ANCHOR,1)+"\n"+POST
            cell["source"]=modified.splitlines(keepends=True)
            hits+=1
    if hits!=1:raise ValueError("expected one notebook instrumentation point")
    full="\n".join(source(c) for c in nb["cells"] if c["cell_type"]=="code")
    if full.count(DATA_MARKER)!=1 or full.count(TIMER_MARKER)!=1:
        raise ValueError("missing synthetic input or elapsed marker")
    for guard in base.base.GUARDS:
        if full.count(guard)!=1:raise ValueError("supervised fitting guard not intact")
    if "preview = False  # diagnostic: force real inference path" not in full:
        raise ValueError("fixture preview still enabled")
    expected=initial["source_code_sha256"]
    if expected!="a588221cec3db8b1fc1ad4b525614fc28dc70582f4f45f5268c58e8bf2373eef":
        raise ValueError("original source provenance SHA mismatch")
    meta_path=root/"kernel-metadata.json"
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    if "docker_image" in meta or [len(meta.get(k) or []) for k in (
        "dataset_sources","kernel_sources","model_sources")]!=[21,8,6]:
        raise ValueError("lost clean native GPU metadata")
    meta["id"]=DEST;meta["title"]=TITLE
    meta_path.write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    content=json.dumps(nb,ensure_ascii=False).encode("utf-8")
    target.write_bytes(content)
    return {
      "slug":DEST,"rows":N,"input":"pure_original_synthetic_text",
      "run_model_weights":True,"no_new_supervised_training":True,
      "no_labels_or_balanced_accuracy":True,"no_official_competition_submission":True,
      "pretrained_mount_counts":{"datasets":21,"kernels":8,"models":6},
      "clean_gpu_metadata":True,"unchanged_model_loader":True,
      "sha256":hashlib.sha256(content).hexdigest()
    }

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print("NTL_SCALE32_CPU_BUILD="+json.dumps(make(a.out),sort_keys=True),flush=True)

if __name__=="__main__":
    main()
