"""Private 84-row paired diagnostic: frozen scored 0.683 Areeb8 vs isolated WeMM fold0.

Only 12 observed true labels, seven complete external prompt blocks: SPENT DIAGNOSTIC.
No Kaggle competition submission, no training, no public row-level artifacts, no LB tuning.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from scripts.ntl_0679_transfer_private_once import _make_private_capsule
from scripts.ntl_wemm_isolated_fold0_smoke_20261113 import (
    SOURCE as W_SOURCE, SHA256 as W_SOURCE_SHA, DIGEST as W_DIGEST,
    WEMM_SOURCE, FROZEN_HEADS, RUNTIME, cell_text as wtext, new_code
)

CHAMPION = "kolihhan/ntl-areeb8-prefit-only-nov13"
CHAMP_CODE_SHA256 = "178587738bf3474e78a7701d78254455192a0d16c97815075fa2560e3b695902"
DEST = "kolihhan/ntl-areeb8-wemm-paired84-nov13"
NIMITHA_PROMPTS = 4
TOAN_PROMPTS = 3
ROWS = 84
LABELS12 = [
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.8-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"
]

# This is source-defined input and scoring code, not public rows.
# Both models get EXACTLY the same unlabeled query CSV. Truth never enters either model.
PREPARE = r"""
import hashlib as _ph
import pandas as _pp
import os as _po
from pathlib import Path as _P
L12 = %r
src = _P('/kaggle/temp/ntl_external_capsule')
nim = _pp.read_csv(src/'generated_proxy_candidates.csv')
toan = _pp.read_json(src/'passages.jsonl',lines=True)
assert len(nim)==240 and len(toan)==107
assert set(nim.model)==set(L12) and set(toan.model)==set(L12)
frames=[]
for name, frame, k in [('nimitha',nim,4),('toan',toan,3)]:
    assert {'model','text','prompt_id'} <= set(frame.columns)
    frame=frame.copy()
    frame['prompt_id']=frame.prompt_id.astype(str)
    allp=sorted(str(p) for p,g in frame.groupby('prompt_id')
                if len(g)==12 and set(g.model)==set(L12))
    if len(allp)<k:
        raise RuntimeError('not enough complete 12-class prompt panels: '+name)
    chosen=frame.loc[frame.prompt_id.isin(allp[:k]),['prompt_id','model','text']].copy()
    assert len(chosen)==12*k
    chosen['source']=name
    chosen['text']=chosen.text.fillna('').astype(str)
    assert chosen.text.str.len().gt(0).all()
    frames.append(chosen)
panel=_pp.concat(frames,ignore_index=True)
panel=panel.sort_values(['source','prompt_id','model'],kind='stable').reset_index(drop=True)
assert len(panel)==84 and not panel.text.isna().any()
assert set(panel.model)==set(L12)
assert set(panel.groupby('model').size())=={7}
panel.insert(0,'id',['diag-%03d'%%i for i in range(84)])
def digest(s):return _ph.sha256(' '.join(str(s).split()).casefold().encode()).hexdigest()
assert panel.text.map(digest).is_unique
official_train=_pp.read_csv(_po.path.join(DATA,'train.csv'),usecols=['text'])
overlap=set(panel.text.map(digest))&set(official_train.text.dropna().map(digest))
if overlap:raise RuntimeError('external panel overlaps official train')
ext='/kaggle/temp/ntl-paired84-input'
_P(ext).mkdir(parents=True,exist_ok=False)
_po.symlink(_po.path.realpath(_po.path.join(DATA,'train.csv')),ext+'/train.csv')
panel[['id','text']].to_csv(ext+'/test.csv',index=False)
panel[['id','source','prompt_id','model']].to_csv('/kaggle/temp/ntl-paired84-truth.csv',index=False)
DATA=ext
print('NTL_PAIRED84_INPUT_VERIFIED='+json.dumps({
    'rows':len(panel),'label_support':len(L12),
    'source_counts':panel.source.value_counts().to_dict(),
    'complete_prompt_blocks':7,'group_target_count':7,
    'training_exact_text_overlap':len(overlap),
    'model_inputs_contain_labels':False,
    'source_status':'SPENT_12_LABEL_DIAGNOSTIC_NOT_NEW_18_CLASS_VALIDATION'
},sort_keys=True),flush=True)
del nim,toan,frames,chosen,panel,official_train
""" % (LABELS12,)

WEMM_SCORER = r"""
from pathlib import Path
import sys,json,time,subprocess
import numpy as np
import pandas as pd
RUNTIME=Path('/kaggle/temp/ntl-gsq-v27-runtime')
W=Path('/kaggle/working/wemm-paired84')
W.mkdir(parents=True,exist_ok=True)
config = _ntl_fix_obj(%r)
base,assets=Path(config['base']),Path(config['assets'])
assert base.is_dir() and assets.is_dir()
assert (assets/'fold0'/'trainables.safetensors').is_file()
assert (assets/'fold0'/'calibration.json').is_file()
query=Path(DATA)/'test.csv'
true=pd.read_csv('/kaggle/temp/ntl-paired84-truth.csv',dtype={'id':str,'prompt_id':str})
frame=pd.read_csv(query,dtype={'id':str})
assert len(frame)==len(true)==84 and frame.id.tolist()==true.id.tolist()
out=W/'raw-fold0.npz'
log=W/'fold0-worker.log'
cmd=[sys.executable,'-u',str(RUNTIME/'wemm_embedding_worker.py'),
    '--base',str(base),'--assets',str(assets),
    '--query',str(query),'--out',str(out),
    '--work',str(W/'worker'),'--folds','0',
    '--deadline',str(time.monotonic()+3400.)]
t0=time.monotonic()
with log.open('w') as f:
    p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=3450)
wall=round(time.monotonic()-t0,2)
if p.returncode:
    print('NTL_PAIRED84_WEMM_WORKER_ERROR='+log.read_text(errors='replace')[-1200:],flush=True)
    raise RuntimeError('WeMM frozen worker failed; no official submission')
assert out.is_file()
with np.load(out,allow_pickle=False) as raw:
    assert raw['ids'].tolist()==frame.id.tolist()
    assert raw['folds'].tolist()==[0]
    assert raw['plain'].shape==raw['centered'].shape==(1,84,18)
    assert np.isfinite(raw['plain']).all()
    assert np.isfinite(raw['centered']).all()
    labels=raw['labels'].tolist()
    assert len(labels)==18 and len(set(labels))==18
sys.path.insert(0,str(RUNTIME))
from wemm_fusion import load_wemm
lp,folds=load_wemm([out],assets,frame.id.tolist(),labels,np.arange(84,dtype=int))
assert folds==[0] and lp.shape==(5,84,18) and np.isfinite(lp).all()
assert np.allclose(np.exp(lp).sum(axis=2),1.,atol=1e-5)
wemm=[labels[int(i)] for i in np.argmax(lp[0],axis=1)]
champ=pd.read_parquet('/kaggle/working/submission.parquet')
assert list(champ.columns)==['id','model']
assert len(champ)==84 and set(champ.id)==set(frame.id) and champ.id.is_unique
assert set(champ.model) <= set(labels)
paired=true.merge(champ,on='id',how='left',validate='one_to_one',suffixes=('_true','_champ'))
assert paired.id.tolist()==frame.id.tolist() and paired.model_champ.notna().all()
paired['wemm']=wemm
# All scoring is behind the completed frozen inference paths, not in model inputs.
def report(x):
    n=len(x)
    a=x.model_true==x.model_champ
    b=x.model_true==x.wemm
    classes=sorted(set(x.model_true))
    assert set(classes)==set(%r)
    def macro(ok):
        return round(float(np.mean([ok[x.model_true==c].mean() for c in classes])),6)
    return {
        'rows':n,'true_labels':len(classes),'champ_correct':int(a.sum()),
        'wemm_correct':int(b.sum()),
        'champ_conditional12_BA':macro(a),'wemm_conditional12_BA':macro(b),
        'rescues':int((~a&b).sum()),'regressions':int((a&~b).sum()),
        'both_right':int((a&b).sum()),'both_wrong':int((~a&~b).sum()),
        'prediction_disagreements':int((x.model_champ!=x.wemm).sum()),
        'oracle_any_correct_upper_bound_not_deployable':round(float((a|b).mean()),6)
    }
result={
  'scope':'SPENT_TWO_SOURCE_12_LABEL_DIAGNOSTIC_ONLY',
  'no_new_training':True,'official_submission':False,
  'champion_pinned_official_public_BA':0.683,
  'champion_official_submission_ref':'58295802',
  'source_prompt_blocks':7,'wemm_folds_evaluated':[0],
  'weMM_calibration_mode':'singleton_no_prompt_oracle',
  'wemm_wall_seconds':wall,
  'combined':report(paired),
  'per_source':{s:report(x) for s,x in paired.groupby('source')},
  'private_row_level_output':'not_published'
}
(W/'paired84-summary.json').write_text(json.dumps(result,indent=2,sort_keys=True))
print('NTL_PAIRED84_FROZEN_RESCUE_REGRESSION='+json.dumps(result,sort_keys=True),flush=True)
print('NTL_PAIRED84_NO_OFFICIAL_SUBMISSION_OR_TRAINING=TRUE',flush=True)
"""


def source_code(nb):
    return [wtext(c) for c in nb["cells"] if c.get("cell_type")=="code"]


def build(champ_dir: Path, omkar_dir: Path, output_dir: Path, capsule_dir: Path):
    champion=list(champ_dir.glob("*.ipynb"))
    public=list(omkar_dir.glob("*.ipynb"))
    if len(champion)!=1 or len(public)!=1:
        raise RuntimeError("expected one notebook per frozen source")
    champ_nb=json.loads(champion[0].read_text())
    champ_src="\n".join(source_code(champ_nb))
    champ_sha=hashlib.sha256(champ_src.encode()).hexdigest()
    if champ_sha!=CHAMP_CODE_SHA256:
        raise RuntimeError("official 0.683 champion executable source sha drift: "+champ_sha)
    omkar_bytes=public[0].read_bytes()
    if hashlib.sha256(omkar_bytes).hexdigest()!=W_SOURCE_SHA:
        raise RuntimeError("public WeMM source hash drift")
    omkar_nb=json.loads(omkar_bytes)
    k=source_code(omkar_nb)
    if (len(k)!=43 or
        hashlib.sha256(json.dumps(k,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()!=W_DIGEST):
        raise RuntimeError("WeMM original executable source changed")
    champ_meta=json.loads((champ_dir/"kernel-metadata.json").read_text())
    omkar_meta=json.loads((omkar_dir/"kernel-metadata.json").read_text())
    if (champ_meta.get("id")!=CHAMPION or champ_meta.get("is_private") is not True
        or champ_meta.get("competition_sources")!=["name-that-llm"]
        or omkar_meta.get("id")!=W_SOURCE or omkar_meta.get("is_private") is not False
        or len(champ_meta.get("model_sources",[]))!=8
        or len(champ_meta.get("dataset_sources",[]))!=17):
        raise RuntimeError("source asset or provenance mismatch")
    if "FAST_COMMIT = False" not in champ_src or champ_src.count('"gauss": 0.0')!=2:
        raise RuntimeError("champion no-fitting/real-inference controls missing")
    if "_ga.fit(" in champ_src or "NO_NEW_TRAINING: supervised classifier fitting forbidden" not in champ_src:
        raise RuntimeError("champion supervised training guard not intact")
    cells=champ_nb["cells"]
    source_index=next((i for i,c in enumerate(cells)
        if c.get("cell_type")=="code" and re.search(r"(?m)^DATA = next\(p for p in .*os\.path\.exists\(p \+ '/train\.csv'\)\)",wtext(c))),None)
    if source_index is None:raise RuntimeError("champion DATA original assignment anchor not found")
    old_code=wtext(cells[source_index])
    needle=re.compile(r"(?m)^DATA = next\(p for p in .*os\.path\.exists\(p \+ '/train\.csv'\)\)$")
    if len(needle.findall(old_code))!=1:
        raise RuntimeError("champion unique Kaggle DATA assignment changed")
    cells[source_index]["source"]=needle.sub(
        lambda m:m.group(0)+"\n"+PREPARE.strip("\n"),old_code,count=1
    ).splitlines(keepends=True)
    capsule_script=_make_private_capsule(capsule_dir)
    cells.insert(source_index,new_code(capsule_script,70))
    src=omkar_nb["cells"]
    shim=wtext(src[3])
    stage=wtext(src[42])
    worker=wtext(src[51])
    fusion=wtext(src[53])
    config_lines=[s for s in wtext(src[54]).splitlines()
                  if s.startswith("CONFIG['wemm_embedding'] = {")]
    if len(config_lines)!=1:raise RuntimeError("WeMM private calibration literal missing")
    wemm_config=ast.literal_eval(config_lines[0].split(" = ",1)[1])
    if wemm_config.get("share")!=.07 or wemm_config.get("deadline_seconds")!=30240:
        raise RuntimeError("WeMM frozen prefit config drift")
    if any(t in worker for t in (".fit(", ".train(", "train_final", "torch.optim", ".backward(")):
        raise RuntimeError("WeMM worker prohibited fit exists")
    # Add only after exact existing frozen 0.683 inference completed, never mutate its submission output.
    cells.extend([
        new_code("from pathlib import Path\nPath('"+RUNTIME+"').mkdir(parents=True,exist_ok=True)\n",71),
        new_code(shim,72),new_code(stage,73),new_code(worker,74),
        new_code(fusion,75),new_code(WEMM_SCORER % (wemm_config,LABELS12),76)
    ])
    for c in cells:
        if c.get("cell_type")=="code":
            c.setdefault("outputs",[])
            c.setdefault("execution_count",None)
    champ_meta.pop("id_no",None)
    champ_meta.pop("docker_image",None)
    champ_meta.update({
        "id":DEST,"title":DEST.split("/")[-1],"is_private":True,
        "enable_gpu":True,"enable_internet":False,"enable_tpu":False,
        "machine_shape":"NvidiaTeslaT4"
    })
    for kind,additional in (("model_sources",WEMM_SOURCE),
                            ("dataset_sources",FROZEN_HEADS)):
        if additional not in champ_meta[kind]:champ_meta[kind].append(additional)
    if len(champ_meta["model_sources"])!=9 or FROZEN_HEADS not in champ_meta["dataset_sources"]:
        raise RuntimeError("new frozen WeMM mount missing/duplicate")
    outname=champ_meta.get("code_file")
    if not outname or Path(outname).name!=outname or not outname.endswith(".ipynb"):
        raise RuntimeError("invalid Notebook path")
    if output_dir.exists():raise FileExistsError("refuse output replacement")
    output_dir.mkdir(parents=True)
    (output_dir/outname).write_text(json.dumps(champ_nb,ensure_ascii=False))
    (output_dir/"kernel-metadata.json").write_text(json.dumps(champ_meta,indent=2,sort_keys=True))
    result={
        "destination":DEST, "source_official_champion":CHAMPION,
        "champion_code_sha256":champ_sha,
        "frozen_weMM_public_source_sha256":W_SOURCE_SHA,
        "source_rows":ROWS,"observed_true_labels":12,
        "source_complete_prompts":NIMITHA_PROMPTS+TOAN_PROMPTS,
        "same_query_for_both":True,"new_training":False,
        "competition_submission":False,
        "not_independent_full18_validation":True,
        "notebook_sha256":hashlib.sha256((output_dir/outname).read_bytes()).hexdigest()
    }
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--champion",type=Path,required=True)
    p.add_argument("--omkar",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--capsules",type=Path,required=True)
    a=p.parse_args()
    print("NTL_PAIRED84_BUILD="+json.dumps(
        build(a.champion,a.omkar,a.out,a.capsules),sort_keys=True),flush=True)

if __name__=="__main__":
    main()
