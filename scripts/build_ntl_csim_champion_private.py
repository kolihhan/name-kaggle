"""Construct one private v36-lineage diagnostic notebook, no official submission."""
import hashlib,json,re
from pathlib import Path
from scripts.build_ntl_v36_trace_preflight import build, code, source
from scripts.ntl_csim_champion_panel import PUBLISHER_SHA,OFFICIAL_SHA

POOL_SHA="fe6ece20e6f9f2fe5fb3c47f2aaa9749394c1056bc688a8eab865e0dc2bb56df"
COHORT_SHA="a59b8150bb1de990209408d7ce42c4b3a115c6eaee6b7cdfe0d14d0574599a74"
DEST="kolihhan/ntl-v36-csim-stage-diagnostic-500-20261128"
PUBLISHER="markbarney/ntl-matched-passages-five-models-1k"
OLD_PANEL="nischaydnk/ntl-topic-synthetic-18llms"
DATA_PATTERN=re.compile(r"(?m)^DATA = next\(p for p in .*os\.path\.exists\(p \+ '/train\.csv'\)\)$")

def construct(raw,md,inventory,trace_src,panel_src):
    mounts=inventory["mounts"]
    for k in ("model_sources","kernel_sources","competition_sources","dataset_sources"):
        expected=set(mounts[k])
        if k=="dataset_sources":expected.add(OLD_PANEL)
        if set(md.get(k,[]))!=expected:raise ValueError("prior private lineage mount drift "+k)
    if not md.get("is_private") or not md.get("enable_gpu"):raise ValueError("unexpected prior private metadata")
    newmd=dict(md);newmd.pop("id_no",None)
    newmd.update(id=DEST,title=DEST.split("/")[1],is_private=True,enable_gpu=True,enable_internet=False,
                 dataset_sources=list(mounts["dataset_sources"])+[PUBLISHER],
                 code_file="diagnostic.ipynb")
    asset_sha=hashlib.sha256(json.dumps({k:newmd[k] for k in mounts},sort_keys=True,separators=(",",":")).encode()).hexdigest()
    nb,report=build(raw,trace_src)
    cells=nb["cells"]
    pre=code("import os as _cs_os,time as _cs_time\n"
             "_cs_os.environ['NTL_PRIVATE_TRACE_AUTHORIZED']='1'\n"
             "_cs_started=_cs_time.monotonic()\n")
    helper=code(
        "from pathlib import Path as _cs_Path\n"
        "import sys as _cs_sys,json as _cs_json\n"
        "_cs_Path('/kaggle/temp').mkdir(parents=True,exist_ok=True)\n"
        "_cs_Path('/kaggle/temp/ntl_csim_champion_panel.py').write_text("+repr(panel_src)+")\n"
        "_cs_sys.path.insert(0,'/kaggle/temp')\n"
        "import ntl_csim_champion_panel as _cs_panel\n"
        "def _cs_prepare(original):\n"
        "    root=_cs_Path('/kaggle/input')\n"
        "    def locate(name,sha):\n"
        "        paths=[p for p in root.rglob(name) if _cs_panel.sha(p)==sha]\n"
        "        assert len(paths)==1, 'pinned mount absent or ambiguous: '+name\n"
        "        return paths[0]\n"
        "    train=_cs_Path(original)/'train.csv'\n"
        "    pub=locate('passages.csv',"+repr(PUBLISHER_SHA)+")\n"
        "    pool=locate('ntl_open_pool.parquet',"+repr(POOL_SHA)+")\n"
        "    pair, receipt=_cs_panel.freeze(pub,train,pool)\n"
        "    assert receipt['cohort_sha']=="+repr(COHORT_SHA)+" and receipt['rows']==500\n"
        "    d=_cs_Path('/kaggle/temp/csim-private-data');d.mkdir(exist_ok=False)\n"
        "    test,truth=pair\n"
        "    (d/'train.csv').symlink_to(train.resolve())\n"
        "    test.to_csv(d/'test.csv',index=False,lineterminator='\\n')\n"
        "    assert _cs_panel.sha(d/'test.csv')==receipt['cohort_sha']\n"
        "    truth.to_csv('/kaggle/temp/csim-private-truth.csv',index=False)\n"
        "    _cs_Path('/kaggle/temp/ntl-private-trace-contract.json').write_text(_cs_json.dumps({'cohort_sha':receipt['cohort_sha'],'asset_manifest_sha':"+repr(asset_sha)+"}))\n"
        "    print('CSIM_PRIVATE_INPUT_READY',len(test),flush=True)\n"
        "    return str(d)\n")
    # Locate original DATA discovery, proven first-use order by prior successful replay.
    matches=[i for i,c in enumerate(cells) if c.get("cell_type")=="code" and DATA_PATTERN.search(source(c))]
    if len(matches)!=1:raise ValueError("DATA discovery anchor not unique")
    i=matches[0];s=source(cells[i])
    changed,n=DATA_PATTERN.subn(lambda m:m.group(0)+"\nDATA = _cs_prepare(DATA)\nimport os\nos.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'",s)
    if n!=1 or changed.index("DATA = _cs_prepare(DATA)")>changed.index("FAST_COMMIT ="):raise ValueError("DATA initialization order")
    compile(changed,"<private data cell>","exec")
    cells[i]=dict(cells[i],source=changed.splitlines(True))
    # Post-scoring diagnostic: no publisher truth is consumed by inference.
    final=code("""
import numpy as _cs_np,pandas as _cs_pd
from scipy.special import softmax as _cs_softmax
from scipy.optimize import linear_sum_assignment as _cs_assign
from sklearn.feature_extraction.text import TfidfVectorizer as _cs_Tfidf
assert not FAST_COMMIT
assert _nt_trace.root.exists()
_cs_truth=_cs_pd.read_csv('/kaggle/temp/csim-private-truth.csv',dtype=str).set_index('id')
_cs_test=_cs_pd.read_csv(_cs_Path(DATA)/'test.csv',dtype=str).set_index('id')
_cs_y=_cs_truth.loc[_nt_ids,'model'].to_numpy(str)
_cs_groups=_cs_truth.loc[_nt_ids,'prompt_id'].to_numpy(str)
assert len(_cs_y)==500 and len(set(_cs_y))==5
def _cs_counts(a,b):
    a=_cs_np.asarray(a);b=_cs_np.asarray(b)
    old=a==_cs_y;new=b==_cs_y
    delta=(new.astype(float)-old.astype(float))
    g,inv=_cs_np.unique(_cs_groups,return_inverse=True)
    assert len(g)==100
    sums=_cs_np.bincount(inv,weights=delta)/5
    rng=_cs_np.random.default_rng(20261128)
    ci=_cs_np.quantile(sums[rng.integers(0,len(sums),size=(2000,len(sums)))].mean(1),[.025,.975])
    return {'base_BA5':float(old.mean()),'BA5':float(new.mean()),'delta':float(delta.mean()),
        'CI95':ci.tolist(),'rescues':int((~old&new).sum()),'regressions':int((old&~new).sum()),
        'changed':int((a!=b).sum()),'recall':{m:float(new[_cs_y==m].mean()) for m in sorted(set(_cs_y))}}
def _cs_scores(name):
    with _cs_np.load(_nt_trace.root/(name+'.npz'),allow_pickle=False) as z:
        assert _cs_np.array_equal(z['ids'],_nt_ids) and _cs_np.array_equal(z['labels'],_nt_labels)
        return z['logits']
# Label-free, reciprocal text similarity pairs; freeze .15 threshold and .5 mix from prior diagnostic.
x=_cs_Tfidf(analyzer='word',ngram_range=(1,2),stop_words='english',min_df=1,sublinear_tf=True,max_features=50000).fit_transform(_cs_test.loc[_nt_ids,'text'])
s=(x@x.T).toarray();_cs_np.fill_diagonal(s,-1)
j=s.argmax(1)
pairs=[(i,int(k)) for i,k in enumerate(j) if i<k and j[k]==i and s[i,k]>=.15]
def _cs_joint(z):
    p=_cs_softmax(z,axis=1);base=p.argmax(1);hard=base.copy();soft=base.copy()
    for i,j in pairs:
        a,b=p[i],p[j];den=1-float(a@b)
        if den>1e-12:
            soft[i]=_cs_np.argmax(.5*a+.5*a*(1-b)/den)
            soft[j]=_cs_np.argmax(.5*b+.5*b*(1-a)/den)
        r,c=_cs_assign(-_cs_np.log(_cs_np.maximum(p[[i,j]],1e-300)))
        hard[_cs_np.array([i,j])[r]]=c
    return {'hard':_cs_counts(_nt_labels[base],_nt_labels[hard]),'soft':_cs_counts(_nt_labels[base],_nt_labels[soft])}
native=_cs_scores('native');native_pred=_nt_labels[native.argmax(1)]
with _cs_np.load(_nt_trace.root/'final.npz',allow_pickle=False) as z:
    assert _cs_np.array_equal(z['ids'],_nt_ids)
    final_pred=z['predictions']
# Trace does not modify final predictions; independent repeated scorer parity remains unverified.
assert _cs_pd.read_parquet(OUT).set_index('id').loc[_nt_ids,'model'].tolist()==final_pred.tolist()
results={'rows':500,'truth_classes':5,'output_classes':18,'native_to_final':_cs_counts(native_pred,final_pred),
    'native_joint':_cs_joint(native),'apex_refined_joint':_cs_joint(_cs_scores('apex_refined')),
    'pairs':len(pairs),'same_prompt_pairs':sum(bool(_cs_groups[i]==_cs_groups[j]) for i,j in pairs),
    'same_label_pairs':sum(bool(_cs_y[i]==_cs_y[j]) for i,j in pairs),
    'v1_status':_nt_v1_status,'elapsed_seconds':_cs_time.monotonic()-_cs_started,
    'source_independence':False,'historical_asset_versions_verified':False,'actual_inference':True,
    'no_new_neural_training':True,'official_submission':False,'promotion':False,
    'independent_replay_parity':False,'per_expert_trace_complete':False,'token_trace_complete':False}
if _nt_v1_status=='applied':
    post=_cs_scores('post_v1');p=_nt_labels[post.argmax(1)]
    results['native_to_post_v1']=_cs_counts(native_pred,p)
    results['post_v1_to_final']=_cs_counts(p,final_pred)
    results['post_v1_joint']=_cs_joint(post)
_cs_Path('/kaggle/working/csim_champion_aggregate.json').write_text(_cs_json.dumps(results,sort_keys=True))
print('CSIM_CHAMPION_AGGREGATE='+_cs_json.dumps(results,sort_keys=True),flush=True)
""")
    cells.insert(i,helper);cells.insert(0,pre);cells.append(final)
    # Strip stale output from the donor, including any displayed data.
    for c in cells:
        if c.get("cell_type")=="code":c.update(outputs=[],execution_count=None)
    report.update({"destination":DEST,"rows":500,"cohort_sha":COHORT_SHA,"asset_manifest_sha":asset_sha,
        "historical_mount_names_restored":True,"historical_asset_versions_verified":False,
        "private":True,"internet":False,"new_neural_training":False,"official_submission":False,
        "max_runtime_hours":4,"total_cells":len(cells)})
    return nb,newmd,report
