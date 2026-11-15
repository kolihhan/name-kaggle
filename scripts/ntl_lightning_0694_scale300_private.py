"""Private Lightning 0.694 V2 300-row *inference timing* pilot.

Only inference; reads 300 text fields, no labels, inside private Kaggle.
The existing V2 trained weights and guards remain untouched.
No leaderboard submission and no accuracy estimates.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

FROM = "kolihhan/lightning-0-694-authentic-no-fit-inference"
TO = "kolihhan/ntl-lightning0694-scale300-nov15"
V2_CODE_SHA = "cae1074e9cea0b3c5f03e221b93e622858c4cbd8183309ba0a9a02e0ee73558c"
DATA_ANCHOR = "DATA = Path('/kaggle/input/competitions/name-that-llm/test.csv')"
CONFIG_ANCHOR = "(RUNTIME/'config.json').write_text(json.dumps(CONFIG),encoding='utf-8')"
MARKER = "NTL0694_SCALE300_INPUT_TEXT_ONLY"
PILOT = r'''
# No label fields are loaded; this is a throughput test, not an evaluation.
import pandas as _ntl_pilot_pd
_ntl_pilot = _ntl_pilot_pd.read_csv(
    '/kaggle/input/competitions/name-that-llm/train.csv',
    usecols=['text'], dtype={'text': str}, nrows=300)
assert len(_ntl_pilot)==300 and _ntl_pilot['text'].notna().all()
_ntl_pilot.insert(0, 'id', ['scale300_%04d'%i for i in range(300)])
_ntl_pilot_file = Path('/kaggle/working/ntl0694_scale300_unlabelled.csv')
_ntl_pilot.to_csv(_ntl_pilot_file,index=False)
DATA = _ntl_pilot_file
print("NTL0694_SCALE300_INPUT_TEXT_ONLY="+str(len(_ntl_pilot)),flush=True)
'''

def source(cell: dict) -> str:
    s=cell.get('source',[])
    return ''.join(s) if isinstance(s,list) else str(s)

def build(previous:Path, output:Path)->dict:
    files=list(previous.glob('*.ipynb'))
    if len(files)!=1:
        raise ValueError('must_pull_one_frozen_owned_V2')
    nb=json.loads(files[0].read_text())
    meta=json.loads((previous/'kernel-metadata.json').read_text())
    if (meta.get('id')!=FROM or meta.get('is_private') is not True or
            meta.get('competition_sources')!=['name-that-llm'] or
            [len(meta.get(k) or []) for k in
             ('dataset_sources','model_sources','kernel_sources')]!=[23,6,9]):
        raise ValueError('owned_V2_manifest_drift')
    cells=[c for c in nb['cells'] if c.get('cell_type')=='code']
    if len(cells)!=50:raise ValueError('unexpected_source_cells')
    original='\n'.join(source(c) for c in cells)
    sha=hashlib.sha256(original.encode()).hexdigest()
    if sha!=V2_CODE_SHA:
        raise ValueError('owned_V2_code_SHA_drift:'+sha)
    for token in ('preview = False  # always genuine model inference',
                  'NO_NEW_TRAINING: Gaussian fit blocked',
                  'NO_NEW_TRAINING: Atlas training blocked'):
        if original.count(token)!=1:
            raise ValueError('missing_no_fit_or_real_path_guard:'+token)
    if 'gauss.fit(' in original:raise ValueError('supervised_gaussian_present')
    edits={'input':0,'config':0}
    for cell in cells:
        text=source(cell)
        if DATA_ANCHOR in text:
            if text.count(DATA_ANCHOR)!=1:raise ValueError('ambiguous_DATA')
            text=text.replace(DATA_ANCHOR,PILOT,1)
            edits['input']+=1
        if CONFIG_ANCHOR in text and 'CONFIG = ' in text:
            if text.count(CONFIG_ANCHOR)!=1:raise ValueError('ambiguous_CONFIG')
            updates=("CONFIG['paths']['test_csv'] = str(DATA)\n"
                     "EXPLICIT_PATHS['paths']['test_csv']['path'] = str(DATA)\n")
            text=text.replace(CONFIG_ANCHOR,updates+CONFIG_ANCHOR,1)
            edits['config']+=1
        cell['source']=text.splitlines(keepends=True)
    if edits!={'input':1,'config':1}:raise ValueError('pilot_route_insertion_missing:'+str(edits))
    full='\n'.join(source(c) for c in cells)
    if (full.count(MARKER)!=1 or DATA_ANCHOR in full or
        'gauss.fit(' in full or
        full.count('NO_NEW_TRAINING: Gaussian fit blocked')!=1 or
        full.count('NO_NEW_TRAINING: Atlas training blocked')!=1):
        raise ValueError('pilot_source_corrupt')
    count=0
    for cell in cells:
        text=source(cell)
        if not text.startswith('%%writefile '):continue
        lines=text.splitlines()
        ast.parse('\n'.join(lines[1:]), filename=lines[0].split(' ',1)[1])
        count+=1
    if count!=37:raise ValueError('expected_37_model_modules')
    meta.pop('id_no',None);meta.pop('docker_image',None)
    meta.update({'id':TO,'title':'ntl-lightning0694-scale300-nov15',
       'is_private':True,'enable_gpu':True,'enable_internet':False,
       'enable_tpu':False,'machine_shape':'NvidiaTeslaT4'})
    name=meta.get('code_file')
    if not name or name!=Path(name).name or not name.endswith('.ipynb'):
        raise ValueError('bad_ipynb_filename')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('refuse_overwrite')
    output.mkdir(parents=True,exist_ok=True)
    data=json.dumps(nb,ensure_ascii=False)
    (output/name).write_text(data)
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True))
    return {
      'source':FROM, 'target':TO, 'text_rows':300,
      'input_origin':'official_train_text_only',
      'model_labels_read':False,
      'fullpath_inference':True,
      'gsq_min_texts':50,'gsq_max_fraction':0.20,
      'source_v2_code_sha':sha,
      'pilot_ipynb_sha256':hashlib.sha256(data.encode()).hexdigest(),
      'generated_modules_compiled':count,
      'retained_mounts':{'datasets':23,'models':6,'notebooks':9},
      'training':False,'official_submission':False,'BA_measured':False,
    }

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print('NTL0694_SCALE300_PILOT_BUILD='+json.dumps(build(a.source,a.out),sort_keys=True),flush=True)
