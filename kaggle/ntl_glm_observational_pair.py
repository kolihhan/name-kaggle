"""Keep labels separate from inference and report only observable GLM fixes."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd

DATA_ROOT = Path('/kaggle/input')
WORK = Path('/kaggle/temp/ntl-glm-ratio')
OUT = Path('/kaggle/working')
CORPUS_SHA = 'bce34670f4d1170375e7baf85eec5577af4282d5f0ca6cf1573ce46e81c5a45c'
BINARY_SHA = '9c39e3afc77c64c1930f438fef50f297a6a8a649252ab4af9f618804be84cc8c'
PAYLOAD_SHA = 'b0bb99ed84e417a8249600fb28f1f69c21641659510bc7ede15b3a73a05a785d'
RESULT = 'glm-ratio-summary.json'
FULL, FLASH = 'glm-5.3', 'glm-5.3-flash'
FAMILY = (FULL, FLASH)
TOPICS = {'short': (98, 25, 65, 83, 16, 61, 68), 'median': (77, 54, 99, 37), 'long': (72, 63, 87)}
STRICT = {'short': (20, 23, 78), 'median': (97,), 'long': (28,)}
PREFIX = '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|><|assistant|><think></think>'


def _digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def _norm(text):
    return hashlib.sha256(' '.join(text.split()).casefold().encode()).hexdigest()


def prepare(data):
    train = Path(data) / 'train.csv'
    official = pd.read_csv(train, usecols=['model', 'text'])
    labels = set(official.model)
    assert len(labels) == 18
    [source] = [path for path in DATA_ROOT.rglob('generations.csv')
                if 'ntl-topic-synthetic-18llms' in path.parts and _digest(path) == CORPUS_SHA]
    external = pd.read_csv(source)
    blocks = []
    for name, selection in (('panel', TOPICS), ('strict', STRICT)):
        for version, topics in selection.items():
            for topic in topics:
                b = external.loc[(external.version == version) & (external.topic_id == topic)].copy()
                expected = labels
                if name == 'strict':
                    expected = set(FAMILY)
                    b = b.loc[b.model.isin(FAMILY)].copy()
                    assert b.instruction.nunique() == b.target_words.nunique() == 1
                assert len(b) == len(expected) and set(b.model) == expected
                b['panel'] = name
                blocks.append(b.sort_values('model', kind='stable'))
    combined = pd.concat(blocks, ignore_index=True)
    assert len(combined) == 262 and combined.text.notna().all()
    assert combined.text.map(_norm).is_unique
    assert not set(combined.text.map(_norm)) & set(official.text.dropna().map(_norm))
    combined.insert(0, 'id', [f'glm-ratio-{i:03}' for i in range(len(combined))])
    panel = combined.loc[combined.panel == 'panel']
    assert len(panel) == 252 and (panel.groupby('model').size() == 14).all()
    private_data = WORK / 'data'
    private_data.mkdir(parents=True, exist_ok=False)
    (private_data / 'train.csv').symlink_to(train.resolve())
    panel[['id', 'text']].to_csv(private_data / 'test.csv', index=False)
    combined[['id', 'text']].to_csv(WORK / 'ratio-input.csv', index=False)
    combined[['id', 'model', 'panel', 'topic_id', 'version']].to_csv(WORK / 'truth.csv', index=False)
    print('NTL_GLM_INPUT_READY rows=252 labels=18 strict_pairs=5', flush=True)
    return private_data


def scores(payload):
    import torch

    if torch.cuda.device_count() != 2:
        raise RuntimeError('frozen likelihood needs two visible T4 GPUs')
    raw = base64.b64decode(payload, validate=True)
    assert hashlib.sha256(raw).hexdigest() == PAYLOAD_SHA
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as bundle:
        for name in ('run_pipeline.py', 'segio.py', 'features.py'):
            f = bundle.getmember(name)
            assert f.isfile()
            code = bundle.extractfile(f).read()
            compile(code, name, 'exec')
            (WORK / name).write_bytes(code)
    [binary] = [path for path in DATA_ROOT.rglob('ntl-segrun')
                if path.parent.name == 'ntl-bin' and _digest(path) == BINARY_SHA
                and (path.parent / 'lib/libcublasLt.so.13').is_file()]
    shutil.copyfile(binary, WORK / 'ntl-segrun').chmod(0o755)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(binary.parent / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    reports, times = {}, {}
    for tag, pattern, count in [('flash', 'flash_iq3xxs-seg*-of-06-*.gguf', 6),
                                ('glm53', 'glm53_iq3xxs-seg*-of-19-*.gguf', 19)]:
        segments = WORK / tag
        segments.mkdir()
        matches = {p.name: p for p in sorted(DATA_ROOT.rglob(pattern), reverse=True)}
        assert len(matches) == count
        for name, path in matches.items():
            (segments / name).symlink_to(path)
        start = time.monotonic()
        cmd = [sys.executable, str(WORK / 'run_pipeline.py'), '--bin', str(WORK),
               '--segs', str(segments / '*.gguf'), '--csv', str(WORK / 'ratio-input.csv'),
               '--work', str(WORK / 'state'), '--out', str(WORK), '--tag', tag,
               '--max-text-tokens', '160', '--seq-batch', '8', '--prefix', PREFIX,
               '--budget-s', '6000']
        with (WORK / f'{tag}-pipeline.log').open('w') as log:
            completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=6060)
        if completed.returncode:
            raise RuntimeError(f'{tag} frozen likelihood stage exit code {completed.returncode}')
        report = pd.read_csv(WORK / f'{tag}_features.csv', dtype={'id': str}).set_index('id')
        assert report.index.is_unique and np.isfinite(report.lp_mean).all()
        reports[tag] = report[['lp_mean', 'n_tok']]
        times[tag] = round(time.monotonic() - start, 1)
        print(f'NTL_GLM_{tag.upper()}_COMPLETE rows={len(report)} seconds={times[tag]}', flush=True)
    flash, glm53 = reports['flash'], reports['glm53']
    assert flash.index.equals(glm53.index) and flash.n_tok.equals(glm53.n_tok) and flash.n_tok.gt(0).all()
    return glm53.lp_mean - flash.lp_mean, times


def outcome(panel):
    original = panel.model == panel.champion
    treated = panel.model == panel.routed
    return {'rows': len(panel), 'champion_correct': int(original.sum()),
            'routed_correct': int(treated.sum()), 'rescues': int((~original & treated).sum()),
            'regressions': int((original & ~treated).sum()),
            'changed': int((panel.champion != panel.routed).sum()),
            'champion_family': int(panel.champion.isin(FAMILY).sum())}


def ranking(likely):
    paired = likely.pivot(index=['topic_id', 'version'], columns='model', values='ratio')
    assert not paired[list(FAMILY)].isna().any().any()
    return {'pairs': len(paired), 'correct_order': int((paired[FULL] > paired[FLASH]).sum()),
            'ties': int((paired[FULL] == paired[FLASH]).sum()),
            'full_positive': int((paired[FULL] > 0).sum()),
            'flash_negative': int((paired[FLASH] < 0).sum())}


def compare(data, payload):
    champion = pd.read_parquet(OUT / 'submission.parquet')
    ids = pd.read_csv(Path(data) / 'test.csv', usecols=['id']).id
    truth = pd.read_csv(WORK / 'truth.csv').set_index('id')
    assert list(champion) == ['id', 'model'] and champion.id.is_unique
    assert set(champion.id) == set(ids) and set(champion.model) <= set(truth.model)

    ratio, times = scores(payload)
    assert set(ratio.index) == set(truth.index)
    truth['ratio'] = ratio
    panel = truth.loc[truth.panel == 'panel'].join(champion.set_index('id').rename(columns={'model': 'champion'}))
    assert panel.champion.notna().all() and len(panel) == 252
    take = panel.champion.isin(FAMILY) & panel.ratio.ne(0)
    panel['routed'] = panel.champion.copy()
    panel.loc[take, 'routed'] = np.where(panel.loc[take, 'ratio'] > 0, FULL, FLASH)
    by_label = {label: outcome(g) for label, g in panel.groupby('model')}
    report = {
        'scope': 'OBSERVATIONAL_DEVELOPMENT_ONLY', 'champion_ref': '58295802',
        'official_submission': False, 'new_training': False, 'independent_holdout': False,
        'overall': outcome(panel), 'by_label': by_label, 'runtime_seconds': times,
        'champion_panel_ba': sum(x['champion_correct'] / x['rows'] for x in by_label.values()) / 18,
        'routed_panel_ba': sum(x['routed_correct'] / x['rows'] for x in by_label.values()) / 18,
        'observational_glm_ranking': ranking(panel.loc[panel.model.isin(FAMILY)]),
        'strict_requested_prompt_glm_ranking': ranking(truth.loc[truth.panel == 'strict']),
        'by_length': {length: outcome(g) for length, g in panel.groupby('version')},
    }
    (OUT / RESULT).write_text(json.dumps(report, indent=2, sort_keys=True))
