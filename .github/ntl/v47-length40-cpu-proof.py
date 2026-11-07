"""Exercise the exact saved SVC treatment with its original fitted state."""
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import types
import zipfile

import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

spec = importlib.util.spec_from_file_location('candidate_transport', '.github/ntl/v47-length40.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
notebook = json.loads(transport.load_packet())
module = types.ModuleType('ntl660_readout')
sys.modules[module.__name__] = module
code = ''.join(notebook['cells'][10]['source']).partition('\n')[2]
exec(compile(code, 'frozen-candidate-ntl660_readout.py', 'exec'), module.__dict__)
api = KaggleApi()
api.authenticate()
dataset = 'lightningv08/ntl-olmo2-native24-final-inference-assets/18'
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    api.dataset_download_file(dataset, 'ntl666-cache-parts.json', path=root, quiet=True, unzip=True)
    manifest = json.loads((root/'ntl666-cache-parts.json').read_text())
    assert manifest['format'] == 'ntl666-cache-parts-v1'
    assert manifest['archive_bytes'] < 2 * 1024**3
    assert [p['name'] for p in manifest['parts']] == [f'ntl666-prefit.part{i:03d}.bin' for i in range(len(manifest['parts']))]
    archive_path = root/'state.zip'
    print('SVC_REAL_WEIGHT_PARTS', len(manifest['parts']), manifest['archive_bytes'], flush=True)
    with archive_path.open('wb') as joined:
        for part in manifest['parts']:
            api.dataset_download_file(dataset, part['name'], path=root, quiet=True, unzip=True)
            file = root/part['name']
            assert file.stat().st_size == part['bytes']
            with file.open('rb') as stream:
                shutil.copyfileobj(stream, joined, length=16*1024**2)
            file.unlink()
    assert archive_path.stat().st_size == manifest['archive_bytes']
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read('ntl666-prefit/COMPLETE') == b'ntl666-v14-data-numeric-v1\n'
        provenance = json.loads(archive.read('ntl666-prefit/provenance.json'))
        assert provenance['status'] == 'COMPLETE' and provenance['pool_rows'] == 190588 and provenance['dataset_version'] == 13
        for entry in archive.infolist():
            if not entry.filename.startswith('ntl666-prefit/core/'):
                continue
            relative = PurePosixPath(entry.filename)
            assert not relative.is_absolute() and '..' not in relative.parts and chr(92) not in entry.filename
            if not entry.is_dir():
                file = root/relative
                file.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as stream, file.open('wb') as out:
                    shutil.copyfileobj(stream,out,length=16*1024**2)
    fitted = module.Readout(root/'ntl666-prefit/core/svc4')
    words = ('A careful researcher compares evidence across topics and records every result clearly so another '
             'reader can repeat each experiment with the same fixed method and decision rule before any outcome '
             'is observed from the collection of ordinary short passages').split()
    assert len(words) >= 40
    block = ' '.join(words[:40])
    copies = [' '.join([block]*n) for n in (1,2,4)]
    baseline, labels = fitted.scores(copies)
    treatment, treated_labels = module.fixed_window_scores(fitted, copies)
    assert np.array_equal(labels, treated_labels) and labels.tolist() == list(range(18))
    assert baseline.shape == treatment.shape == (3,18)
    assert np.isfinite(baseline).all() and np.isfinite(treatment).all()
    np.testing.assert_allclose(treatment, np.broadcast_to(treatment[:1],treatment.shape),rtol=1e-6,atol=1e-7)
    short = ['  a short\nstatement. ']
    votes = module.core_votes(root/'ntl666-prefit',short)
    for name, weight in [('svc4',12.),('tfidf',.25),('distort',.10)]:
        raw, classes = module.Readout(root/'ntl666-prefit/core'/name).scores(short)
        np.testing.assert_allclose(votes[name],weight*module.log_softmax(raw),rtol=1e-7,atol=1e-8)
    print('V47_LENGTH40_REAL_SVC_PASS', json.dumps({
        'code_sha256':transport.CODE_SHA, 'checkpoint_dataset_version':18, 'checkpoint_version_id':21594398,
        'fitted_pool_rows':190588, 'classes':18, 'repetition_multipliers':[1,2,4],
        'baseline_max_drift':float(np.max(np.abs(baseline-baseline[:1]))),
        'treatment_max_drift':float(np.max(np.abs(treatment-treatment[:1]))),
        'short_passage_votes_unchanged':True
    },sort_keys=True),flush=True)
