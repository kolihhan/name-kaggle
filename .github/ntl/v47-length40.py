"""Transport one immutable notebook to a private Kaggle run, then verify its score."""
import base64
import csv
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

COMPETITION = 'name-that-llm'
UPSTREAM = 'lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble'
CONTROL_REF = '58008463'
CONTROL_SCORE = 0.666
UPSTREAM_CODE = 'c50abf03f5d5472dc2404f62d000df659001e2b73def475c17b45d581c4199dc'
CANDIDATE = 'kolihhan/ntl-v47-length40'
CODE_SHA = 'b686a03a1f792ca722cae07a289c44ea2c5314a9eacd627773a5d68c1865d75c'
NOTEBOOK_SHA = '9942051bd6f4ad778e419f0441e9263b960114d888ac021ddbcb462bfd86f63f'
DESCRIPTION = 'Research V47 length40 SVC 61287905 20261107'
DESCRIPTION_PREFIX = 'Research V47 length40 SVC '
CONTEXT = b'ntl-v47-length40-20261107'
SOURCE_KEYS = ('dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources')
PACKET = Path('.github/ntl/v47-length40-payload.json')
ARCHIVED_DATASETS = (
    'dhruvtibarewal/nlm-l24-adapters',
    'lightningv08/ntl-olmo2-native24-final-inference-assets',
    'seyitkaangunes/ntl-extra-api-passages',
    'abdullah1afify/ntl-generations',
    'dophuchung/ntl-proxy-generations',
    'zbigniewkirchner/ntl-qwen3-8b-lora-claude-head',
    'josephadamski91/ntl-atlas-qwen3-8b-v1',
    'lucasboesen/ntl-free-proxy-generations',
    'kirilltym/ntl-spectral-style-heads-paired-4404',
    'dhruvtibarewal/nlm-enc-v5',
    'nishantdahal/ntl-two-stage-lora',
    'nishantdahal/ntl-handwriting-blend',
    'dhruvtibarewal/nlm-l24-adapters-v5',
    'prvsiyan/prvsiyans-extended-name-that-llm-dataset',
    'bbucxi/qwen35-9b-safetensors',
    'pjmathematician/ultrabert-380m-ntl',
)
LATER_DATASETS = {
    'nuridudak/ntl-q14-lora', 'nuridudak/ntl-q14-lora-v4',
    'nuridudak/ntl-svc-prefit-pool-v4n',
}
ARCHIVED_DATASET_VERSIONS = {'dhruvtibarewal/nlm-l24-adapters': '7',
 'lightningv08/ntl-olmo2-native24-final-inference-assets': '18',
 'seyitkaangunes/ntl-extra-api-passages': '2',
 'dophuchung/ntl-proxy-generations': '1',
 'zbigniewkirchner/ntl-qwen3-8b-lora-claude-head': '1',
 'josephadamski91/ntl-atlas-qwen3-8b-v1': '3',
 'kirilltym/ntl-spectral-style-heads-paired-4404': '1',
 'dhruvtibarewal/nlm-enc-v5': '3',
 'nishantdahal/ntl-two-stage-lora': '1',
 'nishantdahal/ntl-handwriting-blend': '1',
 'dhruvtibarewal/nlm-l24-adapters-v5': '1',
 'prvsiyan/prvsiyans-extended-name-that-llm-dataset': '6',
 'bbucxi/qwen35-9b-safetensors': '1',
 'pjmathematician/ultrabert-380m-ntl': '1'}
ARCHIVED_KERNELS = {
    'lucasboesen/name-that-llm-glm4-9b-trunc25',
    'lucasboesen/name-that-llm-extract-encoders2',
    'lucasboesen/name-that-llm-electra-large-weights',
    'lucasboesen/name-that-llm-desklib-weights',
    'lucasboesen/name-that-llm-train-lora24-s7',
    'lucasboesen/name-that-llm-train-tokattn',
    'lucasboesen/name-that-llm-train-tokattn2',
    'vnton37nguyn/ntl-train-lora-qwen25-instyle',
}
ARCHIVED_MODELS = {
    'qwen-lm/qwen-3/Transformers/8b-base/1',
    'qwen-lm/qwen-3/Transformers/14b-base/1',
    'qwen-lm/qwen-3/Transformers/4b-base/1',
    'qwen-lm/qwen2.5/Transformers/7b/1',
    'qwen-lm/qwen2.5/Transformers/14b/1',
}


def encode(data):
    return base64.b64encode(data).decode('ascii')


def decode(text):
    return base64.b64decode(text, validate=True)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def notebook_digest(document):
    cells = [''.join(c['source']) for c in document['cells'] if c['cell_type'] == 'code']
    return sha256('\n'.join(cells).encode())


def wrapping_key():
    token = os.environ['KAGGLE_API_TOKEN'].encode()
    if len(token) < 24:
        raise RuntimeError('Configured Kaggle credential is missing or too short')
    return hashlib.sha256(CONTEXT + b'\0' + token).digest()


def oaep():
    return padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(), label=CONTEXT)


def cli(*args, require_success=True):
    result = subprocess.run(['kaggle', *args], capture_output=True, text=True, timeout=180)
    if require_success and result.returncode:
        raise RuntimeError('KAGGLE_COMMAND_FAILED ' + ' '.join(args[:3])
                           + ' ' + (result.stdout + result.stderr)[-700:])
    return result


def history():
    result = cli('competitions', 'submissions', COMPETITION, '-v', '-q')
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert any(r.get('ref') == CONTROL_REF and float(r.get('publicScore') or 'nan') == CONTROL_SCORE
               for r in rows), 'Authenticated V47 reference missing'
    return rows


def safe_submission_row(row):
    return {k: row.get(k) for k in ('ref', 'status', 'publicScore', 'date')}


def ensure_no_duplicate(rows):
    assert not any(r.get('description', '').startswith(DESCRIPTION_PREFIX) for r in rows), 'Diagnostic already submitted'


def remaining_gpu_hours():
    quota = cli('quota').stdout
    rows = [line.split() for line in quota.splitlines() if line.strip().startswith('GPU ')]
    assert len(rows) == 1 and len(rows[0]) >= 4 and rows[0][2].endswith('h'), 'Unknown Kaggle GPU quota'
    return float(rows[0][2].removesuffix('h'))


def fetch_kernel(ref, path):
    cli('kernels', 'pull', ref, '-p', str(path), '-m')
    md = json.loads((path / 'kernel-metadata.json').read_bytes())
    document = json.loads((path / md['code_file']).read_bytes())
    return md, document


def upstream_at(folder, require_exact_counts=True):
    md, nb = fetch_kernel(f'{UPSTREAM}/47', folder)
    assert md['id'] == UPSTREAM and md['enable_gpu'] is True and md['enable_internet'] is False
    assert notebook_digest(nb) == UPSTREAM_CODE
    assert md['competition_sources'] == [COMPETITION]
    if require_exact_counts:
        datasets = set(md['dataset_sources'])
        archived = set(ARCHIVED_DATASETS)
        assert archived <= datasets <= archived | LATER_DATASETS, 'Archived V47 dataset references changed'
        assert set(md['kernel_sources']) == ARCHIVED_KERNELS
        assert set(md['model_sources']) == ARCHIVED_MODELS
        print('KAGGLE_V47_RESTORED_ARCHIVED_ATTACHMENTS', sorted(datasets - archived), flush=True)
        md['dataset_sources'] = [ref + '/' + ARCHIVED_DATASET_VERSIONS[ref]
                                 if ref in ARCHIVED_DATASET_VERSIONS else ref for ref in ARCHIVED_DATASETS]
    return md, nb


def prepare():
    rows = history()
    ensure_no_duplicate(rows)
    recent = sorted(rows, key=lambda r: int(r['ref']), reverse=True)
    print('NTL_LATEST_OFFICIAL', json.dumps(safe_submission_row(recent[0])), flush=True)
    hours = remaining_gpu_hours()
    print('NTL_GPU_HOURS_REMAINING', hours, flush=True)
    with tempfile.TemporaryDirectory() as directory:
        md, nb = upstream_at(Path(directory), require_exact_counts=False)
    print('NTL_V47_PUBLIC_REFS', json.dumps({k: md[k] for k in SOURCE_KEYS}), flush=True)
    print('NTL_V47_METADATA_HAS_SOURCE_IDS', bool(nb.get('metadata', {}).get('kaggle', {}).get('dataSources')), flush=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    secret = key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                               serialization.NoEncryption())
    public = key.public_key().public_bytes(serialization.Encoding.DER,
                                          serialization.PublicFormat.SubjectPublicKeyInfo)
    nonce = os.urandom(12)
    sealed = nonce + AESGCM(wrapping_key()).encrypt(nonce, secret, CONTEXT)
    # The public key and ciphertext are sufficient for transport. The API token
    # and private key stay inside the authenticated Actions runtime.
    print('NTL_TRANSPORT', json.dumps({'public': encode(public), 'sealed': encode(sealed)}), flush=True)


def load_packet():
    packet = json.loads(PACKET.read_bytes())
    sealed = decode(packet['sealed'])
    private = AESGCM(wrapping_key()).decrypt(sealed[:12], sealed[12:], CONTEXT)
    key = serialization.load_der_private_key(private, password=None)
    message_key = key.decrypt(decode(packet['key']), oaep())
    ciphertext = decode(packet['content'])
    compressed = AESGCM(message_key).decrypt(ciphertext[:12], ciphertext[12:], CONTEXT)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        payload = stream.read(1024 * 1024)
    assert sha256(payload) == NOTEBOOK_SHA, 'Decrypted notebook differs from the frozen bytes'
    assert notebook_digest(json.loads(payload)) == CODE_SHA
    return payload


def kernel_version(output):
    if re.search(r'quota|403|forbidden|push error|permission denied', output, re.I):
        raise RuntimeError('KAGGLE_SAVE_REJECTED ' + output[-950:])
    found = re.findall(r'Kernel version\s+(\d+)\s+successfully pushed\b', output, re.I)
    assert len(found) == 1, 'No unique new saved kernel version: ' + output[-700:]
    return int(found[0])


def notebook_log_text(ref):
    body = cli('kernels', 'logs', ref).stdout
    try:
        records = json.loads(body)
    except json.JSONDecodeError:
        records = []
        for line in body.splitlines():
            try:
                records.append(json.loads(line.strip().strip(',')))
            except json.JSONDecodeError:
                continue
    if isinstance(records, list) and records and isinstance(records[0], dict):
        return ''.join(str(r.get('data', '')) for r in records)
    return body


def poll_kernel(ref):
    previous = None
    deadline = time.monotonic() + 180 * 60
    while time.monotonic() < deadline:
        result = cli('kernels', 'status', ref, require_success=False)
        state = (result.stdout + result.stderr).strip()
        if state != previous:
            print('KAGGLE_PREFLIGHT_STATUS', state[-350:], flush=True)
            previous = state
        if result.returncode == 0 and 'KernelWorkerStatus.COMPLETE' in state:
            return
        if re.search(r'KernelWorkerStatus\.(ERROR|CANCELLED|CANCELED|FAILED)', state):
            logs = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', notebook_log_text(ref))
            for line in logs.splitlines()[-100:]:
                if re.search(r'(V47_FAILED_STAGE|V47_STAGE_TRACE|Cell In|RuntimeError|FileNotFoundError|AssertionError|ImportError|ModuleNotFoundError|CUDA out of memory)', line):
                    print('KAGGLE_PREFLIGHT_ERROR', line[:350], flush=True)
            raise RuntimeError('KAGGLE_PREFLIGHT_FAILED')
        time.sleep(30)
    raise RuntimeError('KAGGLE_PREFLIGHT_NOT_COMPLETE')


def proof_from_logs(ref):
    logs = notebook_log_text(ref)
    proofs = re.findall(r'V47_LENGTH40_REAL_PREFLIGHT_PASS\s+(\{[^\n]+\})', logs)
    assert len(proofs) == 1, 'Real fitted inference proof missing'
    proof = json.loads(proofs[0])
    assert proof['rows'] == 3 and proof['labels'] == 18 and proof['columns'] == ['id', 'model']
    assert proof['treatment_max_drift'] < 1e-5, 'Fixed length repetition check failed'
    print('KAGGLE_REAL_PREFLIGHT_PASS', json.dumps(proof, sort_keys=True), flush=True)


def poll_score(ref):
    previous = None
    deadline = time.monotonic() + 150 * 60
    while time.monotonic() < deadline:
        try:
            matches = [r for r in history() if r['ref'] == ref]
        except (RuntimeError, subprocess.TimeoutExpired):
            time.sleep(45)
            continue
        if matches:
            row = matches[0]
            assert row.get('description') == DESCRIPTION
            state = safe_submission_row(row)
            if state != previous:
                print('KAGGLE_OFFICIAL_SUBMISSION', json.dumps(state, sort_keys=True), flush=True)
                previous = state
            if row.get('publicScore'):
                print('KAGGLE_VERIFIED_PUBLIC_SCORE', row['publicScore'],
                      'DELTA_VS_0666', round(float(row['publicScore']) - CONTROL_SCORE, 6), flush=True)
                return
            if any(term in row['status'].upper() for term in ('COMPLETE', 'ERROR', 'FAIL', 'CANCEL')):
                raise RuntimeError('KAGGLE_TERMINAL_WITHOUT_PUBLIC_SCORE')
        time.sleep(45)
    raise RuntimeError('KAGGLE_ACCEPTED_BUT_PUBLIC_SCORE_PENDING')


def run():
    payload = load_packet()
    initial = history()
    ensure_no_duplicate(initial)
    assert remaining_gpu_hours() >= 4, 'Less than four hours of Kaggle GPU quota'
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        md, _ = upstream_at(root / 'upstream')
        original = {name: md[name] for name in SOURCE_KEYS}
        folder = root / 'private'
        folder.mkdir()
        md.pop('id_no', None)
        md.update(id=CANDIDATE, title='NTL V47 Length40', code_file='solution.ipynb',
                  is_private=True, enable_gpu=True, enable_internet=False, machine_shape='NvidiaTeslaT4')
        (folder / 'solution.ipynb').write_bytes(payload)
        (folder / 'kernel-metadata.json').write_text(json.dumps(md))
        print('KAGGLE_SAVE_START', CANDIDATE, flush=True)
        version = kernel_version(cli('kernels', 'push', '-p', str(folder), '--accelerator',
                                     'NvidiaTeslaT4', '--timeout', '32400').stdout)
        ref = f'{CANDIDATE}/{version}'
        print('KAGGLE_SAVED_EXACT_VERSION', ref, flush=True)
        saved_md, saved = fetch_kernel(ref, root / 'saved')
        assert saved_md['id'] == CANDIDATE and saved_md['is_private'] is True
        assert saved_md['enable_internet'] is False and saved_md['enable_gpu'] is True
        assert notebook_digest(saved) == CODE_SHA
        for key in SOURCE_KEYS:
            requested = original[key]
            actual = saved_md[key]
            if key == 'dataset_sources':
                requested = [v.rsplit('/', 1)[0] if v.rsplit('/', 1)[-1].isdigit() else v for v in requested]
                actual = [v.rsplit('/', 1)[0] if v.rsplit('/', 1)[-1].isdigit() else v for v in actual]
            assert sorted(actual) == sorted(requested), f'Saved attachment names differ for {key}'
        ids = json.loads(payload)['metadata']['kaggle']['dataSources']
        observed_ids = saved.get('metadata', {}).get('kaggle', {}).get('dataSources')
        print('KAGGLE_SAVED_SOURCE_IDS_MATCH', observed_ids == ids, flush=True)
        poll_kernel(ref)
        proof_from_logs(ref)
        assert observed_ids == ids, 'Exact Kaggle source IDs unavailable or drifted'
        current = history()
        ensure_no_duplicate(current)
        assert {r['ref'] for r in current} == {r['ref'] for r in initial}, 'Another submission was created during preflight'
        print('KAGGLE_SUBMIT_ONCE', ref, flush=True)
        reply = cli('competitions', 'submit', COMPETITION, '-k', CANDIDATE, '-v', str(version),
                    '-f', 'submission.parquet', '-m', DESCRIPTION)
        refs = re.findall(r'Submission ref:\s*(\d+)', reply.stdout + reply.stderr, re.I)
        assert len(refs) == 1, 'Submit response lacks a unique official ref. Check My Submissions before retrying.'
        print('KAGGLE_OFFICIAL_ACCEPTED_REF', refs[0], flush=True)
        poll_score(refs[0])


if __name__ == '__main__':
    assert not Path('data/train.csv').exists() and not Path('data/test.csv').exists()
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
