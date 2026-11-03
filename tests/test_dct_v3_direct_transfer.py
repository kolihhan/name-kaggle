import csv
import json
from pathlib import Path

import pytest

from scripts.dct_v3_direct_transfer import (
    KERNEL_REF,
    classify_kernel_status,
    push_kernel_once,
    validate_transfer_artifacts,
)


def test_classify_kernel_status_handles_terminal_and_running_states():
    assert classify_kernel_status('KernelWorkerStatus.COMPLETE') == 'complete'
    assert classify_kernel_status('KernelWorkerStatus.RUNNING') == 'running'
    assert classify_kernel_status('KernelWorkerStatus.QUEUED') == 'running'
    assert classify_kernel_status('KernelWorkerStatus.ERROR') == 'failed'
    assert classify_kernel_status('cancelled by user') == 'failed'
    assert classify_kernel_status('something unexpected') == 'unknown'


def test_push_kernel_once_treats_quota_message_as_failure_even_with_zero_exit(tmp_path, monkeypatch):
    class Result:
        returncode = 0
        stdout = 'Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.\n'
        stderr = ''

    monkeypatch.setattr('scripts.dct_v3_direct_transfer._run', lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match='Maximum weekly GPU quota'):
        push_kernel_once(tmp_path)


def _write_valid_output(out: Path):
    out.mkdir()
    (out / 'v3_stage1_checkpoint').mkdir()
    (out / 'v1_worker.json').write_text('{}', encoding='utf-8')
    (out / 'v3_worker.json').write_text('{}', encoding='utf-8')
    result = {
        'candidate': 'dct-temporal-fingerprint-v3',
        'scope': 'paired_external_source_transfer_no_official_rows_no_public_lb',
        'pool_sha256': '37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec',
        'holdout_sources': ['alperen', 'hailey', 'willrice'],
        'official_rows_scored': 0,
        'competition_test_rows_used': 0,
        'public_lb_tuning': False,
        'v3_stage1_checkpoint_preserved': True,
        'decision': 'pass',
        'transfer_gate': {
            'rows': 14506,
            'decision': {
                'status': 'pass',
                'delta': 0.01,
                'bootstrap_ci_lower': 0.001,
                'bootstrap_ci_upper': 0.02,
            },
        },
    }
    (out / 'paired_transfer_result.json').write_text(json.dumps(result), encoding='utf-8')
    with (out / 'paired_transfer_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['source', 'prompt_id', 'model', 'v1_prediction', 'v3_prediction'])
        for i in range(14506):
            writer.writerow(['alperen', f'p{i}', 'm', 'm', 'm'])


def test_validate_transfer_artifacts_accepts_exact_frozen_lineage(tmp_path):
    out = tmp_path / 'out'
    _write_valid_output(out)
    summary = validate_transfer_artifacts(out)
    assert summary['candidate'] == 'dct-temporal-fingerprint-v3'
    assert summary['decision'] == 'pass'
    assert summary['oof_gate_unlocked'] is True
    assert summary['submission_authorized'] is False
    assert summary['rows'] == 14506


def test_validate_transfer_artifacts_rejects_public_lb_contamination(tmp_path):
    out = tmp_path / 'out'
    _write_valid_output(out)
    path = out / 'paired_transfer_result.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    result['public_lb_tuning'] = True
    path.write_text(json.dumps(result), encoding='utf-8')
    with pytest.raises(RuntimeError, match='public_lb_tuning'):
        validate_transfer_artifacts(out)


def test_kernel_ref_is_frozen_identity():
    assert KERNEL_REF == 'kolihhan/ntl-v3-dct-transfer'
