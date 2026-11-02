import json

import pytest

from scripts.two_stage_lora24_direct_preflight import (
    KERNEL_REF,
    classify_kernel_status,
    validate_submission_artifacts,
)


def test_classify_kernel_status_handles_terminal_and_running_states():
    assert classify_kernel_status('KernelWorkerStatus.COMPLETE') == 'complete'
    assert classify_kernel_status('status "KernelWorkerStatus.RUNNING"') == 'running'
    assert classify_kernel_status('KernelWorkerStatus.QUEUED') == 'running'
    assert classify_kernel_status('KernelWorkerStatus.ERROR') == 'failed'
    assert classify_kernel_status('cancelled by user') == 'failed'
    assert classify_kernel_status('something unexpected') == 'unknown'


def test_validate_submission_artifacts_accepts_exact_schema_and_lineage(tmp_path, monkeypatch):
    import pandas as pd

    out = tmp_path / 'out'
    out.mkdir()
    (out / 'submission.parquet').write_bytes(b'placeholder')
    monkeypatch.setattr(pd, 'read_parquet', lambda path: pd.DataFrame({'id': ['a', 'b'], 'model': ['m0', 'm1']}))
    lineage = {
        'candidate': 'two-stage-lora24-v1',
        'primary_head': 'plain',
        'centered_head_used_for_submission': False,
        'competition_test_rows_used_for_training': 0,
        'competition_test_rows_used_for_model_selection': 0,
        'public_lb_tuning': False,
        'test_rows': 2,
        'submission_rows': 2,
        'submission_columns': ['id', 'model'],
    }
    (out / 'submission_lineage.json').write_text(json.dumps(lineage), encoding='utf-8')

    summary = validate_submission_artifacts(out)
    assert summary['rows'] == 2
    assert summary['candidate'] == 'two-stage-lora24-v1'
    assert summary['primary_head'] == 'plain'
    assert summary['submission_authorized'] is False


def test_validate_submission_artifacts_rejects_contaminated_lineage(tmp_path, monkeypatch):
    import pandas as pd

    out = tmp_path / 'out'
    out.mkdir()
    (out / 'submission.parquet').write_bytes(b'placeholder')
    monkeypatch.setattr(pd, 'read_parquet', lambda path: pd.DataFrame({'id': ['a'], 'model': ['m0']}))
    lineage = {
        'candidate': 'two-stage-lora24-v1',
        'primary_head': 'plain',
        'centered_head_used_for_submission': False,
        'competition_test_rows_used_for_training': 1,
        'competition_test_rows_used_for_model_selection': 0,
        'public_lb_tuning': False,
        'test_rows': 1,
        'submission_rows': 1,
        'submission_columns': ['id', 'model'],
    }
    (out / 'submission_lineage.json').write_text(json.dumps(lineage), encoding='utf-8')

    with pytest.raises(RuntimeError, match='test rows used for training'):
        validate_submission_artifacts(out)


def test_kernel_ref_is_fixed_preflight_identity():
    assert KERNEL_REF == 'kolihhan/name-that-llm-two-stage-lora24-v1-submission-preflight'
