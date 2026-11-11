import hashlib
import importlib.util
from pathlib import Path
import pytest

p = Path(__file__).resolve().parents[1] / 'scripts' / 'stablepair_protocol_gate.py'
spec = importlib.util.spec_from_file_location('stablepair_protocol_gate', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def bank(source, origin, n=105, second='gpt-5.6-sol', identical=False, exposure='none'):
    rows = []
    for k in range(n):
        prompt = h(f'{"shared" if identical else source}-{k}')
        for label in ('gpt-5.6-luna', second):
            rows.append(dict(source_lineage=source, prompt_origin=origin,
                prompt_hash=prompt, model=label, request_hash=h(f'req-{k}'),
                settings_hash=h('temp-0.7'), target_words='120',
                served_model=label, served_revision='rev-20261112',
                rights_verified='true', prefit_exposure=exposure))
    return rows


def test_two_provenance_separated_sources_are_capacity_only():
    out = m.audit(bank('s1', 'o1') + bank('s2', 'o2'))
    p = out['pair_results']['gpt56:gpt-5.6-luna|gpt-5.6-sol']
    assert p['independent_source_pairs'] == 1
    assert p['status'] == 'CAPACITY_ONLY_NOT_VALIDATED'
    assert out['independently_verified_provenance'] is False


def test_same_upstream_prompt_origin_fails():
    out = m.audit(bank('s1', 'common') + bank('s2', 'common'))
    assert not out['any_pair_capacity_eligible']


def test_same_prompt_hashes_across_source_fails():
    out = m.audit(bank('s1', 'o1', identical=True) + bank('s2', 'o2', identical=True))
    assert not out['any_pair_capacity_eligible']


def test_mismatched_requested_length_fails():
    rows = bank('s1', 'o1') + bank('s2', 'o2')
    for row in rows:
        if row['model'] == 'gpt-5.6-sol':
            row['target_words'] = '80'
    assert not m.audit(rows)['any_pair_capacity_eligible']


def test_unverified_exposure_and_invalid_hash_fail_closed():
    rows = bank('s1', 'o1', exposure='unknown') + bank('s2', 'o2')
    rows[0]['prompt_hash'] = 'not-a-real-digest'
    out = m.audit(rows)
    assert not out['any_pair_capacity_eligible']
    assert out['rows_excluded']['unverified_rights_or_prefit_exposure'] > 0
    assert out['rows_excluded']['invalid_hash'] == 1


def test_duplicates_do_not_inflate_prompt_count():
    rows = bank('s1', 'o1', n=2) + bank('s2', 'o2', n=2)
    out = m.audit(rows * 150, min_pairs=100)
    assert not out['any_pair_capacity_eligible']
    count = out['pair_results']['gpt56:gpt-5.6-luna|gpt-5.6-sol']['source_pair_prompt_counts']['s1']
    assert count == 2


def test_bad_threshold_rejected():
    with pytest.raises(ValueError):
        m.audit([], min_pairs=0)
