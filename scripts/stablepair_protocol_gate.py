#!/usr/bin/env python3
"""Fail-closed metadata gate for source-disjoint sibling attribution research.

Input is an authorized, metadata-only CSV. Do not supply raw prompts, responses,
API tokens, or official competition rows. This does NOT train or score a model.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path

LABEL_GROUPS = {
    "gpt56": ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"),
    "gemini_flash": ("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"),
    "claude_opus": ("claude-opus-4-6", "claude-opus-4-8", "claude-opus-5"),
    "glm": ("glm-5.3", "glm-5.3-flash"),
}
REQUIRED = {
    "source_lineage", "prompt_origin", "prompt_hash", "model",
    "request_hash", "settings_hash", "target_words", "served_model",
    "served_revision", "rights_verified", "prefit_exposure",
}
TARGETS = frozenset(itertools.chain.from_iterable(LABEL_GROUPS.values()))
PAIRS = {
    f"{family}:{a}|{b}": (a, b)
    for family, members in LABEL_GROUPS.items()
    for a, b in itertools.combinations(members, 2)
}


def _digest(value: str) -> bool:
    return len(value) == 64 and all(ch in '0123456789abcdef' for ch in value)


def audit(rows: list[dict[str, str]], *, min_pairs: int = 100) -> dict:
    if min_pairs < 1:
        raise ValueError("min_pairs must be positive")
    excluded = collections.Counter()
    groups = collections.defaultdict(lambda: collections.defaultdict(set))
    # Even a verified metadata field is publisher-supplied: original provenance
    # and upstream identity must still be independently audited before training.
    for row in rows:
        x = {key: (row.get(key) or '').strip() for key in REQUIRED}
        if x['model'] not in TARGETS:
            excluded['unsupported_label'] += 1
            continue
        if any(not x[key] for key in REQUIRED):
            excluded['missing_metadata'] += 1
            continue
        if any(not _digest(x[key]) for key in ('prompt_hash', 'request_hash', 'settings_hash')):
            excluded['invalid_hash'] += 1
            continue
        # Fail closed on provider routing/alias ambiguity. A provider alias
        # needs a separately audited, explicit mapping before admission.
        if x['served_model'] != x['model']:
            excluded['served_identity_mismatch'] += 1
            continue
        if x['rights_verified'] != 'true' or x['prefit_exposure'] != 'none':
            excluded['unverified_rights_or_prefit_exposure'] += 1
            continue
        if not x['target_words'].isdigit() or int(x['target_words']) < 1:
            excluded['invalid_requested_length'] += 1
            continue
        key = (
            x['source_lineage'], x['prompt_origin'], x['prompt_hash'],
            x['request_hash'], x['settings_hash'], x['target_words'],
        )
        # Retain source and prompt *lineage*, not response text or row IDs.
        groups[key][x['model']].add((x['served_model'], x['served_revision']))

    pair_by_source = collections.defaultdict(lambda: collections.defaultdict(set))
    source_origins = collections.defaultdict(set)
    for key, present in groups.items():
        lineage, origin, prompt_hash, _, _, _ = key
        # Duplicate rows or ambiguous served revisions do not create evidence.
        good = {model for model, served in present.items() if len(served) == 1}
        source_origins[lineage].add(origin)
        for name, (a, b) in PAIRS.items():
            if a in good and b in good:
                pair_by_source[name][lineage].add(prompt_hash)

    results = {}
    for name in sorted(PAIRS):
        sources = pair_by_source.get(name, {})
        eligible = {source: hashes for source, hashes in sources.items()
                    if len(hashes) >= min_pairs}
        independent = []
        for a, b in itertools.combinations(sorted(eligible), 2):
            if (eligible[a].isdisjoint(eligible[b]) and
                    source_origins[a].isdisjoint(source_origins[b])):
                independent.append((a, b))
        results[name] = {
            "source_pair_prompt_counts": {s: len(v) for s, v in sorted(sources.items())},
            "source_lineages_passing_count_gate": len(eligible),
            "independent_source_pairs": len(independent),
            "status": "CAPACITY_ONLY_NOT_VALIDATED" if independent else "BLOCKED",
        }
    return {
        "scope": "metadata-only source/prompt/protocol sibling-pair feasibility",
        "rows_seen": len(rows),
        "rows_excluded": dict(sorted(excluded.items())),
        "eligible_protocol_prompt_blocks": len(groups),
        "min_distinct_matched_prompts_per_source": min_pairs,
        "pair_results": results,
        "any_pair_capacity_eligible": any(x['independent_source_pairs'] > 0 for x in results.values()),
        "independently_verified_provenance": False,
        "model_training_or_inference": False,
        "official_competition_score": None,
        "cautions": [
            "All hashes, rights flags, exposure flags and source lineages are assertions until source-owner audited.",
            "Same request/settings hashes must be canonicalized independently of the model ID.",
            "Different source names and nonoverlapping prompt hashes alone cannot prove independent origins.",
            "A matched source cohort is not an untouched 18-label holdout or proof of protocol invariance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata_csv', type=Path)
    parser.add_argument('--min-pairs', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with args.metadata_csv.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            parser.error('missing metadata columns: ' + ', '.join(sorted(missing)))
        result = audit(list(reader), min_pairs=args.min_pairs)
    output = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
