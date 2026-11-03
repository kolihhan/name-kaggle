from __future__ import annotations

import ast
from pathlib import Path

from scripts.materialize_dct_v3_worker import materialize


def test_materialized_worker_changes_only_dct_representation_contract() -> None:
    source = Path("scripts/run_two_stage_lora24_gate_a_gpu.py").read_text(encoding="utf-8")
    rendered = materialize(source)
    ast.parse(rendered)

    assert 'CANDIDATE = "dct-temporal-fingerprint-v3"' in rendered
    assert "from scripts.dct_temporal_pool import masked_dct_pool" in rendered
    assert 'representation_dim = hidden_size * len(layers) * 2' in rendered
    assert 'n_modes=2' in rendered
    assert 'mode_features[:, 0]' in rendered
    assert 'mode_features[:, 1]' in rendered
    assert 'mask_first_token=True' in rendered

    # Rejected v2 and unrelated future mechanisms must not leak in.
    forbidden = [
        "CrossPromptPrototypeBank",
        "CONTRASTIVE_WEIGHT",
        "attention_pool",
        "top_k",
        "family_router",
        "likelihood_clip",
    ]
    for marker in forbidden:
        assert marker not in rendered


def test_materialized_worker_preserves_v1_training_constants_and_loop() -> None:
    source = Path("scripts/run_two_stage_lora24_gate_a_gpu.py").read_text(encoding="utf-8")
    rendered = materialize(source)

    invariants = [
        'triplets = build_prompt_triplets(frame, train_idx, seed=seed)',
        'accumulation = int(spec["gradient_accumulation_triplets"])',
        'optimizer = torch.optim.AdamW(',
        'scaler = torch.amp.GradScaler("cuda", init_scale=256.0)',
        'plain_logits = plain(dropout(z))',
        'centered_logits = centered(dropout(center_triplet(z)))',
        'F.cross_entropy(',
        'float(spec["gradient_clip"])',
        'if loss_count != len(triplets):',
    ]
    for marker in invariants:
        assert marker in source
        assert marker in rendered


def test_materializer_fails_closed_on_v1_source_drift() -> None:
    source = Path("scripts/run_two_stage_lora24_gate_a_gpu.py").read_text(encoding="utf-8")
    drifted = source.replace(
        'plain = nn.Linear(hidden_size * len(layers), len(labels)).to(device).float()',
        'plain = nn.Linear(123, len(labels)).to(device).float()',
    )
    try:
        materialize(drifted)
    except RuntimeError as exc:
        assert "v1 worker drift" in str(exc)
    else:
        raise AssertionError("materializer must fail closed when the v1 head block drifts")
