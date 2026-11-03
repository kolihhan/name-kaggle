from __future__ import annotations

import math

import torch

from scripts.dct_temporal_pool import masked_dct_pool
from scripts.run_two_stage_lora24_gate_a import masked_mean_pool


def test_mode_zero_matches_v1_masked_mean() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [99.0, 990.0]],
        ],
        dtype=torch.float32,
    )
    attention = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    encoded = masked_dct_pool(hidden, attention, mask_first_token=True, n_modes=2)
    expected = masked_mean_pool(hidden, attention, mask_first_token=True)
    assert encoded.shape == (2, 2, 2)
    assert torch.allclose(encoded[:, 0], expected, atol=1e-6, rtol=1e-6)


def test_padding_does_not_change_dct_modes() -> None:
    base = torch.tensor([[[1.0], [2.0], [3.0]]])
    padded = torch.tensor([[[1.0], [2.0], [3.0], [777.0], [888.0]]])
    base_mask = torch.tensor([[1, 1, 1]])
    padded_mask = torch.tensor([[1, 1, 1, 0, 0]])
    a = masked_dct_pool(base, base_mask, mask_first_token=True, n_modes=2)
    b = masked_dct_pool(padded, padded_mask, mask_first_token=True, n_modes=2)
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_constant_trajectory_has_zero_first_ac_mode() -> None:
    hidden = torch.ones((2, 6, 4), dtype=torch.float32)
    attention = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    encoded = masked_dct_pool(hidden, attention, mask_first_token=True, n_modes=2)
    assert torch.allclose(encoded[:, 1], torch.zeros_like(encoded[:, 1]), atol=1e-6)


def test_single_effective_token_disables_ac1() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [7.0, 8.0]]])
    attention = torch.tensor([[1, 1]])
    encoded = masked_dct_pool(hidden, attention, mask_first_token=True, n_modes=2)
    assert torch.allclose(encoded[:, 0], hidden[:, 1], atol=1e-6)
    assert torch.allclose(encoded[:, 1], torch.zeros_like(encoded[:, 1]), atol=1e-6)


def test_zero_effective_span_matches_v1_zero_behavior() -> None:
    hidden = torch.tensor([[[3.0, 4.0]]])
    attention = torch.tensor([[1]])
    encoded = masked_dct_pool(hidden, attention, mask_first_token=True, n_modes=2)
    expected = masked_mean_pool(hidden, attention, mask_first_token=True)
    assert torch.allclose(encoded[:, 0], expected)
    assert torch.allclose(encoded, torch.zeros_like(encoded))


def test_first_ac_uses_length_normalized_dct_ii_weights() -> None:
    # Position zero is masked. Effective trajectory is [1, 2, 4].
    hidden = torch.tensor([[[100.0], [1.0], [2.0], [4.0]]])
    attention = torch.tensor([[1, 1, 1, 1]])
    encoded = masked_dct_pool(hidden, attention, mask_first_token=True, n_modes=2)
    values = [1.0, 2.0, 4.0]
    n = len(values)
    expected_ac = sum(
        value * (math.sqrt(2.0) / n) * math.cos(math.pi * (t + 0.5) / n)
        for t, value in enumerate(values)
    )
    assert torch.allclose(encoded[0, 1, 0], torch.tensor(expected_ac), atol=1e-6)
