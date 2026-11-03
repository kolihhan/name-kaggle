from __future__ import annotations

import math


def masked_dct_pool(hidden, attention_mask, *, mask_first_token: bool, n_modes: int = 2):
    """Length-normalized DCT-II over the same token span as v1 masked mean.

    Returns ``(batch, n_modes, hidden)``. Mode zero is exactly the arithmetic
    mean over valid tokens. Modes whose index is not smaller than the effective
    token count are zero, matching the standard DCT convention used by READER.
    """
    import torch

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape (batch, tokens, hidden)")
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError("attention_mask must match hidden batch/token dimensions")
    if n_modes <= 0:
        raise ValueError("n_modes must be positive")

    valid = attention_mask.to(dtype=torch.bool).clone()
    if mask_first_token and valid.shape[1] > 0:
        valid[:, 0] = False

    lengths = valid.sum(dim=1).to(torch.long)
    # Relative positions are counted only among valid tokens, making the helper
    # padding-invariant while preserving the usual contiguous right-padded case.
    relative = valid.to(torch.long).cumsum(dim=1) - 1
    relative = relative.clamp(min=0)

    safe_lengths = lengths.clamp(min=1).to(torch.float32)
    modes = torch.arange(n_modes, device=hidden.device, dtype=torch.float32)
    phase = (
        torch.pi
        * (relative.to(torch.float32)[:, None, :] + 0.5)
        * modes[None, :, None]
        / safe_lengths[:, None, None]
    )
    basis = torch.cos(phase)
    if n_modes > 1:
        basis[:, 1:] *= math.sqrt(2.0)
    basis /= safe_lengths[:, None, None]
    basis *= valid[:, None, :].to(torch.float32)
    basis *= (modes[None, :, None] < lengths[:, None, None]).to(torch.float32)
    basis = basis.to(dtype=hidden.dtype)
    return torch.einsum("bqt,btd->bqd", basis, hidden)
