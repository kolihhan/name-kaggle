"""Short-TRACE frozen rank-bin features. No model fitting or raw text output.

Rank is zero-based number of evaluator vocabulary logits strictly greater than
the logit of the observed next token; this is NOT source-generator decoding.
"""
from __future__ import annotations
import hashlib
import numpy as np

BINS = (1, 2, 4, 8, 16, 32, 64, 128)
N_BINS = 9
SEED = 20261107


def bins_from_ranks(ranks):
    r = np.asarray(ranks)
    if r.ndim != 1 or not np.issubdtype(r.dtype, np.integer):
        raise ValueError("ranks must be 1D integer vector")
    if np.any(r < 0):
        raise ValueError("negative rank")
    return np.searchsorted(BINS, r, side="right").astype(np.int64)


def _normalized(arr):
    a = np.asarray(arr, dtype=np.float64)
    total = a.sum()
    if total:
        a = a / total
    return a.astype(np.float32)


def make_features(ranks, text):
    """Return H, S, T. Shuffle preserves same ranks/marginal exactly.

    Seeds derive from original text only, not any label or source.
    Adjacent pairs are within each document; never across documents.
    """
    b = bins_from_ranks(ranks)
    hist = _normalized(np.bincount(b, minlength=N_BINS))
    true = np.zeros((N_BINS, N_BINS), dtype=np.float64)
    shuffled = np.zeros_like(true)
    if len(b) > 1:
        np.add.at(true, (b[:-1], b[1:]), 1)
        digest = hashlib.sha256((str(SEED) + "|" + text).encode("utf8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        b_shuffle = rng.permutation(b)
        np.add.at(shuffled, (b_shuffle[:-1], b_shuffle[1:]), 1)
    t = np.concatenate([hist, _normalized(true).ravel()])
    s = np.concatenate([hist, _normalized(shuffled).ravel()])
    return {"H": hist, "S": s, "T": t}


def evaluator_ranks(model, tokenizer, text, max_tokens=384):
    """One full causal-LM forward; no generation or model updates.

    Computes rank without sorting the full vocabulary at each position.
    Forces single-example input so no padding crosses token boundaries.
    """
    import torch
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty text")
    ids = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=max_tokens, return_tensors="pt")["input_ids"]
    if ids.shape[1] < 2:
        return np.empty(0, dtype=np.int64), {"tokens": int(ids.shape[1]), "ties": 0}
    device = next(model.parameters()).device
    ids = ids.to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1, :].float()
        targets = ids[0, 1:]
        target_logits = logits.gather(-1, targets[:, None])
        ranks = (logits > target_logits).sum(-1)
        ties = ((logits == target_logits).sum(-1) - 1).clamp_min(0).sum()
    result = ranks.cpu().numpy().astype(np.int64)
    stats = {"tokens": int(ids.shape[1]), "ties": int(ties.item()),
             "truncated": int(ids.shape[1] == max_tokens)}
    del logits
    return result, stats
