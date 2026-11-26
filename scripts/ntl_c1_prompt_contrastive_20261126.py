"""Prompt-matched supervised contrastive readout on frozen strong encoder embeddings.

CE-only and CE+contrastive use the same projector, label head, initialization,
prompt-group batches, optimizer, and data. No external source data or labels.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold


def prompt_group_splits(groups: np.ndarray, labels: np.ndarray, *, n_splits: int = 3):
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    if len(groups) != len(labels) or n_splits < 2:
        raise ValueError("invalid grouped split inputs")
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), labels, groups))


def prompt_supervised_contrast(z: torch.Tensor, labels: torch.Tensor,
                              prompts: torch.Tensor, *, temperature: float = 0.16,
                              same_prompt_negative_weight: float = 2.0):
    """InfoNCE positives share model label across distinct prompts.

    Different-generator responses to exactly the same prompt have boosted
    denominator weights (hard negatives); self edges are masked.
    """
    if z.ndim != 2 or labels.ndim != 1 or prompts.ndim != 1 or len(z) != len(labels) or len(z) != len(prompts):
        raise ValueError("bad tensor dimensions")
    if not 0 < temperature <= 1.0 or same_prompt_negative_weight < 1:
        raise ValueError("invalid contrastive hyperparameter")
    same_label = labels[:, None].eq(labels[None, :])
    same_prompt = prompts[:, None].eq(prompts[None, :])
    diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positive = same_label & ~same_prompt & ~diagonal
    if not torch.any(positive):
        raise ValueError("no positive model matches across different prompts")
    emb = F.normalize(z, dim=1)
    logits = emb @ emb.T / temperature
    logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
    neg_adjust = (same_prompt & ~same_label).float() * np.log(same_prompt_negative_weight)
    log_denom = torch.logsumexp(logits + neg_adjust, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~positive, torch.finfo(logits.dtype).min), dim=1)
    valid = positive.any(dim=1)
    return (log_denom[valid] - log_num[valid]).mean()


class Decoder(nn.Module):
    """Identical readout architecture in CE control and contrastive treatment."""
    def __init__(self, in_dim: int, num_classes: int, dim: int):
        super().__init__()
        self.project = nn.Linear(in_dim, dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        hidden = F.normalize(F.gelu(self.project(x)), p=2, dim=1)
        return self.head(hidden), hidden


def _fit(x, y, prompt, train_index, test_index, *, alpha, epochs, dim, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    nclasses = int(np.unique(y).size)
    if not np.array_equal(np.unique(y), np.arange(nclasses)):
        raise ValueError("labels must be contiguous zero-based")
    xx = torch.tensor(x, dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.long)
    pp = torch.tensor(prompt, dtype=torch.long)
    mdl = Decoder(xx.shape[1], nclasses, dim)
    opt = torch.optim.AdamW(mdl.parameters(), lr=0.002, weight_decay=0.01)
    groups = np.unique(prompt[train_index])
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        shuffled = rng.permutation(groups)
        for j in range(0, len(shuffled), 5):
            select = np.flatnonzero(
                np.isin(prompt, shuffled[j:j + 5])
                & np.isin(np.arange(len(prompt)), train_index)
            )
            if len(select) < 5:
                continue
            pred, zz = mdl(xx[select])
            ce = F.cross_entropy(pred, yy[select])
            if alpha:
                try:
                    cl = prompt_supervised_contrast(zz, yy[select], pp[select])
                except ValueError:
                    cl = ce * 0.
                loss = ce + alpha * cl
            else:
                loss = ce
            opt.zero_grad()
            loss.backward()
            opt.step()
    mdl.eval()
    with torch.inference_mode():
        return mdl(xx[test_index])[0].numpy()


def train_pair(x, y, prompt, train_index, test_index, *,
               epochs=8, dim=256, seed=20261126, alpha=0.2):
    """Matched source, exact same training batches, same random initialization."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    prompt = np.asarray(prompt, dtype=np.int64)
    tr = np.asarray(train_index, dtype=np.int64)
    te = np.asarray(test_index, dtype=np.int64)
    if len(x) != len(y) or len(y) != len(prompt) or not len(tr) or not len(te):
        raise ValueError("bad input lengths")
    if set(prompt[tr]) & set(prompt[te]):
        raise ValueError("prompt leakage into heldout")
    a = _fit(x, y, prompt, tr, te, alpha=0., epochs=epochs, dim=dim, seed=seed)
    b = _fit(x, y, prompt, tr, te, alpha=alpha, epochs=epochs, dim=dim, seed=seed)
    return a, b
