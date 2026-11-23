"""TCAE-v0: one-pass token-conditioned source energy (research prototype).

This is NOT a trained/scored NTL candidate. No competition data, Kaggle API,
third-party model or weights are loaded by this module or its self-tests.
Method contract: docs/research/ntl-token-conditional-identity-design-20261123.md

Causal contract:
  hidden[b,t] is computed from [BOS, target_1, ..., target_(t-1)] ONLY;
  token_embedding[b,t] is embedding of the actually observed target_t.
The caller, not this module, must respect the alignment. See align_next_token.
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


def align_next_token(hidden: torch.Tensor, token_ids: torch.Tensor,
                     output_weight: torch.Tensor):
    """Require caller hidden for [BOS]+targets; strip last state before scoring.

    The LM must run causally with a BOS prefix. No future label or prompt_id
    is ever an input to this method. The target token embedding is deliberately
    the CURRENT observed token, while the context hidden state is PRIOR-only.
    """
    if token_ids.ndim != 2 or hidden.ndim != 3 or token_ids.shape[0] != hidden.shape[0]:
        raise ValueError("Expected aligned [batch,length] token IDs and [batch,length+1,hidden] states")
    if hidden.shape[1] != token_ids.shape[1] + 1:
        raise ValueError("Hidden must be for [BOS]+targets with one extra time step")
    # LM heads are frequently nn.Linear, not nn.Embedding. Gather vocabulary
    # rows from the frozen output projection weight, never call Linear(ids).
    if output_weight.ndim != 2 or output_weight.shape[1] != hidden.shape[2]:
        raise ValueError("LM output matrix must be [vocab, hidden_dim]")
    if token_ids.min() < 0 or token_ids.max() >= output_weight.shape[0]:
        raise ValueError("Invalid target ID")
    # Provide both views from the same frozen causal backbone pass.
    return hidden[:, :-1, :], hidden[:, 1:, :], F.embedding(token_ids,output_weight.detach())


def masked_mean(x: torch.Tensor, mask: torch.Tensor):
    """Accept only finite full batch and at least one real token per response."""
    if x.ndim != 3 or mask.shape != x.shape[:2]:
        raise ValueError("Expected [B,T,D] values with [B,T] mask")
    if not torch.isfinite(x).all():
        raise FloatingPointError("nonfinite token input")
    valid = mask.to(dtype=x.dtype)
    n = valid.sum(dim=1, keepdim=True)
    if (n < 1).any() or not torch.isfinite(n).all():
        raise ValueError("Every response needs >=1 token")
    return (x * valid.unsqueeze(-1)).sum(dim=1) / n


class TokenConditionalAuthorshipEnergy(nn.Module):
    """A 18-way discriminative energy over causal context × chosen token."""

    def __init__(self, dim: int, classes: int = 18, rank: int = 16):
        super().__init__()
        if min(dim, classes, rank) <= 0 or rank > dim:
            raise ValueError("invalid dimensions")
        self.dim, self.classes, self.rank = dim, classes, rank
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Parameter(torch.empty(classes, rank, dim))
        self.bias = nn.Parameter(torch.zeros(classes))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.normal_(self.up, mean=0.0, std=0.02)

    def forward(self, prev_hidden: torch.Tensor,
                observed_token_embedding: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        if prev_hidden.ndim != 3 or observed_token_embedding.shape != prev_hidden.shape:
            raise ValueError("Prior hidden and current token embedding must both be [B,T,D]")
        if prev_hidden.shape[2] != self.dim:
            raise ValueError("dimension mismatch")
        if attention_mask.shape != prev_hidden.shape[:2]:
            raise ValueError("mask shape mismatch")
        if not torch.isfinite(prev_hidden).all() or not torch.isfinite(observed_token_embedding).all():
            raise FloatingPointError("nonfinite inputs")
        # One frozen LM forward supplied by caller. No costly 18x vocab softmax.
        x = self.down(prev_hidden)
        # [B,T,r] × [classes,r,D] × [B,T,D] -> [B,T,classes]
        energy = torch.einsum("btr,crd,btd->btc", x, self.up, observed_token_embedding)
        score = masked_mean(energy, attention_mask)
        return score + self.bias[None, :]


class DirectPooledHead(nn.Module):
    """Equal-backbone, approximately parameter-matched supervised CE control."""

    def __init__(self, dim: int, classes: int = 18, target_params: int = 0):
        super().__init__()
        if dim <= 0 or classes <= 0:
            raise ValueError("invalid dimensions")
        width = max(4, round((target_params - classes) / (dim + 1 + classes)))
        self.body = nn.Sequential(nn.Linear(dim, width), nn.GELU(),
                                  nn.Linear(width, classes))

    def forward(self, current_hidden, observed_token_embedding, attention_mask):
        # Fair direct control sees all current response tokens from hidden[:,1:].
        return self.body(masked_mean(current_hidden, attention_mask))


class TokenOnlyHead(nn.Module):
    """Lexical unigram source-control: no prior-context interaction."""

    def __init__(self, dim: int, classes: int = 18):
        super().__init__()
        self.head = nn.Linear(dim, classes)

    def forward(self, prev_hidden, observed_token_embedding, attention_mask):
        return self.head(masked_mean(observed_token_embedding, attention_mask))


def trainable_count(model):
    return sum(t.numel() for t in model.parameters() if t.requires_grad)


def selftest():
    torch.manual_seed(20261123)
    d, c, r = 4, 2, 2
    model = TokenConditionalAuthorshipEnergy(d, classes=c, rank=r)
    # Make the alternative hypotheses exactly distinguishable on synthetic arrays.
    with torch.no_grad():
        model.down.weight[:] = torch.tensor([[1., 0., 0., 0.],
                                             [0., 1., 0., 0.]])
        model.up[:] = 0
        # class0 rewards context-token basis alignment; class1 rewards crossing
        model.up[0, 0, 0] = model.up[0, 1, 1] = 1.
        model.up[1, 0, 1] = model.up[1, 1, 0] = 1.
        model.bias[:] = 0
    hidden = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]]])
    target = hidden.clone()
    switched = target.flip(dims=[1])
    mask = torch.ones(1, 2)
    s_aligned = model(hidden, target, mask)
    s_crossed = model(hidden, switched, mask)
    assert (s_aligned[0,0] > s_aligned[0,1]).item()
    assert (s_crossed[0,1] > s_crossed[0,0]).item()
    # A token-identity-only observer cannot tell: same multiset, different order.
    lexical = TokenOnlyHead(d, classes=c)
    assert torch.allclose(lexical(hidden,target,mask),
                          lexical(hidden,switched,mask),atol=1e-6)
    # Padding cannot change a document's score.
    h2 = torch.cat([hidden, torch.full((1,1,d), 200.)], dim=1)
    t2 = torch.cat([target, torch.full((1,1,d),-111.)], dim=1)
    mask2 = torch.tensor([[1,1,0]])
    assert torch.allclose(model(h2,t2,mask2),s_aligned,atol=1e-6)
    # A BOS+target causal alignment offset is mandatory.
    fake_emb = nn.Embedding(8,d)
    raw_ids = torch.tensor([[1,2]])
    h_bos = torch.randn(1,3,d)
    h_prev, h_current, e_current = align_next_token(h_bos,raw_ids,fake_emb.weight)
    assert h_prev.shape == h_current.shape == e_current.shape == (1,2,d)
    assert torch.equal(h_prev, h_bos[:,:-1]) and torch.equal(h_current,h_bos[:,1:])
    try:
        align_next_token(h_bos[:,:-1],raw_ids,fake_emb.weight)
        raise AssertionError("causal guard failed")
    except ValueError:
        pass
    # Full 18-class production-shape selftest, masked outputs and optimizer grad.
    b, t, d, c = 3, 5, 48, 18
    tcae=TokenConditionalAuthorshipEnergy(d,classes=c,rank=16)
    control=DirectPooledHead(d,classes=c,target_params=trainable_count(tcae))
    lexical=TokenOnlyHead(d,classes=c)
    h=torch.randn(b,t,d)
    e=torch.randn(b,t,d)
    m=torch.tensor([[1,1,1,1,1],[1,1,1,0,0],[1,0,0,0,0]])
    y=torch.tensor([3,0,17])
    z=tcae(h,e,m)
    assert z.shape == (b,c) and torch.isfinite(z).all()
    F.cross_entropy(z,y).backward()
    assert tcae.up.grad is not None and tcae.down.weight.grad is not None
    assert tcae.up.grad.abs().sum()>0 and tcae.down.weight.grad.abs().sum()>0
    assert control(h,e,m).shape == lexical(h,e,m).shape == (b,c)
    print("NTL_TCAE_V0_SYNTHETIC_CAUSAL_ALIGNMENT=PASS",flush=True)
    print("NTL_TCAE_V0_SYNTHETIC_TOKEN_CONTEXT_INTERACTION=PASS",flush=True)
    print("NTL_TCAE_V0_SYNTHETIC_PADDING_AND_GRADIENTS=PASS",flush=True)
    print("NTL_TCAE_V0_PARAMETER_COUNTS="+json.dumps({
       "tcae":trainable_count(tcae),"direct":trainable_count(control),
       "lexical":trainable_count(lexical),
       "same_backbone_required":True,
       "real_training_or_attribution_BA":False},sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--selftest",action="store_true")
    args=parser.parse_args()
    if not args.selftest:
        raise SystemExit("Design-first only: real training entrypoint not authorized by this standalone prototype.")
    selftest()


if __name__=="__main__":
    main()
