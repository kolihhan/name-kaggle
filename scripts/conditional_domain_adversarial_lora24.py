"""CDA-L24 fixed model-learning mechanism. No data access or model execution.

Train-only conditional gradient reversal; no model/source label in the final
18-class inference API. This module is NOT an end-to-end model or Notebook.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional as F

ADV_LAMBDA = 0.10
MIN_CELL_COUNT = 20


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, representation: Tensor, strength: float) -> Tensor:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength outside [0,1]")
        ctx.strength = float(strength)
        return representation.view_as(representation)

    @staticmethod
    def backward(ctx, upstream: Tensor):
        return upstream.neg() * ctx.strength, None


def reverse_source_gradient(features: Tensor, *, strength: float) -> Tensor:
    """Identity in forward, -strength in backward on *features only*."""
    return _GradientReverse.apply(features, strength)


class ConditionalDomainAdversary(nn.Module):
    """Train-only source prediction conditioned on known source author's LABEL.

    The known label is a TRAINING artifact only. The final Kaggle inference
    excludes this entire module and uses the base LoRA24 18-way plain head.
    """
    def __init__(self, feature_dim: int, label_count: int, source_count: int):
        super().__init__()
        if feature_dim < 1 or label_count < 2 or source_count < 2:
            raise ValueError("invalid architecture dimensions")
        self.label_count = int(label_count)
        self.source_count = int(source_count)
        self.body = nn.Sequential(nn.LayerNorm(feature_dim),
                                  nn.Linear(feature_dim, 256), nn.GELU())
        self.label_embedding = nn.Embedding(label_count, 32)
        self.source_logits = nn.Linear(288, source_count)

    def forward(self, features: Tensor, known_labels: Tensor,
                *, gradient_strength: float = ADV_LAMBDA) -> Tensor:
        if features.ndim != 2 or known_labels.ndim != 1:
            raise ValueError("expected [batch,features] and [batch] labels")
        if features.shape[0] != len(known_labels):
            raise ValueError("batch mismatch")
        if not known_labels.dtype in (torch.long, torch.int64):
            raise TypeError("labels must be torch.long")
        if (known_labels < 0).any() or (known_labels >= self.label_count).any():
            raise ValueError("label index outside source-adversary vocabulary")
        z = reverse_source_gradient(features, strength=gradient_strength)
        a = self.body(z)
        b = self.label_embedding(known_labels)
        return self.source_logits(torch.cat((a, b), dim=-1))


@dataclass(frozen=True)
class CrossedSourceSupport:
    cell_counts: Tensor
    eligible_labels: Tensor
    eligible_source_cells: Tensor


def crossed_source_support(known_labels: Tensor, known_sources: Tensor,
                           *, label_count: int, source_count: int,
                           minimum: int = MIN_CELL_COUNT) -> CrossedSourceSupport:
    """Compute conditional identifiable support on TRAIN data, never a holdout.

    A class needs >=2 source cells with >=minimum examples; a single source
    can never identify within-class domain nuisance.
    """
    if (known_labels.ndim != 1 or known_sources.ndim != 1 or
        len(known_labels) != len(known_sources)):
        raise ValueError("train label/source shape mismatch")
    if known_labels.dtype != torch.long or known_sources.dtype != torch.long:
        raise TypeError("train identifiers must be torch.long")
    if minimum < 1 or label_count < 2 or source_count < 2:
        raise ValueError("invalid crossed-support settings")
    if len(known_labels) == 0:
        raise ValueError("empty training sample")
    if (known_labels < 0).any() or (known_labels >= label_count).any() or (
            known_sources < 0).any() or (known_sources >= source_count).any():
        raise ValueError("invalid source/label ID")
    pair = known_labels * source_count + known_sources
    counts = torch.bincount(pair, minlength=label_count * source_count)
    counts = counts.reshape(label_count, source_count)
    eligible_labels = (counts >= minimum).sum(dim=1) >= 2
    eligible_cells = (counts >= minimum) & eligible_labels.unsqueeze(1)
    return CrossedSourceSupport(counts, eligible_labels, eligible_cells)


def conditional_domain_loss(adversary: ConditionalDomainAdversary,
                            features: Tensor, labels: Tensor, sources: Tensor,
                            support: CrossedSourceSupport,
                            *, gradient_strength: float) -> tuple[Tensor, int]:
    """Cell-macro source CE on supported training rows only.

    No holdout rows or true hidden class labels ever enter here. A 0-row
    microbatch contributes zero, but an entire run MUST independently assert
    sufficient coverage; do not silently train on no eligible examples.
    """
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(sources):
        raise ValueError("invalid batch alignment")
    if labels.dtype != torch.long or sources.dtype != torch.long:
        raise TypeError("identifiers must be torch.long")
    cells = support.eligible_source_cells.to(labels.device)
    counts = support.cell_counts.to(labels.device)
    if cells.shape != (adversary.label_count, adversary.source_count):
        raise ValueError("support tensor has different source/label vocabulary")
    mask = cells[labels, sources]
    if not mask.any():
        return features.sum() * 0.0, 0
    logits = adversary(features[mask], labels[mask],
                       gradient_strength=gradient_strength)
    per_row = F.cross_entropy(logits, sources[mask], reduction="none")
    # Compute weights from complete TRAIN-domain counts; do not renormalize
    # source/class frequencies using validation data.
    w = counts[labels[mask], sources[mask]].to(per_row.dtype).reciprocal()
    loss = (per_row * w).sum() / w.sum()
    return loss, int(mask.sum().item())


def joint_training_loss(model_logits: Tensor, labels: Tensor, features: Tensor,
                        sources: Tensor, adversary: ConditionalDomainAdversary,
                        support: CrossedSourceSupport, *,
                        gradient_strength: float,
                        model_class_weights: Tensor | None = None
                        ) -> tuple[Tensor, dict]:
    """One external-stage step: ordinary identity CE + conditional source CE."""
    if model_logits.shape != (len(labels), adversary.label_count):
        raise ValueError("expected full-label plain-head logits")
    model_loss = F.cross_entropy(model_logits, labels, weight=model_class_weights)
    source_loss, eligible = conditional_domain_loss(
        adversary, features, labels, sources, support,
        gradient_strength=gradient_strength)
    return model_loss + source_loss, {
        "identity_ce": float(model_loss.detach()),
        "source_ce": float(source_loss.detach()),
        "eligible_source_rows": eligible,
        "batch_rows": len(labels),
        "strength": gradient_strength,
    }
