from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def prompt_bucket(prompt_id: str, seed: int = 20261101) -> int:
    payload = f"{seed}\0{prompt_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()[0] & 1


@dataclass
class PrototypeLossResult:
    loss: torch.Tensor
    eligible_anchors: int
    total_anchors: int


class CrossPromptPrototypeBank:
    """Two-bucket, stop-gradient exact-model prototype memory.

    An anchor in prompt bucket b is compared only against historical model
    prototypes from bucket 1-b. A prompt cannot supply its own positive because
    one prompt deterministically belongs to exactly one bucket.
    """

    def __init__(self, num_classes: int, dim: int, *, device, dtype=torch.float32):
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.sums = torch.zeros((2, num_classes, dim), device=device, dtype=dtype)
        self.counts = torch.zeros((2, num_classes), device=device, dtype=torch.long)

    @torch.no_grad()
    def update(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        prompt_ids: list[str],
        *,
        seed: int = 20261101,
    ) -> None:
        if z.ndim != 2 or labels.ndim != 1 or len(prompt_ids) != z.shape[0]:
            raise ValueError("prototype update inputs are not row-aligned")
        normalized = F.normalize(z.detach().float(), dim=1)
        for row, prompt_id in enumerate(prompt_ids):
            bucket = prompt_bucket(str(prompt_id), seed)
            label = int(labels[row].item())
            self.sums[bucket, label].add_(normalized[row])
            self.counts[bucket, label].add_(1)

    def loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        prompt_ids: list[str],
        *,
        tau: float = 0.2,
        seed: int = 20261101,
    ) -> PrototypeLossResult:
        if tau <= 0:
            raise ValueError("tau must be > 0")
        if z.ndim != 2 or labels.ndim != 1 or len(prompt_ids) != z.shape[0]:
            raise ValueError("prototype loss inputs are not row-aligned")

        normalized = F.normalize(z.float(), dim=1)
        losses: list[torch.Tensor] = []
        for row, prompt_id in enumerate(prompt_ids):
            opposite = 1 - prompt_bucket(str(prompt_id), seed)
            available = self.counts[opposite] > 0
            target = int(labels[row].item())
            if not bool(available[target]):
                continue
            class_ids = torch.nonzero(available, as_tuple=False).flatten()
            if class_ids.numel() < 2:
                continue
            prototypes = self.sums[opposite, class_ids] / self.counts[
                opposite, class_ids
            ].unsqueeze(1)
            prototypes = F.normalize(prototypes, dim=1)
            logits = (normalized[row : row + 1] @ prototypes.T) / tau
            target_position = torch.nonzero(
                class_ids == target, as_tuple=False
            ).item()
            losses.append(
                F.cross_entropy(
                    logits,
                    torch.tensor([target_position], device=z.device),
                )
            )

        if not losses:
            return PrototypeLossResult(z.sum() * 0.0, 0, int(z.shape[0]))
        return PrototypeLossResult(
            torch.stack(losses).mean(), len(losses), int(z.shape[0])
        )
