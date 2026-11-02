from __future__ import annotations

from dataclasses import dataclass, field

import torch

from scripts.cross_prompt_prototype import CrossPromptPrototypeBank

CONTRASTIVE_WEIGHT = 1.0
CONTRASTIVE_TAU = 0.2
PROMPT_BUCKET_SEED = 20261101


@dataclass
class PrototypeStepBuffer:
    """Detached examples accumulated during one optimizer step."""

    z_parts: list[torch.Tensor] = field(default_factory=list)
    label_parts: list[torch.Tensor] = field(default_factory=list)
    prompt_ids: list[str] = field(default_factory=list)

    def append(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        prompt_ids: list[str],
    ) -> None:
        if z.ndim != 2:
            raise ValueError("z must be [rows, dim]")
        if labels.ndim != 1 or labels.shape[0] != z.shape[0]:
            raise ValueError("labels must be [rows] aligned with z")
        if len(prompt_ids) != z.shape[0]:
            raise ValueError("prompt_ids must align with z")
        self.z_parts.append(z.detach().float().cpu())
        self.label_parts.append(labels.detach().long().cpu())
        self.prompt_ids.extend(str(value) for value in prompt_ids)

    def clear(self) -> None:
        self.z_parts.clear()
        self.label_parts.clear()
        self.prompt_ids.clear()

    @property
    def rows(self) -> int:
        return int(sum(part.shape[0] for part in self.z_parts))


def optimizer_step_was_applied(previous_scale: float, current_scale: float) -> bool:
    if previous_scale <= 0 or current_scale <= 0:
        raise ValueError("GradScaler scales must be positive")
    return float(current_scale) >= float(previous_scale)


@torch.no_grad()
def commit_prototypes_after_scaler_step(
    bank: CrossPromptPrototypeBank,
    buffer: PrototypeStepBuffer,
    *,
    previous_scale: float,
    current_scale: float,
    seed: int = PROMPT_BUCKET_SEED,
) -> bool:
    """Commit current-step rows only when the optimizer step really applied."""
    applied = optimizer_step_was_applied(previous_scale, current_scale)
    try:
        if not applied or buffer.rows == 0:
            return False
        z = torch.cat(buffer.z_parts, dim=0).to(bank.sums.device)
        labels = torch.cat(buffer.label_parts, dim=0).to(bank.sums.device)
        bank.update(z, labels, buffer.prompt_ids, seed=seed)
        return True
    finally:
        buffer.clear()
