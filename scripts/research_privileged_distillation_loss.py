from __future__ import annotations

from typing import Any


def selective_privileged_distillation(
    *,
    student_logits: Any,
    teacher_logits: Any,
    target: Any,
    temperature: float,
    distill_weight: float,
) -> dict[str, Any]:
    """Synthetic research primitive for training-only privileged distillation.

    The teacher is always stop-gradient. A row is distilled only when the
    detached privileged teacher has lower supervised CE than the student on
    that training label. This function contains no dataset/model-selection
    policy and is not itself a candidate authorization.
    """
    import torch
    import torch.nn.functional as F

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if distill_weight < 0:
        raise ValueError("distill_weight must be non-negative")
    if student_logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError("student_logits and teacher_logits must be rank-2")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logits must have identical shape")
    if target.ndim != 1 or target.shape[0] != student_logits.shape[0]:
        raise ValueError("target must be rank-1 and row-aligned with logits")

    teacher_detached = teacher_logits.detach()

    student_ce_rows = F.cross_entropy(student_logits, target, reduction="none")
    teacher_ce_rows = F.cross_entropy(teacher_detached, target, reduction="none")
    teacher_selected = teacher_ce_rows < student_ce_rows.detach()

    supervised_loss = student_ce_rows.mean()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_detached / temperature, dim=-1)
    kl_rows = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1) * (temperature**2)

    selected_count = int(teacher_selected.sum().item())
    if selected_count:
        distillation_loss = kl_rows[teacher_selected].mean()
    else:
        # Preserve a differentiable zero on the student's device/dtype.
        distillation_loss = student_logits.sum() * 0.0

    loss = supervised_loss + float(distill_weight) * distillation_loss
    return {
        "loss": loss,
        "supervised_loss": supervised_loss,
        "distillation_loss": distillation_loss,
        "teacher_selected": teacher_selected.detach(),
        "selected_count": selected_count,
    }
