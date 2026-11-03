import torch

from scripts.research_privileged_distillation_loss import selective_privileged_distillation


def test_teacher_teaches_only_when_it_has_lower_true_label_ce():
    student = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
    teacher = torch.tensor([[3.0, 0.0], [0.0, 2.0], [1.5, 0.0]], dtype=torch.float32)
    target = torch.tensor([0, 0, 0], dtype=torch.long)

    out = selective_privileged_distillation(
        student_logits=student,
        teacher_logits=teacher,
        target=target,
        temperature=2.0,
        distill_weight=1.0,
    )

    assert out["teacher_selected"].tolist() == [True, False, True]
    assert out["selected_count"] == 2
    assert out["distillation_loss"].item() > 0.0


def test_teacher_is_stop_gradient_privileged_signal():
    student = torch.tensor([[0.0, 2.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[3.0, 0.0]], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([0], dtype=torch.long)

    out = selective_privileged_distillation(
        student_logits=student,
        teacher_logits=teacher,
        target=target,
        temperature=2.0,
        distill_weight=1.0,
    )
    out["loss"].backward()

    assert student.grad is not None
    assert teacher.grad is None


def test_no_selected_teacher_reduces_to_plain_supervised_ce():
    student = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    teacher = torch.tensor([[0.0, 3.0], [3.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([0, 1], dtype=torch.long)

    out = selective_privileged_distillation(
        student_logits=student,
        teacher_logits=teacher,
        target=target,
        temperature=2.0,
        distill_weight=1.0,
    )

    expected = torch.nn.functional.cross_entropy(student, target)
    assert out["selected_count"] == 0
    assert torch.allclose(out["distillation_loss"], torch.zeros_like(expected))
    assert torch.allclose(out["loss"], expected)


def test_invalid_temperature_or_weight_fails_closed():
    student = torch.zeros((1, 2))
    teacher = torch.zeros((1, 2))
    target = torch.zeros((1,), dtype=torch.long)

    for temperature, weight in [(0.0, 1.0), (1.0, -1.0)]:
        try:
            selective_privileged_distillation(
                student_logits=student,
                teacher_logits=teacher,
                target=target,
                temperature=temperature,
                distill_weight=weight,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid frozen hyperparameter must fail closed")
