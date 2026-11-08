"""CDA-L24 engineering controls, synthetic only (not scientific model evaluation)."""
import pytest
import torch
from scripts.conditional_domain_adversarial_lora24 import (
    ADV_LAMBDA,
    ConditionalDomainAdversary,
    conditional_domain_loss,
    crossed_source_support,
    joint_training_loss,
)


def _support():
    # Two class/source crossings are actually supported for class 0.
    labels = torch.tensor([0] * 25 + [0] * 26 + [1] * 30, dtype=torch.long)
    sources = torch.tensor([0] * 25 + [1] * 26 + [0] * 30, dtype=torch.long)
    return crossed_source_support(labels, sources, label_count=2, source_count=2)


def test_crossed_support_excludes_single_source_class():
    s = _support()
    assert s.cell_counts.tolist() == [[25, 26], [30, 0]]
    assert s.eligible_labels.tolist() == [True, False]
    assert s.eligible_source_cells.tolist() == [[True, True], [False, False]]


def test_domain_forward_same_logits_grad_reversed_only_for_representation():
    torch.manual_seed(7)
    head = ConditionalDomainAdversary(4, 2, 2)
    labels = torch.tensor([0, 0], dtype=torch.long)
    src = torch.tensor([0, 1], dtype=torch.long)
    features = torch.tensor([[.2, -.1, .6, -.5], [-.4, .8, .2, .1]],
                            dtype=torch.float32, requires_grad=True)
    support = _support()
    values = []
    feature_grads = []
    weight_grads = []
    for strength in (0., ADV_LAMBDA):
        head.zero_grad(set_to_none=True)
        if features.grad is not None:
            features.grad.zero_()
        # Exactly identical domain forward for the two matched arms.
        logits = head(features, labels, gradient_strength=strength)
        loss, eligible = conditional_domain_loss(head, features, labels, src,
                                                 support, gradient_strength=strength)
        assert eligible == 2
        assert torch.isfinite(loss)
        loss.backward()
        values.append(logits.detach().clone())
        feature_grads.append(features.grad.detach().clone())
        weight_grads.append(head.source_logits.weight.grad.detach().clone())
    torch.testing.assert_close(values[0], values[1])
    torch.testing.assert_close(feature_grads[0], torch.zeros_like(feature_grads[0]))
    assert feature_grads[1].abs().sum() > 0
    torch.testing.assert_close(weight_grads[0], weight_grads[1])


def test_train_only_loss_excludes_unsupported_class_source_cell():
    head = ConditionalDomainAdversary(4, 2, 2)
    f = torch.randn(2, 4, requires_grad=True)
    labels = torch.tensor([1, 1])
    sources = torch.tensor([0, 0])
    loss, eligible = conditional_domain_loss(head, f, labels, sources, _support(),
                                             gradient_strength=ADV_LAMBDA)
    assert eligible == 0
    assert loss.item() == 0.
    loss.backward()
    torch.testing.assert_close(f.grad, torch.zeros_like(f))


def test_joint_objective_tracks_class_loss_and_source_coverage():
    torch.manual_seed(11)
    head = ConditionalDomainAdversary(4, 2, 2)
    features = torch.randn(3, 4, requires_grad=True)
    model_logits = torch.randn(3, 2, requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    sources = torch.tensor([0, 1, 0])
    result, log = joint_training_loss(model_logits, labels, features, sources,
                                      head, _support(), gradient_strength=ADV_LAMBDA)
    assert log["eligible_source_rows"] == 2
    assert log["batch_rows"] == 3
    assert log["source_ce"] > 0
    assert log["identity_ce"] > 0
    result.backward()
    assert model_logits.grad.abs().sum() > 0
    assert features.grad.abs().sum() > 0


def test_invalid_label_and_unsupported_source_ids_fail_closed():
    head = ConditionalDomainAdversary(4, 2, 2)
    with pytest.raises(ValueError, match="outside"):
        head(torch.ones(2, 4), torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="invalid source/label"):
        crossed_source_support(torch.tensor([0, 1]), torch.tensor([0, 2]),
                               label_count=2, source_count=2)
    with pytest.raises(ValueError, match="strength"):
        head(torch.ones(2, 4), torch.tensor([1, 0]), gradient_strength=1.5)
