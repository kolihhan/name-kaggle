import torch

from scripts import run_two_stage_lora24_gate_a as gate


def test_masked_mean_pool_excludes_first_token_and_padding():
    hidden = torch.tensor(
        [[[100.0, 100.0], [1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]]
    )
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    pooled = gate.masked_mean_pool(
        hidden,
        attention_mask,
        mask_first_token=True,
    )

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))


def test_center_triplet_has_zero_mean_per_feature():
    features = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [8.0, 11.0]],
    )

    centered = gate.center_triplet(features)

    torch.testing.assert_close(centered.mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        centered,
        features - features.mean(dim=0, keepdim=True),
    )


def test_truncate_qwen_layers_keeps_first_layers_only():
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2) for _ in range(6)]
            )

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    model = Dummy()
    original = list(model.model.layers)

    gate.truncate_qwen_layers(model, keep_layers=4)

    assert len(model.model.layers) == 4
    assert list(model.model.layers) == original[:4]
