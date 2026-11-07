import numpy as np
import pandas as pd
import pytest
import torch

from scripts.pretrained_source_geometry import (
    align_anchor,
    fold_reference_indices,
    kernel_class_scores,
    masked_mean,
    paired_prompt_interval,
    text_fingerprint,
)


def test_density_preserves_reference_modes_lost_by_one_centroid():
    q = np.array([[1.0, 0.0]])
    ref = np.array([[1.0, 0.0], [0.0, 1.0], [2 ** -0.5, 2 ** -0.5]])
    y = np.array(["a", "a", "b"])
    centroid = kernel_class_scores(q, ref, y, ["a", "b"], "centroid")
    density = kernel_class_scores(q, ref, y, ["a", "b"], "density")
    assert centroid.argmax(axis=1).tolist() == [1]
    assert density.argmax(axis=1).tolist() == [0]


def test_density_normalizes_out_reference_class_frequency():
    q = np.eye(2)
    ref = np.eye(2)
    original = kernel_class_scores(q, ref, np.array(["a", "b"]), ["a", "b"], "density")
    duplicated = kernel_class_scores(q, ref[[0, 0, 0, 1]], np.array(["a", "a", "a", "b"]),
                                    ["a", "b"], "density")
    np.testing.assert_allclose(original, duplicated, atol=1e-10)


def test_missing_reference_label_fails_instead_of_silently_remapping_classes():
    with pytest.raises(ValueError, match="missing reference"):
        kernel_class_scores(np.eye(2), np.eye(2), np.array(["a", "a"]), ["a", "b"], "density")


def test_masked_pooling_excludes_padding_and_keeps_valid_special_tokens():
    hidden = torch.tensor([[[1., 2.], [3., 4.], [100., 200.]]])
    mask = torch.tensor([[1, 1, 0]])
    torch.testing.assert_close(masked_mean(hidden, mask), torch.tensor([[2., 3.]]))
    with pytest.raises(ValueError, match="empty"):
        masked_mean(hidden, torch.zeros_like(mask))


def test_real_tiny_roberta_embedding_is_invariant_to_unrelated_batch_padding():
    from transformers import RobertaConfig, RobertaModel
    torch.manual_seed(3)
    model = RobertaModel(RobertaConfig(vocab_size=50, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         hidden_dropout_prob=0, attention_probs_dropout_prob=0)).eval()
    with torch.inference_mode():
        single = model(input_ids=torch.tensor([[0, 7, 2]]), attention_mask=torch.ones(1, 3)).last_hidden_state
        ids = torch.tensor([[0, 7, 2, 1, 1], [0, 8, 9, 10, 2]])
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        padded = model(input_ids=ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(masked_mean(single, torch.ones(1, 3)),
                               masked_mean(padded, mask)[:1], atol=2e-6, rtol=2e-6)


def test_reference_fold_excludes_whole_prompt_groups():
    frame = pd.DataFrame({"prompt_id": np.repeat(["a", "b", "c", "d", "e"], 2),
                          "fold_id": np.repeat(np.arange(5), 2)})
    fit, valid = fold_reference_indices(frame, 2)
    assert fit.tolist() == [0, 1, 2, 3, 6, 7, 8, 9]
    assert valid.tolist() == [4, 5]
    frame.loc[4, "fold_id"] = 1
    with pytest.raises(ValueError, match="prompt"):
        fold_reference_indices(frame, 2)


def test_anchor_is_joined_by_id_and_rejects_wrong_truth_or_folds():
    official = pd.DataFrame({"id": ["a", "b"], "prompt_id": ["p1", "p2"],
                             "model": ["x", "y"], "fold_id": [0, 1]})
    anchor = official.iloc[::-1].copy()
    anchor["plain_prediction"] = ["y", "x"]
    assert align_anchor(official, anchor).tolist() == ["x", "y"]
    anchor.loc[anchor.id == "a", "model"] = "y"
    with pytest.raises(ValueError, match="model"):
        align_anchor(official, anchor)


def test_prompt_interval_is_paired_and_not_a_row_bootstrap():
    frame = pd.DataFrame({"prompt_id": ["p", "p", "q", "q"],
                          "model": ["a", "b", "a", "b"]})
    y = frame.model.to_numpy()
    wrong = np.where(y == "a", "b", "a")
    result = paired_prompt_interval(frame, y, wrong, replicates=50)
    assert result["ci975"] == pytest.approx([1, 1])
    assert result["retained"] == 50
    assert result["prompt_groups"] == 2


def test_cache_fingerprint_tracks_order_and_each_exact_passage():
    assert text_fingerprint(["ab", "c"]) != text_fingerprint(["a", "bc"])
    assert text_fingerprint(["a", "b"]) != text_fingerprint(["b", "a"])
    assert text_fingerprint(["a", "b"]) != text_fingerprint(["a", "B"])


def test_legacy_position_buffer_is_checked_before_removing_nonparameter_state():
    from scripts import pretrained_source_geometry as geometry
    weight = torch.tensor([[0.3, 0.7]])
    old = {"encoder.weight": weight, "embeddings.position_ids": torch.arange(6).unsqueeze(0)}
    cleaned = geometry.compatible_state(old, 6)
    assert set(cleaned) == {"encoder.weight"}
    torch.testing.assert_close(cleaned["encoder.weight"], weight)
    old["embeddings.position_ids"][0, 2] = 99
    with pytest.raises(ValueError, match="position"):
        geometry.compatible_state(old, 6)
