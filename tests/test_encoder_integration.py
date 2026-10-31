import numpy as np

from scripts.run_frozen_encoder_ablation import encode_texts


def test_encode_texts_returns_finite_unit_normalized_embeddings():
    embeddings = encode_texts(
        ["one short passage", "another short passage"],
        model_name="hf-internal-testing/tiny-random-bert",
        batch_size=2,
        max_length=32,
        device="cpu",
    )

    assert embeddings.shape[0] == 2
    assert embeddings.ndim == 2
    assert np.isfinite(embeddings).all()
    np.testing.assert_allclose(
        np.linalg.norm(embeddings, axis=1),
        np.ones(2),
        atol=1e-5,
    )
