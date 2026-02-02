import numpy as np
import torch
import pytest
from datetime import datetime

from sensemaking.embeddings.encoder import EmbeddingEncoder, attach_embeddings
from sensemaking.data.schemas import Post


def test_encoder_initializes():
    encoder = EmbeddingEncoder(
        batch_size=8,
        require_cuda=False,
    )
    assert encoder.model is not None


def test_encoder_device_resolution():
    encoder = EmbeddingEncoder(require_cuda=False)
    assert encoder.device in {"cpu", "cuda"}


@pytest.mark.parametrize("texts", [
    ["Hello world"],
    ["One", "Two", "Three"],
])
def test_encode_texts_shape(texts):
    encoder = EmbeddingEncoder(batch_size=4)
    embeddings = encoder.encode_texts(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 100  # MPNet dimensionality sanity check


def test_attach_embeddings():
    encoder = EmbeddingEncoder(batch_size=4)

    posts = [
        Post(post_id="1", text="Vaccines save lives", timestamp=datetime.now()),
        Post(post_id="2", text="Vaccines are dangerous", timestamp=datetime.now()),
    ]

    posts = attach_embeddings(posts, encoder)

    for post in posts:
        assert post.embedding is not None
        assert isinstance(post.embedding, np.ndarray)


def test_embeddings_are_normalized():
    encoder = EmbeddingEncoder(normalize=True)

    emb = encoder.encode_texts(["Test sentence"])[0]
    norm = np.linalg.norm(emb)

    assert np.isclose(norm, 1.0, atol=1e-3)
