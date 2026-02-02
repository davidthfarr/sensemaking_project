import torch
import pytest
from datetime import datetime

from sensemaking.embeddings.stance import ZeroShotStanceLabeler
from sensemaking.data.schemas import Post


@pytest.fixture(scope="session")
def stance_model():
    return ZeroShotStanceLabeler(
        batch_size=4,
        require_cuda=False,
    )


def test_stance_model_initializes(stance_model):
    assert stance_model.model is not None
    assert stance_model.tokenizer is not None


def test_predict_batch_output_range(stance_model):
    texts = [
        "Vaccines save lives",
        "Vaccines are dangerous",
        "I am not sure what to think",
    ]

    stances = stance_model.predict_batch(texts)

    assert len(stances) == len(texts)
    for s in stances:
        assert s in {-1, 0, 1}


def test_attach_stance():
    stance_model = ZeroShotStanceLabeler(batch_size=2)

    posts = [
        Post(post_id="1", text="Vaccines save lives", timestamp=datetime.now()),
        Post(post_id="2", text="Vaccines are dangerous", timestamp=datetime.now()),
    ]

    posts = stance_model.attach_stance(posts)

    for post in posts:
        assert post.stance in {-1, 0, 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_stance_model_uses_cuda():
    stance_model = ZeroShotStanceLabeler(require_cuda=True)
    assert stance_model.device.type == "cuda"
