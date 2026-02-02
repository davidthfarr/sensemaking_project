import numpy as np
from datetime import datetime

from sensemaking.data.schemas import Post
from sensemaking.clustering.hdbscan import HDBSCANClusterer


def make_post(post_id, emb, stance):
    return Post(
        post_id=post_id,
        text="dummy",
        timestamp=datetime.now(),
        embedding=emb,
        stance=stance,
    )


def test_joint_vector_construction():
    emb_dim = 5
    posts = [
        make_post("1", np.ones(emb_dim), 1),
        make_post("2", np.zeros(emb_dim), -1),
    ]

    clusterer = HDBSCANClusterer(stance_weight=0.5)
    X_raw = clusterer._build_raw_joint_vectors(posts)

    assert X_raw.shape == (2, emb_dim + 1)
    assert X_raw[0, -1] == 0.5
    assert X_raw[1, -1] == -0.5



def test_missing_embedding_raises():
    post = Post(
        post_id="1",
        text="dummy",
        timestamp=datetime.now(),
        stance=1,
    )

    clusterer = HDBSCANClusterer()

    try:
        clusterer.fit_predict([post])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_missing_stance_raises():
    post = Post(
        post_id="1",
        text="dummy",
        timestamp=datetime.now(),
        embedding=np.ones(5),
    )

    clusterer = HDBSCANClusterer()

    try:
        clusterer.fit_predict([post])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_hdbscan_runs_and_labels():
    emb_dim = 4

    posts = [
        make_post("1", np.array([1, 1, 1, 1]), 1),
        make_post("2", np.array([1.1, 1.0, 1.0, 1.1]), 1),
        make_post("3", np.array([-1, -1, -1, -1]), -1),
        make_post("4", np.array([-1.1, -1.0, -1.0, -1.1]), -1),
        make_post("5", np.array([10, 10, 10, 10]), 0),
    ]

    clusterer = HDBSCANClusterer(
        min_cluster_size=2,
        min_samples=1,  
    )
    posts = clusterer.fit_predict(posts)

    for p in posts:
        assert p.is_noise in {True, False}
        assert p.cluster_id is None or isinstance(p.cluster_id, int)
