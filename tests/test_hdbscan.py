import numpy as np
from datetime import datetime

from sensemaking.data.schemas import Post
from sensemaking.clustering.hdbscan import HDBSCANClusterer


def make_post(post_id, emb):
    return Post(
        post_id=post_id,
        text="dummy",
        timestamp=datetime.now(),
        embedding=emb,
    )


def test_missing_embedding_raises():
    post = Post(
        post_id="1",
        text="dummy",
        timestamp=datetime.now(),
    )

    clusterer = HDBSCANClusterer()

    try:
        clusterer.fit_predict([post])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_embedding_matrix_shape():
    emb_dim = 5
    posts = [
        make_post("1", np.ones(emb_dim)),
        make_post("2", np.array([2.0] * emb_dim)),
    ]
    clusterer = HDBSCANClusterer()
    X = clusterer._build_matrix(posts)
    assert X.shape == (2, emb_dim)


def test_hdbscan_runs_and_labels():
    posts = [
        make_post("1", np.array([1.0, 1.0, 1.0, 1.0])),
        make_post("2", np.array([1.1, 1.0, 1.0, 1.1])),
        make_post("3", np.array([-1.0, -1.0, -1.0, -1.0])),
        make_post("4", np.array([-1.1, -1.0, -1.0, -1.1])),
        make_post("5", np.array([10.0, 10.0, 10.0, 10.0])),
    ]

    clusterer = HDBSCANClusterer(min_cluster_size=2, min_samples=1)
    posts = clusterer.fit_predict(posts)

    for p in posts:
        assert p.is_noise in {True, False}
        assert p.cluster_id is None or isinstance(p.cluster_id, int)
