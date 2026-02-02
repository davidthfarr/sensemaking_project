import numpy as np
from datetime import datetime

from sensemaking.data.schemas import Post
from sensemaking.clustering.alignment import align_clusters


def make_post(post_id, emb, cluster_id):
    return Post(
        post_id=post_id,
        text="dummy",
        timestamp=datetime.now(),
        embedding=emb,
        stance=0,
        cluster_id=cluster_id,
        is_noise=False,
    )


def test_simple_alignment():
    # Window t
    prev_posts = [
        make_post("1", np.array([1, 0, 0]), 0),
        make_post("2", np.array([1.1, 0, 0]), 0),
        make_post("3", np.array([0, 1, 0]), 1),
    ]

    # Window t+1
    curr_posts = [
        make_post("4", np.array([1.05, 0, 0]), 10),
        make_post("5", np.array([0, 1.1, 0]), 11),
    ]

    alignment = align_clusters(prev_posts, curr_posts)

    assert alignment[0] == 10
    assert alignment[1] == 11


def test_threshold_blocks_alignment():
    prev_posts = [
        make_post("1", np.array([1, 0, 0]), 0),
    ]

    curr_posts = [
        make_post("2", np.array([-1, 0, 0]), 5),
    ]

    alignment = align_clusters(
        prev_posts,
        curr_posts,
        similarity_threshold=0.9,
    )

    assert alignment == {}


def test_noise_ignored():
    prev_posts = [
        make_post("1", np.array([1, 0, 0]), None),
    ]

    curr_posts = [
        make_post("2", np.array([1, 0, 0]), 0),
    ]

    alignment = align_clusters(prev_posts, curr_posts)

    assert alignment == {}
