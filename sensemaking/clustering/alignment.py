"""
Cluster alignment utilities.

This module aligns clusters across adjacent time windows using
centroid cosine similarity and Hungarian matching.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

from sensemaking.data.schemas import Post


ClusterID = int
Alignment = Dict[ClusterID, ClusterID]


def _compute_cluster_centroids(
    posts: List[Post],
) -> Dict[ClusterID, np.ndarray]:
    """
    Compute semantic centroids for each cluster.

    Noise points (cluster_id is None) are ignored.
    """
    clusters: Dict[ClusterID, List[np.ndarray]] = {}

    for post in posts:
        if post.cluster_id is None:
            continue

        clusters.setdefault(post.cluster_id, []).append(post.embedding)

    centroids = {
        cid: np.mean(np.vstack(embs), axis=0)
        for cid, embs in clusters.items()
    }

    return centroids


def align_clusters(
    prev_posts: List[Post],
    curr_posts: List[Post],
    similarity_threshold: float = 0.5,
) -> Alignment:
    """
    Align clusters between two adjacent windows.

    Parameters
    ----------
    prev_posts : List[Post]
        Posts from previous window (already clustered).
    curr_posts : List[Post]
        Posts from current window (already clustered).
    similarity_threshold : float
        Minimum cosine similarity required to accept an alignment.

    Returns
    -------
    Dict[int, int]
        Mapping from previous cluster_id to current cluster_id.
    """
    prev_centroids = _compute_cluster_centroids(prev_posts)
    curr_centroids = _compute_cluster_centroids(curr_posts)

    if not prev_centroids or not curr_centroids:
        return {}

    prev_ids = list(prev_centroids.keys())
    curr_ids = list(curr_centroids.keys())

    prev_matrix = np.vstack([prev_centroids[cid] for cid in prev_ids])
    curr_matrix = np.vstack([curr_centroids[cid] for cid in curr_ids])

    sim_matrix = cosine_similarity(prev_matrix, curr_matrix)

    # Hungarian matching maximizes total similarity
    row_idx, col_idx = linear_sum_assignment(-sim_matrix)

    alignment: Alignment = {}

    for r, c in zip(row_idx, col_idx):
        sim = sim_matrix[r, c]
        if sim >= similarity_threshold:
            alignment[prev_ids[r]] = curr_ids[c]

    return alignment
