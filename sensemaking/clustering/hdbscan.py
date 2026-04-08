"""
HDBSCAN-based clustering over joint semantic + stance representations.

Responsibilities:
- Construct joint vectors from Post objects
- Run HDBSCAN
- Assign cluster labels back to Post objects

No embedding logic.
No stance logic.
No windowing logic.
"""

from typing import List
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize

from sensemaking.data.schemas import Post


class HDBSCANClusterer:
    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples: int | None = None,
        stance_weight: float = 0.1,
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.stance_weight = stance_weight
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon

    def _build_raw_joint_vectors(self, posts: List[Post]) -> np.ndarray:
        embeddings = []
        stances = []

        for p in posts:
            if p.embedding is None:
                raise ValueError("Post missing embedding")
            if p.stance is None:
                raise ValueError("Post missing stance")

            embeddings.append(p.embedding)
            stances.append(p.stance)

        E = np.vstack(embeddings)
        s = np.array(stances).reshape(-1, 1)

        return np.hstack([E, self.stance_weight * s])

    def _build_joint_vectors(self, posts: List[Post]) -> np.ndarray:
        X = self._build_raw_joint_vectors(posts)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_predict(self, posts: List[Post]) -> List[Post]:
        """
        Cluster posts and assign labels.
        """
        X = self._build_joint_vectors(posts)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )

        labels = clusterer.fit_predict(X)

        for post, label in zip(posts, labels):
            if label == -1:
                post.cluster_id = None
                post.is_noise = True
            else:
                post.cluster_id = int(label)
                post.is_noise = False

        return posts

