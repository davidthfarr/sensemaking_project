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
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.stance_weight = stance_weight
        self.metric = metric

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
        n = len(posts)
    
        # ---- SAFETY GUARD ----
        min_required = max(
            self.min_cluster_size,
            self.min_samples or 1,
        )
    

        # Not enough points to cluster: mark all as noise
        if n < min_required:
            print(
                f"Skipping clustering: "
                f"{n} posts < min_required={min_required}"
            )
            for post in posts:
                post.cluster_id = None
                post.is_noise = True
                post.cluster_strength = 0.0
            return posts
        # ---------------------

        if not posts:
            return posts

        X = self._build_joint_vectors(posts)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples or self.min_cluster_size,
            metric=self.metric,
            prediction_data=True,
        )

        labels = clusterer.fit_predict(X)
        strengths = clusterer.probabilities_

        for post, label, strength in zip(posts, labels, strengths):
            if label == -1:
                post.cluster_id = None
                post.is_noise = True
                post.cluster_strength = float(strength)
            else:
                post.cluster_id = int(label)
                post.is_noise = False
                post.cluster_strength = float(strength)

        return posts

