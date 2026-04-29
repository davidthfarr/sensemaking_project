"""
Stationary theme generation for global clusters.

Each global cluster receives a theme label exactly once, at cluster birth
(the first window in which it appears). The label is a short, claim-like
summary generated from representative posts at that moment and never updated,
even as the cluster evolves across subsequent windows.

Two public classes:

  ThemeStore
      Lightweight dict-backed store that maps global_cluster_id → theme string.
      Serializes to/from plain JSON so it persists across pipeline runs.

  StationaryThemeLabeler
      Selects representative posts for a new cluster and calls GPT to generate
      the theme. Checks ThemeStore before calling the API so it never labels
      the same cluster twice.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from openai import OpenAI

from sensemaking.data.schemas import Post

_PROMPT_TEMPLATE = """\
You are analyzing social media posts about a conflict event. \
Here are 5 representative posts from a cluster of similar posts:

{posts}

In 10 words or fewer, write a concise claim-like label that \
captures the main narrative theme of this cluster. \
Return only the label, nothing else.\
"""


# ---------------------------------------------------------------------------
# ThemeStore
# ---------------------------------------------------------------------------

class ThemeStore:
    """
    Maps global_cluster_id → stationary theme string.

    A cluster is considered new if its ID is not yet in the store.
    Serializes to/from JSON for persistence across pipeline runs.
    """

    def __init__(self, themes: Optional[Dict[int, str]] = None):
        self._themes: Dict[int, str] = themes or {}

    def is_new(self, global_cluster_id: int) -> bool:
        return global_cluster_id not in self._themes

    def get(self, global_cluster_id: int) -> Optional[str]:
        return self._themes.get(global_cluster_id)

    def set(self, global_cluster_id: int, theme: str) -> None:
        self._themes[global_cluster_id] = theme

    def __len__(self) -> int:
        return len(self._themes)

    def to_dict(self) -> Dict[int, str]:
        return dict(self._themes)

    @classmethod
    def from_dict(cls, d: Dict) -> "ThemeStore":
        return cls({int(k): str(v) for k, v in d.items()})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self._themes.items()}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ThemeStore":
        with open(Path(path), "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)


# ---------------------------------------------------------------------------
# StationaryThemeLabeler
# ---------------------------------------------------------------------------

class StationaryThemeLabeler:
    """
    Generates a stationary theme label for a cluster at birth using GPT.

    Parameters
    ----------
    model : str
        OpenAI model ID.
    n_representative : int
        Number of representative posts to pass to GPT. Posts are selected by
        proximity to the cluster centroid in embedding space. If embeddings are
        absent, posts are taken in their original order up to this limit.
    max_post_chars : int
        Character limit applied to each post before sending to the API.
    api_key : str or None
        OpenAI API key. If None, the client reads OPENAI_API_KEY from the
        environment.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        n_representative: int = 5,
        max_post_chars: int = 400,
        api_key: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.n_representative = n_representative
        self.max_post_chars = max_post_chars

    # ------------------------------------------------------------------
    # Representative post selection
    # ------------------------------------------------------------------

    def _select_representative(self, posts: List[Post]) -> List[Post]:
        """Return up to n_representative posts closest to the cluster centroid."""
        n = min(self.n_representative, len(posts))

        embeddings = [p.embedding for p in posts if p.embedding is not None]
        if len(embeddings) < len(posts):
            # Some posts lack embeddings — fall back to first-n
            return posts[:n]

        matrix = np.vstack(embeddings)
        centroid = matrix.mean(axis=0)

        # Cosine similarity to centroid
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = matrix / norms

        centroid_norm = centroid / (np.linalg.norm(centroid) or 1.0)
        similarities = normed @ centroid_norm

        top_indices = np.argsort(similarities)[::-1][:n]
        return [posts[i] for i in top_indices]

    # ------------------------------------------------------------------
    # GPT call
    # ------------------------------------------------------------------

    def _build_prompt(self, posts: List[Post]) -> str:
        lines = "\n".join(
            f"{i + 1}. {p.text[: self.max_post_chars]}"
            for i, p in enumerate(posts)
        )
        return _PROMPT_TEMPLATE.format(posts=lines)

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def label_cluster(self, posts: List[Post]) -> str:
        """
        Generate a theme for a cluster from its posts at birth.

        Parameters
        ----------
        posts : List[Post]
            All posts belonging to the cluster in its birth window.
            Non-noise posts only; cluster_id should already be assigned.

        Returns
        -------
        str
            A short, claim-like theme string.
        """
        if not posts:
            return "Unknown narrative"

        representative = self._select_representative(posts)
        prompt = self._build_prompt(representative)
        return self._call_api(prompt)

    def assign_new_themes(
        self,
        posts: List[Post],
        global_cluster_ids: Dict[int, int],
        store: ThemeStore,
    ) -> ThemeStore:
        """
        Label all clusters in this window that do not yet have a theme.

        Parameters
        ----------
        posts : List[Post]
            Clustered posts for the current window (noise excluded or included;
            noise posts with cluster_id=None are ignored automatically).
        global_cluster_ids : Dict[int, int]
            Mapping from local cluster_id → global_cluster_id for this window,
            as produced by the alignment step.
        store : ThemeStore
            Existing theme store; updated in place and returned.

        Returns
        -------
        ThemeStore
            Updated store containing themes for any newly-born clusters.
        """
        # Group non-noise posts by local cluster_id
        local_groups: Dict[int, List[Post]] = {}
        for p in posts:
            if p.cluster_id is not None:
                local_groups.setdefault(p.cluster_id, []).append(p)

        for local_id, global_id in global_cluster_ids.items():
            if not store.is_new(global_id):
                continue

            cluster_posts = local_groups.get(local_id, [])
            theme = self.label_cluster(cluster_posts)
            store.set(global_id, theme)

        return store
