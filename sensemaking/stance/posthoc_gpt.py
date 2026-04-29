"""
Post-hoc GPT stance classification against stationary cluster themes.

Classifies each post as support / oppose / neutral relative to the cluster's
fixed theme label. Must be called after clustering and theme assignment are
complete — this module has no knowledge of embeddings or cluster structure.
"""

import json
import time
from typing import Iterable, List, Optional

from openai import OpenAI, RateLimitError
from tqdm import tqdm

from sensemaking.data.schemas import Post

STANCE_LABELS = frozenset({"support", "oppose", "neutral"})

_SYSTEM_PROMPT = """\
You are a stance classifier for social media posts.

Given a narrative claim and a numbered list of posts, classify each post's \
stance toward the claim. Reply with a JSON object in this exact format:
{"stances": ["support", "neutral", "oppose", ...]}

Rules:
- support: the post affirms, endorses, agrees with, or spreads the claim
- oppose: the post rejects, counters, disputes, or contradicts the claim
- neutral: the post is unrelated to the claim, ambiguous, or takes no clear position

Return one label per post, in the same order as the input. Use only the \
words "support", "oppose", or "neutral".\
"""


class PosthocGPTStanceClassifier:
    """
    Classifies posts against a stationary cluster theme using the OpenAI chat API.

    Parameters
    ----------
    model : str
        OpenAI model ID. gpt-4o-mini balances cost and accuracy.
    batch_size : int
        Number of posts per API call.
    max_retries : int
        Retry attempts on rate-limit or malformed response.
    retry_delay : float
        Base wait (seconds) between retries; multiplied by attempt number.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        batch_size: int = 20,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        api_key: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Internal API helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, texts: List[str], theme: str) -> str:
        numbered = "\n".join(f"{i + 1}. {t[:500]}" for i, t in enumerate(texts))
        return f'Narrative claim: "{theme}"\n\nPosts:\n{numbered}'

    def _parse_labels(self, content: str, expected_n: int) -> Optional[List[str]]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None

        labels = parsed if isinstance(parsed, list) else parsed.get("stances")
        if not isinstance(labels, list) or len(labels) != expected_n:
            return None

        normalized = [str(l).lower().strip() for l in labels]
        if not all(l in STANCE_LABELS for l in normalized):
            return None

        return normalized

    def _call_api(self, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return response.choices[0].message.content

    def _classify_batch(self, texts: List[str], theme: str) -> List[str]:
        user_msg = self._build_user_message(texts, theme)

        for attempt in range(self.max_retries):
            try:
                content = self._call_api(user_msg)
                labels = self._parse_labels(content, len(texts))
                if labels is not None:
                    return labels
            except RateLimitError:
                time.sleep(self.retry_delay * (attempt + 1))
                continue
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)

        # Batch failed: fall back to one post at a time
        return [self._classify_single(t, theme) for t in texts]

    def _classify_single(self, text: str, theme: str) -> str:
        user_msg = self._build_user_message([text], theme)
        try:
            content = self._call_api(user_msg)
            labels = self._parse_labels(content, 1)
            if labels:
                return labels[0]
        except Exception:
            pass
        return "neutral"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify_posts(self, posts: List[Post], theme: str) -> List[Post]:
        """
        Classify all posts against a cluster theme and attach stance labels.

        Parameters
        ----------
        posts : List[Post]
            Posts to classify. Each post should already have cluster_id assigned.
        theme : str
            Stationary theme label for the cluster (generated at cluster birth).

        Returns
        -------
        List[Post]
            Same posts with .stance set to 'support', 'oppose', or 'neutral'.
        """
        for start in tqdm(
            range(0, len(posts), self.batch_size),
            desc="GPT stance classification",
        ):
            batch = posts[start : start + self.batch_size]
            labels = self._classify_batch([p.text for p in batch], theme)
            for post, label in zip(batch, labels):
                post.stance = label

        return posts

    def __call__(self, posts: Iterable[Post], theme: str) -> List[Post]:
        return self.classify_posts(list(posts), theme)
