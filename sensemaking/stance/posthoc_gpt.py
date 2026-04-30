"""
Post-hoc stance classification against stationary cluster themes.

Two backends are provided with identical interfaces so callers can swap them:

  PosthocGPTStanceClassifier
      Classifies batches of posts via the OpenAI chat API.
      Uses a JSON-array response for efficiency; falls back to single-post on parse failure.

  LocalLlamaClassifier
      Classifies posts locally using meta-llama/Meta-Llama-3-8B-Instruct via
      HuggingFace transformers. CUDA is used if available; falls back to CPU.
      Requires 'transformers' and 'torch'; model weights require HF access to
      meta-llama/Meta-Llama-3-8B-Instruct (run `huggingface-cli login` first).

Both classes expose:
  classify_posts(posts: List[Post], theme: str) -> List[Post]
  __call__(posts, theme) -> List[Post]

Stance labels: support / oppose / neutral
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

_SINGLE_SYSTEM = (
    "You are a stance classifier. "
    "Reply with exactly one word: SUPPORT, OPPOSE, or NEUTRAL."
)


# ---------------------------------------------------------------------------
# GPT classifier
# ---------------------------------------------------------------------------

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
    api_key : str or None
        OpenAI API key. If None, the client reads OPENAI_API_KEY from the environment.
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

        return [self._classify_single(t, theme) for t in texts]

    def _classify_single(self, text: str, theme: str) -> str:
        prompt = (
            f"Does this post SUPPORT, OPPOSE, or take a NEUTRAL stance toward "
            f"the following claim: '{theme}'?\n"
            f"Reply with exactly one word: SUPPORT, OPPOSE, or NEUTRAL.\n\n"
            f"Post: {text[:500]}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SINGLE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            word = resp.choices[0].message.content.strip().lower()
            if word in STANCE_LABELS:
                return word
            for label in ("support", "oppose", "neutral"):
                if label in word:
                    return label
        except Exception:
            pass
        return "neutral"

    def classify_posts(self, posts: List[Post], theme: str) -> List[Post]:
        """
        Classify all posts against a cluster theme and attach stance labels.

        Parameters
        ----------
        posts : List[Post]
            Posts to classify. Should have .text set.
        theme : str
            Stationary theme label or topic claim for the cluster.

        Returns
        -------
        List[Post]
            Same posts with .stance set to 'support', 'oppose', or 'neutral'.
        """
        for start in tqdm(
            range(0, len(posts), self.batch_size),
            desc="GPT stance",
            position=0,
            leave=True,
        ):
            batch = posts[start: start + self.batch_size]
            labels = self._classify_batch([p.text for p in batch], theme)
            for post, label in zip(batch, labels):
                post.stance = label

        return posts

    def __call__(self, posts: Iterable[Post], theme: str) -> List[Post]:
        return self.classify_posts(list(posts), theme)


# ---------------------------------------------------------------------------
# Local Llama classifier
# ---------------------------------------------------------------------------

class LocalLlamaClassifier:
    """
    Classifies posts locally using meta-llama/Meta-Llama-3-8B-Instruct.

    Requires HuggingFace access to meta-llama/Meta-Llama-3-8B-Instruct:
      huggingface-cli login

    The model is loaded with device_map="auto" so transformers automatically
    shards it across all available GPUs (or falls back to CPU if none).
    float16 is used to reduce memory footprint and speed up inference.

    Parameters
    ----------
    batch_size : int
        Prompts processed per forward pass. 32 works well across two GPUs.
    """

    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

    _SYSTEM = (
        "You are a stance classifier. "
        "Reply with exactly one word: SUPPORT, OPPOSE, or NEUTRAL."
    )

    def __init__(self, batch_size: int = 32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()

        self._torch = torch
        self.batch_size = batch_size

    def _build_prompt(self, text: str, claim: str) -> str:
        return (
            f"Does this post SUPPORT, OPPOSE, or take a NEUTRAL stance toward "
            f"the following claim: '{claim}'?\n"
            f"Reply with exactly one word: SUPPORT, OPPOSE, or NEUTRAL.\n\n"
            f"Post: {text[:500]}"
        )

    def _parse_label(self, output: str) -> str:
        upper = output.upper()
        for label in ("SUPPORT", "OPPOSE", "NEUTRAL"):
            if label in upper:
                return label.lower()
        return "neutral"

    def _classify_batch(self, texts: List[str], claim: str) -> List[str]:
        messages_list = [
            [
                {"role": "system", "content": self._SYSTEM},
                {"role": "user",   "content": self._build_prompt(text, claim)},
            ]
            for text in texts
        ]

        prompts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msgs in messages_list
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        labels = []
        for seq in outputs:
            new_tokens = seq[input_len:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            labels.append(self._parse_label(decoded))
        return labels

    def classify_posts(self, posts: List[Post], theme: str) -> List[Post]:
        """
        Classify all posts against a claim and attach stance labels.

        Parameters
        ----------
        posts : List[Post]
            Posts to classify. Should have .text set.
        theme : str
            Stationary theme label or topic claim.

        Returns
        -------
        List[Post]
            Same posts with .stance set to 'support', 'oppose', or 'neutral'.
        """
        for start in tqdm(
            range(0, len(posts), self.batch_size),
            desc="Llama stance",
            position=0,
            leave=True,
        ):
            batch = posts[start: start + self.batch_size]
            labels = self._classify_batch([p.text for p in batch], theme)
            for post, label in zip(batch, labels):
                post.stance = label

        return posts

    def __call__(self, posts: Iterable[Post], theme: str) -> List[Post]:
        return self.classify_posts(list(posts), theme)
