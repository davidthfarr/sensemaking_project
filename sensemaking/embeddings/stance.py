"""
Zero-shot stance labeling using a locally hosted NLI model.

Stance mapping:
    +1 = Support / Affirm
    -1 = Oppose / Reject
     0 = Neutral / Unclear
"""

from typing import Iterable, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from sensemaking.data.schemas import Post


class ZeroShotStanceLabeler:
    """
    Zero-shot stance classifier using an NLI model.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
        batch_size: int = 16,
        require_cuda: bool = False,
    ):
        self.device = self._resolve_device(device, require_cuda)
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.label2id = self.model.config.label2id
        self.entail_idx = self.label2id["entailment"]

        self._log_device_status()

    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: Optional[str], require_cuda: bool) -> torch.device:
        if device is not None:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"CUDA requested but unavailable: {device}")
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        if require_cuda:
            raise RuntimeError("CUDA is required but not available.")

        return torch.device("cpu")

    def _log_device_status(self) -> None:
        print("===== ZeroShotStanceLabeler Device Status =====")
        print(f"Device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print("==============================================")

    # ------------------------------------------------------------------
    # Stance logic
    # ------------------------------------------------------------------

    def _build_hypotheses(self) -> List[str]:
        """
        Hypotheses must be concrete claims for NLI.
        """
        return [
            "Russia's invasion of Ukraine is justified.",
            "Russia's invasion of Ukraine is not justified.",
            "The post does not take a clear position on whether Russia's invasion of Ukraine is justified.",
        ]

    def predict_batch(self, texts: List[str], show_progress: bool = True) -> List[int]:
        """
        Predict stance labels for a list of texts.
        """
        hypotheses = self._build_hypotheses()
        results: List[int] = []

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Stance labeling", leave=False)

        for start_idx in iterator:
            batch = texts[start_idx : start_idx + self.batch_size]

            inputs = self.tokenizer(
                [t for t in batch for _ in hypotheses],
                hypotheses * len(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits, dim=-1)
            probs = probs.view(len(batch), len(hypotheses), -1)

            for p in probs:
                entail_scores = [
                    p[0, self.entail_idx],  # entails justification
                    p[1, self.entail_idx],  # entails not justified
                    p[2, self.entail_idx],  # entails neutral
                ]

                label = int(torch.argmax(torch.stack(entail_scores)))

                if label == 0:
                    results.append(1)
                elif label == 1:
                    results.append(-1)
                else:
                    results.append(0)

        return results

    # ------------------------------------------------------------------
    # Pipeline integration
    # ------------------------------------------------------------------

    def __call__(self, posts: Iterable[Post]) -> List[Post]:
        return attach_stance(posts, self)


def attach_stance(
    posts: Iterable[Post],
    labeler: ZeroShotStanceLabeler,
) -> List[Post]:
    """
    Attach stance labels to Post objects.
    """
    posts = list(posts)
    texts = [p.text for p in posts]

    stances = labeler.predict_batch(texts)

    for post, stance in zip(posts, stances):
        post.stance = stance

    return posts
