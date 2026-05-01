"""
Semantic embedding encoder with explicit CUDA control.

This module is responsible ONLY for:
- Loading a sentence embedding model
- Encoding text into fixed embeddings
- Attaching embeddings to Post objects

No stance logic.
No clustering logic.
No windowing logic.
"""

from typing import List, Iterable, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from sensemaking.data.schemas import Post


class EmbeddingEncoder:
    """
    Sentence-level semantic embedding encoder with explicit CUDA handling.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 64,
        device: Optional[str] = None,
        normalize: bool = True,
        require_cuda: bool = False,
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace/Sentence-Transformers model name.
        batch_size : int
            Batch size for encoding.
        device : str or None
            'cuda', 'cuda:0', 'cpu', or None to auto-detect.
        normalize : bool
            Whether to L2-normalize embeddings.
        require_cuda : bool
            If True, raise RuntimeError if CUDA is unavailable.
        """

        self.device = self._resolve_device(device, require_cuda)
        self.batch_size = batch_size
        self.normalize = normalize

        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.to(self.device)

        self._log_device_status()

    @staticmethod
    def _resolve_device(device: Optional[str], require_cuda: bool) -> str:
        """
        Resolve computation device with explicit CUDA checks.
        """
        if device is not None:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"Device '{device}' requested but CUDA is unavailable.")
            return device

        if torch.cuda.is_available():
            return "cuda"

        elif torch.backends.mps.is_available():
            return "mps"

        if require_cuda:
            raise RuntimeError("CUDA is required but not available.")

        return "cpu"

    def _log_device_status(self) -> None:
        """
        Log device status for reproducibility and debugging.
        """
        print("===== EmbeddingEncoder Device Status =====")
        print(f"Device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print("==========================================")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into semantic embeddings.

        Parameters
        ----------
        texts : List[str]
            Input texts.

        Returns
        -------
        np.ndarray
            Array of shape (N, d) containing embeddings.
        """
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings

    def __call__(self, posts: Iterable[Post]) -> List[Post]:
        """
        Encode and attach embeddings to Post objects.
        """
        return attach_embeddings(posts, self)



def attach_embeddings(
    posts: Iterable[Post],
    encoder: EmbeddingEncoder,
) -> List[Post]:
    """
    Compute and attach embeddings to Post objects.

    Parameters
    ----------
    posts : Iterable[Post]
        Collection of Post objects.
    encoder : EmbeddingEncoder
        Initialized embedding encoder.

    Returns
    -------
    List[Post]
        Posts with `.embedding` field populated.
    """
    posts = list(posts)

    texts = [post.text for post in posts]
    embeddings = encoder.encode_texts(texts)

    for post, emb in zip(posts, embeddings):
        post.embedding = emb

    return posts
