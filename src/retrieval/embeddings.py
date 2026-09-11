import time
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings


class EmbeddingEngine:
    """
    Configurable multilingual embedding engine.
    Supports pluggable embedding models (e.g. BAAI/bge-m3, paraphrase-multilingual-MiniLM-L12-v2).
    Ensures vector normalization for cosine similarity.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        print(f"[INFO] Initializing EmbeddingEngine with model: {self.model_name}")
        t0 = time.time()
        self.model = SentenceTransformer(self.model_name)
        t1 = time.time()
        self._dim = self.model.get_sentence_embedding_dimension()
        print(f"[INFO] Model {self.model_name} loaded in {t1 - t0:.2f}s (dim: {self._dim}).")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Embeds a list of document strings with L2 normalization.
        Returns np.ndarray of shape (len(texts), dim), float32.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string with L2 normalization.
        Returns np.ndarray of shape (1, dim), float32.
        """
        emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return emb.astype(np.float32)
