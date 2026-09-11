import time
from typing import List, Optional
from config.settings import settings
from .schemas import SearchResult, RetrievalMethod


class Reranker:
    """
    Configurable cross-encoder reranker for retrieved candidate chunks.
    Reranks only top N candidates (e.g. top 40 -> top 10).
    Provides graceful fallback to RRF score order if cross-encoder cannot run.
    """

    def __init__(self, model_name: Optional[str] = None, enabled: bool = True):
        self.model_name = model_name or settings.reranker_model
        self.enabled = enabled
        self._model = None
        self._load_attempted = False

    def _load_model(self):
        if self._load_attempted or not self.enabled:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder
            print(f"[INFO] Initializing CrossEncoder reranker: {self.model_name}...")
            t0 = time.time()
            self._model = CrossEncoder(self.model_name)
            print(f"[SUCCESS] Reranker {self.model_name} loaded in {time.time() - t0:.2f}s.")
        except Exception as e:
            print(f"[WARN] Could not load reranker model '{self.model_name}' ({e}). Falling back to RRF rank order.")
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 10
    ) -> List[SearchResult]:
        if not candidates:
            return []

        if not self.enabled:
            return candidates[:top_k]

        self._load_model()

        if self._model is None:
            # Fallback to current score/rank
            return candidates[:top_k]

        try:
            # Create (query, doc_text) pairs for top candidates
            pairs = [[query, c.text] for c in candidates]
            scores = self._model.predict(pairs)

            # Assign reranker scores and sort
            for c, s in zip(candidates, scores):
                c.rerank_score = float(s)

            reranked = sorted(candidates, key=lambda x: x.rerank_score or -999.0, reverse=True)[:top_k]

            # Update final ranks and method
            for idx, r in enumerate(reranked, 1):
                r.rank = idx
                r.score = r.rerank_score or r.score
                r.retrieval_method = RetrievalMethod.RERANK.value

            return reranked

        except Exception as e:
            print(f"[WARN] Reranking inference failed ({e}). Returning original candidate order.")
            return candidates[:top_k]
