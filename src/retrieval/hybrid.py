from typing import List, Dict, Any, Optional
from config.settings import settings
from .schemas import SearchResult, RetrievalMethod, LegalFilter
from .embeddings import EmbeddingEngine
from .dense_index import DenseIndex
from .bm25_index import BM25Index
from .reranker import Reranker


class HybridRetriever:
    """
    Unified Hybrid Retrieval Engine implementing:
    - Dense Vector Search (cosine similarity)
    - BM25 Lexical Search (legal tokenization)
    - Reciprocal Rank Fusion (RRF) with configurable k
    - Optional Cross-Encoder Reranking
    - Full metadata and rank provenance preservation.
    """

    def __init__(
        self,
        dense_index: Optional[DenseIndex] = None,
        bm25_index: Optional[BM25Index] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        reranker: Optional[Reranker] = None,
        rrf_k: Optional[int] = None,
        dense_top_n: Optional[int] = None,
        bm25_top_n: Optional[int] = None
    ):
        self.dense_index = dense_index or DenseIndex()
        self.bm25_index = bm25_index or BM25Index()
        self.embedding_engine = embedding_engine
        self.reranker = reranker or Reranker()

        self.rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
        self.dense_top_n = dense_top_n or settings.dense_top_n
        self.bm25_top_n = bm25_top_n or settings.bm25_top_n

    def _ensure_loaded(self):
        if not self.dense_index.is_loaded:
            if not self.dense_index.load():
                raise RuntimeError(f"Dense index not found at {self.dense_index.index_dir}. Run build_dense_index.py first.")
        if not self.bm25_index.is_loaded:
            if not self.bm25_index.load():
                raise RuntimeError(f"BM25 index not found at {self.bm25_index.index_dir}. Run build_bm25_index.py first.")
        if self.embedding_engine is None:
            model_name = self.dense_index.config.get("model_name", settings.embedding_model)
            self.embedding_engine = EmbeddingEngine(model_name=model_name)

    def dense_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[LegalFilter] = None
    ) -> List[SearchResult]:
        self._ensure_loaded()
        q_emb = self.embedding_engine.embed_query(query)
        return self.dense_index.search(q_emb, top_k=top_k, filters=filters)

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[LegalFilter] = None
    ) -> List[SearchResult]:
        self._ensure_loaded()
        return self.bm25_index.search(query, top_k=top_k, filters=filters)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[LegalFilter] = None,
        use_reranker: bool = False
    ) -> List[SearchResult]:
        """
        Executes Dense + BM25, fuses with RRF, and optionally applies cross-encoder reranking.
        """
        self._ensure_loaded()

        # Step 1: Retrieve top N candidates from dense and BM25
        dense_results = self.dense_search(query, top_k=self.dense_top_n, filters=filters)
        bm25_results = self.bm25_search(query, top_k=self.bm25_top_n, filters=filters)

        # Step 2: Reciprocal Rank Fusion
        # RRF_Score(d) = sum( 1 / (k + r_i(d)) )
        fused_items: Dict[str, Dict[str, Any]] = {}

        for rank, res in enumerate(dense_results, 1):
            cid = res.chunk_id
            if cid not in fused_items:
                fused_items[cid] = {
                    "base_result": res,
                    "rrf_score": 0.0,
                    "dense_rank": rank,
                    "dense_score": res.score,
                    "bm25_rank": None,
                    "bm25_score": None
                }
            fused_items[cid]["dense_rank"] = rank
            fused_items[cid]["dense_score"] = res.score
            fused_items[cid]["rrf_score"] += 1.0 / (self.rrf_k + rank)

        for rank, res in enumerate(bm25_results, 1):
            cid = res.chunk_id
            if cid not in fused_items:
                fused_items[cid] = {
                    "base_result": res,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": rank,
                    "bm25_score": res.score
                }
            fused_items[cid]["bm25_rank"] = rank
            fused_items[cid]["bm25_score"] = res.score
            fused_items[cid]["rrf_score"] += 1.0 / (self.rrf_k + rank)

        # Sort by RRF score descending
        sorted_fused = sorted(fused_items.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Step 3: Build SearchResult list with diagnostic ranks
        candidate_results: List[SearchResult] = []
        for idx, item in enumerate(sorted_fused, 1):
            base: SearchResult = item["base_result"]
            res = SearchResult(
                chunk_id=base.chunk_id,
                text=base.text,
                score=round(item["rrf_score"], 6),
                retrieval_method=RetrievalMethod.HYBRID.value,
                rank=idx,
                source_file=base.source_file,
                document_title=base.document_title,
                page_start=base.page_start,
                page_end=base.page_end,
                rule_number=base.rule_number,
                section_number=base.section_number,
                sub_rule=base.sub_rule,
                clause=base.clause,
                schedule=base.schedule,
                document_type=base.document_type,
                document_date=base.document_date,
                extraction_method=base.extraction_method,
                ocr_required=base.ocr_required,
                fallback_chunking=base.fallback_chunking,
                dense_rank=item["dense_rank"],
                dense_score=item["dense_score"],
                bm25_rank=item["bm25_rank"],
                bm25_score=item["bm25_score"],
                rrf_score=round(item["rrf_score"], 6)
            )
            candidate_results.append(res)

        # Step 4: Optional Cross-Encoder Reranking
        if use_reranker and self.reranker:
            # Send top candidates (e.g. top 40) to reranker, return final top_k
            rerank_candidates = candidate_results[:settings.hybrid_top_n]
            return self.reranker.rerank(query, rerank_candidates, top_k=top_k)

        return candidate_results[:top_k]
