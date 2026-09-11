import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

from config.settings import settings
from .schemas import SearchResult, RetrievalMethod, LegalFilter
from .embeddings import EmbeddingEngine


class DenseIndex:
    """
    FAISS-based Dense Vector Index with cosine similarity.
    Maintains deterministic 1-to-1 mapping between vector index and chunk_id.
    Fully reloadable from disk without recomputing embeddings.
    """

    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = Path(index_dir or settings.dense_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.chunk_ids: List[str] = []
        self.metadata_by_id: Dict[str, Dict[str, Any]] = {}
        self.config: Dict[str, Any] = {}
        self.is_loaded = False

    def build_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embedding_engine: EmbeddingEngine,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """
        Builds FAISS index for all chunks and persists to disk.
        """
        t0 = time.time()
        print(f"[INFO] Building dense index for {len(chunks)} chunks using {embedding_engine.model_name}...")

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.metadata_by_id = {c["chunk_id"]: c for c in chunks}

        # Extract texts for embedding
        texts = [c["text"] for c in chunks]
        embeddings = embedding_engine.embed_documents(texts, batch_size=batch_size)

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

        self.config = {
            "model_name": embedding_engine.model_name,
            "dimension": dim,
            "total_vectors": self.faiss_index.ntotal,
            "metric": "inner_product_cosine",
            "build_timestamp": time.time(),
            "build_elapsed_seconds": round(time.time() - t0, 2)
        }

        self.save()
        self.is_loaded = True
        print(f"[SUCCESS] Dense index built with {self.faiss_index.ntotal} vectors in {time.time() - t0:.2f}s.")
        return self.config

    def save(self):
        """Persists index, chunk_ids, metadata, and config to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = self.index_dir / "faiss.index"
        faiss.write_index(self.faiss_index, str(faiss_path))

        # Save chunk IDs (deterministic mapping)
        chunk_ids_path = self.index_dir / "chunk_ids.json"
        with open(chunk_ids_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_ids, f, ensure_ascii=False, indent=2)

        # Save metadata lookup
        metadata_path = self.index_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_by_id, f, ensure_ascii=False, indent=2)

        # Save config
        config_path = self.index_dir / "index_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """Reloads index and metadata from disk without recomputing embeddings."""
        faiss_path = self.index_dir / "faiss.index"
        chunk_ids_path = self.index_dir / "chunk_ids.json"
        metadata_path = self.index_dir / "metadata.json"
        config_path = self.index_dir / "index_config.json"

        if not (faiss_path.exists() and chunk_ids_path.exists() and metadata_path.exists()):
            return False

        self.faiss_index = faiss.read_index(str(faiss_path))
        with open(chunk_ids_path, "r", encoding="utf-8") as f:
            self.chunk_ids = json.load(f)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata_by_id = json.load(f)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        self.is_loaded = True
        return True

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 30,
        filters: Optional[LegalFilter] = None
    ) -> List[SearchResult]:
        if not self.is_loaded or self.faiss_index is None:
            raise RuntimeError("DenseIndex is not loaded. Call load() or build_from_chunks() first.")

        # Ensure shape (1, dim) and float32
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        query_vec = query_vec.astype(np.float32)

        # Over-fetch if filters are active to allow post-filtering
        fetch_k = min(len(self.chunk_ids), top_k * 5 if (filters and filters.is_active()) else top_k)

        scores, indices = self.faiss_index.search(query_vec, fetch_k)
        scores = scores[0]
        indices = indices[0]

        results: List[SearchResult] = []
        rank = 1

        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[idx]
            meta = self.metadata_by_id.get(chunk_id)
            if not meta:
                continue

            if filters and filters.is_active():
                if not filters.matches(meta):
                    continue

            result = SearchResult(
                chunk_id=chunk_id,
                text=meta["text"],
                score=float(score),
                retrieval_method=RetrievalMethod.DENSE.value,
                rank=rank,
                source_file=meta["source_file"],
                document_title=meta["document_title"],
                page_start=meta["page_start"],
                page_end=meta["page_end"],
                rule_number=meta.get("rule_number"),
                section_number=meta.get("section_number"),
                sub_rule=meta.get("sub_rule"),
                clause=meta.get("clause"),
                schedule=meta.get("schedule"),
                document_type=meta.get("document_type"),
                document_date=meta.get("document_date"),
                extraction_method=meta.get("extraction_method"),
                ocr_required=meta.get("ocr_required", False),
                fallback_chunking=meta.get("fallback_chunking", False),
                dense_rank=rank,
                dense_score=float(score)
            )
            results.append(result)
            rank += 1
            if len(results) >= top_k:
                break

        return results
