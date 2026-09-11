import re
import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from config.settings import settings
from .schemas import SearchResult, RetrievalMethod, LegalFilter


class LegalBM25Tokenizer:
    """
    Legal-aware tokenizer preserving:
    - Rule numbers and sub-rules: "Rule 6", "Rule 6(1)", "Rule 6(3)"
    - Schedules: "Schedule II", "Schedule I"
    - Legal terms: "MRP", "Maximum Retail Price", "net quantity", "consumer care", "country of origin"
    - Measurement units: "kg", "g", "l", "ml", "cm", "mm"
    - Devanagari (Hindi/Marathi) and Tamil unicode tokens.
    """

    def __init__(self):
        # Key phrase normalization mappings
        self.phrase_mappings = [
            (re.compile(r"\bmaximum\s+retail\s+price\b", re.I), "maximum_retail_price mrp"),
            (re.compile(r"\bnet\s+quantity\b", re.I), "net_quantity"),
            (re.compile(r"\bconsumer\s+care\b", re.I), "consumer_care"),
            (re.compile(r"\bcountry\s+of\s+origin\b", re.I), "country_of_origin"),
            (re.compile(r"\bprincipal\s+display\s+panel\b", re.I), "principal_display_panel"),
            (re.compile(r"\bwholesale\s+package\b", re.I), "wholesale_package"),
            (re.compile(r"\bretail\s+sale\s+price\b", re.I), "retail_sale_price"),
            (re.compile(r"\bpackaged\s+commodities\b", re.I), "packaged_commodities"),
            (re.compile(r"\blegal\s+metrology\b", re.I), "legal_metrology"),
            (re.compile(r"\brule\s+(\d+[A-Za-z]?)\s*\(\s*(\d+[A-Za-z]?)\s*\)", re.I), r"rule_\1 rule_\1_\2"),
            (re.compile(r"\brule\s+(\d+[A-Za-z]?)\b", re.I), r"rule_\1"),
            (re.compile(r"\bschedule\s+([ivx0-9]+)\b", re.I), r"schedule_\1"),
            (re.compile(r"\bअनुसूची\s+([ivx०-९0-9]+)\b", re.I), r"अनुसूची_\1"),
            (re.compile(r"\bनियम\s+([०-९0-9]+)\b", re.I), r"नियम_\1"),
        ]

        # Token pattern: legal identifiers, English words, Devanagari, Tamil, numbers
        self.token_regex = re.compile(
            r"[a-z0-9]+(?:_[a-z0-9]+)*|"
            r"[\u0900-\u097f]+|"
            r"[\u0b80-\u0bff]+|"
            r"[a-z]+|"
            r"\d+",
            re.IGNORECASE
        )

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        processed = text.lower()
        for pattern, replacement in self.phrase_mappings:
            processed = pattern.sub(replacement, processed)

        tokens = self.token_regex.findall(processed)
        return [t.lower() for t in tokens if len(t) > 1 or t.isdigit()]


class BM25Index:
    """
    BM25 Lexical Retrieval Index using rank_bm25.
    Persists tokenized representations and model to data/indices/bm25/.
    Fully reloadable without re-tokenizing.
    """

    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = Path(index_dir or settings.bm25_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = LegalBM25Tokenizer()
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.metadata_by_id: Dict[str, Dict[str, Any]] = {}
        self.config: Dict[str, Any] = {}
        self.is_loaded = False

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        print(f"[INFO] Building BM25 index for {len(chunks)} chunks...")

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.metadata_by_id = {c["chunk_id"]: c for c in chunks}

        tokenized_corpus = []
        for c in chunks:
            # Combine title, rule number, and text to enrich lexical surface
            doc_context = f"{c.get('document_title', '')} Rule {c.get('rule_number', '')} {c['text']}"
            tokenized_corpus.append(self.tokenizer.tokenize(doc_context))

        self.bm25 = BM25Okapi(tokenized_corpus)

        self.config = {
            "total_documents": len(tokenized_corpus),
            "tokenizer": "LegalBM25Tokenizer",
            "build_timestamp": time.time(),
            "build_elapsed_seconds": round(time.time() - t0, 2)
        }

        self.save()
        self.is_loaded = True
        print(f"[SUCCESS] BM25 index built for {len(self.chunk_ids)} documents in {time.time() - t0:.2f}s.")
        return self.config

    def save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save BM25 model
        bm25_path = self.index_dir / "bm25_index.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

        # Save chunk IDs
        chunk_ids_path = self.index_dir / "chunk_ids.json"
        with open(chunk_ids_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_ids, f, ensure_ascii=False, indent=2)

        # Save metadata
        metadata_path = self.index_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_by_id, f, ensure_ascii=False, indent=2)

        # Save config
        config_path = self.index_dir / "index_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        bm25_path = self.index_dir / "bm25_index.pkl"
        chunk_ids_path = self.index_dir / "chunk_ids.json"
        metadata_path = self.index_dir / "metadata.json"
        config_path = self.index_dir / "index_config.json"

        if not (bm25_path.exists() and chunk_ids_path.exists() and metadata_path.exists()):
            return False

        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
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
        query: str,
        top_k: int = 30,
        filters: Optional[LegalFilter] = None
    ) -> List[SearchResult]:
        if not self.is_loaded or self.bm25 is None:
            raise RuntimeError("BM25Index is not loaded. Call load() or build_from_chunks() first.")

        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)

        # Over-fetch if filters are active
        fetch_k = min(len(self.chunk_ids), top_k * 5 if (filters and filters.is_active()) else top_k)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:fetch_k]

        results: List[SearchResult] = []
        rank = 1

        for idx in top_indices:
            score = float(doc_scores[idx])
            if score <= 0.0:
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
                score=score,
                retrieval_method=RetrievalMethod.BM25.value,
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
                bm25_rank=rank,
                bm25_score=score
            )
            results.append(result)
            rank += 1
            if len(results) >= top_k:
                break

        return results
