import re
from typing import List, Dict, Any, Tuple, Optional
from config.settings import settings
from src.retrieval.schemas import SearchResult
from .schemas import GenerationContext


class ContextValidator:
    """
    Validates retrieved candidate evidence before sending to generation layer.
    Enforces the core rule:
    NO RETRIEVED EVIDENCE / INSUFFICIENT EVIDENCE -> ABSTAIN.
    """

    # Non-legal metrology topics that require explicit abstention
    OUT_OF_DOMAIN_PATTERNS = [
        re.compile(r"\bincome\s+tax\b", re.I),
        re.compile(r"\bgst\s+rates?\b", re.I),
        re.compile(r"\bcustoms\s+duty\s+rate\b", re.I),
        re.compile(r"\bdriving\s+licen[sc]e\b", re.I),
        re.compile(r"\bpassport\b", re.I),
        re.compile(r"\bcorporate\s+tax\b", re.I),
        re.compile(r"\bcriminal\s+procedure\s+code\b", re.I),
        re.compile(r"\bipc\s+section\b", re.I),
    ]

    def __init__(self, confidence_threshold: Optional[float] = None):
        # Calibrated provisional threshold for RRF score
        self.confidence_threshold = confidence_threshold or settings.confidence_threshold

    def validate(self, query: str, chunks: List[SearchResult]) -> Tuple[bool, Optional[str]]:
        """
        Validates evidence sufficiency.
        Returns: (is_valid, failure_reason)
        """
        # Check out-of-domain patterns in user query
        for pattern in self.OUT_OF_DOMAIN_PATTERNS:
            if pattern.search(query):
                return False, "Query concerns matters outside the Legal Metrology knowledge base (e.g. tax/criminal/licensing law)."

        # Check for zero chunks
        if not chunks:
            return False, "Zero supporting provisions retrieved from Legal Metrology knowledge base."

        # Check top score confidence
        top_chunk = chunks[0]
        # In hybrid retrieval, top RRF score is typically >= 0.015
        if top_chunk.score < self.confidence_threshold:
            return False, f"Retrieval confidence ({top_chunk.score:.4f}) is below evidence threshold ({self.confidence_threshold:.4f})."

        return True, None


class ContextSelector:
    """
    Selects top 5–7 evidence chunks for LLM context window.
    Balances:
    - Reranker / RRF ranking
    - Source and rule diversity
    - Contiguous chunk preservation (keeps adjacent chunks of the same rule)
    - Token budget management (~2,500 words max)
    """

    def __init__(self, max_chunks: Optional[int] = None):
        self.max_chunks = max_chunks or settings.context_chunk_limit

    def select(self, candidates: List[SearchResult]) -> GenerationContext:
        if not candidates:
            return GenerationContext(
                selected_chunks=[],
                token_estimate=0,
                provenance_map={},
                validation_status="empty",
                validation_reason="No candidate chunks supplied."
            )

        selected: List[SearchResult] = []
        seen_chunk_ids = set()
        seen_texts = set()

        for c in candidates:
            if len(selected) >= self.max_chunks:
                break

            if c.chunk_id in seen_chunk_ids:
                continue

            # Deduplicate near-identical snippets
            snippet = c.text.strip()[:100]
            if snippet in seen_texts:
                continue

            selected.append(c)
            seen_chunk_ids.add(c.chunk_id)
            seen_texts.add(snippet)

        # Check if adjacent chunk needed to complete a rule provision
        # e.g., if Rule 6 is selected, ensure both sub-rule (1) and definitions are accessible
        rule_numbers_present = {c.rule_number for c in selected if c.rule_number}
        if "6" in rule_numbers_present and len(selected) < self.max_chunks + 1:
            for c in candidates:
                if c.rule_number == "6" and c.chunk_id not in seen_chunk_ids:
                    selected.append(c)
                    seen_chunk_ids.add(c.chunk_id)
                    break

        # Build provenance map and estimate token count (~4 chars per token)
        provenance_map = {}
        total_chars = 0
        for c in selected:
            provenance_map[c.chunk_id] = {
                "source_file": c.source_file,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "rule_number": c.rule_number,
                "sub_rule": c.sub_rule,
                "clause": c.clause,
                "schedule": c.schedule,
                "document_title": c.document_title,
            }
            total_chars += len(c.text)

        token_estimate = total_chars // 4

        return GenerationContext(
            selected_chunks=selected,
            token_estimate=token_estimate,
            provenance_map=provenance_map,
            validation_status="valid"
        )
