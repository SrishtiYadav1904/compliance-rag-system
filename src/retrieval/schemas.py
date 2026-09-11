from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class RetrievalMethod(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    RERANK = "rerank"


class LegalFilter(BaseModel):
    rule_number: Optional[str] = None
    section_number: Optional[str] = None
    document_type: Optional[str] = None
    document_date: Optional[str] = None
    schedule: Optional[str] = None
    source_file: Optional[str] = None

    def is_active(self) -> bool:
        return any([
            self.rule_number,
            self.section_number,
            self.document_type,
            self.document_date,
            self.schedule,
            self.source_file
        ])

    def matches(self, meta: Dict[str, Any]) -> bool:
        if self.rule_number and str(meta.get("rule_number")) != str(self.rule_number):
            return False
        if self.section_number and str(meta.get("section_number")) != str(self.section_number):
            return False
        if self.document_type and meta.get("document_type") != self.document_type:
            return False
        if self.document_date and meta.get("document_date") != self.document_date:
            return False
        if self.schedule and meta.get("schedule") != self.schedule:
            return False
        if self.source_file and meta.get("source_file") != self.source_file:
            return False
        return True


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    retrieval_method: str
    rank: int
    source_file: str
    document_title: str
    page_start: int
    page_end: int
    rule_number: Optional[str] = None
    section_number: Optional[str] = None
    sub_rule: Optional[str] = None
    clause: Optional[str] = None
    schedule: Optional[str] = None
    document_type: Optional[str] = None
    document_date: Optional[str] = None
    extraction_method: Optional[str] = None
    ocr_required: bool = False
    fallback_chunking: bool = False

    # Diagnostic provenance
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class EvaluationQuery(BaseModel):
    query_id: str
    query: str
    language: str
    relevant_chunk_ids: List[str]
    topic: str
    description: Optional[str] = None
    is_ocr_typo: bool = False
