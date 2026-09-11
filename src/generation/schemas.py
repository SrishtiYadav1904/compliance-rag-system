from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.retrieval.schemas import SearchResult


class Citation(BaseModel):
    """Structured legal citation referencing authoritative KB source."""
    source_file: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    rule_number: Optional[str] = None
    sub_rule: Optional[str] = None
    clause: Optional[str] = None
    schedule: Optional[str] = None
    document_title: Optional[str] = None
    verified: bool = False
    raw_citation: str = ""

    def formatted(self) -> str:
        parts = [f"Source: {self.source_file}"]
        if self.page_start:
            if self.page_end and self.page_end != self.page_start:
                parts.append(f"pp. {self.page_start}–{self.page_end}")
            else:
                parts.append(f"p. {self.page_start}")
        if self.rule_number:
            r_str = f"Rule {self.rule_number}"
            if self.sub_rule:
                r_str += f"({self.sub_rule})"
            if self.clause:
                r_str += f"({self.clause})"
            parts.append(r_str)
        if self.schedule:
            parts.append(f"Schedule: {self.schedule}")
        return f"[{', '.join(parts)}]"


class GenerationContext(BaseModel):
    """Encapsulates retrieved and validated evidence chunks supplied to LLM."""
    selected_chunks: List[SearchResult] = Field(default_factory=list)
    token_estimate: int = 0
    provenance_map: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    validation_status: str = "valid"
    validation_reason: Optional[str] = None


class GroundedResponse(BaseModel):
    """
    Structured grounded generation output schema.
    Guarantees strict auditability, citation provenance, and latency breakdown.
    """
    answer: str
    language: str
    original_query: str
    retrieval_query: str
    translation_used: bool
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    citation_verified: bool = True
    grounded: bool = True
    abstained: bool = False
    abstention_reason: Optional[str] = None
    latencies: Dict[str, float] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
