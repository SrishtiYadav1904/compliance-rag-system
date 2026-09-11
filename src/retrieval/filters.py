from typing import Dict, Any, List, Optional
from .schemas import LegalFilter


def apply_filter(chunks: List[Dict[str, Any]], legal_filter: LegalFilter) -> List[Dict[str, Any]]:
    """Filters a list of chunks or metadata dictionaries by legal criteria."""
    if not legal_filter or not legal_filter.is_active():
        return chunks
    return [c for c in chunks if legal_filter.matches(c)]


def check_metadata_anomalies(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Diagnostic analysis of chunk metadata quality across the corpus.
    Identifies:
    - Percentage with rule_number, schedule, document_date, title, page numbers
    - Suspicious hierarchy assignments (e.g. 'Definitions' associated with Rule 1 instead of Rule 2)
    """
    total = len(chunks)
    if total == 0:
        return {"total_chunks": 0}

    with_rule = sum(1 for c in chunks if c.get("rule_number") is not None)
    with_schedule = sum(1 for c in chunks if c.get("schedule") is not None)
    with_date = sum(1 for c in chunks if c.get("document_date") is not None)
    with_title = sum(1 for c in chunks if c.get("document_title") is not None)
    with_valid_pages = sum(1 for c in chunks if c.get("page_start") and c.get("page_end"))

    suspicious_assignments = []
    for c in chunks:
        txt = c.get("text", "").lower()
        r = str(c.get("rule_number")) if c.get("rule_number") else ""

        # Case 1: Rule 1 chunk containing definition headings (in Legal Metrology, Rule 2 is Definitions)
        if r == "1" and ("definitions" in txt or "परिभाषाएं" in txt) and "short title" not in txt:
            suspicious_assignments.append({
                "chunk_id": c["chunk_id"],
                "source_file": c["source_file"],
                "assigned_rule": "1",
                "suspected_rule": "2 (Definitions)",
                "reason": "Chunk text contains definitions header but was indexed under Rule 1"
            })

        # Case 2: Rule 6 chunk containing wholesale provisions (Rule 24)
        if r == "6" and "provisions applicable to wholesale packages" in txt:
            suspicious_assignments.append({
                "chunk_id": c["chunk_id"],
                "source_file": c["source_file"],
                "assigned_rule": "6",
                "suspected_rule": "24",
                "reason": "Text mentions wholesale package chapter but has rule_number 6"
            })

    return {
        "total_chunks": total,
        "percentages": {
            "with_rule_number": round(with_rule / total * 100, 2),
            "with_schedule": round(with_schedule / total * 100, 2),
            "with_document_date": round(with_date / total * 100, 2),
            "with_document_title": round(with_title / total * 100, 2),
            "with_valid_page_numbers": round(with_valid_pages / total * 100, 2),
        },
        "suspicious_hierarchy_assignments_count": len(suspicious_assignments),
        "suspicious_assignments_sample": suspicious_assignments[:10]
    }
