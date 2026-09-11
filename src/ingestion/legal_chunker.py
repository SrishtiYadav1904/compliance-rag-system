import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple


class LegalChunker:
    """
    Legal-aware chunker for Indian Legal Metrology statutory documents.
    Detects:
      - Rule (e.g., Rule 6, Rule 18, 6. Declarations...)
      - Sub-rule (e.g., (1), (2), (4A))
      - Clause (e.g., (a), (b), (e))
      - Sub-clause (e.g., (i), (ii), (iv))
      - Schedule (e.g., THE FIRST SCHEDULE, SCHEDULE II)
      - Section (e.g., Section 36, 18. Declarations on pre-packaged...)
    Preserves exact page start and page end numbers.
    Falls back gracefully to paragraph-based chunking if legal hierarchy is absent,
    explicitly marking fallback_chunking=True.
    """

    def __init__(
        self,
        min_chunk_chars: int = 60,
        max_chunk_chars: int = 2200,
        overlap_chars: int = 150
    ):
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

        # Patterns for English and Hindi legal structural elements
        self.schedule_pattern = re.compile(
            r"(?:^|\n)\s*(?:(?:THE\s+)?([A-Z]+|\d+(?:ST|ND|RD|TH)?)\s+SCHEDULE|"
            r"SCHEDULE\s+([I|V|X]+|\d+)|"
            r"(?:(?:पहली|दूसरी|तीसरी|चौथी|पांचवीं|छठी|सातवीं|आठवीं)\s+अनुसूची)|"
            r"(?:अनुसूची\s+([I|V|X]+|[०-९0-9]+)))",
            re.IGNORECASE
        )

        self.rule_pattern = re.compile(
            r"(?:^|\n)\s*(?:(?:Rule|िनयम|नियम)\s+)?([0-9]{1,3}[A-Z]?|[०-९]{1,3})\.\s+([^\n\.\-]{3,90})",
            re.MULTILINE
        )

        self.section_pattern = re.compile(
            r"(?:^|\n)\s*(?:(?:Section|धारा)\s+)([0-9]{1,3}[A-Z]?)\.\s+([^\n\.\-]{3,90})",
            re.IGNORECASE
        )

        self.subrule_pattern = re.compile(
            r"(?:^|\n)\s*\(([0-9]{1,2}[A-Z]?|[०-९]{1,2})\)\s+"
        )

        self.clause_pattern = re.compile(
            r"(?:^|\n)\s*\(([a-z]{1,2}|[क-ह]{1,2})\)\s+"
        )

    def chunk_document(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        document_id = doc_data["document_id"]
        source_file = doc_data["source_file"]
        document_title = doc_data["document_title"]
        pages = doc_data.get("pages", [])

        if not pages:
            return []

        doc_type = self._detect_document_type(document_title, source_file)
        doc_date = self._detect_document_date(doc_data)

        # First attempt: Structural / Legal hierarchy parsing
        chunks = self._chunk_by_legal_hierarchy(
            pages, document_id, source_file, document_title, doc_type, doc_date
        )

        # If structural parser found zero or minimal chunks relative to page count, fallback to paragraph chunking
        if not chunks or len(chunks) < (len(pages) // 3):
            chunks = self._chunk_fallback_paragraphs(
                pages, document_id, source_file, document_title, doc_type, doc_date
            )

        return chunks

    def _detect_document_type(self, title: str, filename: str) -> str:
        s = f"{title} {filename}".lower()
        if "jan vishwas" in s or "jv act" in s or "act 2009" in s or "act, 2009" in s:
            return "Statutory Act"
        elif "amendment" in s:
            return "Amendment Rules"
        elif "advisory" in s:
            return "Advisory"
        elif "sop" in s or "standard operating procedure" in s:
            return "Standard Operating Procedure"
        elif "guideline" in s:
            return "Guidelines"
        elif "corrigendum" in s:
            return "Corrigendum"
        elif "rules" in s:
            return "Rules"
        else:
            return "Gazette Notification"

    def _detect_document_date(self, doc_data: Dict[str, Any]) -> Optional[str]:
        # Check filename for YYYY.MM.DD or YYYY
        fn = doc_data.get("source_file", "")
        m_date = re.search(r"(20[0-2][0-9])[\.\-_]([0-1]?[0-9])[\.\-_]([0-3]?[0-9])", fn)
        if m_date:
            y, m, d = m_date.groups()
            return f"{y}-{int(m):02d}-{int(d):02d}"

        m_year = re.search(r"(20[0-2][0-9])", fn)
        if m_year:
            return f"{m_year.group(1)}-01-01"

        # Check first page text for date patterns
        pages = doc_data.get("pages", [])
        if pages:
            t0 = pages[0].get("text", "")
            m_txt = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(20[0-2][0-9])", t0, re.IGNORECASE)
            if m_txt:
                day, month_str, yr = m_txt.groups()
                months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
                try:
                    mo = months.index(month_str.lower()) + 1
                    return f"{yr}-{mo:02d}-{int(day):02d}"
                except ValueError:
                    pass
        return None

    def _chunk_by_legal_hierarchy(
        self,
        pages: List[Dict[str, Any]],
        document_id: str,
        source_file: str,
        document_title: str,
        doc_type: str,
        doc_date: Optional[str]
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        active_schedule: Optional[str] = None
        active_rule: Optional[str] = None
        active_section: Optional[str] = None
        active_subrule: Optional[str] = None
        active_clause: Optional[str] = None

        current_text_parts: List[str] = []
        page_start: Optional[int] = None
        page_end: Optional[int] = None
        has_ocr: bool = False
        methods: set = set()

        def flush_chunk():
            nonlocal current_text_parts, page_start, page_end, has_ocr, methods
            if not current_text_parts:
                return
            combined_text = "\n".join(current_text_parts).strip()
            if len(combined_text) < self.min_chunk_chars:
                return

            chunk_idx = len(chunks) + 1
            chunk_id = f"{document_id}_c{chunk_idx:04d}"
            primary_method = "ocr" if has_ocr and "ocr" in methods else "text"

            chunk = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source_file": source_file,
                "document_title": document_title,
                "page_start": page_start if page_start is not None else 1,
                "page_end": page_end if page_end is not None else (page_start or 1),
                "text": combined_text,
                "rule_number": active_rule,
                "section_number": active_section,
                "sub_rule": active_subrule,
                "clause": active_clause,
                "schedule": active_schedule,
                "document_type": doc_type,
                "document_date": doc_date,
                "extraction_method": primary_method,
                "ocr_required": has_ocr,
                "fallback_chunking": False
            }
            chunks.append(chunk)
            current_text_parts = []
            page_start = None
            page_end = None
            has_ocr = False
            methods = set()

        for page in pages:
            p_num = page["page_number"]
            p_text = page.get("text", "")
            p_method = page.get("extraction_method", "text")
            is_p_ocr = (p_method == "ocr")

            lines = p_text.split("\n")
            for line in lines:
                raw_line = line.strip()
                if not raw_line:
                    continue

                # Check Schedule boundary
                sched_match = self.schedule_pattern.search(raw_line)
                if sched_match:
                    flush_chunk()
                    active_schedule = sched_match.group(0).strip()
                    active_rule = None
                    active_section = None
                    active_subrule = None
                    active_clause = None

                # Check Section boundary (e.g. in Acts)
                sec_match = self.section_pattern.search(raw_line)
                if sec_match:
                    flush_chunk()
                    active_section = sec_match.group(1).strip()
                    active_rule = None
                    active_subrule = None
                    active_clause = None

                # Check Rule boundary (e.g. "6. Declarations to be made on every package.-")
                rule_match = self.rule_pattern.search(raw_line)
                if rule_match:
                    rule_num = rule_match.group(1).strip()
                    # Filter out False positives (e.g. decimal numbers like 1.5 kg)
                    if not (rule_match.group(2).strip().startswith(("kg", "g", "l", "ml", "cm", "mm", "%"))):
                        flush_chunk()
                        active_rule = rule_num
                        active_section = None
                        active_subrule = None
                        active_clause = None

                # Check Sub-rule boundary (e.g. "(1)", "(2)")
                sub_match = self.subrule_pattern.search(raw_line)
                if sub_match and active_rule:
                    sub_str = f"({sub_match.group(1).strip()})"
                    # If accumulated text is already large, flush to create a sub-rule chunk
                    current_len = sum(len(x) for x in current_text_parts)
                    if current_len > 400:
                        flush_chunk()
                    active_subrule = sub_str
                    active_clause = None

                # Check Clause boundary (e.g. "(a)", "(b)")
                cl_match = self.clause_pattern.search(raw_line)
                if cl_match and active_rule:
                    cl_str = f"({cl_match.group(1).strip()})"
                    current_len = sum(len(x) for x in current_text_parts)
                    if current_len > 600:
                        flush_chunk()
                    active_clause = cl_str

                # Accumulate line
                if page_start is None:
                    page_start = p_num
                page_end = p_num
                if is_p_ocr:
                    has_ocr = True
                methods.add(p_method)

                current_text_parts.append(raw_line)

                # If single chunk exceeds max_chunk_chars, flush
                if sum(len(x) for x in current_text_parts) >= self.max_chunk_chars:
                    flush_chunk()

        flush_chunk()
        return chunks

    def _chunk_fallback_paragraphs(
        self,
        pages: List[Dict[str, Any]],
        document_id: str,
        source_file: str,
        document_title: str,
        doc_type: str,
        doc_date: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Graceful fallback when legal hierarchy headings are not clearly detected
        (e.g., advisories, guidelines, unstructured gazettes).
        Chunks by paragraphs while strictly preserving page boundaries and provenance.
        """
        chunks: List[Dict[str, Any]] = []
        current_paragraphs: List[str] = []
        current_len = 0
        page_start: Optional[int] = None
        page_end: Optional[int] = None
        has_ocr = False
        methods: set = set()

        def flush_fallback():
            nonlocal current_paragraphs, current_len, page_start, page_end, has_ocr, methods
            if not current_paragraphs:
                return
            text_block = "\n\n".join(current_paragraphs).strip()
            if len(text_block) < self.min_chunk_chars:
                return

            chunk_idx = len(chunks) + 1
            chunk_id = f"{document_id}_fb{chunk_idx:04d}"
            primary_method = "ocr" if has_ocr and "ocr" in methods else "text"

            # Fallback chunk has null hierarchy fields
            chunk = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source_file": source_file,
                "document_title": document_title,
                "page_start": page_start if page_start is not None else 1,
                "page_end": page_end if page_end is not None else (page_start or 1),
                "text": text_block,
                "rule_number": None,
                "section_number": None,
                "sub_rule": None,
                "clause": None,
                "schedule": None,
                "document_type": doc_type,
                "document_date": doc_date,
                "extraction_method": primary_method,
                "ocr_required": has_ocr,
                "fallback_chunking": True
            }
            chunks.append(chunk)

            # Keep small overlap if needed
            if len(current_paragraphs) > 1 and len(current_paragraphs[-1]) < self.overlap_chars:
                current_paragraphs = [current_paragraphs[-1]]
                current_len = len(current_paragraphs[0])
            else:
                current_paragraphs = []
                current_len = 0

            page_start = None
            page_end = None
            has_ocr = False
            methods = set()

        for page in pages:
            p_num = page["page_number"]
            p_text = page.get("text", "")
            p_method = page.get("extraction_method", "text")
            is_p_ocr = (p_method == "ocr")

            # Split into paragraphs
            paras = [p.strip() for p in re.split(r"\n\s*\n", p_text) if p.strip()]
            if not paras and p_text.strip():
                paras = [p_text.strip()]

            for para in paras:
                para_len = len(para)
                if current_len + para_len > self.max_chunk_chars and current_len >= self.min_chunk_chars:
                    flush_fallback()

                if page_start is None:
                    page_start = p_num
                page_end = p_num
                if is_p_ocr:
                    has_ocr = True
                methods.add(p_method)

                current_paragraphs.append(para)
                current_len += para_len

        flush_fallback()
        return chunks
