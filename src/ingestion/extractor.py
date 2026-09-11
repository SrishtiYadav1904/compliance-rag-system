import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import pymupdf


class PDFExtractor:
    """
    Extracts text and page-level metadata from PDFs.
    Identifies pages that are scanned (empty or below char threshold) and flags them for OCR.
    Preserves exact page boundaries and detects Devanagari (Hindi) vs English content.
    """

    def __init__(self, min_char_threshold: int = 50):
        self.min_char_threshold = min_char_threshold

    def extract_document(self, file_path: Path) -> Dict[str, Any]:
        doc = pymupdf.open(str(file_path))
        page_count = len(doc)
        pdf_metadata = doc.metadata or {}

        pages_data: List[Dict[str, Any]] = []
        needs_ocr_pages: List[int] = []

        total_chars = 0
        total_hindi_chars = 0
        total_english_chars = 0

        for page_idx in range(page_count):
            page_num = page_idx + 1
            page = doc[page_idx]
            raw_text = page.get_text("text") or ""
            cleaned_text = raw_text.strip()
            char_count = len(cleaned_text)

            hindi_chars = sum(1 for c in cleaned_text if "\u0900" <= c <= "\u097f")
            english_chars = sum(
                1 for c in cleaned_text if c.isascii() and c.isalpha()
            )

            total_chars += char_count
            total_hindi_chars += hindi_chars
            total_english_chars += english_chars

            if hindi_chars > 50 and english_chars > 50:
                lang = "bilingual"
            elif hindi_chars > 50:
                lang = "hi"
            else:
                lang = "en"

            is_scanned = char_count < self.min_char_threshold

            if is_scanned:
                needs_ocr_pages.append(page_num)

            pages_data.append(
                {
                    "page_number": page_num,
                    "text": cleaned_text,
                    "char_count": char_count,
                    "hindi_char_count": hindi_chars,
                    "english_char_count": english_chars,
                    "language": lang,
                    "needs_ocr": is_scanned,
                    "extraction_method": "text" if not is_scanned else None,
                    "ocr_confidence": 1.0 if not is_scanned else None,
                }
            )

        doc.close()

        # Determine overall document language
        if total_hindi_chars > 200 and total_english_chars > 200:
            doc_lang = "bilingual"
        elif total_hindi_chars > 200:
            doc_lang = "hi"
        else:
            doc_lang = "en"

        document_id = file_path.stem
        document_title = (
            pdf_metadata.get("title")
            or self._derive_title_from_filename(file_path.name)
        )

        return {
            "document_id": document_id,
            "source_file": file_path.name,
            "source_path": str(file_path.resolve()),
            "document_title": document_title,
            "page_count": page_count,
            "total_extracted_chars": total_chars,
            "language": doc_lang,
            "needs_ocr": len(needs_ocr_pages) > 0,
            "needs_ocr_pages": needs_ocr_pages,
            "pages": pages_data,
        }

    def _derive_title_from_filename(self, filename: str) -> str:
        name = re.sub(r"_\d{10}\.pdf$", "", filename, flags=re.IGNORECASE)
        name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
        name = name.replace("_", " ").strip()
        return name
