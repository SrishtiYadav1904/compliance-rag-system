import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from config.settings import settings
from .extractor import PDFExtractor
from .ocr_engine import OCREngine
from .legal_chunker import LegalChunker


class IngestionPipeline:
    """
    Master ingestion pipeline:
    - Iterates over all PDFs in kb_dir.
    - Extracts text from digital pages.
    - Runs GPU-accelerated OCR on scanned pages with disk caching.
    - Chunks documents using LegalChunker.
    - Generates chunk metadata and provenance.
    - Persists extracted documents and processed chunks.
    - Records comprehensive statistics and extraction failures.
    """

    def __init__(
        self,
        kb_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        force_ocr: bool = False,
        use_gpu: bool = True
    ):
        self.kb_dir = Path(kb_dir or settings.kb_dir)
        self.data_dir = Path(data_dir or settings.data_dir)
        self.force_ocr = force_ocr
        self.use_gpu = use_gpu

        self.ocr_cache_dir = self.data_dir / "ocr_cache"
        self.extracted_docs_dir = self.data_dir / "extracted_docs"
        self.processed_dir = self.data_dir / "processed"
        self.reports_dir = self.data_dir / "reports"

        for d in [self.ocr_cache_dir, self.extracted_docs_dir, self.processed_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.extractor = PDFExtractor(min_char_threshold=settings.ocr_min_char_threshold)
        self.ocr_engine = OCREngine(
            cache_dir=self.ocr_cache_dir,
            dpi=settings.ocr_dpi,
            use_gpu=self.use_gpu
        )
        self.chunker = LegalChunker(
            min_chunk_chars=settings.chunk_min_chars,
            max_chunk_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars
        )

    def run(self, max_docs: Optional[int] = None) -> Dict[str, Any]:
        start_time = time.time()
        pdf_files = sorted(list(self.kb_dir.glob("*.pdf")))
        if max_docs:
            pdf_files = pdf_files[:max_docs]

        total_files = len(pdf_files)
        print(f"[INFO] Starting ingestion pipeline for {total_files} PDF documents (GPU Acceleration: {self.use_gpu})...")

        all_chunks: List[Dict[str, Any]] = []
        doc_summaries: List[Dict[str, Any]] = []
        all_failures: List[Dict[str, Any]] = []

        total_pages = 0
        total_text_pages = 0
        total_ocr_pages = 0

        for idx, pdf_file in enumerate(pdf_files, 1):
            doc_start_time = time.time()
            doc_name = pdf_file.name
            print(f"[{idx}/{total_files}] Processing: {doc_name}...", end=" ", flush=True)

            try:
                # Stage 1: Extract digital text & detect scanned pages
                doc_data = self.extractor.extract_document(pdf_file)
                num_pages = doc_data["page_count"]
                total_pages += num_pages

                # Stage 2: OCR pipeline for scanned pages
                doc_data, failures = self.ocr_engine.process_document(
                    doc_data, force_ocr=self.force_ocr
                )
                if failures:
                    all_failures.extend(failures)

                # Count page types
                doc_text_pages = 0
                doc_ocr_pages = 0
                for p in doc_data["pages"]:
                    if p.get("extraction_method") == "ocr":
                        doc_ocr_pages += 1
                    else:
                        doc_text_pages += 1

                total_text_pages += doc_text_pages
                total_ocr_pages += doc_ocr_pages

                # Save extracted document representation outside kb/
                doc_json_path = self.extracted_docs_dir / f"{doc_data['document_id']}.json"
                with open(doc_json_path, "w", encoding="utf-8") as f:
                    json.dump(doc_data, f, ensure_ascii=False, indent=2)

                # Stage 3 & 4: Legal-aware chunking & metadata generation
                doc_chunks = self.chunker.chunk_document(doc_data)
                all_chunks.extend(doc_chunks)

                elapsed = time.time() - doc_start_time
                print(f"Done in {elapsed:.1f}s | pgs: {num_pages} (text: {doc_text_pages}, ocr: {doc_ocr_pages}) | chunks: {len(doc_chunks)}")

                doc_summaries.append({
                    "document_id": doc_data["document_id"],
                    "source_file": doc_name,
                    "document_title": doc_data["document_title"],
                    "page_count": num_pages,
                    "text_pages": doc_text_pages,
                    "ocr_pages": doc_ocr_pages,
                    "chunk_count": len(doc_chunks),
                    "extraction_failures": len(failures)
                })

            except Exception as e:
                print(f"FAILED: {e}")
                all_failures.append({
                    "source_file": doc_name,
                    "error": str(e)
                })

        # Save all chunks to data/processed/chunks.json and chunks.jsonl
        chunks_json_path = self.processed_dir / "chunks.json"
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        chunks_jsonl_path = self.processed_dir / "chunks.jsonl"
        with open(chunks_jsonl_path, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        total_elapsed = time.time() - start_time
        print(f"\n[INFO] Ingestion Pipeline Finished in {total_elapsed:.1f}s.")
        print(f"  Total Documents: {total_files}")
        print(f"  Total Pages: {total_pages} (Digital: {total_text_pages}, OCR: {total_ocr_pages})")
        print(f"  Total Chunks Generated: {len(all_chunks)}")
        print(f"  Total Extraction Failures: {len(all_failures)}")

        pipeline_stats = {
            "total_documents": total_files,
            "total_pages": total_pages,
            "digital_pages": total_text_pages,
            "ocr_pages": total_ocr_pages,
            "total_chunks": len(all_chunks),
            "total_failures": len(all_failures),
            "failures": all_failures,
            "document_summaries": doc_summaries,
            "elapsed_seconds": round(total_elapsed, 2)
        }

        # Save stats
        stats_path = self.reports_dir / "pipeline_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_stats, f, ensure_ascii=False, indent=2)

        return pipeline_stats
