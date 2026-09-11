"""
Ingestion, OCR, and Legal-Aware Chunking Module
"""
from .extractor import PDFExtractor
from .ocr_engine import OCREngine
from .legal_chunker import LegalChunker
from .pipeline import IngestionPipeline

__all__ = ["PDFExtractor", "OCREngine", "LegalChunker", "IngestionPipeline"]
