import os
from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    # Base paths
    kb_dir: Path = BASE_DIR / "kb"
    data_dir: Path = BASE_DIR / "data"
    ocr_cache_dir: Path = BASE_DIR / "data" / "ocr_cache"
    extracted_docs_dir: Path = BASE_DIR / "data" / "extracted_docs"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    reports_dir: Path = BASE_DIR / "data" / "reports"
    
    # OCR settings
    ocr_dpi: int = 150
    ocr_min_char_threshold: int = 50  # pages with fewer chars are routed to OCR
    max_ocr_workers: int = 8          # parallel OCR workers for multicore throughput
    
    # Chunking settings
    chunk_min_chars: int = 50
    chunk_max_chars: int = 2500
    chunk_overlap_chars: int = 200

    def init_directories(self):
        for d in [
            self.data_dir,
            self.ocr_cache_dir,
            self.extracted_docs_dir,
            self.processed_dir,
            self.reports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.init_directories()
