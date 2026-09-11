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
    indices_dir: Path = BASE_DIR / "data" / "indices"
    dense_index_dir: Path = BASE_DIR / "data" / "indices" / "dense"
    bm25_index_dir: Path = BASE_DIR / "data" / "indices" / "bm25"
    eval_dir: Path = BASE_DIR / "evaluation"
    eval_datasets_dir: Path = BASE_DIR / "evaluation" / "datasets"
    eval_reports_dir: Path = BASE_DIR / "evaluation" / "reports"

    # OCR settings
    ocr_dpi: int = 150
    ocr_min_char_threshold: int = 50
    max_ocr_workers: int = 8

    # Chunking settings
    chunk_min_chars: int = 60
    chunk_max_chars: int = 2200
    chunk_overlap_chars: int = 150

    # Embedding & Retrieval settings (Configurable via ENV)
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    alternative_embedding_model: str = "BAAI/bge-m3"

    reranker_model: str = Field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL",
            "BAAI/bge-reranker-v2-m3"
        )
    )

    # Retrieval parameters
    rrf_k: int = 60
    dense_top_n: int = 30
    bm25_top_n: int = 30
    hybrid_top_n: int = 40
    final_top_k: int = 10

    # Generation & Orchestration settings (Stage 7)
    generation_provider: str = Field(
        default_factory=lambda: os.getenv("GENERATION_PROVIDER", "auto")
    )
    generation_model: str = Field(
        default_factory=lambda: os.getenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
    )
    retrieval_mode: str = Field(
        default_factory=lambda: os.getenv("RETRIEVAL_MODE", "auto")
    )
    context_chunk_limit: int = 5
    confidence_threshold: float = 0.015

    def init_directories(self):
        for d in [
            self.data_dir,
            self.ocr_cache_dir,
            self.extracted_docs_dir,
            self.processed_dir,
            self.reports_dir,
            self.indices_dir,
            self.dense_index_dir,
            self.bm25_index_dir,
            self.eval_dir,
            self.eval_datasets_dir,
            self.eval_reports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.init_directories()
