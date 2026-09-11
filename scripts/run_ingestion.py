import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run PDF ingestion, GPU-accelerated OCR, legal chunking, and metadata generation."
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR re-processing even if cached pages exist."
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)."
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration and fall back to CPU."
    )
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    pipeline = IngestionPipeline(force_ocr=args.force_ocr, use_gpu=use_gpu)
    pipeline.run(max_docs=args.max_docs)


if __name__ == "__main__":
    main()
