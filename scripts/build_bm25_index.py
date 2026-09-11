import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.retrieval.bm25_index import BM25Index


def main():
    print("=" * 60)
    print("STAGE 6: BUILDING BM25 LEXICAL INDEX")
    print("=" * 60)

    chunks_path = settings.processed_dir / "chunks.json"
    if not chunks_path.exists():
        print(f"[ERROR] Processed chunks file not found at: {chunks_path}")
        sys.exit(1)

    t0 = time.time()
    print(f"[INFO] Loading chunks from {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[INFO] Successfully loaded {len(chunks)} chunks in {time.time() - t0:.2f}s.")

    # Initialize BM25 Index
    bm25_index = BM25Index()
    config = bm25_index.build_from_chunks(chunks)

    print("\n[INFO] Validating index persistence and reloading...")
    reloaded_index = BM25Index()
    success = reloaded_index.load()
    if not success:
        print("[ERROR] Failed to reload BM25 index from disk!")
        sys.exit(1)

    print(f"[SUCCESS] Reloaded BM25 index with {len(reloaded_index.chunk_ids)} indexed chunks.")

    # Quick sanity query
    test_query = "Rule 6 mandatory declarations"
    sample_results = reloaded_index.search(test_query, top_k=3)
    print(f"\n[SANITY CHECK] Query: '{test_query}'")
    for r in sample_results:
        print(f"  - Rank {r.rank}: [{r.chunk_id}] Rule: {r.rule_number} | Score: {r.score:.4f} | Source: {r.source_file}")

    print("\n" + "=" * 60)
    print("BM25 INDEX BUILD COMPLETE")
    print(f"Index directory: {settings.bm25_index_dir}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Elapsed time: {config.get('build_elapsed_seconds', 0)}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
