import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.retrieval.embeddings import EmbeddingEngine
from src.retrieval.dense_index import DenseIndex


def main():
    parser = argparse.ArgumentParser(description="Build FAISS Dense Vector Index")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Embedding model name or HuggingFace ID (default: {settings.embedding_model})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation (default: 64)"
    )
    args = parser.parse_args()

    model_name = args.model or settings.embedding_model

    print("=" * 60)
    print("STAGE 6: BUILDING FAISS DENSE VECTOR INDEX")
    print(f"Embedding Model: {model_name}")
    print(f"Batch Size: {args.batch_size}")
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

    # Initialize Embedding Engine
    emb_engine = EmbeddingEngine(model_name=model_name)

    # Initialize and build Dense Index
    dense_index = DenseIndex()
    config = dense_index.build_from_chunks(chunks, emb_engine, batch_size=args.batch_size)

    print("\n[INFO] Validating index persistence and reloading...")
    reloaded_index = DenseIndex()
    success = reloaded_index.load()
    if not success:
        print("[ERROR] Failed to reload dense index from disk!")
        sys.exit(1)

    print(f"[SUCCESS] Reloaded FAISS index with {reloaded_index.faiss_index.ntotal} vectors.")

    # Quick sanity query
    test_query = "What declarations are mandatory on every package?"
    q_emb = emb_engine.embed_query(test_query)
    sample_results = reloaded_index.search(q_emb, top_k=3)
    print(f"\n[SANITY CHECK] Query: '{test_query}'")
    for r in sample_results:
        print(f"  - Rank {r.rank}: [{r.chunk_id}] Rule: {r.rule_number} | Cosine Sim: {r.score:.4f} | Source: {r.source_file}")

    print("\n" + "=" * 60)
    print("DENSE INDEX BUILD COMPLETE")
    print(f"Index directory: {settings.dense_index_dir}")
    print(f"Total vectors: {reloaded_index.faiss_index.ntotal}")
    print(f"Vector dimension: {reloaded_index.faiss_index.d}")
    print(f"Elapsed time: {config.get('build_elapsed_seconds', 0)}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
