import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output for Windows terminals
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.retrieval.schemas import LegalFilter, RetrievalMethod
from src.retrieval.hybrid import HybridRetriever


def format_chunk_preview(text: str, max_chars: int = 250) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + "..."
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Legal Metrology RAG Diagnostic Retrieval Test")
    parser.add_argument("--query", "-q", type=str, required=True, help="Query text (English, Hindi, Marathi, Tamil)")
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["dense", "bm25", "hybrid", "rerank"],
        default="hybrid",
        help="Retrieval method (default: hybrid)"
    )
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results to retrieve (default: 10)")
    parser.add_argument("--filter-rule", type=str, default=None, help="Filter by rule_number (e.g. '6')")
    parser.add_argument("--filter-section", type=str, default=None, help="Filter by section_number")
    parser.add_argument("--filter-type", type=str, default=None, help="Filter by document_type (e.g. 'Rules')")
    parser.add_argument("--filter-date", type=str, default=None, help="Filter by document_date (e.g. '2023-10-06')")
    parser.add_argument("--filter-schedule", type=str, default=None, help="Filter by schedule (e.g. 'II')")

    args = parser.parse_args()

    # Construct legal filter if any flag is set
    legal_filter = LegalFilter(
        rule_number=args.filter_rule,
        section_number=args.filter_section,
        document_type=args.filter_type,
        document_date=args.filter_date,
        schedule=args.filter_schedule
    )

    print("=" * 80)
    print("LEGAL METROLOGY RETRIEVAL DIAGNOSTIC")
    print(f"Query:   '{args.query}'")
    print(f"Method:  {args.method.upper()}")
    print(f"Top-K:   {args.top_k}")
    if legal_filter.is_active():
        print(f"Filter:  {legal_filter.model_dump(exclude_none=True)}")
    else:
        print("Filter:  None (Default: unfiltered)")
    print("=" * 80)

    retriever = HybridRetriever()

    if args.method == "dense":
        results = retriever.dense_search(args.query, top_k=args.top_k, filters=legal_filter)
    elif args.method == "bm25":
        results = retriever.bm25_search(args.query, top_k=args.top_k, filters=legal_filter)
    elif args.method == "hybrid":
        results = retriever.hybrid_search(args.query, top_k=args.top_k, filters=legal_filter, use_reranker=False)
    elif args.method == "rerank":
        results = retriever.hybrid_search(args.query, top_k=args.top_k, filters=legal_filter, use_reranker=True)

    print(f"\nRetrieved {len(results)} chunks:\n")

    for r in results:
        rule_str = f"Rule {r.rule_number}" if r.rule_number else "No Rule"
        if r.sub_rule:
            rule_str += f"({r.sub_rule})"
        if r.clause:
            rule_str += f"({r.clause})"
        sched_str = f" | Sched: {r.schedule}" if r.schedule else ""
        date_str = f" | Date: {r.document_date}" if r.document_date else ""

        diag_parts = []
        if r.dense_rank is not None:
            diag_parts.append(f"Dense: #{r.dense_rank} ({r.dense_score:.4f})")
        if r.bm25_rank is not None:
            diag_parts.append(f"BM25: #{r.bm25_rank} ({r.bm25_score:.2f})")
        if r.rrf_score is not None:
            diag_parts.append(f"RRF: {r.rrf_score:.6f}")
        if r.rerank_score is not None:
            diag_parts.append(f"Rerank: {r.rerank_score:.4f}")

        diag_str = " | ".join(diag_parts) if diag_parts else f"Score: {r.score:.4f}"

        print(f"[{r.rank}] Chunk ID: {r.chunk_id}  (Score: {r.score:.4f})")
        print(f"    Source:      {r.source_file} (Pages {r.page_start}-{r.page_end})")
        print(f"    Hierarchy:   {rule_str}{sched_str}{date_str} [{r.document_type or 'Doc'}]")
        print(f"    Diagnostics: {diag_str}")
        print(f"    Extraction:  method={r.extraction_method}, ocr_required={r.ocr_required}, fallback={r.fallback_chunking}")
        print(f"    Text:        \"{format_chunk_preview(r.text)}\"")
        print("-" * 80)


if __name__ == "__main__":
    main()
