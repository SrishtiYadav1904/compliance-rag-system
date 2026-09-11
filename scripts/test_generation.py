import sys
import argparse
import time
from pathlib import Path

# Ensure UTF-8 output for Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.generation.orchestration import LegalRAGOrchestrator
from src.generation.generator import GeneratorFactory


def main():
    parser = argparse.ArgumentParser(description="Stage 7 Grounded Multilingual RAG Generation Test CLI")
    parser.add_argument("--query", "-q", type=str, required=True, help="User query text (EN, HI, MR, TA)")
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["auto", "native", "translation"],
        default="auto",
        help="Retrieval strategy mode (default: auto)"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=["auto", "groq", "local"],
        default="auto",
        help="Generation provider (default: auto -> groq if key present, else local)"
    )
    parser.add_argument("--model", type=str, default=None, help="Generation model name")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranker")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose diagnostic details")

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 7: LEGAL METROLOGY GROUNDED GENERATION TEST")
    print(f"Query:        '{args.query}'")
    print(f"Mode:         {args.mode.upper()}")
    print(f"Provider:     {args.provider.upper()}")
    print("=" * 80)

    # Initialize generator and orchestrator
    generator = GeneratorFactory.get_generator(provider=args.provider, model=args.model)
    orchestrator = LegalRAGOrchestrator(
        generator=generator,
        retrieval_mode=args.mode
    )

    t0 = time.time()
    response = orchestrator.answer_query(
        query=args.query,
        retrieval_mode=args.mode,
        use_reranker=not args.no_rerank
    )
    elapsed = time.time() - t0

    # Display results
    print(f"\n[QUERY PROCESSING]")
    print(f"Original Query:    '{response.original_query}'")
    print(f"Detected Language: {response.language.upper()}")
    print(f"Retrieval Query:   '{response.retrieval_query}'")
    print(f"Translation Used:  {response.translation_used}")
    print(f"Abstained:         {response.abstained} {'(' + (response.abstention_reason or '') + ')' if response.abstained else ''}")
    print(f"Grounded:          {response.grounded}")
    print(f"Citations Valid:   {response.citation_verified}")

    print("\n" + "=" * 80)
    print("GENERATED RESPONSE")
    print("=" * 80)
    print(response.answer)
    print("=" * 80)

    print("\n[VERIFIED CITATIONS]")
    if response.citations:
        for idx, cit in enumerate(response.citations, 1):
            status = "[VERIFIED]" if cit.verified else "[UNVERIFIED]"
            print(f"  {idx}. {status} {cit.formatted()}")
    else:
        print("  None (abstained or unreferenced)")

    print(f"\n[RETRIEVED CHUNKS ({len(response.retrieved_chunk_ids)})]")
    print(f"  {', '.join(response.retrieved_chunk_ids)}")

    print(f"\n[LATENCY BREAKDOWN]")
    for k, v in response.latencies.items():
        print(f"  - {k:<30}: {v:.2f} ms")
    print(f"  - Total Elapsed Clock        : {elapsed * 1000:.2f} ms")

    if args.verbose:
        print(f"\n[DIAGNOSTICS]")
        for k, v in response.diagnostics.items():
            print(f"  - {k}: {v}")

    print("=" * 80)


if __name__ == "__main__":
    main()
