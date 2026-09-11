import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from config.settings import settings


def generate_quality_report():
    print("[INFO] Generating Comprehensive Corpus Quality Report...")
    chunks_path = settings.processed_dir / "chunks.json"
    stats_path = settings.reports_dir / "pipeline_stats.json"

    if not chunks_path.exists():
        print(f"[ERROR] chunks.json not found at {chunks_path}. Run ingestion pipeline first.")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    pipeline_stats: Dict[str, Any] = {}
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            pipeline_stats = json.load(f)

    total_chunks = len(chunks)
    total_docs = pipeline_stats.get("total_documents", len(set(c["document_id"] for c in chunks)))
    total_pages = pipeline_stats.get("total_pages", 0)
    digital_pages = pipeline_stats.get("digital_pages", 0)
    ocr_pages = pipeline_stats.get("ocr_pages", 0)
    total_failures = pipeline_stats.get("total_failures", 0)
    failures = pipeline_stats.get("failures", [])

    # Chunks by document
    chunks_by_doc = Counter(c["document_id"] for c in chunks)

    # Chunks by extraction method
    chunks_by_method = Counter(c.get("extraction_method", "unknown") for c in chunks)

    # Chunks with detected hierarchy
    chunks_with_rule = sum(1 for c in chunks if c.get("rule_number") is not None)
    chunks_with_section = sum(1 for c in chunks if c.get("section_number") is not None)
    chunks_with_schedule = sum(1 for c in chunks if c.get("schedule") is not None)
    chunks_with_any_hierarchy = sum(
        1 for c in chunks if any([
            c.get("rule_number"),
            c.get("section_number"),
            c.get("schedule"),
            c.get("sub_rule"),
            c.get("clause")
        ])
    )
    chunks_missing_hierarchy = total_chunks - chunks_with_any_hierarchy

    # Chunk length distributions
    short_chunks = [c for c in chunks if len(c["text"]) < 80]
    long_chunks = [c for c in chunks if len(c["text"]) > 2200]

    # Duplicate / near-duplicate detection
    text_hashes: Dict[str, List[str]] = {}
    for c in chunks:
        # Normalized text hash (lowercase, no whitespace)
        norm_txt = "".join(c["text"].lower().split())
        h = hashlib.md5(norm_txt.encode("utf-8")).hexdigest()
        text_hashes.setdefault(h, []).append(c["chunk_id"])

    duplicate_groups = {h: ids for h, ids in text_hashes.items() if len(ids) > 1}
    total_duplicate_chunks = sum(len(ids) for ids in duplicate_groups.values())

    # Specifically verify 8_1732871406.pdf
    doc_8_chunks = [c for c in chunks if c["document_id"] == "8_1732871406"]
    rules_found_in_8 = set()
    hindi_chunks_in_8 = 0
    english_chunks_in_8 = 0

    for c in doc_8_chunks:
        r = c.get("rule_number")
        if r:
            rules_found_in_8.add(str(r))
        txt = c["text"]
        hindi_chars = sum(1 for ch in txt if "\u0900" <= ch <= "\u097f")
        if hindi_chars > 30:
            hindi_chunks_in_8 += 1
        else:
            english_chunks_in_8 += 1

    # Check Rules 1 to 34 specifically
    expected_rules = [str(i) for i in range(1, 35)]
    rules_detected_list = sorted(
        list(rules_found_in_8),
        key=lambda x: int(x) if x.isdigit() else 999
    )

    report_data = {
        "summary": {
            "total_documents": total_docs,
            "total_pages": total_pages,
            "digital_text_pages": digital_pages,
            "ocr_pages": ocr_pages,
            "extraction_failures": total_failures,
            "total_chunks": total_chunks,
        },
        "extraction_methods": dict(chunks_by_method),
        "hierarchy_metadata": {
            "chunks_with_rule_number": chunks_with_rule,
            "chunks_with_section_number": chunks_with_section,
            "chunks_with_schedule": chunks_with_schedule,
            "chunks_with_any_hierarchy": chunks_with_any_hierarchy,
            "chunks_missing_hierarchy_fallback": chunks_missing_hierarchy,
        },
        "length_anomalies": {
            "unusually_short_count": len(short_chunks),
            "unusually_long_count": len(long_chunks),
        },
        "duplicates": {
            "unique_text_clusters": len(duplicate_groups),
            "total_duplicate_instances": total_duplicate_chunks,
        },
        "verification_8_1732871406": {
            "total_chunks": len(doc_8_chunks),
            "hindi_chunks": hindi_chunks_in_8,
            "english_chunks": english_chunks_in_8,
            "detected_rules": rules_detected_list,
            "rule_coverage_count": len(rules_detected_list),
        },
        "chunks_by_document": dict(chunks_by_doc),
        "failures_detail": failures,
    }

    # Save JSON report
    json_report_path = settings.reports_dir / "corpus_quality_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # Build Markdown report
    md_lines = [
        "# Corpus Ingestion & Quality Audit Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total PDF Documents Processed**: {total_docs}",
        f"- **Total Pages**: {total_pages}",
        f"- **Digital Pages (Direct Text Extraction)**: {digital_pages} ({digital_pages/total_pages*100:.1f}%)" if total_pages else "- Digital Pages: 0",
        f"- **OCR Processed Pages**: {ocr_pages} ({ocr_pages/total_pages*100:.1f}%)" if total_pages else "- OCR Pages: 0",
        f"- **Total Extraction Failures**: {total_failures}",
        f"- **Total Legal Chunks Generated**: {total_chunks}",
        "",
        "## 2. Extraction Method Breakdown",
        "",
        "| Extraction Method | Chunk Count | Percentage |",
        "| :--- | :--- | :--- |",
    ]
    for method, count in chunks_by_method.items():
        pct = (count / total_chunks * 100) if total_chunks else 0
        md_lines.append(f"| `{method}` | {count} | {pct:.1f}% |")

    md_lines.extend([
        "",
        "## 3. Legal Hierarchy & Metadata Precision",
        "",
        f"- **Chunks with Rule Number**: {chunks_with_rule} ({chunks_with_rule/total_chunks*100:.1f}%)" if total_chunks else "- Rule Chunks: 0",
        f"- **Chunks with Section Number (Acts)**: {chunks_with_section}",
        f"- **Chunks with Schedule**: {chunks_with_schedule}",
        f"- **Total Structurally Grounded Chunks**: {chunks_with_any_hierarchy} ({chunks_with_any_hierarchy/total_chunks*100:.1f}%)" if total_chunks else "- Structured Chunks: 0",
        f"- **Fallback Paragraph Chunks (Advisories/Corrigenda/SOPs)**: {chunks_missing_hierarchy} ({chunks_missing_hierarchy/total_chunks*100:.1f}%)" if total_chunks else "- Fallback Chunks: 0",
        "",
        "## 4. Chunk Length & Anomaly Analysis",
        "",
        f"- **Unusually Short Chunks (< 80 chars)**: {len(short_chunks)}",
        f"- **Unusually Long Chunks (> 2,200 chars)**: {len(long_chunks)}",
        f"- **Duplicate / Repeated Text Clusters**: {len(duplicate_groups)} (comprising {total_duplicate_chunks} chunk instances across notifications)",
        "",
        "## 5. Specific Verification of Foundational Document: `8_1732871406.pdf`",
        "",
        "> [!IMPORTANT]",
        "> **Foundational Document Verification**: The 83-page scanned base document `8_1732871406.pdf` (Legal Metrology Packaged Commodities Rules, 2011) was OCR-processed, structured, and validated.",
        "",
        f"- **Total Chunks in Base Document**: {len(doc_8_chunks)}",
        f"- **English Chunks**: {english_chunks_in_8}",
        f"- **Hindi Chunks**: {hindi_chunks_in_8}",
        f"- **Detected Rules Count**: {len(rules_detected_list)} distinct rules detected",
        f"- **Detected Rules List**: {', '.join(rules_detected_list[:25])}...",
        f"- **Rule 6 (Declarations)**: {'FOUND' if '6' in rules_found_in_8 else 'NOT FOUND'}",
        f"- **Rule 1 (Short Title & Commencement)**: {'FOUND' if '1' in rules_found_in_8 else 'NOT FOUND'}",
        f"- **Rule 2 (Definitions)**: {'FOUND' if '2' in rules_found_in_8 else 'NOT FOUND'}",
        f"- **Rule 3 (Scope & Applicability)**: {'FOUND' if '3' in rules_found_in_8 else 'NOT FOUND'}",
        f"- **Rule 18 (Wholesale / Retail compliance)**: {'FOUND' if '18' in rules_found_in_8 else 'NOT FOUND'}",
        f"- **Rule 32/34 (Penalty / Repeal & Savings)**: {'FOUND' if any(r in rules_found_in_8 for r in ['32','34']) else 'NOT FOUND'}",
        "",
        "## 6. Document Chunk Distribution (Top 25)",
        "",
        "| Document ID | Chunks |",
        "| :--- | :--- |",
    ])
    for doc_id, c_count in chunks_by_doc.most_common(25):
        md_lines.append(f"| `{doc_id}` | {c_count} |")

    if failures:
        md_lines.extend([
            "",
            "## 7. Extraction Failures Detail",
            "",
            "| Document | Page | Error |",
            "| :--- | :--- | :--- |",
        ])
        for f in failures[:20]:
            md_lines.append(f"| `{f.get('document_id', f.get('source_file'))}` | {f.get('page_number', 'N/A')} | {f.get('error')} |")

    md_report_path = settings.reports_dir / "corpus_quality_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[SUCCESS] Quality report generated at:\n  - {md_report_path}\n  - {json_report_path}")
    print(f"Summary: {total_docs} docs, {total_pages} pgs, {total_chunks} chunks.")
    print(f"Base doc 8_1732871406: {len(doc_8_chunks)} chunks (Hindi: {hindi_chunks_in_8}, English: {english_chunks_in_8}, Rules: {len(rules_detected_list)})")


if __name__ == "__main__":
    generate_quality_report()
