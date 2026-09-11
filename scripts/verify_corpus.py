import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from config.settings import settings


def run_sanity_checks():
    print("=" * 70)
    print("RUNNING AUTOMATED CORPUS VALIDATION & INGESTION SANITY CHECKS")
    print("=" * 70)

    chunks_path = settings.processed_dir / "chunks.json"
    if not chunks_path.exists():
        print(f"[ERROR] chunks.json not found at {chunks_path}. Run ingestion pipeline first.")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    print(f"[INFO] Loaded {len(chunks)} chunks from {chunks_path}.")

    # Required concept targets
    target_concepts = {
        "Rule 6": {
            "query_patterns": [r"\bRule\s+6\b", r"\bRule\s+6\(", r"\bनियम\s+6\b", r"Declarations to be made on every package"],
            "description": "Rule 6 - Mandatory declarations on packaged commodities"
        },
        "declarations": {
            "query_patterns": [r"\bdeclarations?\b", r"\bघोषणा(?:एं|ओं)?\b"],
            "description": "Mandatory declarations on packages"
        },
        "Maximum Retail Price / MRP": {
            "query_patterns": [r"\bMaximum\s+Retail\s+Price\b", r"\bMRP\b", r"\bखुदरा\s+विक्रय\s+कीमत\b", r"\bएम0आर0पी0\b"],
            "description": "Maximum Retail Price / MRP declarations & inclusive of all taxes"
        },
        "net quantity": {
            "query_patterns": [r"\bnet\s+quantity\b", r"\bशुद्ध\s+मात्रा\b", r"\bstandard\s+units?\s+of\s+weight\b"],
            "description": "Net quantity declarations & unit requirements"
        },
        "manufacturer/packer/importer": {
            "query_patterns": [r"\bmanufactur(?:er|ed)\b", r"\bpacker\b", r"\bimporter\b", r"\bनिर्माता\b", r"\bपैकर\b", r"\bआयातक\b"],
            "description": "Name and complete address of manufacturer, packer, or importer"
        },
        "consumer care": {
            "query_patterns": [r"\bconsumer\s+care\b", r"\bcontact\s+details?\b", r"\bhelpline\b", r"\bउपभोक्ता\s+देखभाल\b", r"\btelephone\s+number\b"],
            "description": "Consumer care / contact details for grievance redressal"
        },
        "schedules": {
            "query_patterns": [r"\bSCHEDULE\s+[I|V|X0-9]+", r"\bFIRST\s+SCHEDULE\b", r"\bSECOND\s+SCHEDULE\b", r"\bTHIRD\s+SCHEDULE\b", r"\bअनुसूची\b"],
            "description": "Schedules (e.g. Schedule II font heights, Schedule III weights)"
        }
    }

    results: Dict[str, Any] = {}
    all_passed = True

    for concept_name, concept_info in target_concepts.items():
        print(f"\n--- Checking Concept: '{concept_name}' ---")
        print(f"Description: {concept_info['description']}")

        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in concept_info["query_patterns"]]
        matching_chunks = []

        for c in chunks:
            text = c.get("text", "")
            if any(p.search(text) for p in compiled_patterns):
                matching_chunks.append(c)

        match_count = len(matching_chunks)
        print(f"Matching Chunks Found: {match_count}")

        if match_count == 0:
            print(f"[FAIL] Zero chunks matched concept '{concept_name}'!")
            all_passed = False
            results[concept_name] = {
                "status": "FAIL",
                "matches": 0,
                "samples": []
            }
        else:
            print(f"[PASS] Concept '{concept_name}' successfully located in corpus.")
            # Select 2 diverse sample chunks with provenance
            samples = []
            for s in matching_chunks[:2]:
                sample_info = {
                    "chunk_id": s["chunk_id"],
                    "source_file": s["source_file"],
                    "document_title": s["document_title"],
                    "page_range": f"p.{s['page_start']}-p.{s['page_end']}",
                    "rule_number": s.get("rule_number"),
                    "sub_rule": s.get("sub_rule"),
                    "clause": s.get("clause"),
                    "extraction_method": s.get("extraction_method"),
                    "text_snippet": s["text"][:250].replace("\n", " ") + "..."
                }
                samples.append(sample_info)
                print(f"  Sample [{s['chunk_id']}] from '{s['source_file']}' ({sample_info['page_range']}, Rule: {s.get('rule_number')}, Method: {s.get('extraction_method')}):")
                print(f"    \"{sample_info['text_snippet']}\"")

            results[concept_name] = {
                "status": "PASS",
                "matches": match_count,
                "samples": samples
            }

    verification_output = {
        "all_passed": all_passed,
        "total_concepts_tested": len(target_concepts),
        "passed_concepts": sum(1 for r in results.values() if r["status"] == "PASS"),
        "results": results
    }

    out_file = settings.reports_dir / "verification_sanity_checks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(verification_output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    if all_passed:
        print(f"[ALL CHECKS PASSED] All 7 target legal concepts successfully located.")
    else:
        print(f"[WARNING] Some checks failed. Review details in {out_file}.")
    print("=" * 70)


if __name__ == "__main__":
    run_sanity_checks()
