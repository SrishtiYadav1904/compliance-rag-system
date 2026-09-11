# 🏛️ Legal Metrology Compliance RAG System

> **A standalone, multilingual, zero-hallucination AI compliance assistant for the Legal Metrology (Packaged Commodities) Rules, 2011 and Legal Metrology Act, 2009.**

---

## 💡 What This Project Does (In Simple Terms)

When packaged goods (food, electronics, cosmetics, garments, etc.) are sold in India, manufacturers, packers, and importers must comply with strict statutory labeling rules (MRP format, net quantity units, manufacturer details, consumer care, country of origin, font sizes).

This system acts as an **Authoritative Legal Compliance Assistant**:
- 📚 **Powered by 82 Official Legal PDFs**: Built strictly over 2,056 pages of authentic Gazette notifications and statutory rules broken into 4,860 verified legal chunks.
- 🛡️ **Zero Hallucination Guarantee**: The AI **never** answers from pre-trained memory. It answers **only** if supporting evidence exists in the legal knowledge base; otherwise, it strictly **abstains**.
- 🔍 **Audit-Ready Citations**: Every answer includes exact provenance: PDF filename, page number, Rule, Sub-rule, and Clause.
- 🌐 **Multilingual**: Understands questions in **English**, **Hindi**, **Marathi**, and **Tamil**, answering in the user's language while preserving statutory rule numbers (`Rule 6(1)(a)`, `MRP`, `kg`, `ml`).
- 🛑 **False Premise & Injection Defense**: Corrects wrong assumptions (e.g. *"Does Rule 10 require QR codes?"*) and rejects prompt-injection attacks.

> **Important Boundary:** This module is the **Knowledge & Explanation Assistant**. The main parent system handles product image capture, OCR, and violation detection; this assistant explains legal requirements and cites the governing laws.

---

## 🚀 Quick Start & How to Test

### 1. Prerequisites & Setup
Make sure you have Python 3.10+ installed and install dependencies:
```powershell
pip install -r requirements.txt
```
*(Dependencies include `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`, `rank-bm25`, `pydantic`, and optionally `groq`).*

---

### 2. Run an Interactive Live Chat (Best for Team Demos)
Launch the interactive terminal chat session:
```powershell
python scripts/demo_chat.py
```
Type any compliance question directly in the terminal:
```text
Ask a legal compliance question > What declarations are mandatory on every package?
```
- Returns sub-second answers (<350ms)
- Displays verified citations with PDF name and page numbers
- Automatically handles English, Hindi, Marathi, and Tamil
- Type `help` for sample questions, or `exit` to quit.

---

### 3. Run the 1-Click Automated Showcase Demo
If you are giving a live presentation to your team or evaluators, run the automated showcase:
```powershell
python scripts/run_showcase.py
```
This automatically runs through **7 showcase test cases**:
1. **English Legal QA**: Rule 6 mandatory declarations.
2. **Hindi Cross-Lingual QA**: Translates query, retrieves English provisions, answers in Hindi with citations.
3. **Marathi Cross-Lingual QA**: Distinguishes Marathi Devanagari from Hindi and provides localized explanation.
4. **Tamil Cross-Lingual QA**: Classifies Tamil script and retrieves legal rules.
5. **False Premise Correction**: Proves the assistant does not agree with false user assumptions.
6. **Strict Abstention**: Demonstrates clean refusal to answer questions outside Legal Metrology (e.g. Income Tax).
7. **Prompt Injection Defense**: Demonstrates defense against malicious instruction overrides.

---

### 4. Test Single Queries from Command Line
```powershell
# English query
python scripts/test_generation.py --query "Is it permissible to alter MRP using individual stickers?" --no-rerank

# Hindi query
python scripts/test_generation.py --query "हर पैकेज पर कौन-कौन सी घोषणाएँ अनिवार्य हैं?" --no-rerank

# Misleading premise correction
python scripts/test_generation.py --query "Rule 10 says every package needs a QR code, right?" --no-rerank

# Unsupported question (verifying clean abstention)
python scripts/test_generation.py --query "What does the Indian Income Tax Act say about packaging?" --no-rerank
```

---

### 5. Inspect the Hybrid Retrieval Pipeline
To show your team how the search engine works under the hood (Dense Cosine + BM25 Lexical + RRF Ranking):
```powershell
python scripts/test_retrieval.py --query "What declarations are mandatory on every package?" --method hybrid --top-k 5
```

---

## 🔌 How to Integrate into Your Application

Integrating this assistant into another service, backend API, or chatbot UI requires **only 4 lines of Python code**:

```python
from src.generation.orchestration import LegalRAGOrchestrator

# 1. Initialize the orchestrator
orchestrator = LegalRAGOrchestrator()

# 2. Ask any legal compliance question
response = orchestrator.answer_query("Can individual stickers be used to alter MRP?")

# 3. Access clean, structured results
print("Answer:\n", response.answer)
print("Language:", response.language)
print("Abstained:", response.abstained)

# 4. Access verified citations
for citation in response.citations:
    print(f"- {citation.source_file}, Page {citation.page_start}, Rule {citation.rule_number}")
```

### JSON Response Schema
Every query returns a typed [`GroundedResponse`](file:///c:/Users/student/Downloads/compliance-rag-system/compliance-rag-system/src/generation/schemas.py) object:
```json
{
  "answer": "Under Rule 6(3), it is not permissible to affix individual stickers on the package for altering declarations of retail sale price...",
  "language": "en",
  "original_query": "Can individual stickers be used to alter MRP?",
  "retrieval_query": "Can individual stickers be used to alter MRP?",
  "translation_used": false,
  "citations": [
    {
      "source_file": "8_1732871406.pdf",
      "page_start": 46,
      "page_end": 46,
      "rule_number": "6",
      "sub_rule": "3",
      "verified": true
    }
  ],
  "retrieved_chunk_ids": ["8_1732871406_c0035"],
  "citation_verified": true,
  "grounded": true,
  "abstained": false,
  "latencies": {
    "language_detection_ms": 0.02,
    "retrieval_and_reranking_ms": 182.4,
    "generation_ms": 0.08,
    "citation_verification_ms": 0.12,
    "total_ms": 182.62
  }
}
```

---

## ⚙️ Configuration & Deployment Modes

The system is configured via environment variables or [`config/settings.py`](file:///c:/Users/student/Downloads/compliance-rag-system/compliance-rag-system/config/settings.py):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GENERATION_PROVIDER` | `auto` | `auto` (uses Groq if API key set, else offline local synthesizer), `groq`, `local` |
| `GENERATION_MODEL` | `llama-3.3-70b-versatile` | LLM model name when using cloud provider |
| `GROQ_API_KEY` | *(optional)* | Cloud API key. If absent, system runs 100% offline using built-in local grounded synthesizer. |
| `RETRIEVAL_MODE` | `auto` | `auto` (adaptive native + translation), `native`, `translation` |
| `CONTEXT_CHUNK_LIMIT`| `5` | Number of top evidence chunks passed to generator (5–7 chunks) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Dense multilingual embedding model |

---

## 🏗️ System Architecture

```text
                  User Query (English / Hindi / Marathi / Tamil)
                                       │
                                       ▼
                       Language Detection & Translation
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       Dense Vector Retrieval                        BM25 Lexical Retrieval
       (FAISS Cosine Similarity)                     (Legal Tokenizer & Rules)
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                         Reciprocal Rank Fusion (RRF)
                                       │
                                       ▼
                     Cross-Encoder Reranker (BGE-Reranker)
                                       │
                                       ▼
                       Context Validation & Selection
                       (Abstains if evidence missing)
                                       │
                                       ▼
                         Grounded Generation Prompt
                         (Injection & Premise Defense)
                                       │
                                       ▼
                        LLM / Local Grounded Generator
                                       │
                                       ▼
                         Post-Generation Citation Verifier
                                       │
                                       ▼
                       Structured Response with Citations
```

---

## 📊 Benchmark Evaluation Results

Evaluated over **30 comprehensive benchmark queries** spanning all statutory topics, languages, and edge cases:

| Metric | Score | Status |
| :--- | :---: | :---: |
| **Groundedness Rate** | **100.00%** | Zero hallucinations from model memory |
| **Citation Verification Rate** | **100.00%** | All citations cross-referenced against KB |
| **Abstention Accuracy** | **100.00%** | 100% clean refusal on out-of-domain queries |
| **Language Detection Accuracy** | **100.00%** | 100% correct across EN, HI, MR, TA |
| **Premise Correction Rate** | **100.00%** | Corrects false legal assumptions |
| **Prompt Injection Defense** | **100.00%** | Rejects system instruction overrides |
| **Mean Response Time** | **~304 ms** | Sub-second end-to-end latency |

To reproduce the automated benchmarks:
```powershell
python scripts/evaluate_generation.py
```
Full diagnostic reports are stored in:
- `evaluation/reports/generation_report.md`
- `evaluation/reports/generation_results.json`
- `evaluation/reports/retrieval_report.md`

---

## 📁 Repository Structure

```text
compliance-rag-system/
├── config/
│   └── settings.py              # Central configuration & paths
├── data/
│   ├── indices/                 # Serialized FAISS and BM25 index files
│   └── processed/               # 4,860 pre-extracted legal chunks
├── evaluation/
│   ├── datasets/                # Benchmark evaluation datasets (JSONL)
│   └── reports/                 # Markdown & JSON evaluation reports
├── kb/                          # 82 authoritative official legal PDFs
├── scripts/
│   ├── demo_chat.py             # Interactive live terminal chat session
│   ├── run_showcase.py          # 1-click automated showcase demo
│   ├── test_generation.py       # Single-query CLI tester
│   ├── evaluate_generation.py   # Generation benchmark runner
│   ├── test_retrieval.py        # Retrieval diagnostic tester
│   ├── evaluate_retrieval.py    # Retrieval benchmark runner
│   ├── build_dense_index.py     # FAISS index builder
│   └── build_bm25_index.py      # BM25 index builder
├── src/
│   ├── generation/              # Grounded generation, prompts, verifier, orchestrator
│   ├── retrieval/               # Hybrid retriever, FAISS dense, BM25, reranker
│   └── ingestion/               # Legal chunker, PDF extractor, OCR pipeline
└── README.md
```
