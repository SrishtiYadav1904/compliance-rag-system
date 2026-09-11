# Corpus Ingestion & Quality Audit Report

## 1. Executive Summary

- **Total PDF Documents Processed**: 82
- **Total Pages**: 2056
- **Digital Pages (Direct Text Extraction)**: 1110 (54.0%)
- **OCR Processed Pages**: 946 (46.0%)
- **Total Extraction Failures**: 0
- **Total Legal Chunks Generated**: 4860

## 2. Extraction Method Breakdown

| Extraction Method | Chunk Count | Percentage |
| :--- | :--- | :--- |
| `ocr` | 2142 | 44.1% |
| `text` | 2718 | 55.9% |

## 3. Legal Hierarchy & Metadata Precision

- **Chunks with Rule Number**: 4606 (94.8%)
- **Chunks with Section Number (Acts)**: 0
- **Chunks with Schedule**: 2254
- **Total Structurally Grounded Chunks**: 4750 (97.7%)
- **Fallback Paragraph Chunks (Advisories/Corrigenda/SOPs)**: 110 (2.3%)

## 4. Chunk Length & Anomaly Analysis

- **Unusually Short Chunks (< 80 chars)**: 129
- **Unusually Long Chunks (> 2,200 chars)**: 250
- **Duplicate / Repeated Text Clusters**: 17 (comprising 37 chunk instances across notifications)

## 5. Specific Verification of Foundational Document: `8_1732871406.pdf`

> [!IMPORTANT]
> **Foundational Document Verification**: The 83-page scanned base document `8_1732871406.pdf` (Legal Metrology Packaged Commodities Rules, 2011) was OCR-processed, structured, and validated.

- **Total Chunks in Base Document**: 126
- **English Chunks**: 106
- **Hindi Chunks**: 20
- **Detected Rules Count**: 30 distinct rules detected
- **Detected Rules List**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 27, 28...
- **Rule 6 (Declarations)**: FOUND
- **Rule 1 (Short Title & Commencement)**: FOUND
- **Rule 2 (Definitions)**: FOUND
- **Rule 3 (Scope & Applicability)**: FOUND
- **Rule 18 (Wholesale / Retail compliance)**: FOUND
- **Rule 32/34 (Penalty / Repeal & Savings)**: FOUND

## 6. Document Chunk Distribution (Top 25)

| Document ID | Chunks |
| :--- | :--- |
| `6_0_1732709495` | 1695 |
| `Gen Rules 6th Amendment Breath Analyser_1764862018` | 464 |
| `2026.4.8 Jan Vishwas Act 2026_1777014384` | 460 |
| `2025.4.21 Gas Meter General Rules_1746001659` | 331 |
| `6(iii)_1732709674` | 272 |
| `Gen Rule- Moisture Meters_1755669735` | 225 |
| `Jan Vishwas (Amendment of Provisions) Act, 2023 (18 of 2023)_1732708241` | 158 |
| `Sphygmomanomneters General Rules_1754973919` | 137 |
| `8_1732871406` | 126 |
| `Continuous_Electrical_Thermometer_1771307283` | 115 |
| `LM_General_Rule_Amendment_2026_1768192719` | 115 |
| `approval_of_models_rules_0_1732709311` | 71 |
| `Radar Equipment Gen Rules Amendment (1)_1746001628` | 70 |
| `stdrules-compressed_0_1732708966` | 69 |
| `267111_1761404639` | 45 |
| `gatc_1732710153` | 43 |
| `8(xii)_0_1732871346` | 41 |
| `2026.08.27 IST Rules_1788026429` | 36 |
| `9_1732872040` | 32 |
| `4_0_1732709211` | 24 |
| `GATC_Amendment_Rules_2026_1778476324` | 24 |
| `2023.12.29 Standard Operating Procedure for Edible oil & Fats Net Quantity Measurement signed copy_1732872010` | 15 |
| `6(ii)_0_1732709637` | 15 |
| `2023.10.6 amendment in PCR_1732871982` | 14 |
| `3_0_0_1732709063` | 13 |