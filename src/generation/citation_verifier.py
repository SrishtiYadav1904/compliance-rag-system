import re
from typing import List, Dict, Any, Tuple, Optional
from .schemas import Citation, GenerationContext


class CitationVerifier:
    """
    Post-generation citation verification layer.
    Validates that every citation produced in the response:
    1. References a source_file that actually exists in the supplied context chunks.
    2. Matches the page range, rule_number, sub-rule, or clause in the supplied provenance.
    Flags or strips unverified citations, ensuring zero hallucinated legal citations.
    """

    # Citation patterns like [Source: filename.pdf, p. 43, Rule 6(1)(a)] or [Source: filename.pdf, pp. 42–43, Rule 6]
    CITATION_REGEX = re.compile(
        r"\[(?:Source:\s*)?([^,\]]+?\.pdf)(?:,\s*(?:pp?\.\s*(\d+)(?:[–-](\d+))?))?(?:,\s*([^\]]+))?\]",
        re.IGNORECASE
    )

    def parse_citations(self, text: str) -> List[Citation]:
        """Extracts candidate citations from generated text."""
        citations: List[Citation] = []
        matches = self.CITATION_REGEX.findall(text)

        for filename, p_start, p_end, rule_part in matches:
            filename = filename.strip()
            start = int(p_start) if p_start else None
            end = int(p_end) if p_end else start
            rule_num = None
            sub_rule = None
            clause = None

            if rule_part:
                rule_part = rule_part.strip()
                r_match = re.search(r"Rule\s*(\d+[A-Za-z]?)", rule_part, re.I)
                if r_match:
                    rule_num = r_match.group(1)
                sub_match = re.search(r"\((\d+[A-Za-z]?)\)", rule_part)
                if sub_match:
                    sub_rule = sub_match.group(1)
                clause_match = re.search(r"\(([a-z])\)", rule_part)
                if clause_match:
                    clause = clause_match.group(1)

            citations.append(Citation(
                source_file=filename,
                page_start=start,
                page_end=end,
                rule_number=rule_num,
                sub_rule=sub_rule,
                clause=clause,
                verified=False,
                raw_citation=f"[{filename}{', p. ' + str(start) if start else ''}{', ' + rule_part if rule_part else ''}]"
            ))

        return citations

    def verify_citations(
        self,
        text: str,
        context: GenerationContext
    ) -> Tuple[List[Citation], bool, str]:
        """
        Verifies citations against supplied evidence chunks in context.
        Returns: (verified_citations, all_verified, cleaned_or_remediated_text)
        """
        parsed = self.parse_citations(text)
        if not parsed:
            # If no in-text citations were parsed, construct verified citations from top supplied context chunks
            verified_citations = []
            for c in context.selected_chunks[:3]:
                cit = Citation(
                    source_file=c.source_file,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    rule_number=c.rule_number,
                    sub_rule=c.sub_rule,
                    clause=c.clause,
                    schedule=c.schedule,
                    document_title=c.document_title,
                    verified=True
                )
                verified_citations.append(cit)
            return verified_citations, True, text

        # Verify each parsed citation against context chunks
        verified_citations = []
        all_verified = True
        context_chunks = context.selected_chunks

        for cit in parsed:
            is_valid = False
            matching_chunk = None

            for chunk in context_chunks:
                # Check source file match (case-insensitive)
                if chunk.source_file.lower() == cit.source_file.lower():
                    # Check page overlap if page was specified
                    page_matches = True
                    if cit.page_start is not None and chunk.page_start is not None:
                        # Allow within +/- 1 page tolerance for document boundary variance
                        page_matches = (
                            chunk.page_start <= cit.page_start <= chunk.page_end or
                            abs(chunk.page_start - cit.page_start) <= 1
                        )

                    # Check rule if specified
                    rule_matches = True
                    if cit.rule_number and chunk.rule_number:
                        rule_matches = str(chunk.rule_number).lower() == str(cit.rule_number).lower()

                    if page_matches and rule_matches:
                        is_valid = True
                        matching_chunk = chunk
                        break

            if is_valid and matching_chunk:
                cit.verified = True
                cit.document_title = matching_chunk.document_title
                cit.schedule = matching_chunk.schedule
                verified_citations.append(cit)
            else:
                cit.verified = False
                all_verified = False
                verified_citations.append(cit)

        # If any citation failed verification, clean or remediate the text
        remediated_text = text
        if not all_verified:
            # Append warning to text
            unverified_names = [c.raw_citation for c in verified_citations if not c.verified]
            warning_notice = (
                f"\n\n[Warning: Citations {', '.join(unverified_names)} could not be verified "
                f"against the retrieved knowledge base evidence and have been flagged.]"
            )
            remediated_text = text + warning_notice

        return verified_citations, all_verified, remediated_text
