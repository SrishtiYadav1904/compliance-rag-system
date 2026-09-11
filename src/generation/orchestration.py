import time
from typing import Optional, Dict, Any, List

from config.settings import settings
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.schemas import SearchResult
from .schemas import GroundedResponse, GenerationContext, Citation
from .language import LanguageDetector
from .translation import QueryTranslator
from .context import ContextValidator, ContextSelector
from .prompts import PromptBuilder
from .generator import BaseGenerator, GeneratorFactory
from .citation_verifier import CitationVerifier


class LegalRAGOrchestrator:
    """
    End-to-End Orchestrator for Grounded Multilingual Legal Metrology RAG.
    Connects:
    1. Language Detection ('en', 'hi', 'mr', 'ta')
    2. Adaptive Multilingual Retrieval ('auto', 'native', 'translation')
    3. Stage 6 Hybrid Retrieval (Dense + BM25 + RRF + Reranker)
    4. Evidence Validation & Abstention Layer
    5. Context Selection & Diversity
    6. Grounded Generation (Groq / Local Synthesizer)
    7. Post-Generation Citation Verification
    8. Diagnostic Latency Tracing
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[BaseGenerator] = None,
        language_detector: Optional[LanguageDetector] = None,
        translator: Optional[QueryTranslator] = None,
        context_validator: Optional[ContextValidator] = None,
        context_selector: Optional[ContextSelector] = None,
        citation_verifier: Optional[CitationVerifier] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        retrieval_mode: Optional[str] = None
    ):
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or GeneratorFactory.get_generator()
        self.lang_detector = language_detector or LanguageDetector()
        self.translator = translator or QueryTranslator()
        self.validator = context_validator or ContextValidator()
        self.selector = context_selector or ContextSelector()
        self.verifier = citation_verifier or CitationVerifier()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.retrieval_mode = retrieval_mode or settings.retrieval_mode

    def answer_query(
        self,
        query: str,
        retrieval_mode: Optional[str] = None,
        use_reranker: bool = True
    ) -> GroundedResponse:
        t_total_start = time.time()
        latencies: Dict[str, float] = {}
        diagnostics: Dict[str, Any] = {}

        # 1. Language Detection
        t0 = time.time()
        detected_lang, lang_conf = self.lang_detector.detect(query)
        latencies["language_detection_ms"] = round((time.time() - t0) * 1000, 2)
        diagnostics["detected_language"] = detected_lang
        diagnostics["language_confidence"] = lang_conf
        target_lang_name = self.lang_detector.get_language_name(detected_lang)

        # 2. Retrieval Strategy
        mode = retrieval_mode or self.retrieval_mode
        diagnostics["retrieval_mode"] = mode
        translation_used = False
        retrieval_query = query

        t_ret_start = time.time()
        candidates: List[SearchResult] = []

        if mode == "native" or detected_lang == "en":
            retrieval_query = query
            candidates = self.retriever.hybrid_search(retrieval_query, top_k=10, use_reranker=use_reranker)

        elif mode == "translation":
            retrieval_query, translation_used = self.translator.translate_query(query, detected_lang)
            candidates = self.retriever.hybrid_search(retrieval_query, top_k=10, use_reranker=use_reranker)

        elif mode == "auto":
            # Step A: Attempt native retrieval
            native_candidates = self.retriever.hybrid_search(query, top_k=10, use_reranker=use_reranker)
            top_native_score = native_candidates[0].score if native_candidates else 0.0

            # Step B: If non-English and native confidence is moderate/low, run translation retrieval
            if detected_lang != "en" and top_native_score < 0.030:
                trans_query, was_trans = self.translator.translate_query(query, detected_lang)
                if was_trans:
                    trans_candidates = self.retriever.hybrid_search(trans_query, top_k=10, use_reranker=use_reranker)
                    top_trans_score = trans_candidates[0].score if trans_candidates else 0.0

                    # Compare candidate sets
                    diagnostics["auto_native_score"] = top_native_score
                    diagnostics["auto_trans_score"] = top_trans_score

                    if top_trans_score >= top_native_score:
                        retrieval_query = trans_query
                        candidates = trans_candidates
                        translation_used = True
                    else:
                        retrieval_query = query
                        candidates = native_candidates
                        translation_used = False
                else:
                    retrieval_query = query
                    candidates = native_candidates
            else:
                retrieval_query = query
                candidates = native_candidates

        latencies["retrieval_and_reranking_ms"] = round((time.time() - t_ret_start) * 1000, 2)
        diagnostics["retrieval_query"] = retrieval_query
        diagnostics["translation_used"] = translation_used

        # 3. Context Validation (Abstention check)
        is_valid, failure_reason = self.validator.validate(query, candidates)
        if not is_valid:
            abstention_text = self.prompt_builder.get_abstention_message(detected_lang)
            latencies["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
            return GroundedResponse(
                answer=abstention_text,
                language=detected_lang,
                original_query=query,
                retrieval_query=retrieval_query,
                translation_used=translation_used,
                citations=[],
                retrieved_chunk_ids=[c.chunk_id for c in candidates[:3]],
                citation_verified=True,
                grounded=True,
                abstained=True,
                abstention_reason=failure_reason,
                latencies=latencies,
                diagnostics=diagnostics
            )

        # 4. Context Selection
        context: GenerationContext = self.selector.select(candidates)
        diagnostics["selected_chunk_count"] = len(context.selected_chunks)
        diagnostics["token_estimate"] = context.token_estimate

        # 5. Build Grounded Prompt
        prompt = self.prompt_builder.build_prompt(
            query=query,
            context=context,
            target_language_name=target_lang_name
        )

        # 6. LLM Generation
        t_gen_start = time.time()
        raw_answer = self.generator.generate(
            prompt=prompt,
            context=context,
            language=detected_lang
        )
        latencies["generation_ms"] = round((time.time() - t_gen_start) * 1000, 2)

        # 7. Post-Generation Citation Verification
        t_ver_start = time.time()
        verified_citations, all_verified, final_answer = self.verifier.verify_citations(
            text=raw_answer,
            context=context
        )
        latencies["citation_verification_ms"] = round((time.time() - t_ver_start) * 1000, 2)
        latencies["total_ms"] = round((time.time() - t_total_start) * 1000, 2)

        diagnostics["generator_provider"] = self.generator.provider_name
        diagnostics["generator_model"] = self.generator.model_name

        return GroundedResponse(
            answer=final_answer,
            language=detected_lang,
            original_query=query,
            retrieval_query=retrieval_query,
            translation_used=translation_used,
            citations=verified_citations,
            retrieved_chunk_ids=[c.chunk_id for c in context.selected_chunks],
            citation_verified=all_verified,
            grounded=True,
            abstained=False,
            latencies=latencies,
            diagnostics=diagnostics
        )
