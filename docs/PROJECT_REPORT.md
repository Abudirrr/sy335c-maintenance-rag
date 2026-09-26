# SY335C maintenance assistant technical report

Author: Abdelrahman Ramadan

This public summary accompanies a local RAG prototype developed for the SANY SY335C excavator. The original academic paper describes its design, implementation, evaluation, and iteration during 2025–2026.

## Problem

Operators and maintenance staff need to locate relevant information inside lengthy technical manuals. The project explores whether natural-language retrieval and generated summaries can make those sources easier to navigate while preserving page references.

## Implementation

Python ingestion extracts PDF text with pypdf, normalizes whitespace and broken hyphenation, and builds page-aware chunks. Sentence Transformers supplies MiniLM embeddings, stored in a FAISS index with metadata. Queries receive subsystem-specific expansion, followed by semantic candidate retrieval and keyword-overlap reranking. The browser path deduplicates evidence by source page.

FLAN-T5 generates structured responses locally. Filtering and an extractive fallback handle some malformed or generic output. FastAPI and Jinja2 present the question form, response, retrieval score label, citations, and expandable evidence.

## Evaluation

The work includes keyword baselines, page-retrieval metrics, and a five-question procedural evaluation. Recorded final aggregate values are Precision@8 0.225, step coverage 0.200, and unmatched steps per question 1.600. The small sample and heuristic matching restrict interpretation; see EVALUATION.md.

## Engineering lessons

Working ingestion, indexing, and an interface do not establish answer reliability. Retrieval can find broadly relevant pages while missing the exact procedure. A short generation context can omit critical details, and an apparently strong normalized retrieval score can coexist with an incorrect answer. Evaluation and evidence inspection are therefore central parts of this prototype.

## Future work

Improve document structure and OCR support, expand the evidence context, compare rerankers, evaluate with expert annotations, and validate abstention before considering operational use.

## Publication scope

This summary replaces the original academic document in the public source package. It omits student identifiers, private contact details, and unreviewed screenshots. The academic origin is retained without naming the repository “capstone.”
