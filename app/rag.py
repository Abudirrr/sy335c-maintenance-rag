# app/rag.py
import json
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.config import VECTORDB_DIR, EMBED_MODEL_NAME, TOP_K
from app.llm import generate_structured_answer

_model = None
_index = None
_meta = None

# --- Retrieval tuning ---
RAW_CANDIDATES = 64
MIN_SCORE_FOR_LLM = 0.28
MIN_SCORE_FOR_PREVIEW = 0.16

# Hybrid weights
SEMANTIC_WEIGHT = 0.75
KEYWORD_WEIGHT = 0.25

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "been", "being", "what", "how", "should", "check", "after",
    "before", "when", "why", "does", "do", "it", "this", "that", "i", "me", "my",
    "we", "our", "you", "your", "can", "could", "would", "first", "procedure", "steps",
}


def _load():
    """Lazy-load embedding model, FAISS index, and metadata only once."""
    global _model, _index, _meta

    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)

    if _index is None or _meta is None:
        idx_path = VECTORDB_DIR / "index.faiss"
        meta_path = VECTORDB_DIR / "meta.json"
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Vector DB missing. Expected {idx_path} and {meta_path}. "
                "Run: python -m app.ingest"
            )

        _index = faiss.read_index(str(idx_path))
        _meta = json.loads(meta_path.read_text(encoding="utf-8"))


def get_all_chunks_meta() -> List[Dict[str, Any]]:
    """Return full chunk metadata list."""
    _load()
    return _meta


def _dedupe_by_page(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If multiple chunks come from the same source_file+page,
    keep only the highest-scoring one.
    """
    best: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in results:
        key = (r["source_file"], int(r["page"]))
        if key not in best or r["score"] > best[key]["score"]:
            best[key] = r

    deduped = list(best.values())
    deduped.sort(key=lambda x: x["score"], reverse=True)
    return deduped


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _keyword_overlap_score(query: str, text: str) -> float:
    """
    Simple keyword overlap score on candidate chunk text.
    """
    q_tokens = {t for t in _tokenize(query) if t not in STOPWORDS}
    t_tokens = {t for t in _tokenize(text) if t not in STOPWORDS}

    if not q_tokens or not t_tokens:
        return 0.0

    return len(q_tokens.intersection(t_tokens)) / len(q_tokens)


def _expand_query(query: str) -> str:
    """
    Light query expansion to improve retrieval for maintenance manuals.
    """
    q = query.lower()
    extra = []

    if "hydraulic" in q or "pressure" in q or "boom" in q:
        extra += ["hydraulic", "pressure", "relief valve", "accumulator", "control lever", "manual"]
    if "engine" in q or "overheat" in q or "temperature" in q:
        extra += ["engine", "cooling", "radiator", "fan", "coolant", "manual"]
    if "start" in q or "battery" in q or "starter" in q:
        extra += ["battery", "starter", "fuel system", "electrical", "manual"]
    if "track" in q or "tension" in q or "undercarriage" in q:
        extra += ["track", "tension", "grease", "undercarriage", "manual"]
    if "safe" in q or "safely" in q or "warning" in q:
        extra += ["safety", "warning", "procedure", "manual"]

    if extra:
        return query + " " + " ".join(extra)
    return query


def _normalize_scores(values: List[float]) -> List[float]:
    """
    Min-max normalize scores to 0..1.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def retrieve_raw(
    query: str,
    k: int = TOP_K,
    k_search: Optional[int] = None,
    dedupe: bool = False,
) -> List[Dict[str, Any]]:
    """
    Raw retrieval for evaluation/analysis:
    - uses semantic retrieval first
    - then reranks candidates with semantic + keyword overlap
    - no score filtering
    - optional dedupe
    """
    _load()

    expanded_query = _expand_query(query)

    q_emb = _model.encode([expanded_query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    k_search = k_search or max(int(k), RAW_CANDIDATES)
    semantic_scores, ids = _index.search(q_emb, k_search)

    candidates: List[Dict[str, Any]] = []
    for sem_score, idx in zip(semantic_scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue

        m = _meta[idx]
        text = m.get("text", "")
        kw_score = _keyword_overlap_score(expanded_query, text)

        candidates.append(
            {
                "semantic_score": float(sem_score),
                "keyword_score": float(kw_score),
                "source_file": m.get("source_file", "unknown"),
                "page": int(m.get("page", -1)),
                "id": m.get("id", str(idx)),
                "text": text,
            }
        )

    if not candidates:
        return []

    # Normalize and combine scores
    sem_norm = _normalize_scores([c["semantic_score"] for c in candidates])
    kw_norm = _normalize_scores([c["keyword_score"] for c in candidates])

    reranked: List[Dict[str, Any]] = []
    for c, s_norm, k_norm in zip(candidates, sem_norm, kw_norm):
        final_score = SEMANTIC_WEIGHT * s_norm + KEYWORD_WEIGHT * k_norm
        reranked.append(
            {
                "score": float(final_score),
                "semantic_score": c["semantic_score"],
                "keyword_score": c["keyword_score"],
                "source_file": c["source_file"],
                "page": c["page"],
                "id": c["id"],
                "text": c["text"],
            }
        )

    reranked.sort(key=lambda x: x["score"], reverse=True)

    if dedupe:
        reranked = _dedupe_by_page(reranked)

    return reranked[: int(k)]


def retrieve(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    UI-friendly retrieval:
    - fetch more candidates
    - rerank
    - dedupe by page
    - filter weak scores for preview
    """
    raw = retrieve_raw(
        query,
        k=max(int(k), RAW_CANDIDATES),
        k_search=RAW_CANDIDATES,
        dedupe=True,
    )

    preview_ok = [r for r in raw if r["score"] >= MIN_SCORE_FOR_PREVIEW]
    return preview_ok[: int(k)]


def _confidence_label(top_score: float) -> str:
    if top_score >= 0.55:
        return "high"
    if top_score >= 0.35:
        return "medium"
    return "low"


def draft_grounded_answer(query: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build final response object:
    - always returns citations + evidence preview
    - only sends sufficiently strong evidence to the LLM
    """
    if not evidence:
        return {
            "query": query,
            "answer_text": (
                "I couldn't find relevant manual sections for this query.\n\n"
                "Try rephrasing with subsystem keywords (engine/cooling/hydraulic/electrical), "
                "include when it happens (cold start/warm-up/after 30 min), and add symptoms "
                "(alarm code, temperature high, fan not running, coolant level, etc.)."
            ),
            "citations": [],
            "evidence_preview": [],
            "top_score": 0.0,
            "confidence": "low",
        }

    top_score = float(evidence[0]["score"])
    confidence = _confidence_label(top_score)

    citations = [
        {
            "source_file": e["source_file"],
            "page": e["page"],
            "score": e["score"],
            "id": e["id"],
        }
        for e in evidence
    ]

    evidence_preview = [
        {
            "source_file": e["source_file"],
            "page": e["page"],
            "score": e["score"],
            "id": e["id"],
            "snippet": (e["text"][:350] + "...") if len(e["text"]) > 350 else e["text"],
        }
        for e in evidence
    ]

    llm_evidence = [e for e in evidence if e["score"] >= MIN_SCORE_FOR_LLM]

    if not llm_evidence:
        return {
            "query": query,
            "answer_text": (
                "Insufficient evidence found in retrieved manual sections.\n\n"
                f"Top retrieval score is {top_score:.3f} (weak match).\n"
                "Try adding specific keywords like:\n"
                "- cooling system / radiator / fan / coolant / thermostat\n"
                "- engine temperature high / overheat / alarm code\n"
                "- hydraulic pressure / relief valve / warm-up\n"
                "- track tension / grease / undercarriage\n"
                "- starter / battery / fuel system\n"
            ),
            "citations": citations,
            "evidence_preview": evidence_preview,
            "top_score": top_score,
            "confidence": confidence,
        }

    answer_text = generate_structured_answer(query, llm_evidence)

    return {
        "query": query,
        "answer_text": answer_text,
        "citations": citations,
        "evidence_preview": evidence_preview,
        "top_score": top_score,
        "confidence": confidence,
    }