# app/baseline.py
import re
from typing import List, Dict, Any

STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were",
    "what","how","should","check","after","before","when","why","does","do","it","this",
    "too","high","low","from","near"
}

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)

def keyword_search(query: str, meta: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """
    Simple keyword baseline (improved):
    - token overlap count excluding stopwords
    """
    q_tokens = {t for t in _tokenize(query) if t not in STOP}
    if not q_tokens:
        return []

    scored = []
    for m in meta:
        t_tokens = {t for t in _tokenize(m.get("text", "")) if t not in STOP}
        score = len(q_tokens.intersection(t_tokens))
        if score > 0:
            scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    results = []
    for score, m in top:
        results.append({
            "score": float(score),
            "source_file": m.get("source_file", "unknown"),
            "page": int(m.get("page", -1)),
            "id": m.get("id", ""),
            "text": m.get("text", ""),
        })
    return results