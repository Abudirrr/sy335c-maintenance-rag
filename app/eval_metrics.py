import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from app.rag import retrieve
from app.baseline import keyword_search
from app.rag import get_all_chunks_meta


def _normalize_expected(expected: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """
    expected = [{"source_file": "...pdf", "page": 123}, ...]
    returns list of tuples for easy matching.
    """
    out = []
    for e in expected:
        sf = str(e["source_file"]).strip()
        pg = int(e["page"])
        out.append((sf, pg))
    return out


def _rank_of_expected(results: List[Dict[str, Any]], expected_pairs: List[Tuple[str, int]]) -> int:
    """
    Returns 1-based rank where an expected (source_file, page) appears in results.
    If not found, returns 0.
    """
    for i, r in enumerate(results, start=1):
        pair = (str(r["source_file"]).strip(), int(r["page"]))
        if pair in expected_pairs:
            return i
    return 0


def _metrics_from_ranks(ranks: List[int], k: int) -> Dict[str, float]:
    """
    ranks: list of 1-based ranks, 0 means not found
    """
    n = len(ranks)
    hit_at_k = sum(1 for r in ranks if 1 <= r <= k) / n if n else 0.0
    hit_at_1 = sum(1 for r in ranks if r == 1) / n if n else 0.0
    mrr = sum((1.0 / r) for r in ranks if r > 0) / n if n else 0.0
    return {
        "n": n,
        "hit@1": round(hit_at_1, 4),
        f"hit@{k}": round(hit_at_k, 4),
        "mrr": round(mrr, 4),
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    testset_path = project_root / "app" / "testset.json"

    if not testset_path.exists():
        raise FileNotFoundError(f"Missing testset.json at: {testset_path}")

    testset = json.loads(testset_path.read_text(encoding="utf-8"))
    if not isinstance(testset, list) or len(testset) == 0:
        raise ValueError("testset.json must be a non-empty list")

    # Load all chunks once for keyword baseline
    all_meta = get_all_chunks_meta()

    rag_ranks_top5: List[int] = []
    kw_ranks_top5: List[int] = []

    detailed_rows = []

    TOPK = 5

    for item in testset:
        query = item["query"]
        expected_pairs = _normalize_expected(item.get("expected", []))

        rag_results = retrieve(query, k=TOPK)
        kw_results = keyword_search(query, meta=all_meta, k=TOPK)

        rag_rank = _rank_of_expected(rag_results, expected_pairs)
        kw_rank = _rank_of_expected(kw_results, expected_pairs)

        rag_ranks_top5.append(rag_rank)
        kw_ranks_top5.append(kw_rank)

        detailed_rows.append(
            {
                "query": query,
                "expected": [{"source_file": sf, "page": pg} for (sf, pg) in expected_pairs],
                "rag_rank": rag_rank,
                "kw_rank": kw_rank,
                "rag_top1": (
                    {"source_file": rag_results[0]["source_file"], "page": rag_results[0]["page"], "score": rag_results[0]["score"]}
                    if rag_results else None
                ),
                "kw_top1": (
                    {"source_file": kw_results[0]["source_file"], "page": kw_results[0]["page"], "score": kw_results[0]["score"]}
                    if kw_results else None
                ),
            }
        )

    rag_metrics = _metrics_from_ranks(rag_ranks_top5, k=TOPK)
    kw_metrics = _metrics_from_ranks(kw_ranks_top5, k=TOPK)

    out_dir = project_root / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "evaluation_metrics.json"
    details_path = out_dir / "evaluation_details.json"

    metrics_payload = {
        "topk": TOPK,
        "rag": rag_metrics,
        "keyword": kw_metrics,
    }

    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    details_path.write_text(json.dumps(detailed_rows, indent=2), encoding="utf-8")

    print("Saved:", metrics_path)
    print("Saved:", details_path)

    print("\n=== Metrics (Top-1 / Top-5 / MRR) ===")
    print("RAG    :", rag_metrics)
    print("Keyword:", kw_metrics)

    print("\n=== Per-query ranks (0 means not found in top-5) ===")
    for row in detailed_rows:
        print(f"- {row['query']}")
        print(f"  Expected: {row['expected']}")
        print(f"  RAG rank: {row['rag_rank']} | top1: {row['rag_top1']}")
        print(f"  KW  rank: {row['kw_rank']} | top1: {row['kw_top1']}")


if __name__ == "__main__":
    main()
