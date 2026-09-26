# app/evaluate.py
import json
from pathlib import Path

from app.rag import retrieve_raw, retrieve, get_all_chunks_meta
from app.baseline import keyword_search
from app.config import TOP_K


GROUND_TRUTH = {
    "Engine overheating after 30 minutes of operation": ("SY335_Maintenance_EN.pdf", 66),
    "Boom loses hydraulic pressure after warm up": ("SY335_Maintenance_EN.pdf", 93),
    "How to relieve hydraulic system pressure safely": ("SY335_Maintenance_EN.pdf", 93),
    "Daily inspection checklist before operation": ("SY335_Maintenance_EN.pdf", 56),
    "Accumulator nitrogen pressure procedure": ("SY335_Operator_Manual.pdf", 277),
    "Hydraulic oil temperature too high": ("SY335_Operator_Manual.pdf", 197),
    "Track tension adjustment procedure": ("SY335_Maintenance_EN.pdf", 101),
    "Abnormal noise from swing motor": ("SY335_Maintenance_EN.pdf", 96),
    "Machine won't start, what to check": ("SY335_Operator_Manual.pdf", 295),
    "Hydraulic leak near boom cylinder": ("SY335_Maintenance_EN.pdf", 113),
}


def hits_and_mrr(hits, expected_file, expected_page, page_tol=0):
    hit1 = 0
    hit3 = 0
    hit5 = 0
    rr = 0.0
    for i, h in enumerate(hits[:5]):
        same_file = h["source_file"] == expected_file
        close_page = abs(int(h["page"]) - expected_page) <= page_tol
        if same_file and close_page:
            if i == 0:
                hit1 = 1
            if i < 3:
                hit3 = 1
            hit5 = 1
            rr = 1.0 / (i + 1)
            break
    return hit1, hit3, hit5, rr


def _agg_init():
    return {"h1": 0.0, "h3": 0.0, "h5": 0.0, "mrr": 0.0}


def _agg_add(agg, h1, h3, h5, rr):
    agg["h1"] += h1
    agg["h3"] += h3
    agg["h5"] += h5
    agg["mrr"] += rr


def _agg_print(name, strict, lenient, n):
    print(f"\n== {name} ==")
    print(f"{'Metric':<12} {'Strict':>10} {'Lenient±1':>12}")
    print("-" * 36)
    print(f"{'Hit@1':<12} {strict['h1']/n:>10.3f} {lenient['h1']/n:>12.3f}")
    print(f"{'Hit@3':<12} {strict['h3']/n:>10.3f} {lenient['h3']/n:>12.3f}")
    print(f"{'Hit@5':<12} {strict['h5']/n:>10.3f} {lenient['h5']/n:>12.3f}")
    print(f"{'MRR':<12} {strict['mrr']/n:>10.3f} {lenient['mrr']/n:>12.3f}")


def main():
    meta = get_all_chunks_meta()
    n = len(GROUND_TRUTH)

    rag_raw_strict = _agg_init()
    rag_raw_lenient = _agg_init()

    rag_ui_strict = _agg_init()
    rag_ui_lenient = _agg_init()

    kw_strict = _agg_init()
    kw_lenient = _agg_init()

    rows = []

    print("\n=== PER-QUERY RESULTS ===\n")

    for q, (exp_file, exp_page) in GROUND_TRUTH.items():
        # RAG raw (no preview filter; optional: dedupe=False for eval)
        rag_raw_hits = retrieve_raw(q, k=TOP_K, k_search=max(24, TOP_K), dedupe=False)

        # RAG UI (your app uses this)
        rag_ui_hits = retrieve(q, k=TOP_K)

        # Keyword baseline
        kw_hits = keyword_search(q, meta=meta, k=TOP_K)

        # Strict
        rrs_h1, rrs_h3, rrs_h5, rrs_rr = hits_and_mrr(rag_raw_hits, exp_file, exp_page, page_tol=0)
        rui_h1, rui_h3, rui_h5, rui_rr = hits_and_mrr(rag_ui_hits, exp_file, exp_page, page_tol=0)
        k_h1, k_h3, k_h5, k_rr = hits_and_mrr(kw_hits, exp_file, exp_page, page_tol=0)

        _agg_add(rag_raw_strict, rrs_h1, rrs_h3, rrs_h5, rrs_rr)
        _agg_add(rag_ui_strict, rui_h1, rui_h3, rui_h5, rui_rr)
        _agg_add(kw_strict, k_h1, k_h3, k_h5, k_rr)

        # Lenient ±1 page
        rrl_h1, rrl_h3, rrl_h5, rrl_rr = hits_and_mrr(rag_raw_hits, exp_file, exp_page, page_tol=1)
        rul_h1, rul_h3, rul_h5, rul_rr = hits_and_mrr(rag_ui_hits, exp_file, exp_page, page_tol=1)
        kl_h1, kl_h3, kl_h5, kl_rr = hits_and_mrr(kw_hits, exp_file, exp_page, page_tol=1)

        _agg_add(rag_raw_lenient, rrl_h1, rrl_h3, rrl_h5, rrl_rr)
        _agg_add(rag_ui_lenient, rul_h1, rul_h3, rul_h5, rul_rr)
        _agg_add(kw_lenient, kl_h1, kl_h3, kl_h5, kl_rr)

        def top1_str(hits):
            if not hits:
                return "NONE"
            h = hits[0]
            return f"{h['source_file']} p{h['page']} (score {h['score']:.3f})"

        print(f"Q: {q}")
        print(f"   Expected : {exp_file} p{exp_page}")
        print(f"   RAG RAW  : {top1_str(rag_raw_hits)}  strict_hit={rrs_h1} lenient_hit={rrl_h1}")
        print(f"   RAG UI   : {top1_str(rag_ui_hits)}   strict_hit={rui_h1} lenient_hit={rul_h1}")
        print(f"   KW       : {top1_str(kw_hits)}       strict_hit={k_h1} lenient_hit={kl_h1}")
        print()

        rows.append({
            "query": q,
            "expected_file": exp_file,
            "expected_page": exp_page,
            "rag_raw_top1": top1_str(rag_raw_hits),
            "rag_ui_top1": top1_str(rag_ui_hits),
            "kw_top1": top1_str(kw_hits),
            "rag_raw_strict": {"hit1": rrs_h1, "hit3": rrs_h3, "hit5": rrs_h5, "mrr": round(rrs_rr, 4)},
            "rag_ui_strict": {"hit1": rui_h1, "hit3": rui_h3, "hit5": rui_h5, "mrr": round(rui_rr, 4)},
            "kw_strict": {"hit1": k_h1, "hit3": k_h3, "hit5": k_h5, "mrr": round(k_rr, 4)},
            "rag_raw_lenient": {"hit1": rrl_h1, "hit3": rrl_h3, "hit5": rrl_h5, "mrr": round(rrl_rr, 4)},
            "rag_ui_lenient": {"hit1": rul_h1, "hit3": rul_h3, "hit5": rul_h5, "mrr": round(rul_rr, 4)},
            "kw_lenient": {"hit1": kl_h1, "hit3": kl_h3, "hit5": kl_h5, "mrr": round(kl_rr, 4)},
        })

    print("\n=== AGGREGATE METRICS ===")
    _agg_print("RAG (RAW, evaluation)", rag_raw_strict, rag_raw_lenient, n)
    _agg_print("RAG (UI filtered)", rag_ui_strict, rag_ui_lenient, n)
    _agg_print("Keyword baseline", kw_strict, kw_lenient, n)

    out_path = Path("logs") / "evaluation_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "RAG_RAW": {
            "strict": {"hit@1": rag_raw_strict["h1"]/n, "hit@3": rag_raw_strict["h3"]/n, "hit@5": rag_raw_strict["h5"]/n, "mrr": rag_raw_strict["mrr"]/n},
            "lenient_pm1": {"hit@1": rag_raw_lenient["h1"]/n, "hit@3": rag_raw_lenient["h3"]/n, "hit@5": rag_raw_lenient["h5"]/n, "mrr": rag_raw_lenient["mrr"]/n},
        },
        "RAG_UI": {
            "strict": {"hit@1": rag_ui_strict["h1"]/n, "hit@3": rag_ui_strict["h3"]/n, "hit@5": rag_ui_strict["h5"]/n, "mrr": rag_ui_strict["mrr"]/n},
            "lenient_pm1": {"hit@1": rag_ui_lenient["h1"]/n, "hit@3": rag_ui_lenient["h3"]/n, "hit@5": rag_ui_lenient["h5"]/n, "mrr": rag_ui_lenient["mrr"]/n},
        },
        "Keyword": {
            "strict": {"hit@1": kw_strict["h1"]/n, "hit@3": kw_strict["h3"]/n, "hit@5": kw_strict["h5"]/n, "mrr": kw_strict["mrr"]/n},
            "lenient_pm1": {"hit@1": kw_lenient["h1"]/n, "hit@3": kw_lenient["h3"]/n, "hit@5": kw_lenient["h5"]/n, "mrr": kw_lenient["mrr"]/n},
        },
        "per_query": rows,
    }, indent=2), encoding="utf-8")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()