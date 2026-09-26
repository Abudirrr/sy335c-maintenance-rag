# app/evaluate_steps.py
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from app.rag import retrieve_raw, draft_grounded_answer
from app.config import TOP_K


# -------------------------------------------------
# GOLD DATASET
# Keep wording short and close to likely manual wording
# -------------------------------------------------
STRUCTURED_GOLD = {
    "How to relieve hydraulic system pressure safely?": {
        "expected_file": "SY335_Maintenance_EN.pdf",
        "expected_page": 93,
        "expected_steps": [
            "stop the engine",
            "operate the control levers",
            "release hydraulic pressure",
            "release accumulator pressure",
        ],
    },
    "Track tension adjustment procedure for SY335C.": {
        "expected_file": "SY335_Maintenance_EN.pdf",
        "expected_page": 101,
        "expected_steps": [
            "park the machine",
            "check track tension",
            "add grease",
            "release grease",
            "recheck track tension",
        ],
    },
    "Engine overheating after 30 minutes of operation. What should I check?": {
        "expected_file": "SY335_Maintenance_EN.pdf",
        "expected_page": 66,
        "expected_steps": [
            "check engine oil",
            "replace the engine oil filter",
            "check the fan belt",
        ],
    },
    "Boom loses hydraulic pressure after warm-up.": {
        "expected_file": "SY335_Maintenance_EN.pdf",
        "expected_page": 93,
        "expected_steps": [
            "check hydraulic pressure",
            "inspect the hydraulic system",
            "check the relief valve",
        ],
    },
    "Machine won't start. What should I check first?": {
        "expected_file": "SY335_Operator_Manual.pdf",
        "expected_page": 295,
        "expected_steps": [
            "check the battery",
            "check the starter",
            "check the fuel system",
        ],
    },
}


# -------------------------------------------------
# TEXT HELPERS
# -------------------------------------------------
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_set(text: str) -> set:
    return set(normalize_text(text).split())


def overlap_score(a: str, b: str) -> float:
    """
    More forgiving token overlap similarity.
    Uses intersection / smaller token set size.
    """
    a_tokens = token_set(a)
    b_tokens = token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens.intersection(b_tokens)) / max(1, min(len(a_tokens), len(b_tokens)))


# -------------------------------------------------
# STEP EXTRACTION FROM GENERATED ANSWER
# -------------------------------------------------
def extract_procedure_section(answer_text: str) -> str:
    """
    Pull text between:
    ### Procedure
    and next ### heading
    """
    if not answer_text:
        return ""

    m = re.search(
        r"### Procedure\s*(.*?)\s*(### Safety Warnings|### Notes / Checks|### Citations|$)",
        answer_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return ""
    return m.group(1).strip()


def extract_steps(answer_text: str) -> List[str]:
    """
    Extract numbered or bullet-like steps from the Procedure section.
    """
    proc = extract_procedure_section(answer_text)
    if not proc:
        return []

    lines = [line.strip() for line in proc.splitlines() if line.strip()]
    steps = []

    for line in lines:
        # numbered: 1) step / 1. step / 1: step
        line = re.sub(r"^\d+\s*[\)\.:-]\s*", "", line).strip()
        # bullet: - step / * step
        line = re.sub(r"^[-*]\s*", "", line).strip()

        if not line:
            continue

        low = normalize_text(line)
        if low in {
            "not found",
            "not found in context",
            "refer to cited manual sections",
            "none found",
            "none found in context",
        }:
            continue

        steps.append(line)

    # fallback: if procedure is one useful sentence and not empty
    if not steps and proc:
        low = normalize_text(proc)
        if low not in {
            "not found",
            "not found in context",
            "refer to cited manual sections",
            "none found",
            "none found in context",
        }:
            steps = [proc]

    return steps


# -------------------------------------------------
# RETRIEVAL METRICS
# -------------------------------------------------
def is_relevant_hit(hit: Dict[str, Any], expected_file: str, expected_page: int, page_tol: int = 1) -> bool:
    return (
        hit["source_file"] == expected_file
        and abs(int(hit["page"]) - int(expected_page)) <= page_tol
    )


def precision_at_k(
    hits: List[Dict[str, Any]],
    expected_file: str,
    expected_page: int,
    k: int = 5,
    page_tol: int = 1,
) -> float:
    top = hits[:k]
    if not top:
        return 0.0
    relevant = sum(1 for h in top if is_relevant_hit(h, expected_file, expected_page, page_tol=page_tol))
    return relevant / len(top)


# -------------------------------------------------
# STEP COVERAGE + HALLUCINATION METRICS
# -------------------------------------------------
def match_generated_to_expected(
    generated_steps: List[str],
    expected_steps: List[str],
    threshold: float = 0.34,
) -> Tuple[int, int, List[str], List[str]]:
    """
    Returns:
    - correct_count
    - hallucinated_count
    - matched_expected_steps
    - hallucinated_generated_steps

    Rule:
    A generated step counts as correct if it overlaps sufficiently
    with any unmatched expected step.
    Otherwise it is hallucinated.
    """
    matched_expected = set()
    correct = 0
    hallucinated = 0
    matched_expected_texts = []
    hallucinated_generated = []

    for g in generated_steps:
        best_idx = None
        best_score = 0.0

        for i, e in enumerate(expected_steps):
            if i in matched_expected:
                continue
            score = overlap_score(e, g)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None and best_score >= threshold:
            matched_expected.add(best_idx)
            correct += 1
            matched_expected_texts.append(expected_steps[best_idx])
        else:
            hallucinated += 1
            hallucinated_generated.append(g)

    return correct, hallucinated, matched_expected_texts, hallucinated_generated


# -------------------------------------------------
# MAIN EVALUATION
# -------------------------------------------------
def main():
    results = []

    total_questions = len(STRUCTURED_GOLD)
    total_precision = 0.0
    total_coverage = 0.0
    total_hallucinations = 0

    print("\n=== STRUCTURED EVALUATION ===\n")

    for query, item in STRUCTURED_GOLD.items():
        expected_file = item["expected_file"]
        expected_page = item["expected_page"]
        expected_steps = item["expected_steps"]

        # Raw retrieval for IR metrics
        hits = retrieve_raw(query, k=TOP_K, k_search=max(48, TOP_K), dedupe=False)
        p_at_k = precision_at_k(hits, expected_file, expected_page, k=TOP_K, page_tol=1)

        # Use top retrieved chunks as evidence for answer generation
        ui_evidence = hits[:TOP_K]
        result_obj = draft_grounded_answer(query, ui_evidence)
        answer_text = result_obj["answer_text"]

        generated_steps = extract_steps(answer_text)

        correct_count, hallucinated_count, matched_expected, hallucinated_generated = match_generated_to_expected(
            generated_steps,
            expected_steps,
            threshold=0.34,
        )

        coverage = correct_count / max(1, len(expected_steps))

        total_precision += p_at_k
        total_coverage += coverage
        total_hallucinations += hallucinated_count

        top1 = "NONE"
        if hits:
            top1 = f"{hits[0]['source_file']} p{hits[0]['page']} (score {hits[0]['score']:.3f})"

        print(f"Q: {query}")
        print(f"   Expected page      : {expected_file} p{expected_page}")
        print(f"   Top1 retrieval     : {top1}")
        print(f"   Precision@{TOP_K}        : {p_at_k:.3f}")
        print(f"   Expected steps     : {len(expected_steps)}")
        print(f"   Generated steps    : {len(generated_steps)}")
        print(f"   Correct steps      : {correct_count}")
        print(f"   Coverage           : {coverage:.3f}")
        print(f"   Hallucinated steps : {hallucinated_count}")
        print()

        results.append({
            "query": query,
            "expected_file": expected_file,
            "expected_page": expected_page,
            "top1_retrieval": top1,
            f"precision@{TOP_K}": round(p_at_k, 4),
            "expected_steps": expected_steps,
            "generated_steps": generated_steps,
            "correct_step_count": correct_count,
            "coverage": round(coverage, 4),
            "hallucinated_step_count": hallucinated_count,
            "matched_expected_steps": matched_expected,
            "hallucinated_generated_steps": hallucinated_generated,
        })

    avg_precision = total_precision / max(1, total_questions)
    avg_coverage = total_coverage / max(1, total_questions)
    avg_hallucinations = total_hallucinations / max(1, total_questions)

    print("=== AGGREGATE RESULTS ===\n")
    print(f"Average Precision@{TOP_K}: {avg_precision:.3f}")
    print(f"Average Step Coverage : {avg_coverage:.3f}")
    print(f"Average Hallucinated Steps / Question: {avg_hallucinations:.3f}")

    out = {
        "summary": {
            f"avg_precision@{TOP_K}": avg_precision,
            "avg_step_coverage": avg_coverage,
            "avg_hallucinated_steps_per_question": avg_hallucinations,
            "num_questions": total_questions,
        },
        "per_question": results,
    }

    out_path = Path("logs") / "structured_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()