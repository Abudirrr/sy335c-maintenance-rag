from __future__ import annotations

from functools import lru_cache
from typing import List, Dict, Any
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

GEN_MODEL_NAME = "google/flan-t5-large"


@lru_cache(maxsize=1)
def _load_model():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    model.eval()
    return tok, model


def _pick_text(e: Dict[str, Any]) -> str:
    return (e.get("text") or e.get("snippet") or "").strip()


def _best_evidence_block(evidence: List[Dict[str, Any]], max_total_chars: int = 900) -> str:
    """
    Keep context shorter to reduce model drift and hallucinations.
    """
    lines = []
    used = 0
    for i, e in enumerate(evidence, start=1):
        t = _pick_text(e).replace("\n", " ").strip()
        if not t:
            continue

        header = f"[{i}] {e.get('source_file', '?')} p.{e.get('page', '?')} ({e.get('id', e.get('chunk_id', '?'))}) "
        block = header + t

        remaining = max_total_chars - used
        if remaining <= 0:
            break

        chunk = block[:remaining]
        lines.append(chunk)
        used += len(chunk)

    return "\n".join(lines)


def _ask_flan(tok, model, prompt: str, max_new_tokens: int = 160) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tok.decode(output_ids[0], skip_special_tokens=True).strip()


def _is_garbage(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return True
    low = text.lower().strip()
    bad = [
        "as an ai",
        "i'm an ai",
        "i cannot access",
        "based on the context above write",
        "output format",
        "rules:",
    ]
    return any(p in low for p in bad)


def _looks_generic_or_invented(step: str) -> bool:
    """
    Filter generic lines that often show up as hallucinations.
    """
    low = step.lower().strip()

    generic_patterns = [
        "refer to cited manual sections",
        "refer to the evidence",
        "follow standard site sops",
        "none found",
        "not found",
        "check all components frequently",
        "determine the cause",
        "perform adjustments or repairs immediately",
        "always let the system cool down at the end of the working day",
    ]

    if any(p in low for p in generic_patterns):
        return True

    if len(low) < 8:
        return True

    return False


def _extract_action_steps_from_evidence(evidence: List[Dict[str, Any]], max_steps: int = 4) -> List[str]:
    """
    Rule-based fallback:
    Pull short procedural sentences from evidence.
    Keep it conservative to reduce hallucinations.
    """
    action_verbs = (
        "check", "inspect", "release", "stop", "turn", "remove", "install",
        "tighten", "loosen", "adjust", "measure", "clean", "replace",
        "verify", "operate", "wait", "park", "start", "shut", "fill"
    )

    candidates = []
    seen = set()

    for e in evidence:
        text = _pick_text(e).replace("\n", " ")
        parts = re.split(r"(?<=[\.\!\?])\s+", text)

        for p in parts:
            s = re.sub(r"\s+", " ", p).strip()
            low = s.lower()

            if len(s) < 12 or len(s) > 140:
                continue

            if _looks_generic_or_invented(s):
                continue

            if any(low.startswith(v + " ") for v in action_verbs) or any(f" {v} " in low for v in action_verbs):
                key = low
                if key not in seen:
                    seen.add(key)
                    candidates.append(s)

            if len(candidates) >= max_steps:
                return candidates

    return candidates[:max_steps]


def _extract_numbered_steps_only(proc_text: str) -> List[str]:
    """
    Keep only numbered steps from model output.
    """
    lines = [line.strip() for line in proc_text.splitlines() if line.strip()]
    steps = []

    for line in lines:
        if re.match(r"^\d+\s*[\)\.:-]\s*", line):
            line = re.sub(r"^\d+\s*[\)\.:-]\s*", "", line).strip()
            if line and not _looks_generic_or_invented(line):
                steps.append(line)

    return steps


def generate_structured_answer(query: str, evidence: List[Dict[str, Any]]) -> str:
    if not evidence or all(len(_pick_text(e)) == 0 for e in evidence):
        return "Insufficient evidence found in retrieved manual sections."

    tok, model = _load_model()
    context = _best_evidence_block(evidence, max_total_chars=900)

    prompt = (
        "You are extracting maintenance information from a technical manual.\n\n"
        "Strict rules:\n"
        "- Use ONLY the exact information explicitly present in the context.\n"
        "- Do NOT infer, guess, explain, or add missing steps.\n"
        "- Do NOT add safety advice unless it is written in the context.\n"
        "- If only one or two steps are present, return only those.\n"
        "- If no steps are explicitly present, write: not found.\n\n"
        "Return exactly this format:\n"
        "### Procedure\n"
        "1) ...\n"
        "2) ...\n"
        "(If no explicit steps: not found)\n\n"
        "### Safety Warnings\n"
        "- ...\n"
        "(If none explicitly present: none found)\n\n"
        "### Notes / Checks\n"
        "- ...\n"
        "(If none explicitly present: none found)\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    answer = _ask_flan(tok, model, prompt, max_new_tokens=160)

    # If model output is weak, go directly to conservative fallback
    if _is_garbage(answer) or "### Procedure" not in answer:
        extracted_steps = _extract_action_steps_from_evidence(evidence, max_steps=4)
        procedure = "\n".join([f"{i+1}) {step}" for i, step in enumerate(extracted_steps)]) if extracted_steps else "not found"

        answer = (
            f"### Procedure\n{procedure}\n\n"
            f"### Safety Warnings\nnone found\n\n"
            f"### Notes / Checks\nnone found"
        )

    # Clean procedure section aggressively
    proc_match = re.search(
        r"### Procedure\s*(.*?)\s*(### Safety Warnings|### Notes / Checks|### Citations|$)",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    proc_text = proc_match.group(1).strip() if proc_match else ""

    numbered_steps = _extract_numbered_steps_only(proc_text)

    # If model did not give usable numbered steps, replace with conservative evidence extraction
    if not numbered_steps:
        extracted_steps = _extract_action_steps_from_evidence(evidence, max_steps=4)
        procedure = "\n".join([f"{i+1}) {step}" for i, step in enumerate(extracted_steps)]) if extracted_steps else "not found"

        answer = re.sub(
            r"### Procedure\s*(.*?)\s*(### Safety Warnings)",
            f"### Procedure\n{procedure}\n\n\\2",
            answer,
            flags=re.DOTALL | re.IGNORECASE,
        )
    else:
        cleaned_procedure = "\n".join([f"{i+1}) {step}" for i, step in enumerate(numbered_steps[:4])])
        answer = re.sub(
            r"### Procedure\s*(.*?)\s*(### Safety Warnings)",
            f"### Procedure\n{cleaned_procedure}\n\n\\2",
            answer,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # Clean safety section if model copied junk
    answer = re.sub(
        r"### Safety Warnings\s*(List all.*?|Write each.*?|Output format.*?)(?=###|$)",
        "### Safety Warnings\nnone found\n\n",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Clean notes section if model copied junk
    answer = re.sub(
        r"### Notes / Checks\s*(Rules:.*?|Output format.*?)(?=###|$)",
        "### Notes / Checks\nnone found\n\n",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )

    citation_line = "Evidence sources: " + ", ".join(
        f"{e.get('source_file', '?')} p.{e.get('page', '?')} ({e.get('id', e.get('chunk_id', '?'))})"
        for e in evidence[:4]
    )

    return answer + "\n\n### Citations\n" + citation_line