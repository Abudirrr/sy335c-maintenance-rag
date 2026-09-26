import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import (
    RAW_DIR,
    PROCESSED_DIR,
    VECTORDB_DIR,
    EMBED_MODEL_NAME,
)

# ---- NEW DEFAULTS (better retrieval for troubleshooting) ----
# Smaller, more semantically focused chunks
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 300

# Drop pages that are too empty (scanned/blank)
MIN_PAGE_TEXT_LEN = 60


@dataclass
class Chunk:
    id: str
    text: str
    source_file: str
    page: int


# -----------------------------
# Text Cleaning (very important)
# -----------------------------
def _clean_pdf_text(text: str) -> str:
    """
    Clean typical PDF extraction artifacts:
    - Fix hyphenated line breaks: "thermo-\nstat" -> "thermostat"
    - Merge line breaks into spaces
    - Remove repeated whitespace
    - Remove weird control characters
    """
    if not text:
        return ""

    # Remove non-printing characters except common whitespace
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)

    # Fix hyphen line breaks: "cool-\nant" -> "coolant"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Convert newlines to spaces
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)

    # Some PDFs put every sentence on a new line; merge lines
    text = re.sub(r"\n", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_pdf_text(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Returns list of dicts: {page: int, text: str}
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = _clean_pdf_text(txt)
        pages.append({"page": i + 1, "text": txt})
    return pages


# ---------------------------------------
# Better chunking: sentence/paragraph-ish
# ---------------------------------------
def _split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter (no extra dependencies).
    Good enough for manuals.
    """
    if not text:
        return []

    # Split on punctuation boundaries, keep it simple.
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    parts = [p.strip() for p in parts if p and len(p.strip()) > 0]
    return parts


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Build chunks by accumulating sentences until chunk_size,
    then overlap by carrying last overlap chars forward.
    This preserves meaning much better than raw character slicing.
    """
    text = _clean_pdf_text(text)
    if not text:
        return []

    sentences = _split_into_sentences(text)
    if not sentences:
        # Fallback: use normalized text as one chunk
        return [text[:chunk_size]]

    chunks: List[str] = []
    current = ""

    for sent in sentences:
        if not current:
            current = sent
            continue

        # If adding this sentence exceeds size, flush
        if len(current) + 1 + len(sent) > chunk_size:
            chunks.append(current.strip())

            # Create overlap by taking last N chars from current chunk
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + " " + sent).strip()
        else:
            current = (current + " " + sent).strip()

    if current.strip():
        chunks.append(current.strip())

    # Filter tiny chunks
    chunks = [c for c in chunks if len(c) >= 80]
    return chunks


def build_chunks() -> List[Chunk]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks: List[Chunk] = []

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {RAW_DIR}. Put your manuals there first.")

    for pdf in pdfs:
        pages = extract_pdf_text(pdf)

        for p in pages:
            page_num = p["page"]
            page_text = (p["text"] or "").strip()

            if len(page_text) < MIN_PAGE_TEXT_LEN:
                continue

            parts = chunk_text(page_text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

            for j, ch in enumerate(parts):
                chunk_id = f"{pdf.name}::p{page_num}::c{j}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=ch,
                        source_file=pdf.name,
                        page=page_num,
                    )
                )

    # Save chunks for debugging
    out_jsonl = PROCESSED_DIR / "chunks.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    return chunks


def build_faiss_index(chunks: List[Chunk]) -> None:
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c.text for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(embeddings, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity if normalized embeddings
    index.add(emb)

    faiss.write_index(index, str(VECTORDB_DIR / "index.faiss"))

    meta = [
        {"id": c.id, "source_file": c.source_file, "page": c.page, "text": c.text}
        for c in chunks
    ]
    (VECTORDB_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run():
    chunks = build_chunks()
    print(f"Built {len(chunks)} chunks.")
    build_faiss_index(chunks)
    print(f"FAISS index saved to {VECTORDB_DIR}.")


if __name__ == "__main__":
    run()
