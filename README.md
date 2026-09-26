# SY335C Maintenance RAG

A local retrieval-augmented generation prototype for exploring SANY SY335C excavator maintenance manuals. Built by Abdelrahman Ramadan with Python, FastAPI, Sentence Transformers, FAISS, and FLAN-T5.

Ask a natural-language question, inspect retrieved manual pages, and review a structured answer alongside its evidence. This project demonstrates the complete path from PDF ingestion to retrieval, generation, a browser interface, and quantitative evaluation.

**Research prototype:** generated maintenance instructions can be incomplete or incorrect. Verify against the correct official manual and a qualified technician before acting. This is an independent project, not a SANY product.

## What I built

- PDF text cleaning and sentence-aware chunking with page and source provenance.
- Normalized MiniLM embeddings indexed with FAISS inner-product search.
- Query expansion and hybrid reranking combining semantic similarity and keyword overlap (75% / 25%).
- A FastAPI/Jinja2 interface with evidence previews and page-level references.
- Local FLAN-T5 generation with structured procedure, warning, and notes sections, plus an extractive fallback.
- Keyword baseline comparisons, retrieval metrics, and a small step-based evaluation suite.

## Architecture

```text
Local manuals → PDF extraction → chunks + page metadata → MiniLM → FAISS
Question → query expansion → semantic candidates → hybrid reranking
         → FLAN-T5 / extractive fallback → answer + citations + evidence
```

The ingestion implementation uses 1,800-character target chunks and 300-character overlap. Retrieval considers up to 64 candidates and returns eight results by default. The generator currently receives at most 900 characters of evidence; this is a significant context limitation.

## Run locally

Python 3.11 is a suggested starting point. The original archive did not contain a dependency lockfile, so the requirements are not a claim of a fully reproduced environment.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place lawfully obtained, text-readable manuals in `data/raw/`. The evaluation scripts expect these filenames and the original PDF page numbering:

- `SY335_Maintenance_EN.pdf`
- `SY335_Operator_Manual.pdf`

```bash
python -m app.ingest
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Initial model loading downloads `sentence-transformers/all-MiniLM-L6-v2` and `google/flan-t5-large` from Hugging Face. Local generation needs substantial memory and can be slow on CPU. No paid inference API key is required by this implementation.

## Evaluation

Run from the repository root after ingestion:

```bash
python -m app.eval
python -m app.eval_metrics
python -m app.evaluate_steps
```

The first two commands use different evaluation schemas and both write `logs/evaluation_metrics.json`; save a copy between runs if you need both outputs.

The supplied historical five-question structured evaluation reports:

| Metric | Recorded result |
| --- | ---: |
| Mean Precision@8 | 0.225 |
| Mean step coverage | 0.200 |
| Mean unmatched generated steps per question | 1.600 |

The original evaluator calls unmatched steps “hallucinated steps.” It uses token overlap against a short expected-step list, not expert factual adjudication. These are archived prototype results, not a fresh benchmark of this publication. See [evaluation notes](docs/EVALUATION.md).

## Repository guide

- `app/ingest.py`: PDF extraction, chunking, embeddings, and index construction.
- `app/rag.py`: retrieval, reranking, evidence previews, and answer assembly.
- `app/llm.py`: local generation and extractive fallback.
- `app/main.py`: web routes.
- `app/eval*.py`: retrieval and generation evaluation.
- `app/testset.json`: page-reference test questions.
- `docs/PROJECT_REPORT.md`: public technical summary of the accompanying paper.

## Limitations and next steps

Retrieval scores are normalized within each candidate set. The UI labels are heuristics, not calibrated probabilities of correctness. Citations identify retrieved evidence, but do not prove that every generated step is supported. Evaluation is small and uses permissive token matching; some matches are false positives. Scanned manuals need a separate OCR step.

Future work includes section-aware chunking, stronger reranking, a larger expert-reviewed test set, sentence-level attribution, and calibrated abstention. The source retains the original prototype algorithm so the project history remains interpretable.

## Source materials

This portfolio edition is based on the author's 2026 ITC 4879 project. Manufacturer PDFs, extracted manual text, generated indexes, the bundled virtual environment, and private academic identifiers are excluded. Obtain the manuals separately and rebuild the index. No redistribution license is asserted for third-party manuals or model weights; their respective terms apply.
