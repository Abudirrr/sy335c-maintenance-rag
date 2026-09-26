from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTORDB_DIR = BASE_DIR / "data" / "vectordb"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE_CHARS = 850
CHUNK_OVERLAP_CHARS = 200

TOP_K = 8