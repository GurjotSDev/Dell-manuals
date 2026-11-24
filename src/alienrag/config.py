"""Central configuration: paths and model names."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUALS_DIR = ROOT / "Manuals"
# Index output: parsed/ (Docling JSON), images/ (exported pictures), chunks/ (JSONL),
# manifest.json (PDF hashes for incremental indexing).
DATA_DIR = ROOT / "data"

# Embedding: Ollama model + matching HF tokenizer (used to size chunks).
EMBED_MODEL = "qwen3-embedding:0.6b"
EMBED_TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_MAX_TOKENS = 512

GEN_MODEL = "qwen3:4b-instruct"
GEN_NUM_CTX = 16384

RETRIEVE_K = 6  # chunks retrieved per question (hybrid search, product-filtered)

IMAGES_SCALE = 2.0  # render scale for exported pictures
