"""Central configuration: paths and model names."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUALS_DIR = ROOT / "Manuals"
# Index output: parsed/ (Docling JSON), images/ (exported pictures), chunks/ (JSONL),
# manifest.json (PDF hashes for incremental indexing).
DATA_DIR = ROOT / "data"
# Hand-editable device nicknames, merged into the device registry at index time:
# {"<model code>": ["alias", ...]}.
ALIASES_FILE = ROOT / "aliases.json"

# Embedding: Ollama model + matching HF tokenizer (used to size chunks).
EMBED_MODEL = "qwen3-embedding:0.6b"
EMBED_TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_MAX_TOKENS = 512

GEN_MODEL = "frob/qwen3.5-instruct:9b"
GEN_NUM_CTX = 16384

RETRIEVE_K = 6  # chunks retrieved per question (hybrid search, product-filtered)

SESSION_TURNS = 4  # earlier turns a chat session keeps for rewriting follow-ups

IMAGES_SCALE = 2.0  # render scale for exported pictures
