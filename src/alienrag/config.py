"""Central configuration: paths and model names."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUALS_DIR = ROOT / "Manuals"
DATA_DIR = ROOT / "data"
PARSED_DIR = DATA_DIR / "parsed"  # Docling JSON per manual
IMAGES_DIR = DATA_DIR / "images"  # exported pictures per manual
CHUNKS_DIR = DATA_DIR / "chunks"  # chunk JSONL per manual
MANIFEST_PATH = DATA_DIR / "manifest.json"  # file hashes for incremental indexing

# Embedding: Ollama model + matching HF tokenizer (used to size chunks).
EMBED_MODEL = "qwen3-embedding:0.6b"
EMBED_TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_MAX_TOKENS = 512

GEN_MODEL = "frob/qwen3.5-instruct:9b"
GEN_NUM_CTX = 16384

IMAGES_SCALE = 2.0  # render scale for exported pictures
