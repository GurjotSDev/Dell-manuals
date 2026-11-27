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
# Golden eval set run by `eval`: one question per line with expected devices, pages,
# status and answer.
GOLDEN_FILE = ROOT / "eval" / "golden.jsonl"

# Embedding: Ollama model + matching HF tokenizer (used to size chunks).
EMBED_MODEL = "qwen3-embedding:0.6b"
EMBED_TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_MAX_TOKENS = 512

GEN_MODEL = "frob/qwen3.5-instruct:9b"
GEN_NUM_CTX = 16384

RETRIEVE_K = 6  # chunks retrieved per question (hybrid search, product-filtered)
# Refuse without generating when no retrieved chunk's cosine similarity to the query
# reaches this: a safety floor below every answerable question measured. Off-topic and
# answerable questions overlap above it, so the prompt-level refusal does the rest.
# See eval/baseline.md.
REFUSE_BELOW = 0.40
# A retrieved chunk brings the rest of its section (a procedure or spec table split
# over chunks) when the section has at most SECTION_MAX_CHUNKS chunks, up to
# CONTEXT_MAX_CHUNKS chunks shown to the generator in all.
SECTION_MAX_CHUNKS = 4
CONTEXT_MAX_CHUNKS = 12

SESSION_TURNS = 4  # earlier turns a chat session keeps for rewriting follow-ups

IMAGES_SCALE = 2.0  # render scale for exported pictures
