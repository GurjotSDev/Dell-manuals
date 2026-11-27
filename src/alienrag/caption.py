"""Image captions: a local vision model describes each procedure image, and the
description is added to the embed text of the chunk the image is anchored to.

Captions are cached in <data dir>/captions.json by model, prompt and image content, so
re-indexing only captions images it hasn't seen.

Captions change both searches: `embed_text` is what gets embedded and what the BM25 index
covers. They are appended after chunking, so a captioned chunk's embed text can exceed
CHUNK_MAX_TOKENS (the embedder's context is far larger) and its `n_tokens` counts the
text without captions.
"""

import hashlib
import json
import os
from pathlib import Path

import ollama
from PIL import Image

from alienrag import config
from alienrag.chunk import Chunk

CAPTION_PROMPT = (
    "This image is from a computer owner's manual. Describe what it shows in one or two "
    "sentences: the parts, connectors, screws, labels and the action illustrated. "
    "Plain text only."
)

CAPTION_PREFIX = "Image: "  # starts each caption line in a chunk's embed text


def _prompt_id() -> str:
    return hashlib.sha256(CAPTION_PROMPT.encode()).hexdigest()[:12]


def signature() -> str:
    """What the captions depend on: changing it re-chunks and re-embeds. The size
    threshold decides which images are captioned, not what a caption says, so it is here
    but not in the cache key."""
    return f"captions|{config.CAPTION_MODEL}|{_prompt_id()}|{config.CAPTION_MIN_PIXELS}"


def _cache_path(data_dir: Path) -> Path:
    return data_dir / "captions.json"


def _read_cache(data_dir: Path) -> dict[str, str]:
    path = _cache_path(data_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_cache(data_dir: Path, cache: dict[str, str]) -> None:
    path = _cache_path(data_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def is_procedure_image(path: Path) -> bool:
    """Big enough to illustrate a step; smaller pictures are note and warning icons."""
    with Image.open(path) as im:
        return max(im.size) >= config.CAPTION_MIN_PIXELS


def _caption(path: Path) -> str:
    resp = ollama.chat(
        model=config.CAPTION_MODEL,
        messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [str(path)]}],
        think=False,
        options={"temperature": 0},
    )
    return " ".join((resp.message.content or "").split())


def add_captions(chunks: list[Chunk], data_dir: Path, log=print) -> list[Chunk]:
    """`chunks` with a caption line per procedure image appended to their embed text."""
    cache = _read_cache(data_dir)
    key_prefix = f"{config.CAPTION_MODEL}|{_prompt_id()}"
    new = 0
    for c in chunks:
        lines = []
        for rel in c.images:
            path = data_dir / rel
            if not is_procedure_image(path):
                continue
            key = f"{key_prefix}|{hashlib.sha256(path.read_bytes()).hexdigest()}"
            if key not in cache:
                text = _caption(path)
                if not text:  # not cached: an empty reply is retried next run
                    continue
                cache[key] = text
                new += 1
                _write_cache(data_dir, cache)  # keep progress if a long run dies
            lines.append(CAPTION_PREFIX + cache[key])
        if lines:
            c.embed_text = c.embed_text + "\n\n" + "\n".join(lines)
    if new:
        log(f"[caption] {new} images captioned with {config.CAPTION_MODEL}")
    return chunks
