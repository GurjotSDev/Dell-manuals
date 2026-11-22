"""DoclingDocument -> structure-aware chunks with page / heading / image metadata."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument, PictureItem
from transformers import AutoTokenizer

from alienrag import config


@dataclass
class Chunk:
    id: str
    source: str  # manual file stem, e.g. "act1250-owners-manual-en-us"
    text: str  # raw chunk text
    embed_text: str  # heading path + text; what gets embedded
    headings: list[str]
    pages: list[int]
    images: list[str] = field(default_factory=list)  # paths relative to data/
    n_tokens: int = 0


def build_chunker() -> HybridChunker:
    tok = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(config.EMBED_TOKENIZER),
        max_tokens=config.CHUNK_MAX_TOKENS,
    )
    return HybridChunker(tokenizer=tok, merge_peers=True)


def _image_path(item: PictureItem, img_dir: Path) -> str | None:
    if item.image is None or item.image.uri is None:
        return None
    p = Path(str(item.image.uri))
    if not p.is_absolute():
        p = img_dir / p.name
    return p.relative_to(config.DATA_DIR).as_posix() if p.is_relative_to(config.DATA_DIR) else str(p)


SKIP_SECTIONS = {"contents"}  # table-of-contents pages are retrieval noise


def _pictures_by_anchor(doc: DoclingDocument, img_dir: Path) -> dict[str, list[str]]:
    """Map each picture to the text item preceding it in reading order.

    HybridChunker drops pictures from chunk doc_items, so pictures are re-attached to
    whichever chunk contains their anchor text (in these manuals, the step they illustrate).
    """
    anchors: dict[str, list[str]] = {}
    last_text: str | None = None
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            if last_text and (path := _image_path(item, img_dir)):
                anchors.setdefault(last_text, []).append(path)
        elif hasattr(item, "text"):
            last_text = item.self_ref
    return anchors


def chunk_document(chunker: HybridChunker, doc: DoclingDocument, source: str) -> list[Chunk]:
    img_dir = config.IMAGES_DIR / source
    anchors = _pictures_by_anchor(doc, img_dir)
    out: list[Chunk] = []
    for ch in chunker.chunk(doc):
        headings = list(ch.meta.headings or [])
        if headings and headings[0].strip().lower() in SKIP_SECTIONS:
            continue
        pages = sorted({p.page_no for it in ch.meta.doc_items for p in it.prov})
        images = [p for it in ch.meta.doc_items for p in anchors.get(it.self_ref, [])]
        embed_text = chunker.contextualize(ch)
        out.append(
            Chunk(
                id=f"{source}:{len(out):05d}",
                source=source,
                text=ch.text,
                embed_text=embed_text,
                headings=headings,
                pages=pages,
                images=images,
                n_tokens=chunker.tokenizer.count_tokens(embed_text),
            )
        )
    return out


def chunks_path(source: str) -> Path:
    return config.CHUNKS_DIR / f"{source}.jsonl"


def save_chunks(chunks: list[Chunk], source: str) -> Path:
    path = chunks_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return path


def load_chunks(source: str) -> list[Chunk]:
    with chunks_path(source).open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
