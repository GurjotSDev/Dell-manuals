"""DoclingDocument -> structure-aware chunks with page / heading / image metadata."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    DocChunk,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
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


class _MarkdownTables(ChunkingSerializerProvider):
    """Serialize tables as Markdown. Docling's default "row, col = value" triplets
    drop cells on these manuals' multi-page spec and BIOS tables; Markdown keeps all."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())


def build_chunker() -> HybridChunker:
    tok = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(config.EMBED_TOKENIZER),
        max_tokens=config.CHUNK_MAX_TOKENS,
    )
    return HybridChunker(tokenizer=tok, merge_peers=True, serializer_provider=_MarkdownTables())


def _image_path(item: PictureItem, source: str) -> str | None:
    """Image path relative to the data dir: images/<source>/<file>.png."""
    if item.image is None or item.image.uri is None:
        return None
    return f"images/{source}/{Path(str(item.image.uri)).name}"


SKIP_SECTIONS = {"contents"}  # table-of-contents pages are retrieval noise


def _pictures_by_anchor(doc: DoclingDocument, source: str) -> dict[str, list[str]]:
    """Map each picture to the text item preceding it in reading order.

    HybridChunker drops pictures from chunk doc_items, so pictures are re-attached to
    whichever chunk contains their anchor text (in these manuals, the step they illustrate).
    """
    anchors: dict[str, list[str]] = {}
    last_text: str | None = None
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            if last_text and (path := _image_path(item, source)):
                anchors.setdefault(last_text, []).append(path)
        elif hasattr(item, "text"):
            last_text = item.self_ref
    return anchors


def chunk_document(chunker: HybridChunker, doc: DoclingDocument, source: str) -> list[Chunk]:
    anchors = _pictures_by_anchor(doc, source)
    # Front matter (cover page) precedes the first outline heading; label it with the
    # manual title: the first body text in reading order (page furniture excluded).
    title = next(
        (t for item, _ in doc.iterate_items() if (t := getattr(item, "text", "").strip())),
        source,
    )
    out: list[Chunk] = []
    for base in chunker.chunk(doc):
        ch = DocChunk.model_validate(base)
        headings = list(ch.meta.headings or []) or [title]
        if headings[0].strip().lower() in SKIP_SECTIONS:
            continue
        ch.meta.headings = headings  # so contextualize() prefixes the fallback title too
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


def chunks_path(source: str, data_dir: Path) -> Path:
    return data_dir / "chunks" / f"{source}.jsonl"


def save_chunks(chunks: list[Chunk], source: str, data_dir: Path) -> Path:
    path = chunks_path(source, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return path


def load_chunks_file(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
