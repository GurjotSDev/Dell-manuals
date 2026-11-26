"""Chunk store: Ollama embeddings in LanceDB with a full-text index for hybrid search."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import lancedb
import ollama
import pyarrow as pa

from alienrag import config
from alienrag.chunk import Chunk
from alienrag.registry import model_code

TABLE = "chunks"
EMBED_BATCH = 32
# Qwen3-Embedding is instruction-aware: queries carry a task prefix, documents don't.
QUERY_INSTRUCTION = "Given a question about a computer, retrieve owner's manual passages that answer it"


@dataclass
class Hit:
    id: str
    source: str
    product: str
    text: str
    headings: list[str]
    pages: list[int]
    images: list[str]
    score: float  # fused (reciprocal-rank) relevance: says how hits rank, not how well they match
    similarity: float  # cosine similarity of the chunk's embedding to the query's


def embed(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = ollama.embed(model=config.EMBED_MODEL, input=texts[i : i + EMBED_BATCH])
        out.extend(list(map(list, resp.embeddings)))
    return out


def _db(data_dir: Path):
    return lancedb.connect(str(data_dir / "lancedb"))


def _open(data_dir: Path):
    db = _db(data_dir)
    return db.open_table(TABLE) if TABLE in db.list_tables().tables else None


def _quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def delete_source(data_dir: Path, source: str) -> None:
    if (tbl := _open(data_dir)) is not None:
        tbl.delete(f"source = {_quote(source)}")


def replace_source(data_dir: Path, source: str, chunks: list[Chunk]) -> int:
    """Embed `chunks` and make them the only rows for `source`. Returns rows written."""
    vectors = embed([c.embed_text for c in chunks]) if chunks else []
    rows = [
        {
            "id": c.id,
            "source": c.source,
            "product": model_code(c.source),
            "text": c.text,
            "embed_text": c.embed_text,
            "headings": c.headings,
            "pages": c.pages,
            "images": c.images,
            "vector": v,
        }
        for c, v in zip(chunks, vectors)
    ]
    tbl = _open(data_dir)
    if tbl is None:
        if not rows:
            return 0
        dim = len(rows[0]["vector"])
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("source", pa.string()),
                ("product", pa.string()),
                ("text", pa.string()),
                ("embed_text", pa.string()),
                ("headings", pa.list_(pa.string())),
                ("pages", pa.list_(pa.int32())),
                ("images", pa.list_(pa.string())),
                ("vector", pa.list_(pa.float32(), dim)),
            ]
        )
        _db(data_dir).create_table(TABLE, data=rows, schema=schema)
        return len(rows)
    tbl.delete(f"source = {_quote(source)}")
    if rows:
        tbl.add(rows)
    return len(rows)


def drop(data_dir: Path) -> None:
    db = _db(data_dir)
    if TABLE in db.list_tables().tables:
        db.drop_table(TABLE)


def rebuild_fts(data_dir: Path) -> None:
    if (tbl := _open(data_dir)) is not None and tbl.count_rows() > 0:
        tbl.create_fts_index("embed_text", replace=True)


def has_fts(data_dir: Path) -> bool:
    tbl = _open(data_dir)
    return tbl is None or tbl.count_rows() == 0 or any(i.index_type == "FTS" for i in tbl.list_indices())


def count(data_dir: Path) -> dict[str, int]:
    """Rows per source manual."""
    tbl = _open(data_dir)
    if tbl is None:
        return {}
    return dict(Counter(tbl.search().select(["source"]).to_arrow().column("source").to_pylist()))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
    return dot / norm if norm else 0.0


def _fts_text(query: str) -> str:
    # The FTS parser treats some punctuation as syntax; keep words only.
    return " ".join(re.findall(r"[\w.]+", query))


def search(data_dir: Path, query: str, products: list[str], k: int = 8) -> list[Hit]:
    """Hybrid (BM25 + vector, reciprocal-rank fused) search restricted to `products`."""
    tbl = _open(data_dir)
    if tbl is None:
        raise RuntimeError(f"no index in {data_dir}; run `index` first")
    qvec = embed([f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"])[0]
    where = "product IN (" + ", ".join(_quote(p) for p in products) + ")"
    rows = (
        tbl.search(query_type="hybrid")
        .vector(qvec)
        .text(_fts_text(query))
        .where(where, prefilter=True)
        .limit(k)
        .to_list()
    )
    return [
        Hit(
            id=r["id"],
            source=r["source"],
            product=r["product"],
            text=r["text"],
            headings=list(r["headings"]),
            pages=list(r["pages"]),
            images=list(r["images"]),
            score=float(r["_relevance_score"]),
            similarity=_cosine(qvec, list(r["vector"])),
        )
        for r in rows
    ]
