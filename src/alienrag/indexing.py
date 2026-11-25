"""The index seam: Manuals folder -> parsed documents, images and chunks on disk."""

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from alienrag import config
from alienrag.chunk import Chunk, load_chunks_file

# Bump whenever chunking output changes (serializer, heading repair, metadata), so
# existing indexes re-chunk automatically instead of silently keeping stale chunks.
CHUNKER_VERSION = 2

# Bump whenever the stored row layout changes, so manuals are re-embedded.
STORE_VERSION = 1

Action = Literal["parsed", "rechunked", "skipped", "removed"]


@dataclass
class IndexReport:
    actions: dict[str, Action] = field(default_factory=dict)  # PDF filename -> action
    chunk_counts: dict[str, int] = field(default_factory=dict)  # PDF filename -> chunks written
    embedded: dict[str, int] = field(default_factory=dict)  # PDF filename -> chunks embedded


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _chunker_signature() -> str:
    return f"v{CHUNKER_VERSION}|{config.EMBED_TOKENIZER}|{config.CHUNK_MAX_TOKENS}"


def _embedder_signature() -> str:
    return f"v{STORE_VERSION}|{config.EMBED_MODEL}"


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Entries from before chunker signatures were recorded are plain hashes.
    return {k: v if isinstance(v, dict) else {"sha256": v, "chunker": ""} for k, v in raw.items()}


def _write_manifest(path: Path, manifest: dict[str, dict[str, str]]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _remove_outputs(data_dir: Path, stem: str) -> None:
    (data_dir / "chunks" / f"{stem}.jsonl").unlink(missing_ok=True)
    (data_dir / "parsed" / f"{stem}.json").unlink(missing_ok=True)
    shutil.rmtree(data_dir / "images" / stem, ignore_errors=True)


def index(
    manuals_dir: Path = config.MANUALS_DIR,
    data_dir: Path = config.DATA_DIR,
    *,
    force: bool = False,
    rechunk: bool = False,
    aliases_file: Path | None = config.ALIASES_FILE,
    log=print,
) -> IndexReport:
    """Parse, chunk and embed every PDF in `manuals_dir` into `data_dir`.

    A PDF is re-parsed when its content changed (or `force`), re-chunked from the cached
    parse when the chunker changed (or `rechunk`), and otherwise skipped. Chunks are
    re-embedded into the hybrid-search store only when they were re-chunked, the embedder
    changed, or the store lacks them. Outputs of PDFs no longer in `manuals_dir` are removed.
    The device registry is rebuilt every run from the PDFs present and `aliases_file`.
    """
    from alienrag import store
    from alienrag.chunk import build_chunker, chunk_document, chunks_path, save_chunks
    from alienrag.registry import build_registry
    from alienrag.parse import build_converter, load_parsed, parse_pdf, parsed_path
    from alienrag.structure import repair_headings

    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest = _read_manifest(manifest_path)
    report = IndexReport()
    signature = _chunker_signature()
    embed_sig = _embedder_signature()
    pdfs = sorted(manuals_dir.glob("*.pdf"))
    if any(e.get("embedder", embed_sig) != embed_sig for e in manifest.values()):
        # Different embedder or row layout: vectors aren't comparable (or even the same
        # size), so start the store from scratch and re-embed every manual.
        log("[embed] embedder changed; rebuilding the store")
        store.drop(data_dir)
    converter = chunker = None

    present = {p.name for p in pdfs}
    for name in sorted(set(manifest) - present):
        _remove_outputs(data_dir, Path(name).stem)
        store.delete_source(data_dir, Path(name).stem)
        del manifest[name]
        report.actions[name] = "removed"
        log(f"[remove] {name} no longer in {manuals_dir.name}/")
    _write_manifest(manifest_path, manifest)
    stored = store.count(data_dir)

    for pdf in pdfs:
        digest = _sha256(pdf)
        entry = manifest.get(pdf.name, {})
        has_parse = parsed_path(pdf, data_dir).exists()
        if force or entry.get("sha256") != digest or not has_parse:
            converter = converter or build_converter()
            t = time.perf_counter()
            log(f"[parse] {pdf.name} ...")
            doc = parse_pdf(converter, pdf, data_dir)
            log(f"[parse] {pdf.name} done in {time.perf_counter() - t:.0f}s")
            report.actions[pdf.name] = "parsed"
        elif rechunk or entry.get("chunker") != signature:
            doc = load_parsed(pdf, data_dir)
            report.actions[pdf.name] = "rechunked"
        else:
            report.actions[pdf.name] = "skipped"
            doc = None

        chunks: list[Chunk] | None = None
        if doc is not None:
            kept, demoted = repair_headings(doc, pdf)
            log(f"[struct] {pdf.name}: {kept} outline headings kept, {demoted} demoted to text")
            chunker = chunker or build_chunker()
            chunks = chunk_document(chunker, doc, pdf.stem)
            save_chunks(chunks, pdf.stem, data_dir)
            report.chunk_counts[pdf.name] = len(chunks)
            log(f"[chunk] {pdf.name}: {len(chunks)} chunks")
            # Fresh chunks invalidate any previous embedding (no "embedder" key yet).
            manifest[pdf.name] = {"sha256": digest, "chunker": signature}
            _write_manifest(manifest_path, manifest)

        entry = manifest[pdf.name]
        if chunks is None and entry.get("embedder") == embed_sig and pdf.stem in stored:
            log(f"[skip]  {pdf.name} unchanged")
            continue
        if chunks is None:
            chunks = load_chunks_file(chunks_path(pdf.stem, data_dir))
        t = time.perf_counter()
        report.embedded[pdf.name] = store.replace_source(data_dir, pdf.stem, chunks)
        log(f"[embed] {pdf.name}: {len(chunks)} chunks in {time.perf_counter() - t:.0f}s")
        entry["embedder"] = embed_sig
        _write_manifest(manifest_path, manifest)

    build_registry(pdfs, data_dir, aliases_file)

    # Also rebuild when missing, e.g. a previous run died between embedding and here.
    if report.embedded or "removed" in report.actions.values() or not store.has_fts(data_dir):
        store.rebuild_fts(data_dir)
    return report


def load_all_chunks(data_dir: Path = config.DATA_DIR) -> list[Chunk]:
    return [c for f in sorted((data_dir / "chunks").glob("*.jsonl")) for c in load_chunks_file(f)]
