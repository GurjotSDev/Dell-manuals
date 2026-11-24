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

Action = Literal["parsed", "rechunked", "skipped", "removed"]


@dataclass
class IndexReport:
    actions: dict[str, Action] = field(default_factory=dict)  # PDF filename -> action
    chunk_counts: dict[str, int] = field(default_factory=dict)  # PDF filename -> chunks written


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _chunker_signature() -> str:
    return f"v{CHUNKER_VERSION}|{config.EMBED_TOKENIZER}|{config.CHUNK_MAX_TOKENS}"


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
    log=print,
) -> IndexReport:
    """Parse and chunk every PDF in `manuals_dir` into `data_dir`.

    A PDF is re-parsed when its content changed (or `force`), re-chunked from the cached
    parse when the chunker changed (or `rechunk`), and otherwise skipped. Outputs of PDFs
    no longer in `manuals_dir` are removed.
    """
    from alienrag.chunk import build_chunker, chunk_document, save_chunks
    from alienrag.parse import build_converter, load_parsed, parse_pdf, parsed_path
    from alienrag.structure import repair_headings

    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest = _read_manifest(manifest_path)
    report = IndexReport()
    signature = _chunker_signature()
    pdfs = sorted(manuals_dir.glob("*.pdf"))
    converter = chunker = None

    present = {p.name for p in pdfs}
    for name in sorted(set(manifest) - present):
        _remove_outputs(data_dir, Path(name).stem)
        del manifest[name]
        report.actions[name] = "removed"
        log(f"[remove] {name} no longer in {manuals_dir.name}/")
    _write_manifest(manifest_path, manifest)

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
            log(f"[skip]  {pdf.name} unchanged")
            report.actions[pdf.name] = "skipped"
            continue

        kept, demoted = repair_headings(doc, pdf)
        log(f"[struct] {pdf.name}: {kept} outline headings kept, {demoted} demoted to text")
        chunker = chunker or build_chunker()
        chunks = chunk_document(chunker, doc, pdf.stem)
        save_chunks(chunks, pdf.stem, data_dir)
        report.chunk_counts[pdf.name] = len(chunks)
        log(f"[chunk] {pdf.name}: {len(chunks)} chunks")
        manifest[pdf.name] = {"sha256": digest, "chunker": signature}
        _write_manifest(manifest_path, manifest)

    return report


def load_all_chunks(data_dir: Path = config.DATA_DIR) -> list[Chunk]:
    return [c for f in sorted((data_dir / "chunks").glob("*.jsonl")) for c in load_chunks_file(f)]
