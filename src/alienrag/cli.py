"""Entry points: `index` and `ask`."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from alienrag import config


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict[str, str]:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text())
    return {}


def index_main() -> None:
    ap = argparse.ArgumentParser(description="Parse and chunk manuals in Manuals/.")
    ap.add_argument("--force", action="store_true", help="re-parse even if unchanged")
    ap.add_argument("--rechunk", action="store_true", help="re-chunk from cached parses")
    args = ap.parse_args()

    from alienrag.chunk import build_chunker, chunk_document, save_chunks
    from alienrag.parse import build_converter, load_parsed, parse_pdf
    from alienrag.structure import repair_headings

    pdfs = sorted(config.MANUALS_DIR.glob("*.pdf"))
    manifest = _load_manifest()
    converter = None
    chunker = build_chunker()

    for pdf in pdfs:
        digest = _sha256(pdf)
        changed = args.force or manifest.get(pdf.name) != digest
        if changed:
            converter = converter or build_converter()
            t = time.perf_counter()
            print(f"[parse] {pdf.name} ...", flush=True)
            doc = parse_pdf(converter, pdf)
            print(f"[parse] {pdf.name} done in {time.perf_counter() - t:.0f}s")
        elif args.rechunk:
            doc = load_parsed(pdf)
        else:
            print(f"[skip]  {pdf.name} unchanged")
            continue

        kept, demoted = repair_headings(doc, pdf)
        print(f"[struct] {pdf.name}: {kept} outline headings kept, {demoted} demoted to text")
        chunks = chunk_document(chunker, doc, pdf.stem)
        path = save_chunks(chunks, pdf.stem)
        print(f"[chunk] {pdf.name}: {len(chunks)} chunks -> {path.relative_to(config.ROOT)}")
        manifest[pdf.name] = digest
        config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def ask_main() -> None:
    raise SystemExit("ask: not implemented yet (build step 4)")
