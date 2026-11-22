"""Property tests for the index seam, run against the real manuals.

The test index lives in data/test-index. The first run parses the PDFs (~2 min);
later runs reuse the cached parses and only re-chunk, so chunking changes are
always exercised.
"""

import re
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from alienrag.indexing import index, load_all_chunks
from alienrag.parse import load_parsed

ROOT = Path(__file__).resolve().parents[1]
MANUALS = ROOT / "Manuals"
TEST_DATA = ROOT / "data" / "test-index"

pytestmark = pytest.mark.skipif(
    not any(MANUALS.glob("*.pdf")), reason="no manual PDFs in Manuals/"
)


@pytest.fixture(scope="session")
def chunks():
    index(MANUALS, TEST_DATA, rechunk=True)
    return load_all_chunks(TEST_DATA)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _pdf_lines(pdf: Path):
    """Yield (1-based page number, line) from the PDF's own text layer."""
    doc = pdfium.PdfDocument(str(pdf))
    try:
        for i in range(len(doc)):
            for line in doc[i].get_textpage().get_text_range().splitlines():
                yield i + 1, line
    finally:
        doc.close()


def test_every_warning_and_caution_in_the_pdf_is_in_a_chunk_on_its_page(chunks):
    missing = []
    for pdf in MANUALS.glob("*.pdf"):
        own = [c for c in chunks if c.source == pdf.stem]
        for page, line in _pdf_lines(pdf):
            m = re.match(r"\s*((WARNING|CAUTION):.*)", line)
            if not m:
                continue
            needle = _norm(m.group(1))[:40]
            if not any(page in c.pages and needle in _norm(c.text) for c in own):
                missing.append(f"{pdf.stem} p{page}: {m.group(1)[:60]}")
    assert not missing, "\n".join(missing)


def test_no_table_of_contents_text_in_chunks(chunks):
    # TOC lines in these manuals use dot leaders: "Front.............7"
    leaders = [c.id for c in chunks if re.search(r"\.{8,}", c.text)]
    assert not leaders, leaders
    assert not [c.id for c in chunks if c.headings[0].strip().lower() == "contents"]


def test_every_table_cell_is_in_a_chunk(chunks):
    missing = []
    for pdf in MANUALS.glob("*.pdf"):
        blob = " ".join(_norm(c.text) for c in chunks if c.source == pdf.stem)
        doc = load_parsed(pdf, TEST_DATA)
        for table in doc.tables:
            if any(re.search(r"\.{8,}", c.text) for c in table.data.table_cells):
                continue  # table-of-contents table, excluded on purpose
            for cell in table.data.table_cells:
                needle = _norm(cell.text)[:30]
                if len(needle) >= 8 and needle not in blob:
                    missing.append(f"{pdf.stem}: {cell.text[:60]}")
    assert not missing, "\n".join(missing[:20])


def test_reindexing_unchanged_manuals_does_nothing(chunks):
    report = index(MANUALS, TEST_DATA, log=lambda *_: None)
    assert report.actions
    assert set(report.actions.values()) == {"skipped"}
    assert report.chunk_counts == {}


def test_manuals_removed_from_the_folder_are_removed_from_the_index(chunks):
    import json

    manifest = TEST_DATA / "manifest.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    entries["gone.pdf"] = {"sha256": "x", "chunker": "x"}
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    (TEST_DATA / "chunks" / "gone.jsonl").write_text(
        (TEST_DATA / "chunks" / f"{chunks[0].source}.jsonl").read_text(encoding="utf-8").replace(
            chunks[0].source, "gone"
        ),
        encoding="utf-8",
    )

    report = index(MANUALS, TEST_DATA, log=lambda *_: None)

    assert report.actions["gone.pdf"] == "removed"
    assert "gone" not in {c.source for c in load_all_chunks(TEST_DATA)}


def test_every_linked_image_exists_on_disk(chunks):
    images = [i for c in chunks for i in c.images]
    assert images
    assert not [i for i in images if not (TEST_DATA / i).is_file()]


def test_every_chunk_has_source_page_and_heading_path(chunks):
    sources = {p.stem for p in MANUALS.glob("*.pdf")}
    assert {c.source for c in chunks} == sources
    for c in chunks:
        assert c.source in sources
        assert c.pages, f"{c.id} has no pages"
        assert c.headings, f"{c.id} has no heading path"
