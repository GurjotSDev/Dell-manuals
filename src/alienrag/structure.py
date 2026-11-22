"""Repair the heading hierarchy of a parsed manual using the PDF's bookmark outline.

Docling assigns every section header level 1 for these PDFs, so procedure sub-labels
("Prerequisites", "Steps", "Next steps", ...) reset the heading path and split each
procedure into fragments. The PDF outline holds the real hierarchy, so:

- headers that match an outline entry get level = outline depth + 1;
- every other header is demoted to body text, keeping procedures in one section.
"""

import re
from pathlib import Path

import pypdfium2 as pdfium
from docling_core.types.doc import DocItemLabel, DoclingDocument, SectionHeaderItem, TextItem


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def outline_levels(pdf: Path) -> dict[str, int]:
    doc = pdfium.PdfDocument(str(pdf))
    try:
        levels: dict[str, int] = {}
        for b in doc.get_toc():
            levels.setdefault(_norm(b.get_title()), b.level + 1)
        return levels
    finally:
        doc.close()


def repair_headings(doc: DoclingDocument, pdf: Path) -> tuple[int, int]:
    """Mutate `doc` in place. Returns (headers kept, headers demoted)."""
    levels = outline_levels(pdf)
    kept = demoted = 0
    for i, item in enumerate(doc.texts):
        if not isinstance(item, SectionHeaderItem):
            continue
        level = levels.get(_norm(item.text))
        if level is not None:
            item.level = level
            kept += 1
            continue
        doc.texts[i] = TextItem(
            self_ref=item.self_ref,
            parent=item.parent,
            children=item.children,
            content_layer=item.content_layer,
            label=DocItemLabel.TEXT,
            prov=item.prov,
            orig=item.orig,
            text=item.text,
        )
        demoted += 1
    return kept, demoted
