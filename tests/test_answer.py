"""Tests for the answer seam, run against the real manuals and real Ollama models.

Shares data/test-index with test_index.py; indexing is a no-op when nothing changed.
"""

from pathlib import Path

import pytest

from alienrag.answer import Session, UnknownDevice, answer
from alienrag.indexing import index

ROOT = Path(__file__).resolve().parents[1]
MANUALS = ROOT / "Manuals"
TEST_DATA = ROOT / "data" / "test-index"

pytestmark = pytest.mark.skipif(
    not any(MANUALS.glob("*.pdf")), reason="no manual PDFs in Manuals/"
)

MEMORY_Q = "how do I remove the memory module?"


@pytest.fixture(scope="module")
def indexed():
    index(MANUALS, TEST_DATA, log=lambda *_: None)


@pytest.fixture(scope="module")
def ask(indexed):
    cache = {}

    def _ask(question: str, product: str):
        key = (question, product)
        if key not in cache:
            cache[key] = answer(question, Session(devices=[product]), data_dir=TEST_DATA)
        return cache[key]

    return _ask


@pytest.mark.parametrize(
    ("product", "manual"),
    [("act1250", "act1250-owners-manual-en-us"), ("r16", "alienware-aurora-r16-owners-manual-en-us")],
)
def test_retrieved_chunks_come_only_from_the_requested_products_manual(ask, product, manual):
    a = ask(MEMORY_Q, product)
    assert a.devices == [product]
    assert a.chunks
    assert {c.source for c in a.chunks} == {manual}


@pytest.mark.parametrize("product", ["act1250", "r16"])
def test_every_citation_points_at_a_retrieved_page(ask, product):
    a = ask(MEMORY_Q, product)
    assert a.status == "answered"
    assert a.text.strip()
    assert a.citations, a.text
    retrieved = {(c.source, p) for c in a.chunks for p in c.pages}
    stray = [c for c in a.citations if (c.source, c.page) not in retrieved]
    assert not stray, f"{stray}\n\n{a.text}"


def test_memory_module_removal_on_act1250_is_answered_from_its_procedure_pages(ask):
    # Ground truth from the PDF: "Removing the memory module" spans pages 54-55.
    a = ask(MEMORY_Q, "act1250")
    procedure = {54, 55}
    assert procedure & {p for c in a.chunks for p in c.pages}
    assert procedure & {c.page for c in a.citations}, a.text
    assert "clip" in a.text.lower(), a.text  # "Push the securing clips away from the memory module"


def test_retrieval_matches_exact_part_numbers(ask):
    # Only the keyword side matches the token reliably; CR2032 is the ACT1250 coin cell.
    a = ask("CR2032", "act1250")
    assert any("CR2032" in c.text for c in a.chunks)


def test_retrieval_matches_paraphrases_with_no_shared_keywords(ask):
    a = ask("taking out a stick of RAM", "act1250")
    assert any("Removing the memory module" in c.headings for c in a.chunks)


@pytest.mark.parametrize("devices", [[], ["nosuchdevice"]])
def test_asking_without_a_known_device_is_rejected_naming_the_known_ones(indexed, devices):
    with pytest.raises(UnknownDevice) as e:
        answer(MEMORY_Q, Session(devices=devices), data_dir=TEST_DATA)
    assert "act1250" in str(e.value) and "r16" in str(e.value)
