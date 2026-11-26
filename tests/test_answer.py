"""Tests for the answer seam, run against the real manuals and real Ollama models.

Shares data/test-index with test_index.py; indexing is a no-op when nothing changed.
"""

from pathlib import Path

import ollama
import pytest

from alienrag import config
from alienrag.answer import SYSTEM_PROMPT, Session, UnknownDevice, answer
from alienrag.indexing import index

ROOT = Path(__file__).resolve().parents[1]
MANUALS = ROOT / "Manuals"
TEST_DATA = ROOT / "data" / "test-index"
ALIASES = ROOT / "tests" / "aliases.json"

pytestmark = pytest.mark.skipif(
    not any(MANUALS.glob("*.pdf")), reason="no manual PDFs in Manuals/"
)

MEMORY_Q = "how do I remove the memory module?"


@pytest.fixture(scope="module")
def indexed():
    index(MANUALS, TEST_DATA, aliases_file=ALIASES, log=lambda *_: None)


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
    assert [d.id for d in a.devices] == [product]
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


def test_forcing_an_unknown_device_is_rejected_naming_the_known_ones(indexed):
    with pytest.raises(UnknownDevice) as e:
        answer(MEMORY_Q, Session(devices=["nosuchdevice"]), data_dir=TEST_DATA)
    assert "act1250" in str(e.value) and "r16" in str(e.value)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many fans does the Alienware Aurora ACT1250 have?", ["act1250"]),
        ("how many fans does my act1250 have", ["act1250"]),
        ("How many fans are in the Aurora R16?", ["r16"]),
        ("how many fans does the 1250 have?", ["act1250"]),  # alias from tests/aliases.json
        ("R-16: how many fans?", ["r16"]),
    ],
)
def test_device_named_in_the_question_is_resolved_without_a_flag(indexed, question, expected):
    a = answer(question, Session(), data_dir=TEST_DATA)
    assert a.status == "answered"
    assert [d.id for d in a.devices] == expected
    assert {c.product for c in a.chunks} == set(expected)


def test_question_naming_two_devices_resolves_to_both(indexed):
    a = answer("Does the R16 have more fans than the Aurora ACT1250?", Session(), data_dir=TEST_DATA)
    assert [d.id for d in a.devices] == ["r16", "act1250"]  # in order of mention
    assert {c.product for c in a.chunks} <= {"r16", "act1250"}


def test_question_naming_no_device_asks_which_device_listing_known_ones(indexed):
    a = answer(MEMORY_Q, Session(), data_dir=TEST_DATA)
    assert a.status == "needs-device"
    assert a.devices == []
    assert a.chunks == [] and a.citations == []  # nothing retrieved, nothing generated
    assert "Alienware Aurora ACT1250" in a.text and "Alienware Aurora R16" in a.text


def test_device_named_in_a_way_no_alias_matches_is_resolved_by_the_llm(indexed):
    a = answer("How many fans does my Aurora ACT 1250 have?", Session(), data_dir=TEST_DATA)
    assert [d.id for d in a.devices] == ["act1250"]
    assert a.status == "answered"


def test_alias_shared_by_two_devices_is_settled_by_a_unique_name(indexed):
    # "aurora" is an alias of both devices in tests/aliases.json
    a = answer("How many fans does my aurora R16 have?", Session(), data_dir=TEST_DATA)
    assert [d.id for d in a.devices] == ["r16"]


def test_alias_shared_by_two_devices_alone_asks_which_device(indexed):
    a = answer("How many fans does my aurora have?", Session(), data_dir=TEST_DATA)
    assert a.status == "needs-device"
    assert a.chunks == []


def test_answer_refers_to_the_device_by_its_marketing_name(ask):
    a = ask(MEMORY_Q, "act1250")
    assert [d.name for d in a.devices] == ["Alienware Aurora ACT1250"]
    assert "Alienware Aurora ACT1250" in a.text, a.text


def test_question_whose_best_passage_scores_below_the_threshold_is_refused_without_generating(
    indexed, monkeypatch
):
    # Model calls go through (reading "I" in the question still asks the model to rewrite
    # it) but are recorded, so we can check none of them generated an answer.
    real_chat = ollama.chat
    system_prompts = []

    def recording_chat(*args, **kwargs):
        system_prompts.append(kwargs["messages"][0]["content"])
        return real_chat(*args, **kwargs)

    monkeypatch.setattr(config, "REFUSE_BELOW", 1.01)  # no passage can reach this
    monkeypatch.setattr(ollama, "chat", recording_chat)
    a = answer("How do I install Ubuntu on the ACT1250?", Session(), data_dir=TEST_DATA)
    assert SYSTEM_PROMPT not in system_prompts, "the generator ran below the threshold"
    assert a.status == "refused"
    assert [d.id for d in a.devices] == ["act1250"]
    assert a.citations == []
    assert "not found in the manuals" in a.text.lower()
    assert "full question" in a.text.lower(), a.text  # nothing matched: maybe it was unclear


def test_question_the_passages_dont_answer_is_refused_by_the_model(indexed, monkeypatch):
    monkeypatch.setattr(config, "REFUSE_BELOW", 0.0)  # only the model can refuse
    a = answer("What is the retail price of the Alienware Aurora ACT1250?", Session(), data_dir=TEST_DATA)
    assert a.status == "refused", a.text
    assert a.citations == []
    assert "not found in the manuals" in a.text.lower()



def test_answerable_question_with_a_weak_best_match_is_still_answered(ask):
    # Short questions built around a part number score low (here ~0.52) though the right
    # passage is retrieved; the threshold must stay below them (see eval/baseline.md).
    a = ask("What is the CR2032 battery used for?", "act1250")
    assert a.status == "answered", a.text
    assert any("CR2032" in c.text for c in a.chunks)
