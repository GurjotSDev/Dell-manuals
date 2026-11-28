"""Multi-turn behaviour: `answer` called repeatedly with the same session.

Runs against the real manuals and real Ollama models, sharing data/test-index with the
other test files; indexing is a no-op when nothing changed.
"""

from pathlib import Path

import pytest

from alienrag.answer import Session, answer
from alienrag.indexing import index

ROOT = Path(__file__).resolve().parents[1]
MANUALS = ROOT / "Manuals"
TEST_DATA = ROOT / "data" / "test-index"
ALIASES = ROOT / "tests" / "aliases.json"

pytestmark = pytest.mark.skipif(
    not any(MANUALS.glob("*.pdf")), reason="no manual PDFs in Manuals/"
)


@pytest.fixture(scope="module")
def indexed():
    index(MANUALS, TEST_DATA, aliases_file=ALIASES, log=lambda *_: None)


@pytest.fixture
def chat(indexed):
    """A fresh session, and a function asking a question in it."""
    session = Session()
    return lambda question: answer(question, session, data_dir=TEST_DATA)


def test_follow_up_naming_no_device_inherits_the_previous_turns_devices(chat):
    chat("How do I remove the memory module from the Aurora ACT1250?")
    a = chat("and how do I put it back in?")
    assert a.status == "answered"
    assert [d.id for d in a.devices] == ["act1250"]
    assert {c.product for c in a.chunks} == {"act1250"}


def test_stating_a_configuration_mid_chat_keeps_the_active_device(chat):
    chat("How do I remove the memory module from the Aurora ACT1250?")
    a = chat("Mine has liquid cooling.")
    assert [d.id for d in a.devices] == ["act1250"]


def test_follow_up_is_rewritten_into_a_standalone_query_used_for_retrieval(chat):
    chat("How do I remove the memory module from the Aurora ACT1250?")
    a = chat("and how do I put it back in?")
    assert "memory" in a.query.lower(), a.query
    assert any("Installing the memory module" in c.headings for c in a.chunks), a.query


def test_follow_up_naming_another_device_switches_to_it(chat):
    chat("How do I remove the memory module from the Aurora ACT1250?")
    a = chat("what about on the R16?")
    assert [d.id for d in a.devices] == ["r16"]
    assert "memory" in a.query.lower(), a.query
    assert {c.product for c in a.chunks} == {"r16"}


def test_naming_the_device_after_being_asked_which_answers_the_original_question(chat):
    assert chat("How do I remove the memory module?").status == "needs-device"
    a = chat("The Aurora R16")
    assert a.status == "answered"
    assert [d.id for d in a.devices] == ["r16"]
    assert any("Removing the memory module" in c.headings for c in a.chunks), a.query


def test_stated_configuration_is_stored_and_tailors_later_answers(indexed):
    session = Session()
    noted = answer("Mine is an Aurora ACT1250 with liquid cooling.", session, data_dir=TEST_DATA)
    assert any("liquid" in c.lower() for c in session.configuration), session.configuration
    assert noted.chunks == [], noted.query  # a statement asks nothing, so nothing is retrieved
    # The ACT1250 ships air-cooled (fan and heat-sink, p.81-82) or liquid-cooled
    # (240 mm liquid-cooling assembly, p.83-84); only the latter applies here.
    a = answer("How do I remove the processor cooler?", session, data_dir=TEST_DATA)
    assert a.status == "answered"
    assert any("Removing the processor liquid-cooling assembly (240 mm)" in c.headings for c in a.chunks), a.query
    assert {83, 84} & {c.page for c in a.citations}, a.text


def test_configuration_stated_alongside_a_question_is_remembered(indexed):
    session = Session()
    a = answer("My ACT1250 has an RTX 5080. How many fans does it have?", session, data_dir=TEST_DATA)
    assert any("5080" in c for c in session.configuration), session.configuration
    assert a.status == "answered" and a.chunks, a.query


def test_a_new_session_remembers_nothing_from_an_earlier_one(chat):
    chat("How do I remove the memory module from the Aurora ACT1250?")
    a = answer("and how do I put it back in?", Session(), data_dir=TEST_DATA)
    assert a.status == "needs-device"


# Replying to "Which device?" with the full marketing name, as the UI's buttons do.


def test_an_identity_question_is_answered_once_the_device_is_named(indexed):
    session = Session()
    assert answer("What is my motherboard", session, data_dir=TEST_DATA).status == "needs-device"
    a = answer("Alienware Aurora ACT1250", session, data_dir=TEST_DATA)
    assert a.status == "answered" and a.chunks, a.text
    assert [d.id for d in a.devices] == ["act1250"]
    assert "motherboard" in a.query.lower() or "system board" in a.query.lower(), a.query
    assert not a.text.startswith("Noted"), a.text
    assert session.configuration == []


def test_a_procedure_question_is_answered_once_the_device_is_named(indexed):
    session = Session()
    assert answer("How do I remove the memory module?", session, data_dir=TEST_DATA).status == "needs-device"
    a = answer("Alienware Aurora R16", session, data_dir=TEST_DATA)
    assert a.status == "answered"
    assert [d.id for d in a.devices] == ["r16"]
    assert any("Removing the memory module" in c.headings for c in a.chunks), a.query
    assert session.configuration == []
