"""The web UI: what a chat turn shows for an Answer, and the chat handler.

`reply` tests build Answers by hand from the test index's real chunks and render real
manual pages, so they need no model. `respond` tests run real Ollama models, like
test_session.py.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from alienrag.answer import Answer, Citation, Session
from alienrag.indexing import index
from alienrag.registry import load_registry
from alienrag.store import Hit
from alienrag.ui import reply, respond

ROOT = Path(__file__).resolve().parents[1]
MANUALS = ROOT / "Manuals"
TEST_DATA = ROOT / "data" / "test-index"
ALIASES = ROOT / "tests" / "aliases.json"
ACT1250 = "act1250-owners-manual-en-us"

pytestmark = pytest.mark.skipif(
    not any(MANUALS.glob("*.pdf")), reason="no manual PDFs in Manuals/"
)


@pytest.fixture(scope="module")
def indexed():
    index(MANUALS, TEST_DATA, aliases_file=ALIASES, log=lambda *_: None)


def _chunk(source: str, heading: str) -> Hit:
    """The first chunk of `source` under `heading`, as search would return it."""
    with open(TEST_DATA / "chunks" / f"{source}.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if heading in c["headings"]:
                return Hit(
                    id=c["id"],
                    source=c["source"],
                    product="act1250",
                    text=c["text"],
                    headings=c["headings"],
                    pages=c["pages"],
                    images=c["images"],
                    score=1.0,
                    similarity=0.8,
                )
    raise LookupError(heading)


def _answered(hits: list[Hit], citations: list[Citation]) -> Answer:
    devices = [d for d in load_registry(TEST_DATA) if d.id == "act1250"]
    return Answer(
        devices=devices,
        status="answered",
        chunks=hits,
        text="1. Remove the memory module [ACT1250 p.54].",
        citations=citations,
    )


def _reply(a: Answer):
    return reply(a, data_dir=TEST_DATA, manuals_dir=MANUALS)


def test_answer_lists_its_sources_and_shows_each_cited_page_rendered(indexed):
    memory = _chunk(ACT1250, "Removing the memory module")  # p.54-55
    r = _reply(_answered([memory], [Citation(ACT1250, 54)]))
    assert "Remove the memory module" in r.text
    assert f"{ACT1250}.pdf, page 54" in r.text
    assert [p.citation for p in r.pages] == [Citation(ACT1250, 54)]
    with Image.open(r.pages[0].path) as im:
        assert im.format == "PNG"
        assert im.height > im.width  # a portrait manual page


def test_answer_shows_the_procedure_images_of_the_cited_chunks_only(indexed):
    memory = _chunk(ACT1250, "Removing the memory module")  # p.54-55: 1 figure, 2 icons
    wireless = _chunk(ACT1250, "Removing the wireless card")  # p.66-67, not cited
    r = _reply(_answered([memory, wireless], [Citation(ACT1250, 54)]))
    sizes = []
    for path in r.images:
        with Image.open(path) as im:
            sizes.append(im.size)
    assert sizes == [(793, 457)]  # the memory-module figure; note and warning icons dropped


def test_needs_device_is_a_prompt_offering_each_device_as_an_answer(indexed):
    a = Answer(devices=[], status="needs-device", chunks=[], text="Which device?", citations=[])
    r = _reply(a)
    assert r.text == "Which device?"
    assert r.options == ["Alienware Aurora ACT1250", "Alienware Aurora R16"]
    assert r.pages == [] and r.images == []


def test_answers_other_than_needs_device_offer_no_options(indexed):
    memory = _chunk(ACT1250, "Removing the memory module")
    clarify = Answer(
        devices=_answered([], []).devices,
        status="needs-config-clarification",
        chunks=[memory],
        text="Is your processor cooled by air or liquid?",
        citations=[],
    )
    r = _reply(clarify)
    assert r.text == "Is your processor cooled by air or liquid?"
    assert r.options == []
    assert _reply(_answered([memory], [Citation(ACT1250, 54)])).options == []


# --- respond: the chat handler, against real models ---------------------------------


def _respond(message: str, history: list[dict], session: Session) -> list[dict]:
    history, session_out = respond(
        message, history, session, data_dir=TEST_DATA, manuals_dir=MANUALS
    )
    assert session_out is session  # one session per tab, kept across turns
    return history


def _last_turn(history: list[dict]) -> list[dict]:
    """The assistant messages after the last user message."""
    last_user = max(i for i, m in enumerate(history) if m["role"] == "user")
    return history[last_user + 1 :]


def _files(messages: list[dict]) -> list[Path]:
    return [Path(m["content"]["path"]) for m in messages if isinstance(m["content"], dict)]


def test_chat_answers_with_cited_pages_and_follow_ups_keep_the_device(indexed):
    session, history = Session(), []
    history = _respond("How do I remove the memory module from the Aurora ACT1250?", history, session)
    turn = _last_turn(history)
    assert history[0] == {"role": "user", "content": "How do I remove the memory module from the Aurora ACT1250?"}
    assert f"{ACT1250}.pdf, page" in turn[0]["content"]
    assert any(p.parent.name == ACT1250 for p in _files(turn))  # rendered pages and images
    history = _respond("and how do I put it back in?", history, session)
    turn = _last_turn(history)
    assert f"{ACT1250}.pdf, page" in turn[0]["content"]
    assert session.active == ["act1250"]


def test_chat_asks_which_device_and_picking_an_option_answers_the_question(indexed):
    session = Session()
    history = _respond("How do I remove the memory module?", [], session)
    prompt = _last_turn(history)[0]
    labels = [o["label"] for o in prompt["options"]]
    assert labels == ["Alienware Aurora ACT1250", "Alienware Aurora R16"]
    history = _respond(prompt["options"][1]["value"], history, session)
    assert "alienware-aurora-r16-owners-manual-en-us.pdf, page" in _last_turn(history)[0]["content"]
