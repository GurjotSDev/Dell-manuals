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
from alienrag.registry import load_registry, model_code
from alienrag.store import Hit
from alienrag.ui import app, reply, respond

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
                    product=model_code(c["source"]),
                    text=c["text"],
                    headings=c["headings"],
                    pages=c["pages"],
                    images=c["images"],
                    score=1.0,
                    similarity=0.8,
                )
    raise LookupError(heading)


def _act1250():
    return [d for d in load_registry(TEST_DATA) if d.id == "act1250"]


def _answered(
    hits: list[Hit],
    citations: list[Citation],
    text: str = "1. Remove the memory module [ACT1250 p.54].",
    query: str = "",
) -> Answer:
    return Answer(
        devices=_act1250(),
        status="answered",
        chunks=hits,
        text=text,
        citations=citations,
        query=query,
    )


def _reply(a: Answer, question: str = ""):
    return reply(a, question, data_dir=TEST_DATA, manuals_dir=MANUALS)


def _sizes(paths: list[Path]) -> list[tuple[int, int]]:
    out = []
    for path in paths:
        with Image.open(path) as im:
            out.append(im.size)
    return out


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
    # The memory-module figure; note and warning icons dropped.
    assert _sizes(r.images) == [(793, 457)]


def test_a_procedure_shows_its_own_figure_not_its_neighbours_on_a_shared_page(indexed):
    removing = _chunk(ACT1250, "Removing the memory module")  # p.54-55, figure 793x457
    installing = _chunk(ACT1250, "Installing the memory module")  # p.55-56, figure 795x455
    text = (
        "Steps\n"
        "1. Lay the computer on the right side [ACT1250 p.54].\n"
        "2. Push the securing clips away from the memory module [ACT1250 p.55].\n"
        "3. Lift the memory module off the memory-module slot [ACT1250 p.55]."
    )
    cited = [Citation(ACT1250, 54), Citation(ACT1250, 55)]
    r = _reply(_answered([removing, installing], cited, text))
    assert _sizes(r.images) == [(793, 457)]


def test_answer_is_headed_by_its_device_and_says_what_a_follow_up_searched_for(indexed):
    memory = _chunk(ACT1250, "Removing the memory module")
    a = _answered([memory], [Citation(ACT1250, 54)], query="install the memory module")
    r = _reply(a, "and how do I put it back in?")
    assert r.text.startswith("**Alienware Aurora ACT1250**\n")
    assert "(searched for: install the memory module)" in r.text
    same = _answered([memory], [Citation(ACT1250, 54)], query="remove the memory module")
    assert "searched for" not in _reply(same, "remove the memory module").text


def test_needs_device_is_a_prompt_offering_each_device_as_an_answer(indexed):
    a = Answer(devices=[], status="needs-device", chunks=[], text="Which device?", citations=[])
    r = _reply(a)
    assert r.text == "Which device?"
    assert r.options == ["Alienware Aurora ACT1250", "Alienware Aurora R16"]
    assert r.pages == [] and r.images == []


def test_answers_other_than_needs_device_offer_no_options(indexed):
    memory = _chunk(ACT1250, "Removing the memory module")
    clarify = Answer(
        devices=_act1250(),
        status="needs-config-clarification",
        chunks=[memory],
        text="Is your processor cooled by air or liquid?",
        citations=[],
    )
    r = _reply(clarify)
    assert r.text.endswith("\nIs your processor cooled by air or liquid?")
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
    question = "How do I remove the memory module from the Aurora ACT1250?"
    history = _respond(question, history, session)
    turn = _last_turn(history)
    assert history[0] == {"role": "user", "content": question}
    assert f"{ACT1250}.pdf, page" in turn[0]["content"]
    assert any(p.parent.name == ACT1250 for p in _files(turn))  # rendered pages and images
    history = _respond("and how do I put it back in?", history, session)
    turn = _last_turn(history)
    assert f"{ACT1250}.pdf, page" in turn[0]["content"]
    assert session.active == ["act1250"]


def _pick(value: str, index: int, history: list[dict], session: Session):
    """Click option `index` (value `value`) of a chat message, as the browser does."""
    import gradio as gr

    demo = app(data_dir=TEST_DATA, manuals_dir=MANUALS)
    pick = next(f for f in demo.fns.values() if f.name == "pick").fn
    assert pick is not None
    return pick(gr.SelectData(None, {"index": index, "value": value}), history, session)


def test_chat_asks_which_device_and_picking_an_option_answers_the_question(indexed):
    session = Session()
    history = _respond("How do I remove the memory module?", [], session)
    prompt = _last_turn(history)[0]
    labels = [o["label"] for o in prompt["options"]]
    assert labels == ["Alienware Aurora ACT1250", "Alienware Aurora R16"]
    # The click goes through the handler Gradio calls, with the payload its Chatbot
    # sends for option_select: {index, value} where value is the option's value.
    history, session_out = _pick(prompt["options"][1]["value"], 1, history, session)
    assert session_out is session
    assert [m for m in history if m["role"] == "user"][-1]["content"] == "Alienware Aurora R16"
    r16_source = "alienware-aurora-r16-owners-manual-en-us.pdf, page"
    assert r16_source in _last_turn(history)[0]["content"]


def test_a_failed_turn_keeps_the_question_and_says_so_in_the_chat(tmp_path):
    session = Session()
    history, _ = respond("How do I remove the memory module?", [], session, data_dir=tmp_path)
    assert history[0] == {"role": "user", "content": "How do I remove the memory module?"}
    assert history[1]["role"] == "assistant"
    assert "run `index` first" in history[1]["content"]
