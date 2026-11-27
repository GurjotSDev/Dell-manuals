"""Comparison questions across devices (#9).

Runs `answer` against the real manuals and real Ollama models, sharing data/test-index
with the other test files; indexing is a no-op when nothing changed.
"""

import re
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

GPU_LENGTH_Q = (
    "What is the maximum graphics card length for the Alienware Aurora ACT1250 "
    "and for the Alienware Aurora R16?"
)


@pytest.fixture(scope="module")
def indexed():
    index(MANUALS, TEST_DATA, aliases_file=ALIASES, log=lambda *_: None)


@pytest.fixture(scope="module")
def gpu_length(indexed):
    return answer(GPU_LENGTH_Q, Session(), data_dir=TEST_DATA)


def test_comparison_retrieves_top_chunks_from_every_device(gpu_length):
    a = gpu_length
    assert [d.id for d in a.devices] == ["act1250", "r16"]
    # One manual must not crowd the other out of the top of the ranking.
    assert {c.product for c in a.chunks[:2]} == {"act1250", "r16"}
    # Ground truth from the PDFs: ACT1250 p.26 (305 mm), R16 p.61 (312 mm).
    top = {(c.product, p) for c in a.chunks[:5] for p in c.pages}
    assert ("act1250", 26) in top, top
    assert ("r16", 61) in top, top


def _sections(text: str, names: list[str]) -> dict[str, str]:
    """The answer's text under each device heading: a line that starts with the
    device's full name (after any Markdown heading or bold markup)."""
    starts = {}
    for name in names:
        m = re.search(rf"^[#*\s]*{re.escape(name)}\b.*$", text, re.MULTILINE)
        assert m, f"no heading for {name}:\n{text}"
        starts[name] = m.start()
    order = sorted(names, key=starts.__getitem__)
    ends = [starts[n] for n in order[1:]] + [len(text)]
    return {n: text[starts[n] : e] for n, e in zip(order, ends)}


def test_comparison_answer_has_a_section_per_device_citing_that_devices_manual(gpu_length):
    a = gpu_length
    assert a.status == "answered", a.text
    sections = _sections(a.text, ["Alienware Aurora ACT1250", "Alienware Aurora R16"])
    act, r16 = sections["Alienware Aurora ACT1250"], sections["Alienware Aurora R16"]
    assert "305" in act and "[ACT1250 p." in act and "[R16 p." not in act, act
    assert "312" in r16 and "[R16 p." in r16 and "[ACT1250 p." not in r16, r16
    assert {c.source for c in a.citations} == {
        "act1250-owners-manual-en-us",
        "alienware-aurora-r16-owners-manual-en-us",
    }


_NOT_COVERED = re.compile(
    r"(doesn't|does not|don't|do not)\s+(cover|mention|include|describe|document)"
    r"|not\s+(covered|mentioned|documented|described|found)|no\s+(information|mention)",
    re.IGNORECASE,
)


def test_comparison_says_which_devices_manual_lacks_the_topic(indexed):
    # Only the R16 manual has an SMA cable (Removing the SMA cable, p.70-71); the ACT1250
    # manual never mentions one.
    a = answer(
        "How do I remove the SMA cable from the Aurora R16 and from the Aurora ACT1250?",
        Session(),
        data_dir=TEST_DATA,
    )
    assert a.status == "answered", a.text
    sections = _sections(a.text, ["Alienware Aurora R16", "Alienware Aurora ACT1250"])
    act, r16 = sections["Alienware Aurora ACT1250"], sections["Alienware Aurora R16"]
    assert "[R16 p." in r16 and not _NOT_COVERED.search(r16), r16
    assert _NOT_COVERED.search(act) and "[R16 p." not in act, act


def test_compare_that_with_another_device_adds_it_to_the_session(indexed):
    session = Session()
    answer(
        "What is the maximum graphics card length for the Aurora ACT1250?",
        session,
        data_dir=TEST_DATA,
    )
    a = answer("Compare that with the R16", session, data_dir=TEST_DATA)
    assert [d.id for d in a.devices] == ["act1250", "r16"]
    assert session.active == ["act1250", "r16"]
    assert "length" in a.query.lower(), a.query
    top = {(c.product, p) for c in a.chunks[:5] for p in c.pages}
    assert ("act1250", 26) in top and ("r16", 61) in top, top
    sections = _sections(a.text, ["Alienware Aurora ACT1250", "Alienware Aurora R16"])
    assert "305" in sections["Alienware Aurora ACT1250"], a.text
    assert "312" in sections["Alienware Aurora R16"], a.text
