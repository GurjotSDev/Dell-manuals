"""Answer rules (#7): procedures, warnings, spec options and configuration variants.

Runs `answer` against the real manuals and real Ollama models, sharing data/test-index
with the other test files; indexing is a no-op when nothing changed.
"""

import re
from pathlib import Path

import pytest

from alienrag.answer import Session, answer
from alienrag.indexing import index
from alienrag.session import Turn

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


def _gpus(text: str) -> set[str]:
    found = re.findall(r"RTX\s*(\d{4})(\s*Ti)?(\s*SUPER)?", text, re.IGNORECASE)
    return {" ".join(p.strip().upper() for p in g if p.strip()) for g in found}


@pytest.fixture(scope="module")
def gpu(indexed):
    return answer("What GPU does the ACT1250 have?", Session(), data_dir=TEST_DATA)


def test_spec_question_lists_every_documented_option(gpu):
    # Table 14 "Discrete graphics specifications" (p.23-24) spans two chunks.
    a = gpu
    assert a.status == "answered"
    expected = {
        "4060", "4060 TI", "4070 SUPER", "4070 TI SUPER", "4080 SUPER", "4090",
        "5060", "5060 TI", "5070", "5070 TI", "5080",
    }  # fmt: skip
    assert expected <= _gpus(a.text), a.text


LIQUID_COOLER_Q = "How do I remove the 240 mm processor liquid-cooling assembly from the ACT1250?"


@pytest.fixture(scope="module")
def liquid_cooler(indexed):
    return answer(LIQUID_COOLER_Q, Session(), data_dir=TEST_DATA)


def test_procedure_is_numbered_steps_in_manual_order_with_prerequisites(liquid_cooler):
    # p.83-84: prerequisites (before working inside, remove the left-side cover), then
    # steps 1-6 from "Lay the computer on the right side" to "Release the spring latch".
    text = liquid_cooler.text.lower()
    assert {83, 84} & {c.page for c in liquid_cooler.citations}, liquid_cooler.text
    assert "left-side cover" in text, liquid_cooler.text
    order = ["right side", "m3x5", "fan pump", "captive screws", "spring latch"]
    at = [text.find(s) for s in order]
    assert -1 not in at and at == sorted(at), (at, liquid_cooler.text)
    assert re.search(r"^\s*1\.", liquid_cooler.text, re.MULTILINE), liquid_cooler.text


def test_every_warning_and_caution_of_the_procedure_is_in_the_answer(liquid_cooler):
    # p.83: CAUTION technicians only; WARNING the assembly may be hot; CAUTION do not
    # touch the heat transfer areas (skin oils).
    text = liquid_cooler.text.lower()
    assert "authorized service technicians" in text, liquid_cooler.text
    assert "sufficient time to cool" in text, liquid_cooler.text
    assert "oils in your skin" in text, liquid_cooler.text


@pytest.mark.parametrize(
    ("product", "question", "warnings"),
    [
        # R16 p.89, Installing the processor liquid-cooling assembly (240 mm)
        (
            "r16",
            "How do I install the 240 mm processor liquid-cooling assembly?",
            [
                "information in this installation section is intended for authorized service technicians",
                "incorrect alignment of the processor liquid-cooling assembly can damage",
                "use the thermal grease that is provided in the kit",
            ],
        ),
        # ACT1250 p.62, Installing the M.2 2230 solid state drive
        ("act1250", "How do I install the M.2 2230 SSD?", ["solid state drives are fragile"]),
        # ACT1250 p.55, Removing the memory module
        ("act1250", "How do I remove the memory module?", ["hold the memory module by the edges"]),
    ],
)
def test_warnings_and_cautions_keep_the_manuals_wording(indexed, product, question, warnings):
    a = answer(question, Session(devices=[product]), data_dir=TEST_DATA)
    text = " ".join(a.text.lower().split())
    missing = [w for w in warnings if w not in text]
    assert not missing, a.text


R16_COOLER_Q = "How do I remove the processor liquid-cooling assembly from the R16?"


def test_configuration_dependent_procedure_gives_every_variant_or_asks_which(indexed):
    # The R16 ships with a 120 mm (p.85) or a 240 mm (p.88) liquid cooler, removed
    # differently (one radiator screw vs two). No configuration is known, so the answer
    # must not pick one: it presents both, labelled, or asks which the user has.
    a = answer(R16_COOLER_Q, Session(), data_dir=TEST_DATA)
    assert a.status in {"answered", "needs-config-clarification"}, a.status
    text = a.text.lower()
    assert "120 mm" in text and "240 mm" in text, a.text
    if a.status == "answered":
        assert {85} & {c.page for c in a.citations} and {88} & {c.page for c in a.citations}, a.text
    else:
        assert "?" in a.text and a.citations == [], a.text


def test_stated_configuration_answers_for_that_variant_only(indexed):
    a = answer(R16_COOLER_Q, Session(configuration=["120 mm liquid cooler"]), data_dir=TEST_DATA)
    assert a.status == "answered"
    cited = {c.page for c in a.citations}
    assert 85 in cited and 88 not in cited, a.text  # p.88: the 240 mm removal


def test_replying_to_a_configuration_question_answers_the_original_question(indexed):
    asked = (
        "Which processor liquid-cooling assembly does your Alienware Aurora R16 have: "
        "the 120 mm or the 240 mm one?"
    )
    session = Session(active=["r16"], turns=[Turn(R16_COOLER_Q, asked)])
    a = answer("Mine has the 120 mm one.", session, data_dir=TEST_DATA)
    assert any("120" in c for c in session.configuration), session.configuration
    assert a.status == "answered"
    assert any(
        "Removing the processor liquid-cooling assembly (120 mm)" in c.headings for c in a.chunks
    ), a.query
    assert 85 in {c.page for c in a.citations}, a.text


def test_a_retrieved_section_reaches_the_model_whole(gpu):
    # Search ranks only the first half of Table 14 (p.23); the rest of the "Video"
    # section (RTX 5060-5080 rows, p.23-24) must come with it.
    a = gpu
    video = [c for c in a.chunks if c.headings[-1] == "Video"]
    assert any("RTX 4090" in c.text for c in video)
    assert any("RTX 5070 Ti" in c.text and "300 W" in c.text for c in video), [c.id for c in a.chunks]
    # Retrieval's own ranking stays first; whole-section additions come after it.
    assert a.chunks[0].headings[-1] == "Video"


def test_an_answer_that_gives_no_procedure_gets_no_safety_notes(indexed):
    # "Press F2" can be cited from the BIOS-update procedures (p.125-126), whose
    # BitLocker and power-off cautions don't apply to entering BIOS setup.
    a = answer("What key do I press to get into BIOS setup on the ACT1250?", Session(), data_dir=TEST_DATA)
    assert a.status == "answered"
    assert "F2" in a.text, a.text
    assert "Safety notes" not in a.text, a.text
