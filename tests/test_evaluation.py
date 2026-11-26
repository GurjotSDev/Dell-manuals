"""Tests for the eval metrics, run over fake Answers (no index or Ollama needed)."""

from pathlib import Path

import pytest

from alienrag.answer import Answer
from alienrag.evaluation import CATEGORIES, EvalItem, evaluate, format_report, load_golden
from alienrag.registry import Device
from alienrag.store import Hit

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden.jsonl"

DEVICES = {
    "act1250": Device("act1250", "Alienware Aurora ACT1250", "act1250-owners-manual-en-us"),
    "r16": Device("r16", "Alienware Aurora R16", "alienware-aurora-r16-owners-manual-en-us"),
}


def item(
    id: str,
    devices: list[str],
    pages: dict[str, list[list[int]]] | None = None,
    status: str = "answered",
    category: str = "single-device",
) -> EvalItem:
    return EvalItem(
        id=id,
        category=category,
        question=f"question {id}",
        devices=devices,
        pages=pages or {},
        status=status,
        answer="",
    )


def hit(product: str, pages: list[int]) -> Hit:
    return Hit(
        id=f"{product}-{pages[0]}",
        source=DEVICES[product].source,
        product=product,
        text="",
        headings=[],
        pages=pages,
        images=[],
        score=1.0,
        similarity=1.0,
    )


def reply(devices: list[str], status: str = "answered", chunks: list[Hit] | None = None) -> Answer:
    return Answer(
        devices=[DEVICES[d] for d in devices],
        status=status,  # type: ignore[arg-type]  # statuses not yet built (e.g. "refused")
        chunks=chunks or [],
        text="",
        citations=[],
    )


def run(pairs: list[tuple[EvalItem, Answer]]):
    answers = {i.id: a for i, a in pairs}
    return evaluate([i for i, _ in pairs], lambda it: answers[it.id])


def test_device_resolution_accuracy_counts_exact_device_set_matches():
    report = run(
        [
            (item("a", ["act1250"]), reply(["act1250"])),
            (item("b", ["r16", "act1250"]), reply(["act1250", "r16"])),  # order doesn't matter
            (item("c", ["r16"]), reply(["act1250"])),
            (item("d", [], status="needs-device"), reply([], status="needs-device")),
        ]
    )
    assert (report.devices.correct, report.devices.total) == (3, 4)


def test_page_hit_at_5_needs_an_expected_page_in_the_top_five_chunks():
    filler = [hit("act1250", [100 + n]) for n in range(5)]
    report = run(
        [
            # hit at rank 1
            (item("a", ["act1250"], {"act1250": [[54, 55]]}), reply(["act1250"], chunks=[hit("act1250", [55, 56])])),
            # right page, but only at rank 6
            (item("b", ["act1250"], {"act1250": [[54]]}), reply(["act1250"], chunks=[*filler, hit("act1250", [54])])),
            # right page number, wrong manual
            (item("c", ["act1250"], {"act1250": [[22]]}), reply(["act1250"], chunks=[hit("r16", [22])])),
            # no expected pages (e.g. not in the manuals): not scored
            (item("d", ["r16"], status="refused"), reply(["r16"])),
        ]
    )
    assert (report.pages.correct, report.pages.total) == (1, 3)


def test_page_hit_at_5_for_a_comparison_needs_a_hit_for_every_device():
    pages = {"act1250": [[22]], "r16": [[23]]}
    report = run(
        [
            (item("both", ["act1250", "r16"], pages), reply(["act1250", "r16"], chunks=[hit("act1250", [22]), hit("r16", [23])])),
            (item("one", ["act1250", "r16"], pages), reply(["act1250", "r16"], chunks=[hit("act1250", [22]), hit("r16", [90])])),
        ]
    )
    assert (report.pages.correct, report.pages.total) == (1, 2)


def test_page_hit_at_5_needs_every_page_group_so_each_configuration_is_retrieved():
    # air-cooler procedure on 81-82, liquid-cooler procedure on 83-84
    pages = {"act1250": [[81, 82], [83, 84]]}
    report = run(
        [
            (item("both", ["act1250"], pages), reply(["act1250"], chunks=[hit("act1250", [83, 84]), hit("act1250", [81, 82])])),
            (item("air-only", ["act1250"], pages), reply(["act1250"], chunks=[hit("act1250", [81, 82])])),
        ]
    )
    assert (report.pages.correct, report.pages.total) == (1, 2)


def test_correct_refusal_rate_counts_refusals_on_questions_the_manuals_dont_answer():
    report = run(
        [
            (item("a", ["r16"], status="refused"), reply(["r16"], status="refused")),
            (item("b", ["r16"], status="refused"), reply(["r16"], status="answered")),
            (item("c", ["act1250"], {"act1250": [[54]]}), reply(["act1250"], status="refused")),
            (item("d", ["act1250"], {"act1250": [[54]]}), reply(["act1250"], status="answered")),
        ]
    )
    assert (report.refusals.correct, report.refusals.total) == (1, 2)
    assert report.false_refusals == 1  # "c" should have been answered


def test_failures_list_each_failing_question_with_what_went_wrong():
    report = run(
        [
            (item("ok", ["act1250"], {"act1250": [[54]]}), reply(["act1250"], chunks=[hit("act1250", [54])])),
            (item("wrong-device", ["r16"], {"r16": [[57]]}), reply(["act1250"], chunks=[hit("act1250", [54])])),
            (item("page-miss", ["act1250"], {"act1250": [[22]]}), reply(["act1250"], chunks=[hit("act1250", [54])])),
            (item("no-refusal", ["r16"], status="refused"), reply(["r16"])),
        ]
    )
    reasons = {f.item.id: f.reasons for f in report.failures}
    assert list(reasons) == ["wrong-device", "page-miss", "no-refusal"]
    assert reasons["wrong-device"][0] == "devices: expected r16, got act1250"
    assert reasons["page-miss"] == ["pages@5: expected act1250 p.22, got act1250 p.54"]
    assert reasons["no-refusal"] == ["status: expected refused, got answered"]


def test_printed_report_gives_the_three_metrics_then_each_failure():
    report = run(
        [
            (item("ok", ["act1250"], {"act1250": [[54]]}), reply(["act1250"], chunks=[hit("act1250", [54])])),
            (item("bad", ["r16"], {"r16": [[57]]}, category="comparison"), reply(["act1250"])),
            (item("refuse", ["r16"], status="refused", category="not-in-manual"), reply(["r16"])),
        ]
    )
    lines = format_report(report).splitlines()
    assert "Device-resolution accuracy: 2/3 (67%)" in lines
    assert "Retrieval page-hit@5: 1/2 (50%)" in lines
    assert "Correct-refusal rate: 0/1 (0%), false refusals: 0" in lines
    text = "\n".join(lines)
    assert "[bad] (comparison) question bad\n  devices: expected r16, got act1250" in text
    assert "[refuse] (not-in-manual) question refuse\n  status: expected refused, got answered" in text
    assert "[ok]" not in text


def test_golden_set_covers_all_five_categories():
    items = load_golden(GOLDEN)
    assert {i.category for i in items} == set(CATEGORIES)
    assert len({i.id for i in items}) == len(items)


def test_golden_set_expectations_fit_their_category():
    for i in load_golden(GOLDEN):
        if i.category == "no-device":
            assert (i.devices, i.pages, i.status) == ([], {}, "needs-device"), i.id
        elif i.category == "not-in-manual":
            assert i.devices and not i.pages and i.status == "refused", i.id
        else:
            # answerable: every expected device has pages that answer it
            assert i.devices and set(i.pages) == set(i.devices), i.id
            assert i.status == "answered", i.id
        if i.category == "comparison":
            assert len(i.devices) >= 2, i.id


@pytest.mark.parametrize(
    ("category", "status", "typo"),
    [("trivia", "needs-device", "trivia"), ("no-device", "need-device", "need-device")],
)
def test_loading_rejects_an_unknown_category_or_status(tmp_path, category, status, typo):
    bad = tmp_path / "golden.jsonl"
    bad.write_text(
        f'{{"id": "x", "category": "{category}", "question": "q", "devices": [], "pages": {{}},'
        f' "status": "{status}", "answer": ""}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=typo):
        load_golden(bad)


def test_printed_report_marks_a_metric_with_nothing_to_score_as_na():
    report = run([(item("d", [], status="needs-device"), reply([], status="needs-device"))])
    lines = format_report(report).splitlines()
    assert "Retrieval page-hit@5: n/a" in lines
    assert "Failures: none" in lines
