"""Golden eval set: run each question through `answer` and score the results.

Metrics read straight off `Answer`: device-resolution accuracy, retrieval page-hit@5
and correct-refusal rate. Answer prose is not graded; the user spot-checks it.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from alienrag.answer import Answer

CATEGORIES = (
    "single-device",
    "comparison",
    "no-device",
    "configuration-dependent",
    "not-in-manual",
)
# Answer statuses from the spec, including ones the pipeline doesn't produce yet
# (needs-config-clarification is a later issue), so the eval set can already expect them.
STATUSES = ("answered", "refused", "needs-device", "needs-config-clarification")
REFUSED = "refused"
HIT_AT = 5  # page-hit@5: only the top 5 retrieved chunks count


@dataclass
class EvalItem:
    id: str
    category: str
    question: str
    devices: list[str]  # expected model codes; empty for a no-device question
    # model code -> page groups that answer it, e.g. one group per configuration
    # ({"act1250": [[81, 82], [83, 84]]}); every group must be retrieved. Empty when
    # the manuals don't answer the question.
    pages: dict[str, list[list[int]]]
    status: str  # expected Answer.status
    answer: str  # expected answer, for the user's spot-check only
    reviewed: bool = False  # the user has verified this item against the manual


def load_golden(path: Path) -> list[EvalItem]:
    """The eval set: one JSON object per line with the EvalItem fields."""
    items: list[EvalItem] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        it = EvalItem(**json.loads(line))
        if it.category not in CATEGORIES:
            raise ValueError(f"{path.name}:{n}: unknown category {it.category!r}")
        if it.status not in STATUSES:
            raise ValueError(f"{path.name}:{n}: unknown status {it.status!r}")
        items.append(it)
    return items


@dataclass
class Rate:
    correct: int = 0
    total: int = 0


@dataclass
class Failure:
    item: EvalItem
    got: Answer
    reasons: list[str]


@dataclass
class Report:
    devices: Rate = field(default_factory=Rate)
    pages: Rate = field(default_factory=Rate)
    refusals: Rate = field(default_factory=Rate)
    false_refusals: int = 0  # questions the manuals answer that were refused
    failures: list[Failure] = field(default_factory=list)


def _top_pages(got: Answer) -> dict[str, set[int]]:
    """Pages of the top-5 chunks, per device."""
    out: dict[str, set[int]] = {}
    for c in got.chunks[:HIT_AT]:
        out.setdefault(c.product, set()).update(c.pages)
    return out


def _page_hit(expected: dict[str, list[list[int]]], top: dict[str, set[int]]) -> bool:
    """Every expected page group has a page in a top-5 chunk of its device's manual."""
    return all(
        top.get(device, set()) & set(group)
        for device, groups in expected.items()
        for group in groups
    )


def _fmt_ids(ids: list[str]) -> str:
    return " + ".join(sorted(ids)) or "none"


def _fmt_pages(by_device: dict[str, list[list[int]]]) -> str:
    return "; ".join(
        f"{d} " + " & ".join("p." + ",".join(map(str, g)) for g in groups)
        for d, groups in by_device.items()
    ) or "none"


def evaluate(items: list[EvalItem], ask: Callable[[EvalItem], Answer]) -> Report:
    """Ask every item's question and score the answers."""
    report = Report()
    for it in items:
        got = ask(it)
        reasons: list[str] = []

        got_ids = [d.id for d in got.devices]
        report.devices.total += 1
        if set(got_ids) == set(it.devices):
            report.devices.correct += 1
        else:
            reasons.append(f"devices: expected {_fmt_ids(it.devices)}, got {_fmt_ids(got_ids)}")

        if it.pages:
            report.pages.total += 1
            top = _top_pages(got)
            if _page_hit(it.pages, top):
                report.pages.correct += 1
            else:
                got_pages = {d: [sorted(ps)] for d, ps in top.items()}
                reasons.append(
                    f"pages@{HIT_AT}: expected {_fmt_pages(it.pages)}, got {_fmt_pages(got_pages)}"
                )

        if it.status == REFUSED:
            report.refusals.total += 1
            report.refusals.correct += got.status == REFUSED
        elif got.status == REFUSED:
            report.false_refusals += 1
        if got.status != it.status:
            reasons.append(f"status: expected {it.status}, got {got.status}")

        if reasons:
            report.failures.append(Failure(it, got, reasons))
    return report


def _fmt_rate(r: Rate) -> str:
    return f"{r.correct}/{r.total} ({r.correct / r.total:.0%})" if r.total else "n/a"


def format_report(report: Report) -> str:
    lines = [
        f"Device-resolution accuracy: {_fmt_rate(report.devices)}",
        f"Retrieval page-hit@{HIT_AT}: {_fmt_rate(report.pages)}",
        f"Correct-refusal rate: {_fmt_rate(report.refusals)}, false refusals: {report.false_refusals}",
        "",
    ]
    if not report.failures:
        lines.append("Failures: none")
    else:
        lines.append(f"Failures ({len(report.failures)}):")
        for f in report.failures:
            lines.append(f"[{f.item.id}] ({f.item.category}) {f.item.question}")
            lines += [f"  {r}" for r in f.reasons]
    return "\n".join(lines)
