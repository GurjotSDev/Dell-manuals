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
# Answer statuses, including ones the pipeline doesn't produce yet (refusal is a later
# issue), so the eval set can already expect them.
STATUSES = ("answered", "refused", "needs-device", "needs-config-clarification")
REFUSED = "refused"


@dataclass
class EvalItem:
    id: str
    category: str
    question: str
    devices: list[str]  # expected model codes; empty for a no-device question
    pages: dict[str, list[int]]  # model code -> pages that answer it; empty when none do
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


HIT_AT = 5


def _page_hit(expected: dict[str, list[int]], got: Answer) -> bool:
    """Every device with expected pages has one of them in a top-5 chunk of its manual."""
    top = got.chunks[:HIT_AT]
    return all(
        any(c.product == device and set(c.pages) & set(pages) for c in top)
        for device, pages in expected.items()
    )


def _ids(ids: list[str]) -> str:
    return " + ".join(sorted(ids)) or "none"


def _pages(by_device: dict[str, list[int]]) -> str:
    return "; ".join(
        f"{d} p.{','.join(map(str, ps))}" for d, ps in by_device.items()
    ) or "none"


def _top_pages(got: Answer) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for c in got.chunks[:HIT_AT]:
        out.setdefault(c.product, [])
        out[c.product] += [p for p in c.pages if p not in out[c.product]]
    return out


def evaluate(items: list[EvalItem], ask: Callable[[str], Answer]) -> Report:
    """Ask every item's question and score the answers."""
    report = Report()
    for it in items:
        got = ask(it.question)
        reasons: list[str] = []

        got_ids = [d.id for d in got.devices]
        report.devices.total += 1
        if set(got_ids) == set(it.devices):
            report.devices.correct += 1
        else:
            reasons.append(f"devices: expected {_ids(it.devices)}, got {_ids(got_ids)}")

        if it.pages:
            report.pages.total += 1
            if _page_hit(it.pages, got):
                report.pages.correct += 1
            else:
                reasons.append(f"pages@{HIT_AT}: expected {_pages(it.pages)}, got {_pages(_top_pages(got))}")

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


def _rate(r: Rate) -> str:
    return f"{r.correct}/{r.total} ({r.correct / r.total:.0%})" if r.total else "n/a"


def format_report(report: Report) -> str:
    lines = [
        f"Device-resolution accuracy: {_rate(report.devices)}",
        f"Retrieval page-hit@{HIT_AT}: {_rate(report.pages)}",
        f"Correct-refusal rate: {_rate(report.refusals)}, false refusals: {report.false_refusals}",
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
