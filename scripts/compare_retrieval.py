"""Compare two indexes on the golden set by where each expected page group ranks.

Finer than page-hit@5, which says only whether a group is in the top 5. Retrieval only:
each question is read as a fresh session's first message and retrieved as `answer` does,
with no generation. Used for the #10 image-caption experiment (eval/baseline.md):

    uv run python scripts/compare_retrieval.py data data/index-captions
"""

import sys
from pathlib import Path

from alienrag import config
from alienrag.answer import _retrieve
from alienrag.evaluation import EvalItem, load_golden
from alienrag.registry import load_registry
from alienrag.session import Session, read


def ranks(
    data_dir: Path, items: list[EvalItem], queries: dict[str, str]
) -> dict[tuple[str, str, tuple[int, ...]], int | None]:
    """Rank (1-based) of the first chunk hitting each expected page group; None if missed."""
    registry = load_registry(data_dir)
    by_id = {d.id: d for d in registry}
    out: dict[tuple[str, str, tuple[int, ...]], int | None] = {}
    for it in items:
        ranked, _ = _retrieve(data_dir, queries[it.id], [by_id[i] for i in it.devices], registry)
        for device, groups in it.pages.items():
            for g in groups:
                out[(it.id, device, tuple(g))] = next(
                    (n for n, h in enumerate(ranked, 1) if h.product == device and set(g) & set(h.pages)),
                    None,
                )
    return out


def _mrr(r: dict) -> float:
    return sum(1 / v for v in r.values() if v) / len(r)


if __name__ == "__main__":
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    items = [i for i in load_golden(config.GOLDEN_FILE) if i.pages]
    queries = {it.id: read(it.question, Session()).query for it in items}
    ra, rb = ranks(a, items, queries), ranks(b, items, queries)
    for key in ra:
        if ra[key] != rb[key]:
            print(f"{key[0]} {key[1]} p.{'-'.join(map(str, key[2]))}: {ra[key]} -> {rb[key]}")
    print(f"{len(ra)} groups, MRR {a}: {_mrr(ra):.3f}, {b}: {_mrr(rb):.3f}")
