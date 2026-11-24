"""Print quality stats and samples for the chunk files in data/chunks/."""

import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

CHUNKS = Path(__file__).resolve().parents[1] / "data" / "chunks"
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


def main(n_samples: int = 3, seed: int = 0) -> None:
    for f in sorted(CHUNKS.glob("*.jsonl")):
        rows = [json.loads(l) for l in f.open(encoding="utf-8")]
        toks = [r["n_tokens"] for r in rows]
        print(f"\n=== {f.stem}: {len(rows)} chunks")
        print(f"tokens  min={min(toks)} median={statistics.median(toks):.0f} max={max(toks)}")
        print(f"tiny (<40 tok): {sum(t < 40 for t in toks)}   over 512: {sum(t > 512 for t in toks)}")
        print(f"no headings: {sum(not r['headings'] for r in rows)}   no pages: {sum(not r['pages'] for r in rows)}")
        print(f"chunks with images: {sum(bool(r['images']) for r in rows)}  "
              f"(images linked: {sum(len(r['images']) for r in rows)})")
        warn = sum(bool(re.search(r"\b(WARNING|CAUTION)\b", r["text"])) for r in rows)
        print(f"chunks containing WARNING/CAUTION: {warn}")
        top = Counter(r["headings"][0] if r["headings"] else "<none>" for r in rows).most_common(12)
        print("top-level headings:", *(f"  {c:3d}  {h}" for h, c in top), sep="\n")
        random.seed(seed)
        for r in random.sample(rows, n_samples):
            print(f"\n--- {r['id']}  pages={r['pages']}  tok={r['n_tokens']}  imgs={len(r['images'])}")
            print(r["embed_text"][:700])


if __name__ == "__main__":
    main()
