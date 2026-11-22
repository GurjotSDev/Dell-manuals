"""Entry points: `index` and `ask`."""

import argparse

from alienrag import config


def index_main() -> None:
    ap = argparse.ArgumentParser(description="Parse and chunk manuals in Manuals/.")
    ap.add_argument("--force", action="store_true", help="re-parse even if unchanged")
    ap.add_argument("--rechunk", action="store_true", help="re-chunk from cached parses")
    args = ap.parse_args()

    from alienrag.indexing import index

    index(config.MANUALS_DIR, config.DATA_DIR, force=args.force, rechunk=args.rechunk)


def ask_main() -> None:
    raise SystemExit("ask: not implemented yet (build step 4)")
