"""Entry points: `index` and `ask`."""

import argparse

from alienrag import config


def index_main() -> None:
    ap = argparse.ArgumentParser(description="Parse, chunk and embed manuals in Manuals/.")
    ap.add_argument("--force", action="store_true", help="re-parse even if unchanged")
    ap.add_argument("--rechunk", action="store_true", help="re-chunk from cached parses")
    args = ap.parse_args()

    from alienrag.indexing import index

    index(config.MANUALS_DIR, config.DATA_DIR, force=args.force, rechunk=args.rechunk)


def ask_main() -> None:
    ap = argparse.ArgumentParser(description="Ask a question about an Alienware computer.")
    ap.add_argument("question")
    ap.add_argument(
        "--product",
        action="append",
        help="force the device by model code, e.g. act1250 or r16 (repeatable); "
        "by default the device is worked out from the question",
    )
    args = ap.parse_args()

    from alienrag.answer import Session, UnknownDevice, answer

    try:
        result = answer(args.question, Session(devices=args.product or []))
    except UnknownDevice as e:
        raise SystemExit(f"ask: {e}")
    if result.devices:
        print(" / ".join(d.name for d in result.devices) + "\n")
    print(result.text.strip())
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  {c.source}.pdf, page {c.page}")
