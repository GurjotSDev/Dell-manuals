"""Entry points: `index` and `ask`."""

import argparse
from typing import TYPE_CHECKING

from alienrag import config

if TYPE_CHECKING:
    from alienrag.answer import Answer


def index_main() -> None:
    ap = argparse.ArgumentParser(description="Parse, chunk and embed manuals in Manuals/.")
    ap.add_argument("--force", action="store_true", help="re-parse even if unchanged")
    ap.add_argument("--rechunk", action="store_true", help="re-chunk from cached parses")
    args = ap.parse_args()

    from alienrag.indexing import index

    index(config.MANUALS_DIR, config.DATA_DIR, force=args.force, rechunk=args.rechunk)


def _print_answer(result: "Answer") -> None:
    if result.devices:
        print(" / ".join(d.name for d in result.devices) + "\n")
    print(result.text.strip())
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  {c.source}.pdf, page {c.page}")


def ask_main() -> None:
    ap = argparse.ArgumentParser(
        description="Ask a question about an Alienware computer. "
        "With no question, start a chat that remembers devices and configuration until you exit."
    )
    ap.add_argument("question", nargs="?", help="omit to start an interactive chat")
    ap.add_argument(
        "--product",
        action="append",
        help="force the device by model code, e.g. act1250 or r16 (repeatable); "
        "by default the device is worked out from the question",
    )
    args = ap.parse_args()

    from alienrag.answer import Session, UnknownDevice, answer

    session = Session(devices=args.product or [])
    if args.question is not None:
        try:
            _print_answer(answer(args.question, session))
        except UnknownDevice as e:
            raise SystemExit(f"ask: {e}")
        return

    print("Ask about your Alienware computer. Empty line, 'exit' or Ctrl+D to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"", "exit", "quit"}:
            break
        try:
            result = answer(question, session)
        except UnknownDevice as e:
            raise SystemExit(f"ask: {e}")
        print()
        _print_answer(result)
        if result.query and result.query != question:
            print(f"\n(searched for: {result.query})")
