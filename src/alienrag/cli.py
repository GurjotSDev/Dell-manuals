"""Entry points: `index`, `ask`, `eval` and `ui`."""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from alienrag import config

if TYPE_CHECKING:
    from alienrag.answer import Answer


def index_main() -> None:
    ap = argparse.ArgumentParser(description="Parse, chunk and embed manuals in Manuals/.")
    ap.add_argument("--force", action="store_true", help="re-parse even if unchanged")
    ap.add_argument("--rechunk", action="store_true", help="re-chunk from cached parses")
    ap.add_argument(
        "--captions",
        action=argparse.BooleanOptionalAction,
        default=config.CAPTIONS,
        help=f"add {config.CAPTION_MODEL} descriptions of procedure images to the embedded "
        f"text (default: {'on' if config.CAPTIONS else 'off'}); toggling re-chunks and re-embeds",
    )
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="index to build")
    args = ap.parse_args()

    from alienrag.indexing import index

    index(
        config.MANUALS_DIR,
        args.data_dir,
        force=args.force,
        rechunk=args.rechunk,
        captions=args.captions,
    )


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

    from ollama import ResponseError

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
        except (ResponseError, ConnectionError) as e:  # Ollama hiccup: keep the session
            print(f"\nask: model call failed ({e}); try again.")
            continue
        print()
        _print_answer(result)
        if result.query and result.query != question:
            print(f"\n(searched for: {result.query})")


def eval_main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the golden eval set through `answer` and print the metrics."
    )
    ap.add_argument("--golden", type=Path, default=config.GOLDEN_FILE, help="eval set (JSONL)")
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="index to evaluate")
    ap.add_argument(
        "--answers", action="store_true", help="also print every answer next to the expected one"
    )
    args = ap.parse_args()

    from alienrag.answer import Answer, Session, answer
    from alienrag.evaluation import EvalItem, evaluate, format_report, load_golden

    items = load_golden(args.golden)

    def ask(item: EvalItem) -> Answer:
        print(f"[{items.index(item) + 1}/{len(items)}] {item.id}", file=sys.stderr, flush=True)
        a = answer(item.question, Session(), data_dir=args.data_dir)  # fresh session each
        if args.answers:
            print(f"\n=== {item.id}: {item.question}\nstatus: {a.status}")
            print(f"expected: {item.answer}\n--- got:\n{a.text.strip()}\n")
        return a

    report = evaluate(items, ask)
    print(format_report(report))
    if unreviewed := sum(not i.reviewed for i in items):
        print(f"\nDRAFT: {unreviewed}/{len(items)} eval items not yet reviewed by the user.")


def ui_main() -> None:
    ap = argparse.ArgumentParser(description="Open the chat in a local web page.")
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="index to use")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()

    from alienrag.ui import app

    data_dir = args.data_dir.resolve()
    app(data_dir=data_dir).launch(
        server_name="127.0.0.1",  # local only
        server_port=args.port,
        inbrowser=not args.no_browser,
        # Only what the chat shows: rendered pages and exported images.
        allowed_paths=[str(data_dir / "pages"), str(data_dir / "images")],
    )
