"""The web UI: a Gradio chat that is a thin shell over `answer` and a session.

`reply` turns an Answer into what a chat turn shows: the answer with its sources, the
cited manual pages rendered from the PDFs, and the procedure images linked to the
cited chunks.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pypdfium2

from alienrag import config
from alienrag.answer import Answer, Citation, Session, answer
from alienrag.caption import is_procedure_image
from alienrag.registry import load_registry

PAGE_SCALE = 2  # render scale for cited pages: 144 dpi


@dataclass
class Page:
    citation: Citation
    path: Path  # the page rendered as PNG


@dataclass
class Reply:
    text: str  # markdown: the answer, then its sources
    pages: list[Page] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)  # procedure images, in manual order
    # Answers the user can pick instead of typing: the devices, when asked which one.
    options: list[str] = field(default_factory=list)


def _page_png(citation: Citation, data_dir: Path, manuals_dir: Path) -> Path | None:
    """`citation`'s page rendered as PNG, cached under data_dir/pages/; None when the
    citation names no manual or page there."""
    out = data_dir / "pages" / citation.source / f"{citation.page:04d}.png"
    if out.exists():
        return out
    pdf_path = manuals_dir / f"{citation.source}.pdf"
    if not pdf_path.exists():
        return None
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        if not 1 <= citation.page <= len(pdf):
            return None
        image = pdf[citation.page - 1].render(scale=PAGE_SCALE).to_pil()
    finally:
        pdf.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    image.save(tmp, format="PNG")
    tmp.replace(out)
    return out


def _sources(citations: list[Citation]) -> str:
    return "\n".join(f"- {c.source}.pdf, page {c.page}" for c in citations)


def _linked_images(result: Answer, data_dir: Path) -> list[Path]:
    """The procedure images of the chunks the answer cites: chunks it was given that are
    on a cited page. Note and warning icons are left out."""
    cited = {(c.source, c.page) for c in result.citations}
    out: list[Path] = []
    for h in sorted(result.chunks, key=lambda h: h.id):
        if not any((h.source, p) in cited for p in h.pages):
            continue
        for rel in h.images:
            path = data_dir / rel
            if path not in out and path.exists() and is_procedure_image(path):
                out.append(path)
    return out


def reply(
    result: Answer, *, data_dir: Path = config.DATA_DIR, manuals_dir: Path = config.MANUALS_DIR
) -> Reply:
    """What the chat shows for `result`."""
    text = result.text.strip()
    if result.citations:
        text += "\n\n**Sources**\n" + _sources(result.citations)
    pages = [
        Page(c, path)
        for c in result.citations
        if (path := _page_png(c, data_dir, manuals_dir)) is not None
    ]
    options = (
        [d.name for d in load_registry(data_dir)] if result.status == "needs-device" else []
    )
    return Reply(text, pages, _linked_images(result, data_dir), options)


def _file(path: Path, name: str) -> dict:
    return {"role": "assistant", "content": {"path": str(path), "orig_name": name}}


def messages(r: Reply) -> list[dict]:
    """`r` as chat messages: the text (with any options), then each cited page and image."""
    text: dict = {"role": "assistant", "content": r.text}
    if r.options:
        text["options"] = [{"value": o, "label": o} for o in r.options]
    return [
        text,
        *(_file(p.path, f"{p.citation.source}.pdf, page {p.citation.page}") for p in r.pages),
        *(_file(path, path.name) for path in r.images),
    ]


def respond(
    message: str,
    history: list[dict],
    session: Session,
    *,
    data_dir: Path = config.DATA_DIR,
    manuals_dir: Path = config.MANUALS_DIR,
) -> tuple[list[dict], Session]:
    """One chat turn: `history` with `message` and the reply to it, and the session."""
    from ollama import ResponseError

    history = [*history, {"role": "user", "content": message}]
    try:
        result = answer(message, session, data_dir=data_dir)
    except (ResponseError, ConnectionError) as e:  # Ollama hiccup: keep the session
        failed = {"role": "assistant", "content": f"Model call failed ({e}); try again."}
        return [*history, failed], session
    shown = messages(reply(result, data_dir=data_dir, manuals_dir=manuals_dir))
    return [*history, *shown], session


def app(*, data_dir: Path = config.DATA_DIR, manuals_dir: Path = config.MANUALS_DIR):
    """The Gradio chat page. Each browser tab gets its own session."""
    import gradio as gr

    def submit(message: str, history: list[dict], session: Session):
        message = message.strip()
        if not message:
            return "", history, session
        history, session = respond(
            message, history, session, data_dir=data_dir, manuals_dir=manuals_dir
        )
        return "", history, session

    def pick(event: gr.SelectData, history: list[dict], session: Session):
        _, history, session = submit(str(event.value["value"]), history, session)
        return history, session

    with gr.Blocks(title="Alienware manuals") as demo:
        gr.Markdown("## Ask about your Alienware computer")
        session = gr.State(Session())  # copied per browser tab
        chat = gr.Chatbot(height=640, buttons=["copy"])
        box = gr.Textbox(placeholder="How do I remove the memory module?", show_label=False)
        box.submit(submit, [box, chat, session], [box, chat, session])
        chat.option_select(pick, [chat, session], [chat, session])
        gr.ClearButton([chat, box]).click(lambda: Session(), None, session)
    return demo
