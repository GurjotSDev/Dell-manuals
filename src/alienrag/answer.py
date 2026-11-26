"""The answer seam: question + session -> cited Answer."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ollama

from alienrag import config, store
from alienrag.registry import Device, load_registry
from alienrag.resolve import UnknownDevice, resolve
from alienrag.session import Session, Turn, read

__all__ = ["Answer", "Citation", "Session", "UnknownDevice", "answer"]

Status = Literal["answered", "needs-device"]  # later: refused, needs-config-clarification


@dataclass
class Citation:
    source: str  # manual file stem
    page: int


@dataclass
class Answer:
    devices: list[Device]
    status: Status
    chunks: list[store.Hit]
    text: str
    citations: list[Citation]
    # The standalone query retrieval ran on: the question itself, or a follow-up
    # rewritten from the session's history.
    query: str = ""


SYSTEM_PROMPT = """You answer questions about Alienware computers using ONLY the manual passages provided. Never use outside knowledge.

Each passage starts with a header like [ACT1250 p.54-55] giving the device and its pages.
Rules:
- Cite every claim with the device and a single page number in square brackets, exactly like [ACT1250 p.54]. Cite only devices and pages that appear in passage headers.
- For a removal or installation procedure, give the prerequisites, then the steps as a numbered list in the manual's order, and include every WARNING and CAUTION attached to it.
- Refer to each device by its full name as given in the device list, never by its label alone.
- If the user's configuration is given and the manual differs by configuration, answer for the user's configuration.
- If the passages do not contain the answer, say it is not found in the manuals."""


def _label(product: str) -> str:
    return product.upper()


def _page_span(pages: list[int]) -> str:
    return f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"


def _context(hits: list[store.Hit]) -> str:
    return "\n\n".join(
        f"[{_label(h.product)} {_page_span(h.pages)}] {' > '.join(h.headings)}\n{h.text}"
        for h in hits
    )


# [ACT1250 p.54], [ACT1250 p. 54-55], [ACT1250 p.54, p.55], [ACT1250 pp. 54, 55]
_CITE = re.compile(r"\[\s*([A-Za-z0-9]+)\s+((?:pp?\.?\s*\d+(?:\s*[-–,]\s*(?:p\.?\s*)?\d+)*))\s*\]")


def _citations(text: str, hits: list[store.Hit]) -> list[Citation]:
    """Citations in `text`, in order, de-duplicated. Labels map to manuals via the hits;
    an unknown label is kept verbatim as the source so bad citations stay visible."""
    sources = {_label(h.product): h.source for h in hits}
    out: list[Citation] = []
    for m in _CITE.finditer(text):
        source = sources.get(m.group(1).upper(), m.group(1))
        nums: list[int] = []
        for lo, hi in re.findall(r"(\d+)(?:\s*[-–]\s*(\d+))?", m.group(2)):
            nums.extend(range(int(lo), int(hi or lo) + 1))
        for n in nums:
            c = Citation(source, n)
            if c not in out:
                out.append(c)
    return out


def _device_list(devices: list[Device]) -> str:
    return "\n".join(f"- {d.name} (label {_label(d.id)})" for d in devices)


def _generate(
    question: str, devices: list[Device], hits: list[store.Hit], configuration: list[str]
) -> str:
    setup = (
        "User's configuration:\n" + "\n".join(f"- {c}" for c in configuration) + "\n\n"
        if configuration
        else ""
    )
    resp = ollama.chat(
        model=config.GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Devices:\n{_device_list(devices)}\n\n{setup}Manual passages:\n\n{_context(hits)}\n\nQuestion: {question}",
            },
        ],
        options={"num_ctx": config.GEN_NUM_CTX, "temperature": 0},
    )
    return resp.message.content or ""


def _which_device(registry: list[Device]) -> str:
    names = "\n".join(f"- {d.name} ({d.id})" for d in registry)
    return f"Which device is your question about? I have manuals for:\n{names}"


def _noted(configuration: list[str], devices: list[Device]) -> str:
    facts = ", ".join(configuration) or "nothing about your configuration yet"
    machine = f" {' / '.join(d.name for d in devices)}" if devices else ""
    return f"Noted for your{machine}: {facts}. I'll use this for the rest of the session."


def answer(question: str, session: Session, *, data_dir: Path = config.DATA_DIR) -> Answer:
    """Answer `question` in `session`, updating the session's devices, configuration and turns."""
    registry = load_registry(data_dir)
    if not registry:
        raise RuntimeError(f"no device registry in {data_dir}; run `index` first")
    reading = read(question, session)
    session.configuration = reading.configuration
    query = reading.query
    devices = resolve(question, session.devices, registry, inherited=session.active)
    if not query:  # the message only stated facts about the user's machine
        if devices:
            session.active = [d.id for d in devices]
        result = Answer(
            devices=devices,
            status="answered",
            chunks=[],
            text=_noted(session.configuration, devices),
            citations=[],
            query=query,
        )
    elif not devices:
        result = Answer(
            devices=[],
            status="needs-device",
            chunks=[],
            text=_which_device(registry),
            citations=[],
            query=query,
        )
    else:
        hits = store.search(data_dir, query, [d.id for d in devices], k=config.RETRIEVE_K)
        text = _generate(query, devices, hits, session.configuration)
        session.active = [d.id for d in devices]
        result = Answer(
            devices=devices,
            status="answered",
            chunks=hits,
            text=text,
            citations=_citations(text, hits),
            query=query,
        )
    session.remember(Turn(question, query, result.text))
    return result
