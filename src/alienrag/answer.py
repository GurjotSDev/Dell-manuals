"""The answer seam: question + session -> cited Answer."""

import re
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Literal

import ollama

from alienrag import config, store
from alienrag.registry import Device, load_registry
from alienrag.resolve import UnknownDevice, resolve
from alienrag.session import Session, Turn, read

__all__ = ["Answer", "Citation", "Session", "UnknownDevice", "answer"]

# needs-config-clarification: the answer differs by configuration and the variants are
# too long to present together, so `text` asks which configuration the user has.
Status = Literal["answered", "needs-device", "refused", "needs-config-clarification"]


@dataclass
class Citation:
    source: str  # manual file stem
    page: int


@dataclass
class Answer:
    devices: list[Device]
    status: Status
    # What the generator saw: the retrieved chunks in rank order, then the chunks that
    # complete their sections.
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
- For a removal or installation procedure, give the prerequisites, then the steps as a numbered list in the manual's order. Quote every WARNING and CAUTION of the procedure word for word, including any that come before its prerequisites, even when the user didn't ask about safety.
- Refer to each device by its full name as given in the device list, never by its label alone.
- For a specification question, list every option or configuration the passages document (for example every graphics card in a table, with its values), never a single guessed value.
- When the manual gives different procedures or values for different configurations (for example air vs. liquid cooling, 120 mm vs. 240 mm radiator, 500 W vs. 1000 W power supply):
  - If the user's configuration is given, answer for the user's configuration only.
  - Otherwise present each variant under its own label naming the configuration, each complete with its own steps and warnings.
  - Only for a removal or installation procedure: if it has more than two configuration variants, or presenting them all would be very long, do not answer. Reply with one line only: CLARIFY: followed by a short question asking which configuration the user has, naming the options. Never do this for a specification or other factual question: list the value for every configuration instead.
- If the passages answer no part of the question, reply with exactly "Not found in the manuals." and nothing else. If they answer only part of it (for example one device of a comparison), answer that part and say which part the manuals don't cover."""

# Added to SYSTEM_PROMPT when the question is about several devices.
COMPARISON_RULE = """
- The question is about several devices, and the passages are grouped under a === header per device. Answer under a heading for each device, in the device list's order: a line with only the device's full name in bold, like **Alienware Aurora R16**. Under each heading use only that device's passages, and cite every claim and step with that device's label and page, like [R16 p.61]. If a device's passages don't cover the topic, say under its heading that its manual doesn't cover it. After the per-device sections you may add one short line comparing them, with no citations."""


def _label(product: str) -> str:
    return product.upper()


def _page_span(pages: list[int]) -> str:
    return f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"


_SectionKey = tuple[str, tuple[str, ...]]  # (manual, heading path)


def _key(h: store.Hit) -> _SectionKey:
    return (h.source, tuple(h.headings))


def _in_manual_order(hits: list[store.Hit]) -> list[store.Hit]:
    """Sections in the order their first chunk ranks; each section's chunks in manual order."""
    rank: dict[_SectionKey, int] = {}
    for h in hits:
        rank.setdefault(_key(h), len(rank))
    return sorted(hits, key=lambda h: (rank[_key(h)], h.id))


def _passages(hits: list[store.Hit]) -> str:
    return "\n\n".join(
        f"[{_label(h.product)} {_page_span(h.pages)}] {' > '.join(h.headings)}\n{h.text}"
        for h in _in_manual_order(hits)
    )


def _context(hits: list[store.Hit], devices: list[Device]) -> str:
    """The passages, grouped under a header per device when there are several."""
    if len(devices) == 1:
        return _passages(hits)
    return "\n\n".join(
        f"=== {d.name} (label {_label(d.id)}) ===\n\n"
        + (_passages([h for h in hits if h.product == d.id]) or "(no passages)")
        for d in devices
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


# The generator's signal that variants are too long to present: a reply that is only
# "CLARIFY: <question>". A marker after an answer is ignored, keeping the answer.
_CLARIFY = re.compile(r"\A\W*CLARIFY\W*\s*(.+)", re.DOTALL)

# A WARNING/CAUTION line, after any list marker or bold markup: "- CAUTION: ...".
_SAFETY = re.compile(r"^[\s\-*•\d.]*\**(WARNING|CAUTION):\**\s*(.+)$", re.MULTILINE)
_STEPS = re.compile(r"^\s*Steps\s*$", re.MULTILINE)


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _step_lines(text: str) -> list[str]:
    """The numbered lines after a "Steps" label, as their first few words."""
    m = _STEPS.search(text)
    if not m:
        return []
    lines = re.findall(r"^\s*\d+\.\s+(.+)$", text[m.end() :].split("Next steps")[0], re.MULTILINE)
    return [k for line in lines if (k := " ".join(_words(line)[:5]))]


def _procedures_given(
    text: str, hits: list[store.Hit], citations: list[Citation]
) -> list[_SectionKey]:
    """The procedure sections `text` gives, in manual order.

    A procedure is given when at least half of its steps appear in the answer.
    Neighbouring procedures share pages (a removal ends where the installation begins),
    so pages are only the fallback for an answer that gives numbered steps but
    paraphrases them all: then a procedure counts through a cited page no other
    procedure is on. An answer with no numbered steps gives no procedure.
    """
    sections: dict[_SectionKey, list[store.Hit]] = {}
    for h in _in_manual_order(hits):
        sections.setdefault(_key(h), []).append(h)
    # Joined, so steps that run on into the section's next chunk are counted too.
    steps = {k: _step_lines("\n".join(h.text for h in hs)) for k, hs in sections.items()}
    procedures = [k for k, hs in sections.items() if any(_STEPS.search(h.text) for h in hs)]
    said = " ".join(_words(text))
    given = [
        k for k in procedures if steps[k] and 2 * sum(s in said for s in steps[k]) >= len(steps[k])
    ]
    if not given and len(re.findall(r"^\s*\d+\.\s", text, re.MULTILINE)) >= 2:
        pages = {k: {(k[0], p) for h in sections[k] for p in h.pages} for k in procedures}
        cited = {(c.source, c.page) for c in citations}
        for k in procedures:
            own = pages[k] - set().union(*(v for o, v in pages.items() if o != k))
            if cited & own:
                given.append(k)
    return sorted(given, key=lambda k: min(h.id for h in sections[k]))


def _with_safety_notes(text: str, hits: list[store.Hit], citations: list[Citation]) -> str:
    """`text` plus every WARNING/CAUTION of the procedures it gives that it leaves out."""
    said = " ".join(_words(text))
    missing: list[str] = []
    noted: set[str] = set()  # by wording: variants of a procedure repeat the same notes
    for k in _procedures_given(text, hits, citations):
        for h in sorted((h for h in hits if _key(h) == k), key=lambda h: h.id):
            for m in _SAFETY.finditer(h.text):
                body = m.group(2).strip()
                gist = " ".join(_words(body)[:8])
                if gist in said or gist in noted:
                    continue
                noted.add(gist)
                missing.append(
                    # The chunk's whole span: which of its pages holds the line is unknown.
                    f"- {m.group(1)}: {body} [{_label(h.product)} {_page_span(h.pages)}]"
                )
    if not missing:
        return text
    return text.rstrip() + "\n\nSafety notes from the manual:\n" + "\n".join(missing)


def _device_list(devices: list[Device]) -> str:
    return "\n".join(f"- {d.name} (label {_label(d.id)})" for d in devices)


def _generate(
    question: str,
    query: str,
    devices: list[Device],
    hits: list[store.Hit],
    configuration: list[str],
) -> str:
    """The answer to `query` (the standalone form of what the user typed, `question`)."""
    setup = (
        "User's configuration (answer for this configuration only; leave out the variants "
        "for other configurations):\n" + "\n".join(f"- {c}" for c in configuration) + "\n\n"
        if configuration
        else ""
    )
    asked = query if query == question else f"{query}\n(The user's words: {question})"
    resp = ollama.chat(
        model=config.GEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + (COMPARISON_RULE if len(devices) > 1 else ""),
            },
            {
                "role": "user",
                "content": f"Devices:\n{_device_list(devices)}\n\n{setup}Manual passages:\n\n{_context(hits, devices)}\n\nQuestion: {asked}",
            },
        ],
        options={"num_ctx": config.GEN_NUM_CTX, "temperature": 0},
    )
    return resp.message.content or ""


NOT_FOUND = "Not found in the manuals."
# Below the retrieval threshold nothing matched at all: the question may be off-topic or
# too vague to search for, so suggest rephrasing.
NOT_FOUND_OR_UNCLEAR = (
    "Not found in the manuals, or the question wasn't clear enough to search for. "
    "Try asking it as a full question."
)


def _refuses(text: str, devices: list[Device]) -> bool:
    """The generator declined: its whole reply is the refusal the prompt asks for, under
    any device headings (a comparison none of whose manuals covers the topic). A reply
    that says more (a partial answer naming what isn't covered) is an answer."""
    headings = {d.name.lower() for d in devices}
    lines = [s for line in text.splitlines() if (s := line.strip(" \t#*_\"'.:").lower())]
    lines = [s for s in lines if s not in headings]
    return bool(lines) and all(s == NOT_FOUND.lower().rstrip(".") for s in lines)


def _which_device(registry: list[Device]) -> str:
    names = "\n".join(f"- {d.name} ({d.id})" for d in registry)
    return f"Which device is your question about? I have manuals for:\n{names}"


def _noted(configuration: list[str], devices: list[Device]) -> str:
    facts = ", ".join(configuration) or "nothing about your configuration yet"
    machine = " / ".join(d.name for d in devices) if devices else "machine"
    return f"Noted for your {machine}: {facts}. I'll use this for the rest of the session."


def _without_device_names(query: str, registry: list[Device]) -> str:
    """`query` with every device's model code, name and aliases taken out (whole words,
    longest first, case-insensitive)."""
    terms = sorted({t for d in registry for t in [d.id, d.name, *d.aliases] if t.strip()}, key=len)
    for t in reversed(terms):
        query = re.sub(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])", "", query, flags=re.I)
    return re.sub(r"\s+", " ", query).strip()


def _retrieve(
    data_dir: Path, query: str, devices: list[Device], registry: list[Device]
) -> tuple[list[store.Hit], list[store.Hit]]:
    """The ranked hits for `query`, and those hits followed by the chunks completing
    their sections.

    A comparison searches each device on its own (top-k, product-filtered) so one manual
    can't crowd out another, and interleaves the rankings: every device's best chunk
    comes before any device's second best. The context budget is shared between the
    devices.

    Each device is searched twice, with and without the device names, and the two
    rankings are interleaved. The names pull up each manual's spec chunks, which are
    headed with the device's name: usually right, but they can crowd out an answering
    passage elsewhere (the R16's graphics card length is on p.61, outside the specs).
    See eval/baseline.md.
    """
    if len(devices) == 1:
        ranked = store.search(data_dir, query, [devices[0].id], k=config.RETRIEVE_K)
        hits = store.complete_sections(
            data_dir,
            ranked,
            max_section=config.SECTION_MAX_CHUNKS,
            max_total=config.CONTEXT_MAX_CHUNKS,
        )
        return ranked, hits
    budget = max(config.RETRIEVE_K, config.CONTEXT_MAX_CHUNKS_COMPARISON // len(devices))
    nameless = _without_device_names(query, registry)
    rankings: list[list[store.Hit]] = []
    added: list[store.Hit] = []
    for d in devices:
        queries = [q for q in (query, nameless) if q]
        both = [store.search(data_dir, q, [d.id], k=config.RETRIEVE_K) for q in queries]
        ranked = _interleave(both)[: config.RETRIEVE_K]
        hits = store.complete_sections(
            data_dir, ranked, max_section=config.SECTION_MAX_CHUNKS, max_total=budget
        )
        rankings.append(ranked)
        added += hits[len(ranked) :]
    ranked = _interleave(rankings)
    return ranked, ranked + added


def _interleave(rankings: list[list[store.Hit]]) -> list[store.Hit]:
    """Every ranking's first hit, then every ranking's second, ...; each chunk once."""
    out: list[store.Hit] = []
    seen: set[str] = set()
    for rank in zip_longest(*rankings):
        for h in rank:
            if h is not None and h.id not in seen:
                seen.add(h.id)
                out.append(h)
    return out


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
        ranked, hits = _retrieve(data_dir, query, devices, registry)
        session.active = [d.id for d in devices]
        status: Status = "answered"
        # The threshold judges what search ranked; chunks added to complete a section
        # carry no similarity of their own.
        if max((h.similarity for h in ranked), default=0.0) < config.REFUSE_BELOW:
            text, status = NOT_FOUND_OR_UNCLEAR, "refused"  # refused without generating
        else:
            text = _generate(question, query, devices, hits, session.configuration)
            if _refuses(text, devices):
                text, status = NOT_FOUND, "refused"  # drop the model's formatting around it
            elif m := _CLARIFY.search(text):
                text, status = _CITE.sub("", m.group(1)).strip(), "needs-config-clarification"
            else:
                text = _with_safety_notes(text, hits, _citations(text, hits))
        result = Answer(
            devices=devices,
            status=status,
            chunks=hits,
            text=text,
            citations=_citations(text, hits) if status == "answered" else [],
            query=query,
        )
    session.remember(Turn(question, result.text))
    return result
