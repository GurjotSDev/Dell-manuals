"""Device resolution: which registry devices a question is about.

Deterministic alias matching first; an LLM call with structured JSON output only when
matching finds nothing or meets a name shared by several devices.
"""

import json
import re

import ollama

from alienrag import config
from alienrag.registry import Device, match_aliases


class UnknownDevice(ValueError):
    """A forced device has no manual in the index."""


RESOLVE_PROMPT = """Decide which of these devices the user's question is about.

Devices (id: name, aliases):
{devices}

Reply with JSON {{"devices": [ids]}} listing only devices the question clearly refers to,
by any name, spelling or nickname. If it names no device, or a name that fits several
devices without saying which, reply {{"devices": []}}."""

_RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {"devices": {"type": "array", "items": {"type": "string"}}},
    "required": ["devices"],
}


def _ask_llm(question: str, registry: list[Device]) -> set[str]:
    """Device ids the LLM says `question` refers to (lower-cased, unvalidated)."""
    listing = "\n".join(
        f"- {d.id}: {d.name}" + (f" (also: {', '.join(d.aliases)})" if d.aliases else "")
        for d in registry
    )
    resp = ollama.chat(
        model=config.GEN_MODEL,
        messages=[
            {"role": "system", "content": RESOLVE_PROMPT.format(devices=listing)},
            {"role": "user", "content": question},
        ],
        format=_RESOLVE_SCHEMA,
        options={"temperature": 0},
    )
    try:
        ids = json.loads(resp.message.content or "{}").get("devices", [])
    except json.JSONDecodeError:
        return set()
    return {i.lower() for i in ids if isinstance(i, str)}


def resolve(
    question: str, forced: list[str], registry: list[Device], inherited: list[str] | None = None
) -> list[Device]:
    """The devices `question` is about; empty when it can't tell.

    `forced` (model codes) overrides resolution. Devices named unambiguously are always
    kept; the LLM may only add to them. An LLM answer that still picks several devices
    for one shared name is a guess, so those devices are left out.

    `inherited` (model codes) is the device set of an ongoing conversation: a question
    that no alias matches goes straight to them without asking the LLM, which tends to
    guess at device-less follow-ups ("mine has liquid cooling" -> every device); so a new
    device named only in a form the aliases miss is not picked up mid-conversation. A
    shared name the LLM can't settle also falls back to them. A question that names
    devices replaces them ("what about the R16?"), unless it asks for a comparison
    ("compare that with the R16"): then the devices it names are added to them.
    """
    by_id = {d.id: d for d in registry}
    if forced:
        return _known(forced, by_id)
    match = match_aliases(question, registry)
    if match.devices and not match.ambiguous:
        return _extending(inherited, match.devices, question, by_id)
    if inherited and not match.ambiguous:
        return _known(inherited, by_id)
    wanted = _ask_llm(question, registry)
    for group in match.ambiguous:
        if sum(d.id in wanted for d in group) > 1:
            wanted -= {d.id for d in group}
    named = {d.id for d in match.devices}
    found = match.devices + [d for d in registry if d.id in wanted and d.id not in named]
    if found:
        return _extending(inherited, found, question, by_id)
    return _known(inherited or [], by_id)


# Words that ask to compare against what's already being discussed.
_COMPARING = re.compile(
    r"\b(compare[ds]?|comparing|comparison|versus|vs|against|differ\w*|both)\b", re.IGNORECASE
)


def _extending(
    inherited: list[str] | None, named: list[Device], question: str, by_id: dict[str, Device]
) -> list[Device]:
    """`named`, after the `inherited` devices when `question` asks for a comparison."""
    if not inherited or not _COMPARING.search(question):
        return named
    kept = _known(inherited, by_id)
    return kept + [d for d in named if d not in kept]


def _known(ids: list[str], by_id: dict[str, Device]) -> list[Device]:
    """The registry devices for model codes `ids`, in order; UnknownDevice if any is missing."""
    ids = list(dict.fromkeys(i.lower() for i in ids))
    if unknown := [i for i in ids if i not in by_id]:
        raise UnknownDevice(
            f"unknown device ({', '.join(unknown)}); known: {', '.join(sorted(by_id))}"
        )
    return [by_id[i] for i in ids]
