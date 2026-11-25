"""Device resolution: which registry devices a question is about.

Deterministic alias matching first; an LLM call with structured JSON output only when
matching finds nothing or meets a name shared by several devices.
"""

import json

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


def resolve(question: str, forced: list[str], registry: list[Device]) -> list[Device]:
    """The devices `question` is about; empty when it can't tell.

    `forced` (model codes) overrides resolution. Devices named unambiguously are always
    kept; the LLM may only add to them. An LLM answer that still picks several devices
    for one shared name is a guess, so those devices are left out.
    """
    by_id = {d.id: d for d in registry}
    if forced:
        ids = list(dict.fromkeys(f.lower() for f in forced))
        if unknown := [i for i in ids if i not in by_id]:
            raise UnknownDevice(
                f"unknown device ({', '.join(unknown)}); known: {', '.join(sorted(by_id))}"
            )
        return [by_id[i] for i in ids]
    match = match_aliases(question, registry)
    if match.devices and not match.ambiguous:
        return match.devices
    wanted = _ask_llm(question, registry)
    for group in match.ambiguous:
        if sum(d.id in wanted for d in group) > 1:
            wanted -= {d.id for d in group}
    named = {d.id for d in match.devices}
    return match.devices + [d for d in registry if d.id in wanted and d.id not in named]
