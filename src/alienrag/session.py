"""Conversation session: what a chat remembers between turns, in memory only.

A session holds the active device set, the user's stated configuration and the last few
turns. Each message is read against that state before retrieval: follow-ups are
rewritten into standalone queries, and any configuration the user states is picked up.
Nothing is written to disk; dropping the session forgets it.
"""

import json
import re
from dataclasses import dataclass, field

import ollama

from alienrag import config
from alienrag.registry import Device, names_only


@dataclass
class Turn:
    question: str  # what the user typed
    answer: str


@dataclass
class Session:
    # Forced device set (model codes, e.g. "act1250"); empty means resolve from the question.
    devices: list[str] = field(default_factory=list)
    # Active device set: the devices the last answered turn was about. A question that
    # names no device inherits it.
    active: list[str] = field(default_factory=list)
    # The user's stated configuration as short facts, e.g. ["RTX 5080 GPU", "liquid cooling"].
    configuration: list[str] = field(default_factory=list)
    # The standalone query of the last question, when it was answered with "Which
    # device?"; empty otherwise.
    pending: str = ""
    # The last few turns, oldest first (at most config.SESSION_TURNS).
    turns: list[Turn] = field(default_factory=list)

    def remember(self, turn: Turn) -> None:
        self.turns = [*self.turns, turn][-config.SESSION_TURNS :]


@dataclass
class Reading:
    """What a message means in its session."""

    query: str  # standalone search query; empty when the message asks nothing
    configuration: list[str]  # the user's full configuration after this message


READ_PROMPT = """You read the user's latest message in a chat about Alienware computer manuals.

1. "query": rewrite the latest message into a standalone search query. Resolve every reference to earlier turns ("it", "that", "and the RAM?", "put it back") so it makes sense on its own. When the user's configuration decides which part or procedure applies (for example liquid cooling vs. air cooling), make the query specific to it. Keep it short, in the user's words where possible, and leave out device names. Do not answer it. If the message only states facts and asks nothing, use "". Exception: when the assistant's previous reply was a question asking which configuration the user has, a message answering it means the user's earlier question, so rewrite that earlier question for the stated configuration.

2. "configuration": the hardware configuration the user says their own machine has (GPU, CPU, cooling, power supply, memory, storage and similar), as short facts like "RTX 5080 GPU" or "liquid cooling". Start from the configuration already known, add what the latest message states, and replace a fact only when the user corrects it. Device model names are not configuration. Never guess.

Reply with JSON {"query": "...", "configuration": ["...", ...]}."""

_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "configuration": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["query", "configuration"],
}

_ANSWER_EXCERPT = 600  # characters of each earlier answer shown to the reader

# A message can only state the user's own configuration if it speaks in the first person.
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|me|my|mine|we|our|ours)\b", re.IGNORECASE)


def _history(turns: list[Turn]) -> str:
    return "\n\n".join(
        f"User: {t.question}\nAssistant: {t.answer[:_ANSWER_EXCERPT]}" for t in turns
    )


def read(question: str, session: Session, registry: list[Device]) -> Reading:
    """Read `question` against the session: its standalone query and the configuration.

    A first message that can't refer back to anything or state a configuration is taken
    as it is, with no model call; so is a reply that only names a `registry` device
    after "Which device?", which means the question that was waiting for it. A reply
    that also states facts means that question too.

    Device names are never configuration, whatever the reader makes of them: a fact
    that only names a device is dropped.
    """
    if session.pending and names_only(question, registry):
        return Reading(session.pending, session.configuration)
    if not session.turns and not session.configuration and not _FIRST_PERSON.search(question):
        return Reading(question, [])
    known = "\n".join(f"- {c}" for c in session.configuration) or "(none)"
    resp = ollama.chat(
        model=config.GEN_MODEL,
        messages=[
            {"role": "system", "content": READ_PROMPT},
            {
                "role": "user",
                "content": f"Conversation so far:\n\n{_history(session.turns) or '(none)'}\n\n"
                f"Configuration already known:\n{known}\n\nLatest message: {question}",
            },
        ],
        format=_READ_SCHEMA,
        options={"temperature": 0},
    )
    try:
        raw = json.loads(resp.message.content or "{}")
    except json.JSONDecodeError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    query = raw.get("query")
    facts = raw.get("configuration")
    if not isinstance(facts, list):
        facts = session.configuration
    facts = list(dict.fromkeys(f.strip() for f in facts if isinstance(f, str) and f.strip()))
    facts = [f for f in facts if not names_only(f, registry)]
    query = query.strip() if isinstance(query, str) else question
    # An empty query means the message only stated facts; one that states nothing new
    # is still a question, whatever the reader made of it.
    if not query and facts == session.configuration:
        query = question
    if not query and session.pending:
        query = session.pending
    return Reading(query, facts)
