"""Device registry: which devices the manuals cover, and what people call them.

Built at index time from each manual's filename (model code) and title page (marketing
name), merged with the hand-editable alias file. Stored as data/devices.json.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pypdfium2 as pdfium


@dataclass
class Device:
    id: str  # model code, e.g. "act1250"
    name: str  # marketing name from the title page, e.g. "Alienware Aurora ACT1250"
    source: str  # manual file stem
    aliases: list[str] = field(default_factory=list)  # from the alias file


def model_code(source: str) -> str:
    """Model code of a manual, from its file stem: the token before "owners".

    "act1250-owners-manual-en-us" -> "act1250", "alienware-aurora-r16-owners-manual-en-us" -> "r16".
    """
    tokens = source.lower().split("-")
    i = tokens.index("owners") if "owners" in tokens else len(tokens)
    return tokens[i - 1] if i > 0 else source.lower()


def _title_page_name(pdf: Path) -> str:
    """Marketing name: the title page's lines before "Owner's Manual", joined.

    Falls back to the PDF's Title metadata, then the model code.
    """
    doc = pdfium.PdfDocument(str(pdf))
    try:
        text = doc[0].get_textpage().get_text_range() if len(doc) else ""
        meta_title = doc.get_metadata_dict().get("Title", "")
    finally:
        doc.close()
    for candidate in (text, meta_title):
        head, sep, _ = re.sub(r"\s+", " ", candidate).partition("Owner")
        if sep and head.strip():
            return head.strip()
    return model_code(pdf.stem).upper()


def _read_aliases(path: Path | None) -> dict[str, list[str]]:
    """Alias file: {"<model code>": ["alias", ...]}. Missing file means no aliases."""
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k.lower(): list(v) for k, v in raw.items()}


def registry_path(data_dir: Path) -> Path:
    return data_dir / "devices.json"


def build_registry(pdfs: list[Path], data_dir: Path, aliases_file: Path | None) -> list[Device]:
    """Derive one device per manual, merge aliases, and write the registry."""
    aliases = _read_aliases(aliases_file)
    devices: list[Device] = []
    for pdf in sorted(pdfs):
        code = model_code(pdf.stem)
        if clash := next((d for d in devices if d.id == code), None):
            raise ValueError(
                f"{pdf.name} and {clash.source}.pdf both have model code {code!r}; "
                "rename one so the token before 'owners' tells them apart"
            )
        devices.append(
            Device(code, _title_page_name(pdf), pdf.stem, aliases.get(code, []))
        )
    path = registry_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(d) for d in devices], indent=2), encoding="utf-8")
    return devices


def load_registry(data_dir: Path) -> list[Device]:
    path = registry_path(data_dir)
    if not path.exists():
        return []
    return [Device(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def _norm(s: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", s.lower())) + " "


@dataclass
class AliasMatch:
    devices: list[Device]  # devices named unambiguously, in order of first mention
    # Names shared by several devices that were used without any of those devices being
    # named on its own: each entry is the devices one such name could mean.
    ambiguous: list[list[Device]]


def match_aliases(question: str, devices: list[Device]) -> AliasMatch:
    """Devices named in `question` by model code, marketing name or alias (whole words,
    case- and punctuation-insensitive). A name shared by several devices only counts as
    ambiguous when none of its devices is also named by a name of its own."""
    q = _norm(question)
    owners: dict[str, list[Device]] = {}
    for d in devices:
        for term in {_norm(t) for t in [d.id, d.name, *d.aliases]}:
            if term.strip():
                owners.setdefault(term, []).append(d)
    first_mention: dict[str, int] = {}
    shared: list[list[Device]] = []
    for term, ds in owners.items():
        if (pos := q.find(term)) < 0:
            continue
        if len(ds) > 1:
            shared.append(ds)
        else:
            first_mention[ds[0].id] = min(first_mention.get(ds[0].id, pos), pos)
    ambiguous = [ds for ds in shared if not any(d.id in first_mention for d in ds)]
    by_id = {d.id: d for d in devices}
    return AliasMatch([by_id[i] for i in sorted(first_mention, key=first_mention.__getitem__)], ambiguous)
