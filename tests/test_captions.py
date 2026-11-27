"""Image captions (#10): an optional index stage where a vision model describes each
procedure image and the description is added to its chunk's embed text.

Captioning runs on its own index (data/test-index-captions), seeded from the shared test
index's cached parses so it doesn't re-parse, and so toggling captions there doesn't
re-embed the index the other test files use. The first run captions every procedure
image (~10-15 min); later runs read the caption cache.
"""

import shutil
from pathlib import Path

import ollama
import pytest
from PIL import Image

from alienrag import config
from alienrag.indexing import index, load_all_chunks

ROOT = Path(__file__).resolve().parents[1]
MANUALS = ROOT / "Manuals"
TEST_DATA = ROOT / "data" / "test-index"
CAPTION_DATA = ROOT / "data" / "test-index-captions"
ALIASES = ROOT / "tests" / "aliases.json"

pytestmark = pytest.mark.skipif(
    not any(MANUALS.glob("*.pdf")), reason="no manual PDFs in Manuals/"
)


def _seed(data_dir: Path) -> None:
    """Give `data_dir` the shared test index's parses, so indexing only re-chunks."""
    index(MANUALS, TEST_DATA, aliases_file=ALIASES, log=lambda *_: None)
    for sub in ("parsed", "images"):
        if not (data_dir / sub).exists():
            shutil.copytree(TEST_DATA / sub, data_dir / sub)
    if not (data_dir / "manifest.json").exists():
        shutil.copy(TEST_DATA / "manifest.json", data_dir / "manifest.json")


@pytest.fixture(scope="module")
def captioned():
    _seed(CAPTION_DATA)
    index(MANUALS, CAPTION_DATA, captions=True, aliases_file=ALIASES, log=lambda *_: None)
    return load_all_chunks(CAPTION_DATA)


def _procedure_images(images: list[str], data_dir: Path) -> list[str]:
    """The images big enough to be procedure illustrations, not icons."""
    out = []
    for rel in images:
        with Image.open(data_dir / rel) as im:
            if max(im.size) >= config.CAPTION_MIN_PIXELS:
                out.append(rel)
    return out


def test_with_captions_each_procedure_image_adds_a_caption_to_its_chunks_embed_text(captioned):
    with_images = [c for c in captioned if _procedure_images(c.images, CAPTION_DATA)]
    assert with_images
    for c in captioned:
        n = len(_procedure_images(c.images, CAPTION_DATA))
        captions = [line for line in c.embed_text.splitlines() if line.startswith("Image: ")]
        assert len(captions) == n, (c.id, c.images, captions)
        assert all(len(line) > len("Image: ") + 10 for line in captions), captions
        assert "Image: " not in c.text  # the text shown to the generator is the manual's


def _recording_vision_calls(monkeypatch) -> list[str]:
    """Let model calls through but record the vision model's."""
    real_chat = ollama.chat
    calls: list[str] = []

    def recording_chat(*args, **kwargs):
        if kwargs.get("model") == config.CAPTION_MODEL:
            calls.append(str(kwargs["messages"][0].get("images")))
        return real_chat(*args, **kwargs)

    monkeypatch.setattr(ollama, "chat", recording_chat)
    return calls


def test_captions_are_off_by_default(monkeypatch):
    calls = _recording_vision_calls(monkeypatch)
    index(MANUALS, TEST_DATA, rechunk=True, aliases_file=ALIASES, log=lambda *_: None)
    assert calls == []
    assert not any("Image: " in c.embed_text for c in load_all_chunks(TEST_DATA))


def test_re_indexing_with_captions_reuses_cached_captions(captioned, monkeypatch):
    calls = _recording_vision_calls(monkeypatch)
    index(
        MANUALS, CAPTION_DATA, rechunk=True, captions=True, aliases_file=ALIASES,
        log=lambda *_: None,
    )  # fmt: skip
    assert calls == [], f"{len(calls)} images captioned again"
    again = {c.id: c.embed_text for c in load_all_chunks(CAPTION_DATA)}
    assert again == {c.id: c.embed_text for c in captioned}
