"""The document's own images: captured per chapter, saved, and summarised.

With illustrations on, every chapter image is saved to ``<stem>_video/book_images/``
and described by the local vision model (:mod:`pdf2video.llm.vision`). The
summaries are kept in ``book_images.json``, so re-runs only describe new images.
The script writer then compares them with the pictures it would generate and may
show the book's image instead (see :func:`pdf2video.video.script.choose_book_images`).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ValidationError

from ..media import Cancelled
from ..model import Document

log = logging.getLogger(__name__)

FOLDER = "book_images"
CATALOG = "book_images.json"
MIN_SIDE = 200  # smaller images are icons or ornaments
UNSUITABLE = ("text", "decorative")  # kinds never offered as illustrations

# "Figure 2.1", "Fig. 3", "figures 4.2", "Table 5", "Obr. 4", "Abb. 1-2"
_FIGURE_REF = re.compile(
    r"\b(fig(?:ure)?s?|obr(?:ázek|ázku|\.)?|abb(?:ildung|\.)?|tab(?:le|ulka|elle)?s?)\.?"
    r"\s*(\d+(?:[.\-–]\d+)*)",
    re.IGNORECASE,
)


def figure_refs(text: str) -> list[str]:
    """The figures and tables ``text`` refers to, normalised: ["figure 2.1", "table 3"]."""
    refs = []
    for match in _FIGURE_REF.finditer(text):
        kind = "table" if match[1].lower().startswith("tab") else "figure"
        refs.append(f"{kind} {re.sub(r'[\-–]', '.', match[2])}")
    return refs


class BookImage(BaseModel):
    file: str  # relative to the video folder, e.g. "book_images/03-1a2b3c4d.png"
    chapter: int
    sha256: str
    width: int
    height: int
    caption: str = ""  # from the document
    kind: str = ""  # from the vision model ("" = not described)
    description: str = ""

    @property
    def figure(self) -> str:
        """The figure this image is, from its caption ("figure 2.1"), or ""."""
        refs = figure_refs(self.caption[:40])
        return refs[0] if refs and _FIGURE_REF.match(self.caption.strip()) else ""

    @property
    def suitable(self) -> bool:
        """Worth offering: known content and not just text or an ornament."""
        return bool(self.description or self.caption) and self.kind not in UNSUITABLE

    def summary(self) -> str:
        text = f"{self.kind or 'image'}: {self.description or '(no description)'}"
        if self.caption:
            text += f' Caption in the book: "{self.caption}"'
        return text


def collect(doc: Document, out_dir: Path) -> list[BookImage]:
    """Save every chapter image (deduplicated, not too small) into ``out_dir/book_images``."""
    folder = out_dir / FOLDER
    images: list[BookImage] = []
    seen: set[tuple[int, str]] = set()
    for chapter in doc.chapters:
        for image in chapter.images:
            digest = hashlib.sha256(image.data).hexdigest()
            if (chapter.number, digest) in seen:
                continue
            seen.add((chapter.number, digest))
            try:
                with Image.open(io.BytesIO(image.data)) as img:
                    size, fmt = img.size, img.format
                    if min(size) < MIN_SIDE:
                        continue
                    suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(fmt or "", "")
                    name = f"{chapter.number:02d}-{digest[:8]}{suffix or '.png'}"
                    path = folder / name
                    if not path.exists():
                        folder.mkdir(parents=True, exist_ok=True)
                        if suffix:
                            path.write_bytes(image.data)
                        else:  # GIF, TIFF, … → PNG
                            img.convert("RGBA").save(path)
            except Exception as exc:
                log.debug("skipping an unreadable image in %r: %s", chapter.title, exc)
                continue
            images.append(
                BookImage(
                    file=f"{FOLDER}/{name}",
                    chapter=chapter.number,
                    sha256=digest,
                    width=size[0],
                    height=size[1],
                    caption=image.caption,
                )
            )
    return images


def _load_catalog(path: Path) -> dict[str, BookImage]:
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        return {e.sha256: e for e in (BookImage.model_validate(x) for x in entries)}
    except (OSError, ValueError, TypeError, ValidationError):
        return {}


def describe(
    images: list[BookImage],
    out_dir: Path,
    model: str,
    auto_download: bool,
    on_progress: Callable[[float, str], None] = lambda fraction, message: None,
    check_cancel: Callable[[], None] = lambda: None,
) -> None:
    """Fill in ``kind`` and ``description`` (from ``book_images.json`` or the vision model).

    Failures are not fatal: an image that can't be described keeps only its caption.
    """
    known = _load_catalog(out_dir / CATALOG)
    todo: list[BookImage] = []
    for image in images:
        if (old := known.get(image.sha256)) is not None and old.kind:
            image.kind, image.description = old.kind, old.description
        else:
            todo.append(image)
    if todo:
        _describe_new(todo, out_dir, model, auto_download, on_progress, check_cancel)
    save_catalog(images, out_dir)


def _describe_new(
    todo: list[BookImage],
    out_dir: Path,
    model: str,
    auto_download: bool,
    on_progress: Callable[[float, str], None],
    check_cancel: Callable[[], None],
) -> None:
    from ..llm import unload as unload_llm
    from ..llm import vision

    try:
        unload_llm()  # free its memory for the vision model
        on_progress(0.0, "Preparing the vision model…")
        llm = vision.VisionLLM(model, auto_download)
        llm.prepare(lambda f, m: on_progress(0.4 * f, m), check_cancel)
    except Cancelled:
        raise
    except Exception as exc:
        log.warning("Vision model unavailable (%s); book images keep only their captions", exc)
        on_progress(1.0, f"Vision model unavailable: {exc}")
        return
    try:
        for n, image in enumerate(todo, start=1):
            check_cancel()
            on_progress(0.4 + 0.6 * (n - 1) / len(todo), f"Describing book image {n}/{len(todo)}")
            try:
                described = llm.describe((out_dir / image.file).read_bytes(), image.caption)
            except Cancelled:
                raise
            except Exception as exc:
                log.warning("Could not describe %s: %s", image.file, exc)
                continue
            image.kind, image.description = described.kind, described.description
    finally:
        vision.unload()


def save_catalog(images: list[BookImage], out_dir: Path) -> None:
    if images:
        (out_dir / CATALOG).write_text(
            json.dumps([i.model_dump() for i in images], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def prepare_book_images(
    doc: Document,
    out_dir: Path,
    model: str,
    auto_download: bool,
    on_progress: Callable[[float, str], None] = lambda fraction, message: None,
    check_cancel: Callable[[], None] = lambda: None,
) -> dict[int, list[BookImage]]:
    """Capture and summarise the book's images; chapter number → its images."""
    images = collect(doc, out_dir)
    if not images:
        return {}
    describe(images, out_dir, model, auto_download, on_progress, check_cancel)
    return by_chapter(images)


def by_chapter(images: list[BookImage]) -> dict[int, list[BookImage]]:
    chapters: dict[int, list[BookImage]] = {}
    for image in images:
        chapters.setdefault(image.chapter, []).append(image)
    return chapters
