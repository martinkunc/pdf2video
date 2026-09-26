"""Document loaders: every format is turned into a :class:`~pdf2video.model.Document`."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..model import Chapter, Document

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".epub", ".txt", ".md", ".markdown", ".text")


class ParseError(Exception):
    """Raised when a document cannot be read or has no usable text."""


@dataclass
class ParseOptions:
    ocr: str = "auto"  # "auto" (scanned pages only) | "always" | "off"
    ocr_languages: str = "auto"  # Tesseract codes such as "eng+ces", or "auto"
    on_progress: Callable[[float, str], None] | None = None

    def progress(self, fraction: float, message: str) -> None:
        if self.on_progress is not None:
            self.on_progress(fraction, message)


def _loader(suffix: str, options: ParseOptions) -> Callable[[Path], Document]:
    match suffix:
        case ".pdf":
            from .pdf import load_pdf

            return lambda path: load_pdf(path, options)
        case ".docx":
            from .docx import load_docx

            return load_docx
        case ".epub":
            from .epub import load_epub

            return load_epub
        case ".txt" | ".md" | ".markdown" | ".text":
            from .text import load_text

            return load_text
    raise ParseError(
        f"Unsupported file type “{suffix or '(none)'}”. "
        f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def load_document(path: str | Path, options: ParseOptions | None = None) -> Document:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ParseError(f"File not found: {path}")
    loader = _loader(path.suffix.lower(), options or ParseOptions())
    try:
        doc = loader(path)
    except ParseError:
        raise
    except Exception as exc:  # library-specific errors → user-facing message
        raise ParseError(f"Could not read {path.name}: {exc}") from exc
    doc = postprocess(doc)
    if not doc.chapters:
        raise ParseError(
            f"No readable text found in {path.name}. "
            "(If it is a scanned document, make sure OCR is not turned off.)"
        )
    return doc


_CONTENTS_TITLE = re.compile(
    r"^\s*(table of )?contents|obsah|inhalt(sverzeichnis)?|table des matières|índice|indice\s*$",
    re.IGNORECASE,
)
_TOC_LINE = re.compile(r"(\.{3,}|\s)\d{1,4}\s*$")


def _is_table_of_contents(chapter: Chapter) -> bool:
    if _CONTENTS_TITLE.match(chapter.title):
        return True
    paragraphs = chapter.paragraphs
    if len(paragraphs) < 5:
        return False
    toc_like = sum(1 for p in paragraphs if len(p) < 100 and _TOC_LINE.search(p))
    return toc_like / len(paragraphs) > 0.6


def _squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).casefold()


_PG_START = re.compile(r"^\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)
_PG_END = re.compile(r"^\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)
_PG_TITLE = re.compile(r"project gutenberg", re.IGNORECASE)


def _strip_gutenberg(chapters: list[Chapter]) -> list[Chapter]:
    """Remove Project Gutenberg header/licence boilerplate around the actual book."""
    flat = [(ci, pi) for ci, ch in enumerate(chapters) for pi, p in enumerate(ch.paragraphs)]
    start = next(
        (k for k, (ci, pi) in enumerate(flat) if _PG_START.match(chapters[ci].paragraphs[pi])), None
    )
    end = next(
        (k for k, (ci, pi) in enumerate(flat) if _PG_END.match(chapters[ci].paragraphs[pi])), None
    )
    if start is None and end is None:
        return [ch for ch in chapters if not _PG_TITLE.search(ch.title)]
    keep = {flat[k] for k in range((start or -1) + 1, end if end is not None else len(flat))}
    result = []
    for ci, ch in enumerate(chapters):
        paragraphs = [p for pi, p in enumerate(ch.paragraphs) if (ci, pi) in keep]
        if paragraphs and not _PG_TITLE.search(ch.title):
            result.append(Chapter(ch.title, paragraphs, ch.images))
    return result


def postprocess(doc: Document) -> Document:
    """Drop boilerplate, empty and table-of-contents chapters; strip repeated headings."""
    chapters: list[Chapter] = []
    for chapter in _strip_gutenberg(doc.chapters):
        paragraphs = [p.strip() for p in chapter.paragraphs if p and p.strip()]
        title = " ".join(chapter.title.split()) or f"Chapter {len(chapters) + 1}"
        # The heading itself (or its parts, e.g. "CHAPTER I." / "Down the Rabbit-Hole")
        # is often repeated at the start of the text.
        key = _squash(title)
        while paragraphs and len(paragraphs[0]) < 120 and _squash(paragraphs[0]) in key:
            paragraphs.pop(0)
        chapter = Chapter(title=title, paragraphs=paragraphs, images=chapter.images)
        if not paragraphs or _is_table_of_contents(chapter):
            continue
        chapters.append(chapter)
    doc.title = " ".join(doc.title.split()) or doc.path.stem
    doc.chapters = _merge_tiny(chapters, doc)
    for i, chapter in enumerate(doc.chapters, start=1):  # final numbering
        chapter.number = i
    doc.chapter_count = len(doc.chapters)
    return doc


TINY_WORDS = 60


def _merge_tiny(chapters: list[Chapter], doc: Document) -> list[Chapter]:
    """Collapse leading tiny chapters (title page, dedication, …) into one front-matter chapter.

    Only applies when the document also has substantial chapters.
    """
    lead = 0
    while lead < len(chapters) and chapters[lead].word_count < TINY_WORDS:
        lead += 1
    if lead >= 2 and lead < len(chapters):
        first = chapters[0]
        for chapter in chapters[1:lead]:
            first.paragraphs.extend(chapter.paragraphs)
            first.images.extend(chapter.images)
        chapters = [first, *chapters[lead:]]
    for chapter in chapters:
        if chapter.title == doc.path.stem and doc.title != doc.path.stem:
            chapter.title = doc.title
    return chapters
