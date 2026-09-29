"""EPUB loader based on ebooklib + BeautifulSoup; chapters come from the TOC."""

from __future__ import annotations

import posixpath
import warnings
from pathlib import Path
from urllib.parse import unquote

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

from ..model import Chapter, ChapterImage, Document
from ..textutil import clean_paragraph

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", module="ebooklib")

_BLOCKS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "dt", "dd", "img"]
_MAX_IMAGES = 6


def _flatten_toc(toc, depth: int = 0) -> list[tuple[int, str, str]]:
    """Return (depth, title, href) for every TOC entry."""
    out: list[tuple[int, str, str]] = []
    for entry in toc:
        if isinstance(entry, tuple):  # (Section, [children])
            section, children = entry
            href = getattr(section, "href", None)
            if href:
                out.append((depth, section.title, href))
            out.extend(_flatten_toc(children, depth + 1))
        elif isinstance(entry, epub.Link):
            out.append((depth, entry.title, entry.href))
    return out


def _chapter_starts(book: epub.EpubBook) -> dict[str, dict[str | None, str]]:
    """Map document href → {fragment or None: chapter title} for top-level TOC entries."""
    entries = _flatten_toc(book.toc)
    if not entries:
        return {}
    depths = sorted({d for d, _, _ in entries})
    level = depths[0]
    if sum(1 for d, *_ in entries if d == level) == 1 and len(depths) > 1:
        level = depths[1]
    starts: dict[str, dict[str | None, str]] = {}
    for depth, title, href in entries:
        if depth != level:
            continue
        file, _, frag = unquote(href).partition("#")
        starts.setdefault(posixpath.normpath(file), {})[frag or None] = title.strip()
    return starts


def _block_elements(soup: BeautifulSoup):
    body = soup.body or soup
    for el in body.find_all(_BLOCKS):
        # Skip blocks nested in other blocks (e.g. <p> inside <li>) to avoid duplicates.
        if el.name != "img" and el.find_parent(["p", "li", "blockquote", "pre", "dd"]):
            continue
        yield el


def _fragment_markers(soup: BeautifulSoup, doc_starts: dict[str | None, str]) -> dict[int, str]:
    """Map id() of the first block element at/after each TOC fragment → chapter title."""
    markers: dict[int, str] = {}
    for frag, chapter_title in doc_starts.items():
        if frag is None:
            continue
        target = soup.find(id=frag) or soup.find(attrs={"name": frag})
        if target is None:
            continue
        block = target if target.name in _BLOCKS else target.find(_BLOCKS)
        if block is None:
            block = target.find_next(_BLOCKS)
        while (
            block is not None
            and block.name != "img"
            and block.find_parent(["p", "li", "blockquote", "pre", "dd"])
        ):
            block = block.find_parent(["p", "li", "blockquote", "pre", "dd"])
        if block is not None:
            markers.setdefault(id(block), chapter_title)
    return markers


def _caption(img) -> str:
    """The enclosing <figure>'s <figcaption>, else the image's alt/title text."""
    figure = img.find_parent("figure")
    caption = figure.find("figcaption") if figure is not None else None
    if caption is not None and (text := clean_paragraph(caption.get_text(" "))):
        return text[:300]
    return clean_paragraph(img.get("alt") or img.get("title") or "")[:300]


def load_epub(path: Path) -> Document:
    book = epub.read_epub(str(path), options={"ignore_ncx": False})
    titles = book.get_metadata("DC", "title")
    creators = book.get_metadata("DC", "creator")
    title = titles[0][0] if titles else path.stem.replace("_", " ")
    author = creators[0][0] if creators else None
    starts = _chapter_starts(book)

    chapters: list[Chapter] = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        if isinstance(item, epub.EpubNav):
            continue
        name = posixpath.normpath(unquote(item.get_name()))
        soup = BeautifulSoup(item.get_content(), "lxml")
        doc_starts = starts.get(name, {})
        if None in doc_starts:
            chapters.append(Chapter(doc_starts[None], []))
        elif not starts:
            heading = soup.find(["h1", "h2"])
            if heading and heading.get_text(strip=True):
                chapters.append(Chapter(clean_paragraph(heading.get_text(" ")), []))
        base = posixpath.dirname(name)
        markers = _fragment_markers(soup, doc_starts)
        for el in _block_elements(soup):
            if (hit := markers.get(id(el))) is not None:
                chapters.append(Chapter(hit, []))
            if not chapters:
                chapters.append(Chapter(title, []))
            if el.name == "img":
                src = el.get("src")
                if src and len(chapters[-1].images) < _MAX_IMAGES:
                    img = book.get_item_with_href(posixpath.normpath(posixpath.join(base, src)))
                    if img is not None:
                        chapters[-1].images.append(ChapterImage(img.get_content(), _caption(el)))
                continue
            text = clean_paragraph(el.get_text(" "))
            if text:
                chapters[-1].paragraphs.append(text)
    return Document(path=path, title=title, author=author, chapters=chapters)
