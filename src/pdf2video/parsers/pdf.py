"""PDF loader based on PyMuPDF.

Chapters come from the PDF outline when available; otherwise headings are
detected from font sizes. Pages without a text layer (scans) are OCR'd with
Tesseract through PyMuPDF.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..model import Chapter, ChapterImage, Document
from ..textutil import (
    CAPTION_LINE,
    CHAPTER_LINE,
    detect_language,
    lines_to_paragraphs,
    looks_like_heading,
    normalize,
)
from . import ParseError, ParseOptions

log = logging.getLogger(__name__)

# MuPDF prints diagnostics about malformed PDFs (e.g. "syntax error: invalid key in
# dict") straight to stderr while it repairs them. Collect them instead and report
# them through logging (see _report_mupdf_messages).
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)

MAX_IMAGES_PER_CHAPTER = 10
MIN_IMAGE_SIDE = 200
CAPTION_GAP = 40  # points between an image and its caption above/below
SIDE_CAPTION_GAP = 250  # points between an image and its caption in the margin
FIGURE_GAP = 25  # drawings closer than this (points) belong to the same figure
MIN_FIGURE_SIDE = 60  # points
LABEL_GAP = 15  # a short text this close to a drawing is one of its labels (points)
LABEL_CHARS = 60
FIGURE_PIXELS = 1200  # longest side of a cropped vector figure
OCR_DPI = 300
MIN_TEXT_CHARS = 25  # a page with less extractable text than this is treated as a scan
TESSERACT_LANGS = {
    "en": "eng", "cs": "ces", "de": "deu", "fr": "fra", "es": "spa",
    "it": "ita", "pt": "por", "nl": "nld", "pl": "pol",
}  # fmt: skip


@dataclass
class Line:
    page: int
    text: str
    size: float
    bold: bool
    block_start: bool  # first line of a text block (paragraph hint)


def _page_lines(page: pymupdf.Page, page_no: int, textpage=None) -> list[Line]:
    lines: list[Line] = []
    data = page.get_text("dict", flags=pymupdf.TEXT_DEHYPHENATE, textpage=textpage)
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        first = True
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            size = max(s["size"] for s in spans)
            bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
            lines.append(Line(page_no, normalize(text), round(size, 1), bold, first))
            first = False
    return lines


_DIGITS = re.compile(r"\d+")


def _remove_running_headers(pages: list[list[Line]]) -> list[list[Line]]:
    """Drop lines that repeat at the top/bottom of many pages (headers, footers)."""
    if len(pages) < 4:
        return pages
    counter: Counter[str] = Counter()
    for lines in pages:
        edge = {_DIGITS.sub("#", ln.text) for ln in lines[:2] + lines[-2:]}
        counter.update(edge)
    threshold = max(3, len(pages) * 0.3)
    repeated = {text for text, n in counter.items() if n >= threshold}
    cleaned = []
    for lines in pages:
        edge_ids = {id(ln) for ln in lines[:2] + lines[-2:]}
        cleaned.append(
            [
                ln
                for ln in lines
                if not (id(ln) in edge_ids and _DIGITS.sub("#", ln.text) in repeated)
            ]
        )
    return cleaned


def _to_paragraphs(lines: list[Line]) -> list[str]:
    raw: list[str] = []
    for ln in lines:
        if ln.block_start and raw:
            raw.append("")
        raw.append(ln.text)
    return lines_to_paragraphs(raw)


def _body_size(lines: list[Line]) -> float:
    weights: Counter[float] = Counter()
    for ln in lines:
        weights[ln.size] += len(ln.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _norm_title(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).casefold()


def _chapters_from_toc(
    toc: list[list], flat: list[Line], page_count: int
) -> list[tuple[str, int]] | None:
    """Map outline entries to (title, index into ``flat``) chapter starts."""
    entries = [(lvl, title.strip(), page - 1) for lvl, title, page, *_ in toc if page >= 1]
    if not entries:
        return None
    level_counts = Counter(lvl for lvl, _, _ in entries)
    level = min(level_counts)
    # A single top-level entry is usually the book title; use the next level.
    if level_counts[level] == 1 and level + 1 in level_counts:
        level += 1
    chosen = [(t, p) for lvl, t, p in entries if lvl == level and 0 <= p < page_count]
    if len(chosen) < 1:
        return None

    page_first_index: dict[int, int] = {}
    for i, ln in enumerate(flat):
        page_first_index.setdefault(ln.page, i)

    starts: list[tuple[str, int]] = []
    for title, page in chosen:
        # Find the heading line on its page so that a chapter starting mid-page is cut there.
        idx = next(
            (page_first_index[p] for p in range(page, page_count) if p in page_first_index),
            len(flat),
        )
        target = _norm_title(title)
        j = idx
        while j < len(flat) and flat[j].page <= page:
            candidate = _norm_title(flat[j].text)
            if candidate and (target.startswith(candidate) or candidate.startswith(target)):
                idx = j
                break
            j += 1
        starts.append((title, idx))
    # Guarantee monotonically increasing positions.
    starts.sort(key=lambda s: s[1])
    return starts


def _chapters_from_fonts(flat: list[Line]) -> list[tuple[str, int]] | None:
    body = _body_size(flat)
    candidates = [
        i
        for i, ln in enumerate(flat)
        if (ln.size >= body * 1.25 and looks_like_heading(ln.text) and len(ln.text) > 1)
        or (CHAPTER_LINE.match(ln.text) and (ln.bold or ln.size > body) and ln.block_start)
    ]
    if not candidates:
        return None
    sizes = Counter(round(flat[i].size) for i in candidates)
    # Take the largest heading size that occurs at least twice (top-level headings).
    threshold = next((s for s in sorted(sizes, reverse=True) if sizes[s] >= 2), min(sizes))
    heads = [
        i
        for i in candidates
        if round(flat[i].size) >= threshold or CHAPTER_LINE.match(flat[i].text)
    ]
    head_set = set(heads)
    starts: list[tuple[str, int]] = []
    skip_until = -1
    for i in heads:
        if i <= skip_until:
            continue
        # Merge multi-line headings ("Chapter 1" / "The Beginning").
        title_parts = [flat[i].text]
        j = i + 1
        while j < len(flat) and flat[j].page == flat[i].page and len(title_parts) < 3:
            nxt = flat[j]
            heading_styled = nxt.size >= body * 1.15 or (nxt.bold and nxt.size >= body)
            if not (j in head_set or (heading_styled and looks_like_heading(nxt.text))):
                break
            title_parts.append(nxt.text)
            j += 1
        skip_until = j - 1
        separator = " " if title_parts[0].endswith((".", ":")) else " – "
        starts.append((separator.join(title_parts), i))
    return starts


Block = tuple[pymupdf.Rect, str]


def _text_blocks(page: pymupdf.Page) -> list[Block]:
    return [
        (pymupdf.Rect(b[:4]), normalize(" ".join(b[4].split())))
        for b in page.get_text("blocks")
        if b[6] == 0 and b[4].strip()
    ]


def _gap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Distance between two rectangles (0 if they touch or overlap)."""
    dx = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return max(dx, dy)


def _caption_block(blocks: list[Block], k: int) -> Block:
    """Caption block ``k``; a bare label ("Figure 1.1") is joined with the text under it."""
    rect, text = blocks[k]
    if len(text) < 20:
        below = [
            (r, t)
            for r, t in blocks
            if 0 <= r.y0 - rect.y1 <= 12 and r.x0 < rect.x1 and r.x1 > rect.x0
        ]
        if below:
            r, t = min(below, key=lambda b: b[0].y0)
            rect, text = rect | r, f"{text} {t}"
    return rect, text[:300]


def _is_caption(text: str) -> bool:
    """ "Figure 3: The heart" or a bare "Figure 1.1", but not "Figure 5.16 shows that …"."""
    match = CAPTION_LINE.match(text)
    rest = text[match.end() :].lstrip() if match else ""
    return match is not None and not rest[:1].islower()


def _captions(blocks: list[Block]) -> list[tuple[pymupdf.Rect, str]]:
    return [_caption_block(blocks, k) for k, (_, t) in enumerate(blocks) if _is_caption(t)]


def _caption_distance(figure: pymupdf.Rect, caption: pymupdf.Rect) -> float | None:
    """How far ``caption`` is from ``figure``, or None if it can't be its caption.

    Captions sit above or below the figure, or beside it in the page margin.
    """
    if caption.x0 < figure.x1 and caption.x1 > figure.x0:  # above / below
        gap = max(0.0, caption.y0 - figure.y1, figure.y0 - caption.y1)
        return gap if gap <= CAPTION_GAP else None
    if caption.y0 < figure.y1 and caption.y1 > figure.y0:  # beside
        gap = max(caption.x0 - figure.x1, figure.x0 - caption.x1)
        return gap if gap <= SIDE_CAPTION_GAP else None
    return None


def _nearest_caption(rect: pymupdf.Rect, captions: list[tuple[pymupdf.Rect, str]]) -> str:
    """The caption above, below or beside ``rect`` ("Figure 3: …"), if any."""
    near = [(d, t) for r, t in captions if (d := _caption_distance(rect, r)) is not None]
    return min(near)[1] if near else ""


def _with_labels(region: pymupdf.Rect, blocks: list[Block], caption: pymupdf.Rect):
    """``region`` grown by the text labels of the drawing: blocks inside or overlapping
    it, and short ones just next to it (not body paragraphs, not the caption)."""
    for _ in range(3):
        grown = pymupdf.Rect(region)
        for rect, text in blocks:
            if rect.intersects(caption) or rect in grown:
                continue
            label = len(text) <= LABEL_CHARS and _gap(rect, grown) <= LABEL_GAP
            if rect.intersects(grown) or label:
                grown |= rect
        if grown == region:
            break
        region = grown
    return region


def _vector_figures(
    page: pymupdf.Page, blocks: list[Block], captions: list[tuple[pymupdf.Rect, str]]
) -> list[ChapterImage]:
    """Figures drawn with vector graphics, cropped from the page.

    Only drawings next to a caption count, so boxes, rules and table borders
    elsewhere on the page are not mistaken for figures.
    """
    if not captions:
        return []
    try:
        clusters = [r for r in page.cluster_drawings() if r.width > 3 and r.height > 3]
    except Exception:
        return []
    groups: list[pymupdf.Rect] = []  # clusters that are close together form one figure
    for rect in sorted(clusters, key=lambda r: (r.y0, r.x0)):
        for k, group in enumerate(groups):
            if _gap(group, rect) <= FIGURE_GAP:
                groups[k] = group | rect
                break
        else:
            groups.append(pymupdf.Rect(rect))
    figures: list[ChapterImage] = []
    used: set[int] = set()
    for cap_rect, caption in captions:
        candidates = [
            (d, k)
            for k, g in enumerate(groups)
            if k not in used and (d := _caption_distance(g, cap_rect)) is not None
        ]
        if not candidates:
            continue
        k = min(candidates)[1]
        used.add(k)
        region = _with_labels(pymupdf.Rect(groups[k]), blocks, cap_rect)
        region = (region + (-6, -6, 6, 6)) & page.rect
        if region.width < MIN_FIGURE_SIDE or region.height < MIN_FIGURE_SIDE:
            continue
        zoom = min(4.0, FIGURE_PIXELS / max(region.width, region.height))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=region, alpha=False)
        figures.append(ChapterImage(pix.tobytes("png"), caption))
    return figures


def _chapter_images(doc: pymupdf.Document, pages: range, skip: set[int]) -> list[ChapterImage]:
    images: list[ChapterImage] = []
    seen: set[int] = set()
    for page_no in pages:
        if page_no in skip:  # the "image" on a scanned page is the page itself
            continue
        page = doc[page_no]
        blocks = _text_blocks(page)
        captions = _captions(blocks)
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)
            if img[2] < MIN_IMAGE_SIDE or img[3] < MIN_IMAGE_SIDE:
                continue
            try:
                pix = pymupdf.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:  # CMYK → RGB
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                rects = page.get_image_rects(xref)
                caption = _nearest_caption(rects[0], captions) if rects else ""
                images.append(ChapterImage(pix.tobytes("png"), caption))
            except Exception:
                continue
        # Captions not taken by a raster image may belong to a vector figure.
        taken = {i.caption for i in images}
        images.extend(_vector_figures(page, blocks, [c for c in captions if c[1] not in taken]))
        if len(images) >= MAX_IMAGES_PER_CHAPTER:
            return images[:MAX_IMAGES_PER_CHAPTER]
    return images


def tesseract_available() -> bool:
    try:
        return bool(pymupdf.get_tessdata())
    except Exception:
        return False


def _needs_ocr(page: pymupdf.Page) -> bool:
    if len(page.get_text("text").strip()) >= MIN_TEXT_CHARS:
        return False
    # Only OCR pages that actually show something: images or vector drawings.
    return bool(page.get_images()) or len(page.get_drawings()) > 20


class _OCR:
    def __init__(self, languages: str):
        self.auto = languages.strip().lower() in ("", "auto")
        self.languages = "eng" if self.auto else languages.strip()
        self.detected = not self.auto

    def page_lines(self, page: pymupdf.Page, page_no: int) -> list[Line]:
        lines = self._ocr(page, page_no)
        if not self.detected:
            # Auto mode: OCR the first scanned page in English, detect the language,
            # then re-run it with the right Tesseract language pack.
            self.detected = True
            lang = detect_language(" ".join(ln.text for ln in lines))
            code = TESSERACT_LANGS.get(lang)
            if code and code != "eng" and code in _installed_langs():
                self.languages = f"{code}+eng"
                lines = self._ocr(page, page_no)
        return lines

    def _ocr(self, page: pymupdf.Page, page_no: int) -> list[Line]:
        textpage = page.get_textpage_ocr(language=self.languages, dpi=OCR_DPI, full=True)
        return _page_lines(page, page_no, textpage)


def _installed_langs() -> set[str]:
    tessdata = Path(pymupdf.get_tessdata())
    return {f.stem for f in tessdata.glob("*.traineddata")}


def _report_mupdf_messages(path: Path) -> None:
    messages = [m for m in pymupdf.TOOLS.mupdf_warnings(reset=1).splitlines() if m.strip()]
    if not messages:
        return
    for message in messages:
        log.debug("MuPDF (%s): %s", path.name, message)
    log.info(
        "%s is malformed; MuPDF repaired it while reading (details with -v / debug logging)",
        path.name,
    )


def load_pdf(path: Path, options: ParseOptions | None = None) -> Document:
    try:
        return _load_pdf(path, options or ParseOptions())
    finally:
        _report_mupdf_messages(path)


def _load_pdf(path: Path, options: ParseOptions) -> Document:
    pymupdf.TOOLS.reset_mupdf_warnings()
    with pymupdf.open(path) as doc:
        pages: list[list[Line]] = []
        ocr_pages: set[int] = set()
        ocr: _OCR | None = None
        for i in range(doc.page_count):
            page = doc[i]
            wants_ocr = options.ocr == "always" or (options.ocr == "auto" and _needs_ocr(page))
            if not wants_ocr:
                pages.append(_page_lines(page, i))
                continue
            if not tesseract_available():
                raise ParseError(
                    f"{path.name} contains scanned pages but Tesseract OCR is not installed. "
                    "Install it (e.g. `brew install tesseract tesseract-lang`) or disable OCR."
                )
            ocr = ocr or _OCR(options.ocr_languages)
            options.progress(i / doc.page_count, f"OCR page {i + 1}/{doc.page_count}…")
            pages.append(ocr.page_lines(page, i))
            ocr_pages.add(i)
        pages = _remove_running_headers(pages)
        flat = [ln for lines in pages for ln in lines]
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip() or path.stem.replace("_", " ")
        author = (meta.get("author") or "").strip() or None

        starts = _chapters_from_toc(doc.get_toc(), flat, doc.page_count) or _chapters_from_fonts(
            flat
        )
        chapters: list[Chapter] = []
        if not starts:
            chapters.append(Chapter(title, _to_paragraphs(flat)))
        else:
            if starts[0][1] > 0:
                preface = flat[: starts[0][1]]
                if sum(len(ln.text.split()) for ln in preface) > 150:
                    chapters.append(Chapter(title, _to_paragraphs(preface)))
            for n, (ch_title, start) in enumerate(starts):
                end = starts[n + 1][1] if n + 1 < len(starts) else len(flat)
                lines = flat[start:end]
                if not lines:
                    continue
                page_range = range(lines[0].page, lines[-1].page + 1)
                chapters.append(
                    Chapter(
                        ch_title,
                        _to_paragraphs(lines),
                        _chapter_images(doc, page_range, ocr_pages),
                    )
                )
    return Document(path=path, title=title, author=author, chapters=chapters)
