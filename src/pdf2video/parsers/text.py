"""Plain text and Markdown loader."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..model import Chapter, Document
from ..textutil import CHAPTER_LINE, lines_to_paragraphs

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_INLINE = [
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),  # images
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # links
    (re.compile(r"`([^`]+)`"), r"\1"),
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),
    (re.compile(r"(?<!\w)[*_](.+?)[*_](?!\w)"), r"\1"),
    (re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+"), ""),  # list markers
    (re.compile(r"^\s{0,3}>\s?"), ""),  # blockquotes
]
_MD_RULE = re.compile(r"^\s*([-*_]\s*){3,}$")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1250", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            continue
        return text
    return raw.decode("utf-8", errors="replace")


def _strip_markdown(line: str) -> str:
    for pattern, repl in _MD_INLINE:
        line = pattern.sub(repl, line)
    return line


def _is_title_case(text: str) -> bool:
    words = [w for w in re.findall(r"[^\W\d_][\w'’-]*", text) if len(w) >= 4]
    return bool(words) and all(w[0].isupper() for w in words)


def _merge_chapter_subtitles(lines: list[str]) -> list[str]:
    """Join "CHAPTER I." with a following short title line ("Down the Rabbit-Hole")."""
    lines = list(lines)
    for i, line in enumerate(lines):
        if not CHAPTER_LINE.match(line):
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j - i > 3 or j >= len(lines):
            continue
        candidate = lines[j].strip()
        followed_by_blank = j + 1 >= len(lines) or not lines[j + 1].strip()
        if (
            followed_by_blank
            and len(candidate) <= 80
            and not CHAPTER_LINE.match(candidate)
            and _is_title_case(candidate)
            and len(line.strip()) <= 20
        ):
            lines[i] = f"{line.strip()} {candidate}"
            lines[j] = ""
    return lines


def load_text(path: Path) -> Document:
    lines = _merge_chapter_subtitles(_read_text(path).splitlines())
    is_markdown = path.suffix.lower() in (".md", ".markdown")

    md_levels = Counter(len(m.group(1)) for line in lines if (m := _MD_HEADING.match(line)))
    title = path.stem.replace("_", " ")
    chapter_level: int | None = None
    if md_levels:
        # A single H1 is the document title; chapters are the next level used repeatedly.
        levels = sorted(md_levels)
        chapter_level = levels[1] if md_levels[levels[0]] == 1 and len(levels) > 1 else levels[0]

    chapters: list[Chapter] = []
    buffer: list[str] = []
    current_title = title
    in_code = False

    def flush() -> None:
        paragraphs = lines_to_paragraphs(buffer)
        if paragraphs or chapters:
            chapters.append(Chapter(current_title, paragraphs))
        buffer.clear()

    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if m := _MD_HEADING.match(line):
            level, text = len(m.group(1)), _strip_markdown(m.group(2))
            if chapter_level and level < chapter_level:
                title = text
                current_title = text if not chapters and not buffer else current_title
                continue
            if chapter_level and level == chapter_level:
                flush()
                current_title = text
                continue
            buffer.extend(["", text, ""])  # sub-heading: its own paragraph
            continue
        if not md_levels and CHAPTER_LINE.match(line) and len(line.strip()) < 90:
            flush()
            current_title = line.strip()
            continue
        if is_markdown:
            if _MD_RULE.match(line):
                buffer.append("")
                continue
            line = _strip_markdown(line)
        buffer.append(line)
    flush()
    return Document(path=path, title=title, chapters=chapters)
