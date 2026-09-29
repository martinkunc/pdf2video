"""Format-independent document model produced by the parsers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass
class ChapterImage:
    """An image embedded in the document (a figure, photo, map, …)."""

    data: bytes  # encoded image file (PNG, JPEG, …)
    caption: str = ""  # figure caption or alt text found in the document, if any


@dataclass
class Chapter:
    title: str
    paragraphs: list[str]
    images: list[ChapterImage] = field(default_factory=list)
    number: int = 0  # 1-based position in the full document (0 = not numbered)

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def word_count(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)


@dataclass
class Document:
    path: Path
    title: str
    chapters: list[Chapter]
    author: str | None = None
    chapter_count: int = 0  # chapters in the full document (a selection may hold fewer)

    def __post_init__(self) -> None:
        self.number_chapters()

    def number_chapters(self) -> None:
        """Give unnumbered chapters their position and remember the full count."""
        for i, chapter in enumerate(self.chapters, start=1):
            if not chapter.number:
                chapter.number = i
        numbers = [c.number for c in self.chapters]
        self.chapter_count = max([self.chapter_count, len(self.chapters), *numbers])

    def select(self, numbers: Iterable[int]) -> Document:
        """A copy containing only the chapters with these (1-based) numbers.

        The chapters keep their original numbers, so output files are named after
        their position in the full document.
        """
        wanted = set(numbers)
        return replace(self, chapters=[c for c in self.chapters if c.number in wanted])

    @property
    def number_width(self) -> int:
        """Digits used for chapter numbers in file names (at least 2)."""
        return max(2, len(str(self.chapter_count or len(self.chapters))))

    @property
    def word_count(self) -> int:
        return sum(c.word_count for c in self.chapters)

    def output_dir(self, suffix: str) -> Path:
        """`<doc dir>/<doc stem>_<suffix>`, e.g. `book_audio`."""
        return self.path.parent / f"{self.path.stem}_{suffix}"
