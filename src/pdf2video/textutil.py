"""Text cleanup, sentence splitting, chunking and related helpers."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

WORDS_PER_MINUTE = 175  # typical neural-voice narration pace, used only for estimates

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTISPACE = re.compile(r"[ \t ]+")
_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "fig",
    "no",
    "vol",
    "approx",
    "cf",
    "al",
    "inc",
    "ltd",
    "co",
    "např",
    "tzv",
    "atd",
    "resp",
}
_SENTENCE_END = re.compile(r"([.!?…]+[\"'”’»)\]]*)\s+(?=[\"'“‘«(\[]?[A-ZÀ-ÖØ-ÞĀ-Ž0-9])")


def normalize(text: str) -> str:
    """Normalize unicode, ligatures and whitespace; join hyphenated line breaks."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return text


def clean_paragraph(text: str) -> str:
    text = normalize(text)
    text = text.replace("\n", " ")
    return _MULTISPACE.sub(" ", text).strip()


def lines_to_paragraphs(lines: Iterable[str]) -> list[str]:
    """Join visually wrapped lines into paragraphs.

    A blank line always ends a paragraph; a line ending with sentence punctuation
    that is noticeably shorter than the typical line also ends one.
    """
    lines = [normalize(line).rstrip() for line in lines]
    lengths = [len(line) for line in lines if line.strip()]
    typical = sorted(lengths)[int(len(lengths) * 0.8)] if lengths else 0
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            paragraphs.append(clean_paragraph(" ".join(current)))
            current.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if _PAGE_NUMBER.match(stripped):
            continue
        if current and current[-1].endswith("-") and stripped[:1].islower():
            current[-1] = current[-1][:-1] + stripped
        else:
            current.append(stripped)
        if typical and stripped[-1:] in '.!?:…"”' and len(stripped) < typical * 0.75:
            flush()
    flush()
    return [p for p in paragraphs if p]


def split_sentences(text: str) -> list[str]:
    text = clean_paragraph(text)
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        candidate = text[start : match.end(1)]
        last_word = candidate.rsplit(" ", 1)[-1].rstrip(".").lower()
        if last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
            continue
        sentences.append(candidate.strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


@dataclass(frozen=True)
class Chunk:
    text: str
    ends_paragraph: bool


def chunk_paragraphs(paragraphs: Sequence[str], max_chars: int = 1500) -> list[Chunk]:
    """Group text into TTS-sized chunks, cut at sentence (preferably paragraph) boundaries."""
    chunks: list[Chunk] = []
    buf = ""

    def emit(text: str, ends_paragraph: bool) -> None:
        if text.strip():
            chunks.append(Chunk(text.strip(), ends_paragraph))

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(buf) + len(paragraph) + 1 <= max_chars:
            buf = f"{buf}\n{paragraph}" if buf else paragraph
            continue
        if buf:
            emit(buf, True)
            buf = ""
        if len(paragraph) <= max_chars:
            buf = paragraph
            continue
        # Long paragraph: split by sentences (and hard-wrap giant sentences).
        piece = ""
        for sentence in split_sentences(paragraph):
            for part in _hard_wrap(sentence, max_chars):
                if piece and len(piece) + len(part) + 1 > max_chars:
                    emit(piece, False)
                    piece = ""
                piece = f"{piece} {part}" if piece else part
        buf = piece
    if buf:
        emit(buf, True)
    return chunks


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for word in text.split(" "):
        if current and len(current) + len(word) + 1 > max_chars:
            parts.append(current)
            current = ""
        current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return parts


def plan_parts(
    durations: Sequence[float], paragraph_ends: Sequence[bool], max_seconds: float
) -> list[list[int]]:
    """Group consecutive chunk indices into parts no longer than ``max_seconds``.

    Parts are balanced: with total ``T`` and ``n = ceil(T / max)`` parts we aim at
    ``T / n`` each and prefer to cut at paragraph boundaries. A single chunk longer
    than the limit forms its own part.
    """
    if not durations:
        return []
    remaining = sum(durations)
    if remaining <= max_seconds:
        return [list(range(len(durations)))]

    def target_for(rest: float) -> float:
        return rest / math.ceil(rest / max_seconds)

    target = target_for(remaining)
    parts: list[list[int]] = []
    current: list[int] = []
    elapsed = 0.0
    for i, d in enumerate(durations):
        if current:
            overshoot = elapsed + d - target
            undershoot = target - elapsed
            # At a paragraph boundary we are happy to cut a bit earlier.
            eagerness = 2.0 if paragraph_ends[current[-1]] else 1.0
            if elapsed + d > max_seconds or (overshoot > 0 and undershoot < eagerness * overshoot):
                parts.append(current)
                remaining -= elapsed
                target = target_for(remaining) if remaining > max_seconds else remaining
                current, elapsed = [], 0.0
        current.append(i)
        elapsed += d
    if current:
        parts.append(current)
    return parts


def estimate_seconds(text_or_words: str | int, rate: float = 1.0) -> float:
    words = text_or_words if isinstance(text_or_words, int) else len(text_or_words.split())
    return words / (WORDS_PER_MINUTE * rate) * 60


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min" if m >= 10 else f"{m} min {s:02d} s"
    return f"{s} s"


_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def safe_filename(name: str, max_len: int = 80) -> str:
    name = _UNSAFE.sub(" ", unicodedata.normalize("NFC", name))
    name = _MULTISPACE.sub(" ", name).strip(" .")
    if len(name) > max_len:
        name = name[:max_len].rsplit(" ", 1)[0].rstrip(" .,;-")
    return name or "Untitled"


_STOPWORDS = {
    "en": "the and of to in is that it for was with as on be this are by not",
    "de": "der die das und ist nicht mit sich auf ein eine den von zu dem des",
    "fr": "le la les et est dans une des pour que qui pas sur du au avec",
    "es": "el la los las y es en que de por con para una del se no",
    "it": "il la di che e è per un una non con del della sono gli",
    "pt": "o a os as e é de que em um uma para com não do da",
    "nl": "de het een en is van dat niet op te zijn met voor",
    "cs": "a je se na že to v s z do jako by jsou ale pro není",
    "pl": "i w na że się nie to jest z do jak po ale od",
}
_STOPSETS = {lang: set(words.split()) for lang, words in _STOPWORDS.items()}


def detect_language(text: str) -> str:
    """Crude stop-word based language guess (ISO 639-1), defaulting to English."""
    words = re.findall(r"[^\W\d_]+", text[:20000].lower())
    if not words:
        return "en"
    counts = Counter(words)
    scores = {lang: sum(counts[w] for w in stops) for lang, stops in _STOPSETS.items()}
    best = max(scores, key=scores.__getitem__)
    return best if scores[best] > 0 else "en"


def looks_like_heading(line: str) -> bool:
    line = line.strip()
    return 0 < len(line) <= 90 and not line.endswith((".", ",", ";")) and len(line.split()) <= 12


CHAPTER_LINE = re.compile(
    r"^\s*(chapter|kapitola|kapitel|chapitre|cap[ií]tulo|capitolo|rozdział|part|část|teil)"
    r"\s+([0-9]+|[ivxlcdm]+|[a-z]+)\b.*$",
    re.IGNORECASE,
)
