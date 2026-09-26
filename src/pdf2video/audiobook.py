"""Audiobook pipeline: Document → one MP3 per chapter (or chapter part)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import media
from .jobs import JobContext
from .model import Chapter, Document
from .narration import Narrator
from .settings import Settings
from .textutil import Chunk, chunk_paragraphs, plan_parts, safe_filename
from .tts import TTSEngine

# Leave room for the lead-in silence and inter-chunk gaps added while encoding.
SPLIT_MARGIN_SECONDS = 3.0
LEAD_SILENCE = 0.5
CHUNK_GAP = 0.25
CHARS_PER_SECOND = 15  # rough narration speed, only used to size chunks


@dataclass
class OutputResult:
    out_dir: Path
    files: list[Path] = field(default_factory=list)


def part_filenames(number: int, width: int, title: str, parts: int, ext: str) -> list[str]:
    """``01 - Title.mp3`` or ``01 - Title_1.mp3``, ``01 - Title_2.mp3``, …

    ``number`` is the chapter's position in the full document and ``width`` the
    zero-padded digit count (see :attr:`Document.number_width`).
    """
    stem = f"{number:0{width}d} - {safe_filename(title)}"
    if parts == 1:
        return [f"{stem}{ext}"]
    return [f"{stem}_{k}{ext}" for k in range(1, parts + 1)]


def chunk_chars_for(max_seconds: float) -> int:
    """TTS chunk size: small enough that a part limit can be met with ~4 chunks."""
    return int(min(900, max(200, max_seconds * CHARS_PER_SECOND / 4)))


def chapter_chunks(chapter: Chapter, max_chars: int = 1500) -> list[Chunk]:
    title = chapter.title.rstrip(".:!?") + "."
    return [Chunk(title, True), *chunk_paragraphs(chapter.paragraphs, max_chars)]


def make_audiobook(
    doc: Document,
    settings: Settings,
    ctx: JobContext | None = None,
    engine: TTSEngine | None = None,
) -> OutputResult:
    ctx = ctx or JobContext()
    if error := media.check_ffmpeg():
        raise media.MediaError(error)
    ctx.progress(0.0, "Preparing…")
    narrator = Narrator(settings, doc.chapters[0].text if doc.chapters else "", engine)
    narrator.prepare(ctx)
    out_dir = doc.output_dir("audio")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = OutputResult(out_dir)

    max_seconds = max(30.0, settings.max_chapter_seconds - SPLIT_MARGIN_SECONDS)
    max_chars = chunk_chars_for(max_seconds)
    all_chunks = [chapter_chunks(ch, max_chars) for ch in doc.chapters]
    total_chars = sum(len(c.text) for chunks in all_chunks for c in chunks) or 1
    done_chars = 0
    n_chapters = len(doc.chapters)
    track = 0

    with tempfile.TemporaryDirectory(prefix="pdf2video-") as tmp:
        workdir = Path(tmp)
        for ch_index, (chapter, chunks) in enumerate(zip(doc.chapters, all_chunks, strict=True)):
            ctx.check()
            label = f"Chapter {chapter.number} ({ch_index + 1}/{n_chapters}): {chapter.title}"
            base_chars = done_chars
            cumulative = [0]
            for c in chunks:
                cumulative.append(cumulative[-1] + len(c.text))

            def on_done(
                n: int, base_chars: int = base_chars, cumulative=cumulative, label: str = label
            ) -> None:
                # Approximate: assume chunks finish roughly in order.
                frac = (base_chars + cumulative[n]) / total_chars
                ctx.progress(frac * 0.97, f"Narrating {label}")

            ctx.progress(done_chars / total_chars * 0.97, f"Narrating {label}")
            clips = narrator.synthesize_many(
                [c.text for c in chunks], workdir, f"ch{ch_index:03d}", ctx, on_done
            )
            parts = plan_parts(
                [c.seconds + CHUNK_GAP for c in clips],
                [c.ends_paragraph for c in chunks],
                max_seconds,
            )
            names = part_filenames(
                chapter.number, doc.number_width, chapter.title, len(parts), ".mp3"
            )
            for part_no, (indices, name) in enumerate(zip(parts, names, strict=True), start=1):
                ctx.check()
                ctx.progress(
                    (done_chars + cumulative[-1]) / total_chars * 0.97,
                    f"Encoding {name}",
                )
                track += 1
                title = chapter.title if len(parts) == 1 else f"{chapter.title} ({part_no})"
                tags = {
                    "title": title,
                    "album": doc.title,
                    "track": str(track),
                    "genre": "Audiobook",
                }
                if doc.author:
                    tags["artist"] = tags["album_artist"] = doc.author
                target = out_dir / name
                media.concat_audio_to_mp3(
                    [clips[i].path for i in indices],
                    target,
                    lead_silence=LEAD_SILENCE,
                    gap=CHUNK_GAP,
                    tags=tags,
                    bitrate=settings.bitrate,
                    cancel=ctx.cancel_event,
                )
                result.files.append(target)
            for clip in clips:
                clip.path.unlink(missing_ok=True)
            done_chars += cumulative[-1]

    ctx.progress(1.0, f"Done: {len(result.files)} files in {out_dir}")
    return result
