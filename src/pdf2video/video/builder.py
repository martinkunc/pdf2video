"""Explanatory video pipeline: Document → script → slides + narration → MP4 per chapter."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .. import media
from ..audiobook import OutputResult, part_filenames
from ..jobs import JobContext
from ..model import ChapterImage, Document
from ..narration import Narrator
from ..settings import Settings
from ..textutil import plan_parts, split_sentences
from ..tts import TTSEngine
from .bookimages import by_chapter, prepare_book_images
from .bookimages import collect as collect_book_images
from .diagrams import render_diagram, reveal_steps
from .script import Illustration, Scene, VideoScript, build_script, show_mentioned_figures
from .slides import fit_rect, illustration_box, render_scene_slide, render_title_slide

log = logging.getLogger(__name__)

CLIP_PAD = 0.6  # silence after each narration (seconds)
AUDIO_DELAY = 0.2  # narration starts this long after the slide appears
SUBTITLE_CHARS = 90
REVEAL_SPAN = 0.85  # a revealed diagram is complete after this share of the narration
FADE_IN_AT = 0.8  # seconds


@dataclass
class _SlidePlan:
    """How one slide becomes a clip: a still, a sequence of frames, or a moving picture."""

    frames: list[Path]
    timing: str = "still"  # "still" | "fade" | "reveal" | "motion"
    picture: Path | None = None  # motion only
    position: tuple[int, int] = (0, 0)
    motion: str = ""

    def frame_starts(self, audio_seconds: float) -> list[float]:
        if self.timing == "fade":
            return [0.0, FADE_IN_AT]
        n = len(self.frames)
        return [0.0] + [AUDIO_DELAY + audio_seconds * REVEAL_SPAN * k / n for k in range(1, n)]


@dataclass
class _Pictures:
    """The files for the script's picture illustrations: generated (prompt → file)
    or, where the AI chose one, the book's own image."""

    root: Path  # the video folder
    files: dict[tuple[str, str], Path] = field(default_factory=dict)

    def book_image(self, ill: Illustration) -> Path | None:
        path = self.root / ill.book_image if ill.book_image else None
        return path if path is not None and path.is_file() else None

    def get(self, ill: Illustration) -> Path | None:
        return self.book_image(ill) or self.files.get((ill.prompt, ill.placement))


@dataclass
class _Clip:
    video: Path
    seconds: float
    narration: str
    audio_seconds: float


def _srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _subtitle_lines(text: str) -> list[str]:
    cues: list[str] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        current = ""
        for word in words:
            if current and len(current) + len(word) + 1 > SUBTITLE_CHARS:
                cues.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            cues.append(current)
    return cues


def write_srt(clips: list[_Clip], out: Path) -> None:
    entries: list[str] = []
    offset = 0.0
    for clip in clips:
        cues = _subtitle_lines(clip.narration)
        total_chars = sum(len(c) for c in cues) or 1
        t = offset + AUDIO_DELAY
        for cue in cues:
            length = clip.audio_seconds * len(cue) / total_chars
            entries.append(
                f"{len(entries) + 1}\n{_srt_time(t)} --> {_srt_time(t + length)}\n{cue}\n"
            )
            t += length
        offset += clip.seconds
    out.write_text("\n".join(entries), encoding="utf-8")


def _picture_aspect(placement: str) -> float:
    x0, y0, x1, y1 = illustration_box(placement)
    return (x1 - x0) / (y1 - y0)


def _generate_pictures(
    script: VideoScript,
    settings: Settings,
    out_dir: Path,
    ctx: JobContext,
    progress,
) -> _Pictures:
    """Create (or reuse from ``out_dir/illustrations``) every picture illustration.

    Failures are not fatal: scenes whose picture can't be made just show no
    illustration.
    """
    from .. import imagegen
    from ..llm import unload as unload_llm

    pictures = _Pictures(out_dir)
    wanted = []
    for section in script.sections:
        for scene in section.scenes:
            ill = scene.illustration
            if ill is None or ill.kind != "picture":
                continue
            if pictures.book_image(ill) is not None:
                ill.image = ill.book_image  # the AI chose the book's own image
            elif not ill.prompt:
                ill.image = ""  # a book image that is gone, and nothing to generate
            else:
                if ill.book_image:
                    log.warning("%s is missing; generating a picture instead", ill.book_image)
                wanted.append(ill)
    if not wanted:
        return pictures
    generator = imagegen.ImageGenerator(settings.image_model, settings.llm_auto_download)
    folder = out_dir / "illustrations"
    todo: dict[Path, list[Illustration]] = {}  # missing file → scenes that show it
    for ill in wanted:
        try:
            key = generator.cache_key(ill.prompt, _picture_aspect(ill.placement))
        except imagegen.ModelError as exc:
            log.warning("No pictures: %s", exc)
            return pictures
        path = folder / f"{key}.png"
        ill.image = f"illustrations/{path.name}"
        pictures.files[(ill.prompt, ill.placement)] = path
        if not path.exists():
            todo.setdefault(path, []).append(ill)
    if not todo:
        return pictures

    def skip(ills: list[Illustration]) -> None:
        for ill in ills:  # the slide shows no picture when its file is missing
            ill.image = ""

    try:
        unload_llm()  # free its memory for the image model
        progress(0.0, "Preparing the image model…")
        generator.prepare(lambda f, m: progress(0.3 * f, m), ctx.check)
    except media.Cancelled:
        raise
    except Exception as exc:
        log.warning("Image model unavailable (%s); pictures are skipped", exc)
        progress(1.0, f"Image model unavailable, pictures skipped: {exc}")
        for ills in todo.values():
            skip(ills)
        return pictures
    folder.mkdir(parents=True, exist_ok=True)
    for n, (path, ills) in enumerate(todo.items(), start=1):
        ctx.check()
        progress(0.3 + 0.7 * (n - 1) / len(todo), f"Drawing picture {n}/{len(todo)}")
        try:
            image = generator.generate(ills[0].prompt, _picture_aspect(ills[0].placement))
            image.save(path)
        except Exception as exc:
            log.warning("Picture failed for %r: %s", ills[0].prompt, exc)
            skip(ills)
    imagegen.unload()
    return pictures


def _scene_slide(
    workdir: Path,
    name: str,
    scene: Scene,
    footer: str,
    progress: float,
    doc_image: ChapterImage | None,
    pictures: _Pictures | None,
) -> _SlidePlan:
    """Render a scene's slide(s) and decide how they are animated."""
    ill = scene.illustration if pictures is not None else None

    def render(suffix: str = "", **kwargs) -> Path:
        return render_scene_slide(
            workdir / f"{name}{suffix}.png", scene.title, scene.bullets, footer, progress, **kwargs
        )

    if ill is not None and ill.kind == "diagram" and ill.diagram is not None:
        diagram = ill.diagram
        if diagram.caption.casefold().strip(" .:") in scene.title.casefold():
            diagram = diagram.model_copy(update={"caption": ""})  # just repeats the title
        x0, y0, x1, y1 = illustration_box(ill.placement)
        size = (x1 - x0, y1 - y0)
        if ill.animation == "reveal" and reveal_steps(diagram) > 1:
            frames = [
                render(f"_{k}", placement=ill.placement, drawing=render_diagram(diagram, size, k))
                for k in range(1, reveal_steps(diagram) + 1)
            ]
            return _SlidePlan(frames, "reveal")
        full = render(placement=ill.placement, drawing=render_diagram(diagram, size))
        if ill.animation == "fade_in":
            return _SlidePlan([render("_0", placement=ill.placement), full], "fade")
        return _SlidePlan([full])

    picture_path = pictures.get(ill) if ill is not None and ill.kind == "picture" else None
    if ill is not None and picture_path is not None and picture_path.exists():
        with Image.open(picture_path) as img:
            picture = img.convert("RGB")
        rect = fit_rect(picture.size, illustration_box(ill.placement))
        if ill.animation in media.MOTIONS:
            fitted = workdir / f"{name}_picture.png"
            picture.resize((rect[2] - rect[0], rect[3] - rect[1]), Image.Resampling.LANCZOS).save(
                fitted
            )
            base = render(placement=ill.placement, picture_rect=rect)
            return _SlidePlan([base], "motion", fitted, (rect[0], rect[1]), ill.animation)
        return _SlidePlan([render(placement=ill.placement, picture=picture, picture_rect=rect)])

    return _SlidePlan([render(image=doc_image.data if doc_image else None)])


def _encode(plan: _SlidePlan, audio, out: Path, ctx: JobContext) -> None:
    if plan.timing == "motion" and plan.picture is not None:
        media.motion_clip(
            plan.frames[0], plan.picture, plan.position, audio.path, out,
            motion=plan.motion, pad=CLIP_PAD, cancel=ctx.cancel_event,
        )  # fmt: skip
    elif len(plan.frames) > 1:
        frames = list(zip(plan.frames, plan.frame_starts(audio.seconds), strict=True))
        media.frames_clip(frames, audio.path, out, pad=CLIP_PAD, cancel=ctx.cancel_event)
    else:
        media.still_clip(plan.frames[0], audio.path, out, pad=CLIP_PAD, cancel=ctx.cancel_event)


def make_video(
    doc: Document,
    settings: Settings,
    ctx: JobContext | None = None,
    engine: TTSEngine | None = None,
    script: VideoScript | None = None,
) -> OutputResult:
    """Build the videos; pass ``script`` to reuse a previously generated script."""
    ctx = ctx or JobContext()
    if error := media.check_ffmpeg():
        raise media.MediaError(error)
    ctx.progress(0.0, "Preparing…")
    narrator = Narrator(settings, doc.chapters[0].text if doc.chapters else "", engine)
    narrator.prepare(ctx)
    out_dir = doc.output_dir("video")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = OutputResult(out_dir)

    # Phase 1: the book's own images (captured and summarised), then the script.
    if script is None:
        mode = settings.video_script
        llm = settings.llm() if mode != "extractive" else None
        uses_ai = llm is not None and llm.status(check_remote=False).ok
        book_images = {}
        book_share = 0.0
        if uses_ai and settings.illustrations and settings.book_images:
            if any(c.images for c in doc.chapters):
                book_share = 0.1
            book_images = prepare_book_images(
                doc,
                out_dir,
                settings.vision_model,
                settings.llm_auto_download,
                lambda f, message: ctx.progress(book_share * f, message),
                ctx.check,
            )
        script_share = book_share + (0.5 if uses_ai else 0.02)
        ctx.progress(book_share, "Preparing script…")
        script = build_script(
            doc,
            mode,
            narrator.language,
            llm,
            on_progress=lambda f, message: ctx.progress(
                book_share + (script_share - book_share) * f, message
            ),
            check_cancel=ctx.check,
            illustrations=settings.illustrations,
            book_images=book_images,
        )
    else:
        script_share = 0.0
        if settings.illustrations:
            # The files a reused script may refer to; figures the narration mentions.
            book_images = by_chapter(collect_book_images(doc, out_dir))
            if settings.book_images:
                for section in script.sections:
                    if section.source == "ai" and section.number in book_images:
                        show_mentioned_figures(section, book_images[section.number])
        # A reused script may cover more chapters than are selected now.
        numbers = {c.number for c in doc.chapters}
        titles = {c.title for c in doc.chapters}
        script = script.model_copy(
            update={
                "sections": [
                    s
                    for s in script.sections
                    if (s.number in numbers if s.number else s.chapter_title in titles)
                ]
            }
        )
    script.save(out_dir / "script.json")

    # Phase 2: pictures for the illustrations (diagrams are drawn with the slides).
    pictures: _Pictures | None = None
    image_share = 0.0
    if settings.illustrations:
        needs_images = any(
            s.illustration is not None and s.illustration.kind == "picture"
            for section in script.sections
            for s in section.scenes
        )
        image_share = 0.25 * (1 - script_share) if needs_images else 0.0
        pictures = _generate_pictures(
            script,
            settings,
            out_dir,
            ctx,
            lambda f, message: ctx.progress(script_share + image_share * f, message),
        )
        script.save(out_dir / "script.json")  # now with the picture file names

    # Phase 3: slides, narration, clips, per-chapter videos.
    total_scenes = sum(len(s.scenes) + 1 for s in script.sections) or 1
    done_scenes = 0
    n_chapters = len(script.sections)
    start_share = script_share + image_share

    def report(extra: float, message: str) -> None:
        frac = start_share + (1 - start_share) * (done_scenes + extra) / total_scenes
        ctx.progress(frac * 0.99, message)

    with tempfile.TemporaryDirectory(prefix="pdf2video-") as tmp:
        workdir = Path(tmp)
        for ch_index, section in enumerate(script.sections):
            ctx.check()
            # Match by title so that a reused script.json still finds the chapter images.
            chapter = next(
                (c for c in doc.chapters if c.number == section.number)
                if section.number
                else (c for c in doc.chapters if c.title == section.chapter_title),
                None,
            )
            images = chapter.images if chapter else []
            if pictures is not None and settings.book_images and section.source == "ai":
                images = []  # the book's images appear only where the AI chose them
            number = section.number or (chapter.number if chapter else ch_index + 1)
            label = f"chapter {number} ({ch_index + 1}/{n_chapters})"
            prefix = f"c{ch_index:03d}"
            n_items = len(section.scenes) + 1

            # Slides.
            report(0, f"Rendering slides for {label}")
            plans = [
                _SlidePlan(
                    [
                        render_title_slide(
                            workdir / f"{prefix}_s000.png",
                            section.chapter_title,
                            f"Chapter {number}",
                            doc.title,
                        )
                    ]
                )
            ]
            for k, scene in enumerate(section.scenes, start=1):
                plans.append(
                    _scene_slide(
                        workdir,
                        f"{prefix}_s{k:03d}",
                        scene,
                        f"{doc.title} · {section.chapter_title}",
                        k / len(section.scenes),
                        images[k - 1] if k - 1 < len(images) else None,
                        pictures,
                    )
                )

            # Narration.
            texts = [section.chapter_title.rstrip(".:!?") + "."] + [
                s.narration for s in section.scenes
            ]
            audio = narrator.synthesize_many(
                texts,
                workdir,
                prefix + "_n",
                ctx,
                lambda n, label=label, n_items=n_items: report(
                    0.5 * n / n_items, f"Narrating {label}"
                ),
            )

            # Clips.
            clips: list[_Clip] = []
            for k, (plan, narration) in enumerate(zip(plans, audio, strict=True)):
                ctx.check()
                report(0.5 + 0.5 * k / n_items, f"Encoding {label}, scene {k + 1}/{n_items}")
                clip_path = workdir / f"{prefix}_v{k:03d}.mp4"
                _encode(plan, narration, clip_path, ctx)
                clips.append(
                    _Clip(clip_path, media.duration(clip_path), texts[k], narration.seconds)
                )

            # Split long chapters at scene boundaries and join.
            parts = plan_parts(
                [c.seconds for c in clips], [True] * len(clips), settings.max_chapter_seconds
            )
            names = part_filenames(
                number, doc.number_width, section.chapter_title, len(parts), ".mp4"
            )
            for part_no, (indices, name) in enumerate(zip(parts, names, strict=True), start=1):
                ctx.check()
                target = out_dir / name
                title = section.chapter_title
                if len(parts) > 1:
                    title += f" ({part_no})"
                part_clips = [clips[i] for i in indices]
                media.concat_videos(
                    [c.video for c in part_clips],
                    target,
                    tags={"title": title, "album": doc.title, "track": str(number)},
                    cancel=ctx.cancel_event,
                )
                write_srt(part_clips, target.with_suffix(".srt"))
                result.files.append(target)
            for path in workdir.iterdir():
                path.unlink(missing_ok=True)
            done_scenes += n_items

    ctx.progress(1.0, f"Done: {len(result.files)} videos in {out_dir}")
    return result
