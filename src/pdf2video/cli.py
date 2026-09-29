"""Command line interface: everything the GUI does, without the GUI.

Examples::

    pdf2video-cli info book.pdf
    pdf2video-cli text scan.pdf --ocr always --ocr-lang ces+eng --chapters 1
    pdf2video-cli audio book.epub --max-minutes 15 --chapters 2-3
    pdf2video-cli script book.pdf --model qwen3:8b
    pdf2video-cli video book.pdf --reuse-script
    pdf2video-cli video book.pdf --illustrations --image-model sdxl-turbo
    pdf2video-cli voices --engine edge --language cs
    pdf2video-cli models
    pdf2video-cli models --pull qwen3:4b
    pdf2video-cli models --pull-image sdxl-turbo
    pdf2video-cli models --pull-vision qwen3-vl-4b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .jobs import JobContext
from .media import Cancelled
from .model import Document
from .parsers import ParseError, load_document
from .settings import Settings
from .textutil import detect_language, estimate_seconds, format_duration
from .tts import ENGINES

COMMANDS = ("info", "text", "audio", "video", "script", "voices", "models")


def _progress(fraction: float, message: str) -> None:
    if sys.stderr.isatty():
        sys.stderr.write(f"\r\033[K{fraction * 100:5.1f}%  {message[:100]}")
    else:
        sys.stderr.write(f"{fraction * 100:5.1f}%  {message}\n")
    sys.stderr.flush()


def parse_chapter_selection(spec: str, count: int) -> list[int]:
    """ "1,3-5" → [0, 2, 3, 4] (0-based indices, validated against ``count``)."""
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        first, _, last = part.partition("-")
        start = int(first)
        end = int(last) if last else start
        if not (1 <= start <= end <= count):
            raise ValueError(f"chapter range {part!r} is outside 1-{count}")
        indices.extend(i - 1 for i in range(start, end + 1) if i - 1 not in indices)
    return indices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2video-cli",
        description="Create audiobooks and explanatory videos from documents.",
        epilog=__doc__.split("Examples::", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    doc_opts = argparse.ArgumentParser(add_help=False)
    doc_opts.add_argument("file", type=Path, help="PDF, DOCX, EPUB, TXT or MD file")
    doc_opts.add_argument("--chapters", help="only these chapters, e.g. 1,3-5")
    doc_opts.add_argument("--ocr", choices=["auto", "always", "off"], help="PDF OCR mode")
    doc_opts.add_argument("--ocr-lang", help="Tesseract languages, e.g. eng+ces (default auto)")

    tts_opts = argparse.ArgumentParser(add_help=False)
    tts_opts.add_argument("--engine", choices=list(ENGINES), help="TTS engine")
    tts_opts.add_argument("--voice", help="voice id (see the `voices` command)")
    tts_opts.add_argument("--rate", type=float, help="speaking speed multiplier, e.g. 1.1")
    tts_opts.add_argument("--max-minutes", type=float, help="max chapter length (default 10)")

    llm_opts = argparse.ArgumentParser(add_help=False)
    llm_opts.add_argument(
        "--script-mode", choices=["auto", "ai", "extractive"], help="how to write the script"
    )
    llm_opts.add_argument("--model", help="model name (e.g. qwen3:8b) or path to a .gguf file")
    llm_opts.add_argument(
        "--no-download", action="store_true", help="never download a missing model"
    )
    llm_opts.add_argument(
        "--illustrations",
        action=argparse.BooleanOptionalAction,
        help="let the AI add diagrams and generated pictures to scenes that need them",
    )
    llm_opts.add_argument(
        "--image-model", help="image model for pictures (e.g. sdxl-turbo) or a checkpoint path"
    )
    llm_opts.add_argument(
        "--book-images",
        action=argparse.BooleanOptionalAction,
        help="with --illustrations: show the book's own image where it fits better "
        "than a generated picture",
    )
    llm_opts.add_argument(
        "--vision-model", help="model that describes the book's images (e.g. qwen3-vl-4b)"
    )

    sub.add_parser("info", parents=[doc_opts], help="show detected chapters")
    text = sub.add_parser("text", parents=[doc_opts], help="print the extracted text")
    text.add_argument("--json", action="store_true", help="output JSON")
    sub.add_parser("audio", parents=[doc_opts, tts_opts], help="make the audiobook (MP3)")
    video = sub.add_parser(
        "video", parents=[doc_opts, tts_opts, llm_opts], help="make the explanatory video"
    )
    video.add_argument(
        "--reuse-script",
        action="store_true",
        help="use the existing <name>_video/script.json instead of generating a new one",
    )
    sub.add_parser(
        "script", parents=[doc_opts, llm_opts], help="only write the video script (script.json)"
    )
    voices = sub.add_parser("voices", help="list TTS voices")
    voices.add_argument("--engine", choices=list(ENGINES))
    voices.add_argument("--language", help="filter by language code, e.g. cs")
    models = sub.add_parser("models", help="list local LLM models / download one")
    models.add_argument("--pull", metavar="NAME", help="download a model, e.g. qwen3:4b")
    models.add_argument("--model", help="model to check (default from settings)")
    models.add_argument("--pull-image", metavar="NAME", help="download an image model")
    models.add_argument("--pull-vision", metavar="NAME", help="download a vision model")
    return parser


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> None:
    mapping = {
        "max_minutes": "max_chapter_minutes",
        "engine": "tts_engine",
        "voice": "voice",
        "rate": "rate",
        "script_mode": "video_script",
        "model": "llm_model",
        "ocr": "ocr",
        "ocr_lang": "ocr_languages",
        "illustrations": "illustrations",
        "image_model": "image_model",
        "book_images": "book_images",
        "vision_model": "vision_model",
    }
    for arg, field in mapping.items():
        value = getattr(args, arg, None)
        if value is not None:
            setattr(settings, field, value)
            if arg == "engine" and not getattr(args, "voice", None):
                settings.voice = ""
    if getattr(args, "no_download", False):
        settings.llm_auto_download = False


def _load(args: argparse.Namespace, settings: Settings) -> Document:
    doc = load_document(args.file, settings.parse_options(_progress))
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K")
    if args.chapters:
        indices = parse_chapter_selection(args.chapters, len(doc.chapters))
        doc = doc.select(i + 1 for i in indices)
    return doc


def _cmd_info(doc: Document, settings: Settings) -> None:
    print(f"{doc.title}" + (f" — {doc.author}" if doc.author else ""))
    total = estimate_seconds(doc.word_count, settings.rate)
    language = detect_language(" ".join(c.text[:3000] for c in doc.chapters[:5]))
    print(
        f"{len(doc.chapters)} chapters, {doc.word_count} words, ~{format_duration(total)}, "
        f"language: {language}"
    )
    for ch in doc.chapters:
        secs = estimate_seconds(ch.word_count, settings.rate)
        images = f"  [{len(ch.images)} img]" if ch.images else ""
        duration = format_duration(secs)
        print(f"{ch.number:3d}. {ch.title[:60]:60s} {ch.word_count:7d} w  ~{duration}{images}")


def _cmd_text(doc: Document, as_json: bool) -> None:
    if as_json:
        data = {
            "title": doc.title,
            "author": doc.author,
            "chapters": [
                {"title": c.title, "paragraphs": c.paragraphs, "images": len(c.images)}
                for c in doc.chapters
            ],
        }
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    for chapter in doc.chapters:
        print(f"# {chapter.title}\n")
        for paragraph in chapter.paragraphs:
            print(f"{paragraph}\n")


def _cmd_voices(settings: Settings, language: str | None) -> None:
    from .tts import get_engine

    engine = get_engine(settings.tts_engine)
    for voice in engine.voices():
        if language and voice.language != language:
            continue
        print(f"{voice.id:40s} {voice.locale:8s} {voice.gender}")
    if language:
        print(f"\ndefault for {language}: {engine.default_voice(language)}", file=sys.stderr)


def _cmd_models(
    settings: Settings,
    pull: str | None,
    pull_image: str | None = None,
    pull_vision: str | None = None,
) -> int:
    from . import imagegen
    from .llm import RECOMMENDED, LocalLLM, list_local, vision
    from .llm.store import download, format_size

    if pull_vision:
        vision.VisionLLM(pull_vision).prepare(_progress)
        vision.unload()
        print(f"\n{pull_vision} is ready")
        return 0
    if pull_image:
        imagegen.ImageGenerator(pull_image).prepare(_progress)
        imagegen.unload()
        print(f"\n{pull_image} is ready")
        return 0
    if pull:
        model = download(
            pull,
            lambda done, total: _progress(
                done / total, f"{format_size(done)} / {format_size(total)}"
            ),
        )
        print(f"\n{model.name} → {model.path}")
        return 0
    status = LocalLLM(settings.llm_model, settings.llm_auto_download).status()
    print(("OK: " if status.ok else "NOT READY: ") + status.message)
    local = list_local()
    print("\nAvailable offline:")
    for m in local:
        marker = "*" if m.name == settings.llm_model else " "
        print(f" {marker} {m.name:28s} {format_size(m.size):>8s}  ({m.source})")
    if not local:
        print("   (none)")
    names = {m.name for m in local}
    missing = {k: v for k, v in RECOMMENDED.items() if k not in names}
    if missing:
        print("\nRecommended (downloaded automatically when selected):")
        for name, note in missing.items():
            print(f"   {name:28s} {note}")
    image_status = imagegen.ImageGenerator(settings.image_model).status()
    print("\nImage models (pictures for --illustrations):")
    for name, spec in imagegen.IMAGE_MODELS.items():
        marker = "*" if name == settings.image_model else " "
        print(f" {marker} {name:28s} {spec.note}")
    print(("   OK: " if image_status.ok else "   NOT READY: ") + image_status.message)
    vision_status = vision.VisionLLM(settings.vision_model, settings.llm_auto_download).status()
    print("\nVision models (describe the book's images for --book-images):")
    for name, spec in vision.VISION_MODELS.items():
        marker = "*" if name == settings.vision_model else " "
        print(f" {marker} {name:28s} {spec.note}")
    print(("   OK: " if vision_status.ok else "   NOT READY: ") + vision_status.message)
    return 0 if status.ok else 1


def _cmd_script(doc: Document, settings: Settings, ctx: JobContext) -> Path:
    from .video.bookimages import prepare_book_images
    from .video.script import build_script

    language = detect_language(doc.chapters[0].text if doc.chapters else "")
    out_dir = doc.output_dir("video")
    out_dir.mkdir(parents=True, exist_ok=True)
    book_images = {}
    if settings.illustrations and settings.book_images and settings.video_script != "extractive":
        book_images = prepare_book_images(
            doc,
            out_dir,
            settings.vision_model,
            settings.llm_auto_download,
            ctx.progress,
            ctx.check,
        )
    script = build_script(
        doc,
        settings.video_script,
        language,
        settings.llm() if settings.video_script != "extractive" else None,
        on_progress=ctx.progress,
        check_cancel=ctx.check,
        illustrations=settings.illustrations,
        book_images=book_images,
    )
    path = out_dir / "script.json"
    script.save(path)
    print(file=sys.stderr)
    for section in script.sections:
        print(f"{section.chapter_title}  [{section.source}, {len(section.scenes)} scenes]")
        for scene in section.scenes:
            ill = scene.illustration
            extra = ""
            if ill is not None:
                what = ill.diagram.type if ill.diagram else ill.prompt[:50]
                if ill.book_image:
                    what = f"from the book, {ill.book_image}"
                extra = f"  [{ill.kind}: {what}, {ill.placement}, {ill.animation}]"
                if ill.book_image_reason:
                    extra += f"\n       ({ill.book_image_reason})"
            print(f"   - {scene.title}  ({len(scene.narration.split())} words){extra}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    settings = Settings.load()
    _apply_overrides(settings, args)

    if args.command == "voices":
        _cmd_voices(settings, args.language)
        return 0
    if args.command == "models":
        return _cmd_models(settings, args.pull, args.pull_image, args.pull_vision)

    try:
        doc = _load(args, settings)
    except (ParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "info":
        _cmd_info(doc, settings)
        return 0
    if args.command == "text":
        _cmd_text(doc, args.json)
        return 0

    ctx = JobContext(on_progress=_progress)
    try:
        if args.command == "script":
            print(_cmd_script(doc, settings, ctx))
            return 0
        if args.command == "audio":
            from .audiobook import make_audiobook

            result = make_audiobook(doc, settings, ctx)
        else:
            from .video.builder import make_video
            from .video.script import VideoScript

            script = None
            if args.reuse_script:
                path = doc.output_dir("video") / "script.json"
                if not path.exists():
                    print(f"error: {path} does not exist", file=sys.stderr)
                    return 2
                script = VideoScript.model_validate_json(path.read_text(encoding="utf-8"))
            result = make_video(doc, settings, ctx, script=script)
    except (KeyboardInterrupt, Cancelled):
        print("\ncancelled", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.debug("failed", exc_info=True)
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    print(file=sys.stderr)
    for path in result.files:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
