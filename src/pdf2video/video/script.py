"""Turn chapters into a video script: scenes with slide bullets and narration.

Two strategies:

* **AI** — a local LLM (built-in llama.cpp runner, see :mod:`pdf2video.llm`)
  writes a teaching-style explanation of each chapter.
* **Extractive** — offline; scenes are consecutive slices of the original text,
  bullets are the leading sentences of its paragraphs.

With ``illustrations`` on, the AI also decides which scenes need a visual and
whether it is a *diagram* (structured data drawn by :mod:`.diagrams`) or a
*picture* (a prompt for the local image model, see :mod:`pdf2video.imagegen`).
When the chapter has images of its own (summarised by :mod:`.bookimages`), the AI
then compares them with the pictures it would generate and may pick the book's
image instead (:func:`choose_book_images`).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from ..media import Cancelled

if TYPE_CHECKING:
    from ..llm import LocalLLM as LLM
    from .bookimages import BookImage
from ..model import Chapter, Document
from ..textutil import chunk_paragraphs, split_sentences

log = logging.getLogger(__name__)

SCENE_WORDS = 170
MAX_BULLETS = 4
BULLET_WORDS = 14

# Local models have small context windows: long chapters are scripted in segments
# of at most this many characters (never truncated).
SEGMENT_CHARS = 20_000

LANGUAGE_NAMES = {
    "en": "English", "cs": "Czech", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
}  # fmt: skip


DiagramType = Literal["flow", "cycle", "hierarchy", "hub", "comparison", "timeline", "bars"]
Placement = Literal["right", "left", "full"]
Animation = Literal["none", "fade_in", "reveal", "zoom_in", "zoom_out", "pan"]
PICTURE_ANIMATIONS = ("none", "fade_in", "zoom_in", "zoom_out", "pan")
DIAGRAM_ANIMATIONS = ("none", "fade_in", "reveal")
MAX_DIAGRAM_ITEMS = 7
MAX_ILLUSTRATED = 0.6  # at most this share of a segment's scenes get an illustration


class DiagramItem(BaseModel):
    label: str
    detail: str = ""
    value: float | None = None  # bars only


class Diagram(BaseModel):
    type: DiagramType
    caption: str = ""
    items: list[DiagramItem]


class Illustration(BaseModel):
    kind: Literal["diagram", "picture"]
    placement: Placement = "right"
    animation: Animation = "none"
    prompt: str = ""  # picture: English description for the image model
    diagram: Diagram | None = None
    image: SkipJsonSchema[str] = (
        ""  # picture: the file shown, relative to the video folder (informational)
    )
    # picture: an image from the book shown instead of a generated one (relative to
    # the video folder), and why the AI chose it or kept the generated picture
    book_image: SkipJsonSchema[str] = ""
    book_image_reason: SkipJsonSchema[str] = ""


class SceneText(BaseModel):
    title: str = Field(description="Short slide heading (max ~8 words)")
    bullets: list[str] = Field(description="2-4 concise key points shown on the slide")
    narration: str = Field(description="What the narrator says while the slide is shown")


class Scene(SceneText):
    illustration: Illustration | None = None


class VideoSection(BaseModel):
    chapter_title: str
    number: int = 0  # chapter number in the full document
    scenes: list[Scene]
    source: str = "extractive"  # "ai" | "extractive"


class VideoScript(BaseModel):
    title: str
    language: str
    sections: list[VideoSection]

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


class _AIScenes(BaseModel):
    scenes: list[SceneText]


class _AIIllustration(Illustration):
    # Every choice is required, so that the model makes it explicitly.
    placement: Placement
    animation: Animation
    prompt: str
    diagram: Diagram | None


class _AIIllustratedScene(SceneText):
    # Required (but nullable) so that the model decides for every scene.
    illustration: _AIIllustration | None


class _AIIllustratedScenes(BaseModel):
    scenes: list[_AIIllustratedScene]


class _ParsedIllustratedScenes(BaseModel):
    # The answer is parsed leniently: a missing field must not discard the segment.
    scenes: list[Scene]


# ---------------------------------------------------------------- extractive


def _shorten(sentence: str, words: int = BULLET_WORDS) -> str:
    parts = sentence.split()
    if len(parts) <= words:
        return sentence.rstrip()
    return " ".join(parts[:words]).rstrip(",;:—-") + "…"


def extractive_section(chapter: Chapter) -> VideoSection:
    # Split into sentence-aligned slices of about SCENE_WORDS words.
    slices: list[list[str]] = [[]]
    count = 0
    for paragraph in chapter.paragraphs:
        units = [paragraph] if len(paragraph.split()) <= SCENE_WORDS else split_sentences(paragraph)
        for i, unit in enumerate(units):
            if count >= SCENE_WORDS * 0.7 and slices[-1]:
                slices.append([])
                count = 0
            # Mark paragraph starts so we can pick bullet sentences from them.
            slices[-1].append(("¶" if i == 0 else "") + unit)
            count += len(unit.split())
    scenes: list[Scene] = []
    total = len([s for s in slices if s])
    for n, units in enumerate((s for s in slices if s), start=1):
        paragraph_starts = [u[1:] for u in units if u.startswith("¶")] or [units[0].lstrip("¶")]
        bullets = []
        for p in paragraph_starts[:MAX_BULLETS]:
            sentences = split_sentences(p)
            if sentences:
                bullets.append(_shorten(sentences[0]))
        narration = " ".join(u.lstrip("¶") for u in units)
        title = chapter.title if total == 1 else f"{chapter.title} ({n}/{total})"
        scenes.append(Scene(title=title, bullets=bullets, narration=narration))
    return VideoSection(chapter_title=chapter.title, scenes=scenes, source="extractive")


# ---------------------------------------------------------------------- AI


SYSTEM_PROMPT = """\
You are the scriptwriter for short explanatory videos made of narrated slides.
You are given (part of) one chapter of a document. Write a script that *explains* it
to a curious, intelligent viewer who has not read it: cover the key ideas, arguments,
facts and examples faithfully, in a logical order, and make the reasoning clear.
Do not invent facts that are not supported by the text.

Rules:
- Produce between 2 and 8 scenes, proportional to how much substance the text has.
- Each scene has a short slide title, 2-4 terse bullet points (max ~10 words each),
  and narration of roughly 60-150 words that is meant to be spoken aloud.
- The narration must read naturally when heard: no markdown, no lists, no references
  to "this slide", no URLs; spell out symbols where a listener would need it.
- Write everything (titles, bullets, narration) in {language}.
- Answer with JSON only, matching the requested schema."""

ILLUSTRATION_RULES = """

Illustrations:
- Illustrations never replace scenes: first plan the scenes exactly as you would
  without them (the number the text's substance deserves, often one per main idea
  or paragraph), then decide for each scene whether it needs an illustration.
- A scene may get an "illustration", but only where a visual genuinely helps the
  viewer understand: about half of the scenes at most. Set it to null for the
  others, especially for introductions and recaps.
- Choose "kind":
  - "diagram" for abstract or structured ideas: processes and steps, cause and
    effect, cycles, hierarchies and classifications, a concept and its aspects,
    comparisons, timelines, numbers and proportions. Diagrams are drawn precisely
    with readable labels, so prefer them whenever the idea has parts and relations.
  - "picture" for concrete, visual subjects: physical objects, places, organisms,
    historical scenes, people at work, or a vivid metaphor. A picture cannot
    contain any readable text, so never ask for a picture of a flowchart, chart,
    infographic, labelled map, document or screen: use a diagram for those.
- For a diagram, fill "diagram" with "type", "items" and optionally "caption":
  - flow: ordered steps (2-6 items)
  - cycle: steps that repeat in a loop, where the last leads back to the first (3-6)
  - hierarchy: the first item is the whole, the others its parts or kinds (3-7)
  - hub: the first item is the central concept, the others related aspects (3-7)
  - comparison: 2-3 alternatives side by side; every item needs its key traits
    in "detail", separated by semicolons
  - timeline: events in time order (2-6); "label" is the date, "detail" the event
  - bars: quantities (2-7); every item needs a numeric "value" taken from the
    text; put the unit in "caption"
  Item labels have 1-4 words, details at most ~12 words. Use only facts from the
  text. Diagram text is in {language}.
- For a picture, write "prompt": an English description for an image generator:
  subject, setting, composition, mood; 20-50 words; no text, labels or letters.
  Set "diagram" to null. For a diagram, set "prompt" to "".
- "placement": "right" (next to the bullets; the default), "left", or "full"
  (large, under the title, replacing the bullets; good for timelines, wide flows
  and key pictures).
- "animation": for a diagram "reveal" (items appear one by one while the
  narration explains them; best for flows, cycles and timelines) or "fade_in";
  for a picture "zoom_in", "zoom_out", "pan" (slow camera motion) or "fade_in";
  or "none"."""


def _segments(chapter: Chapter) -> list[str]:
    return [c.text for c in chunk_paragraphs(chapter.paragraphs, SEGMENT_CHARS)] or [""]


# Pictures of these are mostly text, which image models can't draw.
_TEXT_PICTURE = re.compile(
    r"\b(flow ?charts?|diagrams?|charts?|graphs?|infographics?|spreadsheets?|mind ?maps?"
    r"|timelines?|screenshots?|user interfaces?)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(no|without)\b[^,.;]*", re.IGNORECASE)  # "no text or charts"


def clean_illustration(ill: Illustration | None) -> Illustration | None:
    """Drop illustrations the model got wrong and fix incompatible options."""
    if ill is None:
        return None
    ill = ill.model_copy(deep=True)
    ill.image = ill.book_image = ill.book_image_reason = ""
    if ill.kind == "picture":
        ill.prompt = " ".join(ill.prompt.split())
        ill.diagram = None
        if len(ill.prompt.split()) < 3 or _TEXT_PICTURE.search(_NEGATION.sub("", ill.prompt)):
            return None  # the image model would draw gibberish text
        if ill.animation not in PICTURE_ANIMATIONS:
            ill.animation = "zoom_in"
        return ill
    diagram = ill.diagram
    if diagram is None:
        return None
    items = [
        DiagramItem(label=i.label.strip(), detail=i.detail.strip(), value=i.value)
        for i in diagram.items
        if i.label.strip()
    ][:MAX_DIAGRAM_ITEMS]
    if diagram.type == "bars":
        items = [i for i in items if i.value is not None and i.value >= 0]
        if len(items) < 2 or not any(i.value for i in items):
            return None
    if diagram.type == "comparison" and not all(i.detail for i in items):
        return None  # without traits the panels would be empty
    if len(items) < (3 if diagram.type in ("cycle", "hierarchy", "hub") else 2):
        return None
    ill.prompt = ""
    ill.diagram = Diagram(type=diagram.type, caption=diagram.caption.strip(), items=items)
    if ill.animation not in DIAGRAM_ANIMATIONS:
        ill.animation = "reveal"
    return ill


def _chat_scenes(llm: LLM, system: str, user: str, illustrations: bool = False) -> list[Scene]:
    schema = (_AIIllustratedScenes if illustrations else _AIScenes).model_json_schema()
    content = llm.chat_json(system, user, schema)
    parsed = (_ParsedIllustratedScenes if illustrations else _AIScenes).model_validate_json(content)
    scenes = []
    for scene in parsed.scenes:
        if not scene.narration.strip():
            continue
        bullets = [b.strip() for b in scene.bullets if b.strip()][:MAX_BULLETS]
        scenes.append(
            Scene(
                title=scene.title.strip(),
                bullets=bullets,
                narration=scene.narration.strip(),
                illustration=clean_illustration(getattr(scene, "illustration", None)),
            )
        )
    if not scenes:
        raise ValueError("the model returned an empty script")
    _limit_illustrations(scenes)
    return scenes


def _limit_illustrations(scenes: list[Scene]) -> None:
    """Keep at most MAX_ILLUSTRATED of the scenes illustrated. The opening scene's
    illustration goes first, then pictures, then the latest diagrams."""
    illustrated = [k for k, s in enumerate(scenes) if s.illustration is not None]
    excess = len(illustrated) - max(1, math.ceil(len(scenes) * MAX_ILLUSTRATED))
    if excess <= 0:
        return

    def priority(k: int) -> tuple[bool, bool, int]:
        kind = scenes[k].illustration.kind
        return (k != 0 or len(scenes) == 1, kind == "diagram", -k)

    for k in sorted(illustrated, key=priority)[:excess]:
        scenes[k].illustration = None


def ai_section(
    llm: LLM,
    chapter: Chapter,
    doc: Document,
    language: str,
    on_segment: Callable[[], None] = lambda: None,
    check_cancel: Callable[[], None] = lambda: None,
    illustrations: bool = False,
) -> VideoSection:
    prompt = SYSTEM_PROMPT + (ILLUSTRATION_RULES if illustrations else "")
    system = prompt.format(language=LANGUAGE_NAMES.get(language, language))
    segments = _segments(chapter)
    scenes: list[Scene] = []
    for k, segment in enumerate(segments, start=1):
        check_cancel()
        if len(segments) == 1:
            position = (
                "This is the whole chapter. "
                "Start by introducing its topic and end with a short recap."
            )
        else:
            position = (
                f"This is part {k} of {len(segments)} of the chapter. "
                + (
                    "Start by introducing the chapter's topic."
                    if k == 1
                    else "Continue the explanation; do not re-introduce the chapter."
                )
                + (" End with a short recap of the whole chapter." if k == len(segments) else "")
            )
        user = (
            f"Document: {doc.title}\nChapter: {chapter.title}\n{position}\n\n"
            f"<text>\n{segment}\n</text>\n\nWrite the explanatory video script as JSON."
        )
        for attempt in (1, 2):
            try:
                scenes.extend(_chat_scenes(llm, system, user, illustrations))
                break
            except ValueError:  # includes pydantic.ValidationError (malformed JSON)
                if attempt == 2:
                    raise
        on_segment()
    return VideoSection(chapter_title=chapter.title, scenes=scenes, source="ai")


BOOK_IMAGE_PROMPT = """\
You choose the pictures for an explanatory video about one chapter of a document.
For some scenes an image generator will paint a new picture from a description.
The chapter also contains its own images; you get a summary of each.
For every listed scene, compare the planned generated picture with the chapter's
images and decide which one serves the viewer better:
- Choose a book image (its number) when it shows the scene's subject or directly
  illustrates what the narration explains. The original is authentic and is what
  the text refers to, so prefer it when it fits well.
- Keep the generated picture ("book_image": null) when no book image fits the
  scene, or when the fitting one is mostly text or a table that would be
  unreadable on a slide.
- Use each book image at most once.
- "reason": one short sentence explaining the choice.
Answer with JSON only, matching the requested schema."""


class _BookChoice(BaseModel):
    scene: int
    book_image: int | None
    reason: str


class _BookChoices(BaseModel):
    choices: list[_BookChoice]


def show_mentioned_figures(section: VideoSection, images: list[BookImage]) -> None:
    """A scene that refers to a figure of the book ("see Figure 2.1") shows that figure,
    instead of any other illustration."""
    from .bookimages import figure_refs

    figures: dict[str, BookImage] = {}
    for image in images:
        if image.figure:
            figures.setdefault(image.figure, image)
    if not figures:
        return
    for scene in section.scenes:
        text = " ".join([scene.title, *scene.bullets, scene.narration])
        image = next((figures[r] for r in figure_refs(text) if r in figures), None)
        if image is None:
            continue
        old = scene.illustration
        if old is not None and old.book_image == image.file:
            continue  # already shown (e.g. a reused script)
        wide = image.width >= 1.4 * image.height
        scene.illustration = Illustration(
            kind="picture",
            placement="full" if wide else "right",
            animation="fade_in",
            prompt=old.prompt if old is not None and old.kind == "picture" else "",
            book_image=image.file,
            book_image_reason=f"The scene refers to {image.figure.capitalize()}.",
        )


def choose_book_images(
    llm: LLM, section: VideoSection, doc: Document, images: list[BookImage]
) -> None:
    """Let the AI swap generated pictures for the chapter's own images where they fit better.

    Scenes that already show a book image (see :func:`show_mentioned_figures`) and
    the images they show are left out; so are images that are mostly text.
    """
    shown = {s.illustration.book_image for s in section.scenes if s.illustration is not None}
    images = [i for i in images if i.suitable and i.file not in shown]
    scenes = {
        k: s
        for k, s in enumerate(section.scenes, start=1)
        if s.illustration is not None
        and s.illustration.kind == "picture"
        and not s.illustration.book_image
    }
    if not scenes or not images:
        return
    listing = "\n".join(f"[{n}] {image.summary()}" for n, image in enumerate(images, start=1))
    planned = "\n\n".join(
        f"Scene {k}: {s.title}\nNarration: {_shorten(s.narration, 90)}\n"
        f"Generated picture: {s.illustration.prompt}"
        for k, s in scenes.items()
    )
    user = (
        f"Document: {doc.title}\nChapter: {section.chapter_title}\n\n"
        f"Images in the chapter:\n{listing}\n\nScenes that get a picture:\n{planned}\n\n"
        "Decide for every scene, as JSON."
    )
    content = llm.chat_json(BOOK_IMAGE_PROMPT, user, _BookChoices.model_json_schema())
    used: set[int] = set()
    for choice in _BookChoices.model_validate_json(content).choices:
        scene = scenes.get(choice.scene)
        if scene is None or scene.illustration.book_image_reason:
            continue  # unknown scene, or already decided
        ill = scene.illustration
        ill.book_image_reason = " ".join(choice.reason.split()) or "-"
        number = choice.book_image
        if number is None:
            continue
        if not 1 <= number <= len(images):
            ill.book_image_reason = f"Kept the generated picture: there is no book image {number}."
        elif number in used:
            ill.book_image_reason = (
                f"Kept the generated picture: book image {number} is shown in another scene."
            )
        else:
            used.add(number)
            ill.book_image = images[number - 1].file


def count_ai_steps(doc: Document) -> int:
    return sum(len(_segments(ch)) for ch in doc.chapters)


def build_script(
    doc: Document,
    mode: str,
    language: str,
    llm: LLM | None = None,
    on_progress: Callable[[float, str], None] = lambda fraction, message: None,
    check_cancel: Callable[[], None] = lambda: None,
    illustrations: bool = False,
    book_images: dict[int, list[BookImage]] | None = None,
) -> VideoScript:
    """``mode``: "ai", "extractive" or "auto" (AI when the local model is usable).

    ``illustrations``: let the AI mark scenes that need a diagram or picture
    (ignored for the extractive script).

    ``book_images``: chapter number → the chapter's own (summarised) images; with
    illustrations on, a scene that mentions one of the book's figures shows it, and
    the AI may show one of them instead of a generated picture.

    ``llm`` must provide ``prepare(on_progress, check_cancel)`` (download + load) and
    ``chat_json(system, user, schema)``. In "auto" mode, a model that cannot be
    prepared or a chapter whose AI script fails falls back to the extractive
    script; in "ai" mode those failures are raised.
    """
    if mode == "extractive":
        llm = None
    if llm is not None:
        try:
            # First 20 % of the progress: download (if needed) and load the model.
            llm.prepare(lambda f, m: on_progress(0.2 * f, m), check_cancel)
        except Cancelled:
            raise
        except Exception as exc:
            if mode == "ai":
                raise RuntimeError(f"Local model unavailable: {exc}") from exc
            log.warning("Local model unavailable (%s); using the extractive script", exc)
            llm = None

    total = count_ai_steps(doc) if llm else len(doc.chapters)
    done = 0

    def step() -> None:
        nonlocal done
        done = min(total, done + 1)
        on_progress(0.2 + 0.8 * done / total, f"Writing script: step {done}/{total}")

    sections: list[VideoSection] = []
    for chapter in doc.chapters:
        check_cancel()
        section = None
        if llm is not None:
            try:
                section = ai_section(llm, chapter, doc, language, step, check_cancel, illustrations)
            except Cancelled:
                raise
            except Exception as exc:
                if mode == "ai":
                    raise RuntimeError(f"Local model failed on “{chapter.title}”: {exc}") from exc
                log.warning("AI script failed for %r (%s); using extractive", chapter.title, exc)
        if section is None:
            section = extractive_section(chapter)
            step()
        elif illustrations and (images := (book_images or {}).get(chapter.number)):
            check_cancel()
            show_mentioned_figures(section, images)
            try:
                choose_book_images(llm, section, doc, images)
            except Cancelled:
                raise
            except Exception as exc:  # the generated pictures are kept
                log.warning("Choosing book images failed for %r: %s", chapter.title, exc)
        section.number = chapter.number
        sections.append(section)
    return VideoScript(title=doc.title, language=language, sections=sections)
