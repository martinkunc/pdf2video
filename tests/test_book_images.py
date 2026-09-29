import io
import json

from PIL import Image
from test_illustrations import FakeImageGenerator, _decode_errors
from test_llm_script import FakeLLM

from pdf2video import imagegen
from pdf2video.llm import vision
from pdf2video.model import Chapter, ChapterImage, Document
from pdf2video.settings import Settings
from pdf2video.video import bookimages
from pdf2video.video.bookimages import BookImage, prepare_book_images
from pdf2video.video.builder import make_video
from pdf2video.video.script import (
    Diagram,
    DiagramItem,
    Illustration,
    Scene,
    VideoScript,
    VideoSection,
    build_script,
    choose_book_images,
)


def _png(color="red", size=(300, 200)) -> bytes:
    data = io.BytesIO()
    Image.new("RGB", size, color).save(data, "PNG")
    return data.getvalue()


def _book(tmp_path, images):
    src = tmp_path / "Book.pdf"
    src.write_text("x")
    return Document(src, "Book", [Chapter("One", ["Some text here."], images)])


class FakeVision:
    calls: list[str] = []
    kinds = {"red": "photo", "blue": "text"}

    def __init__(self, model, auto_download=True):
        self.model = model

    def prepare(self, on_progress=lambda f, m: None, check_cancel=lambda: None):
        on_progress(1.0, "loaded")

    def describe(self, image, caption=""):
        color = {(255, 0, 0): "red", (0, 0, 255): "blue"}[
            Image.open(io.BytesIO(image)).convert("RGB").getpixel((0, 0))
        ]
        FakeVision.calls.append(color)
        return vision.ImageDescription(kind=self.kinds[color], description=f"A {color} fox.")


def test_images_are_saved_described_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "VisionLLM", FakeVision)
    FakeVision.calls = []
    doc = _book(
        tmp_path,
        [
            ChapterImage(_png("red"), "Figure 1: A fox"),
            ChapterImage(_png("red"), "duplicate"),
            ChapterImage(_png("blue")),
            ChapterImage(_png("red", (50, 50))),  # too small
        ],
    )
    out = tmp_path / "out"
    images = prepare_book_images(doc, out, "qwen3-vl-4b", True)[1]
    assert [i.caption for i in images] == ["Figure 1: A fox", ""]
    assert images[0].description == "A red fox." and images[0].kind == "photo"
    # The page of text ("blue") is kept (a scene may mention it) but not offered to the AI.
    assert [i.suitable for i in images] == [True, False]
    assert images[0].figure == "figure 1" and images[1].figure == ""
    assert sorted(FakeVision.calls) == ["blue", "red"]
    files = sorted(p.name for p in (out / "book_images").iterdir())
    assert len(files) == 2 and all(f.startswith("01-") and f.endswith(".png") for f in files)
    catalog = json.loads((out / "book_images.json").read_text())
    assert {e["kind"] for e in catalog} == {"photo", "text"}

    # A second run reuses the summaries.
    again = prepare_book_images(doc, out, "qwen3-vl-4b", True)
    assert len(FakeVision.calls) == 2
    assert again[1][0].description == "A red fox."


def test_unavailable_vision_model_keeps_captions(tmp_path, monkeypatch):
    class Broken(FakeVision):
        def prepare(self, on_progress=lambda f, m: None, check_cancel=lambda: None):
            raise vision.ModelError("not downloaded")

    monkeypatch.setattr(vision, "VisionLLM", Broken)
    doc = _book(
        tmp_path, [ChapterImage(_png("red"), "Figure 1: A fox"), ChapterImage(_png("blue"))]
    )
    images = prepare_book_images(doc, tmp_path / "out", "qwen3-vl-4b", True)[1]
    # Without a description, only the captioned image can be judged.
    assert [(i.caption, i.kind, i.suitable) for i in images] == [
        ("Figure 1: A fox", "", True),
        ("", "", False),
    ]


def _picture_scene(title, prompt):
    return Scene(
        title=title,
        bullets=["a point"],
        narration="word " * 12,
        illustration=Illustration(kind="picture", prompt=prompt, animation="zoom_in"),
    )


def _images():
    return [
        BookImage(
            file=f"book_images/01-{n}.png",
            chapter=1,
            sha256=str(n),
            width=300,
            height=200,
            kind="photo",
            description=d,
        )
        for n, d in enumerate(["A red fox in snow.", "A map of Europe."], start=1)
    ]


def test_choose_book_images(tmp_path):
    section = VideoSection(
        chapter_title="One",
        scenes=[
            Scene(title="Intro", bullets=[], narration="Hello."),
            _picture_scene("Fox", "a red fox in the snow"),
            _picture_scene("Tree", "an old oak tree"),
            _picture_scene("Again", "another fox"),
        ],
    )
    reply = {
        "choices": [
            {"scene": 2, "book_image": 1, "reason": "The book's photo shows the fox."},
            {"scene": 3, "book_image": None, "reason": "No book image shows a tree."},
            {"scene": 4, "book_image": 1, "reason": "Also a fox."},  # already used
            {"scene": 1, "book_image": 2, "reason": "Not a picture scene."},
            {"scene": 2, "book_image": 2, "reason": "A second answer is ignored."},
        ]
    }
    llm = FakeLLM(reply)
    doc = Document(tmp_path / "x.pdf", "Doc", [Chapter("One", ["Text."])])
    choose_book_images(llm, section, doc, _images())
    ills = [s.illustration for s in section.scenes]
    assert ills[0] is None
    assert ills[1].book_image == "book_images/01-1.png"
    assert ills[1].book_image_reason == "The book's photo shows the fox."
    assert ills[2].book_image == "" and ills[2].book_image_reason.startswith("No book image")
    assert ills[3].book_image == "" and "shown in another scene" in ills[3].book_image_reason
    user = llm.calls[0]["user"]
    assert "[1] photo: A red fox in snow." in user and "[2] photo: A map of Europe." in user
    assert "Scene 2: Fox" in user and "Generated picture: an old oak tree" in user
    assert "Scene 1:" not in user
    assert "book_image" in llm.calls[0]["schema"]["$defs"]["_BookChoice"]["required"]


class ScriptedLLM(FakeLLM):
    """Answers the script request first, then the book-image choice."""

    def __init__(self, *replies):
        super().__init__()
        self.replies = list(replies)

    def chat_json(self, system, user, schema):
        self.calls.append({"system": system, "user": user, "schema": schema})
        return json.dumps(self.replies.pop(0))


def test_build_script_offers_book_images(tmp_path):
    script_reply = {
        "scenes": [
            {"title": "Intro", "bullets": ["a"], "narration": "Hello."},
            {
                "title": "Fox",
                "bullets": ["b"],
                "narration": "A fox.",
                "illustration": {"kind": "picture", "prompt": "a red fox in the snow"},
            },
        ]
    }
    choice = {"choices": [{"scene": 2, "book_image": 1, "reason": "It is the same fox."}]}
    llm = ScriptedLLM(script_reply, choice)
    doc = Document(tmp_path / "x.pdf", "Doc", [Chapter("One", ["Text."])])
    script = build_script(doc, "ai", "en", llm, illustrations=True, book_images={1: _images()})
    assert script.sections[0].scenes[1].illustration.book_image == "book_images/01-1.png"
    assert len(llm.calls) == 2

    # Without illustrations the book images are not considered.
    llm = ScriptedLLM({"scenes": script_reply["scenes"][:1]})
    build_script(doc, "ai", "en", llm, book_images={1: _images()})
    assert len(llm.calls) == 1


def test_build_script_keeps_generated_pictures_if_choice_fails(tmp_path):
    script_reply = {
        "scenes": [
            {
                "title": "Fox",
                "bullets": ["b"],
                "narration": "A fox.",
                "illustration": {"kind": "picture", "prompt": "a red fox in the snow"},
            }
        ]
    }
    llm = ScriptedLLM(script_reply, {"choices": "garbage"})
    doc = Document(tmp_path / "x.pdf", "Doc", [Chapter("One", ["Text."])])
    script = build_script(doc, "ai", "en", llm, illustrations=True, book_images={1: _images()})
    ill = script.sections[0].scenes[0].illustration
    assert ill.kind == "picture" and ill.book_image == ""


def test_video_shows_the_chosen_book_image(tmp_path, silent_tts, monkeypatch):
    monkeypatch.setattr(imagegen, "ImageGenerator", FakeImageGenerator)
    FakeImageGenerator.calls = []
    doc = _book(tmp_path, [ChapterImage(_png("red"), "Figure 1: A fox")])
    doc.number_chapters()
    out = tmp_path / "Book_video"
    [saved] = bookimages.collect(doc, out)
    from_book = _picture_scene("Fox", "a red fox in the snow")
    from_book.illustration.book_image = saved.file
    missing = _picture_scene("Oak", "an old oak tree")
    missing.illustration.book_image = "book_images/gone.png"  # falls back to generating
    script = VideoScript(
        title="Book",
        language="en",
        sections=[
            VideoSection(
                chapter_title="One",
                number=1,
                source="ai",
                scenes=[from_book, missing, Scene(title="Plain", bullets=["x"], narration="Hi.")],
            )
        ],
    )
    result = make_video(doc, Settings(illustrations=True), engine=silent_tts, script=script)
    assert _decode_errors(result.files[0]) == ""
    assert FakeImageGenerator.calls == ["an old oak tree"]
    scenes = json.loads((out / "script.json").read_text())["sections"][0]["scenes"]
    assert scenes[0]["illustration"]["image"] == saved.file
    assert scenes[1]["illustration"]["image"].startswith("illustrations/")


def test_figure_refs():
    from pdf2video.video.bookimages import figure_refs

    text = "As Figure 2.1 shows (see also fig. 3 and Table 4-2), unlike Obr. 5; figures 6.1."
    assert figure_refs(text) == ["figure 2.1", "figure 3", "table 4.2", "figure 5", "figure 6.1"]
    assert figure_refs("A configuration of tables") == []


def _figure(n, caption, size=(300, 200)):
    return BookImage(
        file=f"book_images/02-{n}.png",
        chapter=2,
        sha256=str(n),
        width=size[0],
        height=size[1],
        caption=caption,
        kind="text",  # even an image the AI would not be offered
    )


def test_mentioned_figures_are_shown():
    from pdf2video.video.script import show_mentioned_figures

    diagram = Illustration(
        kind="diagram",
        diagram=Diagram(type="flow", items=[DiagramItem(label="a"), DiagramItem(label="b")]),
    )
    section = VideoSection(
        chapter_title="Two",
        scenes=[
            Scene(title="Map", bullets=[], narration="Figure 2.1 maps the process.",
                  illustration=diagram),
            Scene(title="Card", bullets=["See Table 2"], narration="The snow card."),
            _picture_scene("Fox", "a red fox"),
            Scene(title="Other", bullets=[], narration="Figure 9.9 is not in this chapter."),
        ],
    )  # fmt: skip
    images = [
        _figure(1, "Figure 2.1 This map of the process", (900, 500)),
        _figure(2, "Table 2 The snow card", (300, 400)),
        _figure(3, "A photo without a number"),
    ]
    show_mentioned_figures(section, images)
    ills = [s.illustration for s in section.scenes]
    assert ills[0].kind == "picture" and ills[0].book_image == "book_images/02-1.png"
    assert ills[0].placement == "full" and ills[0].book_image_reason.endswith("Figure 2.1.")
    assert ills[1].book_image == "book_images/02-2.png" and ills[1].placement == "right"
    assert ills[2].book_image == "" and ills[3] is None


def test_choose_skips_shown_and_text_images(tmp_path):
    shown = _picture_scene("Map", "a map")
    shown.illustration.book_image = "book_images/01-1.png"
    section = VideoSection(chapter_title="One", scenes=[shown, _picture_scene("Fox", "a fox")])
    images = _images()
    images[1].kind = "text"
    llm = FakeLLM({"choices": []})
    choose_book_images(llm, section, Document(tmp_path / "x.pdf", "Doc", []), images)
    assert llm.calls == []  # image 1 is shown already and image 2 is text: nothing to offer


def test_reused_script_shows_mentioned_figures(tmp_path, silent_tts, monkeypatch):
    monkeypatch.setattr(imagegen, "ImageGenerator", FakeImageGenerator)
    FakeImageGenerator.calls = []
    doc = _book(tmp_path, [ChapterImage(_png("red"), "Figure 1.1 A fox")])
    doc.number_chapters()
    scene = _picture_scene("Fox", "a painted fox in a forest")
    scene.narration = "The fox in Figure 1.1 hunts at night."
    script = VideoScript(
        title="Book",
        language="en",
        sections=[
            VideoSection(
                chapter_title="One",
                number=1,
                source="ai",
                scenes=[scene],
            )
        ],
    )
    make_video(doc, Settings(illustrations=True), engine=silent_tts, script=script)
    assert FakeImageGenerator.calls == []  # the book's figure is shown instead
    saved = json.loads((tmp_path / "Book_video" / "script.json").read_text())
    ill = saved["sections"][0]["scenes"][0]["illustration"]
    assert ill["book_image"].startswith("book_images/01-") and ill["image"] == ill["book_image"]
