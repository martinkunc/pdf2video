import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from test_llm_script import FakeLLM

from pdf2video import imagegen
from pdf2video.model import Chapter, Document
from pdf2video.settings import Settings
from pdf2video.video.builder import make_video
from pdf2video.video.diagrams import LAYOUTS, render_diagram
from pdf2video.video.script import (
    Diagram,
    DiagramItem,
    Illustration,
    Scene,
    VideoScript,
    VideoSection,
    ai_section,
    clean_illustration,
)
from pdf2video.video.slides import illustration_box


def _items(n, value=None):
    return [DiagramItem(label=f"Step {i}", detail=f"detail {i}", value=value) for i in range(n)]


def _diagram(kind="flow", n=4, animation="reveal", placement="right"):
    value = 3.0 if kind == "bars" else None
    return Illustration(
        kind="diagram",
        placement=placement,
        animation=animation,
        diagram=Diagram(type=kind, items=_items(n, value)),
    )


# ------------------------------------------------------------ script / LLM


def test_clean_illustration_fixes_and_drops():
    assert clean_illustration(None) is None
    # A picture needs a real prompt; diagram-only animations are replaced.
    assert clean_illustration(Illustration(kind="picture", prompt="cat")) is None
    pic = clean_illustration(
        Illustration(kind="picture", prompt="  a  red\nfox in snow ", animation="reveal")
    )
    assert pic.prompt == "a red fox in snow" and pic.animation == "zoom_in"
    # A diagram needs its data and enough items.
    assert clean_illustration(Illustration(kind="diagram")) is None
    assert clean_illustration(_diagram("cycle", 2)) is None
    ok = clean_illustration(_diagram("flow", 9, animation="pan"))
    assert len(ok.diagram.items) == 7 and ok.animation == "reveal"
    # Bars need values.
    bars = _diagram("bars", 3)
    bars.diagram.items[0].value = None
    assert len(clean_illustration(bars).diagram.items) == 2
    bars.diagram.items[1].value = None
    assert clean_illustration(bars) is None


def test_ai_section_with_illustrations(tmp_path):
    reply = {
        "scenes": [
            {"title": "Intro", "bullets": ["a"], "narration": "Hello."},
            {
                "title": "Steps",
                "bullets": ["b"],
                "narration": "First this, then that.",
                "illustration": {
                    "kind": "diagram",
                    "placement": "full",
                    "animation": "reveal",
                    "diagram": {
                        "type": "flow",
                        "items": [{"label": "This"}, {"label": "That"}],
                    },
                },
            },
            {
                "title": "Fox",
                "bullets": ["c"],
                "narration": "A fox.",
                "illustration": {"kind": "picture", "prompt": "a red fox in the snow"},
            },
        ]
    }
    llm = FakeLLM(reply)
    doc = Document(tmp_path / "x.pdf", "Doc", [Chapter("Ch", ["Text."])])
    section = ai_section(llm, doc.chapters[0], doc, "en", illustrations=True)
    assert section.scenes[0].illustration is None
    assert section.scenes[1].illustration.diagram.type == "flow"
    assert section.scenes[2].illustration.kind == "picture"
    call = llm.calls[0]
    assert "Illustrations:" in call["system"]
    # The model must decide for every scene and every option (the answer above,
    # with missing fields, is still accepted).
    defs = call["schema"]["$defs"]
    assert "illustration" in defs["_AIIllustratedScene"]["required"]
    ill = defs["_AIIllustration"]
    assert set(ill["required"]) == {"kind", "placement", "animation", "prompt", "diagram"}
    assert "image" not in ill["properties"]


def test_at_most_60_percent_of_scenes_are_illustrated(tmp_path):
    diagram = {"kind": "diagram", "diagram": {"type": "flow", "items": [{"label": "A"}] * 3}}
    picture = {"kind": "picture", "prompt": "a red fox in the snow"}
    kinds = [diagram, picture, diagram, picture, diagram]
    reply = {
        "scenes": [
            {"title": f"S{k}", "bullets": ["x"], "narration": "Text.", "illustration": ill}
            for k, ill in enumerate(kinds)
        ]
    }
    doc = Document(tmp_path / "x.pdf", "Doc", [Chapter("Ch", ["Text."])])
    section = ai_section(FakeLLM(reply), doc.chapters[0], doc, "en", illustrations=True)
    kept = [s.illustration.kind if s.illustration else None for s in section.scenes]
    # 3 of 5 kept: the opening scene's goes first, then the pictures.
    assert kept == [None, "picture", "diagram", None, "diagram"]


def test_ai_section_without_illustrations_uses_plain_schema(tmp_path):
    llm = FakeLLM()
    doc = Document(tmp_path / "x.pdf", "Doc", [Chapter("Ch", ["Text."])])
    ai_section(llm, doc.chapters[0], doc, "en")
    assert "Illustrations:" not in llm.calls[0]["system"]
    assert "illustration" not in json.dumps(llm.calls[0]["schema"])


# ---------------------------------------------------------------- drawing


@pytest.mark.parametrize("kind", sorted(LAYOUTS))
@pytest.mark.parametrize("placement", ["right", "full"])
def test_every_diagram_type_renders_and_reveals(kind, placement):
    x0, y0, x1, y1 = illustration_box(placement)
    diagram = _diagram(kind, 5).diagram
    diagram.caption = "Caption"
    empty = render_diagram(diagram, (x1 - x0, y1 - y0), 0)
    one = render_diagram(diagram, (x1 - x0, y1 - y0), 1)
    full = render_diagram(diagram, (x1 - x0, y1 - y0))
    assert full.size == (x1 - x0, y1 - y0)

    def ink(img):
        return img.getchannel("A").point(lambda a: 255 if a else 0).histogram()[255]

    assert ink(empty) < ink(one) < ink(full)


# ------------------------------------------------------------- end to end


class FakeImageGenerator:
    calls: list[str] = []

    def __init__(self, model, auto_download=True):
        self.model = model

    def cache_key(self, prompt, aspect):
        return f"{abs(hash((prompt, round(aspect, 2)))):016x}"[:16]

    def prepare(self, on_progress=lambda f, m: None, check_cancel=lambda: None):
        on_progress(1.0, "loaded")

    def generate(self, prompt, aspect=1.0):
        FakeImageGenerator.calls.append(prompt)
        return Image.new("RGB", (round(256 * aspect), 256), (200, 120, 40))


def _decode_errors(path: Path) -> str:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    ).stderr


def test_video_with_illustrations(tmp_path, silent_tts, monkeypatch):
    monkeypatch.setattr(imagegen, "ImageGenerator", FakeImageGenerator)
    FakeImageGenerator.calls = []
    src = tmp_path / "Book.pdf"
    src.write_text("x")
    doc = Document(src, "Book", [Chapter("One", ["Some text here."])])
    doc.number_chapters()

    def scene(title, ill):
        return Scene(title=title, bullets=["a point"], narration="word " * 12, illustration=ill)

    script = VideoScript(
        title="Book",
        language="en",
        sections=[
            VideoSection(
                chapter_title="One",
                number=1,
                source="ai",
                scenes=[
                    scene("Reveal", _diagram("cycle", 4, "reveal", "full")),
                    scene("Fade", _diagram("comparison", 2, "fade_in")),
                    scene(
                        "Zoom", Illustration(kind="picture", prompt="a fox", animation="zoom_in")
                    ),
                    scene(
                        "Pan",
                        Illustration(
                            kind="picture", prompt="a fox", animation="pan", placement="full"
                        ),
                    ),
                    scene("Still", Illustration(kind="picture", prompt="a fox", placement="left")),
                    scene(
                        "FadePic", Illustration(kind="picture", prompt="a fox", animation="fade_in")
                    ),
                    scene("Plain", None),
                ],
            )
        ],
    )
    settings = Settings(illustrations=True)
    result = make_video(doc, settings, engine=silent_tts, script=script)
    video = result.files[0]
    assert _decode_errors(video) == ""
    out = tmp_path / "Book_video"
    pictures = sorted((out / "illustrations").glob("*.png"))
    assert len(pictures) == 2  # "right"/"left" share one picture; "full" has another aspect
    assert FakeImageGenerator.calls == ["a fox", "a fox"]
    saved = json.loads((out / "script.json").read_text())
    images = [
        s["illustration"] and s["illustration"]["image"] for s in saved["sections"][0]["scenes"]
    ]
    assert all(i.startswith("illustrations/") for i in images[2:5])

    # Re-rendering reuses the pictures.
    make_video(doc, settings, engine=silent_tts, script=script)
    assert len(FakeImageGenerator.calls) == 2


def test_illustrations_off_ignores_them(tmp_path, silent_tts, monkeypatch):
    monkeypatch.setattr(imagegen, "ImageGenerator", FakeImageGenerator)
    FakeImageGenerator.calls = []
    src = tmp_path / "Book.pdf"
    src.write_text("x")
    doc = Document(src, "Book", [Chapter("One", ["Some text here."])])
    doc.number_chapters()
    ill = Illustration(kind="picture", prompt="a fox", animation="zoom_in")
    script = VideoScript(
        title="Book",
        language="en",
        sections=[
            VideoSection(
                chapter_title="One",
                number=1,
                scenes=[Scene(title="T", bullets=["b"], narration="word " * 5, illustration=ill)],
            )
        ],
    )
    make_video(doc, Settings(illustrations=False), engine=silent_tts, script=script)
    assert FakeImageGenerator.calls == []
    assert not (tmp_path / "Book_video" / "illustrations").exists()


def test_image_model_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    spec = imagegen.resolve("z-image-turbo")
    assert imagegen.dimensions(spec, 1.0) == (1024, 1024)
    w, h = imagegen.dimensions(spec, 2.4)  # clamped to 2:1
    assert w % 64 == 0 and h % 64 == 0 and 1.9 < w / h < 2.2
    with pytest.raises(imagegen.ModelError):
        imagegen.resolve("nope")
    status = imagegen.ImageGenerator("sdxl-turbo", auto_download=False).status()
    assert not status.ok
    custom = tmp_path / "model.safetensors"
    custom.write_bytes(b"x")
    assert imagegen.resolve(str(custom)).files[0].name == str(custom)


def test_clean_illustration_drops_unusable_visuals():
    # Image models can't draw text: no pictures of flowcharts or charts…
    flowchart = Illustration(kind="picture", prompt="A flowchart of the Volere process steps")
    assert clean_illustration(flowchart) is None
    # …but asking for "no text, no charts" is fine.
    ok = Illustration(kind="picture", prompt="A busy office at dawn, no text, no charts")
    assert clean_illustration(ok) is not None
    # A comparison without traits would show empty panels.
    empty = _diagram("comparison", 2)
    empty.diagram.items[1].detail = ""
    assert clean_illustration(empty) is None
    assert clean_illustration(_diagram("comparison", 2)) is not None
