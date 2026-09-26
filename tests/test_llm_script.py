import json

import pytest

from pdf2video.model import Chapter, Document
from pdf2video.video.script import SEGMENT_CHARS, ai_section, build_script

SCENES = {
    "scenes": [
        {"title": "Intro", "bullets": ["a", "b", " "], "narration": "Hello there."},
        {"title": "Empty", "bullets": ["x"], "narration": "  "},
    ]
}


class FakeLLM:
    """Stands in for pdf2video.llm.LocalLLM."""

    def __init__(self, reply=SCENES, fail=False, prepare_error=None):
        self.calls = []
        self.reply = reply
        self.fail = fail
        self.prepare_error = prepare_error
        self.prepared = False

    def prepare(self, on_progress=lambda f, m: None, check_cancel=lambda: None):
        if self.prepare_error:
            raise self.prepare_error
        on_progress(1.0, "loaded")
        self.prepared = True

    def chat_json(self, system, user, schema):
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self.fail:
            raise ConnectionError("boom")
        return json.dumps(self.reply)


def _doc(tmp_path, paragraphs):
    return Document(tmp_path / "x.pdf", "Doc", [Chapter("Ch", paragraphs)])


def test_ai_section_request_and_cleanup(tmp_path):
    llm = FakeLLM()
    doc = _doc(tmp_path, ["Some text."])
    section = ai_section(llm, doc.chapters[0], doc, "cs")
    assert section.source == "ai"
    assert [s.title for s in section.scenes] == ["Intro"]  # empty narration dropped
    assert section.scenes[0].bullets == ["a", "b"]
    call = llm.calls[0]
    assert call["schema"]["type"] == "object"
    assert "Czech" in call["system"]
    assert "Some text." in call["user"]


def test_long_chapter_is_split_into_segments_not_truncated(tmp_path):
    paragraph = "word " * 1000  # ~5000 chars
    doc = _doc(tmp_path, [paragraph.strip()] * 10)
    llm = FakeLLM()
    section = ai_section(llm, doc.chapters[0], doc, "en")
    assert len(llm.calls) >= 3
    sent = "".join(c["user"] for c in llm.calls)
    assert sent.count("word") == 10_000
    assert all(len(c["user"]) < SEGMENT_CHARS + 500 for c in llm.calls)
    assert len(section.scenes) == len(llm.calls)


def test_build_script_prepares_model_and_reports_progress(tmp_path):
    llm = FakeLLM()
    seen = []
    result = build_script(_doc(tmp_path, ["Text."]), "auto", "en", llm, lambda f, m: seen.append(f))
    assert llm.prepared
    assert result.sections[0].source == "ai"
    assert seen[-1] == pytest.approx(1.0)


def test_auto_mode_falls_back_to_extractive(tmp_path):
    result = build_script(_doc(tmp_path, ["Some text here."]), "auto", "en", FakeLLM(fail=True))
    assert result.sections[0].source == "extractive"
    missing = FakeLLM(prepare_error=RuntimeError("no model"))
    result = build_script(_doc(tmp_path, ["Some text here."]), "auto", "en", missing)
    assert result.sections[0].source == "extractive"


def test_ai_mode_raises(tmp_path):
    with pytest.raises(RuntimeError, match="Local model failed"):
        build_script(_doc(tmp_path, ["Some text here."]), "ai", "en", FakeLLM(fail=True))
    with pytest.raises(RuntimeError, match="unavailable: no model"):
        missing = FakeLLM(prepare_error=RuntimeError("no model"))
        build_script(_doc(tmp_path, ["Text."]), "ai", "en", missing)


def test_extractive_mode_never_touches_the_model(tmp_path):
    llm = FakeLLM()
    build_script(_doc(tmp_path, ["Text."]), "extractive", "en", llm)
    assert not llm.prepared and not llm.calls
