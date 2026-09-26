import json
import subprocess
from pathlib import Path

from pdf2video.audiobook import make_audiobook
from pdf2video.media import duration
from pdf2video.model import Chapter, Document
from pdf2video.settings import Settings
from pdf2video.video.builder import make_video


def _doc(tmp_path: Path, lorem: str) -> Document:
    src = tmp_path / "My Book.pdf"
    src.write_text("placeholder")
    return Document(
        path=src,
        title="My Book",
        author="Tester",
        chapters=[
            Chapter("Short: intro", [lorem]),  # ~25 words ≈ 8 s
            Chapter("Long one", [lorem * 3] * 10),  # ~750 words ≈ 250 s
        ],
    )


def test_audiobook_splits_long_chapters(tmp_path, lorem, silent_tts):
    doc = _doc(tmp_path, lorem)
    settings = Settings(max_chapter_minutes=1.5)
    result = make_audiobook(doc, settings, engine=silent_tts)
    out = tmp_path / "My Book_audio"
    assert result.out_dir == out
    names = sorted(p.name for p in out.iterdir())
    assert names[0] == "01 - Short intro.mp3"
    long_parts = [n for n in names if n.startswith("02 - Long one_")]
    assert len(long_parts) >= 3
    assert long_parts[0] == "02 - Long one_1.mp3"
    for p in out.iterdir():
        assert duration(p) <= 90.5
    tags = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=album,artist",
            "-of",
            "json",
            str(out / names[0]),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert json.loads(tags)["format"]["tags"]["album"] == "My Book"


def test_video_extractive(tmp_path, lorem, silent_tts):
    doc = _doc(tmp_path, lorem)
    doc.chapters = doc.chapters[:1]
    settings = Settings(video_script="extractive")
    result = make_video(doc, settings, engine=silent_tts)
    out = tmp_path / "My Book_video"
    assert [p.name for p in result.files] == ["01 - Short intro.mp4"]
    assert (out / "01 - Short intro.srt").read_text().startswith("1\n00:00:00,200")
    assert json.loads((out / "script.json").read_text())["sections"][0]["source"] == "extractive"
    assert duration(result.files[0]) > 8


def test_selected_chapters_keep_their_numbers(tmp_path, lorem, silent_tts):
    doc = _doc(tmp_path, lorem)
    doc.chapters.append(Chapter("Third", [lorem]))
    doc.number_chapters()
    selection = doc.select([1, 3])
    assert [c.number for c in selection.chapters] == [1, 3]
    assert selection.chapter_count == 3

    audio = make_audiobook(selection, Settings(), engine=silent_tts)
    assert [p.name for p in audio.files] == ["01 - Short intro.mp3", "03 - Third.mp3"]

    video = make_video(
        selection.select([3]), Settings(video_script="extractive"), engine=silent_tts
    )
    assert [p.name for p in video.files] == ["03 - Third.mp4"]
    script = json.loads((tmp_path / "My Book_video" / "script.json").read_text())
    assert [s["number"] for s in script["sections"]] == [3]
