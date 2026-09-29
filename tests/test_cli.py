import json

import pytest

from pdf2video.cli import main, parse_chapter_selection


def test_chapter_selection():
    assert parse_chapter_selection("1,3-4", 5) == [0, 2, 3]
    assert parse_chapter_selection("2,2", 3) == [1]
    with pytest.raises(ValueError):
        parse_chapter_selection("4", 3)


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    f = tmp_path / "book.md"
    f.write_text("# Book\n\n## One\n\nFirst text.\n\n## Two\n\nSecond text.\n")
    return f


def test_info(book, capsys):
    assert main(["info", str(book)]) == 0
    out = capsys.readouterr().out
    assert "2 chapters" in out and "One" in out and "Two" in out


def test_text_json_with_selection(book, capsys):
    assert main(["text", str(book), "--json", "--chapters", "2"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [c["title"] for c in data["chapters"]] == ["Two"]
    assert data["chapters"][0]["paragraphs"] == ["Second text."]


def test_script_extractive(book, capsys):
    assert main(["script", str(book), "--script-mode", "extractive"]) == 0
    script = json.loads((book.parent / "book_video" / "script.json").read_text())
    assert [s["chapter_title"] for s in script["sections"]] == ["One", "Two"]


def test_bad_file(tmp_path, capsys):
    assert main(["info", str(tmp_path / "missing.pdf")]) == 2
    assert "not found" in capsys.readouterr().err


def test_chapter_selection_keeps_numbers(book, capsys):
    assert main(["info", str(book), "--chapters", "2"]) == 0
    assert "  2. Two" in capsys.readouterr().out


def test_settings_round_trip(tmp_path, monkeypatch):
    from pdf2video.settings import Settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert Settings().tts_engine == "kokoro"
    Settings(illustrations=True, voice='a "b"', rate=1.2).save()
    loaded = Settings.load()
    assert loaded.illustrations and loaded.voice == 'a "b"' and loaded.rate == 1.2
    # Files written by older versions (Python booleans) still load.
    path = tmp_path / "pdf2video" / "settings.toml"
    path.write_text('tts_engine = "edge"\nillustrations = True\nbook_images = False\n')
    old = Settings.load()
    assert old.tts_engine == "edge" and old.illustrations and not old.book_images
