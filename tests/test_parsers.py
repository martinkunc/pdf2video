import io
from pathlib import Path

import docx
import pymupdf
import pytest
from ebooklib import epub
from PIL import Image

from pdf2video.parsers import ParseError, load_document


def _paragraph(n: int) -> str:
    return " ".join(f"Sentence number {i} of the text is here." for i in range(n))


def test_markdown_chapters(tmp_path: Path):
    f = tmp_path / "book.md"
    f.write_text(
        "# My Book\n\nIntro text that is a preface.\n\n## First\n\nHello **world** "
        "and [a link](http://x).\n\n### Sub\n\nMore.\n\n## Second\n\n- item one\n- item two\n"
    )
    doc = load_document(f)
    assert doc.title == "My Book"
    assert [c.title for c in doc.chapters] == ["My Book", "First", "Second"]
    assert doc.chapters[1].paragraphs[0] == "Hello world and a link."
    assert "Sub" in doc.chapters[1].paragraphs


def test_plain_text_chapter_lines(tmp_path: Path):
    f = tmp_path / "novel.txt"
    f.write_text("CHAPTER I\n\nIt was a dark night.\n\nCHAPTER II\n\nThe sun rose.\n")
    doc = load_document(f)
    assert [c.title for c in doc.chapters] == ["CHAPTER I", "CHAPTER II"]


def test_docx_headings(tmp_path: Path):
    d = docx.Document()
    d.core_properties.title = "Doc Title"
    d.add_heading("Alpha", level=1)
    d.add_paragraph(_paragraph(3))
    d.add_heading("Alpha detail", level=2)
    d.add_paragraph("Detail text.")
    d.add_heading("Beta", level=1)
    d.add_paragraph("Beta text.")
    f = tmp_path / "test.docx"
    d.save(f)
    doc = load_document(f)
    assert doc.title == "Doc Title"
    assert [c.title for c in doc.chapters] == ["Alpha", "Beta"]
    assert "Alpha detail" in doc.chapters[0].paragraphs


def test_epub_toc(tmp_path: Path):
    book = epub.EpubBook()
    book.set_identifier("id1")
    book.set_title("Epub Title")
    book.add_author("Someone")
    png = io.BytesIO()
    Image.new("RGB", (300, 200), "red").save(png, "PNG")
    book.add_item(
        epub.EpubImage(file_name="img/pic.png", media_type="image/png", content=png.getvalue())
    )
    chapters = []
    for i, name in enumerate(["One", "Two"], 1):
        c = epub.EpubHtml(title=name, file_name=f"c{i}.xhtml")
        c.content = (
            f"<html><body><h1>{name}</h1><p>{_paragraph(2)}</p>"
            '<div><img src="img/pic.png"/></div></body></html>'
        )
        book.add_item(c)
        chapters.append(c)
    book.toc = chapters
    book.spine = ["nav", *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    f = tmp_path / "test.epub"
    epub.write_epub(str(f), book)
    doc = load_document(f)
    assert doc.title == "Epub Title"
    assert doc.author == "Someone"
    assert [c.title for c in doc.chapters] == ["One", "Two"]
    assert doc.chapters[0].paragraphs[0].startswith("Sentence number 0")
    assert len(doc.chapters[1].images) == 1


def _make_pdf(path: Path, with_toc: bool) -> None:
    pdf = pymupdf.open()
    toc = []
    for i, title in enumerate(["Introduction", "Methods", "Results"], 1):
        page = pdf.new_page()
        page.insert_text((72, 90), title, fontsize=24)
        y = 130
        for line in range(12):
            page.insert_text((72, y), f"Body line {line} of {title} with some words.", fontsize=11)
            y += 16
        toc.append([1, title, i])
    if with_toc:
        pdf.set_toc(toc)
    pdf.set_metadata({"title": "PDF Title", "author": "A. Author"})
    pdf.save(path)


@pytest.mark.parametrize("with_toc", [True, False])
def test_pdf_chapters(tmp_path: Path, with_toc: bool):
    f = tmp_path / "paper.pdf"
    _make_pdf(f, with_toc)
    doc = load_document(f)
    assert doc.title == "PDF Title"
    assert [c.title for c in doc.chapters] == ["Introduction", "Methods", "Results"]
    assert "Body line 0 of Methods" in doc.chapters[1].text
    assert "Methods" not in doc.chapters[0].text


def test_unsupported(tmp_path: Path):
    f = tmp_path / "x.xyz"
    f.write_text("hi")
    with pytest.raises(ParseError):
        load_document(f)


def test_malformed_pdf_is_repaired_quietly(tmp_path: Path, capfd, caplog):
    import logging
    import re

    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 90), "Chapter One", fontsize=24)
    page.insert_text((72, 130), "Readable body text survives the repair.", fontsize=11)
    raw = pdf.tobytes(deflate=False)
    # An invalid (non-name) key in the font dictionary, like buggy PDF generators write.
    raw, n = re.subn(rb"/Type\s*/Font", rb"/Type/Font 123 (x)", raw, count=1)
    assert n == 1
    f = tmp_path / "broken.pdf"
    f.write_bytes(raw)

    with caplog.at_level(logging.DEBUG, logger="pdf2video.parsers.pdf"):
        doc = load_document(f)
    assert "Readable body text survives the repair." in doc.chapters[0].text
    assert "MuPDF error" not in capfd.readouterr().err  # nothing printed by the C library
    assert any("invalid key in dict" in r.getMessage() for r in caplog.records)
    assert any(r.levelno == logging.INFO and "repaired" in r.getMessage() for r in caplog.records)
