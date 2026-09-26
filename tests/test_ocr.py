import io
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from pdf2video.parsers import ParseOptions, load_document
from pdf2video.parsers.pdf import tesseract_available

pytestmark = pytest.mark.skipif(not tesseract_available(), reason="Tesseract not installed")

CZECH = [
    "Kapitola první",
    "",
    "Byl jednou jeden malý program, který se naučil číst knihy nahlas.",
    "Každý den pilně pracoval a učil se rozdělovat text na věty a kapitoly.",
    "Nakonec dokázal vytvořit zvukovou knihu i vysvětlující video, a to je",
    "přece skvělé. Příliš žluťoučký kůň úpěl ďábelské ódy.",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    pytest.skip("no TrueType font available")


def _scanned_pdf(path: Path, lines: list[str]) -> None:
    """A PDF whose only content is an image of the text (like a scan)."""
    img = Image.new("L", (1700, 2200), 255)
    draw = ImageDraw.Draw(img)
    y = 200
    for i, line in enumerate(lines):
        draw.text((150, y), line, fill=0, font=_font(64 if i == 0 else 40))
        y += 90 if i == 0 else 60
    buf = io.BytesIO()
    img.save(buf, "PNG")
    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=buf.getvalue())
    pdf.save(path)


def test_ocr_detects_language_and_keeps_diacritics(tmp_path: Path):
    f = tmp_path / "scan.pdf"
    _scanned_pdf(f, CZECH)
    doc = load_document(f)
    text = doc.chapters[-1].text
    assert "žluťoučký" in text
    assert "vysvětlující" in text


def test_ocr_off_reports_no_text(tmp_path: Path):
    from pdf2video.parsers import ParseError

    f = tmp_path / "scan.pdf"
    _scanned_pdf(f, CZECH)
    with pytest.raises(ParseError, match="OCR"):
        load_document(f, ParseOptions(ocr="off"))
