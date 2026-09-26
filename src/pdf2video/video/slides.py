"""Render 1920×1080 slides with Pillow."""

from __future__ import annotations

import io
from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
MARGIN = 120
BG_TOP = (15, 23, 42)
BG_BOTTOM = (30, 41, 59)
ACCENT = (56, 189, 248)
TITLE_COLOR = (248, 250, 252)
TEXT_COLOR = (203, 213, 225)
MUTED = (100, 116, 139)

_REGULAR = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
_BOLD = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


@cache
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for candidate in _BOLD if bold else _REGULAR:
        if Path(candidate).exists():
            try:
                index = 1 if bold and candidate.endswith(".ttc") else 0
                return ImageFont.truetype(candidate, size, index=index)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _wrap(text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and fnt.getlength(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


@cache
def _background() -> Image.Image:
    img = Image.new("RGB", (W, H), BG_TOP)
    draw = ImageDraw.Draw(img)
    for y in range(H):
        t = y / (H - 1)
        color = tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM, strict=True))
        draw.line([(0, y), (W, y)], fill=color)
    return img


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = _background().copy()
    return img, ImageDraw.Draw(img)


def _fit_block(
    paragraphs: list[str], width: int, height: int, start: int, minimum: int, bold=False
) -> tuple[ImageFont.FreeTypeFont, list[list[str]], int]:
    """Largest font size at which the wrapped paragraphs fit into width×height."""
    size = start
    while True:
        fnt = font(size, bold)
        wrapped = [_wrap(p, fnt, width) for p in paragraphs]
        line_h = round(size * 1.3)
        gap = round(size * 0.7)
        needed = sum(len(w) for w in wrapped) * line_h + gap * max(0, len(wrapped) - 1)
        if needed <= height or size <= minimum:
            return fnt, wrapped, line_h
        size -= 2


def _footer(draw: ImageDraw.ImageDraw, left: str, progress: float | None) -> None:
    f = font(26)
    draw.text((MARGIN, H - 70), left, font=f, fill=MUTED, anchor="ls")
    if progress is not None:
        draw.rectangle([0, H - 8, W, H], fill=(51, 65, 85))
        draw.rectangle([0, H - 8, round(W * progress), H], fill=ACCENT)


def _paste_image(img: Image.Image, data: bytes, box: tuple[int, int, int, int]) -> bool:
    try:
        picture = Image.open(io.BytesIO(data))
        picture.load()
    except Exception:
        return False
    picture = picture.convert("RGBA")
    x0, y0, x1, y1 = box
    picture.thumbnail((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    px = x0 + (x1 - x0 - picture.width) // 2
    py = y0 + (y1 - y0 - picture.height) // 2
    backing = Image.new("RGBA", (picture.width + 24, picture.height + 24), (255, 255, 255, 255))
    img.paste(backing.convert("RGB"), (px - 12, py - 12))
    img.paste(picture, (px, py), picture)
    return True


def render_title_slide(
    out: Path, chapter_title: str, chapter_label: str, document_title: str
) -> Path:
    img, draw = _canvas()
    draw.rectangle([MARGIN, 380, MARGIN + 140, 388], fill=ACCENT)
    draw.text((MARGIN, 360), chapter_label.upper(), font=font(34, True), fill=ACCENT, anchor="ls")
    fnt, wrapped, line_h = _fit_block([chapter_title], W - 2 * MARGIN, 380, 96, 48, bold=True)
    y = 440
    for line in wrapped[0]:
        draw.text((MARGIN, y), line, font=fnt, fill=TITLE_COLOR, anchor="lt")
        y += line_h
    _footer(draw, document_title, None)
    img.save(out, optimize=True)
    return out


Rect = tuple[int, int, int, int]  # x0, y0, x1, y1


def illustration_box(placement: str) -> Rect:
    """The area an illustration may use for ``placement`` ("right", "left", "full")."""
    if placement == "full":
        return (MARGIN, 300, W - MARGIN, H - 130)
    if placement == "left":
        return (MARGIN, 200, W // 2 - 60, H - 150)
    return (W // 2 + 60, 200, W - MARGIN, H - 150)


def fit_rect(size: tuple[int, int], box: Rect) -> Rect:
    """Largest rect with the aspect ratio of ``size``, centred in ``box``."""
    x0, y0, x1, y1 = box
    scale = min((x1 - x0) / size[0], (y1 - y0) / size[1])
    w, h = round(size[0] * scale), round(size[1] * scale)
    left, top = x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2
    return (left, top, left + w, top + h)


def _picture_frame(img: Image.Image, rect: Rect) -> None:
    """Soft shadow and a thin border around where a picture goes."""
    x0, y0, x1, y1 = rect
    shadow = Image.new("L", img.size, 0)
    ImageDraw.Draw(shadow).rounded_rectangle((x0 + 6, y0 + 14, x1 + 6, y1 + 14), 16, fill=150)
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    img.paste((2, 6, 18), (0, 0), shadow)
    ImageDraw.Draw(img).rectangle((x0 - 3, y0 - 3, x1 + 2, y1 + 2), outline=(71, 85, 105), width=3)


def render_scene_slide(
    out: Path,
    title: str,
    bullets: list[str],
    footer: str,
    progress: float,
    image: bytes | None = None,
    *,
    placement: str | None = None,
    drawing: Image.Image | None = None,
    picture: Image.Image | None = None,
    picture_rect: Rect | None = None,
) -> Path:
    """A scene slide.

    ``image``: a picture from the document, shown on the right. With ``placement``
    the slide gets an illustration instead: a transparent ``drawing`` (diagram)
    centred in :func:`illustration_box`, and/or a framed ``picture`` at
    ``picture_rect`` (the frame alone if ``picture`` is None, for an animated
    overlay added later). "full" leaves out the bullets.
    """
    img, draw = _canvas()
    draw.rectangle([0, 0, 14, H], fill=ACCENT)
    text_left, text_right = MARGIN, W - MARGIN
    title_h, show_bullets = 190, True
    if placement is not None:
        box = illustration_box(placement)
        if placement == "full":
            title_h, show_bullets = 160, False
        elif placement == "left":
            text_left = W // 2 + 20
        else:
            text_right = W // 2 + 20
        if picture_rect is not None:
            _picture_frame(img, picture_rect)
            if picture is not None:
                x0, y0, x1, y1 = picture_rect
                fitted = picture.convert("RGB").resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
                img.paste(fitted, (x0, y0))
        if drawing is not None:
            x0, y0, x1, y1 = box
            img.paste(
                drawing,
                (x0 + (x1 - x0 - drawing.width) // 2, y0 + (y1 - y0 - drawing.height) // 2),
                drawing,
            )
        draw = ImageDraw.Draw(img)
    elif image is not None and _paste_image(img, image, (W // 2 + 60, 200, W - MARGIN, H - 150)):
        text_right = W // 2 + 20

    t_font, t_wrapped, t_line = _fit_block(
        [title], text_right - text_left, title_h, 68, 44, bold=True
    )
    y = 110
    for line in t_wrapped[0][: 2 if placement == "full" else 3]:
        draw.text((text_left, y), line, font=t_font, fill=TITLE_COLOR, anchor="lt")
        y += t_line
    y += 50
    if show_bullets and bullets:
        bullet_x = text_left + 50
        b_font, b_wrapped, b_line = _fit_block(bullets, text_right - bullet_x, H - 150 - y, 48, 26)
        for lines in b_wrapped:
            dot_y = y + round(b_font.size * 0.55)
            r = max(6, b_font.size // 6)
            draw.ellipse([text_left + 8, dot_y - r, text_left + 8 + 2 * r, dot_y + r], fill=ACCENT)
            for line in lines:
                draw.text((bullet_x, y), line, font=b_font, fill=TEXT_COLOR, anchor="lt")
                y += b_line
            y += round(b_font.size * 0.7)
    _footer(draw, footer, progress)
    img.save(out, optimize=True)
    return out
