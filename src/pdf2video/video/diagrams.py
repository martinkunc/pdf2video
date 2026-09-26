"""Draw explanatory diagrams with Pillow.

The LLM describes a diagram as data (:class:`~.script.Diagram`: a type and a few
labelled items); this module lays it out and draws it in the slides' style. Local
models are good at choosing *what* to show and bad at emitting precise SVG, so the
drawing itself is deterministic, always legible and can be revealed item by item.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from .script import Diagram, DiagramItem
from .slides import ACCENT, MUTED, TEXT_COLOR, TITLE_COLOR, font

PALETTE = [
    ACCENT,
    (167, 139, 250),  # violet
    (52, 211, 153),  # emerald
    (251, 191, 36),  # amber
    (248, 113, 113),  # red
    (244, 114, 182),  # pink
    (45, 212, 191),  # teal
]
NODE_BG = (39, 52, 73)
MAX_NODE_W = 440  # wider boxes look empty
LINE = (148, 163, 184)
DARK_TEXT = (15, 23, 42)

Box = tuple[float, float, float, float]  # x0, y0, x1, y1


def reveal_steps(diagram: Diagram) -> int:
    """How many frames a "reveal" animation has (one per item)."""
    return len(diagram.items)


def render_diagram(
    diagram: Diagram, size: tuple[int, int], visible: int | None = None
) -> Image.Image:
    """Transparent RGBA image of ``size`` with the first ``visible`` items drawn."""
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    items = diagram.items
    n_visible = len(items) if visible is None else max(0, min(visible, len(items)))
    top = 0
    if diagram.caption:
        cap = font(28)
        draw.text((0, 0), _ellipsize(diagram.caption, cap, w), font=cap, fill=MUTED, anchor="lt")
        top = 56
    area = (0.0, float(top), float(w), float(h))
    LAYOUTS[diagram.type](draw, area, items, n_visible)
    return img


# ------------------------------------------------------------------ helpers


def _ellipsize(text: str, fnt: ImageFont.FreeTypeFont, width: float) -> str:
    if fnt.getlength(text) <= width:
        return text
    while text and fnt.getlength(text + "…") > width:
        text = text[:-1]
    return text.rstrip() + "…"


def _wrap(text: str, fnt: ImageFont.FreeTypeFont, width: float) -> list[str] | None:
    """Word-wrap; None if a single word is wider than ``width``."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        if fnt.getlength(word) > width:
            return None
        candidate = f"{current} {word}" if current else word
        if current and fnt.getlength(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _detail_lines(detail: str) -> list[str]:
    return [p.strip() for p in detail.split(";") if p.strip()]


class _Text:
    """A label (bold) and optional detail lines, wrapped and measured at one size."""

    def __init__(self, label: str, details: list[str], size: int, width: float):
        self.size = size
        self.label_font = font(size, True)
        self.detail_font = font(max(14, round(size * 0.74)))
        self.label = _wrap(label, self.label_font, width)
        wrapped = [_wrap(d, self.detail_font, width) for d in details]
        self.details = (
            None if any(d is None for d in wrapped) else [ln for d in wrapped for ln in d]
        )
        self.bullets = len(details) > 1

    @property
    def ok(self) -> bool:
        return self.label is not None and self.details is not None

    @property
    def height(self) -> float:
        h = len(self.label or []) * self.size * 1.2
        if self.details:
            h += self.size * 0.35 + len(self.details) * self.detail_font.size * 1.3
        return h

    def draw(self, draw: ImageDraw.ImageDraw, cx: float, y: float, color, align="center", x=0.0):
        for line in self.label or []:
            if align == "center":
                draw.text((cx, y), line, font=self.label_font, fill=color, anchor="mt")
            else:
                draw.text((x, y), line, font=self.label_font, fill=color, anchor="lt")
            y += self.size * 1.2
        if self.details:
            y += self.size * 0.35
            for line in self.details:
                if align == "center":
                    draw.text((cx, y), line, font=self.detail_font, fill=TEXT_COLOR, anchor="mt")
                else:
                    draw.text((x, y), line, font=self.detail_font, fill=TEXT_COLOR, anchor="lt")
                y += self.detail_font.size * 1.3


def _fit(
    items: list[DiagramItem], width: float, height: float, start: int, minimum: int = 18
) -> tuple[int, bool]:
    """Common font size at which every item fits ``width``×``height``; details dropped
    if they would make the text too small."""
    for with_detail, floor in ((True, 22), (False, minimum)):
        for size in range(start, floor - 1, -2):
            texts = [
                _Text(i.label, _detail_lines(i.detail) if with_detail else [], size, width)
                for i in items
            ]
            if all(t.ok and t.height <= height for t in texts):
                return size, with_detail
    return minimum, False


def _text(item: DiagramItem, size: int, with_detail: bool, width: float) -> _Text:
    return _Text(item.label, _detail_lines(item.detail) if with_detail else [], size, width)


def _node(
    draw: ImageDraw.ImageDraw, rect: Box, item: DiagramItem, color, size: int, with_detail: bool
) -> None:
    x0, y0, x1, y1 = rect
    radius = min(18, (y1 - y0) / 4)
    draw.rounded_rectangle(rect, radius=radius, fill=NODE_BG, outline=color, width=4)
    pad = size * 0.5
    text = _text(item, size, with_detail, x1 - x0 - 2 * pad)
    text.draw(draw, (x0 + x1) / 2, (y0 + y1) / 2 - text.height / 2, TITLE_COLOR)


def _nodes_fit(items, rect_w: float, rect_h: float, start: int = 40) -> tuple[int, bool]:
    pad = start * 0.5
    return _fit(items, rect_w - 2 * pad, rect_h - 2 * pad, start)


def _edge_point(rect: Box, toward: tuple[float, float], gap: float = 8) -> tuple[float, float]:
    """Where the ray from the rect's centre toward ``toward`` leaves the rect."""
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == dy == 0:
        return cx, cy
    t = min(
        (x1 - x0) / 2 / abs(dx) if dx else math.inf,
        (y1 - y0) / 2 / abs(dy) if dy else math.inf,
    )
    length = math.hypot(dx, dy)
    t += gap / length
    return cx + dx * t, cy + dy * t


def _center(rect: Box) -> tuple[float, float]:
    return (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2


def _arrow(draw: ImageDraw.ImageDraw, p0, p1, color=LINE, width: int = 5, head: float = 20) -> None:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    head = min(head, length * 0.6)
    base = (p1[0] - ux * head, p1[1] - uy * head)
    draw.line([p0, base], fill=color, width=width)
    wing = head * 0.6
    draw.polygon(
        [
            p1,
            (base[0] - uy * wing, base[1] + ux * wing),
            (base[0] + uy * wing, base[1] - ux * wing),
        ],
        fill=color,
    )


def _connect(draw, a: Box, b: Box, color=LINE, arrow: bool = True) -> None:
    p0 = _edge_point(a, _center(b))
    p1 = _edge_point(b, _center(a))
    if arrow:
        _arrow(draw, p0, p1, color)
    else:
        draw.line([p0, p1], fill=color, width=4)


def _color(i: int):
    return PALETTE[i % len(PALETTE)]


# ------------------------------------------------------------------ layouts


def _flow(draw, area: Box, items, visible: int) -> None:
    x0, y0, x1, y1 = area
    w, h, n = x1 - x0, y1 - y0, len(items)
    rects: list[Box] = []
    if w / h >= 1.5:
        gap = 70
        node_w = min((w - gap * (n - 1)) / n, MAX_NODE_W)
        node_h = min(h * 0.7, max(node_w * 0.8, 160))
        top = y0 + (h - node_h) / 2
        left = x0 + (w - (node_w * n + gap * (n - 1))) / 2
        rects = [
            (left + i * (node_w + gap), top, left + i * (node_w + gap) + node_w, top + node_h)
            for i in range(n)
        ]
    else:
        gap = 44
        node_h = min((h - gap * (n - 1)) / n, 180)
        node_w = min(w * 0.85, 620)
        top = y0 + (h - (node_h * n + gap * (n - 1))) / 2
        left = x0 + (w - node_w) / 2
        rects = [
            (left, top + i * (node_h + gap), left + node_w, top + i * (node_h + gap) + node_h)
            for i in range(n)
        ]
    size, detail = _nodes_fit(items, rects[0][2] - rects[0][0], rects[0][3] - rects[0][1])
    for i in range(visible):
        if i:
            _connect(draw, rects[i - 1], rects[i])
        _node(draw, rects[i], items[i], _color(i), size, detail)


def _ring(area: Box, n: int, node_w: float, node_h: float) -> list[Box]:
    x0, y0, x1, y1 = area
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0 - node_w) / 2, (y1 - y0 - node_h) / 2
    rects = []
    for i in range(n):
        angle = -math.pi / 2 + 2 * math.pi * i / n
        px, py = cx + rx * math.cos(angle), cy + ry * math.sin(angle)
        rects.append((px - node_w / 2, py - node_h / 2, px + node_w / 2, py + node_h / 2))
    return rects


def _cycle(draw, area: Box, items, visible: int) -> None:
    x0, y0, x1, y1 = area
    w, h, n = x1 - x0, y1 - y0, len(items)
    node_w = min(w * (0.36 if n <= 4 else 0.28), 340)
    node_h = min(h * (0.24 if n <= 4 else 0.17), 160)
    rects = _ring(area, n, node_w, node_h)
    size, detail = _nodes_fit(items, node_w, node_h, 36)
    for i in range(visible):
        if i:
            _connect(draw, rects[i - 1], rects[i])
        if i == n - 1:
            _connect(draw, rects[i], rects[0])
        _node(draw, rects[i], items[i], _color(i), size, detail)


def _hub(draw, area: Box, items, visible: int) -> None:
    x0, y0, x1, y1 = area
    w, h = x1 - x0, y1 - y0
    spokes = items[1:]
    n = len(spokes)
    node_w = min(w * (0.32 if n <= 4 else 0.27), 320)
    node_h = min(h * 0.19, 140)
    rects = _ring(area, n, node_w, node_h)
    cx, cy = _center(area)
    hub_w, hub_h = min(node_w * 1.15, w * 0.34), min(node_h * 1.25, h * 0.26)
    hub = (cx - hub_w / 2, cy - hub_h / 2, cx + hub_w / 2, cy + hub_h / 2)
    size, detail = _nodes_fit(spokes, node_w, node_h, 34)
    if visible:
        hub_size, _ = _nodes_fit([DiagramItem(label=items[0].label)], hub_w, hub_h, 44)
        for i in range(min(n, visible - 1)):
            _connect(draw, hub, rects[i], _color(i + 1), arrow=False)
        draw.rounded_rectangle(hub, radius=22, fill=ACCENT)
        text = _Text(items[0].label, [], hub_size, hub_w - hub_size)
        text.draw(draw, cx, cy - text.height / 2, DARK_TEXT)
    for i in range(min(n, visible - 1)):
        _node(draw, rects[i], spokes[i], _color(i + 1), size, detail)


def _hierarchy(draw, area: Box, items, visible: int) -> None:
    x0, y0, x1, y1 = area
    w, h = x1 - x0, y1 - y0
    root, children = items[0], items[1:]
    n = len(children)
    gap = 28
    if (w - gap * (n - 1)) / n >= 170:
        root_w, root_h = min(w * 0.5, 560), min(h * 0.28, 170)
        root_rect = (x0 + (w - root_w) / 2, y0, x0 + (w + root_w) / 2, y0 + root_h)
        child_w = min((w - gap * (n - 1)) / n, MAX_NODE_W)
        child_top = y0 + root_h + min(h * 0.18, 110)
        child_h = min(y1 - child_top, 260)
        left = x0 + (w - (child_w * n + gap * (n - 1))) / 2
        rects = [
            (
                left + i * (child_w + gap),
                child_top,
                left + i * (child_w + gap) + child_w,
                child_top + child_h,
            )
            for i in range(n)
        ]
        bus_y = (root_rect[3] + child_top) / 2
        size, detail = _nodes_fit(children, child_w, child_h, 40)
        if visible:
            root_size, root_detail = _nodes_fit([root], root_w, root_h, 44)
            _node(draw, root_rect, root, ACCENT, root_size, root_detail)
        shown = min(n, visible - 1)
        if shown > 0:
            cx = (root_rect[0] + root_rect[2]) / 2
            draw.line([(cx, root_rect[3]), (cx, bus_y)], fill=LINE, width=4)
            centers = [(r[0] + r[2]) / 2 for r in rects[:shown]]
            draw.line(
                [(min(centers + [cx]), bus_y), (max(centers + [cx]), bus_y)], fill=LINE, width=4
            )
        for i in range(shown):
            ccx = (rects[i][0] + rects[i][2]) / 2
            _arrow(draw, (ccx, bus_y), (ccx, rects[i][1] - 6), LINE, 4, 16)
            _node(draw, rects[i], children[i], _color(i + 1), size, detail)
    else:  # narrow area: an indented tree list
        root_h = min(h * 0.16, 120)
        root_rect = (x0, y0, x0 + min(w * 0.7, 480), y0 + root_h)
        trunk_x = x0 + 40
        left = trunk_x + 50
        top = root_rect[3] + 30
        child_h = min((y1 - top - gap * (n - 1)) / n, 150)
        rects = [
            (left, top + i * (child_h + gap), x1, top + i * (child_h + gap) + child_h)
            for i in range(n)
        ]
        size, detail = _nodes_fit(children, x1 - left, child_h, 32)
        if visible:
            root_size, root_detail = _nodes_fit([root], root_rect[2] - x0, root_h, 44)
            _node(draw, root_rect, root, ACCENT, root_size, root_detail)
        shown = min(n, visible - 1)
        if shown > 0:
            last_y = (rects[shown - 1][1] + rects[shown - 1][3]) / 2
            draw.line([(trunk_x, root_rect[3]), (trunk_x, last_y)], fill=LINE, width=4)
        for i in range(shown):
            cy = (rects[i][1] + rects[i][3]) / 2
            _arrow(draw, (trunk_x, cy), (left - 6, cy), LINE, 4, 16)
            _node(draw, rects[i], children[i], _color(i + 1), size, detail)


def _comparison(draw, area: Box, items, visible: int) -> None:
    x0, y0, x1, y1 = area
    w, h, n = x1 - x0, y1 - y0, len(items)
    gap = 36
    col_w = (w - gap * (n - 1)) / n
    head_h = min(110, h * 0.18)
    pad = 24
    body_top = y0 + head_h + 16
    # Font sizes shared by all columns.
    head_size, _ = _fit(
        [DiagramItem(label=i.label) for i in items], col_w - 2 * pad, head_h - 16, 40
    )
    body_size = 44
    while body_size > 16:
        fnt = font(body_size)
        line_h = body_size * 1.3
        ok = True
        for item in items:
            lines = 0
            for point in _detail_lines(item.detail):
                wrapped = _wrap(point, fnt, col_w - 2 * pad - 30)
                if wrapped is None:
                    ok = False
                    break
                lines += len(wrapped)
            gaps = max(0, len(_detail_lines(item.detail)) - 1) * body_size * 0.5
            if not ok or lines * line_h + gaps > y1 - body_top - 2 * pad:
                ok = False
                break
        if ok:
            break
        body_size -= 2
    fnt = font(body_size)
    for i in range(visible):
        item, color = items[i], _color(i)
        cx0 = x0 + i * (col_w + gap)
        draw.rounded_rectangle((cx0, y0, cx0 + col_w, y0 + head_h), radius=16, fill=color)
        text = _Text(item.label, [], head_size, col_w - 2 * pad)
        text.draw(draw, cx0 + col_w / 2, y0 + head_h / 2 - text.height / 2, DARK_TEXT)
        draw.rounded_rectangle(
            (cx0, body_top, cx0 + col_w, y1), radius=16, fill=NODE_BG, outline=color, width=3
        )
        y = body_top + pad
        for point in _detail_lines(item.detail):
            r = max(5, body_size // 6)
            dot_y = y + body_size * 0.55
            draw.ellipse([cx0 + pad, dot_y - r, cx0 + pad + 2 * r, dot_y + r], fill=color)
            for line in _wrap(point, fnt, col_w - 2 * pad - 30) or [point]:
                draw.text((cx0 + pad + 30, y), line, font=fnt, fill=TEXT_COLOR, anchor="lt")
                y += body_size * 1.3
            y += body_size * 0.5


def _timeline(draw, area: Box, items, visible: int) -> None:
    x0, y0, x1, y1 = area
    w, h, n = x1 - x0, y1 - y0, len(items)
    if w / h >= 1.5:
        axis_y = y0 + h * 0.42
        slot = w / n
        size, detail = _fit(items, slot - 24, y1 - axis_y - 40, 40)
        if visible:
            end = x0 + slot * (visible - 0.5) + (slot / 2 if visible == n else 0)
            _arrow(draw, (x0, axis_y), (min(x1, end + 30), axis_y), LINE, 5, 22)
        for i in range(visible):
            cx = x0 + slot * (i + 0.5)
            color = _color(i)
            draw.ellipse(
                [cx - 16, axis_y - 16, cx + 16, axis_y + 16], fill=color, outline=NODE_BG, width=4
            )
            label_font = font(size + 6, True)
            draw.text(
                (cx, axis_y - 34), _ellipsize(items[i].label, label_font, slot - 12),
                font=label_font, fill=color, anchor="mb",
            )  # fmt: skip
            if detail and items[i].detail:
                detail_font = font(size)
                y = axis_y + 36
                for line in _wrap(items[i].detail, detail_font, slot - 24) or [items[i].detail]:
                    draw.text((cx, y), line, font=detail_font, fill=TEXT_COLOR, anchor="mt")
                    y += size * 1.3
    else:
        axis_x = x0 + 24
        slot = h / n
        size, detail = _fit(items, w - 90, slot - 16, 42)
        if visible:
            end = y0 + slot * (visible - 0.5) + (slot / 2 if visible == n else 0)
            _arrow(draw, (axis_x, y0), (axis_x, min(y1, end + 10)), LINE, 5, 22)
        for i in range(visible):
            cy = y0 + slot * i + 20
            color = _color(i)
            draw.ellipse([axis_x - 14, cy - 14 + 8, axis_x + 14, cy + 14 + 8], fill=color)
            text = _text(items[i], size, detail, w - 90)
            for line_no, line in enumerate(text.label or []):
                draw.text(
                    (axis_x + 50, cy + line_no * size * 1.2 - 4), line,
                    font=text.label_font, fill=color, anchor="lt",
                )  # fmt: skip
            if text.details:
                y = cy - 4 + len(text.label or []) * size * 1.2 + size * 0.25
                for line in text.details:
                    draw.text(
                        (axis_x + 50, y), line, font=text.detail_font, fill=TEXT_COLOR, anchor="lt"
                    )
                    y += text.detail_font.size * 1.3


def _bars(draw, area: Box, items, visible: int) -> None:
    x0, y0, x1, y1 = area
    w, h, n = x1 - x0, y1 - y0, len(items)
    row = h / n
    bar_h = min(row * 0.62, 90)
    label_w = w * 0.32
    size, _ = _fit([DiagramItem(label=i.label) for i in items], label_w - 20, row * 0.9, 34)
    label_font = font(size, True)
    values = [i.value or 0.0 for i in items]
    top = max(values) or 1.0
    value_font = font(max(18, size - 2), True)
    widest = max(value_font.getlength(_number(v)) for v in values)
    bar_max = w - label_w - widest - 30
    for i in range(visible):
        cy = y0 + row * (i + 0.5)
        lines = _wrap(items[i].label, label_font, label_w - 20) or [items[i].label]
        ly = cy - len(lines) * size * 1.2 / 2
        for line in lines:
            draw.text((x0 + label_w - 20, ly), line, font=label_font, fill=TITLE_COLOR, anchor="rt")
            ly += size * 1.2
        length = max(6, bar_max * values[i] / top)
        bx = x0 + label_w
        draw.rounded_rectangle(
            (bx, cy - bar_h / 2, bx + length, cy + bar_h / 2),
            radius=min(12, bar_h / 3),
            fill=_color(i),
        )
        draw.text(
            (bx + length + 16, cy),
            _number(values[i]),
            font=value_font,
            fill=TITLE_COLOR,
            anchor="lm",
        )


def _number(value: float) -> str:
    if value == int(value):
        return f"{int(value):,}".replace(",", " ")
    return f"{value:,.2f}".rstrip("0").rstrip(".").replace(",", " ")


LAYOUTS = {
    "flow": _flow,
    "cycle": _cycle,
    "hierarchy": _hierarchy,
    "hub": _hub,
    "comparison": _comparison,
    "timeline": _timeline,
    "bars": _bars,
}
