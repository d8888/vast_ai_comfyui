#!/usr/bin/env python3
"""Generate the deterministic smoke-test assets for vast-comfy.

Pure standard library (no Pillow).  Writes two 1024x1024 PNG files into
``assets/smoke/``:

* ``vast_comfy_smoke_control.png``      - RGB line-art control image for the
  LLLite "any-test-like" weights (white background, thin black strokes).
* ``vast_comfy_smoke_inpaint_src.png``  - RGBA scene whose centre circle has
  ``alpha = 0`` (transparent == area to repaint).  ComfyUI's ``LoadImage``
  turns that into ``MASK = 1 - alpha``, so mask 1 = inpaint area.

Everything is deterministic: no randomness, no timestamps.  PNG data uses
filter type 0 on every scanline and ``zlib`` level 9; each file stays well
below 300 KB.

Usage::

    python tools/make_smoke_assets.py [--out-dir assets/smoke]
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import zlib
from pathlib import Path

WIDTH = 1024
HEIGHT = 1024
MAX_BYTES = 300 * 1024

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

PAPER = (255, 255, 255)
INK = (0, 0, 0)


# --------------------------------------------------------------------------
# PNG writer (stdlib only: struct + zlib)
# --------------------------------------------------------------------------
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def write_png(path: Path, width: int, height: int, channels: int,
              pixels: bytearray) -> int:
    """Write an 8-bit RGB/RGBA PNG (filter type 0, zlib level 9)."""
    if channels not in (3, 4):
        raise ValueError(f"channels must be 3 or 4, got {channels}")
    if len(pixels) != width * height * channels:
        raise ValueError("pixel buffer size does not match width*height*channels")

    stride = width * channels
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (None) on every scanline
        start = y * stride
        raw += pixels[start:start + stride]

    color_type = 2 if channels == 3 else 6
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    blob = (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(blob)
    return len(blob)


# --------------------------------------------------------------------------
# Tiny software canvas
# --------------------------------------------------------------------------
class Canvas:
    """Row-major interleaved 8-bit RGB/RGBA pixel buffer."""

    __slots__ = ("width", "height", "channels", "px")

    def __init__(self, width: int, height: int, channels: int, background):
        self.width = width
        self.height = height
        self.channels = channels
        if channels == 3:
            r, g, b = background[:3]
            a = 255
        else:
            r, g, b, a = background
        n = width * height
        self.px = bytearray(n * channels)
        self.px[0::channels] = bytes([r]) * n
        self.px[1::channels] = bytes([g]) * n
        self.px[2::channels] = bytes([b]) * n
        if channels == 4:
            self.px[3::channels] = bytes([a]) * n

    # -- primitives --------------------------------------------------------
    def set_px(self, x: int, y: int, color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            i = (y * self.width + x) * self.channels
            px = self.px
            px[i] = color[0]
            px[i + 1] = color[1]
            px[i + 2] = color[2]
            if self.channels == 4:
                px[i + 3] = 255

    def clear_alpha_disc(self, cx: float, cy: float, radius: float) -> None:
        """Set alpha = 0 inside a disc (RGBA canvases only)."""
        if self.channels != 4:
            raise ValueError("clear_alpha_disc requires an RGBA canvas")
        r2 = radius * radius
        y0 = max(0, int(cy - radius) - 1)
        y1 = min(self.height - 1, int(cy + radius) + 1)
        x0 = max(0, int(cx - radius) - 1)
        x1 = min(self.width - 1, int(cx + radius) + 1)
        px = self.px
        for y in range(y0, y1 + 1):
            dy = y - cy
            row = y * self.width
            for x in range(x0, x1 + 1):
                dx = x - cx
                if dx * dx + dy * dy <= r2:
                    px[(row + x) * 4 + 3] = 0

    def fill_rows(self, y0: int, y1: int, color) -> None:
        width, ch = self.width, self.channels
        y0 = max(0, y0)
        y1 = min(self.height - 1, y1)
        if y1 < y0:
            return
        px = self.px
        for y in range(y0, y1 + 1):
            start = y * width * ch
            end = start + width * ch
            px[start:end:ch] = bytes([color[0]]) * width
            px[start + 1:end:ch] = bytes([color[1]]) * width
            px[start + 2:end:ch] = bytes([color[2]]) * width
            if ch == 4:
                px[start + 3:end:ch] = bytes([255]) * width

    def fill_column(self, x: int, y0: int, y1: int, color) -> None:
        if not (0 <= x < self.width):
            return
        for y in range(max(0, y0), min(self.height - 1, y1) + 1):
            self.set_px(x, y, color)

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color) -> None:
        x0, x1 = max(0, min(x0, x1)), min(self.width - 1, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(self.height - 1, max(y0, y1))
        for y in range(y0, y1 + 1):
            row = y * self.width
            for x in range(x0, x1 + 1):
                i = (row + x) * self.channels
                px = self.px
                px[i] = color[0]
                px[i + 1] = color[1]
                px[i + 2] = color[2]
                if self.channels == 4:
                    px[i + 3] = 255

    def fill_disc(self, cx: float, cy: float, radius: float, color) -> None:
        r2 = radius * radius
        for y in range(max(0, int(cy - radius)), min(self.height - 1, int(cy + radius)) + 1):
            dy = y - cy
            for x in range(max(0, int(cx - radius)), min(self.width - 1, int(cx + radius)) + 1):
                dx = x - cx
                if dx * dx + dy * dy <= r2:
                    self.set_px(x, y, color)

    def fill_triangle(self, a, b, c, color) -> None:
        xs = (a[0], b[0], c[0])
        ys = (a[1], b[1], c[1])
        x0, x1 = max(0, int(min(xs))), min(self.width - 1, int(max(xs)) + 1)
        y0, y1 = max(0, int(min(ys))), min(self.height - 1, int(max(ys)) + 1)

        def cross(p, q, pt):
            return (q[0] - p[0]) * (pt[1] - p[1]) - (q[1] - p[1]) * (pt[0] - p[0])

        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                p = (x + 0.5, y + 0.5)
                d1 = cross(a, b, p)
                d2 = cross(b, c, p)
                d3 = cross(c, a, p)
                if (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0):
                    self.set_px(x, y, color)

    def stroke(self, points, color, thickness: float = 6.0) -> None:
        """Draw a polyline with a round brush (no anti-aliasing)."""
        r = max(0.6, thickness / 2.0)
        ri = int(math.ceil(r))
        r2 = r * r
        px, ch = self.px, self.channels
        w, h = self.width, self.height
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            length = math.hypot(x1 - x0, y1 - y0)
            steps = max(2, int(length / 0.5) + 1)
            for step in range(steps + 1):
                t = step / steps
                bx = int(round(x0 + (x1 - x0) * t))
                by = int(round(y0 + (y1 - y0) * t))
                for dy in range(-ri, ri + 1):
                    yy = by + dy
                    if yy < 0 or yy >= h:
                        continue
                    row = yy * w
                    for dx in range(-ri, ri + 1):
                        if dx * dx + dy * dy > r2:
                            continue
                        xx = bx + dx
                        if xx < 0 or xx >= w:
                            continue
                        i = (row + xx) * ch
                        px[i] = color[0]
                        px[i + 1] = color[1]
                        px[i + 2] = color[2]
                        if ch == 4:
                            px[i + 3] = 255


# --------------------------------------------------------------------------
# Geometry helpers (deterministic point lists)
# --------------------------------------------------------------------------
def circle_points(cx: float, cy: float, rx: float, ry: float, segments: int = 96):
    pts = []
    for i in range(segments + 1):
        t = 2.0 * math.pi * i / segments
        pts.append((cx + rx * math.cos(t), cy + ry * math.sin(t)))
    return pts


def arc_points(cx: float, cy: float, rx: float, ry: float,
               start_deg: float, end_deg: float, segments: int = 48):
    pts = []
    for i in range(segments + 1):
        t = math.radians(start_deg + (end_deg - start_deg) * i / segments)
        pts.append((cx + rx * math.cos(t), cy + ry * math.sin(t)))
    return pts


def bezier_points(p0, p1, p2, segments: int = 48):
    pts = []
    for i in range(segments + 1):
        t = i / segments
        u = 1.0 - t
        x = u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0]
        y = u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1]
        pts.append((x, y))
    return pts


# --------------------------------------------------------------------------
# Asset 1: LLLite line-art control image (RGB, white background)
# --------------------------------------------------------------------------
def build_line_art() -> Canvas:
    c = Canvas(WIDTH, HEIGHT, 3, PAPER)
    s = c.stroke

    # Head.
    s(circle_points(512, 400, 150, 140), INK, 7)
    # Ears (outer + inner).
    s([(392, 296), (348, 152), (474, 246)], INK, 7)
    s([(632, 296), (676, 152), (550, 246)], INK, 7)
    s([(408, 282), (380, 196), (452, 262)], INK, 4)
    s([(616, 282), (644, 196), (572, 262)], INK, 4)
    # Eyes.
    s(circle_points(452, 372, 36, 40), INK, 6)
    s(circle_points(572, 372, 36, 40), INK, 6)
    s(circle_points(452, 372, 15, 17), INK, 5)
    s(circle_points(572, 372, 15, 17), INK, 5)
    # Nose + mouth.
    s([(492, 452), (532, 452), (512, 472), (492, 452)], INK, 5)
    s([(512, 472), (512, 480)], INK, 4)
    s(arc_points(492, 474, 22, 22, 15, 165), INK, 4)
    s(arc_points(532, 474, 22, 22, 15, 165), INK, 4)
    # Whiskers.
    s([(430, 452), (268, 428)], INK, 4)
    s([(430, 470), (262, 470)], INK, 4)
    s([(430, 488), (268, 510)], INK, 4)
    s([(594, 452), (756, 428)], INK, 4)
    s([(594, 470), (762, 470)], INK, 4)
    s([(594, 488), (756, 510)], INK, 4)
    # Body.
    s(circle_points(512, 700, 214, 160, segments=128), INK, 7)
    # Front legs + paws.
    s([(452, 820), (444, 926)], INK, 6)
    s([(572, 820), (580, 926)], INK, 6)
    s(arc_points(444, 926, 24, 14, 0, 180), INK, 6)
    s(arc_points(580, 926, 24, 14, 0, 180), INK, 6)
    # Tail.
    s(bezier_points((716, 796), (960, 900), (908, 596), segments=64), INK, 7)
    # Ground line + grass tufts.
    s([(120, 944), (904, 944)], INK, 6)
    for gx in (170, 250, 330, 690, 770, 850):
        s([(gx, 944), (gx - 18, 906)], INK, 3)
        s([(gx, 944), (gx, 898)], INK, 3)
        s([(gx, 944), (gx + 18, 906)], INK, 3)
    # Sun + rays.
    s(circle_points(858, 148, 58, 58, segments=64), INK, 5)
    for k in range(8):
        ang = math.radians(k * 45.0)
        s([(858 + 74 * math.cos(ang), 148 + 74 * math.sin(ang)),
           (858 + 106 * math.cos(ang), 148 + 106 * math.sin(ang))], INK, 5)
    # Birds.
    for bx, by in ((180, 190), (262, 158)):
        s(arc_points(bx, by, 24, 18, 180, 360, segments=24), INK, 4)
        s(arc_points(bx + 48, by, 24, 18, 180, 360, segments=24), INK, 4)
    return c


# --------------------------------------------------------------------------
# Asset 2: LLLite inpainting source (RGBA, transparent centre disc)
# --------------------------------------------------------------------------
def build_inpaint_source() -> Canvas:
    c = Canvas(WIDTH, HEIGHT, 4, (255, 255, 255, 255))

    # Sky: vertical gradient, quantised in 8-row bands / steps of 4 for good
    # PNG compression.
    top = (92, 148, 230)
    bottom = (252, 226, 178)
    for band in range(0, HEIGHT, 8):
        t = band / float(HEIGHT - 1)
        color = tuple(int((top[i] + (bottom[i] - top[i]) * t) // 4) * 4 for i in range(3))
        c.fill_rows(band, band + 7, color)

    # Sun.
    c.fill_disc(768, 232, 74, (255, 214, 102))

    # Two hills (back, then front).
    for x in range(WIDTH):
        y1 = 636 + int(46 * math.sin((x + 180) / 210.0))
        c.fill_column(x, y1, HEIGHT - 1, (96, 150, 84))
    for x in range(WIDTH):
        y2 = 726 + int(38 * math.sin((x - 260) / 170.0))
        c.fill_column(x, y2, HEIGHT - 1, (72, 124, 70))

    # House: chimney -> roof -> walls -> door -> windows.
    c.fill_rect(556, 356, 596, 430, (150, 62, 54))
    c.fill_triangle((366, 452), (512, 336), (658, 452), (176, 74, 62))
    c.fill_rect(402, 452, 622, 616, (232, 214, 186))
    c.fill_rect(486, 520, 538, 616, (122, 82, 52))
    c.fill_disc(530, 570, 5, (240, 220, 120))
    for wx in (424, 552):
        c.fill_rect(wx, 484, wx + 48, 532, (160, 210, 235))
        c.stroke([(wx - 2, 482), (wx + 50, 482), (wx + 50, 534),
                  (wx - 2, 534), (wx - 2, 482)], (250, 250, 250), 5)

    # Trees: trunk then canopy.
    c.fill_rect(196, 560, 216, 704, (104, 70, 46))
    c.fill_disc(206, 520, 86, (58, 130, 74))
    c.fill_disc(160, 556, 54, (48, 116, 66))
    c.fill_disc(252, 556, 52, (66, 142, 80))
    c.fill_rect(824, 540, 844, 704, (104, 70, 46))
    c.fill_disc(834, 496, 92, (58, 130, 74))
    c.fill_disc(786, 536, 56, (48, 116, 66))
    c.fill_disc(882, 536, 56, (66, 142, 80))

    # The area to repaint: fully transparent disc in the middle.  RGB content
    # (the house) stays plausible underneath.
    c.clear_alpha_disc(512, 512, 132)
    return c


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def default_out_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "smoke"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic vast-comfy smoke-test PNG assets.")
    parser.add_argument("--out-dir", default=str(default_out_dir()),
                        help="output directory (default: assets/smoke next to the repo root)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("vast_comfy_smoke_control.png", build_line_art),
        ("vast_comfy_smoke_inpaint_src.png", build_inpaint_source),
    ]

    ok = True
    for name, builder in targets:
        canvas = builder()
        path = out_dir / name
        size = write_png(path, WIDTH, HEIGHT, canvas.channels, canvas.px)
        exists = path.is_file()
        blob = path.read_bytes() if exists else b""
        header_ok = blob[:8] == PNG_SIGNATURE and b"IEND" in blob[-16:]
        status = "ok" if (exists and header_ok and size < MAX_BYTES) else "FAILED"
        if status != "ok":
            ok = False
        print(f"[{status}] {path}  {size} bytes ({size / 1024.0:.1f} KiB)  "
              f"color_type={'RGB' if canvas.channels == 3 else 'RGBA'}")
    if not ok:
        print("one or more assets failed validation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
