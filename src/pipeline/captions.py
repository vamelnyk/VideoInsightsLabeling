"""Burn-in captions: a headline banner on top and subtitles at the bottom.

Captions are drawn as transparent PNGs with Pillow and composited with
ffmpeg's `overlay` filter. That works with any ffmpeg build — many builds
(e.g. Homebrew's) ship without libass/freetype, so `subtitles` and
`drawtext` are not available.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.models import TranscriptSegment

logger = logging.getLogger(__name__)

# Font candidates: (path, ttc index for regular, ttc index for bold)
_FONT_CANDIDATES = [
    ("/System/Library/Fonts/HelveticaNeue.ttc", 0, 1),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0, 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0, 0),
    ("C:/Windows/Fonts/arial.ttf", 0, 0),
]
_BOLD_FILES = {
    "/System/Library/Fonts/Supplemental/Arial.ttf": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arial.ttf": "C:/Windows/Fonts/arialbd.ttf",
}

MAX_WORDS_PER_CAPTION = 14
MAX_CHARS_PER_CAPTION = 84


@dataclass
class CaptionLine:
    start: float  # seconds, relative to clip start
    end: float
    text: str
    speaker: str = ""
    secondary: bool = False  # e.g. interviewer — drawn in a softer color


@dataclass
class Overlay:
    image: Path
    x: int
    y: int
    start: float | None = None  # None = whole clip
    end: float | None = None


def captions_available() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _load_font(size: int, bold: bool):
    from PIL import ImageFont

    custom = os.environ.get("CAPTION_FONT")
    if custom:
        return ImageFont.truetype(custom, size)
    for path, regular_idx, bold_idx in _FONT_CANDIDATES:
        if bold and path in _BOLD_FILES and Path(_BOLD_FILES[path]).exists():
            return ImageFont.truetype(_BOLD_FILES[path], size)
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=bold_idx if bold else regular_idx)
    return ImageFont.load_default(size=size)


# ── Subtitle timing ───────────────────────────────────────────────────────────


def _chunk_words(text: str) -> list[str]:
    """Split text into caption-sized chunks, preferring sentence/clause breaks."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    for w in words:
        if current and (
            len(current) >= MAX_WORDS_PER_CAPTION
            or len(" ".join(current + [w])) > MAX_CHARS_PER_CAPTION
        ):
            chunks.append(" ".join(current))
            current = []
        current.append(w)
        if len(current) >= 6 and re.search(r"[.!?,;…]$", w):
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_caption_lines(
    transcript: list[TranscriptSegment],
    clip_start: float,
    clip_end: float,
    secondary_speakers: set[str] | None = None,
) -> list[CaptionLine]:
    """Turn transcript segments overlapping a clip into timed caption lines.

    Long segments are split into chunks; each chunk gets a share of the
    segment's time proportional to its length. Times are clip-relative.
    """
    secondary_speakers = secondary_speakers or set()
    lines: list[CaptionLine] = []
    for seg in transcript:
        if seg.end <= clip_start or seg.start >= clip_end:
            continue
        chunks = _chunk_words(seg.text)
        total_chars = sum(len(c) for c in chunks) or 1
        t = seg.start
        for chunk in chunks:
            dur = (seg.end - seg.start) * len(chunk) / total_chars
            c_start, c_end = max(t, clip_start), min(t + dur, clip_end)
            t += dur
            if c_end - c_start < 0.4:
                continue
            lines.append(CaptionLine(
                start=round(c_start - clip_start, 3),
                end=round(c_end - clip_start, 3),
                text=chunk,
                speaker=seg.speaker,
                secondary=seg.speaker in secondary_speakers,
            ))
    return lines


# ── Drawing ───────────────────────────────────────────────────────────────────


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) != 6:
        return (245, 158, 11)
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


BAND_COLOR = (16, 18, 24)
HEADER_H_1080 = 120  # header band height at 1080p
SUBTITLE_H_1080 = 130  # subtitle band height at 1080p (fits two lines)


@dataclass
class CaptionLayout:
    """Overlays plus the base filter that makes room for them."""

    overlays: list[Overlay]
    base_filter: str  # applied to the video before the overlays


def _even(n: float) -> int:
    return int(n) // 2 * 2


def _draw_headline(title: str, label_name: str, label_color: str, width: int, height: int, scale: float, path: Path) -> None:
    """Draw the header band: label tag + one-line headline (font shrinks to fit)."""
    from PIL import Image, ImageDraw

    pad_x = int(48 * scale)
    accent_w = int(10 * scale)
    tag_font = _load_font(int(24 * scale), bold=True)
    max_w = width - 2 * pad_x - accent_w

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    size = int(52 * scale)
    title_font = _load_font(size, bold=True)
    while probe.textlength(title, font=title_font) > max_w and size > int(28 * scale):
        size -= 2
        title_font = _load_font(size, bold=True)

    img = Image.new("RGBA", (width, height), BAND_COLOR + (255,))
    d = ImageDraw.Draw(img)
    accent = _hex_to_rgb(label_color)
    d.rectangle([0, 0, accent_w, height], fill=accent + (255,))
    x = pad_x + accent_w
    tag_h = int(30 * scale)
    gap = int(8 * scale)
    block_h = tag_h + gap + size
    y = (height - block_h) // 2
    d.text((x, y), label_name.upper(), font=tag_font, fill=accent + (255,))
    d.text((x, y + tag_h + gap), title, font=title_font, fill=(255, 255, 255, 255))
    img.save(path)


def _draw_subtitle(line: CaptionLine, width: int, band_h: int, scale: float, path: Path) -> tuple[int, int]:
    """Draw one subtitle (up to two lines). Returns (image_width, image_height)."""
    from PIL import Image, ImageDraw

    font = _load_font(int(42 * scale), bold=False)
    name_font = _load_font(int(42 * scale), bold=True)
    line_h = int(54 * scale)
    max_w = int(width * 0.9)

    prefix = f"{line.speaker.split()[0]}: " if line.secondary and line.speaker else ""
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    wrapped = _wrap(probe, prefix + line.text, font, max_w)
    if len(wrapped) > 2:  # rare: squeeze into two lines with a smaller font
        font = _load_font(int(34 * scale), bold=False)
        name_font = _load_font(int(34 * scale), bold=True)
        line_h = int(44 * scale)
        wrapped = _wrap(probe, prefix + line.text, font, max_w)[:2]
    img_w = int(max(probe.textlength(ln, font=font) for ln in wrapped)) + 4
    img_h = min(band_h, line_h * len(wrapped) + int(8 * scale))

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = (255, 214, 128, 255) if line.secondary else (255, 255, 255, 255)
    y = 0
    for i, ln in enumerate(wrapped):
        x = 0
        if i == 0 and prefix and ln.startswith(prefix.strip()):
            d.text((x, y), prefix, font=name_font, fill=color)
            x += int(probe.textlength(prefix, font=name_font))
            ln = ln[len(prefix):]
        d.text((x, y), ln, font=font, fill=color)
        y += line_h
    img.save(path)
    return img_w, img_h


def render_overlays(
    title: str,
    label_name: str,
    label_color: str,
    lines: list[CaptionLine],
    width: int,
    height: int,
    out_dir: Path,
    prefix: str,
) -> CaptionLayout:
    """Draw headline + subtitle PNGs for one clip and lay them out.

    The output keeps the source size: the video is scaled down and framed
    between a header band (headline) and a footer band (subtitles), so
    captions never cover the picture.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = height / 1080
    header_h = _even(HEADER_H_1080 * scale) if title else 0
    footer_h = _even(SUBTITLE_H_1080 * scale) if lines else 0
    video_h = _even(height - header_h - footer_h)
    color = "0x%02x%02x%02x" % BAND_COLOR
    base = f"scale=-2:{video_h},pad={width}:{height}:(ow-iw)/2:{header_h}:color={color},setsar=1"

    overlays: list[Overlay] = []
    if title:
        p = out_dir / f"{prefix}_title.png"
        _draw_headline(title, label_name, label_color, width, header_h, scale, p)
        overlays.append(Overlay(p, 0, 0))

    for i, line in enumerate(lines):
        p = out_dir / f"{prefix}_sub_{i:03d}.png"
        w, h = _draw_subtitle(line, width, footer_h, scale, p)
        y = header_h + video_h + (footer_h - h) // 2
        overlays.append(Overlay(p, (width - w) // 2, y, line.start, line.end))

    return CaptionLayout(overlays, base)


def overlay_ffmpeg_args(layout: CaptionLayout) -> tuple[list[str], str, str]:
    """Build extra -i inputs and a filter_complex for a caption layout.

    Input 0 is the video. Returns (input_args, filter_complex, output_label).
    """
    inputs: list[str] = []
    filters: list[str] = [f"[0:v]{layout.base_filter}[base]"]
    current = "[base]"
    for i, ov in enumerate(layout.overlays, start=1):
        inputs += ["-i", str(ov.image)]
        out = f"[v{i}]"
        enable = ""
        if ov.start is not None and ov.end is not None:
            enable = f":enable='between(t,{ov.start:.3f},{ov.end:.3f})'"
        filters.append(f"{current}[{i}:v]overlay={ov.x}:{ov.y}:eof_action=repeat{enable}{out}")
        current = out
    return inputs, ";".join(filters), current
