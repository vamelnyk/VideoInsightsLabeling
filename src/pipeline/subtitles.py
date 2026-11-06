"""Import existing subtitle transcripts (WebVTT / SRT) instead of running Whisper.

Meeting tools such as Zoom, Teams and Google Meet export a transcript next to the
recording. These files already carry speaker names ("Jane Doe: text"), which is
more reliable than anonymous diarization, and importing them skips the slow
Whisper stage entirely.
"""

import logging
import re
from pathlib import Path

from src.models import TranscriptSegment
from src.pipeline.transcription import _merge_short_segments

logger = logging.getLogger(__name__)

SUBTITLE_EXTENSIONS = {".vtt", ".srt"}

# 00:01:02.345 / 00:01:02,345 / 01:02.345 (hours optional in WebVTT)
_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_CUE_TIMING = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}")
# "Speaker Name: text" — name is short and contains no sentence punctuation
_SPEAKER_PREFIX = re.compile(r"^([^:.!?\n]{1,60}?):\s+(.*)$", re.DOTALL)
# WebVTT voice tag: <v Speaker Name>text</v>
_VOICE_TAG = re.compile(r"^<v(?:\.[^ >]*)?\s+([^>]+)>(.*?)(?:</v>)?$", re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")


def _to_seconds(h: str | None, m: str, s: str, frac: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(frac.ljust(3, "0")) / 1000


def _split_speaker(text: str) -> tuple[str, str]:
    voice = _VOICE_TAG.match(text)
    if voice:
        return voice.group(1).strip(), _TAGS.sub("", voice.group(2)).strip()
    text = _TAGS.sub("", text).strip()
    m = _SPEAKER_PREFIX.match(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text


def parse_subtitles(path: Path) -> list[TranscriptSegment]:
    """Parse a .vtt or .srt file into transcript segments.

    Handles CRLF line endings, optional cue identifiers, NOTE/STYLE blocks,
    and speaker prefixes ("Name: text" or WebVTT <v Name> voice tags).
    """
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", raw.strip())

    segments: list[TranscriptSegment] = []
    for block in blocks:
        lines = block.split("\n")
        timing_idx = next((i for i, ln in enumerate(lines) if _CUE_TIMING.match(ln)), None)
        if timing_idx is None:
            continue  # header, NOTE, STYLE, REGION
        t = _CUE_TIMING.match(lines[timing_idx]).groups()
        start = _to_seconds(*t[:4])
        end = _to_seconds(*t[4:])
        text = " ".join(ln.strip() for ln in lines[timing_idx + 1:] if ln.strip())
        if not text or end <= start:
            continue
        speaker, text = _split_speaker(text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        segments.append(TranscriptSegment(
            index=len(segments),
            start=round(start, 3),
            end=round(end, 3),
            text=text,
            confidence=1.0,
            speaker=speaker,
        ))

    if not segments:
        raise ValueError(f"No subtitle cues found in {path}")

    segments = _merge_short_segments(segments)
    for i, seg in enumerate(segments):
        seg.index = i

    speakers = sorted({s.speaker for s in segments if s.speaker})
    logger.info(
        "Imported %s: %d segments, %d words%s",
        path.name,
        len(segments),
        sum(len(s.text.split()) for s in segments),
        f", speakers: {', '.join(speakers)}" if speakers else "",
    )
    return segments
