"""Pydantic data models for the pipeline."""

from datetime import datetime
from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    """A single segment from Whisper transcription."""

    index: int
    start: float  # seconds
    end: float  # seconds
    text: str
    confidence: float = 0.0  # 0–1, average word confidence
    speaker: str = ""  # speaker label from diarization, e.g. "SPEAKER_00"


class LabeledSegment(BaseModel):
    """A segment with an assigned label from Claude."""

    label_id: str
    start: float  # seconds
    end: float  # seconds
    duration: float  # end - start
    reason: str  # Claude's one-sentence justification
    confidence: float  # 0–1
    transcript_text: str
    speaker: str = ""  # primary speaker in this segment
    speaker_role: str = ""  # resolved role: "interviewer", "interviewee", etc.


class VideoResult(BaseModel):
    """Summary of processing results for one video."""

    filename: str
    filepath: str
    duration_s: float
    transcribed_at: str = ""
    labeled_at: str = ""
    segment_count: int = 0
    label_summary: dict[str, dict[str, float]] = {}
    # e.g. {"intro": {"count": 1, "total_duration_s": 312.1}}


class ReelClip(BaseModel):
    """A single clip within a reel."""

    source: str
    start: float
    end: float
    confidence: float = 0.0


class ReelManifest(BaseModel):
    """Manifest for a rendered reel."""

    label: str
    reel_path: str
    total_duration_s: float
    clip_count: int
    clips: list[ReelClip]
    rendered_at: str = ""
