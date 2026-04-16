"""Whisper transcription wrapper with post-processing and optional speaker diarization."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from src.models import TranscriptSegment
from src.utils.timestamps import format_timestamp

logger = logging.getLogger(__name__)

# Patterns to remove from transcription output
FILLER_PATTERNS = re.compile(
    r"\[BLANK_AUDIO\]|\[MUSIC\]|\(applause\)|\(laughter\)|\(music\)|\[NOISE\]|\[SILENCE\]",
    re.IGNORECASE,
)

# Lazy-loaded model cache
_whisper_model = None
_whisper_model_name = None

# Diarization availability
_diarization_available: bool | None = None


def _get_model(model_size: str, device: str = "auto", compute_type: str = "auto"):
    """Get or create a cached WhisperModel instance."""
    global _whisper_model, _whisper_model_name

    if _whisper_model is not None and _whisper_model_name == model_size:
        return _whisper_model

    from faster_whisper import WhisperModel

    logger.info("Loading Whisper model: %s (device=%s, compute=%s)", model_size, device, compute_type)
    _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _whisper_model_name = model_size
    return _whisper_model


def _check_diarization_available() -> bool:
    """Check if pyannote.audio is installed and usable."""
    global _diarization_available
    if _diarization_available is not None:
        return _diarization_available
    try:
        from pyannote.audio import Pipeline  # noqa: F401
        _diarization_available = True
    except ImportError:
        _diarization_available = False
    return _diarization_available


def _run_diarization(audio_path: Path) -> list[dict]:
    """Run speaker diarization using pyannote.audio.

    Returns a list of dicts: [{"start": float, "end": float, "speaker": str}, ...]
    """
    import os
    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_TOKEN", ""))
    if not hf_token:
        logger.warning(
            "Speaker diarization requires a HuggingFace token. "
            "Set HF_TOKEN env var. Skipping diarization."
        )
        return []

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(str(audio_path))

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            })

        logger.info("Diarization complete: %d turns, %d speakers",
                     len(turns), len(set(t["speaker"] for t in turns)))
        return turns

    except Exception as e:
        logger.warning("Diarization failed: %s. Continuing without speaker labels.", e)
        return []


def _assign_speakers(
    segments: list[TranscriptSegment],
    diarization_turns: list[dict],
) -> list[TranscriptSegment]:
    """Assign speaker labels to transcript segments based on diarization turns.

    For each segment, finds the speaker with the most overlapping time.
    """
    if not diarization_turns:
        return segments

    for seg in segments:
        # Find speaker with maximum overlap
        speaker_overlap: dict[str, float] = {}
        for turn in diarization_turns:
            overlap_start = max(seg.start, turn["start"])
            overlap_end = min(seg.end, turn["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > 0:
                speaker = turn["speaker"]
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + overlap

        if speaker_overlap:
            seg.speaker = max(speaker_overlap, key=speaker_overlap.get)

    return segments


def transcribe(
    audio_path: Path,
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    diarize: bool = False,
) -> tuple[list[TranscriptSegment], list[dict]]:
    """Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to WAV audio file.
        model_size: Whisper model size.
        device: Device to use (auto, cpu, cuda, mps).
        compute_type: Compute type (auto, float16, int8, etc.).
        language: Language code or None for auto-detect.
        diarize: Whether to run speaker diarization (requires pyannote.audio).

    Returns:
        Tuple of (processed segments, raw whisper segments for debugging).
    """
    model = _get_model(model_size, device, compute_type)

    kwargs = {
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    if language:
        kwargs["language"] = language

    segments_gen, info = model.transcribe(str(audio_path), **kwargs)

    raw_segments: list[dict] = []
    processed: list[TranscriptSegment] = []

    for i, seg in enumerate(segments_gen):
        # Store raw data
        raw_seg = {
            "index": i,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in (seg.words or [])
            ],
        }
        raw_segments.append(raw_seg)

        text = seg.text.strip()
        if not text:
            continue

        # Remove filler tokens
        text = FILLER_PATTERNS.sub("", text).strip()
        if not text:
            continue

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Calculate average word confidence
        confidence = 0.0
        if seg.words:
            confidence = round(sum(w.probability for w in seg.words) / len(seg.words), 3)

        processed.append(TranscriptSegment(
            index=len(processed),
            start=round(seg.start, 3),
            end=round(seg.end, 3),
            text=text,
            confidence=confidence,
        ))

    # Post-processing: merge short segments
    processed = _merge_short_segments(processed)

    # Speaker diarization
    if diarize and _check_diarization_available():
        logger.info("Running speaker diarization...")
        turns = _run_diarization(audio_path)
        processed = _assign_speakers(processed, turns)
    elif diarize:
        logger.warning(
            "Speaker diarization requested but pyannote.audio is not installed. "
            "Install with: pip install pyannote.audio"
        )

    # Re-index after merge
    for i, seg in enumerate(processed):
        seg.index = i

    speakers_found = set(s.speaker for s in processed if s.speaker)
    logger.info(
        "Transcription complete: %d segments, %d words%s",
        len(processed),
        sum(len(s.text.split()) for s in processed),
        f", {len(speakers_found)} speakers" if speakers_found else "",
    )

    return processed, raw_segments


def _merge_short_segments(
    segments: list[TranscriptSegment],
    min_duration: float = 3.0,
    max_gap: float = 1.0,
) -> list[TranscriptSegment]:
    """Merge segments shorter than min_duration into the preceding segment.

    Only merges if the gap between segments is <= max_gap seconds
    and the speaker is the same (or speaker info is not available).
    """
    if not segments:
        return segments

    merged: list[TranscriptSegment] = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        duration = seg.end - seg.start
        gap = seg.start - prev.end
        same_speaker = (not seg.speaker and not prev.speaker) or seg.speaker == prev.speaker

        if duration < min_duration and gap <= max_gap and same_speaker:
            # Merge into previous
            merged[-1] = TranscriptSegment(
                index=prev.index,
                start=prev.start,
                end=seg.end,
                text=f"{prev.text} {seg.text}",
                confidence=round((prev.confidence + seg.confidence) / 2, 3),
                speaker=prev.speaker,
            )
        else:
            merged.append(seg)

    return merged


def write_transcript_json(segments: list[TranscriptSegment], path: Path) -> None:
    """Write structured transcript to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [seg.model_dump() for seg in segments]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def write_transcript_txt(
    segments: list[TranscriptSegment],
    path: Path,
    video_filename: str,
    duration_s: float,
) -> None:
    """Write human-readable transcript to TXT."""
    path.parent.mkdir(parents=True, exist_ok=True)

    total_words = sum(len(s.text.split()) for s in segments)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"TRANSCRIPT — {video_filename}",
        f"Duration: {format_timestamp(duration_s)} | Words: {total_words:,} | Generated: {now}",
        "",
        "─" * 60,
        "",
    ]

    for seg in segments:
        speaker_prefix = f" {seg.speaker}:" if seg.speaker else ""
        lines.append(f"[{format_timestamp(seg.start)}]{speaker_prefix} {seg.text}")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_whisper_raw(raw_segments: list[dict], path: Path) -> None:
    """Write raw Whisper output for debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw_segments, indent=2, ensure_ascii=False))


def load_transcript(path: Path) -> list[TranscriptSegment]:
    """Load a transcript from JSON file."""
    data = json.loads(path.read_text())
    return [TranscriptSegment(**seg) for seg in data]
