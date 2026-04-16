"""Parallel transcription orchestration."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import Settings
from src.models import TranscriptSegment
from src.pipeline.audio import extract_audio
from src.pipeline.transcription import (
    transcribe,
    write_transcript_json,
    write_transcript_txt,
    write_whisper_raw,
)
from src.utils.ffmpeg import get_video_duration
from src.utils.files import (
    ensure_video_dir,
    get_audio_path,
    get_transcript_json_path,
    get_transcript_txt_path,
    get_whisper_raw_path,
    video_stem,
)

logger = logging.getLogger(__name__)


def transcribe_video_full(
    video_path: Path,
    output_dir: Path,
    settings: Settings,
    language: str | None = None,
    diarize: bool = False,
) -> tuple[str, list[TranscriptSegment], float] | tuple[str, None, float]:
    """Transcribe a single video: extract audio → run Whisper → write outputs.

    Returns:
        Tuple of (video_filename, segments_or_None, duration).
    """
    stem = video_stem(video_path)
    ensure_video_dir(output_dir, stem)

    try:
        # Get duration
        duration = get_video_duration(video_path)

        # Extract audio
        audio_path = get_audio_path(output_dir, stem)
        extract_audio(video_path, audio_path)

        # Transcribe
        segments, raw = transcribe(
            audio_path,
            model_size=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            language=language,
            diarize=diarize,
        )

        # Write outputs
        write_transcript_json(segments, get_transcript_json_path(output_dir, stem))
        write_transcript_txt(segments, get_transcript_txt_path(output_dir, stem), video_path.name, duration)
        write_whisper_raw(raw, get_whisper_raw_path(output_dir, stem))

        total_words = sum(len(s.text.split()) for s in segments)
        logger.info(
            "%s — transcript done (%s, %d words)",
            video_path.name,
            _fmt_duration(duration),
            total_words,
        )

        return video_path.name, segments, duration

    except Exception as e:
        logger.error("%s — transcription failed: %s", video_path.name, e)
        return video_path.name, None, 0.0


def run_transcription_pool(
    video_paths: list[Path],
    output_dir: Path,
    settings: Settings,
    language: str | None = None,
    skip_existing: bool = False,
    diarize: bool = False,
) -> dict[str, tuple[list[TranscriptSegment], float]]:
    """Transcribe multiple videos in parallel.

    Args:
        video_paths: List of video file paths.
        output_dir: Output directory.
        settings: Application settings.
        language: Language override, or None for auto-detect.
        skip_existing: Skip videos that already have transcripts.
        diarize: Whether to run speaker diarization.

    Returns:
        Dict mapping video filename → (segments, duration).
        Failed videos are omitted.
    """
    results: dict[str, tuple[list[TranscriptSegment], float]] = {}
    to_process: list[Path] = []

    for vp in video_paths:
        stem = video_stem(vp)
        json_path = get_transcript_json_path(output_dir, stem)
        if skip_existing and json_path.exists():
            logger.info("%s — transcript exists, skipping", vp.name)
            from src.pipeline.transcription import load_transcript
            segments = load_transcript(json_path)
            duration = get_video_duration(vp)
            results[vp.name] = (segments, duration)
        else:
            to_process.append(vp)

    if not to_process:
        return results

    max_workers = min(settings.max_parallel_transcriptions, len(to_process))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(transcribe_video_full, vp, output_dir, settings, language, diarize): vp
            for vp in to_process
        }

        for future in as_completed(futures):
            vp = futures[future]
            try:
                filename, segments, duration = future.result()
                if segments is not None:
                    results[filename] = (segments, duration)
            except Exception as e:
                logger.error("%s — unexpected error: %s", vp.name, e)

    return results


def _fmt_duration(seconds: float) -> str:
    """Quick HH:MM:SS format."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
