"""Audio extraction from video files using FFmpeg."""

import logging
from pathlib import Path

from src.utils.ffmpeg import run_ffmpeg, get_video_duration, FFmpegError

logger = logging.getLogger(__name__)


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from a video file as 16kHz mono WAV.

    Args:
        video_path: Path to the input video file.
        output_path: Path where the WAV file should be written.

    Returns:
        Path to the extracted audio file.

    Raises:
        FFmpegError: If audio extraction fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_ffmpeg(
        [
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
            "-y",
            "-loglevel", "error",
        ],
        description=f"extract audio from {video_path.name}",
    )

    logger.info("Audio extracted: %s → %s", video_path.name, output_path.name)
    return output_path
