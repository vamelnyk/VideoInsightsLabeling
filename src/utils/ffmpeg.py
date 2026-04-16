"""FFmpeg subprocess wrapper with logging."""

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """Raised when an FFmpeg command fails."""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def check_ffmpeg() -> bool:
    """Check if ffmpeg and ffprobe are available on PATH."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def get_ffmpeg_path() -> str:
    """Return the path to ffmpeg, respecting FFMPEG_PATH env var."""
    import os
    return os.environ.get("FFMPEG_PATH", "ffmpeg")


def get_ffprobe_path() -> str:
    """Return the path to ffprobe."""
    import os
    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg")
    return ffmpeg.replace("ffmpeg", "ffprobe") if "ffmpeg" in ffmpeg else "ffprobe"


def run_ffmpeg(
    args: list[str],
    description: str = "",
    timeout: int = 3600,
) -> subprocess.CompletedProcess:
    """Run an ffmpeg command with logging.

    Args:
        args: Arguments to pass to ffmpeg (without the 'ffmpeg' binary itself).
        description: Human-readable description for logging.
        timeout: Timeout in seconds.

    Returns:
        CompletedProcess result.

    Raises:
        FFmpegError: If the command exits with non-zero status.
    """
    cmd = [get_ffmpeg_path()] + args
    logger.debug("FFmpeg: %s — %s", description, " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"FFmpeg timed out after {timeout}s: {description}") from e

    if result.returncode != 0:
        logger.error("FFmpeg failed: %s\nstderr: %s", description, result.stderr)
        raise FFmpegError(
            f"FFmpeg failed (exit {result.returncode}): {description}",
            stderr=result.stderr,
        )

    logger.debug("FFmpeg OK: %s", description)
    return result


def run_ffprobe(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run an ffprobe command."""
    cmd = [get_ffprobe_path()] + args
    logger.debug("FFprobe: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise FFmpegError(
            f"FFprobe failed (exit {result.returncode})",
            stderr=result.stderr,
        )
    return result


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = run_ffprobe([
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ])
    return float(result.stdout.strip())


def get_video_info(video_path: Path) -> dict:
    """Get video codec, resolution, and framerate using ffprobe."""
    import json as _json

    result = run_ffprobe([
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate",
        "-of", "json",
        str(video_path),
    ])
    data = _json.loads(result.stdout)
    if data.get("streams"):
        stream = data["streams"][0]
        return {
            "codec": stream.get("codec_name", "unknown"),
            "width": stream.get("width", 0),
            "height": stream.get("height", 0),
            "fps": stream.get("r_frame_rate", "30/1"),
        }
    return {"codec": "unknown", "width": 0, "height": 0, "fps": "30/1"}
