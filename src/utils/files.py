"""Path helpers, output directory setup, and cleanup."""

import shutil
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def video_stem(video_path: Path) -> str:
    """Get the stem of a video filename for use as directory name."""
    return video_path.stem


def ensure_output_dirs(output_dir: Path) -> None:
    """Create the top-level output directory structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries").mkdir(exist_ok=True)
    (output_dir / "reels").mkdir(exist_ok=True)
    (output_dir / "tmp").mkdir(exist_ok=True)


def ensure_video_dir(output_dir: Path, stem: str) -> Path:
    """Create and return the per-video output directory."""
    d = output_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_audio_path(output_dir: Path, stem: str) -> Path:
    return output_dir / stem / "audio.wav"


def get_transcript_json_path(output_dir: Path, stem: str) -> Path:
    return output_dir / stem / "transcript.json"


def get_transcript_txt_path(output_dir: Path, stem: str) -> Path:
    return output_dir / stem / "transcript.txt"


def get_whisper_raw_path(output_dir: Path, stem: str) -> Path:
    return output_dir / stem / "whisper_raw.json"


def get_labels_path(output_dir: Path, stem: str) -> Path:
    return output_dir / stem / "labels.json"


def get_summary_path(output_dir: Path, label_id: str) -> Path:
    return output_dir / "summaries" / f"{label_id}.md"


def get_reel_path(output_dir: Path, label_id: str) -> Path:
    return output_dir / "reels" / f"{label_id}.mp4"


def get_reel_manifest_path(output_dir: Path, label_id: str) -> Path:
    return output_dir / "reels" / f"{label_id}_manifest.json"


def get_tmp_dir(output_dir: Path) -> Path:
    d = output_dir / "tmp"
    d.mkdir(exist_ok=True)
    return d


def cleanup_tmp(output_dir: Path) -> None:
    """Remove the tmp directory."""
    tmp = output_dir / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)


def cleanup_audio_files(output_dir: Path) -> None:
    """Remove all extracted audio.wav files."""
    for wav in output_dir.rglob("audio.wav"):
        wav.unlink(missing_ok=True)


def resolve_video_inputs(inputs: tuple[str, ...] | list[str]) -> list[Path]:
    """Resolve a mix of video file paths and directories into a flat list of video files.

    Each input can be:
    - A path to a video file
    - A path to a directory (scanned for video files with supported extensions)

    Files are sorted alphabetically within each directory.
    Duplicates are removed (by resolved path), preserving order.
    """
    seen: set[Path] = set()
    result: list[Path] = []

    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            # Scan directory for video files (non-recursive)
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    resolved = child.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        result.append(child)
        else:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(p)

    return result


def validate_video_path(path: Path) -> str | None:
    """Validate a video file path. Returns error message or None."""
    if not path.exists():
        return f"video file \"{path}\" does not exist"
    if not path.is_file():
        return f"\"{path}\" is not a file"
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return f"\"{path.suffix}\" is not a supported extension (supported: {exts})"
    return None
