"""FFmpeg-based reel rendering — trim and concatenate segments per label."""

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.context import LabelDefinition
from src.models import LabeledSegment, ReelClip, ReelManifest
from src.utils.ffmpeg import run_ffmpeg, get_video_info, FFmpegError

logger = logging.getLogger(__name__)


def render_reel(
    label: LabelDefinition,
    segments_by_video: dict[str, list[LabeledSegment]],
    video_paths: dict[str, Path],
    output_path: Path,
    tmp_dir: Path,
) -> ReelManifest | None:
    """Render a reel by trimming and concatenating all segments for a label.

    Args:
        label: The label definition.
        segments_by_video: Dict mapping video filename → segments for this label.
        video_paths: Dict mapping video filename → full path to video file.
        output_path: Where to write the output .mp4.
        tmp_dir: Directory for temporary trimmed clips.

    Returns:
        ReelManifest if successful, None if no segments.
    """
    # Collect all clips in order
    clips: list[tuple[Path, LabeledSegment]] = []
    for video_filename in video_paths:
        if video_filename in segments_by_video:
            for seg in segments_by_video[video_filename]:
                clips.append((video_paths[video_filename], seg))

    if not clips:
        logger.info("No segments for label \"%s\" — skipping reel", label.id)
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Trim each segment into a temporary clip
    trimmed_paths: list[Path] = []
    reel_clips: list[ReelClip] = []

    for i, (video_path, seg) in enumerate(clips):
        trimmed = tmp_dir / f"{label.id}_clip_{i:04d}.mp4"
        duration = seg.end - seg.start

        try:
            run_ffmpeg(
                [
                    "-ss", str(seg.start),
                    "-i", str(video_path),
                    "-t", str(duration),
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    str(trimmed),
                    "-y",
                    "-loglevel", "error",
                ],
                description=f"trim {video_path.name} [{seg.start:.1f}–{seg.end:.1f}]",
            )
            trimmed_paths.append(trimmed)
            reel_clips.append(ReelClip(
                source=video_path.name,
                start=seg.start,
                end=seg.end,
                confidence=seg.confidence,
            ))
        except FFmpegError as e:
            logger.error(
                "Failed to trim %s [%.1f–%.1f]: %s",
                video_path.name, seg.start, seg.end, e,
            )

    if not trimmed_paths:
        logger.error("All clip trims failed for label \"%s\"", label.id)
        return None

    # Step 2: Check if we need to re-encode (different codecs/resolutions)
    needs_reencode = _check_needs_reencode(trimmed_paths)

    if needs_reencode:
        logger.info("Reel \"%s\": inputs differ, re-encoding for compatibility", label.id)
        normalized = _normalize_clips(trimmed_paths, tmp_dir, label.id)
        trimmed_paths = normalized

    # Step 3: Concatenate using concat demuxer
    try:
        _concat_clips(trimmed_paths, output_path, tmp_dir)
    except FFmpegError as e:
        logger.error("Failed to concat reel for \"%s\": %s", label.id, e)
        # Write error log
        error_log = output_path.with_suffix(".error.log")
        error_log.write_text(f"FFmpeg error:\n{e}\n\nstderr:\n{e.stderr}")
        return None

    # Step 4: Clean up trimmed clips
    for p in tmp_dir.glob(f"{label.id}_*"):
        p.unlink(missing_ok=True)

    total_duration = sum(c.end - c.start for c in reel_clips)

    manifest = ReelManifest(
        label=label.id,
        reel_path=str(output_path),
        total_duration_s=round(total_duration, 1),
        clip_count=len(reel_clips),
        clips=reel_clips,
        rendered_at=datetime.now(timezone.utc).isoformat(),
    )

    return manifest


def _check_needs_reencode(clip_paths: list[Path]) -> bool:
    """Check if clips have different codecs or resolutions."""
    if len(clip_paths) <= 1:
        return False

    try:
        first_info = get_video_info(clip_paths[0])
        for p in clip_paths[1:]:
            info = get_video_info(p)
            if (
                info["codec"] != first_info["codec"]
                or info["width"] != first_info["width"]
                or info["height"] != first_info["height"]
            ):
                return True
    except Exception:
        # If we can't probe, assume re-encode needed
        return True

    return False


def _normalize_clips(clip_paths: list[Path], tmp_dir: Path, label_id: str) -> list[Path]:
    """Re-encode all clips to a common format."""
    normalized = []
    for i, p in enumerate(clip_paths):
        out = tmp_dir / f"{label_id}_norm_{i:04d}.mp4"
        try:
            run_ffmpeg(
                [
                    "-i", str(p),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                    "-r", "30",
                    str(out),
                    "-y",
                    "-loglevel", "error",
                ],
                description=f"normalize clip {i}",
            )
            normalized.append(out)
        except FFmpegError as e:
            logger.error("Failed to normalize clip %d: %s", i, e)
    return normalized


def _concat_clips(clip_paths: list[Path], output_path: Path, tmp_dir: Path) -> None:
    """Concatenate clips using the concat demuxer."""
    # Write concat list file
    list_file = tmp_dir / "concat_list.txt"
    lines = [f"file '{p.resolve()}'" for p in clip_paths]
    list_file.write_text("\n".join(lines))

    run_ffmpeg(
        [
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_path),
            "-y",
            "-loglevel", "error",
        ],
        description=f"concat {len(clip_paths)} clips → {output_path.name}",
    )

    list_file.unlink(missing_ok=True)


def write_reel_manifest(manifest: ReelManifest, path: Path) -> None:
    """Write reel manifest JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False)
    )


def merge_adjacent_segments(
    segments: list[LabeledSegment],
    label: LabelDefinition,
) -> list[LabeledSegment]:
    """Merge adjacent segments with the same label if merge_adjacent is True."""
    if not label.merge_adjacent or not segments:
        return segments

    merged = [segments[0].model_copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        # Adjacent if gap is less than 1 second
        if seg.start - prev.end < 1.0:
            prev.end = seg.end
            prev.duration = prev.end - prev.start
            prev.transcript_text = f"{prev.transcript_text} {seg.transcript_text}"
            prev.confidence = min(prev.confidence, seg.confidence)
        else:
            merged.append(seg.model_copy())

    return merged
