"""FFmpeg-based reel rendering — trim and concatenate segments per label."""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.context import LabelDefinition
from src.models import LabeledSegment, ReelClip, ReelManifest, TranscriptSegment
from src.pipeline.captions import (
    CaptionLayout,
    build_caption_lines,
    captions_available,
    overlay_ffmpeg_args,
    render_overlays,
)
from src.utils.ffmpeg import run_ffmpeg, get_video_info, FFmpegError

logger = logging.getLogger(__name__)


def render_reel(
    label: LabelDefinition,
    segments_by_video: dict[str, list[LabeledSegment]],
    video_paths: dict[str, Path],
    output_path: Path,
    tmp_dir: Path,
    accurate: bool = True,
    clips_dir: Path | None = None,
    captions: bool = False,
    transcripts: dict[str, list[TranscriptSegment]] | None = None,
    secondary_speakers: set[str] | None = None,
) -> ReelManifest | None:
    """Render a reel by trimming and concatenating all segments for a label.

    Args:
        label: The label definition.
        segments_by_video: Dict mapping video filename → segments for this label.
        video_paths: Dict mapping video filename → full path to video file.
        output_path: Where to write the output .mp4.
        tmp_dir: Directory for temporary trimmed clips.
        accurate: Re-encode each clip so cuts land on the exact timestamp.
            Stream copy (accurate=False) is faster but snaps the start to the
            previous keyframe, which can be tens of seconds early in screen
            recordings with sparse keyframes.
        clips_dir: If set, also keep every trimmed clip as a standalone file here.
        captions: Burn a headline (segment title, or reason as fallback) on top
            and subtitles from `transcripts` at the bottom. Implies accurate cuts.
        transcripts: Dict mapping video filename → transcript, for subtitles.
        secondary_speakers: Speakers drawn in a softer color with a name prefix
            (e.g. interviewers).

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

    if captions and not captions_available():
        logger.warning("Captions need Pillow (pip install pillow) — rendering without captions")
        captions = False
    video_sizes: dict[Path, tuple[int, int]] = {}

    for i, (video_path, seg) in enumerate(clips):
        trimmed = tmp_dir / f"{label.id}_clip_{i:04d}.mp4"
        duration = seg.end - seg.start

        layout: CaptionLayout | None = None
        if captions:
            if video_path not in video_sizes:
                info = get_video_info(video_path)
                video_sizes[video_path] = (info["width"], info["height"])
            width, height = video_sizes[video_path]
            lines = build_caption_lines(
                (transcripts or {}).get(video_path.name, []),
                seg.start, seg.end, secondary_speakers,
            )
            layout = render_overlays(
                seg.title or seg.reason, label.name, label.color, lines,
                width, height, tmp_dir, f"{label.id}_{i:04d}",
            )

        try:
            run_ffmpeg(
                _trim_args(video_path, seg.start, duration, trimmed, accurate or captions, layout),
                description=f"trim {video_path.name} [{seg.start:.1f}–{seg.end:.1f}]",
            )
            trimmed_paths.append(trimmed)
            reel_clips.append(ReelClip(
                source=video_path.name,
                start=seg.start,
                end=seg.end,
                confidence=seg.confidence,
                reason=seg.reason,
                title=seg.title,
                speaker=seg.speaker,
            ))
        except FFmpegError as e:
            logger.error(
                "Failed to trim %s [%.1f–%.1f]: %s",
                video_path.name, seg.start, seg.end, e,
            )

    if not trimmed_paths:
        logger.error("All clip trims failed for label \"%s\"", label.id)
        return None

    if clips_dir is not None:
        _export_clips(trimmed_paths, reel_clips, clips_dir)

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


def _trim_args(
    video_path: Path,
    start: float,
    duration: float,
    out: Path,
    accurate: bool,
    layout: CaptionLayout | None = None,
) -> list[str]:
    """Build ffmpeg args to cut [start, start+duration] from a video."""
    args = ["-ss", f"{start:.3f}", "-i", str(video_path)]
    video_map = "0:v:0"
    if layout:
        extra_inputs, graph, video_map = overlay_ffmpeg_args(layout)
        args += extra_inputs + ["-filter_complex", graph]
    args += ["-t", f"{duration:.3f}"]
    if accurate or layout:
        args += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k",
            "-map", video_map, "-map", "0:a:0?",
            "-map_metadata", "-1", "-dn",
            "-movflags", "+faststart",
        ]
    else:
        args += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    return args + [str(out), "-y", "-loglevel", "error"]


def _clip_filename(index: int, clip: ReelClip) -> str:
    def ts(t: float) -> str:
        t = int(t)
        return f"{t // 3600:02d}{t % 3600 // 60:02d}{t % 60:02d}"
    return f"{index:02d}_{Path(clip.source).stem}_{ts(clip.start)}-{ts(clip.end)}.mp4"


def _export_clips(trimmed: list[Path], clips: list[ReelClip], clips_dir: Path) -> None:
    """Copy each trimmed clip to clips_dir as a standalone snippet."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("*.mp4"):
        old.unlink()
    for i, (path, clip) in enumerate(zip(trimmed, clips), start=1):
        dest = clips_dir / _clip_filename(i, clip)
        shutil.copy2(path, dest)
        clip.clip_path = str(dest)


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
