"""CLI entry point for Video Insights Labeling tool."""

import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from src.config import Settings, load_settings
from src.context import Context, load_context, resolve_context, generate_context_yaml, PRESETS
from src.models import LabeledSegment, TranscriptSegment
from src.pipeline.subtitles import SUBTITLE_EXTENSIONS
from src.utils.ffmpeg import check_ffmpeg, get_video_duration
from src.utils.files import (
    SUPPORTED_EXTENSIONS,
    cleanup_audio_files,
    cleanup_tmp,
    ensure_output_dirs,
    ensure_video_dir,
    get_labels_path,
    get_reel_manifest_path,
    get_reel_path,
    get_summary_path,
    get_transcript_json_path,
    resolve_video_inputs,
    video_stem,
)
from src.utils.timestamps import format_duration_human

console = Console(stderr=True)
logger = logging.getLogger("vil")

# ── Logging setup ─────────────────────────────────────────────────────────────


def setup_logging(log_level: str, output_dir: Path | None = None) -> None:
    """Configure Rich console + file logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Clear existing handlers
    root.handlers.clear()

    # Rich console handler (stderr)
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
        show_time=False,
        markup=True,
    )
    rich_handler.setLevel(level)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(rich_handler)

    # File handler
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "run.log"
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        root.addHandler(file_handler)


# ── Logging helpers ───────────────────────────────────────────────────────────


def log_info(msg: str) -> None:
    logger.info(msg)


def log_ok(msg: str) -> None:
    logger.info("[green]✓[/green] %s", msg)


def log_warn(msg: str) -> None:
    logger.warning(msg)


def log_error(msg: str) -> None:
    logger.error(msg)


# ── Startup validation ────────────────────────────────────────────────────────


def validate_startup(
    video_paths: list[Path],
    context_input: str | None,
    settings: Settings,
    require_context: bool = True,
    require_api_key: bool = True,
    preset: str | None = None,
) -> tuple[list[Path], Context | None]:
    """Validate all startup preconditions. Exits on fatal errors."""
    errors: list[str] = []

    # Check ffmpeg
    if not check_ffmpeg():
        errors.append("ffmpeg is not found on PATH")

    # Check API key
    if require_api_key and not settings.anthropic_api_key:
        errors.append("ANTHROPIC_API_KEY is not set")

    # Validate video files
    from src.utils.files import validate_video_path
    validated_videos: list[Path] = []
    for vp in video_paths:
        err = validate_video_path(vp)
        if err:
            errors.append(err)
        else:
            validated_videos.append(vp)

    # Load and validate context
    ctx = None
    if require_context and context_input:
        try:
            ctx = resolve_context(context_input, preset=preset)
        except ValueError as e:
            errors.append(str(e))
    elif require_context:
        errors.append("--context is required")

    if errors:
        log_error("Startup validation failed:")
        for e in errors:
            log_error(f"  - {e}")
        sys.exit(1)

    return validated_videos, ctx


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(version="0.1.0", prog_name="video-insights-labeling")
def cli():
    """Video Insights Labeling — transcribe, label, summarize, and render video reels."""
    pass


@cli.command()
@click.option("--videos", "-v", multiple=True, required=True, type=click.Path(), help="Video files or directories")
@click.option("--context", "-c", "context_input", required=True, help="Context: YAML file, markdown file, or inline text")
@click.option("--output", "-o", "output_dir", default="./output", type=click.Path(), help="Output directory")
@click.option("--preset", type=click.Choice(list(PRESETS.keys())), default=None, help="Label preset (required when context is text/markdown)")
@click.option("--whisper-model", default=None, help="Whisper model size override")
@click.option("--skip-transcription", is_flag=True, help="Re-use existing transcripts")
@click.option("--skip-labeling", is_flag=True, help="Re-use existing labels")
@click.option("--skip-reels", is_flag=True, help="Skip video rendering")
@click.option("--no-summaries", is_flag=True, help="Skip summary generation")
@click.option("--labels", multiple=True, help="Only process these label IDs")
@click.option("--dry-run", is_flag=True, help="Print plan without executing")
@click.option("--force", is_flag=True, help="Ignore cached results, rerun everything")
@click.option("--export-clips", is_flag=True, help="Also save each reel segment as a standalone snippet in reels/<label>/")
@click.option("--fast-cuts", is_flag=True, help="Stream-copy cuts (fast, but starts snap to the previous keyframe)")
@click.option("--captions", is_flag=True, help="Burn a headline and subtitles into every clip (needs Pillow)")
def run(
    videos: tuple[str, ...],
    context_input: str,
    output_dir: str,
    preset: str | None,
    whisper_model: str | None,
    skip_transcription: bool,
    skip_labeling: bool,
    skip_reels: bool,
    no_summaries: bool,
    labels: tuple[str, ...],
    dry_run: bool,
    force: bool,
    export_clips: bool,
    fast_cuts: bool,
    captions: bool,
) -> None:
    """Run the full pipeline: transcribe → label → summarize → render."""
    start_time = time.time()
    out = Path(output_dir)

    overrides = {}
    if whisper_model:
        overrides["whisper_model"] = whisper_model
    settings = load_settings(**overrides)

    setup_logging(settings.log_level, out)

    video_paths = resolve_video_inputs(videos)
    if not video_paths:
        log_error("No video files found in the provided paths")
        sys.exit(1)

    validated, ctx = validate_startup(
        video_paths, context_input, settings,
        require_api_key=not dry_run and not (skip_labeling and no_summaries),
        preset=preset,
    )
    assert ctx is not None

    ensure_output_dirs(out)

    # Filter labels if specified
    active_labels = ctx.labels
    if labels:
        active_labels = [l for l in ctx.labels if l.id in labels]
        if not active_labels:
            log_error(f"No matching labels found for: {', '.join(labels)}")
            sys.exit(1)

    log_info(f"Loading context from {context_input}")
    log_info(f"Found {len(validated)} video file(s)")

    if dry_run:
        _print_dry_run(validated, ctx, active_labels, skip_transcription, skip_labeling, skip_reels, no_summaries)
        return

    # ── Stage 1: Transcription ────────────────────────────────────────────
    stats = {"transcribed": 0, "labeled": 0, "summaries": 0, "reels": 0}
    stats_total = {
        "transcribed": len(validated),
        "labeled": len(validated),
        "summaries": len([l for l in active_labels if l.include_in_output]),
        "reels": len([l for l in active_labels if l.include_in_output]),
    }
    warnings: list[str] = []
    errors_list: list[str] = []

    transcripts: dict[str, tuple[list[TranscriptSegment], float]] = {}

    if not skip_transcription:
        diarize = ctx.has_speakers()
        if diarize:
            log_info("Starting transcription with speaker diarization...")
        else:
            log_info("Starting transcription...")
        # Videos with a sidecar subtitle file (<stem>.vtt / <stem>.srt) skip Whisper
        to_whisper: list[Path] = []
        for vp in validated:
            json_path = get_transcript_json_path(out, video_stem(vp))
            sidecar = _find_sidecar_transcript(vp)
            if sidecar and (force or not json_path.exists()):
                log_info(f"{vp.name} — importing {sidecar.name} instead of running Whisper")
                transcripts[vp.name] = _import_transcript(vp, sidecar, out)
            else:
                to_whisper.append(vp)

        from src.workers import run_transcription_pool
        transcripts |= run_transcription_pool(
            to_whisper, out, settings,
            language=ctx.language if ctx.language != "en" else None,
            skip_existing=not force,
            diarize=diarize,
        )
        stats["transcribed"] = len(transcripts)
    else:
        log_info("Skipping transcription (--skip-transcription)")
        from src.pipeline.transcription import load_transcript
        for vp in validated:
            stem = video_stem(vp)
            json_path = get_transcript_json_path(out, stem)
            if json_path.exists():
                segments = load_transcript(json_path)
                duration = get_video_duration(vp)
                transcripts[vp.name] = (segments, duration)
                stats["transcribed"] += 1
            else:
                log_error(f"{vp.name} — no existing transcript found")
                errors_list.append(f"{vp.name} — no transcript")

    # ── Stage 2: Labeling ─────────────────────────────────────────────────
    all_labels: dict[str, list[LabeledSegment]] = {}  # video_name → segments

    if not skip_labeling:
        log_info("Starting labeling...")
        from src.pipeline.labeling import label_transcript, write_labels_json

        for vp in validated:
            if vp.name not in transcripts:
                continue
            segments, duration = transcripts[vp.name]
            stem = video_stem(vp)
            labels_path = get_labels_path(out, stem)

            if not force and labels_path.exists():
                log_info(f"{vp.name} — labels exist, skipping")
                from src.pipeline.labeling import load_labels
                all_labels[vp.name] = load_labels(labels_path)
                stats["labeled"] += 1
                continue

            log_info(f"{vp.name} — labeling with Claude...")
            try:
                labeled = label_transcript(segments, vp.name, duration, ctx, settings)
                write_labels_json(labeled, vp.name, duration, labels_path)
                all_labels[vp.name] = labeled

                label_counts = defaultdict(int)
                for seg in labeled:
                    label_counts[seg.label_id] += 1
                total_segs = len(labeled)
                total_labels = len(label_counts)
                log_ok(f"{vp.name} — labeled ({total_segs} segments across {total_labels} labels)")
                stats["labeled"] += 1

                # Warn on low confidence segments
                low_conf = [s for s in labeled if s.confidence < 0.65]
                if low_conf:
                    warnings.append(f"{vp.name} — {len(low_conf)} segments had confidence < 0.65")

            except Exception as e:
                log_error(f"{vp.name} — labeling failed: {e}")
                errors_list.append(f"{vp.name} — labeling failed")
    else:
        log_info("Skipping labeling (--skip-labeling)")
        from src.pipeline.labeling import load_labels
        for vp in validated:
            stem = video_stem(vp)
            labels_path = get_labels_path(out, stem)
            if labels_path.exists():
                all_labels[vp.name] = load_labels(labels_path)
                stats["labeled"] += 1
            else:
                log_error(f"{vp.name} — no existing labels found")
                errors_list.append(f"{vp.name} — no labels")

    # ── Stage 3: Summaries ────────────────────────────────────────────────
    if not no_summaries:
        log_info("Generating summaries...")
        from src.pipeline.summarizer import generate_summary, write_summary

        for label in active_labels:
            if not label.include_in_output:
                continue

            summary_path = get_summary_path(out, label.id)
            if not force and summary_path.exists():
                log_info(f"summary: {label.id}.md exists, skipping")
                stats["summaries"] += 1
                continue

            # Collect segments for this label
            segments_by_video: dict[str, list[LabeledSegment]] = {}
            for vname, segs in all_labels.items():
                matching = [s for s in segs if s.label_id == label.id]
                if matching:
                    segments_by_video[vname] = matching

            if not segments_by_video:
                log_info(f"No segments for label \"{label.id}\" — skipping summary")
                continue

            try:
                summary = generate_summary(label, segments_by_video, ctx, settings)
                write_summary(summary, summary_path)
                word_count = len(summary.split())
                log_ok(f"summary: {label.id}.md ({word_count} words)")
                stats["summaries"] += 1
            except Exception as e:
                log_error(f"Summary for \"{label.id}\" failed: {e}")
                errors_list.append(f"summary {label.id} — failed")
    else:
        log_info("Skipping summaries (--no-summaries)")
        stats_total["summaries"] = 0

    # ── Stage 4: Reels ────────────────────────────────────────────────────
    if not skip_reels:
        log_info("Rendering reels...")
        from src.pipeline.renderer import render_reel, write_reel_manifest, merge_adjacent_segments
        from src.utils.files import get_tmp_dir

        video_path_map = {vp.name: vp for vp in validated}
        tmp_dir = get_tmp_dir(out)

        for label in active_labels:
            if not label.include_in_output:
                continue

            reel_path = get_reel_path(out, label.id)
            if not force and reel_path.exists():
                log_info(f"reel: {label.id}.mp4 exists, skipping")
                stats["reels"] += 1
                continue

            # Collect and optionally merge segments
            segments_by_video: dict[str, list[LabeledSegment]] = {}
            for vname, segs in all_labels.items():
                matching = sorted(
                    [s for s in segs if s.label_id == label.id],
                    key=lambda s: s.start,
                )
                if matching:
                    matching = merge_adjacent_segments(matching, label)
                    segments_by_video[vname] = matching

            if not segments_by_video:
                log_info(f"No segments for label \"{label.id}\" — skipping reel")
                continue

            try:
                manifest = render_reel(
                    label, segments_by_video, video_path_map, reel_path, tmp_dir,
                    accurate=not fast_cuts,
                    clips_dir=(out / "reels" / label.id) if export_clips else None,
                    **_caption_kwargs(captions, ctx, out, video_path_map),
                )
                if manifest:
                    write_reel_manifest(manifest, get_reel_manifest_path(out, label.id))
                    total_dur = format_duration_human(manifest.total_duration_s)
                    src_count = len(segments_by_video)
                    log_ok(
                        f"reels/{label.id}.mp4 ({total_dur}, "
                        f"{manifest.clip_count} clips from {src_count} video(s))"
                    )
                    stats["reels"] += 1
            except Exception as e:
                log_error(f"Reel for \"{label.id}\" failed: {e}")
                errors_list.append(f"reel {label.id} — failed")

        cleanup_tmp(out)
    else:
        log_info("Skipping reels (--skip-reels)")
        stats_total["reels"] = 0

    # ── Cleanup ───────────────────────────────────────────────────────────
    if not settings.keep_audio_files:
        cleanup_audio_files(out)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    _print_run_summary(stats, stats_total, warnings, errors_list, out, elapsed, len(validated))

    if errors_list:
        sys.exit(1)


@cli.command()
@click.option("--videos", "-v", multiple=True, required=True, type=click.Path(), help="Video files or directories")
@click.option("--output", "-o", "output_dir", default="./output", type=click.Path())
@click.option("--model", default=None, help="Whisper model size")
def transcribe(videos: tuple[str, ...], output_dir: str, model: str | None) -> None:
    """Transcribe videos only (no labeling or rendering)."""
    out = Path(output_dir)
    overrides = {}
    if model:
        overrides["whisper_model"] = model
    settings = load_settings(**overrides)

    setup_logging(settings.log_level, out)

    video_paths = resolve_video_inputs(videos)
    if not video_paths:
        log_error("No video files found in the provided paths")
        sys.exit(1)

    validated, _ = validate_startup(
        video_paths, None, settings, require_context=False, require_api_key=False,
    )

    ensure_output_dirs(out)
    log_info(f"Transcribing {len(validated)} video(s)...")

    from src.workers import run_transcription_pool
    results = run_transcription_pool(validated, out, settings)

    log_ok(f"Transcribed {len(results)} / {len(validated)} video(s)")
    if len(results) < len(validated):
        sys.exit(1)


@cli.command()
@click.option("--output", "-o", "output_dir", required=True, type=click.Path(exists=True))
@click.option("--context", "-c", "context_input", required=True, help="Context: YAML file, markdown file, or inline text")
@click.option("--preset", type=click.Choice(list(PRESETS.keys())), default=None, help="Label preset (required when context is text/markdown)")
@click.option("--videos", "-v", multiple=True, type=click.Path(), help="Source videos or directories (used for exact durations)")
def label(output_dir: str, context_input: str, preset: str | None, videos: tuple[str, ...]) -> None:
    """Label existing transcripts (requires previous transcription)."""
    out = Path(output_dir)
    settings = load_settings()
    setup_logging(settings.log_level, out)

    _, ctx = validate_startup([], context_input, settings, require_api_key=True, preset=preset)
    assert ctx is not None

    from src.pipeline.transcription import load_transcript
    from src.pipeline.labeling import label_transcript, write_labels_json

    # Find videos with existing transcripts
    labeled_count = 0
    for video_dir in sorted(out.iterdir()):
        if not video_dir.is_dir() or video_dir.name in ("summaries", "reels", "tmp"):
            continue

        json_path = video_dir / "transcript.json"
        if not json_path.exists():
            continue

        stem = video_dir.name
        labels_path = get_labels_path(out, stem)
        log_info(f"{stem} — labeling with Claude...")

        segments = load_transcript(json_path)

        source = _find_source_video(stem, out, videos)
        duration = get_video_duration(source) if source else 0.0
        video_name = source.name if source else f"{stem}.mp4"

        if duration == 0 and segments:
            duration = segments[-1].end + 1.0

        try:
            labeled = label_transcript(segments, video_name, duration, ctx, settings)
            write_labels_json(labeled, video_name, duration, labels_path)
            log_ok(f"{stem} — labeled ({len(labeled)} segments)")
            labeled_count += 1
        except Exception as e:
            log_error(f"{stem} — labeling failed: {e}")

    log_ok(f"Labeled {labeled_count} video(s)")


@cli.command()
@click.option("--output", "-o", "output_dir", required=True, type=click.Path(exists=True))
@click.option("--context", "-c", "context_input", required=True, help="Context: YAML file, markdown file, or inline text")
@click.option("--preset", type=click.Choice(list(PRESETS.keys())), default=None, help="Label preset (required when context is text/markdown)")
@click.option("--labels", multiple=True, help="Only render these label IDs")
@click.option("--videos", "-v", multiple=True, type=click.Path(), help="Source videos or directories (default: look next to the output dir and in cwd)")
@click.option("--export-clips", is_flag=True, help="Also save each segment as a standalone snippet in reels/<label>/")
@click.option("--fast-cuts", is_flag=True, help="Stream-copy cuts (fast, but starts snap to the previous keyframe)")
@click.option("--captions", is_flag=True, help="Burn a headline and subtitles into every clip (needs Pillow)")
def render(
    output_dir: str,
    context_input: str,
    preset: str | None,
    labels: tuple[str, ...],
    videos: tuple[str, ...],
    export_clips: bool,
    fast_cuts: bool,
    captions: bool,
) -> None:
    """Render reels from existing labels (requires previous labeling)."""
    out = Path(output_dir)
    settings = load_settings()
    setup_logging(settings.log_level, out)

    _, ctx = validate_startup([], context_input, settings, require_context=True, require_api_key=False, preset=preset)
    assert ctx is not None

    from src.pipeline.labeling import load_labels
    from src.pipeline.renderer import render_reel, write_reel_manifest, merge_adjacent_segments
    from src.utils.files import get_tmp_dir

    # Load all labels
    all_labels: dict[str, list[LabeledSegment]] = {}
    video_path_map: dict[str, Path] = {}

    for video_dir in sorted(out.iterdir()):
        if not video_dir.is_dir() or video_dir.name in ("summaries", "reels", "tmp"):
            continue
        labels_file = video_dir / "labels.json"
        if not labels_file.exists():
            continue
        stem = video_dir.name
        source = _find_source_video(stem, out, videos)
        if source is None:
            log_warn(f"{stem} — source video not found, skipping (pass it with --videos)")
            continue
        video_path_map[source.name] = source
        all_labels[source.name] = load_labels(labels_file)

    active_labels = ctx.labels
    if labels:
        active_labels = [l for l in ctx.labels if l.id in labels]

    tmp_dir = get_tmp_dir(out)
    rendered = 0
    caption_kwargs = _caption_kwargs(captions, ctx, out, video_path_map)

    for lbl in active_labels:
        if not lbl.include_in_output:
            continue

        segments_by_video: dict[str, list[LabeledSegment]] = {}
        for vname, segs in all_labels.items():
            matching = sorted([s for s in segs if s.label_id == lbl.id], key=lambda s: s.start)
            if matching:
                matching = merge_adjacent_segments(matching, lbl)
                segments_by_video[vname] = matching

        if not segments_by_video:
            continue

        reel_path = get_reel_path(out, lbl.id)
        manifest = render_reel(
            lbl, segments_by_video, video_path_map, reel_path, tmp_dir,
            accurate=not fast_cuts,
            clips_dir=(out / "reels" / lbl.id) if export_clips else None,
            **caption_kwargs,
        )
        if manifest:
            write_reel_manifest(manifest, get_reel_manifest_path(out, lbl.id))
            log_ok(f"reels/{lbl.id}.mp4 rendered")
            rendered += 1

    cleanup_tmp(out)
    log_ok(f"Rendered {rendered} reel(s)")


@cli.command("import-transcript")
@click.option("--video", "-v", "video", required=True, type=click.Path(exists=True, dir_okay=False), help="Source video file")
@click.option("--transcript", "-t", "transcript", required=True, type=click.Path(exists=True, dir_okay=False), help="Subtitle transcript (.vtt or .srt)")
@click.option("--output", "-o", "output_dir", default="./output", type=click.Path())
def import_transcript(video: str, transcript: str, output_dir: str) -> None:
    """Import a meeting transcript (.vtt/.srt) instead of running Whisper.

    Speaker names in the file (e.g. "Jane Doe: ...") are kept. Afterwards, run
    `vil run ... --skip-transcription` or `vil label`.
    """
    out = Path(output_dir)
    settings = load_settings()
    setup_logging(settings.log_level, out)

    tp = Path(transcript)
    if tp.suffix.lower() not in SUBTITLE_EXTENSIONS:
        log_error(f"Unsupported transcript format \"{tp.suffix}\" (supported: .vtt, .srt)")
        sys.exit(1)

    try:
        segments, _ = _import_transcript(Path(video), tp, out)
    except ValueError as e:
        log_error(str(e))
        sys.exit(1)
    log_ok(f"{Path(video).name} — imported {len(segments)} segments from {tp.name}")


@cli.command("init-context")
@click.option("--output", "-o", default="context.yaml", type=click.Path(), help="Output path")
@click.option(
    "--preset",
    type=click.Choice(list(PRESETS.keys())),
    default=None,
    help="Preset taxonomy template",
)
def init_context(output: str, preset: str | None) -> None:
    """Generate a starter context.yaml file."""
    out = Path(output)
    if out.exists():
        click.confirm(f"{out} already exists. Overwrite?", abort=True)

    yaml_text = generate_context_yaml(preset)
    out.write_text(yaml_text)

    preset_msg = f" (preset: {preset})" if preset else ""
    click.echo(f"Created {out}{preset_msg}")
    click.echo("Edit the file to match your project, then run:")
    click.echo(f"  vil run --videos <files> --context {out}")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_source_video(stem: str, out: Path, videos: tuple[str, ...] = ()) -> Path | None:
    """Locate the source video for an output folder stem.

    Looks in the explicit --videos inputs first, then next to the output dir
    and in the current directory.
    """
    for vp in resolve_video_inputs(videos):
        if video_stem(vp) == stem and vp.exists():
            return vp
    for ext in sorted(SUPPORTED_EXTENSIONS):
        for candidate in (out.parent / f"{stem}{ext}", Path(f"{stem}{ext}")):
            if candidate.exists():
                return candidate
    return None


def _caption_kwargs(captions: bool, ctx: Context, out: Path, video_path_map: dict[str, Path]) -> dict:
    """Build render_reel caption arguments: transcripts and interviewer speakers."""
    if not captions:
        return {}
    from src.pipeline.transcription import load_transcript

    transcripts = {}
    for name, path in video_path_map.items():
        json_path = get_transcript_json_path(out, video_stem(path))
        if json_path.exists():
            segments = load_transcript(json_path)
            for seg in segments:
                seg.text = ctx.apply_corrections(seg.text)
            transcripts[name] = segments
        else:
            log_warn(f"{name} — no transcript.json, clips get a headline but no subtitles")
    secondary = {s.id for s in ctx.interviewers()} | {s.name for s in ctx.interviewers() if s.name}
    return {"captions": True, "transcripts": transcripts, "secondary_speakers": secondary}


def _find_sidecar_transcript(video: Path) -> Path | None:
    """Return <stem>.vtt or <stem>.srt next to the video, if present."""
    for ext in sorted(SUBTITLE_EXTENSIONS):
        candidate = video.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _import_transcript(video: Path, transcript: Path, out: Path) -> tuple[list[TranscriptSegment], float]:
    """Parse a subtitle file and write transcript.json/.txt for the video."""
    from src.pipeline.subtitles import parse_subtitles
    from src.pipeline.transcription import write_transcript_json, write_transcript_txt
    from src.utils.files import get_transcript_txt_path

    segments = parse_subtitles(transcript)
    duration = get_video_duration(video)
    if segments[-1].end > duration + 5:
        log_warn(
            f"{transcript.name} ends at {segments[-1].end:.0f}s but {video.name} is "
            f"{duration:.0f}s long — is it the right transcript?"
        )
    stem = video_stem(video)
    ensure_video_dir(out, stem)
    write_transcript_json(segments, get_transcript_json_path(out, stem))
    write_transcript_txt(segments, get_transcript_txt_path(out, stem), video.name, duration)
    return segments, duration



def _print_dry_run(
    videos: list[Path],
    ctx: Context,
    active_labels: list,
    skip_transcription: bool,
    skip_labeling: bool,
    skip_reels: bool,
    no_summaries: bool,
) -> None:
    """Print what would be done without doing it."""
    console.print("\n[bold]Dry run — nothing will be executed[/bold]\n")

    console.print(f"  Project:  {ctx.project_name}")
    console.print(f"  Videos:   {len(videos)}")
    for v in videos:
        console.print(f"            - {v.name}")
    console.print(f"  Labels:   {len(active_labels)}")
    for l in active_labels:
        tag = "" if l.include_in_output else " (excluded from output)"
        console.print(f"            - {l.id}: {l.name}{tag}")
    console.print()

    steps = []
    if not skip_transcription:
        steps.append(f"1. Transcribe {len(videos)} video(s)")
    if not skip_labeling:
        steps.append(f"2. Label transcripts with Claude")
    if not no_summaries:
        out_labels = [l for l in active_labels if l.include_in_output]
        steps.append(f"3. Generate {len(out_labels)} summary/summaries")
    if not skip_reels:
        out_labels = [l for l in active_labels if l.include_in_output]
        steps.append(f"4. Render {len(out_labels)} reel(s)")

    console.print("  Pipeline steps:")
    for s in steps:
        console.print(f"    {s}")
    console.print()


def _print_run_summary(
    stats: dict,
    totals: dict,
    warnings: list[str],
    errors: list[str],
    output_dir: Path,
    elapsed: float,
    video_count: int,
) -> None:
    """Print end-of-run summary."""
    console.print()
    console.print("─" * 40)
    console.print(f"Run complete — {video_count} video(s) processed")
    console.print()

    for key, label in [
        ("transcribed", "Transcribed"),
        ("labeled", "Labeled"),
        ("summaries", "Summaries"),
        ("reels", "Reels"),
    ]:
        total = totals.get(key, 0)
        if total > 0:
            console.print(f"  {label + ':':14s} {stats.get(key, 0)} / {total}")

    console.print()

    if warnings:
        console.print(f"  [yellow]Warnings:  {len(warnings)}[/yellow]")
        for w in warnings:
            console.print(f"    {w}")
        console.print()

    if errors:
        console.print(f"  [red]Errors:    {len(errors)}[/red]")
        for e in errors:
            console.print(f"    {e}")
    else:
        console.print("  Errors:    0")

    console.print()
    console.print(f"  Finished in {format_duration_human(elapsed)}. Output in {output_dir}")
    console.print("─" * 40)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
