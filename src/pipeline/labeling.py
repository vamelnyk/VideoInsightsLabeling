"""Claude-based transcript labeling with validation and repair."""

import json
import logging
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import anthropic

from src.config import Settings
from src.context import Context
from src.models import LabeledSegment, TranscriptSegment
from src.utils.timestamps import format_timestamp, format_duration

logger = logging.getLogger(__name__)

# Chunking constants
CHUNK_DURATION_S = 2700  # 45 minutes
OVERLAP_S = 300  # 5 minutes
MAX_TRANSCRIPT_CHARS = 100_000


def label_transcript(
    segments: list[TranscriptSegment],
    video_filename: str,
    video_duration: float,
    context: Context,
    settings: Settings,
) -> list[LabeledSegment]:
    """Label all transcript segments using Claude.

    Handles long-video chunking, response parsing, and validation/repair.
    """
    transcript_text = _build_transcript_text(segments)

    if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
        logger.info(
            "%s — transcript is %d chars, chunking for labeling",
            video_filename, len(transcript_text),
        )
        return _label_chunked(segments, video_filename, video_duration, context, settings)

    system_prompt = _build_system_prompt(context)
    user_prompt = _build_user_prompt(segments, video_filename, video_duration, context)

    raw_response = _call_claude(system_prompt, user_prompt, settings)
    labeled = _parse_response(raw_response, video_filename, settings)

    if labeled is None:
        # Retry once
        logger.warning("%s — first Claude response was not valid JSON, retrying", video_filename)
        retry_prompt = user_prompt + (
            "\n\nYour previous response was not valid JSON. "
            "Return ONLY the JSON array, no other text."
        )
        raw_response = _call_claude(system_prompt, retry_prompt, settings)
        labeled = _parse_response(raw_response, video_filename, settings)

        if labeled is None:
            raise ValueError(f"Claude returned unparseable JSON for {video_filename} after 2 attempts")

    # Validation and repair
    labeled = _validate_label_ids(labeled, context, video_filename)
    labeled = _resolve_overlaps(labeled, video_filename)
    labeled = _fill_coverage_gaps(labeled, video_duration, video_filename)
    labeled.sort(key=lambda s: s.start)

    # Attach transcript text to each segment
    labeled = _attach_transcript_text(labeled, segments)

    return labeled


def _build_system_prompt(context: Context) -> str:
    base = (
        "You are an expert video editor and content analyst. Your task is to analyse "
        "a video transcript and assign labels to time segments according to the user's "
        "defined taxonomy.\n\n"
        "Rules:\n"
        "- Every second of the video must be covered by exactly one label.\n"
        "- Prefer longer segments — do not split what is naturally continuous content.\n"
        "- Return a JSON array ONLY. No preamble, no markdown fences, no explanation "
        "outside the JSON.\n"
        '- The "reason" field must be exactly one sentence.\n'
        '- The "confidence" field is a float from 0.0 to 1.0.\n'
        '- The "speaker" field must identify who is primarily speaking in the segment.\n'
        "- Use ONLY the label IDs provided. Any other value is invalid."
    )

    if context.has_speakers():
        speaker_rules = "\n\nSpeaker awareness:\n"
        speaker_rules += "- This video has multiple speakers with distinct roles.\n"
        speaker_rules += "- Pay close attention to WHO is speaking when assigning labels.\n"

        interviewees = context.interviewees()
        if interviewees:
            speaker_rules += (
                "- The interviewee's words carry the primary value — their insights, "
                "stories, reactions, and opinions are the most important content.\n"
                "- When labeling insights or highlights, strongly prefer moments where "
                "the INTERVIEWEE is speaking.\n"
                "- The interviewer's questions provide structure but are secondary content.\n"
            )

        base += speaker_rules

    return base


def _build_user_prompt(
    segments: list[TranscriptSegment],
    video_filename: str,
    video_duration: float,
    context: Context,
) -> str:
    parts = []

    # Context block
    parts.append("## Project context\n")
    parts.append(f"Project: {context.project_name}")
    if context.description:
        parts.append(f"Description: {context.description}")
    if context.video_type:
        parts.append(f"Video type: {context.video_type}")
    if context.audience:
        parts.append(f"Audience: {context.audience}")
    if context.goals:
        parts.append(f"Goals: {', '.join(context.goals)}")
    parts.append(f"Language: {context.language}")
    if context.notes:
        parts.append(f"Notes: {context.notes}")

    parts.append(f"\n## This video\n")
    parts.append(f"Filename: {video_filename}")
    parts.append(f"Duration: {format_duration(video_duration)}")

    # Speakers block
    if context.has_speakers():
        parts.append("\n## Speakers\n")
        parts.append("This video has the following speakers:\n")
        for speaker in context.speakers:
            role_tag = f" ({speaker.role})" if speaker.role else ""
            name_tag = f" — {speaker.name}" if speaker.name else ""
            parts.append(f"- **{speaker.id}**{name_tag}{role_tag}")
            if speaker.description:
                parts.append(f"  {speaker.description}")
        parts.append("")

    # Label taxonomy block
    parts.append("\n## Label taxonomy\n")
    parts.append("Use ONLY these label IDs. Each segment must have exactly one.\n")
    for label in context.labels:
        parts.append(f"**{label.id}** — {label.name}")
        parts.append(f"{label.description}\n")

    # Transcript block
    has_speakers = any(seg.speaker for seg in segments)
    parts.append("## Transcript\n")
    if has_speakers:
        parts.append("Timestamps are in [HH:MM:SS] format. Speaker labels are from diarization.\n")
    else:
        parts.append("Timestamps are in [HH:MM:SS] format.\n")
        if context.has_speakers():
            parts.append(
                "Note: no automatic speaker labels are available. "
                "Infer who is speaking from conversational patterns "
                "(questions vs answers, tone, and context).\n"
            )

    for seg in segments:
        if seg.speaker:
            parts.append(f"[{format_timestamp(seg.start)}] {seg.speaker}: {seg.text}")
        else:
            parts.append(f"[{format_timestamp(seg.start)}] {seg.text}")

    return "\n".join(parts)


def _build_transcript_text(segments: list[TranscriptSegment]) -> str:
    return "\n".join(f"[{format_timestamp(s.start)}] {s.text}" for s in segments)


def _call_claude(system_prompt: str, user_prompt: str, settings: Settings) -> str:
    """Call Claude API with retry on rate limit."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=settings.claude_model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text
            logger.debug(
                "Claude response: %d chars, %d input tokens, %d output tokens",
                len(text),
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            return text
        except anthropic.RateLimitError:
            wait = (2 ** attempt) * 5
            logger.warning("Rate limited, waiting %ds before retry %d/3", wait, attempt + 1)
            time.sleep(wait)
        except anthropic.APIError as e:
            logger.error("Claude API error: %s", e)
            raise

    raise RuntimeError("Claude API rate limit exceeded after 3 retries")


def _parse_response(raw: str, video_filename: str, settings: Settings) -> list[LabeledSegment] | None:
    """Parse Claude's JSON response into LabeledSegment list."""
    # Try to extract JSON array from the response
    text = raw.strip()

    # Remove markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to find a JSON array
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        text = array_match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from Claude response for %s", video_filename)
        return None

    if not isinstance(data, list):
        logger.warning("Claude response is not a JSON array for %s", video_filename)
        return None

    segments = []
    for item in data:
        try:
            seg = LabeledSegment(
                label_id=item.get("label", item.get("label_id", "")),
                start=float(item.get("start", 0)),
                end=float(item.get("end", 0)),
                duration=float(item.get("end", 0)) - float(item.get("start", 0)),
                reason=item.get("reason", ""),
                confidence=float(item.get("confidence", 0.5)),
                transcript_text=item.get("transcript_text", ""),
                speaker=item.get("speaker", ""),
                speaker_role=item.get("speaker_role", ""),
            )
            segments.append(seg)
        except (ValueError, TypeError) as e:
            logger.warning("Skipping invalid segment in %s: %s", video_filename, e)

    return segments if segments else None


def _validate_label_ids(
    segments: list[LabeledSegment],
    context: Context,
    video_filename: str,
) -> list[LabeledSegment]:
    """Replace unknown label IDs with the closest valid match."""
    valid_ids = set(context.label_ids())

    for seg in segments:
        if seg.label_id not in valid_ids:
            # Find closest match
            best_match = max(
                valid_ids,
                key=lambda vid: SequenceMatcher(None, seg.label_id.lower(), vid).ratio(),
            )
            logger.warning(
                "%s — unknown label \"%s\" → replaced with \"%s\"",
                video_filename, seg.label_id, best_match,
            )
            seg.label_id = best_match

    return segments


def _resolve_overlaps(
    segments: list[LabeledSegment],
    video_filename: str,
) -> list[LabeledSegment]:
    """Remove overlaps by keeping the higher-confidence segment."""
    if not segments:
        return segments

    segments.sort(key=lambda s: s.start)
    resolved = [segments[0]]

    for seg in segments[1:]:
        prev = resolved[-1]

        if seg.start < prev.end:
            overlap = prev.end - seg.start
            if overlap > 0.5:
                logger.warning(
                    "%s — overlap of %.1fs between segments at %.1f and %.1f",
                    video_filename, overlap, prev.start, seg.start,
                )

            if seg.confidence >= prev.confidence:
                # Trim previous segment
                prev.end = seg.start
                prev.duration = prev.end - prev.start
                resolved.append(seg)
            else:
                # Trim current segment
                seg.start = prev.end
                seg.duration = seg.end - seg.start
                if seg.duration > 0.1:
                    resolved.append(seg)
        else:
            resolved.append(seg)

    return resolved


def _fill_coverage_gaps(
    segments: list[LabeledSegment],
    video_duration: float,
    video_filename: str,
) -> list[LabeledSegment]:
    """Fill gaps in coverage by extending the nearest segment."""
    if not segments:
        return segments

    segments.sort(key=lambda s: s.start)

    # Fill gap at the start
    if segments[0].start > 0.5:
        logger.warning(
            "%s — gap of %.1fs at video start, extending first segment",
            video_filename, segments[0].start,
        )
        segments[0].start = 0.0
        segments[0].duration = segments[0].end - segments[0].start

    # Fill gaps between segments
    for i in range(len(segments) - 1):
        gap = segments[i + 1].start - segments[i].end
        if gap > 0.5:
            logger.warning(
                "%s — gap of %.1fs between %.1f and %.1f, extending previous segment",
                video_filename, gap, segments[i].end, segments[i + 1].start,
            )
        if gap > 0:
            segments[i].end = segments[i + 1].start
            segments[i].duration = segments[i].end - segments[i].start

    # Fill gap at the end
    if segments[-1].end < video_duration - 0.5:
        logger.warning(
            "%s — gap of %.1fs at video end, extending last segment",
            video_filename, video_duration - segments[-1].end,
        )
    segments[-1].end = video_duration
    segments[-1].duration = segments[-1].end - segments[-1].start

    return segments


def _attach_transcript_text(
    labeled: list[LabeledSegment],
    transcript: list[TranscriptSegment],
) -> list[LabeledSegment]:
    """Fill in transcript_text and speaker info for each labeled segment."""
    for seg in labeled:
        texts = []
        speaker_time: dict[str, float] = {}
        for tseg in transcript:
            # Segment overlaps with labeled region
            if tseg.end > seg.start and tseg.start < seg.end:
                if tseg.speaker:
                    prefix = f"{tseg.speaker}: "
                    texts.append(f"{prefix}{tseg.text}")
                    overlap = min(tseg.end, seg.end) - max(tseg.start, seg.start)
                    speaker_time[tseg.speaker] = speaker_time.get(tseg.speaker, 0) + overlap
                else:
                    texts.append(tseg.text)
        seg.transcript_text = " ".join(texts)

        # Set speaker to the one who spoke the most in this segment
        if speaker_time and not seg.speaker:
            seg.speaker = max(speaker_time, key=speaker_time.get)
    return labeled


def _label_chunked(
    segments: list[TranscriptSegment],
    video_filename: str,
    video_duration: float,
    context: Context,
    settings: Settings,
) -> list[LabeledSegment]:
    """Label a long video by splitting into overlapping chunks."""
    chunks = _chunk_segments(segments, video_duration)
    all_labeled: list[LabeledSegment] = []

    for chunk_idx, (chunk_start, chunk_end, chunk_segs) in enumerate(chunks):
        logger.info(
            "%s — labeling chunk %d/%d (%.0f–%.0fs)",
            video_filename, chunk_idx + 1, len(chunks), chunk_start, chunk_end,
        )

        system_prompt = _build_system_prompt(context)
        user_prompt = _build_user_prompt(chunk_segs, video_filename, chunk_end - chunk_start, context)
        user_prompt += f"\n\nNote: this is a chunk of a longer video, covering {format_timestamp(chunk_start)} to {format_timestamp(chunk_end)}."

        raw_response = _call_claude(system_prompt, user_prompt, settings)
        chunk_labeled = _parse_response(raw_response, video_filename, settings)

        if chunk_labeled is None:
            logger.error("%s — chunk %d returned unparseable JSON, skipping", video_filename, chunk_idx + 1)
            continue

        # Adjust timestamps to absolute positions
        for seg in chunk_labeled:
            seg.start += chunk_start
            seg.end += chunk_start
            seg.duration = seg.end - seg.start

        all_labeled.extend(chunk_labeled)

    # Resolve overlaps from chunk boundaries
    all_labeled = _validate_label_ids(all_labeled, context, video_filename)
    all_labeled = _resolve_overlaps(all_labeled, video_filename)
    all_labeled = _fill_coverage_gaps(all_labeled, video_duration, video_filename)
    all_labeled = _attach_transcript_text(all_labeled, segments)

    return all_labeled


def _chunk_segments(
    segments: list[TranscriptSegment],
    video_duration: float,
) -> list[tuple[float, float, list[TranscriptSegment]]]:
    """Split transcript into overlapping chunks."""
    chunks = []
    pos = 0.0

    while pos < video_duration:
        chunk_end = min(pos + CHUNK_DURATION_S, video_duration)
        chunk_segs = [s for s in segments if s.start >= pos and s.start < chunk_end]
        chunks.append((pos, chunk_end, chunk_segs))
        pos += CHUNK_DURATION_S - OVERLAP_S

        if chunk_end >= video_duration:
            break

    return chunks


def write_labels_json(
    labeled_segments: list[LabeledSegment],
    video_filename: str,
    video_duration: float,
    path: Path,
) -> None:
    """Write labels.json for a video."""
    # Build label summary
    label_summary: dict[str, dict[str, float]] = {}
    for seg in labeled_segments:
        if seg.label_id not in label_summary:
            label_summary[seg.label_id] = {"count": 0, "total_duration_s": 0.0}
        label_summary[seg.label_id]["count"] += 1
        label_summary[seg.label_id]["total_duration_s"] = round(
            label_summary[seg.label_id]["total_duration_s"] + seg.duration, 1
        )

    data = {
        "video": video_filename,
        "duration_s": round(video_duration, 1),
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "label_summary": label_summary,
        "segments": [seg.model_dump() for seg in labeled_segments],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_labels(path: Path) -> list[LabeledSegment]:
    """Load labeled segments from a labels.json file."""
    data = json.loads(path.read_text())
    return [LabeledSegment(**seg) for seg in data["segments"]]
