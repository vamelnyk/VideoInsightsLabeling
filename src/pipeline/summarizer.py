"""Per-label summary generation using Claude."""

import logging
from pathlib import Path

import anthropic

from src.config import Settings
from src.context import Context, LabelDefinition
from src.models import LabeledSegment
from src.utils.timestamps import format_timestamp, format_duration

logger = logging.getLogger(__name__)


def generate_summary(
    label: LabelDefinition,
    segments_by_video: dict[str, list[LabeledSegment]],
    context: Context,
    settings: Settings,
) -> str:
    """Generate a markdown summary for a label across all videos.

    Args:
        label: The label definition.
        segments_by_video: Dict mapping video filename → list of segments with this label.
        context: Project context.
        settings: Application settings.

    Returns:
        Markdown summary text.
    """
    # Count totals
    total_videos = len(segments_by_video)
    total_segments = sum(len(segs) for segs in segments_by_video.values())
    total_duration = sum(s.duration for segs in segments_by_video.values() for s in segs)

    # Build prompt
    prompt = _build_summary_prompt(label, segments_by_video, context, total_videos)

    system_msg = (
        "You are summarising video content. Write clear, well-structured markdown. "
        "Do not pad or repeat yourself."
    )
    if context.has_speakers():
        system_msg += (
            " These videos feature multiple speakers. "
            "Always attribute quotes and insights to the specific speaker. "
            "Focus on what the interviewee(s) said — their insights, stories, "
            "and reactions are the most valuable content."
        )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=4096,
        system=system_msg,
        messages=[{"role": "user", "content": prompt}],
    )

    summary_body = response.content[0].text

    # Build full markdown file
    header = (
        f"# {label.name} — Summary\n\n"
        f"*Across {total_videos} video{'s' if total_videos != 1 else ''} · "
        f"{total_segments} segment{'s' if total_segments != 1 else ''} · "
        f"{format_duration(total_duration)} total*\n\n"
    )

    return header + summary_body


def _build_summary_prompt(
    label: LabelDefinition,
    segments_by_video: dict[str, list[LabeledSegment]],
    context: Context,
    total_videos: int,
) -> str:
    parts = [
        f'You are summarising video content. Below are all transcript segments '
        f'labeled "{label.name}" across {total_videos} video(s).',
        "",
        f"Project context: {context.description}",
        f"Label definition: {label.description}",
        "",
    ]

    # Speaker context
    if context.has_speakers():
        parts.append("## Speakers")
        for speaker in context.speakers:
            role_tag = f" ({speaker.role})" if speaker.role else ""
            name_tag = speaker.name if speaker.name else speaker.id
            desc = f" — {speaker.description}" if speaker.description else ""
            parts.append(f"- {name_tag}{role_tag}{desc}")
        parts.append("")

    parts.append("Segments:")
    parts.append("---")

    for video_filename, segs in segments_by_video.items():
        for seg in segs:
            header = f"[{video_filename} — {format_timestamp(seg.start)}–{format_timestamp(seg.end)}]"
            if seg.speaker:
                speaker_name = context.speaker_display_name(seg.speaker)
                header += f" ({speaker_name})"
            parts.append(header)
            parts.append(seg.transcript_text)
            parts.append("")

    parts.append("---")
    parts.append("")
    parts.append(f'Write a clear, structured summary of everything covered under "{label.name}".')
    parts.append("- Use markdown formatting.")
    parts.append("- Group related points together.")
    parts.append("- Include specific timestamps in the form (video_filename, HH:MM:SS) when referencing notable moments.")
    if context.has_speakers():
        parts.append("- Always attribute quotes and key points to the specific speaker by name.")
        if context.interviewees():
            parts.append("- Prioritize the interviewee's insights, stories, and reactions — these are the most valuable content.")
    parts.append("- Length should be proportional to content: 1–2 paragraphs for short content, structured sections for long content.")
    parts.append("- Do not pad or repeat yourself.")

    return "\n".join(parts)


def write_summary(summary_text: str, path: Path) -> None:
    """Write summary markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_text, encoding="utf-8")
