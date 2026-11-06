"""Context YAML loading, validation, and preset generation."""

from pathlib import Path
from pydantic import BaseModel, field_validator
import yaml


class SpeakerDefinition(BaseModel):
    """A speaker in the video."""

    id: str  # short identifier, e.g. "host", "guest1"
    name: str = ""  # display name, e.g. "Sarah Chen"
    role: str = ""  # "interviewer" or "interviewee"
    description: str = ""  # e.g. "CEO of Acme Corp, expert in AI tooling"


class LabelDefinition(BaseModel):
    """A label in the taxonomy."""

    id: str
    name: str
    description: str
    include_in_output: bool = True
    merge_adjacent: bool = True
    color: str = "#6B7280"

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if " " in v:
            raise ValueError(f"label id \"{v}\" contains spaces — use \"{v.lower().replace(' ', '_')}\"")
        if v != v.lower():
            raise ValueError(f"label id \"{v}\" contains uppercase — use \"{v.lower()}\"")
        return v


class Context(BaseModel):
    """Project context loaded from YAML."""

    project_name: str
    description: str = ""
    video_type: str = ""
    audience: str = ""
    goals: list[str] = []
    language: str = "en"
    notes: str = ""
    speakers: list[SpeakerDefinition] = []
    labels: list[LabelDefinition]
    # Fix recurring transcription mistakes in captions, e.g. {"Commander": "Camunda"}.
    # Keys match whole words/phrases, case-insensitively.
    corrections: dict[str, str] = {}

    def apply_corrections(self, text: str) -> str:
        import re

        for wrong in sorted(self.corrections, key=len, reverse=True):
            text = re.sub(rf"(?<!\w){re.escape(wrong)}(?!\w)", self.corrections[wrong], text, flags=re.IGNORECASE)
        return text

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v: list[LabelDefinition]) -> list[LabelDefinition]:
        if not v:
            raise ValueError("labels array must not be empty")
        ids = [label.id for label in v]
        dupes = [i for i in ids if ids.count(i) > 1]
        if dupes:
            raise ValueError(f"duplicate label IDs: {', '.join(set(dupes))}")
        return v

    def get_label(self, label_id: str) -> LabelDefinition | None:
        for label in self.labels:
            if label.id == label_id:
                return label
        return None

    def output_labels(self) -> list[LabelDefinition]:
        """Labels with include_in_output=True."""
        return [l for l in self.labels if l.include_in_output]

    def label_ids(self) -> list[str]:
        return [l.id for l in self.labels]

    def has_speakers(self) -> bool:
        return len(self.speakers) > 0

    def interviewers(self) -> list[SpeakerDefinition]:
        return [s for s in self.speakers if s.role == "interviewer"]

    def interviewees(self) -> list[SpeakerDefinition]:
        return [s for s in self.speakers if s.role == "interviewee"]

    def get_speaker(self, speaker_id: str) -> SpeakerDefinition | None:
        for s in self.speakers:
            if s.id == speaker_id:
                return s
        return None

    def speaker_display_name(self, speaker_id: str) -> str:
        """Get a human-readable name for a speaker ID."""
        s = self.get_speaker(speaker_id)
        if s and s.name:
            return s.name
        return speaker_id


def load_context(path: Path) -> Context:
    """Load and validate a context YAML file.

    Returns a Context object.
    Raises ValueError with a list of issues if validation fails.
    """
    if not path.exists():
        raise ValueError(f"context file \"{path}\" does not exist")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"context file \"{path}\" is not a valid YAML mapping")

    errors: list[str] = []

    # Check required top-level fields
    for field in ("project_name", "labels"):
        if field not in raw:
            errors.append(f"required field \"{field}\" is missing")

    if "labels" in raw:
        if not isinstance(raw["labels"], list):
            errors.append("\"labels\" must be a list")
        else:
            for i, label in enumerate(raw["labels"]):
                if not isinstance(label, dict):
                    errors.append(f"labels[{i}] must be a mapping")
                    continue
                for field in ("id", "name", "description"):
                    if field not in label or not label[field]:
                        errors.append(f"labels[{i}].{field} is required")
                lid = label.get("id", "")
                if isinstance(lid, str):
                    if " " in lid:
                        errors.append(
                            f"labels[{i}].id \"{lid}\" contains spaces — use \"{lid.lower().replace(' ', '_')}\""
                        )
                    elif lid != lid.lower():
                        errors.append(f"labels[{i}].id \"{lid}\" contains uppercase — use \"{lid.lower()}\"")

    if errors:
        msg = f"context file \"{path}\" is invalid:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)

    return Context(**raw)


# ── Preset templates ──────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "conference": {
        "project_name": "Conference Talks",
        "description": "Recordings from a conference or meetup.",
        "video_type": "conference talk",
        "audience": "developers",
        "goals": ["Create highlight clips", "Extract Q&A segments"],
        "language": "en",
        "notes": "",
        "labels": [
            {
                "id": "intro",
                "name": "Intro",
                "description": "Speaker introduction, agenda overview, and housekeeping announcements.",
                "include_in_output": True,
                "merge_adjacent": True,
                "color": "#6B7280",
            },
            {
                "id": "keynote",
                "name": "Keynote",
                "description": "Main content of the talk — core ideas, arguments, explanations, and insights.",
                "include_in_output": True,
                "merge_adjacent": True,
                "color": "#8B5CF6",
            },
            {
                "id": "demo",
                "name": "Demo",
                "description": "Live coding, screen share, product walkthrough, or hands-on demonstration.",
                "include_in_output": True,
                "merge_adjacent": False,
                "color": "#3B82F6",
            },
            {
                "id": "qa",
                "name": "Q&A",
                "description": "Questions from the audience and the speaker's answers.",
                "include_in_output": True,
                "merge_adjacent": True,
                "color": "#10B981",
            },
            {
                "id": "off_topic",
                "name": "Off-topic",
                "description": "Technical issues, long pauses, sponsor reads, small talk, or irrelevant content.",
                "include_in_output": False,
                "merge_adjacent": True,
                "color": "#EF4444",
            },
        ],
    },
    "tutorial": {
        "project_name": "Tutorial Series",
        "description": "Educational tutorial or course recording.",
        "video_type": "tutorial",
        "audience": "learners",
        "goals": ["Build chapter-based navigation", "Extract key tips"],
        "language": "en",
        "notes": "",
        "labels": [
            {"id": "intro", "name": "Intro", "description": "Introduction and overview of what will be covered.", "include_in_output": True, "merge_adjacent": True, "color": "#6B7280"},
            {"id": "concept", "name": "Concept", "description": "Explanation of a concept, theory, or principle.", "include_in_output": True, "merge_adjacent": True, "color": "#8B5CF6"},
            {"id": "demo", "name": "Demo", "description": "Hands-on demonstration, live coding, or walkthrough.", "include_in_output": True, "merge_adjacent": False, "color": "#3B82F6"},
            {"id": "tip", "name": "Tip", "description": "A practical tip, best practice, or shortcut.", "include_in_output": True, "merge_adjacent": False, "color": "#F59E0B"},
            {"id": "summary", "name": "Summary", "description": "Recap or summary of what was covered.", "include_in_output": True, "merge_adjacent": True, "color": "#10B981"},
            {"id": "off_topic", "name": "Off-topic", "description": "Irrelevant content, pauses, or filler.", "include_in_output": False, "merge_adjacent": True, "color": "#EF4444"},
        ],
    },
    "interview": {
        "project_name": "Interview",
        "description": "Interview or conversation recording with an interviewer and one or more interviewees.",
        "video_type": "interview",
        "audience": "general audience",
        "goals": [
            "Extract key insights and quotable moments from interviewees",
            "Create shareable clips of the strongest answers",
            "Build a highlight reel of interviewee reactions and stories",
        ],
        "language": "en",
        "notes": "Pay close attention to WHO is speaking. Insights, stories, and quotable moments from the interviewee(s) are the most valuable content. The interviewer's questions set context but are less important for output reels.",
        "speakers": [
            {"id": "interviewer", "name": "", "role": "interviewer", "description": "The person asking questions and guiding the conversation."},
            {"id": "interviewee", "name": "", "role": "interviewee", "description": "The person being interviewed — their responses are the primary content of interest."},
        ],
        "labels": [
            {"id": "intro", "name": "Intro", "description": "Introduction of participants, greetings, and topic overview. Usually involves both speakers.", "include_in_output": True, "merge_adjacent": True, "color": "#6B7280"},
            {"id": "question", "name": "Question", "description": "The interviewer asking a question or setting up a topic. Includes follow-up probes and clarifying questions.", "include_in_output": True, "merge_adjacent": False, "color": "#3B82F6"},
            {"id": "answer", "name": "Answer", "description": "The interviewee's direct response to a question — their explanation, opinion, or analysis on the topic.", "include_in_output": True, "merge_adjacent": True, "color": "#8B5CF6"},
            {"id": "story", "name": "Story", "description": "The interviewee sharing a personal anecdote, case study, or real-world example from their experience. These are typically vivid, narrative moments.", "include_in_output": True, "merge_adjacent": False, "color": "#F59E0B"},
            {"id": "insight", "name": "Insight", "description": "A notable insight, surprising revelation, strong opinion, or quotable moment from the interviewee. Should be compelling as a standalone clip — something that makes you stop and think.", "include_in_output": True, "merge_adjacent": False, "color": "#10B981"},
            {"id": "reaction", "name": "Reaction", "description": "A notable emotional reaction from the interviewee — surprise, passion, humor, disagreement, excitement. The tone or delivery is as important as the words.", "include_in_output": True, "merge_adjacent": False, "color": "#EC4899"},
            {"id": "off_topic", "name": "Off-topic", "description": "Small talk, technical issues, filler, or tangents not relevant to the interview topic.", "include_in_output": False, "merge_adjacent": True, "color": "#EF4444"},
        ],
    },
    "podcast": {
        "project_name": "Podcast Episode",
        "description": "Podcast episode recording with a host and one or more guests.",
        "video_type": "podcast",
        "audience": "podcast listeners",
        "goals": ["Create highlight clips", "Remove sponsor segments"],
        "language": "en",
        "notes": "Focus on guest insights and stories. Host questions set context but guest responses are the primary value.",
        "speakers": [
            {"id": "host", "name": "", "role": "interviewer", "description": "The podcast host guiding the conversation."},
            {"id": "guest", "name": "", "role": "interviewee", "description": "The guest being interviewed."},
        ],
        "labels": [
            {"id": "intro", "name": "Intro", "description": "Episode introduction and overview.", "include_in_output": True, "merge_adjacent": True, "color": "#6B7280"},
            {"id": "topic", "name": "Topic", "description": "Discussion of a main topic or segment.", "include_in_output": True, "merge_adjacent": True, "color": "#3B82F6"},
            {"id": "story", "name": "Story", "description": "A personal story or anecdote.", "include_in_output": True, "merge_adjacent": False, "color": "#8B5CF6"},
            {"id": "insight", "name": "Insight", "description": "A key takeaway, strong opinion, or memorable quote.", "include_in_output": True, "merge_adjacent": False, "color": "#F59E0B"},
            {"id": "sponsor", "name": "Sponsor", "description": "Sponsor reads, ads, or promotional segments.", "include_in_output": False, "merge_adjacent": True, "color": "#EF4444"},
            {"id": "off_topic", "name": "Off-topic", "description": "Tangents, filler, or irrelevant banter.", "include_in_output": False, "merge_adjacent": True, "color": "#9CA3AF"},
        ],
    },
    "sales-call": {
        "project_name": "Sales Call",
        "description": "Sales call or product demo recording.",
        "video_type": "sales call",
        "audience": "sales team",
        "goals": ["Extract objection handling", "Build training reels"],
        "language": "en",
        "notes": "",
        "labels": [
            {"id": "intro", "name": "Intro", "description": "Initial greetings and agenda setting.", "include_in_output": True, "merge_adjacent": True, "color": "#6B7280"},
            {"id": "discovery", "name": "Discovery", "description": "Needs assessment, pain point exploration, and qualifying questions.", "include_in_output": True, "merge_adjacent": True, "color": "#3B82F6"},
            {"id": "demo", "name": "Demo", "description": "Product demonstration or solution presentation.", "include_in_output": True, "merge_adjacent": True, "color": "#8B5CF6"},
            {"id": "objection", "name": "Objection", "description": "Customer objections and the rep's responses.", "include_in_output": True, "merge_adjacent": False, "color": "#F59E0B"},
            {"id": "close", "name": "Close", "description": "Closing statements, next steps, and commitments.", "include_in_output": True, "merge_adjacent": True, "color": "#10B981"},
            {"id": "off_topic", "name": "Off-topic", "description": "Small talk, technical issues, or irrelevant tangents.", "include_in_output": False, "merge_adjacent": True, "color": "#EF4444"},
        ],
    },
    "event": {
        "project_name": "Event Recording",
        "description": "Event or ceremony recording.",
        "video_type": "event",
        "audience": "attendees",
        "goals": ["Create event highlights", "Extract speeches"],
        "language": "en",
        "notes": "",
        "labels": [
            {"id": "arrival", "name": "Arrival", "description": "Arrivals, setup, and pre-event footage.", "include_in_output": True, "merge_adjacent": True, "color": "#6B7280"},
            {"id": "speech", "name": "Speech", "description": "Speeches, presentations, or formal addresses.", "include_in_output": True, "merge_adjacent": True, "color": "#3B82F6"},
            {"id": "performance", "name": "Performance", "description": "Musical performances, acts, or demonstrations.", "include_in_output": True, "merge_adjacent": False, "color": "#8B5CF6"},
            {"id": "reaction", "name": "Reaction", "description": "Audience reactions, applause, emotional moments.", "include_in_output": True, "merge_adjacent": False, "color": "#F59E0B"},
            {"id": "off_topic", "name": "Off-topic", "description": "Dead time, technical issues, or irrelevant footage.", "include_in_output": False, "merge_adjacent": True, "color": "#EF4444"},
        ],
    },
}


def resolve_context(context_input: str, preset: str | None = None) -> Context:
    """Resolve context from a YAML file, markdown file, or inline text.

    Detection logic:
    - If the input is a path to an existing file ending in .yaml/.yml → load as YAML
    - If the input is a path to an existing file ending in .md/.txt → read as description text
    - Otherwise → treat as inline description text

    When context is provided as text/markdown (not YAML), a --preset is required
    to supply the label taxonomy.

    Args:
        context_input: A file path or inline text string.
        preset: Preset name to supply labels when context is text/markdown.

    Returns:
        A validated Context object.

    Raises:
        ValueError: If validation fails or labels cannot be determined.
    """
    path = Path(context_input)

    # Case 1: YAML file
    if path.is_file() and path.suffix.lower() in (".yaml", ".yml"):
        return load_context(path)

    # Case 2: Markdown or text file
    if path.is_file() and path.suffix.lower() in (".md", ".txt", ".markdown"):
        description = path.read_text(encoding="utf-8").strip()
        return _context_from_text(description, preset, source=str(path))

    # Case 3: Could be a file path that doesn't exist, or inline text
    if path.is_file():
        # Some other file extension — try reading as text
        description = path.read_text(encoding="utf-8").strip()
        return _context_from_text(description, preset, source=str(path))

    # If it looks like a file path but doesn't exist, give a helpful error
    if not path.exists() and (
        path.suffix.lower() in (".yaml", ".yml", ".md", ".txt", ".markdown")
    ):
        raise ValueError(f"context file \"{path}\" does not exist")

    # Case 4: Inline text
    return _context_from_text(context_input.strip(), preset, source="inline text")


def _context_from_text(
    description: str,
    preset: str | None,
    source: str,
) -> Context:
    """Build a Context from a text description + preset labels."""
    if not preset:
        raise ValueError(
            f"Context from {source} has no label taxonomy. "
            f"Use --preset to supply labels (options: {', '.join(PRESETS.keys())})"
        )

    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset \"{preset}\". "
            f"Options: {', '.join(PRESETS.keys())}"
        )

    preset_data = PRESETS[preset]
    labels = [LabelDefinition(**l) for l in preset_data["labels"]]
    speakers = [SpeakerDefinition(**s) for s in preset_data.get("speakers", [])]

    return Context(
        project_name=preset_data["project_name"],
        description=description,
        video_type=preset_data.get("video_type", ""),
        audience=preset_data.get("audience", ""),
        goals=preset_data.get("goals", []),
        language=preset_data.get("language", "en"),
        notes=preset_data.get("notes", ""),
        speakers=speakers,
        labels=labels,
    )


def generate_context_yaml(preset: str | None = None) -> str:
    """Generate a context YAML string, optionally from a preset."""
    if preset and preset in PRESETS:
        data = PRESETS[preset]
    else:
        data = {
            "project_name": "My Project",
            "description": "Describe what these videos are about.",
            "video_type": "conference talk",
            "audience": "your target audience",
            "goals": ["Goal 1", "Goal 2"],
            "language": "en",
            "notes": "Any additional notes for the AI labeler.",
            "labels": [
                {
                    "id": "highlight",
                    "name": "Highlight",
                    "description": "Memorable or quotable moments worth clipping.",
                    "include_in_output": True,
                    "merge_adjacent": False,
                    "color": "#F59E0B",
                },
                {
                    "id": "off_topic",
                    "name": "Off-topic",
                    "description": "Content not worth including in output reels.",
                    "include_in_output": False,
                    "merge_adjacent": True,
                    "color": "#EF4444",
                },
            ],
        }

    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
