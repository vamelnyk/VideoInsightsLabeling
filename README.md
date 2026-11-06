# Video Insights Labeling

A command-line tool that transcribes video files, uses Claude to label transcript segments according to a user-defined taxonomy, generates written summaries per label, and renders one merged video reel per label.

## How It Works

The tool runs a four-stage pipeline:

```
Video files + Context YAML
        │
        ▼
┌─────────────────┐
│  1. Transcribe   │  Whisper extracts word-level transcripts from each video
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Label        │  Claude assigns a label to every second of each video
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Summarize    │  Claude writes a synthesis per label across all videos
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. Render       │  FFmpeg trims and concatenates segments into reels
└────────┬────────┘
         ▼
   Output directory
```

**Input:** One or more video files and a context — provided as a YAML file, a markdown file, or inline text.

**Output:** Per-video transcripts and labels, per-label summaries (markdown), and per-label video reels (mp4).

## Requirements

- **Python** >= 3.10
- **FFmpeg** >= 6.0 (must include `ffprobe`)
- **CUDA** (optional) — Whisper uses GPU automatically when available

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd VideoInsightsLabeling

# Create a virtual environment (Python 3.10+)
python3.12 -m venv .venv
source .venv/bin/activate

# Install the package (registers the `vil` command)
pip install -e .

# Verify
vil --version
ffmpeg -version
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

```bash
# Option A: Quick start with inline text + preset labels
vil run --videos talk1.mp4 talk2.mp4 \
  --context "Conference talks about AI tooling and platform engineering" \
  --preset conference

# Option B: Point to a markdown file with detailed notes
vil run --videos talk1.mp4 talk2.mp4 \
  --context notes.md --preset conference

# Option C: Full control with a YAML context file
vil init-context --preset conference -o context.yaml
# Edit context.yaml to customize labels, then:
vil run --videos talk1.mp4 talk2.mp4 --context context.yaml
```

## CLI Commands

### `run` — Full Pipeline

Transcribe, label, summarize, and render in one command.

```bash
vil run \
  --videos talk1.mp4 talk2.mp4 interview.mp4 \
  --context context.yaml \
  --output ./output
```

| Flag | Description |
|------|-------------|
| `--videos`, `-v` | Video files or directories containing videos (required) |
| `--context`, `-c` | Context: YAML file, markdown file, or inline text (required) |
| `--preset` | Label preset — required when context is text/markdown (see [Presets](#presets)) |
| `--output`, `-o` | Output directory (default: `./output`) |
| `--whisper-model` | Whisper model size (default: `large-v3`) |
| `--skip-transcription` | Reuse existing transcripts from the output directory |
| `--skip-labeling` | Reuse existing labels |
| `--skip-reels` | Skip video rendering (produce transcripts and labels only) |
| `--no-summaries` | Skip summary generation |
| `--labels` | Only process specific label IDs (e.g. `--labels highlight qa`) |
| `--dry-run` | Print what would be done without executing |
| `--force` | Ignore cached results and rerun everything |
| `--export-clips` | Also save every segment as a standalone snippet in `reels/<label>/` |
| `--fast-cuts` | Stream-copy cuts instead of re-encoding (faster, but clip starts snap to the previous keyframe) |

**Examples:**

```bash
# Process all videos in a folder
vil run --videos ./recordings/ --context context.yaml

# Mix individual files and a directory
vil run --videos keynote.mp4 --videos ./day2_talks/ --context context.yaml

# Re-render reels after manually editing labels.json
vil run --videos *.mp4 --context context.yaml \
  --skip-transcription --skip-labeling

# Transcribe and label only, no video output
vil run --videos *.mp4 --context context.yaml \
  --skip-reels --no-summaries

# Only render the highlight reel
vil run --videos *.mp4 --context context.yaml \
  --skip-transcription --skip-labeling --labels highlight
```

### `transcribe` — Transcription Only

```bash
vil transcribe --videos talk1.mp4 talk2.mp4 --output ./output

# Or point to a folder
vil transcribe --videos ./recordings/ --output ./output
```

Does not require an API key or context file.

### `label` — Labeling Only

Requires existing transcripts in the output directory.

```bash
vil label --output ./output --context context.yaml
```

### `render` — Rendering Only

Requires existing labels in the output directory.

```bash
vil render --output ./output --context context.yaml
vil render --output ./output --context context.yaml --labels highlight demo

# Source videos stored elsewhere + one standalone file per snippet
vil render --output ./output --context context.yaml \
  --videos ~/Downloads/interview.mp4 --export-clips
```

By default every clip is re-encoded so cuts land on the exact timestamp. Screen
recordings (Zoom, Teams) often have a keyframe only every 30 seconds or more, so
stream-copy cuts (`--fast-cuts`) can start far too early.

### `import-transcript` — Use an Existing Transcript

Meeting tools export a `.vtt` or `.srt` transcript with speaker names. Import it
to skip Whisper and keep the real names:

```bash
vil import-transcript --video interview.mp4 --transcript interview.vtt --output ./output
vil run --videos interview.mp4 --context context.yaml --skip-transcription
```

`vil run` also picks up a sidecar transcript automatically when it sits next to
the video with the same name (`interview.mp4` + `interview.vtt`).

### `init-context` — Generate Starter Context

```bash
vil init-context --preset conference -o context.yaml
```

Available presets: `conference`, `tutorial`, `interview`, `podcast`, `sales-call`, `event`. See [Presets](#presets) for the labels included in each.

## Providing Context

The `--context` flag accepts three formats, auto-detected by the tool:

### 1. Inline text

Pass a description string directly. Requires `--preset` to supply labels.

```bash
vil run --videos *.mp4 \
  --context "Sales calls from Q1 2025. Focus on objection handling." \
  --preset sales-call
```

### 2. Markdown or text file

Point to a `.md` or `.txt` file containing your project description. Requires `--preset` to supply labels. This is ideal when you have detailed notes about the videos.

```bash
vil run --videos *.mp4 \
  --context project-notes.md \
  --preset conference
```

### 3. YAML file (full control)

A `.yaml` or `.yml` file that includes both the project description and a custom label taxonomy. No `--preset` needed — labels are defined in the file.

```bash
vil run --videos *.mp4 --context context.yaml
```

Use `vil init-context --preset conference` to generate a starter YAML.

### When to use which

| Approach | Best for |
|----------|----------|
| Inline text + preset | Quick one-off runs where built-in labels are good enough |
| Markdown file + preset | Detailed project notes but standard label taxonomy |
| YAML file | Custom labels, fine-tuned descriptions, or repeated use |

### Presets

When using text or markdown context, `--preset` supplies the label taxonomy:

| Preset | Labels | Speakers |
|--------|--------|----------|
| `conference` | intro, keynote, demo, qa, off_topic | — |
| `tutorial` | intro, concept, demo, tip, summary, off_topic | — |
| `interview` | intro, question, answer, story, insight, reaction, off_topic | interviewer + interviewee |
| `podcast` | intro, topic, story, insight, sponsor, off_topic | host + guest |
| `sales-call` | intro, discovery, demo, objection, close, off_topic | — |
| `event` | arrival, speech, performance, reaction, off_topic | — |

---

## Context YAML Reference

The YAML format defines project metadata, an optional speaker roster, and the label taxonomy. Claude reads the label descriptions and speaker roles to decide how to classify each segment.

### Interview example (with speakers)

```yaml
project_name: "AI Research Interview"
description: "Interview with Dr. Sarah Chen about LLM reasoning."
video_type: "interview"
audience: "AI practitioners"
language: "en"

goals:
  - "Extract the most quotable insights from the interviewee"
  - "Build a highlight reel of strongest moments"

notes: "Focus on what the interviewee says, not the host questions."

speakers:
  - id: host
    name: "Alex Rivera"
    role: interviewer
    description: "Host of the AI Research podcast"
  - id: guest
    name: "Dr. Sarah Chen"
    role: interviewee
    description: "Principal researcher at DeepMind, expert in reasoning"

labels:
  - id: insight
    name: "Insight"
    description: >
      A notable insight or surprising claim from the interviewee.
      Should be compelling as a standalone clip.
    include_in_output: true
    merge_adjacent: false
    color: "#F59E0B"

  - id: story
    name: "Story"
    description: >
      A personal anecdote or case study from the interviewee.
    include_in_output: true
    color: "#8B5CF6"

  - id: off_topic
    name: "Off-topic"
    description: "Filler, small talk, or irrelevant content."
    include_in_output: false
    color: "#EF4444"
```

### Conference example (without speakers)

```yaml
project_name: "Dev Conference 2025 — Day 1"
description: "Three talks on AI tooling, platform engineering, and performance."
video_type: "conference talk"
audience: "software engineers"
language: "en"

labels:
  - id: highlight
    name: "Highlight"
    description: >
      A memorable or quotable statement, surprising revelation, or emotional peak.
    include_in_output: true
    merge_adjacent: false
    color: "#F59E0B"

  - id: off_topic
    name: "Off-topic"
    description: "Technical issues, long pauses, or content not useful in a reel."
    include_in_output: false
    color: "#EF4444"
```

### Speaker Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Short identifier (e.g. `host`, `guest1`) |
| `name` | No | Display name (e.g. "Dr. Sarah Chen") |
| `role` | No | `interviewer` or `interviewee` — drives labeling priorities |
| `description` | No | Background info for Claude (e.g. "CEO of Acme, expert in AI") |

When speakers are defined:
- **Diarization** is automatically enabled during transcription (requires `pyannote.audio`)
- **Labeling** prioritizes interviewee content for insight/highlight labels
- **Summaries** attribute quotes and key points to specific speakers

### Label Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Short slug — lowercase, no spaces (e.g. `key_insight`) |
| `name` | Yes | Display name (e.g. "Key Insight") |
| `description` | Yes | What Claude reads to classify segments — be specific |
| `include_in_output` | No | If `false`, no reel or summary is generated (default: `true`) |
| `merge_adjacent` | No | Merge consecutive segments with the same label (default: `true`) |
| `color` | No | Hex color for visual reference (default: `#6B7280`) |

## Output Structure

```
output/
  run.log                        # Full timestamped log

  talk1/
    transcript.json              # Structured transcript (machine-readable)
    transcript.txt               # Human-readable transcript with timestamps
    labels.json                  # Labeled segments with confidence and reasoning
    whisper_raw.json             # Raw Whisper output (for debugging)

  talk2/
    ...

  summaries/
    highlight.md                 # Markdown summary of all "highlight" segments
    demo.md
    qa.md

  reels/
    highlight.mp4                # Merged video of all "highlight" segments
    highlight_manifest.json      # Clip list with sources, timestamps and reasons
    highlight/                   # Standalone snippets (with --export-clips)
      01_talk1_000312-000358.mp4
    demo.mp4
    demo_manifest.json
    qa.mp4
    qa_manifest.json
```

Labels marked `include_in_output: false` (e.g. `off_topic`) produce no reel or summary.

### labels.json Format

```json
{
  "video": "talk1.mp4",
  "duration_s": 2712.4,
  "labeled_at": "2025-04-16T14:32:11+00:00",
  "label_summary": {
    "intro": { "count": 1, "total_duration_s": 312.1 },
    "demo":  { "count": 3, "total_duration_s": 847.6 }
  },
  "segments": [
    {
      "label_id": "intro",
      "start": 0.0,
      "end": 312.1,
      "duration": 312.1,
      "reason": "Speaker introduces herself and outlines three topics.",
      "confidence": 0.97,
      "transcript_text": "Welcome everyone..."
    }
  ]
}
```

## Resumption

The tool automatically skips completed stages when re-run with the same output directory:

- If `transcript.json` exists for a video, transcription is skipped
- If `labels.json` exists for a video, labeling is skipped
- If a reel `.mp4` exists, rendering is skipped

Use `--force` to ignore all cached results, or use `--skip-transcription` / `--skip-labeling` for explicit control.

## Configuration

All configuration is via environment variables. CLI flags override where applicable.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `WHISPER_DEVICE` | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |
| `WHISPER_COMPUTE_TYPE` | `auto` | Compute type: `auto`, `float16`, `int8` |
| `MAX_PARALLEL_TRANSCRIPTIONS` | `2` | Concurrent transcription workers |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model for labeling and summaries |
| `FFMPEG_PATH` | `ffmpeg` | Path to ffmpeg binary |
| `KEEP_AUDIO_FILES` | `false` | Keep extracted audio.wav files after transcription |
| `LOG_LEVEL` | `INFO` | Logging level |

### Model Selection

| Scenario | Whisper model | Claude model |
|----------|--------------|--------------|
| Development / quick tests | `tiny` or `base` | `claude-haiku-4-5` |
| Standard use | `large-v3` | `claude-sonnet-4-6` |
| Maximum quality | `large-v3` | `claude-opus-4-6` |

## How the Pipeline Stages Work

### 1. Transcription

Each video is processed in parallel (up to `MAX_PARALLEL_TRANSCRIPTIONS` workers):

1. **Audio extraction** — FFmpeg converts the video to 16kHz mono WAV
2. **Whisper transcription** — `faster-whisper` produces word-level timestamps with VAD filtering
3. **Speaker diarization** (optional) — When the context defines speakers, `pyannote.audio` identifies who is speaking at each moment and labels each transcript segment with a speaker ID
4. **Post-processing** — Segments shorter than 3 seconds are merged into neighbors (only if same speaker); filler tokens (`[BLANK_AUDIO]`, `[MUSIC]`, `(applause)`) are removed; whitespace is normalized

The Whisper model is loaded once and reused across videos.

To enable speaker diarization, install the optional dependency and set a HuggingFace token:

```bash
pip install "video-insights-labeling[diarization]"
export HF_TOKEN=hf_...
```

Diarization activates automatically when `speakers` are defined in the context. Without `pyannote.audio` installed, Claude will still attempt to infer speakers from conversational patterns.

### 2. Labeling

Each video is labeled sequentially (to manage API rate limits):

1. A **multi-part prompt** is assembled: system instructions, project context, speaker definitions (if any), label taxonomy, and the full transcript with timestamps and speaker labels
2. Claude returns a **JSON array** of labeled time segments, each with a label ID, start/end times, speaker, confidence score, and one-sentence reasoning. When speakers are defined, Claude is instructed to prioritize interviewee content for insight and highlight labels.
3. The response is **validated and repaired**:
   - Unknown label IDs are fuzzy-matched to the closest valid ID
   - Overlapping segments are resolved by keeping the higher-confidence assignment
   - Coverage gaps are filled by extending the nearest segment
4. For videos longer than ~90 minutes (>100k transcript characters), the transcript is split into **45-minute chunks with 5-minute overlap**. Each chunk is labeled independently, then merged with overlap resolution

### 3. Summaries

For each label with `include_in_output: true`, Claude receives all matching segments across all videos and generates a structured markdown summary. Summaries reference specific timestamps and video filenames.

### 4. Rendering

For each output label, FFmpeg creates a merged reel:

1. Each matching segment is **trimmed** from its source video using stream copy (no re-encoding)
2. If all clips share the same codec and resolution, they are **concatenated** via the FFmpeg concat demuxer (fast, lossless)
3. If clips differ in format, they are **normalized** to H.264 1080p 30fps before concatenation
4. A **manifest JSON** is written alongside each reel documenting the source clips

Adjacent segments with the same label are merged before rendering (if `merge_adjacent: true`).

## Error Handling

- **Startup errors** (missing context, missing videos, no ffmpeg, no API key) are fatal — the tool exits immediately
- **Per-video errors** (transcription or labeling failure) are non-fatal — the video is skipped and processing continues
- **Render errors** (FFmpeg failure) are non-fatal — an error log is written and the reel is skipped
- A **run summary** is printed at the end showing success/failure counts and warnings
- The tool exits with code 1 if any errors occurred, 0 otherwise

## Project Structure

```
main.py                       # Fallback entry point (python main.py ...)
pyproject.toml                # Dependencies and metadata
src/
  main.py                     # CLI commands and pipeline orchestration
  config.py                   # Settings from environment variables
  context.py                  # YAML loading, validation, preset templates
  models.py                   # Pydantic data models
  workers.py                  # Parallel transcription orchestration
  pipeline/
    audio.py                  # FFmpeg audio extraction
    transcription.py          # Whisper wrapper and post-processing
    labeling.py               # Claude prompt builder and response validator
    summarizer.py             # Per-label summary generation
    renderer.py               # FFmpeg trim and concat for reels
  utils/
    timestamps.py             # Timestamp formatting helpers
    ffmpeg.py                 # FFmpeg/ffprobe subprocess wrapper
    files.py                  # Path helpers and output directory management
```
