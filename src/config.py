"""Configuration loaded from environment variables."""

import os
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings, loaded from environment variables."""

    # Required
    anthropic_api_key: str = ""

    # Whisper
    whisper_model: str = "large-v3"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"

    # Parallelism
    max_parallel_transcriptions: int = 2

    # Claude
    claude_model: str = "claude-sonnet-4-6"

    # FFmpeg
    ffmpeg_path: str = "ffmpeg"
    ffmpeg_threads: int = 0  # 0 = auto

    # Behavior
    keep_audio_files: bool = False
    log_level: str = "INFO"


def load_settings(**overrides: object) -> Settings:
    """Load settings from environment variables, with optional overrides."""
    env_map = {
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "whisper_model": "WHISPER_MODEL",
        "whisper_device": "WHISPER_DEVICE",
        "whisper_compute_type": "WHISPER_COMPUTE_TYPE",
        "max_parallel_transcriptions": "MAX_PARALLEL_TRANSCRIPTIONS",
        "claude_model": "CLAUDE_MODEL",
        "ffmpeg_path": "FFMPEG_PATH",
        "ffmpeg_threads": "FFMPEG_THREADS",
        "keep_audio_files": "KEEP_AUDIO_FILES",
        "log_level": "LOG_LEVEL",
    }

    values: dict[str, object] = {}
    for field, env_var in env_map.items():
        env_val = os.environ.get(env_var)
        if env_val is not None:
            # Type coercion for non-string fields
            field_info = Settings.model_fields[field]
            if field_info.annotation is int:
                values[field] = int(env_val)
            elif field_info.annotation is bool:
                values[field] = env_val.lower() in ("true", "1", "yes")
            else:
                values[field] = env_val

    # CLI overrides take precedence
    for k, v in overrides.items():
        if v is not None:
            values[k] = v

    return Settings(**values)
