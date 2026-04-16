"""Timestamp formatting utilities."""


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.

    >>> format_timestamp(0)
    '00:00:00'
    >>> format_timestamp(3661.5)
    '01:01:01'
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration.

    >>> format_duration(45.2)
    '00:00:45'
    >>> format_duration(3661)
    '01:01:01'
    """
    return format_timestamp(seconds)


def format_duration_human(seconds: float) -> str:
    """Format seconds as compact human string like '4m 32s'.

    >>> format_duration_human(272)
    '4m 32s'
    >>> format_duration_human(3661)
    '1h 1m 1s'
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)
