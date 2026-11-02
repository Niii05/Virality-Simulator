"""Central configuration for tunable runtime values."""

import os

LLM_RETRY_BACKOFF_BASE_SECONDS = 1
LLM_RETRY_BACKOFF_MAX_SECONDS = 16
LLM_RETRY_JITTER_SECONDS = 0.5
LLM_ERROR_SNIPPET_LENGTH = 160
TRANSCRIPT_SOURCE = os.getenv("TRANSCRIPT_SOURCE", "gemini").strip().lower()


def transcript_source():
    """Return the active transcript source, allowing environment overrides."""
    value = os.getenv("TRANSCRIPT_SOURCE", TRANSCRIPT_SOURCE).strip().lower()
    if value not in {"gemini", "whisper", "both"}:
        raise ValueError("TRANSCRIPT_SOURCE must be 'gemini', 'whisper', or 'both'.")
    return value
