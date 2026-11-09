"""Central configuration for tunable runtime values."""

import os

LLM_RETRY_BACKOFF_BASE_SECONDS = 1
LLM_RETRY_BACKOFF_MAX_SECONDS = 16
LLM_RETRY_JITTER_SECONDS = 0.5
LLM_RETRY_INITIAL_SECONDS = 2
LLM_RETRY_MAX_SECONDS = 60
LLM_CIRCUIT_BREAKER_FAILURES = 5
LLM_CIRCUIT_BREAKER_SECONDS = 60
MAX_CALLS_PER_MIN = 10
MIN_COMPLETE_FRACTION = 0.8
LLM_ERROR_SNIPPET_LENGTH = 160
TRANSCRIPT_SOURCE = os.getenv("TRANSCRIPT_SOURCE", "gemini").strip().lower()

# Phase 3 tunables. Score weights are initial, uncalibrated guesses.
SCALE_MAX = 10
PERSONA_TEMPERATURE = 0.3
WAVE1_SIZE = 8
SCORE_WEIGHTS = {"stop_scrolling": 0.35, "share_likelihood": 0.25,
                 "watch_to_end": 0.25, "comment_likelihood": 0.15}
SOUND_MODE = "mixed"
GATE_THRESHOLD = 3.0
MAX_WORKERS = 2
REACH_TIERS = {"Low": 0, "Medium": 5, "High": 7, "Very high": 8.5}


def transcript_source():
    """Return the active transcript source, allowing environment overrides."""
    value = os.getenv("TRANSCRIPT_SOURCE", TRANSCRIPT_SOURCE).strip().lower()
    if value not in {"gemini", "whisper", "both"}:
        raise ValueError("TRANSCRIPT_SOURCE must be 'gemini', 'whisper', or 'both'.")
    return value
