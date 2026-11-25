"""Central configuration for tunable runtime values."""

import os

LLM_RETRY_BACKOFF_BASE_SECONDS = 1
LLM_RETRY_BACKOFF_MAX_SECONDS = 16
LLM_RETRY_JITTER_SECONDS = 0.5
LLM_RETRY_INITIAL_SECONDS = 2
LLM_RETRY_MAX_SECONDS = 20
VISION_BRIEF_TIMEOUT_SEC = 120
RETRY_503_MAX = 3
RETRY_503_BASE_SECONDS = 5
RETRY_503_MAX_WAIT_SECONDS = 20
LLM_CIRCUIT_BREAKER_FAILURES = 5
LLM_CIRCUIT_BREAKER_SECONDS = 60
MAX_CALLS_PER_MIN = 10
MIN_COMPLETE_FRACTION = 0.8
LLM_ERROR_SNIPPET_LENGTH = 160
FALLBACK_MODEL = ""
TRANSCRIPT_SOURCE = os.getenv("TRANSCRIPT_SOURCE", "gemini").strip().lower()

# Phase 3 tunables. Score weights are initial, uncalibrated guesses.
SCALE_MAX = 10
WAVE1_RUNS = 3
WAVE2_RUNS = 1
BATCH_SIZE = 3
PERSONA_TEMPERATURE = 0.1
WAVE1_SIZE = 8
WAVE2_SIZE = 16
AUDIENCE_PERSONA_TEMPERATURE = 0.3
WAVE1_SLOT_ARCHETYPES = [
    "gen_z_scroller", "gen_z_scroller",
    "casual_viewer", "casual_viewer",
    "niche_fan", "niche_fan",
    "skeptic", "creator_peer",
]
SCORE_WEIGHTS = {"stop_scrolling": 0.35, "share_likelihood": 0.25,
                 "watch_to_end": 0.25, "comment_likelihood": 0.15}
SOUND_MODE = "mixed"
GATE_THRESHOLD = 3.0
MAX_WORKERS = 1
REACH_TIERS = {"Low": 0, "Medium": 5, "High": 7, "Very high": 8.5}
HIGH_CONFIDENCE_MAX_STDDEV = 1.0
MEDIUM_CONFIDENCE_MAX_STDDEV = 2.0
SCORE_INTERVAL_MIN_PERSONAS = 3
SCORE_INTERVAL_CONFIDENCE = 0.95
SCORE_INTERVAL_BOOTSTRAP_SAMPLES = 10000


def transcript_source():
    """Return the active transcript source, allowing environment overrides."""
    value = os.getenv("TRANSCRIPT_SOURCE", TRANSCRIPT_SOURCE).strip().lower()
    if value not in {"gemini", "whisper", "both"}:
        raise ValueError("TRANSCRIPT_SOURCE must be 'gemini', 'whisper', or 'both'.")
    return value
