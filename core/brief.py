"""Build visual and combined content briefs."""

import copy
import json
from pathlib import Path

from core import llm

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "brief.txt"
AUDIO_ONLY_FIELDS = {
    "transcript", "transcript_best_effort", "language", "music_present", "mood",
    "energy", "voice_clarity", "background_noise", "first_3_seconds_sound",
    "sound_dependent", "mean_volume_db", "max_volume_db", "silence_ratio",
    "first_sound_sec", "available", "reason",
}


def build_brief(extract_result, audio=None):
    """Make one vision call using frames and the approximate local transcript."""
    data = {
        "duration_sec": extract_result.get("duration_sec"),
        "transcript_approximate": extract_result.get("transcript", ""),
        "transcript_language_guess": extract_result.get("language", ""),
    }
    prompt = PROMPT_PATH.read_text(encoding="utf-8") + "\n\nExtracted context (transcript is approximate):\n" + json.dumps(data, ensure_ascii=False)
    return llm.ask_json(prompt, images=extract_result.get("frames", []))


def build_full_brief(extract_result, audio=None):
    result = build_brief(extract_result, audio=audio)
    if audio:
        result.update(copy.deepcopy(audio))
    return result


def build_silent_brief(full_brief):
    result = copy.deepcopy(full_brief)
    for field in AUDIO_ONLY_FIELDS:
        result.pop(field, None)
    return result
