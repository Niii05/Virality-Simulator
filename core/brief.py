"""Build visual and combined content briefs."""

import copy
import json
from pathlib import Path

from core import framework, llm

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "brief.txt"
AUDIO_ONLY_FIELDS = {
    "transcript", "transcript_best_effort", "language", "music_present", "mood",
    "energy", "voice_clarity", "background_noise", "first_3_seconds_sound",
    "sound_dependent", "mean_volume_db", "max_volume_db", "silence_ratio",
    "first_sound_sec", "available", "reason",
}


def build_brief(extract_result, audio=None, creator_context=None):
    """Make one vision call using frames and the approximate local transcript."""
    data = {
        "duration_sec": extract_result.get("duration_sec"),
        "transcript_approximate": extract_result.get("transcript", ""),
        "transcript_language_guess": extract_result.get("language", ""),
    }
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{{NICHE_DETECTION}}", framework.render_niche_detection())
    if creator_context:
        prompt += ("\n\nCreator-provided context (this is what a viewer would know from the caption, "
                   "the account and recognisable people; it is NOT visible in the video):\n"
                   "Treat this context as true for purpose, product, audience and niche. Do not invent "
                   "anything beyond it. Never identify a real person from their face.\n"
                   + creator_context)
    prompt += "\n\nExtracted context (transcript is approximate):\n" + json.dumps(data, ensure_ascii=False)
    result = llm.ask_json(prompt, images=extract_result.get("frames", []))
    if not isinstance(result, dict):
        raise ValueError("Brief response must be a JSON object")
    result["niche_category"] = framework.normalize_niche(result.get("niche_category"))
    niche_reason = result.get("niche_reason")
    result["niche_reason"] = niche_reason.strip() if isinstance(niche_reason, str) and niche_reason.strip() else "The video did not clearly fit a specific niche based on its purpose."
    result["framework_version"] = framework.get_version()
    result["creator_context"] = creator_context
    return result


def build_full_brief(extract_result, audio=None, creator_context=None):
    result = build_brief(extract_result, audio=audio, creator_context=creator_context)
    if audio:
        result.update(copy.deepcopy(audio))
    return result


def build_silent_brief(full_brief):
    result = copy.deepcopy(full_brief)
    for field in AUDIO_ONLY_FIELDS:
        result.pop(field, None)
    return result
