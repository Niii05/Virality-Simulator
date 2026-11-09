"""Generate a single persona's reaction to a brief."""

import json
from pathlib import Path

from core import config, llm

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "persona.txt"


def react(persona, brief, condition):
    if condition not in {"sound_on", "sound_off"}:
        raise ValueError("condition must be 'sound_on' or 'sound_off'.")
    visible_brief = brief
    if condition == "sound_off":
        from core.brief import build_silent_brief
        visible_brief = build_silent_brief(brief)
    condition_instruction = {
        "sound_on": "Assume sound is ON and audible; do not mention muting or missing audio.\n",
        "sound_off": "Assume you cannot hear any audio; judge only visuals and on-screen text.\n",
    }[condition]
    prompt = (PROMPT_PATH.read_text(encoding="utf-8") + "\n\nCONDITION: " + condition
              + "\nCONDITION INSTRUCTION: " + condition_instruction + "PERSONA:\n"
              + json.dumps(persona, ensure_ascii=False) + "\nBRIEF:\n"
              + json.dumps(visible_brief, ensure_ascii=False))
    result = llm.ask_json(prompt, temperature=config.PERSONA_TEMPERATURE)
    for field in ("stop_scrolling", "watch_to_end", "share_likelihood", "comment_likelihood"):
        value = result.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Persona reaction field {field!r} must be numeric.")
        result[field] = max(0, min(config.SCALE_MAX, int(round(value))))
    if not isinstance(result.get("reason"), str) or not isinstance(result.get("biggest_issue"), str):
        raise ValueError("Persona reaction reason and biggest_issue must be strings.")
    if not isinstance(result.get("stop_scrolling"), int) or not isinstance(result.get("watch_to_end"), int):
        raise ValueError("Persona reaction scores must be integers.")
    return result
