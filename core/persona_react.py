"""Generate one or more personas' reactions to a brief."""

import json
from statistics import median
from pathlib import Path

from core import config, framework, llm
from core.brief import build_silent_brief

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "persona.txt"
SCORE_FIELDS = ("stop_scrolling", "watch_to_end", "share_likelihood", "comment_likelihood")
REACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "stop_scrolling": {"type": "INTEGER", "minimum": 0, "maximum": 10},
        "watch_to_end": {"type": "INTEGER", "minimum": 0, "maximum": 10},
        "share_likelihood": {"type": "INTEGER", "minimum": 0, "maximum": 10},
        "comment_likelihood": {"type": "INTEGER", "minimum": 0, "maximum": 10},
        "reason": {"type": "STRING"},
        "biggest_issue": {"type": "STRING"},
        "criteria_cited": {
            "type": "ARRAY",
            "items": {"type": "STRING", "enum": [
                "hook", "clarity", "retention", "shareability", "sound_and_text", "cta",
            ]},
        },
    },
    "required": [*SCORE_FIELDS, "reason", "biggest_issue", "criteria_cited"],
    "additionalProperties": False,
}
PERSONA_RESPONSE_SCHEMA = REACTION_SCHEMA


def _batch_schema(personas, conditions):
    reaction_properties = {condition: REACTION_SCHEMA for condition in conditions}
    reaction_required = list(conditions)
    item_properties = {"persona_id": {"type": "STRING"}, **reaction_properties}
    return {
        "type": "OBJECT",
        "properties": {
            "reactions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT", "properties": item_properties,
                    "required": ["persona_id", *reaction_required],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reactions"],
        "additionalProperties": False,
    }


def _validate_reaction(result):
    if not isinstance(result, dict):
        raise ValueError("Persona reaction must be a JSON object.")
    clean = dict(result)
    for field in SCORE_FIELDS:
        value = clean.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Persona reaction field {field!r} must be numeric.")
        clean[field] = max(0, min(config.SCALE_MAX, int(round(value))))
    if not isinstance(clean.get("reason"), str) or not isinstance(clean.get("biggest_issue"), str):
        raise ValueError("Persona reaction reason and biggest_issue must be strings.")
    if not isinstance(clean.get("criteria_cited"), list):
        raise ValueError("Persona reaction criteria_cited must be an array.")
    valid_criteria = {item["id"] for item in framework._load()["common_criteria"]}
    clean["criteria_cited"] = list(dict.fromkeys(
        item for item in clean["criteria_cited"]
        if isinstance(item, str) and item in valid_criteria
    ))
    return clean


def _aggregate_reactions(results):
    middle = median(sum(config.SCORE_WEIGHTS[key] * result[key] for key in SCORE_FIELDS)
                    for result in results)
    representative = min(results, key=lambda result: abs(
        sum(config.SCORE_WEIGHTS[key] * result[key] for key in SCORE_FIELDS) - middle
    ))
    aggregated = dict(representative)
    aggregated.update({key: median(result[key] for result in results) for key in SCORE_FIELDS})
    aggregated["criteria_cited"] = list(dict.fromkeys(
        item for result in results for item in result["criteria_cited"]
    ))
    return aggregated


def react_batch(personas, brief, conditions=("sound_on", "sound_off"), runs=1):
    """Return median reactions by persona id and requested sound condition."""
    personas = list(personas)
    conditions = tuple(conditions)
    if not personas:
        return {}
    if any(condition not in {"sound_on", "sound_off"} for condition in conditions):
        raise ValueError("conditions must contain only 'sound_on' and 'sound_off'.")
    if not conditions:
        raise ValueError("At least one sound condition is required.")

    niche = framework.normalize_niche(brief.get("niche_category"))
    full_brief = json.dumps(brief, ensure_ascii=False)
    silent_brief = json.dumps(build_silent_brief(brief), ensure_ascii=False) if "sound_off" in conditions else None
    persona_data = [{"persona_id": persona["id"], "profile": persona} for persona in personas]
    prompt = (PROMPT_PATH.read_text(encoding="utf-8")
              + "\n\nEVALUATION FRAMEWORK (selected niche only):\n"
              + framework.render_rules(niche)
              + "\n\nReact to each persona independently. Return exactly one reaction object for every supplied persona_id. "
                "Return JSON in the supplied schema with a reactions array. Each entry must contain persona_id "
                "and a nested reaction for every requested condition. Keep each nested reason and biggest_issue "
                "specific to that persona. Do not let one persona's preferences or reasoning leak into another.\nPERSONAS:\n"
              + json.dumps(persona_data, ensure_ascii=False)
              + "\nBRIEF WITH SOUND AUDIBLE (sound_on):\n" + full_brief)
    if silent_brief is not None:
        prompt += "\nBRIEF WITH AUDIO REMOVED (sound_off):\n" + silent_brief
    schema = _batch_schema(personas, conditions)
    by_persona = {persona["id"]: {condition: [] for condition in conditions} for persona in personas}
    expected = set(by_persona)
    for _ in range(max(1, int(runs))):
        result = llm.ask_json(
            prompt, temperature=config.PERSONA_TEMPERATURE, response_json_schema=schema,
        )
        entries = result.get("reactions") if isinstance(result, dict) else None
        if not isinstance(entries, list):
            raise ValueError("Persona batch response reactions must be an array.")
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("persona_id") not in expected:
                raise ValueError("Persona batch response contains an unknown persona_id.")
            persona_id = entry["persona_id"]
            if persona_id in seen:
                raise ValueError(f"Persona batch response repeats persona_id {persona_id!r}.")
            seen.add(persona_id)
            for condition in conditions:
                by_persona[persona_id][condition].append(_validate_reaction(entry.get(condition)))
        if seen != expected:
            raise ValueError("Persona batch response must include every supplied persona_id.")
    return {
        persona_id: {condition: _aggregate_reactions(condition_results)
                     for condition, condition_results in conditions_by_persona.items()}
        for persona_id, conditions_by_persona in by_persona.items()
    }


def react(persona, brief, condition):
    """Generate the legacy single-condition reaction shape for callers that need it."""
    if condition not in {"sound_on", "sound_off"}:
        raise ValueError("condition must be 'sound_on' or 'sound_off'.")
    visible_brief = brief
    if condition == "sound_off":
        visible_brief = build_silent_brief(brief)
    instruction = {
        "sound_on": "Assume sound is ON and audible; do not mention muting or missing audio.\n",
        "sound_off": "Assume you cannot hear any audio; judge only visuals and on-screen text.\n",
    }[condition]
    prompt = (PROMPT_PATH.read_text(encoding="utf-8")
              + "\n\nEVALUATION FRAMEWORK (selected niche only):\n"
              + framework.render_rules(framework.normalize_niche(brief.get("niche_category")))
              + "\n\nCONDITION: " + condition + "\nCONDITION INSTRUCTION: " + instruction
              + "PERSONA:\n" + json.dumps(persona, ensure_ascii=False)
              + "\nBRIEF:\n" + json.dumps(visible_brief, ensure_ascii=False))
    results = [llm.ask_json(
        prompt, temperature=config.PERSONA_TEMPERATURE,
        response_json_schema=PERSONA_RESPONSE_SCHEMA,
    ) for _ in range(config.WAVE1_RUNS)]
    cleaned = [_validate_reaction(result) for result in results]
    return _aggregate_reactions(cleaned)
