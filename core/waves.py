"""Select and simulate audience waves."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core import config, llm
from core.personas import load_personas
from core.persona_react import react

logger = logging.getLogger(__name__)
AUDIENCE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "audience_match.txt"
DIMENSIONS = ("stop_scrolling", "watch_to_end", "share_likelihood", "comment_likelihood")


class _Wave1People(list):
    def __init__(self, values, selection, raw_ids):
        super().__init__(values)
        self.selection = selection
        self.raw_ids = raw_ids


def _balanced(personas, count):
    groups = {}
    for persona in personas:
        groups.setdefault(persona["archetype"], []).append(persona)
    selected = []
    while len(selected) < min(count, len(personas)):
        progressed = False
        for key in sorted(groups):
            if groups[key]:
                selected.append(groups[key].pop(0))
                progressed = True
                if len(selected) >= min(count, len(personas)):
                    break
        if not progressed:
            break
    return selected


def choose_wave1(brief, personas=None):
    personas = personas or load_personas()
    compact = [{"id": p["id"], "archetype": p["archetype"], "interests": p["interests"]} for p in personas]
    template = AUDIENCE_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (template.replace("{{WAVE1_SIZE}}", str(config.WAVE1_SIZE))
              .replace("{{BRIEF}}", json.dumps({k: brief.get(k) for k in ("target_audience_guess", "topic_niche")}, ensure_ascii=False))
              .replace("{{PERSONAS}}", json.dumps(compact, ensure_ascii=False)))
    valid = {p["id"]: p for p in personas}
    raw_ids = []
    try:
        with llm.call_stage("audience_match"):
            result = llm.ask_json(prompt)
        ids = result.get("ids") if isinstance(result, dict) else None
        raw_ids = ids if isinstance(ids, list) else []
        selected_ids = list(dict.fromkeys(i for i in raw_ids if isinstance(i, str) and i in valid))[:config.WAVE1_SIZE]
        valid_count = len(selected_ids)
        logger.info("audience_match ids returned=%d valid=%d filled=%d", len(raw_ids), valid_count,
                    max(0, min(config.WAVE1_SIZE, len(personas)) - valid_count))
        if not selected_ids:
            return _Wave1People(_balanced(personas, config.WAVE1_SIZE), "fallback", raw_ids), "fallback"
        if len(selected_ids) < min(config.WAVE1_SIZE, len(personas)):
            already = set(selected_ids)
            fill = [p["id"] for p in _balanced(personas, config.WAVE1_SIZE) if p["id"] not in already]
            selected_ids.extend(fill[:config.WAVE1_SIZE - len(selected_ids)])
            return _Wave1People([valid[i] for i in selected_ids], "llm_filled", raw_ids), None
        return _Wave1People([valid[i] for i in selected_ids], "llm", raw_ids), None
    except Exception as error:
        logger.info("audience_match ids returned=%d valid=0 filled=%d", len(raw_ids), min(config.WAVE1_SIZE, len(personas)))
        return _Wave1People(_balanced(personas, config.WAVE1_SIZE), "fallback", raw_ids), f"audience_match fallback ({type(error).__name__}: {error})"


def _one_persona(persona, full_brief, audio_available, reaction_cache=None):
    def reaction(condition):
        cached = reaction_cache.get(persona["id"], condition) if reaction_cache else None
        if cached is not None:
            return cached
        with llm.call_stage(f"persona:{persona['id']}:{condition}"):
            value = react(persona, full_brief, condition)
        if reaction_cache:
            reaction_cache.set(persona["id"], condition, value)
        return value

    on = reaction("sound_on")
    if audio_available:
        off = reaction("sound_off")
    else:
        off = on
    def score(reaction):
        return sum(config.SCORE_WEIGHTS[field] * reaction[field] for field in config.SCORE_WEIGHTS)
    mode = config.SOUND_MODE
    if mode == "on_only":
        value = score(on)
    elif mode == "off_only":
        value = score(off)
    elif mode == "mixed":
        p = persona["sound_on_probability"] if audio_available else 1.0
        value = p * score(on) + (1 - p) * score(off)
    else:
        raise ValueError(f"Unknown SOUND_MODE: {mode}")
    return {"persona_id": persona["id"], "score": round(value, 4), "sound_on_probability": persona["sound_on_probability"],
            "sound_on": on, "sound_off": off if audio_available else None}


def run_wave(personas, full_brief, audio_available=True, reaction_cache=None):
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=max(1, min(config.MAX_WORKERS, 5))) as pool:
        futures = {pool.submit(_one_persona, p, full_brief, audio_available, reaction_cache): p for p in personas}
        for future in as_completed(futures):
            p = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append({"persona_id": p["id"], "error": f"{type(error).__name__}: {error}"})
    return sorted(results, key=lambda x: x["persona_id"]), errors


def dimension_means(results):
    means = {}
    for condition in ("sound_on", "sound_off"):
        reactions = [item.get(condition) for item in results if isinstance(item.get(condition), dict)]
        means[condition] = {
            dimension: (sum(reaction[dimension] for reaction in reactions) / len(reactions)
                        if reactions else None)
            for dimension in DIMENSIONS
        }
    return means


def simulate(brief, audio_available=True, personas=None, reaction_cache=None):
    all_personas = personas or load_personas()
    selection = choose_wave1(brief, all_personas)
    wave1_people, selection_error = selection[:2]
    selection_type = getattr(wave1_people, "selection", "fallback" if selection_error else "llm")
    raw_ids = getattr(wave1_people, "raw_ids", [])
    def execute_wave(selected_people):
        if reaction_cache is None:
            return run_wave(selected_people, brief, audio_available)
        return run_wave(selected_people, brief, audio_available, reaction_cache)

    wave1, errors = execute_wave(wave1_people)
    mean = sum(x["score"] for x in wave1) / len(wave1) if wave1 else 0.0
    passed = bool(wave1) and mean >= config.GATE_THRESHOLD
    wave2, wave2_errors = ([], [])
    means_by_wave = {"wave1": dimension_means(wave1)}
    if passed:
        selected = {p["id"] for p in wave1_people}
        wave2_people = [p for p in all_personas if p["id"] not in selected]
        wave2, wave2_errors = execute_wave(wave2_people)
        means_by_wave["wave2"] = dimension_means(wave2)
    if selection_error:
        errors.append({"stage": "audience_match", "error": selection_error})
    scored_ids = {item["persona_id"] for item in wave1 + wave2}
    failed_ids = sorted({item["persona_id"] for item in errors + wave2_errors if "persona_id" in item})
    intended = len(all_personas)
    scored = len(scored_ids)
    return {"wave1": wave1, "wave2": wave2, "gate_passed": passed,
            "wave1_selection": selection_type, "wave1_raw_ids": raw_ids,
            "dimension_means": means_by_wave,
            "personas_intended": intended, "personas_scored": scored,
            "personas_failed": failed_ids, "complete": scored == intended,
            "errors": errors + wave2_errors}
