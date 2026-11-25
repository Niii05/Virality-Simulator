"""Select and simulate audience waves."""
import copy
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core import config, llm
from core.personas import load_personas
from core.persona_react import react, react_batch

logger = logging.getLogger(__name__)
AUDIENCE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "audience_match.txt"
AUDIENCE_PERSONAS_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "audience_personas.txt"
DIMENSIONS = ("stop_scrolling", "watch_to_end", "share_likelihood", "comment_likelihood")


def _batch_attempt_events(batch_ids, first_event):
    batch_ids = set(batch_ids)
    events = []
    for event in llm.CALL_EVENTS[first_event:]:
        stage = event.get("stage", "")
        if not stage.startswith("persona:") or not stage.endswith(":batch"):
            continue
        stage_ids = set(stage[len("persona:"):-len(":batch")].split(","))
        if stage_ids & batch_ids:
            events.append(event)
    return events


def _log_persona_batch_start(batch, wave_number, batch_id):
    for persona in batch:
        logger.info(
            "Persona start persona_id=%s archetype=%s wave=%s batch_id=%s model=pending attempt_count=0 final_status=STARTED",
            persona["id"], persona.get("archetype", "unknown"), wave_number, batch_id,
        )


def _log_persona_batch_end(batch, wave_number, batch_id, events, status, error=None):
    model_names = list(dict.fromkeys(event.get("model", "unknown") for event in events))
    model_attempts = ",".join(
        f"{event.get('model', 'unknown')}:{event.get('attempt', 0)}:{event.get('status', 'UNKNOWN')}"
        for event in events
    ) or "none"
    reason = (f" error={type(error).__name__}: {llm._redact_error(error)}"
              if error is not None else "")
    for persona in batch:
        logger.info(
            "Persona completion persona_id=%s archetype=%s wave=%s batch_id=%s model=%s attempt_count=%s model_attempts=%s final_status=%s%s",
            persona["id"], persona.get("archetype", "unknown"), wave_number,
            batch_id, ",".join(model_names) or "cache", len(events),
            model_attempts, status, reason,
        )


def _execute_logged_batch(batch, full_brief, audio_available, reaction_cache,
                          persona_runs, wave_number, batch_id):
    _log_persona_batch_start(batch, wave_number, batch_id)
    first_event = len(llm.CALL_EVENTS)
    try:
        results = _persona_batch(batch, full_brief, audio_available,
                                 reaction_cache, persona_runs)
    except Exception as error:
        events = _batch_attempt_events((p["id"] for p in batch), first_event)
        _log_persona_batch_end(batch, wave_number, batch_id, events, "FAILED", error)
        raise
    events = _batch_attempt_events((p["id"] for p in batch), first_event)
    _log_persona_batch_end(batch, wave_number, batch_id, events, "COMPLETED")
    return results


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


def select_wave2_personas(personas=None, exclude_ids=()):
    """Choose the same sorted, archetype-balanced fixed audience on every run."""
    personas = personas or load_personas()
    excluded = set(exclude_ids)
    groups = {}
    for persona in sorted(personas, key=lambda item: item["id"]):
        if persona["id"] not in excluded:
            groups.setdefault(persona["archetype"], []).append(persona)
    archetypes = sorted(groups)
    selected = []
    target_size = min(config.WAVE2_SIZE, sum(map(len, groups.values())))
    while len(selected) < target_size:
        added = False
        for archetype in archetypes:
            if groups[archetype]:
                selected.append(groups[archetype].pop(0))
                added = True
                if len(selected) == target_size:
                    break
        if not added:
            break
    return selected


def _age_bounds(age_range):
    if not isinstance(age_range, str):
        return 18, 65
    numbers = [int(value) for value in re.findall(r"\d+", age_range)]
    if not numbers:
        return 18, 65
    if "under" in age_range.lower():
        return 13, max(13, numbers[0] - 1)
    if "+" in age_range or "over" in age_range.lower() or "older" in age_range.lower():
        return numbers[0], 100
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


ARCHETYPE_ID_PREFIX = {
    "gen_z_scroller": "genz",
    "casual_viewer": "casual",
    "niche_fan": "niche_fan",
    "skeptic": "skeptic",
    "creator_peer": "creator",
    "brand_marketer": "marketer",
}


def _dynamic_prompt(brief):
    slots = list(config.WAVE1_SLOT_ARCHETYPES)
    if len(slots) != config.WAVE1_SIZE:
        raise ValueError("WAVE1_SLOT_ARCHETYPES must contain exactly WAVE1_SIZE entries")
    audience = brief.get("target_audience_guess")
    audience = audience if isinstance(audience, dict) else {}
    region = brief.get("region") or audience.get("region") or "global"
    input_data = {
        "target_audience_guess": audience,
        "topic_niche": brief.get("topic_niche"),
        "niche_category": brief.get("niche_category"),
        "creator_context": brief.get("creator_context"),
        "region": region,
    }
    prompt = (AUDIENCE_PERSONAS_PROMPT_PATH.read_text(encoding="utf-8")
              .replace("{{AUDIENCE_INPUT}}", json.dumps(input_data, ensure_ascii=False))
              .replace("{{SLOTS}}", json.dumps([
                  {"slot": index, "archetype": archetype}
                  for index, archetype in enumerate(slots, start=1)
              ], ensure_ascii=False)))
    with llm.call_stage("audience_personas"):
        response = llm.ask_json(prompt, temperature=config.AUDIENCE_PERSONA_TEMPERATURE)
    return response, slots, audience, region


def _entry_values(entry, slot, archetype, min_age, max_age):
    if not isinstance(entry, dict) or entry.get("slot") != slot or entry.get("archetype") != archetype:
        return None
    age = entry.get("age")
    context = entry.get("context")
    cares = entry.get("cares_about")
    peeves = entry.get("pet_peeves")
    if not isinstance(age, int) or isinstance(age, bool) or not min_age <= age <= max_age:
        return None
    if not isinstance(context, str) or not context.strip():
        return None
    if not isinstance(cares, list) or len(cares) != 3 or not all(isinstance(item, str) and item.strip() for item in cares):
        return None
    if not isinstance(peeves, list) or len(peeves) != 2 or not all(isinstance(item, str) and item.strip() for item in peeves):
        return None
    return {"context": context.strip(), "cares_about": list(cares), "pet_peeves": list(peeves)}


def _dynamic_template(templates, archetype, slot_values, count):
    group = templates[archetype]
    persona = copy.deepcopy(group[count % len(group)])
    prefix = ARCHETYPE_ID_PREFIX.get(archetype, archetype.split("_", 1)[0])
    persona.update({
        "id": f"{prefix}_dyn_{count + 1}",
        "interests": list(slot_values["cares_about"]),
        "cares_about": list(slot_values["cares_about"]),
        "pet_peeves": list(slot_values["pet_peeves"]),
        "context": slot_values["context"],
    })
    return persona


def _fixed_selection_for_slots(brief, personas):
    selected, error = _choose_wave1_by_match(brief, personas)
    return list(selected), error


def _generated_wave1_people(brief, personas, response, slots, audience):
    templates = {}
    for persona in sorted(personas, key=lambda item: item["id"]):
        templates.setdefault(persona["archetype"], []).append(persona)
    if any(archetype not in templates for archetype in slots):
        raise ValueError("WAVE1_SLOT_ARCHETYPES contains an archetype without a persona template")

    entries = response.get("personas") if isinstance(response, dict) else None
    min_age, max_age = _age_bounds(audience.get("age_range"))
    valid_by_slot = {}
    sub_interests = set()
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            slot = entry.get("slot")
            if not isinstance(slot, int) or isinstance(slot, bool) or not 1 <= slot <= len(slots) or slot in valid_by_slot:
                continue
            values = _entry_values(entry, slot, slots[slot - 1], min_age, max_age)
            if values is None:
                continue
            primary_interest = values["cares_about"][0].strip().casefold()
            if primary_interest in sub_interests:
                continue
            sub_interests.add(primary_interest)
            valid_by_slot[slot] = values

    missing_slots = [slot for slot in range(1, len(slots) + 1) if slot not in valid_by_slot]
    fixed_people = []
    fixed_error = None
    if missing_slots:
        fixed_people, fixed_error = _fixed_selection_for_slots(brief, personas)
    selected_fixed_ids = set()
    fixed_by_archetype = {}
    for persona in fixed_people:
        fixed_by_archetype.setdefault(persona["archetype"], []).append(persona)
    pool_by_archetype = {}
    for persona in sorted(personas, key=lambda item: item["id"]):
        pool_by_archetype.setdefault(persona["archetype"], []).append(persona)

    templates_count = {}
    generated = []
    for slot, archetype in enumerate(slots, start=1):
        values = valid_by_slot.get(slot)
        if values is not None:
            count = templates_count.get(archetype, 0)
            templates_count[archetype] = count + 1
            generated.append(_dynamic_template(templates, archetype, values, count))
            continue
        candidates = fixed_by_archetype.get(archetype, []) + pool_by_archetype.get(archetype, [])
        fixed = next((item for item in candidates if item["id"] not in selected_fixed_ids), None)
        if fixed is None:
            raise ValueError(f"No unused fixed persona available for slot {slot} ({archetype})")
        selected_fixed_ids.add(fixed["id"])
        generated.append(copy.deepcopy(fixed))

    selection = "dynamic_filled" if missing_slots else "dynamic"
    raw_ids = [persona["id"] for persona in generated]
    return _Wave1People(generated, selection, raw_ids), fixed_error


def choose_wave1(brief, personas=None):
    """Generate audience-first Wave 1 personas; retain audience matching as fallback."""
    personas = personas or load_personas()
    try:
        response, slots, audience, _ = _dynamic_prompt(brief)
    except Exception as error:
        logger.info("audience_personas generation failed (%s: %s); using audience_match fallback",
                    type(error).__name__, error)
        selected, match_error = _choose_wave1_by_match(brief, personas)
        selected = _Wave1People(selected, "fixed_fallback", getattr(selected, "raw_ids", []))
        fallback_error = f"audience_personas failed ({type(error).__name__}: {error}); used audience_match"
        if match_error:
            fallback_error += f"; audience_match failed ({match_error})"
        return selected, fallback_error
    try:
        selected, fill_error = _generated_wave1_people(brief, personas, response, slots, audience)
    except Exception as error:
        logger.info("audience_personas response could not be built (%s: %s); using audience_match fallback",
                    type(error).__name__, error)
        fixed, match_error = _choose_wave1_by_match(brief, personas)
        selected = _Wave1People(fixed, "fixed_fallback", getattr(fixed, "raw_ids", []))
        fallback_error = f"audience_personas response invalid ({type(error).__name__}: {error}); used audience_match"
        if match_error:
            fallback_error += f"; audience_match failed ({match_error})"
        return selected, fallback_error
    if fill_error:
        return selected, f"audience_match failed while filling invalid slots ({fill_error})"
    return selected, None


def _choose_wave1_by_match(brief, personas):
    personas = personas or load_personas()
    compact = [{"id": p["id"], "archetype": p["archetype"], "interests": p["interests"]} for p in personas]
    template = AUDIENCE_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (template.replace("{{WAVE1_SIZE}}", str(config.WAVE1_SIZE))
              .replace("{{BRIEF}}", json.dumps({k: brief.get(k) for k in ("target_audience_guess", "topic_niche", "creator_context")}, ensure_ascii=False))
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


def _score_persona(persona, on, off, audio_available):
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


def _persona_batch(personas, full_brief, audio_available, reaction_cache, persona_runs):
    cached = {}
    missing = []
    conditions = ("sound_on", "sound_off") if audio_available else ("sound_on",)
    for persona in personas:
        values = {condition: reaction_cache.get(persona, condition) if reaction_cache else None
                  for condition in conditions}
        if all(value is not None for value in values.values()):
            cached[persona["id"]] = values
        else:
            missing.append(persona)

    generated = {}
    if missing:
        ids = ",".join(persona["id"] for persona in missing)
        with llm.call_stage(f"persona:{ids}:batch"):
            generated = react_batch(missing, full_brief, conditions=conditions, runs=persona_runs)
        if reaction_cache:
            for persona in missing:
                for condition, value in generated[persona["id"]].items():
                    reaction_cache.set(persona, condition, value)

    results = []
    for persona in personas:
        reactions = cached.get(persona["id"]) or generated[persona["id"]]
        on = reactions["sound_on"]
        off = reactions.get("sound_off") if audio_available else on
        results.append(_score_persona(persona, on, off, audio_available))
    return results


def _one_persona(persona, full_brief, audio_available, reaction_cache=None):
    """Score one persona with separate condition cache entries for compatibility."""
    conditions = ("sound_on", "sound_off") if audio_available else ("sound_on",)
    reactions = {}
    for condition in conditions:
        value = reaction_cache.get(persona, condition) if reaction_cache else None
        if value is None:
            with llm.call_stage(f"persona:{persona['id']}:{condition}"):
                value = react(persona, full_brief, condition)
            if reaction_cache:
                reaction_cache.set(persona, condition, value)
        reactions[condition] = value
    on = reactions["sound_on"]
    off = reactions.get("sound_off") if audio_available else on
    return _score_persona(persona, on, off, audio_available)


def run_wave(personas, full_brief, audio_available=True, reaction_cache=None, persona_runs=None,
             wave_number=1):
    results, errors = [], []
    persona_runs = config.WAVE1_RUNS if persona_runs is None else persona_runs
    size = max(1, int(config.BATCH_SIZE))
    batches = [(index // size + 1, personas[index:index + size])
               for index in range(0, len(personas), size)]
    with ThreadPoolExecutor(max_workers=max(1, min(config.MAX_WORKERS, 5))) as pool:
        futures = {}
        for batch_number, batch in batches:
            batch_id = f"wave{wave_number}-batch{batch_number:02d}"
            future = pool.submit(_execute_logged_batch, batch, full_brief,
                                 audio_available, reaction_cache, persona_runs,
                                 wave_number, batch_id)
            futures[future] = (batch_number, batch, batch_id)
        for future in as_completed(futures):
            _, batch, batch_id = futures[future]
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as error:
                safe_error = llm._redact_error(error)
                for persona in batch:
                    errors.append({"stage": "persona", "wave": wave_number,
                                   "batch_id": batch_id, "persona_id": persona["id"],
                                   "error": f"{type(error).__name__}: {safe_error}"})
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


def simulate(brief, audio_available=True, personas=None, reaction_cache=None, wave1_persona_cache=None):
    all_personas = personas or load_personas()
    cached_wave1 = wave1_persona_cache.get(brief) if wave1_persona_cache else None
    if cached_wave1:
        wave1_people = _Wave1People(cached_wave1["wave1_personas"], cached_wave1["wave1_selection"],
                                    cached_wave1.get("wave1_raw_ids", []))
        selection_error = None
    else:
        selection = choose_wave1(brief, all_personas)
        wave1_people, selection_error = selection[:2]
        if (wave1_persona_cache and getattr(wave1_people, "selection", None) in {"dynamic", "dynamic_filled"}):
            wave1_persona_cache.set(brief, list(wave1_people), wave1_people.selection,
                                    getattr(wave1_people, "raw_ids", []))
    selection_type = getattr(wave1_people, "selection", "fallback" if selection_error else "llm")
    raw_ids = getattr(wave1_people, "raw_ids", [])
    def execute_wave(selected_people, persona_runs, wave_number):
        if reaction_cache is None:
            return run_wave(selected_people, brief, audio_available,
                            persona_runs=persona_runs, wave_number=wave_number)
        return run_wave(selected_people, brief, audio_available, reaction_cache,
                        persona_runs=persona_runs, wave_number=wave_number)

    logger.info("Pipeline stage event stage=core_audience status=started")
    wave1, errors = execute_wave(wave1_people, config.WAVE1_RUNS, 1)
    wave1_complete = bool(wave1_people) and len(wave1) == len(wave1_people)
    logger.info("Pipeline stage event stage=core_audience status=%s",
                "completed" if wave1_complete else "incomplete")
    core_resonance = sum(x["score"] for x in wave1) / len(wave1) if wave1 else 0.0
    logger.info("Pipeline stage event stage=gate status=started")
    passed = (len(wave1) == len(wave1_people) and bool(wave1)
              and core_resonance >= config.GATE_THRESHOLD)
    gate_outcome = ("passed" if passed else
                    "wave1_incomplete" if not wave1_complete else "threshold_not_met")
    logger.info("Pipeline stage event stage=gate status=completed outcome=%s", gate_outcome)
    wave2, wave2_errors = ([], [])
    wave2_people = []
    means_by_wave = {"wave1": dimension_means(wave1)}
    if passed:
        selected = {p["id"] for p in wave1_people}
        wave2_people = select_wave2_personas(all_personas, exclude_ids=selected)
        logger.info("Pipeline stage event stage=wider_audience status=started")
        wave2, wave2_errors = execute_wave(wave2_people, config.WAVE2_RUNS, 2)
        wave2_complete = (len(wave2) == len(wave2_people)
                          and not any(error.get("wave") == 2 for error in wave2_errors))
        logger.info("Pipeline stage event stage=wider_audience status=%s",
                    "completed" if wave2_complete else "incomplete")
        means_by_wave["wave2"] = dimension_means(wave2)
    else:
        logger.info("Pipeline stage event stage=wider_audience status=skipped outcome=%s",
                    gate_outcome)
    if selection_error:
        errors.append({"stage": "audience_personas", "error": selection_error})
    scored_ids = {item["persona_id"] for item in wave1 + wave2}
    failed_ids = sorted({item["persona_id"] for item in errors + wave2_errors if "persona_id" in item})
    waves_run = [1, 2] if passed else [1]
    intended = len(all_personas) if passed else len(wave1_people)
    scored = len(scored_ids)
    wave2_candidates = (wave2_people if passed else
                        select_wave2_personas(all_personas,
                                              exclude_ids={p["id"] for p in wave1_people}))
    gate_skipped_ids = ({p["id"] for p in wave2_candidates}
                        if not passed and len(wave1) == len(wave1_people) else set())
    failed_ids_set = set(failed_ids)
    known_ids = scored_ids | failed_ids_set | gate_skipped_ids
    not_started_ids = {p["id"] for p in wave1_people + wave2_candidates} - known_ids
    completion_status = "complete" if scored == intended else "incomplete"
    execution_summary = {
        "total_personas": len(all_personas),
        "completed": len(scored_ids),
        "failed": len(failed_ids_set),
        "gate_skipped": len(gate_skipped_ids),
        "not_started": len(not_started_ids),
        "overall_completion_status": completion_status,
    }
    logger.info("Persona execution summary %s", json.dumps(execution_summary, sort_keys=True))
    crossover_score = (sum(item["score"] for item in wave2) / len(wave2)
                       if 2 in waves_run and wave2 else None)
    return {"wave1": wave1, "wave1_personas": list(wave1_people), "wave2": wave2,
            "core_resonance": round(core_resonance, 4),
            "crossover_score": round(crossover_score, 4) if crossover_score is not None else None,
            "gate_passed": passed,
            "waves_run": waves_run,
            "wave1_selection": selection_type, "wave1_raw_ids": raw_ids,
            "dimension_means": means_by_wave,
            "personas_intended": intended, "personas_scored": scored,
            "personas_failed": failed_ids, "complete": scored == intended,
            "persona_execution_summary": execution_summary,
            "personas_gate_skipped": sorted(gate_skipped_ids),
            "personas_not_started": sorted(not_started_ids),
            "errors": errors + wave2_errors}
