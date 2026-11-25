import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from core import config, llm, persona_react, scoring, suggestions, waves
from core.personas import load_personas


def reaction(value=5):
    return {"stop_scrolling": value, "share_likelihood": value, "watch_to_end": value,
            "comment_likelihood": value, "reason": "specific reason", "biggest_issue": "slow opening",
            "criteria_cited": ["hook"]}


def test_score_interval_is_unavailable_below_minimum_persona_count():
    assert scoring.score_interval([]) is None
    assert scoring.score_interval([4]) is None
    assert scoring.score_interval([4, 6]) is None


def test_score_interval_adapts_to_persona_score_distribution():
    narrow = scoring.score_interval([5, 5, 5, 5, 5, 5, 5, 5])
    broad = scoring.score_interval([1, 2, 3, 7, 8, 9, 4, 6])
    assert narrow is not None and broad is not None
    assert narrow[0] == narrow[1] == 5
    assert broad[1] - broad[0] > narrow[1] - narrow[0]
    assert broad == scoring.score_interval([1, 2, 3, 7, 8, 9, 4, 6])


@pytest.mark.parametrize("scores,expected", [([0, 0, 0], [0, 0]), ([10, 10, 10], [10, 10])])
def test_score_interval_respects_zero_to_ten_boundaries(scores, expected):
    assert scoring.score_interval(scores) == expected


def test_score_interval_contains_unchanged_point_score():
    scores = [1, 4, 5, 6, 9, 3, 7, 2]
    result = scoring.score_result({
        "wave1": [{"score": value} for value in scores],
        "wave2": [], "waves_run": [1], "gate_passed": False,
    })
    lower, upper = result["core_score_interval"]
    assert result["core_resonance"] == round(sum(scores) / len(scores), 4)
    assert lower <= result["core_resonance"] <= upper
    assert result["final_score"] == result["core_resonance"]


def test_display_confidence_fallback_derives_from_wave1_without_mutating_saved_result():
    result = {"wave1": [{"score": score} for score in [4, 5, 6, 7]]}
    assert scoring.confidence_for_display(result) == "Medium"
    assert "confidence" not in result


def test_display_confidence_preserves_existing_saved_value():
    result = {
        "confidence": "Low",
        "wave1": [{"score": score} for score in [4, 5, 6, 7]],
    }
    assert scoring.confidence_for_display(result) == "Low"
    assert result["confidence"] == "Low"


def test_display_confidence_is_unavailable_without_scored_personas():
    result = {"wave1": []}
    assert scoring.confidence_for_display(result) is None
    assert "confidence" not in result


def dynamic_audience_response():
    return {"personas": [
        {"slot": index, "archetype": archetype, "age": 29,
         "context": f"Balances a busy week with interest area {index}.",
         "cares_about": [f"sub-interest-{index}", f"taste-{index}-b", f"taste-{index}-c"],
         "pet_peeves": [f"peeve-{index}-a", f"peeve-{index}-b"]}
        for index, archetype in enumerate(config.WAVE1_SLOT_ARCHETYPES, start=1)
    ]}


def fixed_ids_for_slots(people):
    queues = {}
    for persona in sorted(people, key=lambda item: item["id"]):
        queues.setdefault(persona["archetype"], []).append(persona["id"])
    consumed = {}
    ids = []
    for archetype in config.WAVE1_SLOT_ARCHETYPES:
        index = consumed.get(archetype, 0)
        ids.append(queues[archetype][index])
        consumed[archetype] = index + 1
    return ids


def test_scale_anchor_and_persona_prompt(monkeypatch):
    captured = {}
    calls = []
    def fake_ask_json(prompt, **kwargs):
        calls.append(prompt)
        captured.update(kwargs)
        return reaction(99)
    monkeypatch.setattr(llm, "ask_json", fake_ask_json)
    result = persona_react.react(load_personas()[0], {"transcript": "music"}, "sound_on")
    prompt = persona_react.PROMPT_PATH.read_text(encoding="utf-8")
    assert "0 = no interest, 5 = average, 10 = would definitely do it" in prompt
    assert "Use the full scale" in prompt
    assert all(dimension in prompt for dimension in (
        "stop_scrolling:", "watch_to_end:", "share_likelihood:", "comment_likelihood:"
    ))
    assert "must not drive stop_scrolling near zero" in prompt
    assert "Never claim a sound is trending or viral" in prompt
    assert len(calls) == config.WAVE1_RUNS == 3
    assert captured["temperature"] == config.PERSONA_TEMPERATURE
    assert config.PERSONA_TEMPERATURE <= 0.1
    assert captured["response_json_schema"] == persona_react.PERSONA_RESPONSE_SCHEMA
    assert captured["response_json_schema"]["additionalProperties"] is False
    assert result["stop_scrolling"] == config.SCALE_MAX


def test_sound_conditions_have_distinct_instructions(monkeypatch):
    prompts = []
    monkeypatch.setattr(llm, "ask_json", lambda prompt, **kwargs: prompts.append(prompt) or reaction())
    persona = load_personas()[0]
    brief = {"music_present": True, "transcript": "spoken hook", "on_screen_text": "caption"}
    persona_react.react(persona, brief, "sound_on")
    persona_react.react(persona, brief, "sound_off")
    assert "Assume sound is ON and audible" in prompts[0]
    assert "do not mention muting or missing audio" in prompts[0]
    assert "Assume you cannot hear any audio" in prompts[config.WAVE1_RUNS]
    assert prompts[0] != prompts[config.WAVE1_RUNS]


def test_persona_reaction_uses_per_dimension_medians(monkeypatch):
    responses = [
        {"stop_scrolling": 2, "watch_to_end": 4, "share_likelihood": 8, "comment_likelihood": 6,
         "reason": "first", "biggest_issue": "first issue", "criteria_cited": ["hook"]},
        {"stop_scrolling": 9, "watch_to_end": 5, "share_likelihood": 3, "comment_likelihood": 1,
         "reason": "second", "biggest_issue": "second issue", "criteria_cited": ["retention"]},
        {"stop_scrolling": 6, "watch_to_end": 10, "share_likelihood": 5, "comment_likelihood": 9,
         "reason": "third", "biggest_issue": "third issue", "criteria_cited": ["cta"]},
    ]
    calls = iter(responses)
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: next(calls))
    result = persona_react.react(load_personas()[0], {"niche_category": "comedy"}, "sound_on")
    assert [result[key] for key in (
        "stop_scrolling", "watch_to_end", "share_likelihood", "comment_likelihood"
    )] == [6, 5, 5, 6]
    assert result["criteria_cited"] == ["hook", "retention", "cta"]


def test_batched_persona_reactions_preserve_persona_and_condition_reasons(monkeypatch):
    personas = load_personas()[:2]
    calls = []

    def fake_ask(prompt, **kwargs):
        calls.append((prompt, kwargs))
        profiles = json.loads(prompt.split("PERSONAS:", 1)[1].split("BRIEF WITH SOUND", 1)[0].strip())
        return {"reactions": [{
            "persona_id": item["persona_id"],
            "sound_on": dict(reaction(7), reason=f"on for {item['persona_id']}"),
            "sound_off": dict(reaction(3), reason=f"off for {item['persona_id']}"),
        } for item in profiles]}

    monkeypatch.setattr(llm, "ask_json", fake_ask)
    result = persona_react.react_batch(personas, {"on_screen_text": "caption"}, runs=1)
    assert set(result) == {persona["id"] for persona in personas}
    assert result[personas[0]["id"]]["sound_on"]["reason"] == f"on for {personas[0]['id']}"
    assert result[personas[1]["id"]]["sound_off"]["reason"] == f"off for {personas[1]['id']}"
    assert len(calls) == 1
    schema = calls[0][1]["response_json_schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["reactions"]["items"]["additionalProperties"] is False
    assert schema["properties"]["reactions"]["items"]["properties"]["sound_on"] == persona_react.REACTION_SCHEMA


def test_wave1_and_wave2_use_distinct_run_counts(monkeypatch):
    people = load_personas()
    monkeypatch.setattr(waves, "choose_wave1", lambda brief, personas: (people[:config.WAVE1_SIZE], None))
    calls = []

    def fake_wave(selected, brief, audio_available, reaction_cache=None, persona_runs=None,
                  wave_number=1):
        calls.append((len(selected), persona_runs))
        return ([{"persona_id": persona["id"], "score": 8} for persona in selected], [])

    monkeypatch.setattr(waves, "run_wave", fake_wave)
    waves.simulate({}, personas=people)
    assert calls == [(config.WAVE1_SIZE, config.WAVE1_RUNS),
                     (config.WAVE2_SIZE, config.WAVE2_RUNS)]
    assert config.WAVE1_RUNS == 3
    assert config.WAVE2_RUNS == 1
    assert config.BATCH_SIZE == 3
    assert config.MAX_CALLS_PER_MIN == 10


def test_batched_reactions_are_cached_for_retry_resume(monkeypatch, tmp_path):
    from core.pipeline import _ReactionCache

    persona = load_personas()[0]
    cache = _ReactionCache(tmp_path, "batch-cache-key")
    calls = []

    def fake_ask(prompt, **kwargs):
        calls.append(prompt)
        profiles = json.loads(prompt.split("PERSONAS:", 1)[1].split("BRIEF WITH SOUND", 1)[0].strip())
        return {"reactions": [{"persona_id": item["persona_id"],
                               "sound_on": reaction(7), "sound_off": reaction(5)}
                              for item in profiles]}

    monkeypatch.setattr(llm, "ask_json", fake_ask)
    first, first_errors = waves.run_wave([persona], {}, True, cache, persona_runs=1)
    second, second_errors = waves.run_wave([persona], {}, True, cache, persona_runs=1)
    assert not first_errors and not second_errors
    assert first == second
    assert len(calls) == 1


def test_persona_call_stages_survive_wave_thread_pool(monkeypatch):
    seen = []
    def fake_ask(prompt, **kwargs):
        seen.append(llm._CALL_STAGE.get())
        profiles = json.loads(prompt.split("PERSONAS:", 1)[1].split("BRIEF WITH SOUND", 1)[0].strip())
        return {"reactions": [{"persona_id": item["persona_id"],
                               "sound_on": reaction(), "sound_off": reaction()}
                              for item in profiles]}
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    persona = load_personas()[0]
    results, errors = waves.run_wave([persona], {"on_screen_text": "hi"}, audio_available=True)
    assert not errors and len(results) == 1
    assert len(seen) == 3
    assert all(stage.startswith(f"persona:{persona['id']}:batch") for stage in seen)


def test_successful_persona_batch_logs_each_persona_and_model_attempts(monkeypatch, caplog):
    caplog.set_level("INFO")
    people = load_personas()[:2]
    monkeypatch.setattr(llm, "CALL_EVENTS", [])

    def fake_batch(personas, *args, **kwargs):
        stage = llm._CALL_STAGE.get()
        llm.CALL_EVENTS.append({"stage": stage, "model": "mock-primary",
                                "attempt": 1, "status": "OK"})
        return {persona["id"]: {"sound_on": reaction(), "sound_off": reaction()}
                for persona in personas}

    monkeypatch.setattr(waves, "react_batch", fake_batch)
    results, errors = waves.run_wave(people, {}, persona_runs=1, wave_number=2)

    assert len(results) == 2
    assert not errors
    for persona in people:
        assert f"Persona start persona_id={persona['id']}" in caplog.text
        assert (f"Persona completion persona_id={persona['id']}" in caplog.text
                and "wave=2 batch_id=wave2-batch01 model=mock-primary attempt_count=1" in caplog.text
                and "final_status=COMPLETED" in caplog.text)


def test_failed_persona_batch_logs_original_safely_and_sanitizes_result(monkeypatch, caplog):
    caplog.set_level("INFO")
    people = load_personas()[:2]
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret-marker")
    monkeypatch.setattr(llm, "CALL_EVENTS", [])

    def fail_batch(personas, *args, **kwargs):
        stage = llm._CALL_STAGE.get()
        llm.CALL_EVENTS.append({"stage": stage, "model": "mock-primary",
                                "attempt": 2, "status": "OVERLOADED"})
        raise RuntimeError("provider error key=test-secret-marker")

    monkeypatch.setattr(waves, "react_batch", fail_batch)
    results, errors = waves.run_wave(people, {}, persona_runs=1, wave_number=1)

    assert results == []
    assert len(errors) == len(people)
    assert all("test-secret-marker" not in error["error"] for error in errors)
    assert all(error["batch_id"] == "wave1-batch01" for error in errors)
    assert "Persona completion" in caplog.text
    assert "RuntimeError: provider error key=[redacted]" in caplog.text
    assert "test-secret-marker" not in caplog.text


def test_default_persona_concurrency_is_one():
    assert config.MAX_WORKERS == 1


@pytest.mark.parametrize("workers", [1, 2])
def test_run_wave_respects_configured_worker_limit(monkeypatch, workers):
    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_batch(batch, *args):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return [{"persona_id": person["id"], "score": 5} for person in batch]

    monkeypatch.setattr(config, "MAX_WORKERS", workers)
    monkeypatch.setattr(config, "BATCH_SIZE", 1)
    monkeypatch.setattr(waves, "_persona_batch", fake_batch)
    results, errors = waves.run_wave(load_personas()[:7], {}, audio_available=False)

    assert len(results) == 7
    assert not errors
    assert peak <= workers


@pytest.mark.parametrize("mode", ["mixed", "on_only", "off_only"])
def test_sound_mixing_modes(monkeypatch, mode):
    monkeypatch.setattr(config, "SOUND_MODE", mode)
    monkeypatch.setattr(config, "SCORE_WEIGHTS", {"stop_scrolling": 1.0})
    monkeypatch.setattr(waves, "react", lambda p, b, condition: reaction(10 if condition == "sound_on" else 0))
    persona = dict(load_personas()[0], sound_on_probability=0.25)
    result = waves._one_persona(persona, {}, True)
    assert result["score"] == {"mixed": 2.5, "on_only": 10, "off_only": 0}[mode]


def test_no_audio_skips_silent_reaction(monkeypatch):
    calls = []
    monkeypatch.setattr(waves, "react", lambda p, b, condition: calls.append(condition) or reaction())
    result = waves._one_persona(load_personas()[0], {}, False)
    assert calls == ["sound_on"]
    assert result["sound_off"] is None


def test_failed_persona_call_is_recorded_and_tolerated(monkeypatch):
    monkeypatch.setattr(waves, "react_batch", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mock issue")))
    results, errors = waves.run_wave(load_personas()[:1], {}, True)
    assert results == []
    assert "mock issue" in errors[0]["error"]


@pytest.mark.parametrize("value,passed", [(8, True), (2, False)])
def test_gate_pass_and_fail(monkeypatch, value, passed):
    people = load_personas()
    monkeypatch.setattr(waves, "choose_wave1", lambda brief, personas: (people[:2], None))
    monkeypatch.setattr(waves, "run_wave", lambda selected, brief, audio_available, **kwargs: (
        [{"persona_id": p["id"], "score": value} for p in selected], []))
    output = waves.simulate({}, personas=people)
    assert output["gate_passed"] is passed
    assert bool(output["wave2"]) is passed


@pytest.mark.parametrize("returned,expected,selection", [
    (lambda people: [p["id"] for p in people[:config.WAVE1_SIZE]], "llm", "llm"),
    (lambda people: [p["id"] for p in people] + [people[0]["id"]], "llm", "llm"),
    (lambda people: [people[0]["id"]], "llm_filled", "llm_filled"),
    (lambda people: [people[0]["id"], people[0]["id"]], "llm_filled", "llm_filled"),
    (lambda people: ["unknown", people[0]["id"]], "llm_filled", "llm_filled"),
    (lambda people: ["unknown"], "fallback", "fallback"),
])
def test_audience_selection_tolerates_ids(monkeypatch, returned, expected, selection):
    people = load_personas()
    ids = returned(people)
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: {"ids": ids})
    chosen, error = waves._choose_wave1_by_match({}, people)
    assert len(chosen) == config.WAVE1_SIZE
    assert len({p["id"] for p in chosen}) == config.WAVE1_SIZE
    assert chosen.selection == selection
    assert chosen.raw_ids == ids
    if expected == "fallback":
        assert "fallback" in error
    else:
        assert error is None
    if ids and ids[0] in {p["id"] for p in people}:
        assert chosen[0]["id"] == ids[0]


def test_malformed_audience_json_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("malformed JSON")))
    chosen, error = waves.choose_wave1({}, load_personas())
    assert len(chosen) == config.WAVE1_SIZE
    assert chosen.selection == "fixed_fallback"
    assert chosen.raw_ids == []
    assert "malformed JSON" in error


def test_dynamic_audience_prompt_uses_configured_slots_and_stage(monkeypatch):
    people = load_personas()
    captured = {}
    def fake_ask(prompt, **kwargs):
        captured.update(prompt=prompt, kwargs=kwargs, stage=llm._CALL_STAGE.get())
        return dynamic_audience_response()
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    chosen, _ = waves.choose_wave1({}, people)
    assert "never describe, refer to, imply, or hint at the supplied reel" in captured["prompt"].lower()
    assert "real person" in captured["prompt"].lower()
    assert captured["stage"] == "audience_personas"
    assert captured["kwargs"]["temperature"] == config.AUDIENCE_PERSONA_TEMPERATURE
    assert len(chosen) == config.WAVE1_SIZE
    assert [item["archetype"] for item in chosen] == config.WAVE1_SLOT_ARCHETYPES
    slots_block = captured["prompt"].split("WAVE 1 SLOTS:\n", 1)[1]
    assert len(json.loads(slots_block)) == config.WAVE1_SIZE


def test_dynamic_profiles_preserve_template_behaviour_and_use_deterministic_ids(monkeypatch):
    people = load_personas()
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: dynamic_audience_response())
    chosen, error = waves.choose_wave1({"target_audience_guess": {"age_range": "20-40"}}, people)
    assert error is None
    assert chosen.selection == "dynamic"
    assert [p["id"] for p in chosen] == [
        "genz_dyn_1", "genz_dyn_2", "casual_dyn_1", "casual_dyn_2",
        "niche_fan_dyn_1", "niche_fan_dyn_2", "skeptic_dyn_1", "creator_dyn_1",
    ]
    templates = {}
    for persona in sorted(people, key=lambda item: item["id"]):
        templates.setdefault(persona["archetype"], []).append(persona)
    counts = {}
    for index, persona in enumerate(chosen, start=1):
        archetype = persona["archetype"]
        template_index = counts.get(archetype, 0)
        template = templates[archetype][template_index % len(templates[archetype])]
        counts[archetype] = template_index + 1
        for field in ("sound_on_probability", "base_skip_rate", "scroll_habit", "age", "gender", "region", "language_pref"):
            assert persona[field] == template[field]
        assert persona["interests"] == [f"sub-interest-{index}", f"taste-{index}-b", f"taste-{index}-c"]
        assert persona["cares_about"] == persona["interests"]
        assert persona["pet_peeves"] == [f"peeve-{index}-a", f"peeve-{index}-b"]
        assert persona["context"] == f"Balances a busy week with interest area {index}."


def test_malformed_or_missing_dynamic_slots_are_filled_from_fixed_selection(monkeypatch):
    people = load_personas()
    response = dynamic_audience_response()
    response["personas"] = [item for item in response["personas"] if item["slot"] != 4]
    response["personas"][2]["cares_about"] = ["", "bad", "slot"]
    fixed_ids = fixed_ids_for_slots(people)
    calls = iter([response, {"ids": fixed_ids}])
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: next(calls))
    chosen, error = waves.choose_wave1({}, people)
    assert error is None
    assert chosen.selection == "dynamic_filled"
    assert len(chosen) == config.WAVE1_SIZE
    assert [p["archetype"] for p in chosen] == config.WAVE1_SLOT_ARCHETYPES
    assert chosen[2]["id"] in fixed_ids
    assert chosen[3]["id"] in fixed_ids


def test_dynamic_call_failure_uses_old_fixed_audience_match(monkeypatch):
    people = load_personas()
    fixed_ids = fixed_ids_for_slots(people)
    calls = iter([RuntimeError("mock audience generation failure"), {"ids": fixed_ids}])
    def fake_ask(*args, **kwargs):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    chosen, error = waves.choose_wave1({}, people)
    assert chosen.selection == "fixed_fallback"
    assert [item["id"] for item in chosen] == fixed_ids
    assert "mock audience generation failure" in error


def test_wave2_sample_is_fixed_balanced_and_has_no_repeats():
    people = load_personas()
    first = waves.select_wave2_personas(people)
    second = waves.select_wave2_personas(people)
    assert config.WAVE2_SIZE == 16
    assert len(first) == config.WAVE2_SIZE
    assert [p["id"] for p in first] == [p["id"] for p in second]
    assert len({p["id"] for p in first}) == config.WAVE2_SIZE
    counts = {}
    for persona in first:
        counts[persona["archetype"]] = counts.get(persona["archetype"], 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_simulate_exposes_wave1_selection_and_raw_ids(monkeypatch):
    people = load_personas()
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: {"ids": [people[0]["id"]]})
    monkeypatch.setattr(waves, "run_wave", lambda selected, brief, audio_available, **kwargs: ([], []))
    result = waves.simulate({}, personas=people)
    assert result["wave1_selection"] == "dynamic_filled"
    assert result["wave1_raw_ids"] == [persona["id"] for persona in result["wave1_personas"]]


def test_dimension_means_are_per_wave_and_sound_condition(monkeypatch):
    people = load_personas()
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: {"ids": [p["id"] for p in people[:config.WAVE1_SIZE]]})
    wave1_results = [
        {"persona_id": p["id"], "score": 8,
         "sound_on": reaction(2), "sound_off": reaction(4)}
        for p in people[:config.WAVE1_SIZE]
    ]
    wave2_results = [
        {"persona_id": p["id"], "score": 8,
         "sound_on": reaction(6), "sound_off": reaction(10)}
        for p in people[config.WAVE1_SIZE:]
    ]
    calls = iter([(wave1_results, []), (wave2_results, [])])
    monkeypatch.setattr(waves, "run_wave", lambda *args, **kwargs: next(calls))
    result = waves.simulate({}, personas=people)
    assert result["dimension_means"] == {
        "wave1": {
            "sound_on": {"stop_scrolling": 2, "watch_to_end": 2, "share_likelihood": 2, "comment_likelihood": 2},
            "sound_off": {"stop_scrolling": 4, "watch_to_end": 4, "share_likelihood": 4, "comment_likelihood": 4},
        },
        "wave2": {
            "sound_on": {"stop_scrolling": 6, "watch_to_end": 6, "share_likelihood": 6, "comment_likelihood": 6},
            "sound_off": {"stop_scrolling": 10, "watch_to_end": 10, "share_likelihood": 10, "comment_likelihood": 10},
        },
    }


def test_reach_tier_is_capped_after_gate_failure(monkeypatch):
    monkeypatch.setattr(config, "REACH_TIERS", {"Low": 0, "Medium": 4, "High": 7, "Very high": 9})
    result = scoring.score_result({"wave1": [{"score": 10}], "wave2": [], "gate_passed": False})
    assert result["reach_tier"] == "Low"
    assert result["tier_note"] == "stays in test audience"
    assert result["label"] == "Virality potential estimate"


def test_core_crossover_and_final_score_use_wave1_resonance():
    result = scoring.score_result({
        "wave1": [{"score": 2}, {"score": 4}],
        "wave2": [{"score": 8}, {"score": 10}],
        "waves_run": [1, 2], "gate_passed": True,
        "personas_intended": 4, "personas_scored": 4, "complete": True,
    })
    assert result["core_resonance"] == 3
    assert result["crossover_score"] == 9
    assert result["final_score"] == result["core_resonance"] == 3
    assert result["core_resonance_spread"] == 1.0
    assert result["confidence"] == "High"


@pytest.mark.parametrize(
    "scores,expected_confidence",
    [([3, 5], "High"), ([2, 6], "Medium"), ([2, 7], "Low")],
)
def test_confidence_uses_population_spread_thresholds(scores, expected_confidence):
    result = scoring.score_result({
        "wave1": [{"score": score} for score in scores],
        "wave2": [], "gate_passed": False,
    })
    assert result["core_resonance"] == sum(scores) / len(scores)
    expected_spread = (
        sum((score - result["core_resonance"]) ** 2 for score in scores) / len(scores)
    ) ** 0.5
    assert result["core_resonance_spread"] == pytest.approx(expected_spread)
    assert result["confidence"] == expected_confidence


def test_gate_failure_keeps_core_final_and_null_crossover(monkeypatch):
    people = load_personas()
    selected = people[:config.WAVE1_SIZE]
    monkeypatch.setattr(waves, "choose_wave1", lambda brief, personas: (selected, None))
    monkeypatch.setattr(waves, "run_wave", lambda selected_people, brief, audio_available, **kwargs: (
        [{"persona_id": p["id"], "score": 2, "sound_on": reaction(), "sound_off": reaction()}
         for p in selected_people], []))
    output = waves.simulate({}, personas=people)
    result = scoring.score_result(output)
    assert output["core_resonance"] == 2
    assert output["gate_passed"] is False
    assert output["waves_run"] == [1]
    assert output["complete"] is True
    assert result["crossover_score"] is None
    assert result["final_score"] == result["core_resonance"] == 2


def test_gate_failure_with_all_wave1_scored_is_complete(monkeypatch):
    people = load_personas()
    selected = people[:config.WAVE1_SIZE]
    monkeypatch.setattr(waves, "choose_wave1", lambda brief, personas: (selected, None))
    monkeypatch.setattr(waves, "run_wave", lambda selected_people, brief, audio_available, **kwargs: (
        [{"persona_id": p["id"], "score": 2, "sound_on": reaction(), "sound_off": reaction()}
         for p in selected_people], []))
    output = waves.simulate({}, personas=people)
    scored = scoring.score_result(output)
    assert output["gate_passed"] is False
    assert output["waves_run"] == [1]
    assert output["personas_intended"] == config.WAVE1_SIZE
    assert output["personas_scored"] == config.WAVE1_SIZE
    assert output["personas_failed"] == []
    assert output["complete"] is True
    assert output["persona_execution_summary"] == {
        "total_personas": len(people), "completed": config.WAVE1_SIZE,
        "failed": 0, "gate_skipped": len(people) - config.WAVE1_SIZE,
        "not_started": 0, "overall_completion_status": "complete",
    }
    assert len(output["personas_gate_skipped"]) == len(people) - config.WAVE1_SIZE
    assert scored["complete"] is True
    assert scored["gate_passed"] is False
    assert scored["final_score"] == 2
    assert scored["reach_tier"] == "Low"
    assert scored["tier_note"] == "stays in test audience"


def test_gate_pass_with_wave2_failure_is_incomplete(monkeypatch):
    people = load_personas()
    selected = people[:config.WAVE1_SIZE]
    monkeypatch.setattr(waves, "choose_wave1", lambda brief, personas: (selected, None))
    calls = iter([
        ([{"persona_id": p["id"], "score": 5, "sound_on": reaction(), "sound_off": reaction()}
          for p in selected], []),
        ([{"persona_id": p["id"], "score": 5, "sound_on": reaction(), "sound_off": reaction()}
          for p in people[config.WAVE1_SIZE:-1]],
         [{"persona_id": people[-1]["id"], "error": "RuntimeError: mock failure"}]),
    ])
    monkeypatch.setattr(waves, "run_wave", lambda *args, **kwargs: next(calls))
    output = waves.simulate({}, personas=people)
    scored = scoring.score_result(output)
    assert output["gate_passed"] is True
    assert output["waves_run"] == [1, 2]
    assert output["personas_intended"] == len(people)
    assert output["personas_scored"] == len(people) - 1
    assert output["complete"] is False
    assert scored["complete"] is False
    assert scored["reach_tier"] == "Incomplete - rerun"
    assert scored["final_score"] == scored["core_resonance"] == 5
    assert scored["crossover_score"] == 5


def test_incomplete_wave1_does_not_gate_or_start_wave2(monkeypatch):
    people = load_personas()
    selected = people[:config.WAVE1_SIZE]
    monkeypatch.setattr(waves, "choose_wave1", lambda brief, personas: (selected, None))
    calls = []

    def partial_wave(selected_people, brief, audio_available, **kwargs):
        calls.append(len(selected_people))
        return ([{"persona_id": person["id"], "score": 9,
                  "sound_on": reaction(9), "sound_off": reaction(9)}
                 for person in selected_people[:4]],
                [{"persona_id": person["id"], "error": "RuntimeError: 503 overloaded"}
                 for person in selected_people[4:]])

    monkeypatch.setattr(waves, "run_wave", partial_wave)
    output = waves.simulate({}, personas=people)
    scored = scoring.score_result(output)

    assert calls == [config.WAVE1_SIZE]
    assert output["gate_passed"] is False
    assert output["waves_run"] == [1]
    assert output["personas_intended"] == config.WAVE1_SIZE
    assert output["personas_scored"] == 4
    assert output["complete"] is False
    assert output["persona_execution_summary"] == {
        "total_personas": len(people), "completed": 4, "failed": 4,
        "gate_skipped": 0, "not_started": len(people) - config.WAVE1_SIZE,
        "overall_completion_status": "incomplete",
    }
    assert scored["complete"] is False
    assert scored["reach_tier"] == "Incomplete - rerun"


def test_gate_pass_with_all_personas_scored_is_complete(monkeypatch):
    people = load_personas()
    selected = people[:config.WAVE1_SIZE]
    monkeypatch.setattr(waves, "choose_wave1", lambda brief, personas: (selected, None))
    monkeypatch.setattr(waves, "run_wave", lambda selected_people, brief, audio_available, **kwargs: (
        [{"persona_id": p["id"], "score": 5, "sound_on": reaction(), "sound_off": reaction()}
         for p in selected_people], []))
    output = waves.simulate({}, personas=people)
    scored = scoring.score_result(output)
    assert output["gate_passed"] is True
    assert output["waves_run"] == [1, 2]
    assert output["personas_intended"] == len(people)
    assert output["personas_scored"] == len(people)
    assert output["complete"] is True
    assert output["persona_execution_summary"] == {
        "total_personas": len(people), "completed": len(people), "failed": 0,
        "gate_skipped": 0, "not_started": 0,
        "overall_completion_status": "complete",
    }
    assert scored["complete"] is True
    assert scored["final_score"] == 5
    assert scored["reach_tier"] == "Medium"


def test_reaction_cache_skips_successful_condition_on_resume(monkeypatch, tmp_path):
    import uuid
    from core.pipeline import _ReactionCache
    cache = _ReactionCache(tmp_path / "reaction_cache_test", f"video-model-{uuid.uuid4().hex}")
    calls = []
    monkeypatch.setattr(waves, "react", lambda persona, brief, condition: calls.append(condition) or reaction(7))
    persona = load_personas()[0]
    first = waves._one_persona(persona, {}, True, cache)
    second = waves._one_persona(persona, {}, True, cache)
    assert calls == ["sound_on", "sound_off"]
    assert first == second


def test_reaction_cache_path_hashes_full_persona_profile(tmp_path):
    from core.pipeline import _ReactionCache
    cache = _ReactionCache(tmp_path, "video-model-hash")
    profile = dict(load_personas()[0])
    changed = dict(profile, cares_about=["new audience taste"])
    assert cache._path(profile, "sound_on") == cache._path(profile, "sound_on")
    assert cache._path(profile, "sound_on") != cache._path(changed, "sound_on")


def test_wave1_persona_cache_keys_video_brief_and_prompt(tmp_path):
    from core.pipeline import _Wave1PersonaCache
    cache = _Wave1PersonaCache(tmp_path, "video-hash", "prompt-hash")
    brief = {"topic_niche": "software"}
    path, parts = cache._path(brief)
    profiles = [{"id": f"genz_dyn_{i}"} for i in range(config.WAVE1_SIZE)]
    cache.set(brief, profiles, "dynamic", [p["id"] for p in profiles])
    assert set(parts) == {"video_hash", "brief_hash", "prompt_hash", "context_hash"}
    assert cache.get(brief)["wave1_personas"] == profiles
    assert cache._path({"topic_niche": "cooking"})[0] != path
    assert _Wave1PersonaCache(tmp_path, "other-video", "prompt-hash")._path(brief)[0] != path
    assert _Wave1PersonaCache(tmp_path, "video-hash", "other-prompt")._path(brief)[0] != path


def test_wave_report_runs_on_utf8_sample_result(tmp_path):
    result_path = tmp_path / "result_sample.json"
    result_path.write_text(json.dumps({
        "wave1_selection": "dynamic", "core_resonance": 4.25, "crossover_score": None,
        "wave1": [{"persona_id": "genz_dyn_1", "score": 4.25}],
        "wave1_personas": [{"id": "genz_dyn_1", "cares_about": ["கதை", "design"]}],
    }, ensure_ascii=False), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "wave_report.py"
    output = subprocess.run([sys.executable, "-X", "utf8", str(script), str(result_path)],
                            check=True, capture_output=True, text=True, encoding="utf-8").stdout
    assert "wave1_selection: dynamic" in output
    assert "core_resonance: 4.25" in output
    assert "crossover_score: null" in output
    assert "score=4.25" in output
    assert "கதை" in output


def test_suggestions_validate_and_retry_invalid_json(monkeypatch):
    valid = {"suggestions": [{"suggestion": f"Improve {i}", "evidence": "slow opening", "persona_id": "p1", "quote": "slow opening", "criterion": "hook"} for i in range(3)]}
    calls = iter([{"suggestions": []}, valid])
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: next(calls))
    result, errors = suggestions.generate({}, [{"persona_id": "p1", "sound_on": reaction()}])
    assert len(result) == 3 and len(errors) == 1
    assert all(item["evidence"] for item in result)


def test_suggestions_reject_unknown_persona_and_unquoted_evidence(monkeypatch):
    valid = {"suggestions": [{"suggestion": f"Improve {i}", "evidence": "some evidence",
                              "persona_id": "p1", "quote": "not in the persona text", "criterion": "hook"} for i in range(3)]}
    calls = iter([valid, valid])
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: next(calls))
    result, errors = suggestions.generate({}, [{"persona_id": "p1", "sound_on": reaction()}])
    assert result == []
    assert len(errors) == 2 and "quote is not present" in errors[-1]


def test_suggestions_reject_persona_id_absent_from_this_run(monkeypatch):
    payload = {"suggestions": [{"suggestion": "Change it", "evidence": "because", "persona_id": "outsider",
                                "quote": "specific reason", "criterion": "hook"} for _ in range(3)]}
    calls = iter([payload, payload])
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: next(calls))
    result, errors = suggestions.generate({}, [{"persona_id": "p1", "sound_on": reaction()}])
    assert result == []
    assert "must reference a persona" in errors[-1]


def test_suggestions_accept_case_insensitive_quote_from_named_persona(monkeypatch):
    payload = {"suggestions": [{"suggestion": f"Change {i}", "evidence": "reaction reason",
                                "persona_id": "p1", "quote": "SPECIFIC REASON", "criterion": "hook"} for i in range(3)]}
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: payload)
    result, errors = suggestions.generate({}, [{"persona_id": "p1", "sound_on": reaction()}])
    assert len(result) == 3 and not errors
    prompt = suggestions.PROMPT_PATH.read_text(encoding="utf-8")
    assert "Do not invent numbers, quantities, tools or techniques" in prompt


def test_cache_hit_skips_calls(monkeypatch):
    from core import pipeline
    root = pipeline.ROOT / "tmp" / "phase3_cache_test"
    root.mkdir(parents=True, exist_ok=True)
    video = root / "clip.mp4"
    video.write_bytes(b"mock video bytes")
    monkeypatch.setattr(pipeline, "CACHE_DIR", root / "cache")
    monkeypatch.setattr(pipeline, "audio", type("Audio", (), {"audio_brief": staticmethod(lambda *_: pytest.fail("cache must skip"))}))
    monkeypatch.setattr(llm, "CALL_EVENTS", [])
    cached = {"brief": {}, "wave1": [], "wave2": [], "per_persona_scores": {}, "final_score": 1,
              "reach_tier": "Low", "gate_passed": False, "suggestions": [], "errors": []}
    key = pipeline._key(video, llm._model_name())
    (root / "cache").mkdir(exist_ok=True)
    (root / "cache" / f"{key}.json").write_text(json.dumps(cached), encoding="utf-8")
    out = root / "result.json"
    result = pipeline.run(video, out)
    assert result["call_count"] == 0
    assert json.loads(out.read_text(encoding="utf-8"))["final_score"] == 1


def test_empty_persona_scores_skip_suggestions_and_keep_incomplete_placeholder(monkeypatch, tmp_path):
    from core import pipeline

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(pipeline, "audio", type("Audio", (), {
        "audio_brief": staticmethod(lambda *_: {"available": False, "reason": "no audio"})
    }))
    monkeypatch.setattr(pipeline, "extract_all", lambda *_a, **_k: {"frames": []})
    monkeypatch.setattr(pipeline.brief, "build_full_brief", lambda *_a, **_k: {
        "niche_category": "comedy", "topic": "skit"
    })
    wave_result = {
        "complete": False, "personas_scored": 0, "wave1": [], "wave2": [],
        "wave1_personas": [], "waves_run": [1], "wave1_selection": [],
        "wave1_raw_ids": [], "dimension_means": {}, "gate_passed": False,
        "errors": [{"stage": "audience_personas", "error": "failed"}],
        "persona_execution_summary": {}, "personas_gate_skipped": [],
        "personas_not_started": [],
    }
    monkeypatch.setattr(pipeline, "simulate", lambda *_a, **_k: wave_result)
    monkeypatch.setattr(pipeline, "score_result", lambda _waves: {
        "complete": False, "final_score": 0.0, "reach_tier": "Incomplete - rerun"
    })
    monkeypatch.setattr(pipeline, "generate", lambda *_a, **_k: pytest.fail(
        "suggestions must not call the model without scored persona results"
    ))

    result = pipeline.run(video, tmp_path / "result.json", use_cache=False)

    assert result["complete"] is False
    assert result["final_score"] == 0.0
    assert result["suggestions"] == []
    assert result["suggestions_status"] == "skipped"
    assert not any(error.get("stage") == "suggestions" for error in result["errors"])


def _write_suggestions_retry_result(path):
    persona_result = {
        "persona_id": "p1", "score": 7,
        "sound_on": {"reason": "strong hook", "biggest_issue": "slow setup"},
        "sound_off": {"reason": "clear visuals", "biggest_issue": "small text"},
    }
    result = {
        "complete": True, "brief": {"niche_category": "comedy", "topic": "skit"},
        "wave1": [persona_result], "wave2": [],
        "per_persona_scores": {"p1": 7}, "final_score": 7,
        "suggestions": [], "suggestions_status": "failed", "call_count": 18,
        "errors": [{"stage": "suggestions", "error": "old failure"},
                   {"stage": "audio", "error": "optional audio unavailable"}],
    }
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def _forbid_pipeline_stages(monkeypatch, pipeline):
    monkeypatch.setattr(pipeline, "audio", type("Audio", (), {
        "audio_brief": staticmethod(lambda *_: pytest.fail("audio stage must not run"))
    }))
    monkeypatch.setattr(pipeline, "extract_all", lambda *a, **k: pytest.fail("extract must not run"))
    monkeypatch.setattr(pipeline.brief, "build_full_brief",
                        lambda *a, **k: pytest.fail("brief stage must not run"))
    monkeypatch.setattr(pipeline, "simulate", lambda *a, **k: pytest.fail("persona stages must not run"))
    monkeypatch.setattr(pipeline, "score_result", lambda *a, **k: pytest.fail("scoring must not run"))


def test_suggestions_only_retry_success_persists_without_rerunning_persona_stages(monkeypatch, tmp_path):
    from core import pipeline

    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(pipeline.llm, "CALL_EVENTS", [])
    _forbid_pipeline_stages(monkeypatch, pipeline)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    out = tmp_path / "result.json"
    original = _write_suggestions_retry_result(out)
    cache = pipeline.CACHE_DIR / f"{pipeline._key(video, pipeline.llm._model_name())}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(original), encoding="utf-8")
    new_suggestions = [{"suggestion": f"New {index}"} for index in range(3)]
    captured = {}

    def fake_generate(brief, persona_results):
        captured["brief"] = brief
        captured["personas"] = persona_results
        return new_suggestions, []

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    result = pipeline.run(video, out, retry_suggestions=True)

    assert captured["brief"] == original["brief"]
    assert captured["personas"] == original["wave1"] + original["wave2"]
    assert result["suggestions"] == new_suggestions
    assert result["suggestions_status"] == "completed"
    assert result["errors"] == [{"stage": "audio", "error": "optional audio unavailable"}]
    for key in ("brief", "wave1", "wave2", "per_persona_scores", "final_score", "call_count"):
        assert result[key] == original[key]
    assert json.loads(out.read_text(encoding="utf-8")) == result
    assert json.loads(cache.read_text(encoding="utf-8")) == result


def test_suggestions_only_retry_failure_preserves_saved_analysis(monkeypatch, tmp_path):
    from core import pipeline

    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(pipeline.llm, "CALL_EVENTS", [])
    _forbid_pipeline_stages(monkeypatch, pipeline)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    out = tmp_path / "result.json"
    original = _write_suggestions_retry_result(out)
    cache = pipeline.CACHE_DIR / f"{pipeline._key(video, pipeline.llm._model_name())}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(pipeline, "generate", lambda *_: ([], ["RuntimeError: DAILY_QUOTA"]))

    result = pipeline.run(video, out, retry_suggestions=True)

    assert result["suggestions"] == original["suggestions"]
    assert result["suggestions_status"] == "failed"
    assert result["errors"] == [
        {"stage": "audio", "error": "optional audio unavailable"},
        {"stage": "suggestions", "error": "RuntimeError: DAILY_QUOTA"},
    ]
    for key in ("brief", "wave1", "wave2", "per_persona_scores", "final_score", "call_count"):
        assert result[key] == original[key]
    assert json.loads(out.read_text(encoding="utf-8")) == result
    assert json.loads(cache.read_text(encoding="utf-8")) == result
