import json
import subprocess
import sys
from pathlib import Path

import pytest

from core import config, llm, persona_react, scoring, suggestions, waves
from core.personas import load_personas


def reaction(value=5):
    return {"stop_scrolling": value, "share_likelihood": value, "watch_to_end": value,
            "comment_likelihood": value, "reason": "specific reason", "biggest_issue": "slow opening"}


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
    monkeypatch.setattr(llm, "ask_json", lambda prompt, **kwargs: captured.update(kwargs) or reaction(99))
    result = persona_react.react(load_personas()[0], {"transcript": "music"}, "sound_on")
    prompt = persona_react.PROMPT_PATH.read_text(encoding="utf-8")
    assert "0 = no interest, 5 = average, 10 = would definitely do it" in prompt
    assert "must not drive stop_scrolling near zero" in prompt
    assert "Never claim a sound is trending or viral" in prompt
    assert captured["temperature"] == config.PERSONA_TEMPERATURE
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
    assert "Assume you cannot hear any audio" in prompts[1]
    assert prompts[0] != prompts[1]


def test_persona_call_stages_survive_wave_thread_pool(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "ask_json", lambda prompt, **kwargs: (seen.append(llm._CALL_STAGE.get()) or reaction()))
    persona = load_personas()[0]
    results, errors = waves.run_wave([persona], {"on_screen_text": "hi"}, audio_available=True)
    assert not errors and len(results) == 1
    assert set(seen) == {f"persona:{persona['id']}:sound_on", f"persona:{persona['id']}:sound_off"}


def test_default_persona_concurrency_is_two():
    assert config.MAX_WORKERS == 2


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
    monkeypatch.setattr(waves, "react", lambda p, b, c: (_ for _ in ()).throw(RuntimeError("mock issue")))
    results, errors = waves.run_wave(load_personas()[:1], {}, True)
    assert results == []
    assert "mock issue" in errors[0]["error"]


@pytest.mark.parametrize("value,passed", [(8, True), (2, False)])
def test_gate_pass_and_fail(monkeypatch, value, passed):
    people = load_personas()
    monkeypatch.setattr(waves, "choose_wave1", lambda brief, personas: (people[:2], None))
    monkeypatch.setattr(waves, "run_wave", lambda selected, brief, audio_available: (
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
    monkeypatch.setattr(waves, "run_wave", lambda selected, brief, audio_available: ([], []))
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


def test_gate_failure_keeps_core_final_and_null_crossover(monkeypatch):
    people = load_personas()
    selected = people[:config.WAVE1_SIZE]
    monkeypatch.setattr(waves, "choose_wave1", lambda brief, personas: (selected, None))
    monkeypatch.setattr(waves, "run_wave", lambda selected_people, brief, audio_available: (
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
    monkeypatch.setattr(waves, "run_wave", lambda selected_people, brief, audio_available: (
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


def test_gate_pass_with_all_personas_scored_is_complete(monkeypatch):
    people = load_personas()
    selected = people[:config.WAVE1_SIZE]
    monkeypatch.setattr(waves, "choose_wave1", lambda brief, personas: (selected, None))
    monkeypatch.setattr(waves, "run_wave", lambda selected_people, brief, audio_available: (
        [{"persona_id": p["id"], "score": 5, "sound_on": reaction(), "sound_off": reaction()}
         for p in selected_people], []))
    output = waves.simulate({}, personas=people)
    scored = scoring.score_result(output)
    assert output["gate_passed"] is True
    assert output["waves_run"] == [1, 2]
    assert output["personas_intended"] == len(people)
    assert output["personas_scored"] == len(people)
    assert output["complete"] is True
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
