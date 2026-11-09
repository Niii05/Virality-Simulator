import json

import pytest

from core import config, llm, persona_react, scoring, suggestions, waves
from core.personas import load_personas


def reaction(value=5):
    return {"stop_scrolling": value, "share_likelihood": value, "watch_to_end": value,
            "comment_likelihood": value, "reason": "specific reason", "biggest_issue": "slow opening"}


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
    chosen, error = waves.choose_wave1({}, people)
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
    assert chosen.selection == "fallback"
    assert chosen.raw_ids == []
    assert "malformed JSON" in error


def test_audience_prompt_requests_exact_count_and_logs_counts(monkeypatch, caplog):
    caplog.set_level("INFO", logger="core.waves")
    people = load_personas()
    captured = {}
    monkeypatch.setattr(llm, "ask_json", lambda prompt, **kwargs: captured.update(prompt=prompt, kwargs=kwargs) or {"ids": [people[0]["id"]]})
    chosen, _ = waves.choose_wave1({}, people)
    assert f"exactly {config.WAVE1_SIZE}" in captured["prompt"]
    assert '"ids": ["id", ...]' in captured["prompt"]
    assert len(chosen) == config.WAVE1_SIZE
    assert "returned=1 valid=1 filled=7" in caplog.text


def test_simulate_exposes_wave1_selection_and_raw_ids(monkeypatch):
    people = load_personas()
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: {"ids": [people[0]["id"]]})
    monkeypatch.setattr(waves, "run_wave", lambda selected, brief, audio_available: ([], []))
    result = waves.simulate({}, personas=people)
    assert result["wave1_selection"] == "llm_filled"
    assert result["wave1_raw_ids"] == [people[0]["id"]]


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


def test_partial_persona_coverage_is_marked_incomplete(monkeypatch):
    people = load_personas()
    selected = people[:config.WAVE1_SIZE]
    monkeypatch.setattr(waves, "choose_wave1", lambda brief, personas: (selected, None))
    monkeypatch.setattr(waves, "run_wave", lambda selected_people, brief, audio_available: (
        [{"persona_id": p["id"], "score": 1, "sound_on": reaction(), "sound_off": reaction()}
         for p in selected_people], []))
    output = waves.simulate({}, personas=people)
    scored = scoring.score_result(output)
    assert output["personas_intended"] == len(people)
    assert output["personas_scored"] == config.WAVE1_SIZE
    assert output["personas_failed"] == []
    assert output["complete"] is False
    assert scored["reach_tier"] == "Incomplete - rerun"
    assert scored["final_score"] is None


def test_reaction_cache_skips_successful_condition_on_resume(monkeypatch):
    import uuid
    from core.pipeline import _ReactionCache
    from core.pipeline import ROOT
    cache = _ReactionCache(ROOT / "tmp" / "reaction_cache_test", f"video-model-{uuid.uuid4().hex}")
    calls = []
    monkeypatch.setattr(waves, "react", lambda persona, brief, condition: calls.append(condition) or reaction(7))
    persona = load_personas()[0]
    first = waves._one_persona(persona, {}, True, cache)
    second = waves._one_persona(persona, {}, True, cache)
    assert calls == ["sound_on", "sound_off"]
    assert first == second


def test_suggestions_validate_and_retry_invalid_json(monkeypatch):
    valid = {"suggestions": [{"suggestion": f"Improve {i}", "evidence": "slow opening", "persona_id": "p1", "quote": "slow opening"} for i in range(3)]}
    calls = iter([{"suggestions": []}, valid])
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: next(calls))
    result, errors = suggestions.generate({}, [{"persona_id": "p1", "sound_on": reaction()}])
    assert len(result) == 3 and len(errors) == 1
    assert all(item["evidence"] for item in result)


def test_suggestions_reject_unknown_persona_and_unquoted_evidence(monkeypatch):
    valid = {"suggestions": [{"suggestion": f"Improve {i}", "evidence": "some evidence",
                              "persona_id": "p1", "quote": "not in the persona text"} for i in range(3)]}
    calls = iter([valid, valid])
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: next(calls))
    result, errors = suggestions.generate({}, [{"persona_id": "p1", "sound_on": reaction()}])
    assert result == []
    assert len(errors) == 2 and "quote is not present" in errors[-1]


def test_suggestions_reject_persona_id_absent_from_this_run(monkeypatch):
    payload = {"suggestions": [{"suggestion": "Change it", "evidence": "because", "persona_id": "outsider",
                                "quote": "specific reason"} for _ in range(3)]}
    calls = iter([payload, payload])
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: next(calls))
    result, errors = suggestions.generate({}, [{"persona_id": "p1", "sound_on": reaction()}])
    assert result == []
    assert "must reference a persona" in errors[-1]


def test_suggestions_accept_case_insensitive_quote_from_named_persona(monkeypatch):
    payload = {"suggestions": [{"suggestion": f"Change {i}", "evidence": "reaction reason",
                                "persona_id": "p1", "quote": "SPECIFIC REASON"} for i in range(3)]}
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: payload)
    result, errors = suggestions.generate({}, [{"persona_id": "p1", "sound_on": reaction()}])
    assert len(result) == 3 and not errors
    prompt = suggestions.PROMPT_PATH.read_text(encoding="utf-8")
    assert "do not add numbers, quantities, or techniques" in prompt


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
