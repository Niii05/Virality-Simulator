import json
from pathlib import Path

import pytest

from core import brief, framework, llm, persona_react, pipeline, suggestions


SOURCE = Path(__file__).resolve().parents[1] / "knowledge_base" / "evaluation_framework.json"


def framework_data():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def test_framework_loads_and_renders_only_selected_niche():
    framework._load.cache_clear()
    assert framework.get_version() == framework_data()["meta"]["version"]
    block = framework.render_rules("education")
    assert "Niche guidance: Open with the question" in block
    assert "Joke needs context" not in block


@pytest.mark.parametrize("mutation,match", [
    (lambda data: data["common_criteria"][0]["sources"].append("missing_source"), "missing source id"),
    (lambda data: data["niches"]["education"]["rules"].pop(), "exactly one rule per common criterion"),
])
def test_framework_loader_reports_invalid_references_and_rules(tmp_path, monkeypatch, mutation, match):
    data = framework_data()
    mutation(data)
    target = tmp_path / "framework.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(framework, "FRAMEWORK_PATH", target)
    framework._load.cache_clear()
    with pytest.raises(ValueError, match=match):
        framework.get_version()
    framework._load.cache_clear()


def test_unknown_niche_normalizes_to_general():
    assert framework.normalize_niche("not-a-niche") == "general"


def test_brief_prompt_gets_framework_niche_hints_and_saves_reason(monkeypatch):
    framework._load.cache_clear()
    data = framework_data()
    captured = {}
    # Return an unknown value to exercise the general fallback as well.
    monkeypatch.setattr(llm, "ask_json", lambda prompt, **kwargs: (captured.update(prompt=prompt, kwargs=kwargs) or {
        "niche_category": "tech_news", "niche_reason": "The clip reacts to a software demo without teaching a step."}))
    result = brief.build_brief({"frames": [], "transcript": "software demo"})
    for niche, entry in data["niches"].items():
        assert entry["label"] in captured["prompt"]
        assert json.dumps(entry["detect_hints"], ensure_ascii=False) in captured["prompt"]
    assert "When unsure, general is better than a wrong niche" in captured["prompt"]
    assert result["niche_category"] == "general"
    assert result["niche_reason"] == "The clip reacts to a software demo without teaching a step."
    assert captured["kwargs"]["retries"] == captured["kwargs"]["retry_cap"] == llm.BRIEF_RETRY_CAP == 5
    assert captured["kwargs"]["timeout_sec"] == 120
    assert framework.get_version() == data["meta"]["version"]


def test_brief_missing_reason_gets_general_fallback(monkeypatch):
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: {"niche_category": "education"})
    result = brief.build_brief({"frames": []})
    assert result["niche_category"] == "education"
    assert result["niche_reason"] == "The video did not clearly fit a specific niche based on its purpose."


def test_creator_context_is_separated_in_brief_prompt_and_saved(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm, "ask_json", lambda prompt, **kwargs: (captured.update(prompt=prompt) or {}))
    context = "Raw Mango iPhone promotion featuring a well-known actress."
    result = brief.build_brief({"frames": []}, creator_context=context)
    assert "Creator-provided context (this is what a viewer would know from the caption, the account and recognisable people; it is NOT visible in the video)" in captured["prompt"]
    assert "Treat this context as true for purpose, product, audience and niche" in captured["prompt"]
    assert context in captured["prompt"]
    assert result["creator_context"] == context


def test_audience_persona_prompt_receives_creator_context(monkeypatch):
    from core import waves
    captured = {}
    monkeypatch.setattr(llm, "ask_json", lambda prompt, **kwargs: (captured.update(prompt=prompt) or {}))
    waves._dynamic_prompt({"creator_context": "iPhone campaign by Raw Mango", "topic_niche": "fashion"})
    assert '"creator_context": "iPhone campaign by Raw Mango"' in captured["prompt"]
    assert "never describe, refer to, imply, or hint at the supplied reel" in captured["prompt"].lower()


def test_context_hash_separates_all_cache_layers(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    assert pipeline._key(video, "model") != pipeline._key(video, "model", context="iPhone ad")
    from core.pipeline import _ReactionCache, _Wave1PersonaCache
    persona = {"id": "p1"}
    assert _ReactionCache(tmp_path, "video")._path(persona, "sound_on") != _ReactionCache(tmp_path, "video", "ctx")._path(persona, "sound_on")
    brief_value = {"creator_context": "iPhone ad"}
    assert _Wave1PersonaCache(tmp_path, "video", "prompt")._path(brief_value)[0] != _Wave1PersonaCache(tmp_path, "video", "prompt", "ctx")._path(brief_value)[0]


def test_reaction_cache_is_separated_by_niche(tmp_path):
    from core.pipeline import _ReactionCache
    persona = {"id": "p1"}
    assert _ReactionCache(tmp_path, "video", niche="education")._path(persona, "sound_on") != \
        _ReactionCache(tmp_path, "video", niche="comedy")._path(persona, "sound_on")


def _mock_pipeline_run(monkeypatch):
    monkeypatch.setattr(pipeline, "audio", type("Audio", (), {"audio_brief": staticmethod(lambda *_: {"available": False})}))
    monkeypatch.setattr(pipeline, "extract_all", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline.brief, "build_full_brief", lambda *args, **kwargs: {
        "niche_category": "general", "niche_reason": "detected by mock"})
    captured = {}
    def simulate(*args, **kwargs):
        captured["reaction_cache"] = kwargs.get("reaction_cache")
        return {"wave1": [], "wave2": [], "wave1_selection": "fallback", "wave1_raw_ids": [],
                "wave1_personas": [], "dimension_means": {}, "errors": [], "gate_passed": False, "waves_run": [1]}
    monkeypatch.setattr(pipeline, "simulate", simulate)
    monkeypatch.setattr(pipeline, "score_result", lambda *_: {})
    monkeypatch.setattr(pipeline, "generate", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(pipeline.llm, "CALL_EVENTS", [])
    return captured


def test_pipeline_valid_niche_override(monkeypatch, tmp_path):
    captured = _mock_pipeline_run(monkeypatch)
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    output = pipeline.run(video, tmp_path / "result.json", use_cache=True, niche="education")
    assert output["niche_category"] == "education"
    assert output["niche_reason"] == "user override"
    assert captured["reaction_cache"].niche == "education"


def test_pipeline_rejects_invalid_niche_before_model_access(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.llm, "_model_name", lambda: pytest.fail("model must not be accessed"))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    with pytest.raises(ValueError, match="Invalid niche.*Allowed values"):
        pipeline.run(video, tmp_path / "result.json", niche="technology")


def test_context_truncation_warns_and_result_fields(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(pipeline, "audio", type("Audio", (), {"audio_brief": staticmethod(lambda *_: {"available": False})}))
    monkeypatch.setattr(pipeline, "extract_all", lambda *args, **kwargs: {})
    captured = {}
    monkeypatch.setattr(pipeline.brief, "build_full_brief", lambda *args, **kwargs: (captured.update(kwargs) or {"niche_category": "general"}))
    monkeypatch.setattr(pipeline, "simulate", lambda *args, **kwargs: {"wave1": [], "wave2": [], "wave1_selection": "fallback", "wave1_raw_ids": [], "wave1_personas": [], "dimension_means": {}, "errors": [], "gate_passed": False, "waves_run": [1]})
    monkeypatch.setattr(pipeline, "score_result", lambda *_: {})
    monkeypatch.setattr(pipeline, "generate", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(pipeline.llm, "CALL_EVENTS", [])
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    output = pipeline.run(video, tmp_path / "result.json", use_cache=False, context="x" * 601)
    assert captured["creator_context"] == "x" * 600
    assert "truncating" in caplog.text
    assert output["user_context"] == "x" * 600 and output["context_used"] is True


def test_pipeline_cli_context_options_and_absent_context(monkeypatch, tmp_path):
    import sys
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    context_file = tmp_path / "context.txt"
    context_file.write_text("UTF-8 café", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs))
    monkeypatch.setattr(sys, "argv", ["pipeline", str(video), "--context-file", str(context_file)])
    pipeline.main()
    assert captured["kwargs"]["context"] == "UTF-8 café"
    monkeypatch.setattr(sys, "argv", ["pipeline", str(video), "--context", "caption details"])
    pipeline.main()
    assert captured["kwargs"]["context"] == "caption details"
    assert pipeline._normalize_context(None) is None


def test_persona_drops_invalid_criteria_cited(monkeypatch):
    response = {"stop_scrolling": 5, "watch_to_end": 5, "share_likelihood": 5,
                "comment_likelihood": 5, "reason": "reason", "biggest_issue": "issue",
                "criteria_cited": ["hook", "not-real", 3, "hook"]}
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: response.copy())
    result = persona_react.react({"cares_about": []}, {"niche_category": "comedy"}, "sound_on")
    assert result["criteria_cited"] == ["hook"]


def test_suggestion_criterion_is_validated(monkeypatch):
    data = {"suggestions": [{"suggestion": "Change", "evidence": "detail", "persona_id": "p1",
                             "quote": "reason", "criterion": "invalid"} for _ in range(3)]}
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: data)
    result, errors = suggestions.generate({}, [{"persona_id": "p1", "sound_on": {"reason": "reason"}}])
    assert result == []
    assert "valid criterion id" in errors[-1]


@pytest.mark.parametrize("relative", ["prompts/persona.txt", "prompts/brief.txt", "prompts/suggestions.txt",
                                       "prompts/json_repair.txt", "data/personas.json",
                                       "knowledge_base/evaluation_framework.json"])
def test_cache_key_changes_for_every_hashed_input(tmp_path, relative):
    root = tmp_path / "root"
    (root / "prompts").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "knowledge_base").mkdir()
    for name in ("persona.txt", "brief.txt", "suggestions.txt", "json_repair.txt"):
        (root / "prompts" / name).write_text(name, encoding="utf-8")
    (root / "data" / "personas.json").write_text("[]", encoding="utf-8")
    (root / "knowledge_base" / "evaluation_framework.json").write_text("{}", encoding="utf-8")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    before = pipeline._key(video, "model", root=root)
    changed = root / relative
    changed.write_text(changed.read_text(encoding="utf-8") + " changed", encoding="utf-8")
    assert pipeline._key(video, "model", root=root) != before


def test_result_fields_are_added_by_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "audio", type("Audio", (), {"audio_brief": staticmethod(lambda *_: {"available": False})}))
    monkeypatch.setattr(pipeline, "extract_all", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline.brief, "build_full_brief", lambda *args, **kwargs: {
        "niche_category": "comedy", "niche_reason": "The clip is a short skit.",
        "framework_version": framework.get_version()})
    monkeypatch.setattr(pipeline, "simulate", lambda *args, **kwargs: {
        "wave1": [], "wave2": [], "wave1_selection": "fallback", "wave1_raw_ids": [],
        "wave1_personas": [{"id": "genz_dyn_1", "context": "mock audience member"}],
        "dimension_means": {}, "errors": [], "gate_passed": False, "waves_run": [1]})
    monkeypatch.setattr(pipeline, "score_result", lambda *_: {"label": "Virality potential estimate"})
    monkeypatch.setattr(pipeline, "generate", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(pipeline.llm, "CALL_EVENTS", [])
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    output = pipeline.run(video, tmp_path / "result.json", use_cache=False)
    assert output["niche_category"] == "comedy"
    assert output["niche_reason"] == "The clip is a short skit."
    assert output["wave1_personas"] == [{"id": "genz_dyn_1", "context": "mock audience member"}]
    assert output["framework_version"] == framework.get_version()
    assert output["user_context"] is None
    assert output["context_used"] is False


def test_vision_brief_failure_writes_incomplete_result_without_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(pipeline.audio, "audio_brief", lambda *_: {"available": False})
    monkeypatch.setattr(pipeline, "extract_all", lambda *args, **kwargs: {"frames": []})
    monkeypatch.setattr(pipeline.brief, "build_full_brief",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("brief unavailable")))
    monkeypatch.setattr(pipeline, "simulate", lambda *args, **kwargs: pytest.fail("must skip waves"))
    monkeypatch.setattr(pipeline, "score_result", lambda *args, **kwargs: pytest.fail("must skip scoring"))
    monkeypatch.setattr(pipeline, "generate", lambda *args, **kwargs: pytest.fail("must skip suggestions"))
    monkeypatch.setattr(pipeline.llm, "CALL_EVENTS", [])
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    output = pipeline.run(video, tmp_path / "result.json")

    assert output["complete"] is False
    assert output["failure_stage"] == "vision_brief"
    assert output["errors"] == [
        {"stage": "audio", "error": "Audio unavailable"},
        {"stage": "vision_brief", "error": "RuntimeError: brief unavailable"},
    ]
    assert output["final_score"] is None
    assert output["core_resonance"] is None
    assert output["per_persona_scores"] == {}
    assert output["wave1"] == output["wave2"] == []
    assert output["brief"] is None
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["errors"] == output["errors"]


def test_vision_failure_rerun_reuses_cached_successful_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    audio_calls = []
    monkeypatch.setattr(pipeline.audio, "audio_brief", lambda *_: audio_calls.append(1) or {
        "available": True, "transcript": "cached audio"})
    monkeypatch.setattr(pipeline, "extract_all", lambda *args, **kwargs: {"frames": []})
    vision_calls = []
    def build_brief(*args, **kwargs):
        vision_calls.append(1)
        if len(vision_calls) == 1:
            raise RuntimeError("brief unavailable")
        return {"niche_category": "general", "niche_reason": "mock"}
    monkeypatch.setattr(pipeline.brief, "build_full_brief", build_brief)
    monkeypatch.setattr(pipeline, "simulate", lambda *args, **kwargs: {
        "wave1": [], "wave2": [], "wave1_selection": "fallback", "wave1_raw_ids": [],
        "wave1_personas": [], "dimension_means": {}, "errors": [], "gate_passed": False,
        "waves_run": [1]})
    monkeypatch.setattr(pipeline, "score_result", lambda *_: {"complete": True, "final_score": 0})
    monkeypatch.setattr(pipeline, "generate", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(pipeline.llm, "CALL_EVENTS", [])
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    first = pipeline.run(video, tmp_path / "first.json")
    second = pipeline.run(video, tmp_path / "second.json")

    assert first["failure_stage"] == "vision_brief"
    assert second["complete"] is True
    assert len(audio_calls) == 1
    assert len(vision_calls) == 2


def test_pipeline_marks_missing_required_personas_incomplete(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(pipeline.audio, "audio_brief", lambda *_: {"available": False})
    monkeypatch.setattr(pipeline, "extract_all", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline.brief, "build_full_brief", lambda *args, **kwargs: {
        "niche_category": "general", "niche_reason": "mock"})
    wave1 = [{"persona_id": f"persona-{index}", "score": 8,
              "sound_on": {"stop_scrolling": 8, "share_likelihood": 8,
                           "watch_to_end": 8, "comment_likelihood": 8,
                           "reason": "mock", "biggest_issue": "mock"}}
             for index in range(4)]
    missing_errors = [{"persona_id": f"persona-{index}", "error": "RuntimeError: overloaded"}
                      for index in range(4, 8)]
    monkeypatch.setattr(pipeline, "simulate", lambda *args, **kwargs: {
        "wave1": wave1, "wave2": [], "wave1_personas": [{"id": f"persona-{i}"} for i in range(8)],
        "wave1_selection": "fixed_fallback", "wave1_raw_ids": [],
        "dimension_means": {}, "gate_passed": False, "waves_run": [1],
        "personas_intended": 8, "personas_scored": 4, "personas_failed": [f"persona-{i}" for i in range(4, 8)],
        "complete": False, "errors": missing_errors})
    monkeypatch.setattr(pipeline, "generate", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(pipeline.llm, "CALL_EVENTS", [])
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    output = pipeline.run(video, tmp_path / "result.json", use_cache=False)

    assert output["complete"] is False
    assert output["personas_scored"] == 4
    assert output["personas_intended"] == 8
    assert output["reach_tier"] == "Incomplete - rerun"
    assert {item["persona_id"] for item in output["errors"] if "persona_id" in item} == {
        f"persona-{index}" for index in range(4, 8)
    }
