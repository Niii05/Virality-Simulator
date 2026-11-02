import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import audio, brief, extract, llm, persona_react, personas
from scripts import try_brief


def test_silent_brief_removes_audio_without_api_call(monkeypatch):
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: pytest.fail("must not call model"))
    source = {"on_screen_text": "hello", "has_captions": True, "transcript": "spoken",
              "transcript_best_effort": "spoken", "music_present": True, "mood": "calm",
              "energy": "low", "voice_clarity": 4, "background_noise": "low",
              "first_3_seconds_sound": "a chime", "sound_dependent": True,
              "mean_volume_db": -22, "max_volume_db": -2, "silence_ratio": 0.1,
              "first_sound_sec": 0.3}
    result = brief.build_silent_brief(source)
    assert result == {"on_screen_text": "hello", "has_captions": True}
    assert result is not source


def test_persona_validation_and_clamping(monkeypatch):
    data = [{"stop_scrolling": 99, "watch_to_end": -2, "share_likelihood": 4.5,
             "comment_likelihood": 2, "reason": "specific", "biggest_issue": "slow"}]
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: data[0])
    persona = personas.load_personas()[0]
    result = persona_react.react(persona, {"on_screen_text": "x"}, "sound_off")
    assert result["stop_scrolling"] == 10
    assert result["watch_to_end"] == 0
    assert result["share_likelihood"] == 4
    with pytest.raises(ValueError, match="condition"):
        persona_react.react(persona, {}, "muted")


def test_audio_stats_parses_filter_stderr(monkeypatch):
    stderr = "[Parsed_volumedetect] mean_volume: -21.4 dB\n[Parsed_volumedetect] max_volume: -2.0 dB\n[silencedetect] silence_start: 0\n[silencedetect] silence_end: 1.5 | silence_duration: 1.5\n[silencedetect] silence_start: 8"
    monkeypatch.setattr(audio.extract, "has_audio_track", lambda _: True)
    monkeypatch.setattr(audio.extract, "get_duration", lambda _: 10)
    monkeypatch.setattr(audio.subprocess, "run", lambda *a, **k: SimpleNamespace(stderr=stderr))
    result = audio.audio_stats("clip.mp4")
    assert result == {"mean_volume_db": -21.4, "max_volume_db": -2.0,
                      "silence_ratio": 0.35, "first_sound_sec": 1.5}


@pytest.mark.parametrize(
    ("stderr", "duration", "expected_first_sound"),
    [
        ("[silencedetect] silence_start: 5.0", 9.7, 0.0),
        ("[silencedetect] silence_start: 0\\n[silencedetect] silence_end: 1.5 | silence_duration: 1.5", 9.7, 1.5),
        ("[silencedetect] silence_start: 8.0", 9.7, 0.0),
    ],
    ids=["no-leading-silence", "leading-1.5-seconds", "trailing-silence-only"],
)
def test_first_sound_is_start_of_non_silence(monkeypatch, stderr, duration, expected_first_sound):
    monkeypatch.setattr(audio.extract, "has_audio_track", lambda _: True)
    monkeypatch.setattr(audio.extract, "get_duration", lambda _: duration)
    monkeypatch.setattr(audio.subprocess, "run", lambda *a, **k: SimpleNamespace(stderr=stderr))
    assert audio.audio_stats("clip.mp4")["first_sound_sec"] == expected_first_sound


def test_prompts_prevent_identifying_people_by_appearance():
    root = Path(__file__).resolve().parent.parent
    rule = "Never identify real people from their face or appearance."
    for prompt_name in ("brief.txt", "audio_brief.txt"):
        prompt = (root / "prompts" / prompt_name).read_text(encoding="utf-8")
        assert rule in prompt
        assert "Name a person only if their name appears" in prompt


def test_persona_schema_requires_specific_cares_and_pet_peeves(monkeypatch):
    entries = personas.load_personas()
    assert all(item["cares_about"] and item["pet_peeves"] for item in entries)
    assert len({tuple(item["cares_about"]) for item in entries}) == len(entries)
    assert len({tuple(item["pet_peeves"]) for item in entries}) == len(entries)
    broken = [dict(item) for item in entries]
    broken[0].pop("cares_about")
    monkeypatch.setattr(personas.json, "loads", lambda _: broken)
    with pytest.raises(ValueError, match="cares_about"):
        personas.load_personas()


def test_persona_prompt_requires_personalized_reactions():
    prompt = (Path(__file__).resolve().parent.parent / "prompts" / "persona.txt").read_text(encoding="utf-8")
    assert "cares_about and pet_peeves" in prompt
    assert "unless that exact concern is represented" in prompt


def test_transcript_source_gemini_does_not_load_whisper(monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_SOURCE", "gemini")
    monkeypatch.setattr(extract, "transcribe", lambda _: pytest.fail("Whisper should not load"))
    assert extract._transcript_for_source("clip.mp4", {
        "available": True, "transcript_best_effort": "Gemini words", "language": "hi"
    }) == ("Gemini words", "hi")


def test_transcript_source_gemini_falls_back_to_whisper(monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_SOURCE", "gemini")
    monkeypatch.setattr(extract, "transcribe", lambda _: ("Whisper words", "en"))
    assert extract._transcript_for_source("clip.mp4", {"available": False}) == ("Whisper words", "en")


def test_transcript_source_whisper_skips_gemini(monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_SOURCE", "whisper")
    monkeypatch.setattr(extract, "transcribe", lambda _: ("Whisper words", "en"))
    monkeypatch.setattr(audio, "audio_brief", lambda _: pytest.fail("Gemini should not run"))
    assert extract._transcript_for_source("clip.mp4") == ("Whisper words", "en")


def test_transcript_source_both_combines_outputs(monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_SOURCE", "both")
    monkeypatch.setattr(extract, "transcribe", lambda _: ("Whisper words", "en"))
    result = extract._transcript_for_source("clip.mp4", {
        "available": True, "transcript_best_effort": "Gemini words", "language": "hi"
    })
    assert result == ("Gemini: Gemini words\nWhisper: Whisper words", "hi")


def test_transcript_source_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_SOURCE", "invalid")
    with pytest.raises(ValueError, match="TRANSCRIPT_SOURCE"):
        extract._transcript_for_source("clip.mp4")


def test_persona_selection_spreads_across_archetypes_and_is_seeded():
    entries = personas.load_personas()
    first = try_brief.select_personas(entries, 5, seed=17)
    second = try_brief.select_personas(entries, 5, seed=17)
    assert first == second
    assert len({item["archetype"] for item in first}) == 5


def test_no_audio_stream_returns_none(monkeypatch):
    monkeypatch.setattr(audio.extract, "check_media_tools", lambda: None)
    monkeypatch.setattr(audio.extract, "has_audio_track", lambda _: False)
    assert audio.extract_audio(Path("silent.mp4")) is None


def test_audio_brief_failure_returns_unavailable(monkeypatch):
    monkeypatch.setattr(audio, "extract_audio", lambda _: "clip.wav")
    monkeypatch.setattr(audio.llm, "ask_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("503")))
    result = audio.audio_brief("clip.mp4")
    assert result["available"] is False
    assert "RuntimeError" in result["reason"]
    assert "503" in result["reason"]


def test_fallback_model_after_retries(monkeypatch):
    class Error(Exception):
        code = 503

    class Models:
        def __init__(self):
            self.models = []

        def generate_content(self, **kwargs):
            self.models.append(kwargs["model"])
            if kwargs["model"] == "main":
                raise Error("unavailable")
            return SimpleNamespace(text="ok")

    models = Models()
    monkeypatch.setenv("GEMINI_MODEL", "main")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "fallback")
    monkeypatch.setattr(llm, "_get_client", lambda: SimpleNamespace(models=models))
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    assert llm.ask("x", retries=2) == "ok"
    assert models.models == ["main", "main", "fallback"]


def test_persona_data_is_24_valid_entries():
    entries = personas.load_personas()
    assert len(entries) == 24
    assert len({item["id"] for item in entries}) == 24
