from types import SimpleNamespace

import pytest

from core import llm
from scripts import model_probe


class ProviderError(Exception):
    def __init__(self, status):
        super().__init__(f"provider status {status}")
        self.code = status


@pytest.mark.parametrize(
    ("status", "expected"),
    [(404, "NOT_AVAILABLE"), (503, "OVERLOADED"), (429, "RATE_LIMITED"),
     (401, "AUTH_ERROR"), (403, "AUTH_ERROR")],
)
def test_classifies_provider_statuses(status, expected):
    assert model_probe.classify_error(ProviderError(status))[0] == expected


def test_classifies_timeout_and_other_without_exposing_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "never-print-this-key")
    assert model_probe.classify_error(TimeoutError("request timed out"))[0] == "TIMEOUT"
    status, message = model_probe.classify_error(RuntimeError("unexpected failure"))
    assert status == "OTHER"
    assert message == "unexpected failure"
    assert "never-print-this-key" not in message


@pytest.mark.parametrize("status", [404, 503, 429, 401])
def test_probe_uses_fake_client_one_attempt_and_skips_image_on_text_error(monkeypatch, status):
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            raise ProviderError(status)

    fake_client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(llm, "_get_client", lambda timeout_sec=None: fake_client)
    result = model_probe._run_model("candidate", "frame.jpg")
    expected = model_probe.classify_error(ProviderError(status))[0]
    assert result["text"] == expected
    assert result["image"] == "SKIPPED"
    assert result["status"] == expected
    assert len(calls) == 1
    assert calls[0]["model"] == "candidate"


def test_probe_calls_image_after_text_passes(monkeypatch):
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="OK")

    monkeypatch.setattr(llm, "_get_client", lambda timeout_sec=None: SimpleNamespace(models=FakeModels()))
    monkeypatch.setattr(llm, "Path", lambda _: SimpleNamespace(read_bytes=lambda: b"frame bytes"))
    result = model_probe._run_model("candidate", "frame.jpg")
    assert result["text"] == result["image"] == result["status"] == "OK", result
    assert len(calls) == 2
    assert calls[0]["contents"] == [model_probe.TEXT_PROMPT]
    assert calls[1]["contents"][0] == "Describe this image in a few words."


def test_suggestions_choose_two_fastest_full_passes():
    rows = [
        {"model": "slow", "text": "OK", "image": "OK", "seconds": 4},
        {"model": "bad-image", "text": "OK", "image": "OTHER", "seconds": 0.1},
        {"model": "fast", "text": "OK", "image": "OK", "seconds": 1},
        {"model": "middle", "text": "OK", "image": "OK", "seconds": 2},
    ]
    assert model_probe._suggestions(rows) == ("fast", "middle")
