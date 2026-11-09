import pytest

from core import config, llm
_REAL_RATE_LIMIT_CALL = llm._rate_limit_call


@pytest.fixture(autouse=True)
def reset_llm_process_state(monkeypatch):
    llm._next_call_at = 0.0
    llm._model_failures.clear()
    llm._model_circuit_until.clear()
    monkeypatch.setattr(config, "MAX_CALLS_PER_MIN", 600)
    monkeypatch.setattr(llm, "_rate_limit_call", lambda: None)


class ProviderError(Exception):
    def __init__(self, code):
        super().__init__(f"provider status {code}")
        self.code = code


class ConnectTimeout(Exception):
    pass


@pytest.mark.parametrize("status", [429, 500, 503, 599])
def test_rate_limit_and_server_errors_are_retryable(status):
    assert llm._is_retryable(ProviderError(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_other_http_errors_are_not_retryable(status):
    assert not llm._is_retryable(ProviderError(status))


def test_timeouts_are_retryable():
    assert llm._is_retryable(ConnectTimeout("timed out"))
    assert llm._is_retryable(TimeoutError("timed out"))


def test_ask_uses_sdk_timeout_setting(monkeypatch):
    captured = {}
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    monkeypatch.setenv("LLM_TIMEOUT_SEC", "12")
    monkeypatch.setattr(llm, "_client", None)

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm.genai, "Client", fake_client)
    llm._get_client()

    assert captured["http_options"].timeout == 12_000


def test_ask_accepts_per_call_model_and_timeout(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"text": "OK"})()

    def fake_get_client(timeout_sec=None):
        captured["timeout_sec"] = timeout_sec
        return type("Client", (), {"models": FakeModels()})()

    monkeypatch.setattr(llm, "_get_client", fake_get_client)
    assert llm.ask("hello", model="probe-model", timeout_sec=45, retries=1) == "OK"
    assert captured["model"] == "probe-model"
    assert captured["timeout_sec"] == 45


def test_model_call_records_stage_attempt_seconds_and_status(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            return type("Response", (), {"text": "safe result"})()

    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    before = len(llm.CALL_EVENTS)
    with llm.call_stage("brief"):
        assert llm.ask("private prompt", model="test-model", retries=1) == "safe result"
    event = llm.CALL_EVENTS[before]
    assert event["stage"] == "brief"
    assert event["model"] == "test-model"
    assert event["attempt"] == 1
    assert event["seconds"] >= 0
    assert event["status"] == "OK"
    assert "private prompt" not in str(event)


def test_retry_and_fallback_calls_are_marked(monkeypatch):
    class ProviderUnavailable(Exception):
        code = 503

    class FakeModels:
        def generate_content(self, **kwargs):
            if kwargs["model"] == "primary":
                raise ProviderUnavailable("unavailable")
            return type("Response", (), {"text": "ok"})()

    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    before = len(llm.CALL_EVENTS)
    assert llm.ask("x", retries=2) == "ok"
    events = llm.CALL_EVENTS[before:]
    assert [event["reason"] for event in events] == ["initial", "retry", "fallback"]
    assert [event["status"] for event in events] == ["OVERLOADED", "OVERLOADED", "OK"]


def test_ask_timeout_preserves_key_free_original_error(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            raise TimeoutError("internal detail must not be shown")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setenv("LLM_TIMEOUT_SEC", "7")
    monkeypatch.setattr(llm, "_get_client", lambda: FakeClient())

    with pytest.raises(RuntimeError) as exc_info:
        llm.ask("hello", retries=1)

    assert "Gemini request timed out after 7 seconds." in str(exc_info.value)
    assert "TimeoutError: internal detail must not be shown" in str(exc_info.value)
    assert "Check your network/VPN." in str(exc_info.value)


def test_ask_json_strips_json_fence(monkeypatch):
    monkeypatch.setattr(llm, "ask", lambda *args, **kwargs: '```json\n{"score": 7}\n```')

    assert llm.ask_json("Return an object") == {"score": 7}


def test_ask_json_repairs_invalid_response_once(monkeypatch):
    responses = iter(["not json", '{"fixed": true}'])
    calls = []

    def fake_ask(prompt, images=None, json_mode=False, **kwargs):
        calls.append((prompt, images, json_mode))
        return next(responses)

    monkeypatch.setattr(llm, "ask", fake_ask)
    assert llm.ask_json("Return an object") == {"fixed": True}
    assert len(calls) == 2
    assert "not json" in calls[1][0]


def test_ask_json_raises_short_error_after_repair(monkeypatch):
    monkeypatch.setattr(llm, "ask", lambda *args, **kwargs: "still not json")

    with pytest.raises(ValueError, match="still not json"):
        llm.ask_json("Return an object")


def test_rate_limiter_spaces_calls_across_threads(monkeypatch):
    now = [100.0]
    waits = []
    monkeypatch.setattr(config, "MAX_CALLS_PER_MIN", 2)
    monkeypatch.setattr(llm, "_rate_limit_call", _REAL_RATE_LIMIT_CALL)
    monkeypatch.setattr(llm.time, "monotonic", lambda: now[0])
    def sleep(delay):
        waits.append(delay)
        now[0] += delay
    monkeypatch.setattr(llm.time, "sleep", sleep)
    llm._next_call_at = 0.0
    llm._rate_limit_call()
    llm._rate_limit_call()
    assert waits == [30.0]


def test_429_honors_retry_delay_and_logs_full_error(monkeypatch, caplog):
    class Limited(Exception):
        code = 429
        retry_delay = 47
    calls, waits = [], []
    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if len(calls) == 1:
                raise Limited("quota detail " + "x" * 300)
            return type("Response", (), {"text": "ok"})()
    monkeypatch.setenv("GEMINI_MODEL", "primary-delay-test")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    assert llm.ask("x", retries=2) == "ok"
    assert waits == [47]
    assert "x" * 300 in caplog.text


@pytest.mark.parametrize("status", [503, 504])
def test_server_errors_backoff_and_fallback(monkeypatch, status):
    class Unavailable(Exception):
        code = status
    calls, waits = [], []
    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "retry-primary":
                raise Unavailable("busy")
            return type("Response", (), {"text": "fallback"})()
    monkeypatch.setenv("GEMINI_MODEL", "retry-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "retry-backup")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm.random, "uniform", lambda *_: 0)
    assert llm.ask("x") == "fallback"
    assert calls == ["retry-primary"] * 3 + ["retry-backup"]
    assert waits == [2, 4]


def test_daily_quota_stops_without_retry_or_fallback(monkeypatch):
    class DailyLimit(Exception):
        code = 429
    calls = []
    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            raise DailyLimit("Quota exceeded for requests per day")
    monkeypatch.setenv("GEMINI_MODEL", "daily-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "daily-backup")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    with pytest.raises(RuntimeError, match="daily quota is exhausted"):
        llm.ask("x")
    assert calls == ["daily-primary"]


def test_circuit_breaker_routes_directly_to_fallback(monkeypatch):
    class Broken(Exception):
        code = 503
    calls = []
    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "breaker-primary":
                raise Broken("still down")
            return type("Response", (), {"text": "ok"})()
    monkeypatch.setenv("GEMINI_MODEL", "breaker-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "breaker-backup")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    monkeypatch.setattr(config, "LLM_CIRCUIT_BREAKER_FAILURES", 5)
    for _ in range(5):
        assert llm.ask("x", retries=1) == "ok"
    calls.clear()
    assert llm.ask("x", retries=1) == "ok"
    assert calls == ["breaker-backup"]
