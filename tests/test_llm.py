import pytest

from core import config, llm
_REAL_RATE_LIMIT_CALL = llm._rate_limit_call


@pytest.fixture(autouse=True)
def reset_llm_process_state(monkeypatch):
    llm._next_call_at = 0.0
    llm._model_failures.clear()
    llm._model_circuit_until.clear()
    llm._unavailable_models.clear()
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


def test_ask_json_forwards_explicit_timeout_and_other_calls_use_global(monkeypatch):
    timeouts = []

    class FakeModels:
        def generate_content(self, **kwargs):
            return type("Response", (), {"text": "OK"})()

    def fake_get_client(timeout_sec=None):
        timeouts.append(timeout_sec)
        return type("Client", (), {"models": FakeModels()})()

    monkeypatch.setenv("LLM_TIMEOUT_SEC", "30")
    monkeypatch.setattr(llm, "_get_client", fake_get_client)

    assert llm.ask("vision", timeout_sec=120, retries=1) == "OK"
    with llm.call_stage("audio_brief"):
        assert llm.ask("audio", retries=1) == "OK"

    assert timeouts == [120.0, None]


def test_ask_json_passes_timeout_to_ask(monkeypatch):
    captured = {}
    def fake_ask(prompt, **kwargs):
        captured.update(kwargs)
        return '{"ok": true}'
    monkeypatch.setattr(llm, "ask", fake_ask)
    assert llm.ask_json("vision", timeout_sec=120) == {"ok": True}
    assert captured["timeout_sec"] == 120


def test_timeout_retries_primary_once_then_uses_fallback(monkeypatch):
    calls, waits = [], []
    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "primary":
                raise TimeoutError("request timed out")
            return type("Response", (), {"text": "fallback result"})()
    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup")
    monkeypatch.setattr(llm, "_get_client", lambda timeout_sec=None: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm.random, "uniform", lambda *_: 0)

    assert llm.ask("vision", timeout_sec=120, retries=3, retry_cap=3) == "fallback result"
    assert calls == ["primary", "primary", "backup"]
    assert waits == [2]
    assert [event["model"] for event in llm.CALL_EVENTS[-3:]] == ["primary", "primary", "backup"]


def test_timeout_attempts_are_bounded_per_configured_model(monkeypatch, caplog):
    calls = []
    secret_marker = "unit-test-secret-marker"
    class FakeModels:
        def generate_content(self, **kwargs):
            model = kwargs["model"]
            calls.append(model)
            raise TimeoutError(f"request timed out; detail={secret_marker}")
    monkeypatch.setenv("GEMINI_API_KEY", secret_marker)
    monkeypatch.setenv("GEMINI_MODEL", "primary-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "fallback-model")
    monkeypatch.setattr(llm, "_get_client", lambda timeout_sec=None: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    monkeypatch.setattr(llm.random, "uniform", lambda *_: 0)

    with pytest.raises(RuntimeError, match="Gemini request timed out"):
        llm.ask("vision", timeout_sec=120, retries=5, retry_cap=5)

    assert calls == ["primary-model", "primary-model", "fallback-model", "fallback-model"]
    assert [(event["model"], event["attempt"], event["status"])
            for event in llm.CALL_EVENTS[-4:]] == [
                ("primary-model", 1, "TIMEOUT"), ("primary-model", 2, "TIMEOUT"),
                ("fallback-model", 1, "TIMEOUT"), ("fallback-model", 2, "TIMEOUT"),
            ]
    assert "TimeoutError: request timed out; detail=[redacted]" in caplog.text
    assert secret_marker not in caplog.text


def test_vision_brief_thinking_is_omitted_for_unsupported_models(monkeypatch):
    configs = []
    class FakeModels:
        def generate_content(self, **kwargs):
            configs.append(kwargs["config"])
            return type("Response", (), {"text": "OK"})()
    monkeypatch.setattr(llm, "_get_client", lambda timeout_sec=None: type("Client", (), {"models": FakeModels()})())

    with llm.call_stage("vision_brief"):
        assert llm.ask("vision", model="gemini-2.5-flash", retries=1) == "OK"
        assert llm.ask("vision", model="gemini-3.6-flash", retries=1) == "OK"

    assert configs[0].thinking_config is None
    assert configs[1].thinking_config.thinking_level == "MINIMAL"


def test_gemini_38_vision_brief_omits_thinking_config(monkeypatch):
    configs = []

    class FakeModels:
        def generate_content(self, **kwargs):
            configs.append(kwargs["config"])
            return type("Response", (), {"text": "OK"})()

    monkeypatch.setattr(llm, "_get_client", lambda timeout_sec=None: type("Client", (), {"models": FakeModels()})())

    with llm.call_stage("vision_brief"):
        assert llm.ask("vision", model="gemini-3.8-flash", retries=1) == "OK"

    assert configs[0].thinking_config is None


def test_non_vision_stage_keeps_thinking_config_omitted(monkeypatch):
    configs = []

    class FakeModels:
        def generate_content(self, **kwargs):
            configs.append(kwargs["config"])
            return type("Response", (), {"text": "OK"})()

    monkeypatch.setattr(llm, "_get_client", lambda timeout_sec=None: type("Client", (), {"models": FakeModels()})())

    with llm.call_stage("audio_brief"):
        assert llm.ask("audio", model="gemini-3.6-flash", retries=1) == "OK"

    assert configs[0].thinking_config is None


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
    assert [event["reason"] for event in events] == ["initial", "fallback"]
    assert [event["status"] for event in events] == ["OVERLOADED", "OK"]


def test_fallback_404_is_skipped_and_primary_succeeds(monkeypatch, caplog):
    calls = []
    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "primary":
                if calls.count("primary") == 1:
                    raise ProviderError(503)
                return type("Response", (), {"text": "primary result"})()
            raise ProviderError(404)
    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    assert llm.ask("x", retries=2) == "primary result"
    assert calls == ["primary", "backup", "primary"]
    assert "Skipping unavailable Gemini model=backup" in caplog.text
    assert llm.ask("x", retries=2) == "primary result"
    assert calls[-1] == "primary"


def test_fallback_404_returns_to_primary_with_normal_retry_cap(monkeypatch):
    class Busy(Exception):
        code = 503
    calls, waits = [], []
    class FakeModels:
        def generate_content(self, **kwargs):
            model = kwargs["model"]
            calls.append(model)
            if model == "backup":
                raise ProviderError(404)
            if calls.count("primary") == 1:
                raise Busy("busy before fallback")
            raise Busy("busy after fallback")
    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm.random, "uniform", lambda *_: 0)
    with pytest.raises(RuntimeError, match="Gemini request failed"):
        llm.ask("x", retries=3)
    assert calls == ["primary", "backup", "primary", "primary", "primary"]
    assert waits == [2, 4]


def test_primary_404_uses_working_fallback(monkeypatch):
    calls = []
    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "primary":
                raise ProviderError(404)
            return type("Response", (), {"text": "fallback result"})()
    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    assert llm.ask("x") == "fallback result"
    assert calls == ["primary", "backup"]
    assert [event["model"] for event in llm.CALL_EVENTS[-2:]] == ["primary", "backup"]


def test_fallback_503_uses_jittered_backoff_and_stops_at_attempt_cap(monkeypatch):
    class Busy(Exception):
        code = 503
    calls, waits = [], []

    class FakeModels:
        def generate_content(self, **kwargs):
            model = kwargs["model"]
            calls.append(model)
            raise Busy(f"{model} overloaded")

    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup")
    monkeypatch.setattr(llm, "_get_client", lambda timeout_sec=None: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm.random, "uniform", lambda *_: 0)

    with pytest.raises(RuntimeError, match="temporary Google service overload.*backup overloaded"):
        llm.ask("x", retries=3, retry_cap=3)

    assert calls == ["primary", "backup", "backup", "backup", "backup"]
    assert waits == [5, 10, 20]
    assert [event["attempt"] for event in llm.CALL_EVENTS[-5:]] == [1, 1, 2, 3, 4]


def test_model_alias_is_not_used_as_its_own_fallback(monkeypatch):
    class Busy(Exception):
        code = 503
    calls, waits = [], []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            raise Busy("provider overloaded")

    monkeypatch.setenv("GEMINI_MODEL", "models/gemini-3.6-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
    monkeypatch.setattr(llm, "_get_client", lambda timeout_sec=None: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm.random, "uniform", lambda *_: 0)

    with pytest.raises(RuntimeError, match="temporary Google service overload"):
        llm.ask("x")

    assert calls == ["models/gemini-3.6-flash"] * 4
    assert waits == [5, 10, 20]
    assert all(event["reason"] != "fallback" for event in llm.CALL_EVENTS[-4:])


def test_both_models_404_raise_with_actual_failing_model(monkeypatch):
    calls = []
    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            raise ProviderError(404)
    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    with pytest.raises(RuntimeError, match="model 'backup'.*404"):
        llm.ask("x")
    assert calls == ["primary", "backup"]


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
    assert waits == [20]
    assert "x" * 300 in caplog.text
    assert llm.CALL_EVENTS[-2]["status"] == "RATE_LIMITED"


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
    if status == 503:
        assert calls == ["retry-primary", "retry-backup"]
        assert waits == []
    else:
        assert calls == ["retry-primary"] * 3 + ["retry-backup"]
        assert waits == [2, 4]


def test_config_fallback_model_is_used_after_first_primary_503(monkeypatch):
    class Busy(Exception):
        code = 503

    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "configured-primary":
                raise Busy("busy")
            return type("Response", (), {"text": "fallback"})()

    monkeypatch.setenv("GEMINI_MODEL", "configured-primary")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    monkeypatch.setattr(config, "FALLBACK_MODEL", "configured-fallback")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", lambda _: pytest.fail("first 503 should switch to fallback"))
    assert llm.ask("x") == "fallback"
    assert calls == ["configured-primary", "configured-fallback"]


def test_503_without_fallback_retries_configured_count_and_reports_waits(monkeypatch, capsys):
    class Busy(Exception):
        code = 503
    calls, waits = [], []
    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if len(calls) <= 3:
                raise Busy("temporary busy")
            return type("Response", (), {"text": "recovered"})()
    monkeypatch.setenv("GEMINI_MODEL", "busy-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm.random, "uniform", lambda low, high: high / 2)
    assert llm.ask("x") == "recovered"
    assert len(calls) == 4  # initial request plus three retries
    assert waits == [5.25, 10.5, 20]
    assert capsys.readouterr().err.splitlines() == [
        "Gemini busy (503), waiting 6s, attempt 1 of 3",
        "Gemini busy (503), waiting 11s, attempt 2 of 3",
        "Gemini busy (503), waiting 20s, attempt 3 of 3",
    ]


def test_504_without_fallback_caps_wait_at_20_seconds(monkeypatch):
    class Busy(Exception):
        code = 504
    waits = []
    class FakeModels:
        def generate_content(self, **kwargs):
            raise Busy("temporary busy")
    monkeypatch.setenv("GEMINI_MODEL", "busy-cap-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm.random, "uniform", lambda *_: 0)
    with pytest.raises(RuntimeError, match="temporary Google service overload"):
        llm.ask("x")
    assert waits == [5, 10, 20]
    assert all(delay <= 20 for delay in waits)


def test_503_exhaustion_explains_safe_rerun_and_original_error(monkeypatch):
    class Busy(Exception):
        code = 503
    calls = []
    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            raise Busy("high demand")
    monkeypatch.setenv("GEMINI_MODEL", "busy-final-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "")
    monkeypatch.setattr(config, "RETRY_503_MAX", 2)
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    monkeypatch.setattr(llm.random, "uniform", lambda *_: 0)
    with pytest.raises(RuntimeError) as caught:
        llm.ask("x")
    message = str(caught.value)
    assert "temporary Google service overload" in message
    assert "Rerunning later is safe" in message
    assert "successful steps are cached" in message
    assert "Busy: high demand" in message
    assert calls == ["busy-final-primary"] * 3


def test_503_fallback_attempts_stop_at_configured_retry_cap(monkeypatch):
    class Busy(Exception):
        code = 503

    calls, waits = [], []

    class FakeModels:
        def generate_content(self, **kwargs):
            model = kwargs["model"]
            calls.append(model)
            raise Busy(f"{model} high demand")

    monkeypatch.setenv("GEMINI_MODEL", "overloaded-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "overloaded-fallback")
    monkeypatch.setattr(config, "RETRY_503_MAX", 2)
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm.random, "uniform", lambda *_: 0)

    with pytest.raises(RuntimeError, match="temporary Google service overload.*after 2 retries"):
        llm.ask("x", retries=5, retry_cap=5)

    assert calls == [
        "overloaded-primary",
        "overloaded-fallback", "overloaded-fallback", "overloaded-fallback",
    ]
    assert waits == [5, 10]
    assert [event["attempt"] for event in llm.CALL_EVENTS[-4:]] == [1, 1, 2, 3]
    assert [event["model"] for event in llm.CALL_EVENTS[-4:]] == [
        "overloaded-primary", "overloaded-fallback",
        "overloaded-fallback", "overloaded-fallback",
    ]


def test_primary_circuit_open_routes_503_to_fallback(monkeypatch):
    class Busy(Exception):
        code = 503

    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            model = kwargs["model"]
            calls.append(model)
            if model == "primary":
                raise Busy("primary overloaded")
            return type("Response", (), {"text": "fallback result"})()

    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "fallback")
    monkeypatch.setattr(config, "LLM_CIRCUIT_BREAKER_FAILURES", 1)
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", lambda _: pytest.fail("primary must route directly to fallback"))

    assert llm.ask("x", retries=5, retry_cap=5) == "fallback result"
    assert calls == ["primary", "fallback"]
    assert llm._circuit_open("primary")


def test_fallback_circuit_open_stops_retries_immediately(monkeypatch):
    class Busy(Exception):
        code = 503

    calls, waits = [], []

    class FakeModels:
        def generate_content(self, **kwargs):
            model = kwargs["model"]
            calls.append(model)
            raise Busy(f"{model} overloaded")

    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "fallback")
    monkeypatch.setattr(config, "LLM_CIRCUIT_BREAKER_FAILURES", 1)
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm.random, "uniform", lambda *_: 0)

    with pytest.raises(RuntimeError, match="Gemini request failed.*fallback overloaded"):
        llm.ask("x", retries=5, retry_cap=5)

    assert calls == ["primary", "fallback"]
    assert waits == []
    assert llm._circuit_open("primary")
    assert llm._circuit_open("fallback")


def test_daily_quota_stops_without_retry_or_fallback_and_logs_safely(monkeypatch, caplog):
    class DailyLimit(Exception):
        code = 429
    calls = []
    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            raise DailyLimit("Quota exceeded for requests per day; key=unit-test-secret")
    monkeypatch.setenv("GEMINI_MODEL", "daily-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "daily-backup")
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-secret")
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"models": FakeModels()})())
    with pytest.raises(RuntimeError, match="daily quota is exhausted"):
        with llm.call_stage("vision_brief"):
            llm.ask("x")
    assert calls == ["daily-primary"]
    assert llm.CALL_EVENTS[-1]["status"] == "DAILY_QUOTA"
    assert ("stage=vision_brief model=daily-primary attempt=1 http_status=429 classification=DAILY_QUOTA "
            "error=DailyLimit: Quota exceeded for requests per day; key=[redacted]") in caplog.text
    assert "unit-test-secret" not in caplog.text


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
