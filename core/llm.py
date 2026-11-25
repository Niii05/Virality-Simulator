"""Provider wrapper for text and image model requests."""

import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from core import config

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_client = None
_timeout_clients = {}
_client_lock = threading.Lock()
_rate_limit_lock = threading.Lock()
_next_call_at = 0.0
_circuit_lock = threading.Lock()
_model_failures = {}
_model_circuit_until = {}
_unavailable_models = set()
API_CALL_COUNT = 0
CALL_EVENTS = []
_CALL_STAGE = ContextVar("llm_call_stage", default="unspecified")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPAIR_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "json_repair.txt"
BRIEF_RETRY_CAP = 5
TIMEOUT_SAME_MODEL_RETRIES = 1


def _get_client(timeout_sec=None):
    global _client
    if timeout_sec is not None:
        timeout_sec = float(timeout_sec)
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be a positive number.")
        with _client_lock:
            if timeout_sec not in _timeout_clients:
                key = os.getenv("GEMINI_API_KEY")
                if not key:
                    raise RuntimeError("GEMINI_API_KEY is missing. Set it in the local .env file.")
                _timeout_clients[timeout_sec] = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(timeout=int(timeout_sec * 1000)),
                )
            return _timeout_clients[timeout_sec]
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Set it in the local .env file."
            )
        timeout_milliseconds = int(_timeout_seconds() * 1000)
        _client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=timeout_milliseconds),
        )
    return _client


def _model_name():
    return os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def _canonical_model_name(model):
    """Normalize common model resource-name aliases for fallback deduplication."""
    return str(model).strip().rstrip("/").rsplit("/", 1)[-1].casefold()


def _timeout_seconds():
    raw_value = os.getenv("LLM_TIMEOUT_SEC", "60")
    try:
        seconds = float(raw_value)
    except ValueError:
        raise RuntimeError("LLM_TIMEOUT_SEC must be a positive number.") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("LLM_TIMEOUT_SEC must be a positive number.")
    return seconds


def _supports_minimal_thinking(model):
    model_name = str(model).rsplit("/", 1)[-1].lower()
    return re.match(r"^gemini-3(?:[.-])", model_name) is not None


def _status_code(error):
    """Return an HTTP status code when exposed by a provider exception."""
    for value in (
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _is_timeout(error):
    name = type(error).__name__.lower()
    message = str(error).lower()
    return (
        "timeout" in name
        or isinstance(error, TimeoutError)
        or "timeout" in message
        or "timed out" in message
    )


def _is_retryable(error):
    status = _status_code(error)
    return _is_timeout(error) or status == 429 or (status is not None and 500 <= status <= 599)


def _redact_error(error):
    message = str(error).replace("\r", " ").replace("\n", " ")
    key = os.getenv("GEMINI_API_KEY")
    if key:
        message = message.replace(key, "[redacted]")
    return re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted]", message)


def _daily_quota_exhausted(error):
    message = str(error).lower()
    return any(token in message for token in ("per_day", "per day", "daily quota", "quota for today", "requests_per_day"))


def _retry_after_seconds(error):
    candidates = [getattr(error, "retry_delay", None), getattr(error, "retry_after", None)]
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    try:
        candidates.append(headers.get("Retry-After"))
    except AttributeError:
        pass
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "total_seconds"):
            return max(0.0, float(candidate.total_seconds()))
        if hasattr(candidate, "seconds"):
            try:
                return max(0.0, float(candidate.seconds))
            except (TypeError, ValueError):
                pass
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?", str(candidate), re.I)
        if match:
            return max(0.0, float(match.group(1)))
    match = re.search(r"(?:retry\s+in|retry_delay[\"']?\s*[:=]\s*[\"']?)([0-9]+(?:\.[0-9]+)?)\s*s", str(error), re.I)
    return float(match.group(1)) if match else None


def _rate_limit_call():
    global _next_call_at
    interval = 60.0 / max(1, int(config.MAX_CALLS_PER_MIN))
    now = time.monotonic()
    with _rate_limit_lock:
        scheduled = max(now, _next_call_at)
        _next_call_at = scheduled + interval
    delay = scheduled - now
    if delay > 0:
        time.sleep(delay)


def _circuit_open(model):
    with _circuit_lock:
        until = _model_circuit_until.get(model, 0.0)
        if until <= time.monotonic():
            _model_circuit_until.pop(model, None)
            return False
        return True


def _note_model_failure(model):
    with _circuit_lock:
        failures = _model_failures.get(model, 0) + 1
        _model_failures[model] = failures
        if failures >= config.LLM_CIRCUIT_BREAKER_FAILURES:
            _model_circuit_until[model] = time.monotonic() + config.LLM_CIRCUIT_BREAKER_SECONDS


def _note_model_success(model):
    with _circuit_lock:
        _model_failures[model] = 0
        _model_circuit_until.pop(model, None)


@contextmanager
def call_stage(stage):
    """Attach a safe stage label to every provider attempt made in this block."""
    token = _CALL_STAGE.set(str(stage))
    try:
        yield
    finally:
        _CALL_STAGE.reset(token)


def _call_status(error):
    status = _status_code(error)
    if status == 404:
        return "NOT_AVAILABLE"
    if status == 503:
        return "OVERLOADED"
    if status == 429:
        if _daily_quota_exhausted(error):
            return "DAILY_QUOTA"
        return "RATE_LIMITED"
    if status in (401, 403):
        return "AUTH_ERROR"
    if _is_timeout(error):
        return "TIMEOUT"
    return "OTHER"


def _record_call(model, attempt, seconds, status, reason):
    event = {"stage": _CALL_STAGE.get(), "model": model, "attempt": attempt,
             "seconds": round(seconds, 3), "status": status, "reason": reason}
    CALL_EVENTS.append(event)
    logging.getLogger(__name__).info(
        "Model call stage=%s model=%s attempt=%s seconds=%.3f status=%s",
        event["stage"], model, attempt, event["seconds"], status,
    )


def record_model_call(model, attempt, seconds, status, reason="initial"):
    """Record a local model call in the same safe run summary as provider calls."""
    _record_call(model, attempt, seconds, status, reason)


def _safe_error_message(error):
    """Build a key-free, single-line error suitable for console output."""
    message = str(error).replace("\r", " ").replace("\n", " ")
    key = os.getenv("GEMINI_API_KEY")
    if key:
        message = message.replace(key, "[redacted]")
    message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted]", message)
    return message


def _raise_provider_error(error, model=None):
    status = _status_code(error)
    detail = f"{type(error).__name__}: {_redact_error(error)}"
    if status == 401:
        raise RuntimeError(f"Gemini authentication failed (401; {detail}). Check GEMINI_API_KEY.") from error
    if status == 403:
        raise RuntimeError(f"Gemini access was denied (403; {detail}). Check API key permissions.") from error
    if status == 404:
        raise RuntimeError(f"Gemini model {model!r} was not found (404; {detail}).") from error
    raise RuntimeError(
        f"Gemini request failed ({type(error).__name__}: {_redact_error(error)})"
    ) from error


def ask(prompt, images=None, json_mode=False, temperature=0.7, retries=3,
        audio_path=None, model=None, timeout_sec=None, response_json_schema=None,
        retry_cap=3):
    """Send a prompt and optional JPEG images to the configured model."""
    parts = [prompt]
    for path in images or []:
        data = Path(path).read_bytes()
        parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
    if audio_path is not None:
        parts.append(
            types.Part.from_bytes(data=Path(audio_path).read_bytes(), mime_type="audio/wav")
        )

    use_timeout_client = timeout_sec is not None
    if timeout_sec is not None:
        timeout_seconds = float(timeout_sec)
    else:
        timeout_seconds = _timeout_seconds()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_sec must be a positive number.")
    attempts = min(max(1, int(retries)), max(1, int(retry_cap)))
    explicit_model = model is not None
    if explicit_model:
        models = [str(model)]
    else:
        models = [_model_name()]
        fallback = (os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
                    or config.FALLBACK_MODEL.strip())
        if fallback and all(_canonical_model_name(fallback) != _canonical_model_name(item)
                            for item in models):
            models.append(fallback)
    if models and _circuit_open(models[0]):
        logging.getLogger(__name__).warning("Model circuit open model=%s; routing directly to fallback", models[0])
        models = models[1:]
        if not models:
            raise RuntimeError("Gemini model circuit is open and no fallback model is configured.")
    model_index = 0
    while model_index < len(models):
        model = models[model_index]
        if not explicit_model and model in _unavailable_models:
            logging.getLogger(__name__).warning("Skipping unavailable Gemini model=%s", model)
            if not explicit_model and model_index > 0 and models[0] not in _unavailable_models:
                models.append(models[0])
            model_index += 1
            continue
        if _circuit_open(model):
            model_index += 1
            continue
        thinking_config = None
        if (_CALL_STAGE.get() == "vision_brief"
                and _canonical_model_name(model) != "gemini-3.8-flash"
                and _supports_minimal_thinking(model)):
            thinking_config = types.ThinkingConfig(thinking_level="MINIMAL")
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
            response_json_schema=response_json_schema,
            thinking_config=thinking_config,
        )
        attempt = 0
        while True:
            call_started = time.perf_counter()
            try:
                global API_CALL_COUNT
                _rate_limit_call()
                API_CALL_COUNT += 1
                client = (_get_client(timeout_sec=timeout_seconds)
                          if use_timeout_client else _get_client())
                response = client.models.generate_content(
                    model=model, contents=parts, config=generation_config
                )
                reason = "fallback" if model_index else ("retry" if attempt else "initial")
                _record_call(model, attempt + 1, time.perf_counter() - call_started, "OK", reason)
                _note_model_success(model)
                return response.text
            except Exception as error:
                reason = "fallback" if model_index else ("retry" if attempt else "initial")
                call_status = _call_status(error)
                status = _status_code(error)
                _record_call(model, attempt + 1, time.perf_counter() - call_started,
                             call_status, reason)
                logging.getLogger(__name__).error(
                    "Model request error stage=%s model=%s attempt=%s http_status=%s classification=%s error=%s: %s",
                    _CALL_STAGE.get(), model, attempt + 1, status, call_status,
                    type(error).__name__, _redact_error(error),
                )
                _note_model_failure(model)
                model_circuit_open = _circuit_open(model)
                if status == 404:
                    _unavailable_models.add(model)
                    logging.getLogger(__name__).warning("Skipping unavailable Gemini model=%s", model)
                    if not explicit_model and model_index == 0 and len(models) > 1:
                        break
                    if not explicit_model and model_index > 0 and models[0] not in _unavailable_models:
                        models.append(models[0])
                        break
                    _raise_provider_error(error, model)
                if status in (401, 403):
                    _raise_provider_error(error, model)
                if status == 429 and _daily_quota_exhausted(error):
                    raise RuntimeError(f"Gemini daily quota is exhausted; stopping retries. Original error: {type(error).__name__}: {_redact_error(error)}") from error
                if status == 503 and model_index == 0 and len(models) > 1:
                    break
                if model_circuit_open:
                    if model_index + 1 < len(models):
                        break
                    _raise_provider_error(error, model)
                no_fallback = len(models) == 1 and not explicit_model
                using_fallback_model = (
                    model_index > 0
                    and _canonical_model_name(model) != _canonical_model_name(models[0])
                )
                long_overload_retry = status in (503, 504) and (no_fallback or using_fallback_model)
                if long_overload_retry and attempt >= max(0, int(config.RETRY_503_MAX)):
                    raise RuntimeError(
                        "Gemini is experiencing temporary Google service overload "
                        f"(HTTP {status}) after {attempt} retries. Rerunning later is safe "
                        "because successful steps are cached. Original error: "
                        f"{type(error).__name__}: {_redact_error(error)}"
                    ) from error
                if _is_timeout(error):
                    same_model_retries = min(TIMEOUT_SAME_MODEL_RETRIES, attempts - 1)
                    if attempt >= same_model_retries:
                        if model_index + 1 < len(models):
                            break
                        raise RuntimeError(
                            f"Gemini request timed out after {timeout_seconds:g} seconds. "
                            f"Original error: {type(error).__name__}: {_safe_error_message(error)}. "
                            "Check your network/VPN."
                        ) from error
                if not _is_retryable(error):
                    _raise_provider_error(error, model)
                if not long_overload_retry and attempt == attempts - 1:
                    if model_index + 1 < len(models):
                        break
                    _raise_provider_error(error, model)
                retry_after = _retry_after_seconds(error) if status == 429 else None
                if retry_after is not None:
                    delay = min(retry_after, config.LLM_RETRY_MAX_SECONDS)
                elif long_overload_retry:
                    backoff = min(config.RETRY_503_BASE_SECONDS * (2 ** attempt),
                                  config.RETRY_503_MAX_WAIT_SECONDS)
                    jitter = random.uniform(0, min(5, backoff * 0.1))
                    delay = min(config.RETRY_503_MAX_WAIT_SECONDS, backoff + jitter)
                    retry_number = attempt + 1
                    print(f"Gemini busy ({status}), waiting {math.ceil(delay)}s, "
                          f"attempt {retry_number} of {config.RETRY_503_MAX}",
                          file=sys.stderr, flush=True)
                else:
                    backoff = min(config.LLM_RETRY_INITIAL_SECONDS * (2 ** attempt),
                                  config.LLM_RETRY_MAX_SECONDS)
                    delay = min(config.LLM_RETRY_MAX_SECONDS,
                                backoff + random.uniform(0, config.LLM_RETRY_JITTER_SECONDS))
                time.sleep(delay)
                attempt += 1
        model_index += 1
    raise RuntimeError("No available Gemini model remains after retries or circuit-breaker checks.")


def _strip_json_fence(raw):
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def _raw_snippet(raw):
    key = os.getenv("GEMINI_API_KEY")
    snippet = str(raw).replace("\r", " ").replace("\n", " ")
    if key:
        snippet = snippet.replace(key, "[redacted]")
    snippet = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted]", snippet)
    return snippet[: config.LLM_ERROR_SNIPPET_LENGTH]


def ask_json(prompt, images=None, audio_path=None, temperature=0.7, response_json_schema=None,
             retries=3, retry_cap=3, timeout_sec=None):
    """Return a JSON object, allowing one model repair attempt if needed."""
    raw = ask(prompt, images=images, audio_path=audio_path, json_mode=True,
              temperature=temperature, response_json_schema=response_json_schema,
              retries=retries, retry_cap=retry_cap, timeout_sec=timeout_sec)
    for attempt in range(2):
        try:
            result = json.loads(_strip_json_fence(raw))
            if not isinstance(result, dict):
                raise ValueError("JSON response must be an object")
            return result
        except (json.JSONDecodeError, ValueError):
            if attempt == 0:
                template = _REPAIR_PROMPT_PATH.read_text(encoding="utf-8")
                raw = ask(
                    template.replace("{{PROMPT}}", str(prompt)).replace(
                        "{{RESPONSE}}", str(raw)
                    ),
                    images=images,
                    audio_path=audio_path,
                    json_mode=True,
                    temperature=temperature,
                    response_json_schema=response_json_schema,
                    retries=retries,
                    retry_cap=retry_cap,
                    timeout_sec=timeout_sec,
                )
    raise ValueError(f"Could not parse JSON response: {_raw_snippet(raw)}")
