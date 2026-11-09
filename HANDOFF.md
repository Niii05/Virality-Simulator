# Reliability and Resume Handoff

## Changes

- Added a shared, thread-safe provider request limiter. `MAX_CALLS_PER_MIN` defaults to 10.
- Provider retries honor a 429 retry delay from the error, capped at 60 seconds. 503/504 and other retryable server errors use exponential backoff starting at 2 seconds with jitter, capped at 60 seconds, with at most three attempts per model. Daily quota errors stop immediately with a clear message. Full key-redacted provider error text is logged.
- Added a per-model circuit breaker: five consecutive failures opens the circuit for 60 seconds; calls route directly to the configured fallback while it is open.
- Results include `personas_intended`, `personas_scored`, `personas_failed`, and `complete`. Coverage below `MIN_COMPLETE_FRACTION` (default 0.8) sets `reach_tier` to `Incomplete - rerun` and `final_score` to null.
- Successful persona reactions are cached immediately by video hash, configured model, persona ID, and sound condition. Incomplete cached runs resume by default when caching is enabled; `--retry-failed` enables this behavior explicitly, and `--no-retry-failed` disables it.
- Added mock tests for rate spacing, retry delays, full error logging, daily quota stop, 503/504 backoff/fallback, circuit breaking, incomplete coverage, and reaction cache reuse.

## Files changed

- `core/config.py`
- `core/llm.py`
- `core/waves.py`
- `core/scoring.py`
- `core/pipeline.py`
- `tests/test_llm.py`
- `tests/test_phase3.py`
- `HANDOFF.md`

## Verification and limits

- Mock-only test suite: `93 passed`.
- No real API calls or media runs were performed. `.env` and API keys were not opened or printed.
- Live Gemini retry metadata shapes and real quota behavior remain unverified.
- `--no-cache` does **not** write to the cache. It still writes the requested result JSON to `--out`.

## Commands

From the project root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\venv\Scripts\python.exe -m core.pipeline "D:\path\to\video.mp4" --out tmp\result.json --retry-failed
```
