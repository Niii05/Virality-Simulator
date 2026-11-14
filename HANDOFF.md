# Wave Completion Handoff

## Confirmed cause

`tmp\result5.json` has `gate_passed: false`, 8 scored Wave 1 personas, no Wave 2 results, no failed personas, and an empty `errors` list. `core/waves.py` nevertheless set `personas_intended` to all 24 personas. The scoring layer then classified the valid gate-stopped run as incomplete because it compared 8 scored against 24 intended.

## Files changed for this task

- `core/waves.py`: intended count is the Wave 1 size when the gate fails and all personas when it passes; adds `waves_run` as `[1]` or `[1, 2]`.
- `core/scoring.py`: marks a run incomplete only when its intended personas are not all scored. Failed Wave 1 or Wave 2 personas therefore retain the rerun status.
- `core/pipeline.py`: includes `waves_run` in the saved result JSON.
- `tests/test_phase3.py`: covers a complete gate failure, an incomplete Wave 2, and a fully scored two-wave run.
- `tests/test_framework.py`: updates the mocked simulation payload with `waves_run`.

## Run and test (PowerShell)

```powershell
.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

To rerun the prior video and write an updated result:

```powershell
.\venv\Scripts\python.exe -m core.pipeline tmp\<video-file> --out tmp\result5.json
```

## Verification and gaps

- Mock test suite: `113 passed in 31.83s`.
- No real model/API calls were made. `.env` and API keys were not opened or printed.
- The old `tmp\result5.json` was inspected but not edited; rerun the source video to produce a result with the corrected fields and scoring.
