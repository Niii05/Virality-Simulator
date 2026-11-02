# Targeted Fixes Handoff

## Completed

- Fixed `audio_stats` first-sound semantics: only a silence interval starting at the beginning can delay `first_sound_sec`; trailing/interior silence leaves it at `0.0`.
- Added privacy guidance to both brief prompts: people are described generically unless a name is present in supplied text or audio.
- Added individually tailored `cares_about` and `pet_peeves` to all 24 personas and made them required non-empty string lists in schema validation. Updated persona behavior prompt to use them and avoid unsupported generic critiques.
- Added seeded, archetype-spread persona sampling to `scripts/try_brief.py` with `--personas N` and `--seed`.
- Recorded each Gemini request and local Whisper transcription with stage, model, attempt, elapsed seconds, status, and safe retry/fallback reason. The demo prints the model-call summary table and extra-call explanation.
- Added `TRANSCRIPT_SOURCE` (`gemini` default, `whisper`, or `both`). Gemini mode uses the audio brief transcript and invokes Whisper only when the audio brief is unavailable. `extract_all` retains its return dictionary shape.

## Files changed

- `core/audio.py`, `core/config.py`, `core/extract.py`, `core/llm.py`
- `data/personas.json`, `core/personas.py`
- `prompts/brief.txt`, `prompts/audio_brief.txt`, `prompts/persona.txt`
- `scripts/try_brief.py`
- `tests/test_phase2.py`, `tests/test_llm.py`
- `HANDOFF.md`

## Commands

From the project root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\venv\Scripts\python.exe scripts\try_brief.py "D:\path\to\video.mp4" --personas 6 --seed 23
```

## Verification limits

- Tests use fake model calls. Live Gemini and real-media behavior were not exercised.
- `AGENTS.md` was read. No `.env` file or key was opened or printed.
