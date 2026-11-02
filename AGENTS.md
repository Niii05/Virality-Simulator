@'
# Project: Virality Simulator (side prototype for an AI marketing platform)

Goal: user uploads a short video (reel/short). Pipeline: extract frames + audio + transcript, build a text content brief (full and silent versions), simulate audience waves with AI personas, output a "virality potential estimate" (never call it a prediction or a guarantee), a reach tier, and 3 improvement suggestions.

Stack: Python 3.10+, Streamlit, ffmpeg/ffprobe via subprocess, faster-whisper (local), Gemini via google-genai. Primary dev environment: Windows + VS Code. Use pathlib; give PowerShell commands.

Architecture rules:
- core/llm.py is the ONLY module that talks to a model provider. Everything else calls ask() / ask_json().
- Prompts live in prompts/*.txt. Persona data in data/personas.json. No long prompts inline.
- Tunable numbers (gate threshold, score weights, persona counts, concurrency, sound mix) live in core/config.py.
- Free tier: at most 5 concurrent model calls; retry only on 429/5xx/timeouts; never retry auth or model-not-found errors. Model names are pinned in .env (GEMINI_MODEL, GEMINI_FALLBACK_MODEL); never use "-latest" aliases.
- Transcripts are approximate, especially for Hindi/Hinglish. Treat them as such in prompts.
- Sound handling: each persona has sound_on_probability; run each persona with a full brief and a silent brief and combine as p*on + (1-p)*off. Do not hardcode a sound-on/off ratio.

Security rules:
- Never read, print, log or commit .env or any API key. Never hardcode keys or put them in error messages.
- Do not add a dependency without saying which one and why.

Working rules:
- Do only the phase you are asked to do. Do not build ahead.
- Keep existing public function names unless told otherwise.
- Tests never call the real API; use a fake llm.
- If a test you wrote fails: diagnose, fix the test or the code, rerun (max 3 attempts). Never weaken, delete or skip a test just to pass. If the cause is the environment, avoid that dependency or explain why it cannot be fixed. Skip fixing only when the prompt says "report only, do not fix".
- Never hide the root cause behind a generic error: include the original exception type and message (key-free).
- Always use the project venv (.\venv), never global Python.
- When finished, overwrite HANDOFF.md: files changed, exact Windows commands to run and test, what you could NOT run or verify, known gaps.
'@ | Set-Content -Path AGENTS.md -Encoding ascii