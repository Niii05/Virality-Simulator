# Virality Simulator

A prototype pipeline for short-form video analysis. Phase 1 checks Gemini connectivity and extracts sampled frames and a local transcript. The broader audience simulation and virality potential estimate are future work.

## Requirements

- Python 3.10 or newer
- Windows PowerShell and VS Code for the primary development setup
- ffmpeg and ffprobe
- A Gemini API key and a model name available to your account

## Windows setup

Run these commands from the project root in PowerShell:

```powershell
winget install Gyan.FFmpeg
```

Restart VS Code after installing ffmpeg so its terminal refreshes `PATH`. Then create and activate the project environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Copy the example environment file and edit `.env` in a local editor. Do not print, commit, or share the API key:

```powershell
Copy-Item ".env.example" ".env"
```

Set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`. The preflight prints only whether each value is set or empty.

## Preflight and tests

From the project root with the `venv` environment active:

```powershell
python scripts/preflight.py
python -m pytest -q
```

The preflight checks Python and venv status, required imports, ffmpeg tools, required `.env` variable names and non-empty values, Gemini connectivity, and Git ignore behavior. It never prints `.env` values.

## Video extraction

Run from the project root and quote the video path, especially if it contains spaces:

```powershell
python -m core.extract "C:\path\to\your reel.mp4"
```

Frames are written under `tmp\frames`. Whisper uses `WHISPER_MODEL` from `.env`, defaulting to `base`. A video with no audio returns an empty transcript.

Probe the default Gemini model candidates concurrently, or pass a comma-separated `--models` list:

```powershell
.\venv\Scripts\python.exe scripts\model_probe.py
.\venv\Scripts\python.exe scripts\model_probe.py --models gemini-2.5-flash,gemini-2.5-flash-lite
```

The probe uses one text attempt per model, then one image attempt when a frame exists under `tmp\frames`, prints model suggestions without changing `.env`, and saves details to `tmp\model_probe.json`.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| `ffmpeg` or `ffprobe` is missing | Run `winget install Gyan.FFmpeg`, restart VS Code, and rerun preflight. |
| `.env` is missing a required variable | Copy `.env.example` to `.env`, then set non-empty `GEMINI_API_KEY` and `GEMINI_MODEL` values. Never share the key. |
| Gemini reports authentication failure | Check that the key is valid and enabled for the Gemini API. The error output does not include the key. |
| Gemini reports model not found | Set `GEMINI_MODEL` to a model name available in your AI Studio account. |
| Video cannot be probed or decoded | Check that the file is a supported, playable video and that ffmpeg tools are on `PATH`. |
| Whisper is slow on first run | The selected model may need to download. Set `WHISPER_MODEL` to a smaller supported model if appropriate. |

## Current scope

The current code provides a Gemini key check and video extraction. Persona waves, a virality potential estimate, reach tiers, improvement suggestions, and the Streamlit upload interface are not implemented yet.
