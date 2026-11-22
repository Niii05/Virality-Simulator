# Virality Simulator Handoff

## History naming, thumbnails, and result layout

- `data/history_names.json`: editable legacy result-name map with boundary-aware prefixes.
- `ui_helpers.py`: longest-prefix title lookup, metadata-name priority, and date-based fallbacks for old result filenames.
- `run_helpers.py`: best-effort legacy `testN.mp4` thumbnail extraction with ffmpeg and optional OpenCV fallback.
- `app.py`: reads the name map, shows `Niche not recorded` when absent, generates missing legacy thumbnails outside Demo mode, removes saved-result stage pills, and orders result sections as cards, suggestions, Audience Wall, archetype heat strip, Details, Limitations, then footer.
- `ui_theme.py`: uses the new missing-niche fallback in History tiles while escaping display strings.
- `tests/test_ui_helpers.py`, `tests/test_run_helpers.py`: cover prefix boundaries and priority, fallback titles, absent niche, display-name placeholder, section order, saved-result pills, and mocked thumbnail extraction.

## PowerShell commands

Launch the app:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Run the full project suite:

```powershell
.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
```

## Verification and limits

- Full pytest suite passed: 240 tests (`.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests`).
- Thumbnail extraction was tested with a mocked extractor; ffmpeg/OpenCV were not used on project videos.
- The app and pipeline were not launched. No LLM/API calls were made, and `.env` was not opened.
- No changes were made under `core/`, `prompts/`, or `knowledge_base/`.
