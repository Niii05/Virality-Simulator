# Virality Simulator Handoff

## Hero estimate presentation

- The explanatory score note now has a high-contrast, slightly muted treatment on the gradient.
- Confidence is positioned at the bottom-right of the hero card in a dark translucent pill for legibility.
- Range text, score logic, bootstrap, confidence logic, and other card content are unchanged.

## Files changed for this task

- `ui_theme.py`: estimate note and bottom-right confidence styling and markup.
- `tests/test_ui_helpers.py`: focused layout and contrast-hook assertions.
- `HANDOFF.md`: refreshed.

## Verification

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_ui_helpers.py -q
```

Result: 98 passed in 0.91s. No API calls or commits were made. The full suite was not run.
