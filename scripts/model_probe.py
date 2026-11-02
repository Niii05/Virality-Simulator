"""Probe Gemini models concurrently with one text and optional image call each."""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import llm

DEFAULT_MODELS = [
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
    "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-flash-lite",
]
TEXT_PROMPT = "Reply with only OK."


def classify_error(error):
    """Map a provider exception (or llm wrapper) to a short safe status/message."""
    current = error
    while current is not None:
        status = llm._status_code(current)
        if status == 404:
            return "NOT_AVAILABLE", "model not found"
        if status == 503:
            return "OVERLOADED", "service overloaded"
        if status == 429:
            return "RATE_LIMITED", "rate limited"
        if status in (401, 403):
            return "AUTH_ERROR", "authentication or access denied"
        if llm._is_timeout(current):
            return "TIMEOUT", "request timed out"
        current = current.__cause__
    message = llm._safe_error_message(error).strip() or type(error).__name__
    return "OTHER", message[:100]


def _run_model(model, frame):
    started = time.perf_counter()
    result = {"model": model, "text": None, "image": "SKIPPED",
              "seconds": None, "status": "OTHER", "message": ""}
    try:
        llm.ask(TEXT_PROMPT, model=model, timeout_sec=45, retries=1)
        result["text"] = "OK"
    except Exception as error:
        status, message = classify_error(error)
        result.update(text=status, status=status, message=message)
        result["seconds"] = round(time.perf_counter() - started, 3)
        return result

    if frame is not None:
        try:
            llm.ask("Describe this image in a few words.", images=[frame],
                    model=model, timeout_sec=45, retries=1)
            result["image"] = "OK"
        except Exception as error:
            status, message = classify_error(error)
            result.update(image=status, status=status, message=message)
    else:
        result["message"] = "no frame found; image probe skipped"
    if result["image"] == "OK" or frame is None:
        result["status"] = "OK"
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def probe_models(models, frame=None):
    """Probe all requested models with at most five concurrent workers."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, max(1, len(models)))) as pool:
        futures = {pool.submit(_run_model, model, frame): model for model in models}
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    order = {model: index for index, model in enumerate(models)}
    return sorted(results, key=lambda row: order[row["model"]])


def _first_frame():
    frames = sorted((PROJECT_ROOT / "tmp" / "frames").glob("*.jpg"))
    return frames[0] if frames else None


def _suggestions(results):
    passing = sorted(
        (row for row in results if row["text"] == "OK" and row["image"] == "OK"),
        key=lambda row: (row["seconds"], row["model"]),
    )
    return (passing[0]["model"] if passing else None,
            passing[1]["model"] if len(passing) > 1 else None)


def _print_table(results):
    headers = ("model", "text", "image", "seconds", "status")
    rows = [[row["model"], row["text"], row["image"], f'{row["seconds"]:.3f}', row["status"]]
            for row in results]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="comma-separated model names")
    args = parser.parse_args(argv)
    models = [name.strip() for name in args.models.split(",") if name.strip()] if args.models else DEFAULT_MODELS[:]
    if not models:
        parser.error("--models must contain at least one model name")
    frame = _first_frame()
    results = probe_models(models, frame)
    _print_table(results)
    first, second = _suggestions(results)
    print(f"Suggested GEMINI_MODEL: {first or 'None (no model passed both text and image)'}")
    print(f"Suggested GEMINI_FALLBACK_MODEL: {second or 'None (no second model passed both)'}")
    output_path = PROJECT_ROOT / "tmp" / "model_probe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {output_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
