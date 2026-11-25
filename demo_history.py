"""Load the tracked, saved analyses used in the founder demo history."""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from core.scoring import confidence_for_display
from ui_helpers import format_history_date, history_display_name


DEMO_RESULT_CREATED_AT = {
    "result3.json": "2026-11-09T21:53:55+05:30",
    "result4.json": "2026-11-15T00:51:54+05:30",
    "result6b.json": "2026-11-15T23:06:13+05:30",
    "result_ui_4205f564da77423dbc619e87b222c27c.json": "2026-11-17T00:47:53+05:30",
    "result_ui_84746f80427b41d6bcf033e194058539.json": "2026-11-23T01:02:11+05:30",
}


def result_fingerprint(result):
    """Return a stable fingerprint for a saved JSON result payload."""
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_successful_demo_result(result):
    score = result.get("final_score")
    if (result.get("complete") is not True or isinstance(score, bool)
            or not isinstance(score, (int, float)) or not math.isfinite(score)
            or not 0 <= score <= 10):
        return False

    reactions = [
        persona
        for wave_name in ("wave1", "wave2")
        for persona in result.get(wave_name, [])
        if isinstance(persona, dict)
    ]
    if not reactions or not all(
        any(isinstance(persona.get(condition), dict) and persona[condition]
            for condition in ("sound_on", "sound_off"))
        for persona in reactions
    ):
        return False

    suggestions = result.get("suggestions")
    return (
        isinstance(suggestions, list)
        and len(suggestions) == 3
        and all(isinstance(item, dict) and any(value for value in item.values())
                for item in suggestions)
    )


def load_demo_history_items(demo_results_dir, existing_results=()):
    """Load complete tracked demo results, omitting payloads already in local history."""
    known_fingerprints = {
        result_fingerprint(result)
        for result in existing_results
        if isinstance(result, dict)
    }
    items = []
    for result_path in sorted(Path(demo_results_dir).glob("*.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict) or not _is_successful_demo_result(result):
            continue

        fingerprint = result_fingerprint(result)
        if fingerprint in known_fingerprints:
            continue
        known_fingerprints.add(fingerprint)

        created_at = DEMO_RESULT_CREATED_AT.get(result_path.name)
        if created_at:
            try:
                sort_timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                created_at = None
        if not created_at:
            sort_timestamp = result_path.stat().st_mtime
            created_at = datetime.fromtimestamp(sort_timestamp).astimezone().isoformat(timespec="seconds")

        brief = result.get("brief") if isinstance(result.get("brief"), dict) else {}
        display_name = history_display_name(result_path.name, modified_at=sort_timestamp)
        items.append({
            "result_file": result_path.name,
            "history_id": f"demo:{result_path.name}",
            "result_path": str(result_path.resolve()),
            "result": result,
            "meta_id": "demo_" + hashlib.sha256(result_path.name.encode("utf-8")).hexdigest()[:16],
            "metadata": {},
            "display_name": f"Saved demo analysis · {display_name}",
            "created_at": created_at,
            "date_label": format_history_date(created_at),
            "sort_timestamp": sort_timestamp,
            "niche": result.get("niche_category") or brief.get("niche_category") or "Niche not recorded",
            "tier": result.get("reach_tier"),
            "core_score": result.get("core_resonance") if result.get("core_resonance") is not None
            else result.get("final_score"),
            "confidence": confidence_for_display(result),
            "incomplete": False,
            "thumbnail_path": None,
            "is_demo": True,
        })
    return items
