import json
from pathlib import Path

from core.scoring import confidence_for_display
from demo_history import load_demo_history_items, result_fingerprint


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DEMO_FILES = {
    "result3.json",
    "result4.json",
    "result6b.json",
    "result_ui_4205f564da77423dbc619e87b222c27c.json",
    "result_ui_84746f80427b41d6bcf033e194058539.json",
}


def test_tracked_demo_history_contains_exactly_five_complete_saved_results():
    demo_dir = ROOT / "demo_results"
    assert {path.name for path in demo_dir.glob("*.json")} == EXPECTED_DEMO_FILES

    items = load_demo_history_items(demo_dir)
    assert len(items) == 5
    assert all(item["is_demo"] is True for item in items)
    assert all(item["display_name"].startswith("Saved demo analysis · ") for item in items)

    for item in items:
        source = json.loads((ROOT / "tmp" / item["result_file"]).read_text(encoding="utf-8"))
        saved = json.loads((demo_dir / item["result_file"]).read_text(encoding="utf-8"))
        assert saved == source
        assert item["result"] == source
        assert item["result"].get("complete") is True
        assert item["result"].get("final_score") is not None
        assert item["result"].get("reach_tier")
        assert len(item["result"].get("suggestions", [])) == 3
        assert item["confidence"] == confidence_for_display(item["result"])


def test_demo_history_omits_exact_payload_duplicates_from_local_history():
    demo_dir = ROOT / "demo_results"
    existing = json.loads((demo_dir / "result3.json").read_text(encoding="utf-8"))
    items = load_demo_history_items(demo_dir, existing_results=[existing])

    assert len(items) == 4
    assert all(result_fingerprint(item["result"]) != result_fingerprint(existing) for item in items)


def test_demo_history_deduplicates_equivalent_demo_payloads():
    demo_dir = ROOT / "demo_results"
    items = load_demo_history_items(demo_dir)
    duplicate = items[0]["result"]
    assert len(load_demo_history_items(demo_dir, existing_results=[duplicate, duplicate])) == 4

