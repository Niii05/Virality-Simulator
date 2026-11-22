"""Print offline Wave 1 audience and score summaries for result JSON files."""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]


def _short_cares(value, limit=120):
    if isinstance(value, list):
        text = "; ".join(str(item) for item in value)
    elif isinstance(value, str):
        text = value
    else:
        text = ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _display(value):
    return "null" if value is None else value


def format_result(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    scores = {item.get("persona_id"): item.get("score")
              for item in result.get("wave1", []) if isinstance(item, dict)}
    profiles = result.get("wave1_personas", [])
    lines = [
        f"{Path(path).name}:",
        f"  wave1_selection: {result.get('wave1_selection')}",
        f"  core_resonance: {_display(result.get('core_resonance'))}",
        f"  crossover_score: {_display(result.get('crossover_score'))}",
        "  Wave 1 personas:",
    ]
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        persona_id = profile.get("id", "unknown")
        lines.append(f"    {persona_id}: score={scores.get(persona_id)}; cares_about={_short_cares(profile.get('cares_about'))}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", help="result JSON file(s); defaults to tmp/result*.json")
    args = parser.parse_args(argv)
    paths = [Path(item) for item in args.results] if args.results else sorted((ROOT / "tmp").glob("result*.json"))
    for path in paths:
        print(format_result(path))


if __name__ == "__main__":
    main()
