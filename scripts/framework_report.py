"""Summarize niche framework usage in saved result JSON files (offline)."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRITERION_IDS = ("hook", "clarity", "retention", "shareability", "sound_and_text", "cta")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for path in sorted((ROOT / "tmp").glob("result*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cited = Counter({criterion: 0 for criterion in CRITERION_IDS})
        for wave in ("wave1", "wave2"):
            for person in data.get(wave, []):
                for condition in ("sound_on", "sound_off"):
                    reaction = person.get(condition) or {}
                    cited.update(item for item in reaction.get("criteria_cited", []) if isinstance(item, str))
        print(f"{path.name}: niche_category={data.get('niche_category')} framework_version={data.get('framework_version')}")
        print("criteria_cited: " + json.dumps({key: cited[key] for key in CRITERION_IDS}, ensure_ascii=False))
        print("suggestion criteria: " + ", ".join(str(item.get("criterion")) for item in data.get("suggestions", [])))


if __name__ == "__main__":
    main()
