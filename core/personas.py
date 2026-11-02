"""Load and validate the audience persona library."""

import json
from pathlib import Path

PERSONAS_PATH = Path(__file__).resolve().parent.parent / "data" / "personas.json"
ARCHETYPES = {"gen_z_scroller", "skeptic", "niche_fan", "casual_viewer", "brand_marketer", "creator_peer"}
REQUIRED = {"id", "archetype", "age", "gender", "region", "interests", "language_pref", "sound_on_probability", "scroll_habit", "base_skip_rate", "cares_about", "pet_peeves"}


def load_personas(path=PERSONAS_PATH):
    try:
        items = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load personas from {path} ({type(error).__name__}: {error})") from error
    if not isinstance(items, list):
        raise ValueError("Persona data must be a JSON array.")
    seen = set()
    counts = {key: 0 for key in ARCHETYPES}
    for index, persona in enumerate(items):
        prefix = f"Persona at index {index}"
        if not isinstance(persona, dict):
            raise ValueError(f"{prefix} must be an object.")
        missing = REQUIRED - persona.keys()
        if missing:
            raise ValueError(f"{prefix} is missing fields: {', '.join(sorted(missing))}.")
        if not isinstance(persona["id"], str) or not persona["id"].strip() or persona["id"] in seen:
            raise ValueError(f"{prefix} has an empty or duplicate id.")
        seen.add(persona["id"])
        if persona["archetype"] not in ARCHETYPES:
            raise ValueError(f"{prefix} has unknown archetype {persona['archetype']!r}.")
        counts[persona["archetype"]] += 1
        if not isinstance(persona["age"], str) or not persona["age"].strip():
            raise ValueError(f"{prefix} age must be a non-empty range string.")
        if not isinstance(persona["interests"], list) or not all(isinstance(x, str) for x in persona["interests"]):
            raise ValueError(f"{prefix} interests must be a list of strings.")
        for field in ("cares_about", "pet_peeves"):
            if not isinstance(persona[field], list) or not persona[field] or not all(isinstance(x, str) and x.strip() for x in persona[field]):
                raise ValueError(f"{prefix} {field} must be a non-empty list of specific strings.")
        if not isinstance(persona["sound_on_probability"], (int, float)) or isinstance(persona["sound_on_probability"], bool) or not 0 <= persona["sound_on_probability"] <= 1:
            raise ValueError(f"{prefix} sound_on_probability must be between 0 and 1.")
        if not isinstance(persona["base_skip_rate"], (int, float)) or not 0.6 <= persona["base_skip_rate"] <= 0.85:
            raise ValueError(f"{prefix} base_skip_rate must be between 0.6 and 0.85.")
        if any(not isinstance(persona[field], str) or not persona[field].strip() for field in ("gender", "region", "language_pref", "scroll_habit")):
            raise ValueError(f"{prefix} text fields must be non-empty strings.")
    if not 20 <= len(items) <= 24:
        raise ValueError(f"Expected 20-24 personas, found {len(items)}.")
    if any(not 3 <= count <= 4 for count in counts.values()):
        raise ValueError("Expected 3-4 variations for each of the six archetypes.")
    return items
