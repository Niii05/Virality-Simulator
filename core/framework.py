"""Load and render the research-informed evaluation framework."""
import json
from functools import lru_cache
from pathlib import Path

FRAMEWORK_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "evaluation_framework.json"


@lru_cache(maxsize=1)
def _load():
    try:
        data = json.loads(FRAMEWORK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load evaluation framework {FRAMEWORK_PATH}: {type(error).__name__}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("Evaluation framework root must be a JSON object")
    criteria = data.get("common_criteria")
    sources = data.get("sources")
    enum = data.get("niche_category_enum")
    niches = data.get("niches")
    if not isinstance(criteria, list) or not isinstance(sources, list) or not isinstance(enum, list) or not isinstance(niches, dict):
        raise ValueError("Evaluation framework requires common_criteria, sources, niche_category_enum, and niches")
    criterion_ids = [item.get("id") for item in criteria if isinstance(item, dict)]
    if len(criterion_ids) != len(criteria) or len(set(criterion_ids)) != len(criterion_ids):
        raise ValueError("Evaluation framework common_criteria must have unique ids")
    source_ids = {item.get("id") for item in sources if isinstance(item, dict)}
    if set(niches) != set(enum):
        raise ValueError("Evaluation framework niche keys must exactly match niche_category_enum")
    for item in criteria:
        for source in item.get("sources", []):
            if source not in source_ids:
                raise ValueError(f"Common criterion {item['id']!r} references missing source id {source!r}")
    for niche, content in niches.items():
        rules = content.get("rules") if isinstance(content, dict) else None
        if not isinstance(rules, list):
            raise ValueError(f"Niche {niche!r} must contain rules")
        ids = [rule.get("criterion") for rule in rules if isinstance(rule, dict)]
        if len(ids) != len(rules) or len(ids) != len(criterion_ids) or set(ids) != set(criterion_ids):
            missing = sorted(set(criterion_ids) - set(ids))
            extra = sorted(set(ids) - set(criterion_ids))
            raise ValueError(f"Niche {niche!r} must have exactly one rule per common criterion (missing={missing}, extra={extra})")
        for rule in rules:
            for source in rule.get("sources", []):
                if source not in source_ids:
                    raise ValueError(f"Niche {niche!r} criterion {rule['criterion']!r} references missing source id {source!r}")
    return data


def get_version():
    return _load().get("meta", {}).get("version")


def normalize_niche(value):
    return value if value in _load()["niche_category_enum"] else "general"


def render_niche_detection():
    """Render classification labels and hints directly from the framework file."""
    data = _load()
    lines = ["Choose niche_category by the video's PURPOSE, not merely its topic or keywords."]
    for niche in data["niche_category_enum"]:
        content = data["niches"][niche]
        label = content.get("label", niche)
        hints = content.get("detect_hints", [])
        lines.append(f"- {niche} — {label}: detect_hints={json.dumps(hints, ensure_ascii=False)}")
    lines.extend([
        "Purpose definitions: education = the video teaches the viewer to do something or explains a fact;",
        "product_marketing = a brand or seller promotes its own product or offer. For comedy, beauty, and fitness,",
        "use the purposes indicated by their framework detect_hints; a topic mention alone is not enough.",
        "If the video does not clearly fit one of the five specific niches, answer general. When unsure, general is better than a wrong niche.",
        "A reaction to a software or tech demo is not automatically education or comedy; use general unless the video's purpose clearly matches a specific niche.",
    ])
    return "\n".join(lines)


def render_rules(niche):
    data = _load()
    chosen = normalize_niche(niche)
    selected = data["niches"][chosen]
    lines = [f"NICHE RULES: {selected.get('label', chosen)} ({chosen})"]
    for criterion in data["common_criteria"]:
        lines.append(f"{criterion['id']}: {criterion['question_for_persona']}")
        rule = next(item for item in selected["rules"] if item["criterion"] == criterion["id"])
        lines.append(f"  Niche guidance: {rule['rule']}")
        warning = rule.get("extrapolation_warning")
        if warning:
            lines.append(f"  Extrapolation: {warning[:180]}")
    mistakes = selected.get("common_mistakes", [])
    lines.append("Common mistakes: " + "; ".join(str(item)[:100] for item in mistakes[:5]))
    note = selected.get("evidence_note")
    if note:
        lines.append(f"Evidence note: {note[:220]}")
    return "\n".join(lines)
