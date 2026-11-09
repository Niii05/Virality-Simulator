"""Summarize audience scores as a virality potential estimate."""
from core import config


def score_result(waves):
    values = [item["score"] for key in ("wave1", "wave2") for item in waves.get(key, [])]
    final = sum(values) / len(values) if values else 0.0
    tier = "Low"
    for name, threshold in sorted(config.REACH_TIERS.items(), key=lambda pair: pair[1]):
        if final >= threshold:
            tier = name
    intended = waves.get("personas_intended", len(values))
    scored = waves.get("personas_scored", len(values))
    complete = waves.get("complete", scored == intended)
    incomplete = intended > 0 and scored / intended < config.MIN_COMPLETE_FRACTION
    capped = not waves.get("gate_passed", False)
    if capped:
        tier = "Low"
    if incomplete:
        tier = "Incomplete - rerun"
    return {"label": "Virality potential estimate", "final_score": None if incomplete else round(final, 4), "reach_tier": tier,
            "tier_note": "insufficient persona coverage; rerun to complete the estimate" if incomplete else ("stays in test audience" if capped else ""),
            "gate_passed": not capped, "personas_intended": intended,
            "personas_scored": scored, "personas_failed": waves.get("personas_failed", []),
            "complete": complete}
