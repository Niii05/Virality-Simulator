"""Summarize audience scores as a virality potential estimate."""
from core import config


def score_result(waves):
    wave1_values = [item["score"] for item in waves.get("wave1", [])]
    wave2_values = [item["score"] for item in waves.get("wave2", [])]
    total_values = wave1_values + wave2_values
    core_resonance = sum(wave1_values) / len(wave1_values) if wave1_values else 0.0
    waves_run = waves.get("waves_run", [1, 2] if waves.get("gate_passed") else [1])
    crossover_score = (sum(wave2_values) / len(wave2_values)
                       if 2 in waves_run and wave2_values else None)
    final = core_resonance
    tier = "Low"
    for name, threshold in sorted(config.REACH_TIERS.items(), key=lambda pair: pair[1]):
        if final >= threshold:
            tier = name
    intended = waves.get("personas_intended", len(total_values))
    scored = waves.get("personas_scored", len(total_values))
    complete = waves.get("complete", scored == intended)
    incomplete = not complete
    capped = not waves.get("gate_passed", False)
    if capped:
        tier = "Low"
    if incomplete:
        tier = "Incomplete - rerun"
    return {"label": "Virality potential estimate", "core_resonance": round(core_resonance, 4),
            "crossover_score": round(crossover_score, 4) if crossover_score is not None else None,
            "final_score": round(final, 4), "reach_tier": tier,
            "tier_note": "insufficient persona coverage; rerun to complete the estimate" if incomplete else ("stays in test audience" if capped else ""),
            "gate_passed": not capped, "personas_intended": intended,
            "personas_scored": scored, "personas_failed": waves.get("personas_failed", []),
            "complete": complete}
