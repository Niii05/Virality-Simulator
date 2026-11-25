"""Summarize audience scores as a virality potential estimate."""
from statistics import mean, pstdev
import hashlib
import random

from core import config


def score_interval(scores):
    """Return a reproducible percentile-bootstrap CI for a persona-score mean.

    The interval estimates uncertainty in the mean from the observed persona
    distribution. It is intentionally unavailable for fewer than the configured
    minimum number of scores.
    """
    values = sorted(float(value) for value in scores)
    if len(values) < config.SCORE_INTERVAL_MIN_PERSONAS:
        return None

    digest = hashlib.sha256(
        ",".join(format(value, ".12g") for value in values).encode("ascii")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    sample_count = len(values)
    bootstrap_means = sorted(
        sum(values[rng.randrange(sample_count)] for _ in range(sample_count)) / sample_count
        for _ in range(config.SCORE_INTERVAL_BOOTSTRAP_SAMPLES)
    )
    tail = (1.0 - config.SCORE_INTERVAL_CONFIDENCE) / 2.0

    def percentile(fraction):
        position = fraction * (len(bootstrap_means) - 1)
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(bootstrap_means) - 1)
        weight = position - lower_index
        return bootstrap_means[lower_index] * (1.0 - weight) + bootstrap_means[upper_index] * weight

    point = mean(values)
    lower = min(point, max(0.0, percentile(tail)))
    upper = max(point, min(float(config.SCALE_MAX), percentile(1.0 - tail)))
    return [round(lower, 4), round(upper, 4)]


def _confidence_from_spread(spread):
    if spread is None:
        return None
    if spread <= config.HIGH_CONFIDENCE_MAX_STDDEV:
        return "High"
    if spread <= config.MEDIUM_CONFIDENCE_MAX_STDDEV:
        return "Medium"
    return "Low"


def confidence_for_display(result):
    """Return saved confidence or derive it for display from saved Wave 1 scores.

    This fallback is read-only: it does not add or change fields in ``result``.
    """
    if "confidence" in result and result["confidence"] is not None:
        return result["confidence"]
    wave1 = result.get("wave1")
    values = [
        item["score"] for item in wave1
        if isinstance(item, dict)
        and isinstance(item.get("score"), (int, float))
        and not isinstance(item.get("score"), bool)
    ] if isinstance(wave1, list) else []
    spread = pstdev(values) if values else None
    return _confidence_from_spread(spread)


def score_result(waves):
    wave1_values = [item["score"] for item in waves.get("wave1", [])]
    wave2_values = [item["score"] for item in waves.get("wave2", [])]
    total_values = wave1_values + wave2_values
    core_resonance = mean(wave1_values) if wave1_values else 0.0
    core_score_interval = score_interval(wave1_values)
    core_resonance_spread = pstdev(wave1_values) if wave1_values else None
    confidence = _confidence_from_spread(core_resonance_spread)
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
            "core_resonance_spread": round(core_resonance_spread, 4) if core_resonance_spread is not None else None,
            "confidence": confidence,
            "core_score_interval": core_score_interval,
            "crossover_score": round(crossover_score, 4) if crossover_score is not None else None,
            "final_score": round(final, 4), "reach_tier": tier,
            "tier_note": "insufficient persona coverage; rerun to complete the estimate" if incomplete else ("stays in test audience" if capped else ""),
            "gate_passed": not capped, "personas_intended": intended,
            "personas_scored": scored, "personas_failed": waves.get("personas_failed", []),
            "complete": complete}
