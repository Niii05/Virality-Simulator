"""Generate and validate three evidence-linked improvements."""
import json
from collections import Counter
from pathlib import Path

from core import llm

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "suggestions.txt"


def _valid(data, persona_evidence):
    items = data.get("suggestions") if isinstance(data, dict) else None
    if not isinstance(items, list) or len(items) != 3:
        raise ValueError("suggestions must contain exactly three items")
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("suggestion"), str) or not item["suggestion"].strip():
            raise ValueError("each suggestion needs non-empty suggestion text")
        if not isinstance(item.get("evidence"), str) or not item["evidence"].strip():
            raise ValueError("each suggestion needs non-empty evidence")
        persona_id = item.get("persona_id")
        quote = item.get("quote")
        if not isinstance(persona_id, str) or persona_id not in persona_evidence:
            raise ValueError("each suggestion must reference a persona from this run")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("each suggestion needs a short persona quote")
        valid_sources = persona_evidence[persona_id]
        if not any(quote.casefold() in source.casefold() for source in valid_sources):
            raise ValueError(f"quote is not present in persona {persona_id}'s reason or biggest_issue")
    return items


def generate(brief, persona_results):
    issues = Counter()
    reasons = Counter()
    persona_evidence = {}
    for result in persona_results:
        persona_id = result.get("persona_id")
        if not isinstance(persona_id, str):
            continue
        sources = persona_evidence.setdefault(persona_id, [])
        for condition in ("sound_on", "sound_off"):
            reaction = result.get(condition) or {}
            if reaction.get("biggest_issue"):
                issues[reaction["biggest_issue"]] += 1
                sources.append(str(reaction["biggest_issue"]))
            if reaction.get("reason"):
                reasons[reaction["reason"]] += 1
                sources.append(str(reaction["reason"]))
    evidence = {"common_biggest_issues": issues.most_common(5), "common_reasons": reasons.most_common(5)}
    prompt = (PROMPT_PATH.read_text(encoding="utf-8") + "\nBRIEF:\n" + json.dumps(brief, ensure_ascii=False)
              + "\nPERSONA EVIDENCE BY ID:\n" + json.dumps(persona_evidence, ensure_ascii=False)
              + "\nCOMMON ISSUES AND REASONS:\n" + json.dumps(evidence, ensure_ascii=False))
    errors = []
    for attempt in range(2):
        try:
            with llm.call_stage("suggestions"):
                result = llm.ask_json(prompt)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            break
        try:
            return _valid(result, persona_evidence), errors
        except (TypeError, ValueError) as error:
            errors.append(f"{type(error).__name__}: {error}")
            prompt += "\nReturn exactly three valid suggestion objects. Every persona_id must be listed above and every quote must be copied verbatim from that persona's reason or biggest_issue."
    return [], errors
