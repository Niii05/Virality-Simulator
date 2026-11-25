"""Run video extraction, audience waves, scoring, and suggestions."""
import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from core import audio, brief, config, llm
from core import framework
from core.extract import FRAMES_DIR, extract_all
from core.scoring import score_result
from core.suggestions import generate
from core.waves import dimension_means, simulate
from scripts.try_brief import _print_call_summary

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "tmp" / "cache"
MAX_CONTEXT_LENGTH = 600


def _normalize_context(context):
    if context is None:
        return None
    context = str(context).strip()
    if len(context) > MAX_CONTEXT_LENGTH:
        logging.getLogger(__name__).warning("Creator context exceeds %d characters; truncating.", MAX_CONTEXT_LENGTH)
        context = context[:MAX_CONTEXT_LENGTH]
    return context or None


def _context_hash(context):
    return hashlib.sha256((context or "").encode("utf-8")).hexdigest()


class _ReactionCache:
    """Persist each completed persona-condition reaction independently."""
    def __init__(self, directory, video_model_key, context_hash="", niche=""):
        self.directory = Path(directory)
        self.video_model_key = video_model_key
        self.context_hash = context_hash
        self.niche = niche

    def _path(self, persona, condition):
        profile = persona if isinstance(persona, dict) else {"id": str(persona)}
        persona_id = profile.get("id", "unknown")
        safe_id = hashlib.sha256(str(persona_id).encode("utf-8")).hexdigest()
        profile_payload = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        profile_hash = hashlib.sha256(profile_payload.encode("utf-8")).hexdigest()
        logging.getLogger(__name__).info(
            "reaction cache key parts: video_model=%s persona_id=%s profile_sha256=%s condition=%s",
            self.video_model_key, persona_id, profile_hash, condition)
        cache_key = hashlib.sha256(
            f"{self.video_model_key}:{self.context_hash}:{self.niche}:{safe_id}:{profile_hash}:{condition}".encode("utf-8")
        ).hexdigest()
        return self.directory / "reactions" / f"{cache_key}.json"

    def get(self, persona, condition):
        path = self._path(persona, condition)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, persona, condition, value):
        path = self._path(persona, condition)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)


class _Wave1PersonaCache:
    """Persist generated Wave 1 profiles against their video, brief, and prompt."""
    def __init__(self, directory, video_hash, prompt_hash, context_hash=""):
        self.directory = Path(directory)
        self.video_hash = video_hash
        self.prompt_hash = prompt_hash
        self.context_hash = context_hash

    def _parts(self, brief_value):
        brief_payload = json.dumps(brief_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return {"video_hash": self.video_hash,
                "brief_hash": hashlib.sha256(brief_payload.encode("utf-8")).hexdigest(),
                "prompt_hash": self.prompt_hash,
                "context_hash": self.context_hash}

    def _path(self, brief_value):
        parts = self._parts(brief_value)
        key = hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()
        return self.directory / f"{key}.json", parts

    def get(self, brief_value):
        path, parts = self._path(brief_value)
        logging.getLogger(__name__).info("wave1 personas cache key parts: %s",
                                         json.dumps(parts, ensure_ascii=False, sort_keys=True))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            profiles = value.get("wave1_personas") if isinstance(value, dict) else None
            selection = value.get("wave1_selection") if isinstance(value, dict) else None
            if (isinstance(profiles, list) and len(profiles) == config.WAVE1_SIZE
                    and all(isinstance(item, dict) for item in profiles)
                    and selection in {"dynamic", "dynamic_filled"}):
                return {"wave1_personas": profiles, "wave1_selection": selection,
                        "wave1_raw_ids": value.get("wave1_raw_ids", [])}
        except (OSError, json.JSONDecodeError):
            pass
        return None

    def set(self, brief_value, profiles, selection, raw_ids):
        path, _ = self._path(brief_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"wave1_personas": profiles,
                                         "wave1_selection": selection,
                                         "wave1_raw_ids": raw_ids}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)


def _key_parts(video, model, root=ROOT, context=None, niche=None):
    digest = hashlib.sha256()
    parts = {}
    with Path(video).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    parts["video"] = digest.hexdigest()
    hashed_files = sorted((Path(root) / "prompts").glob("*.txt"))
    hashed_files += [Path(root) / "data" / "personas.json", Path(root) / "knowledge_base" / "evaluation_framework.json"]
    for path in hashed_files:
        parts[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    parts["model"] = hashlib.sha256(model.encode("utf-8")).hexdigest()
    parts["context_hash"] = _context_hash(context)
    parts["niche_override"] = niche
    parts["result_schema"] = "core-context-v3-batched-reactions"
    parts["persona_runs"] = [config.WAVE1_RUNS, config.WAVE2_RUNS]
    parts["persona_batch_size"] = config.BATCH_SIZE
    parts["persona_temperature"] = config.PERSONA_TEMPERATURE
    parts["score_weights"] = config.SCORE_WEIGHTS
    parts["confidence_stddev_thresholds"] = [
        config.HIGH_CONFIDENCE_MAX_STDDEV,
        config.MEDIUM_CONFIDENCE_MAX_STDDEV,
    ]
    parts["wave1_slot_archetypes"] = config.WAVE1_SLOT_ARCHETYPES
    parts["wave2_size"] = config.WAVE2_SIZE
    return parts


def _wave1_prompt_hash():
    path = ROOT / "prompts" / "audience_personas.txt"
    payload = path.read_bytes() + json.dumps({
        "slots": config.WAVE1_SLOT_ARCHETYPES,
        "size": config.WAVE1_SIZE,
        "temperature": config.AUDIENCE_PERSONA_TEMPERATURE,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _key(video, model, root=ROOT, context=None, niche=None):
    parts = _key_parts(video, model, root, context, niche)
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def _vision_brief_failure_result(context, errors, niche=None):
    from core.personas import load_personas

    persona_ids = [persona["id"] for persona in load_personas()]
    execution_summary = {
        "total_personas": len(persona_ids), "completed": 0, "failed": 0,
        "gate_skipped": 0, "not_started": len(persona_ids),
        "overall_completion_status": "incomplete",
    }
    logging.getLogger(__name__).info(
        "Persona execution summary %s", json.dumps(execution_summary, sort_keys=True)
    )
    return {
        "framework_version": framework.get_version(),
        "niche_category": framework.normalize_niche(niche) if niche else "general",
        "user_context": context, "context_used": bool(context), "niche_reason": "",
        "brief": None, "wave1": [], "wave1_personas": [], "wave2": [], "waves_run": [],
        "wave1_selection": None, "wave1_raw_ids": [],
        "dimension_means": dimension_means([]), "per_persona_scores": {},
        "label": "Virality potential estimate", "core_resonance": None,
        "core_resonance_spread": None, "confidence": None, "crossover_score": None,
        "final_score": None, "reach_tier": "Incomplete - rerun",
        "tier_note": "vision brief failed; rerun to complete the estimate",
        "gate_passed": None, "personas_intended": 0, "personas_scored": 0,
        "personas_failed": [], "complete": False, "scoring_status": "skipped",
        "failure_stage": "vision_brief",
        "persona_execution_summary": execution_summary,
        "personas_gate_skipped": [], "personas_not_started": persona_ids,
        "suggestions": [], "suggestions_status": "skipped", "errors": errors,
    }


def run(video_path, out_path, use_cache=True, retry_failed=None, context=None, niche=None,
        retry_suggestions=False):
    video_path, out_path = Path(video_path), Path(out_path)
    allowed_niches = framework._load()["niche_category_enum"]
    if niche is not None and niche not in allowed_niches:
        raise ValueError(f"Invalid niche {niche!r}. Allowed values: {', '.join(allowed_niches)}")
    context = _normalize_context(context)
    context_hash = _context_hash(context)
    model = llm._model_name()
    key_parts = _key_parts(video_path, model, context=context, niche=niche)
    video_model_key = hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode("utf-8")).hexdigest()
    logging.getLogger(__name__).info("cache key parts: %s", json.dumps(key_parts, ensure_ascii=False, sort_keys=True))
    cache_file = CACHE_DIR / (video_model_key + ".json")
    audio_cache_file = CACHE_DIR / (video_model_key + ".audio.json")
    if retry_failed is None:
        retry_failed = use_cache
    start = len(llm.CALL_EVENTS)
    errors = []
    if retry_suggestions:
        if not out_path.is_file():
            raise FileNotFoundError(f"Saved result not found: {out_path}")
        result = json.loads(out_path.read_text(encoding="utf-8"))
        if (not isinstance(result, dict) or result.get("complete") is not True
                or not isinstance(result.get("brief"), dict)
                or not isinstance(result.get("wave1"), list)
                or not isinstance(result.get("wave2"), list)):
            raise ValueError("Suggestions-only retry requires a completed saved result with brief and persona results.")

        persona_results = result["wave1"] + result["wave2"]
        logging.getLogger(__name__).info(
            "Pipeline stage event stage=suggestions status=started"
        )
        if persona_results:
            try:
                regenerated, suggestion_errors = generate(result["brief"], persona_results)
            except Exception as error:
                regenerated = None
                suggestion_errors = [f"{type(error).__name__}: {llm._redact_error(error)}"]
        else:
            regenerated, suggestion_errors = None, []
        existing_errors = result.get("errors")
        if not isinstance(existing_errors, list):
            existing_errors = []
        if not persona_results:
            result["suggestions_status"] = "skipped"
            result["errors"] = [
                item for item in existing_errors
                if not isinstance(item, dict) or item.get("stage") != "suggestions"
            ]
            stage_status = "skipped"
        elif not suggestion_errors and isinstance(regenerated, list) and len(regenerated) == 3:
            result["suggestions"] = regenerated
            result["suggestions_status"] = "completed"
            result["errors"] = [
                item for item in existing_errors
                if not isinstance(item, dict) or item.get("stage") != "suggestions"
            ]
            stage_status = "completed"
        else:
            reason = suggestion_errors or ["ValueError: suggestions generator returned an invalid result"]
            result["suggestions_status"] = "failed"
            result["errors"] = [
                item for item in existing_errors
                if not isinstance(item, dict) or item.get("stage") != "suggestions"
            ] + [{"stage": "suggestions", "error": llm._redact_error(item)} for item in reason]
            stage_status = "failed"

        logging.getLogger(__name__).info(
            "Pipeline stage event stage=suggestions status=%s", stage_status
        )
        serialized = json.dumps(result, ensure_ascii=False, indent=2)
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(serialized, encoding="utf-8")
        out_path.write_text(serialized, encoding="utf-8")
        _print_call_summary(llm.CALL_EVENTS[start:])
        logging.getLogger(__name__).info(
            "Pipeline terminal result_status=%s suggestions_status=%s process_status=normal",
            "complete" if result.get("complete") is True else "incomplete",
            result.get("suggestions_status", "unknown"),
        )
        return result

    if use_cache and cache_file.exists():
        result = json.loads(cache_file.read_text(encoding="utf-8"))
        if retry_failed and result.get("complete") is not False:
            retry_cached = False
        else:
            retry_cached = bool(retry_failed and result.get("complete") is False)
        if "dimension_means" not in result:
            result["dimension_means"] = {key: dimension_means(result.get(key, []))
                                          for key in ("wave1", "wave2")
                                          if key == "wave1" or result.get("gate_passed")}
    else:
        retry_cached = False
    if not (use_cache and cache_file.exists() and not retry_cached):
        audio_result = None
        if use_cache and audio_cache_file.exists():
            try:
                audio_result = json.loads(audio_cache_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                audio_result = None
        logging.getLogger(__name__).info(
            "Pipeline stage event stage=reading_reel status=started"
        )
        if audio_result is None:
            with llm.call_stage("audio_brief"):
                audio_result = audio.audio_brief(video_path)
            if use_cache and audio_result.get("available"):
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                audio_cache_file.write_text(json.dumps(audio_result, ensure_ascii=False, indent=2), encoding="utf-8")
        if not audio_result.get("available"):
            errors.append({"stage": "audio", "error": audio_result.get("reason", "Audio unavailable")})
        full = None
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="run_", dir=FRAMES_DIR) as frames_dir:
            with llm.call_stage("extract"):
                extracted = extract_all(
                    video_path, audio_result=audio_result, frames_dir=frames_dir
                )
            try:
                with llm.call_stage("vision_brief"):
                    full = brief.build_full_brief(extracted, audio=audio_result, creator_context=context)
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                errors.append({"stage": "vision_brief", "error": reason})
                logging.getLogger(__name__).error("vision_brief failed (%s)", reason)
                logging.getLogger(__name__).info(
                    "Pipeline stage event stage=reading_reel status=failed"
                )
                result = _vision_brief_failure_result(context, errors, niche)
        if full is not None:
            logging.getLogger(__name__).info(
                "Pipeline stage event stage=reading_reel status=completed"
            )
            if niche is not None:
                full["niche_category"] = niche
                full["niche_reason"] = "user override"
            reaction_cache_niche = niche or framework.normalize_niche(full.get("niche_category"))
            reaction_cache = _ReactionCache(CACHE_DIR, video_model_key, context_hash, reaction_cache_niche) if use_cache else None
            wave1_persona_cache = (_Wave1PersonaCache(CACHE_DIR / "wave1_personas", key_parts["video"],
                                                      _wave1_prompt_hash(), context_hash) if use_cache else None)
            waves = simulate(full, audio_available=bool(audio_result.get("available")),
                             reaction_cache=reaction_cache, wave1_persona_cache=wave1_persona_cache)
            logging.getLogger(__name__).info(
                "Pipeline stage event stage=scoring status=started"
            )
            scored = score_result(waves)
            score_status = ("completed" if waves.get("complete") is True and waves.get("personas_scored", 0) > 0
                            else "incomplete")
            logging.getLogger(__name__).info(
                "Pipeline stage event stage=scoring status=%s", score_status
            )
            all_scores = waves["wave1"] + waves["wave2"]
            logging.getLogger(__name__).info(
                "Pipeline stage event stage=suggestions status=started"
            )
            if all_scores:
                suggestions, suggestion_errors = generate(full, all_scores)
                if (not suggestion_errors and isinstance(suggestions, list)
                        and len(suggestions) == 3):
                    suggestions_status = "completed"
                else:
                    suggestions_status = "failed"
                    if not suggestion_errors:
                        suggestion_errors = [
                            "ValueError: suggestions generator returned an invalid result"
                        ]
            else:
                suggestions, suggestion_errors = [], []
                suggestions_status = "skipped"
            logging.getLogger(__name__).info(
                "Pipeline stage event stage=suggestions status=%s", suggestions_status
            )
            errors.extend(waves["errors"])
            errors.extend({"stage": "suggestions", "error": item} for item in suggestion_errors)
            niche_category = framework.normalize_niche(full.get("niche_category"))
            result = {"framework_version": framework.get_version(), "niche_category": niche_category,
                      "user_context": context, "context_used": bool(context),
                      "niche_reason": full.get("niche_reason", "The video did not clearly fit a specific niche based on its purpose."),
                      "brief": full, "wave1": waves["wave1"], "wave1_personas": waves.get("wave1_personas", []),
                      "wave2": waves["wave2"],
                      "waves_run": waves.get("waves_run", [1, 2] if waves.get("gate_passed") else [1]),
                      "wave1_selection": waves["wave1_selection"], "wave1_raw_ids": waves["wave1_raw_ids"],
                      "dimension_means": waves["dimension_means"],
                      "per_persona_scores": {item["persona_id"]: item["score"] for item in all_scores},
                      "persona_execution_summary": waves.get("persona_execution_summary"),
                      "personas_gate_skipped": waves.get("personas_gate_skipped", []),
                      "personas_not_started": waves.get("personas_not_started", []),
                      **scored, "scoring_status": score_status, "suggestions": suggestions,
                      "suggestions_status": suggestions_status,
                      "errors": errors}
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["call_count"] = len(llm.CALL_EVENTS) - start
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_call_summary(llm.CALL_EVENTS[start:])
    logging.getLogger(__name__).info(
        "Pipeline terminal result_status=%s suggestions_status=%s process_status=normal",
        "complete" if result.get("complete") is True else "incomplete",
        result.get("suggestions_status", "unknown"),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--out", default="tmp/result.json")
    context_group = parser.add_mutually_exclusive_group()
    context_group.add_argument("--context", help="creator-provided context (maximum 600 characters)")
    context_group.add_argument("--context-file", type=Path, help="UTF-8 file containing creator-provided context")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--niche", choices=framework._load()["niche_category_enum"],
                        help="override niche detection with a framework niche category")
    parser.add_argument("--retry-failed", dest="retry_failed", action="store_true", default=None,
                        help="resume incomplete runs from successful cached persona reactions (default with cache)")
    parser.add_argument("--no-retry-failed", dest="retry_failed", action="store_false",
                        help="return an incomplete cached result without rerunning it")
    parser.add_argument("--retry-suggestions", action="store_true",
                        help="retry only suggestions from the completed saved result at --out")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    context = args.context
    if args.context_file:
        context = args.context_file.read_text(encoding="utf-8")
    run(args.video, args.out, use_cache=not args.no_cache, retry_failed=args.retry_failed,
        context=context, niche=args.niche, retry_suggestions=args.retry_suggestions)


if __name__ == "__main__":
    main()
