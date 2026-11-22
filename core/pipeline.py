"""Run video extraction, audience waves, scoring, and suggestions."""
import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from core import audio, brief, config, llm
from core import framework
from core.extract import extract_all
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
    parts["result_schema"] = "core-context-v1"
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


def run(video_path, out_path, use_cache=True, retry_failed=None, context=None, niche=None):
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
    if retry_failed is None:
        retry_failed = use_cache
    start = len(llm.CALL_EVENTS)
    errors = []
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
        with llm.call_stage("audio_brief"):
            audio_result = audio.audio_brief(video_path)
        if not audio_result.get("available"):
            errors.append({"stage": "audio", "error": audio_result.get("reason", "Audio unavailable")})
        with llm.call_stage("extract"):
            extracted = extract_all(video_path, audio_result=audio_result)
        with llm.call_stage("vision_brief"):
            full = brief.build_full_brief(extracted, audio=audio_result, creator_context=context)
        if niche is not None:
            full["niche_category"] = niche
            full["niche_reason"] = "user override"
        reaction_cache_niche = niche or framework.normalize_niche(full.get("niche_category"))
        reaction_cache = _ReactionCache(CACHE_DIR, video_model_key, context_hash, reaction_cache_niche) if use_cache else None
        wave1_persona_cache = (_Wave1PersonaCache(CACHE_DIR / "wave1_personas", key_parts["video"],
                                                  _wave1_prompt_hash(), context_hash) if use_cache else None)
        waves = simulate(full, audio_available=bool(audio_result.get("available")),
                         reaction_cache=reaction_cache, wave1_persona_cache=wave1_persona_cache)
        scored = score_result(waves)
        all_scores = waves["wave1"] + waves["wave2"]
        suggestions, suggestion_errors = generate(full, all_scores)
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
                  **scored, "suggestions": suggestions, "errors": errors}
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["call_count"] = len(llm.CALL_EVENTS) - start
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_call_summary(llm.CALL_EVENTS[start:])
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    context = args.context
    if args.context_file:
        context = args.context_file.read_text(encoding="utf-8")
    run(args.video, args.out, use_cache=not args.no_cache, retry_failed=args.retry_failed,
        context=context, niche=args.niche)


if __name__ == "__main__":
    main()
