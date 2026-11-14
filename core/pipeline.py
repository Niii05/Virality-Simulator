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

from core import audio, brief, llm
from core import framework
from core.extract import extract_all
from core.scoring import score_result
from core.suggestions import generate
from core.waves import dimension_means, simulate
from scripts.try_brief import _print_call_summary

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "tmp" / "cache"


class _ReactionCache:
    """Persist each completed persona-condition reaction independently."""
    def __init__(self, directory, video_model_key):
        self.directory = Path(directory)
        self.video_model_key = video_model_key

    def _path(self, persona_id, condition):
        safe_id = hashlib.sha256(str(persona_id).encode("utf-8")).hexdigest()
        return self.directory / "reactions" / f"{self.video_model_key}_{safe_id}_{condition}.json"

    def get(self, persona_id, condition):
        path = self._path(persona_id, condition)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, persona_id, condition, value):
        path = self._path(persona_id, condition)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)


def _key_parts(video, model, root=ROOT):
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
    return parts


def _key(video, model, root=ROOT):
    parts = _key_parts(video, model, root)
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def run(video_path, out_path, use_cache=True, retry_failed=None):
    video_path, out_path = Path(video_path), Path(out_path)
    model = llm._model_name()
    key_parts = _key_parts(video_path, model)
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
            full = brief.build_full_brief(extracted, audio=audio_result)
        reaction_cache = _ReactionCache(CACHE_DIR, video_model_key) if use_cache else None
        waves = simulate(full, audio_available=bool(audio_result.get("available")),
                         reaction_cache=reaction_cache)
        scored = score_result(waves)
        all_scores = waves["wave1"] + waves["wave2"]
        suggestions, suggestion_errors = generate(full, all_scores)
        errors.extend(waves["errors"])
        errors.extend({"stage": "suggestions", "error": item} for item in suggestion_errors)
        niche_category = framework.normalize_niche(full.get("niche_category"))
        result = {"framework_version": framework.get_version(), "niche_category": niche_category,
                  "niche_reason": full.get("niche_reason", "The video did not clearly fit a specific niche based on its purpose."),
                  "brief": full, "wave1": waves["wave1"], "wave2": waves["wave2"],
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
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--retry-failed", dest="retry_failed", action="store_true", default=None,
                        help="resume incomplete runs from successful cached persona reactions (default with cache)")
    parser.add_argument("--no-retry-failed", dest="retry_failed", action="store_false",
                        help="return an incomplete cached result without rerunning it")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.video, args.out, use_cache=not args.no_cache, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
