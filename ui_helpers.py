"""Pure helpers for displaying saved virality result files."""

from collections import defaultdict
from datetime import datetime
from html import escape
import math
import os
import re
from pathlib import Path


def is_incomplete(result):
    """Whether a saved result lacks a finished final score."""
    return result.get("complete") is False or result.get("final_score") is None


def suggestions_error_count(result):
    """Count saved errors from the optional suggestions stage."""
    errors = result.get("errors")
    if not isinstance(errors, list):
        return 0
    return sum(1 for item in errors
               if isinstance(item, dict) and item.get("stage") == "suggestions")


def should_show_last_run_warning(last_run_failed, last_run_result_path,
                                 selected_result_path, selected_result):
    """Avoid repeating the selected result's incomplete warning for that same run."""
    if not last_run_failed:
        return False
    same_selected_result = (
        last_run_result_path is not None
        and selected_result_path is not None
        and str(last_run_result_path) == str(selected_result_path)
        and isinstance(selected_result, dict)
        and is_incomplete(selected_result)
    )
    return not same_selected_result


ARCHETYPES = {
    "genz": "Gen Z scroller",
    "casual": "Casual viewer",
    "niche_fan": "Niche fan",
    "skeptic": "Skeptic",
    "creator": "Creator peer",
    "marketer": "Brand marketer",
}
AUDIENCE_SCORE_THRESHOLD = 6
AUDIENCE_ARCHETYPE_LABELS = {
    "genz": "Gen Z",
    "skeptic": "Skeptic",
    "niche_fan": "Niche fan",
    "casual": "Casual",
    "marketer": "Marketer",
    "creator": "Creator",
}
AUDIENCE_ARCHETYPE_ORDER = tuple(AUDIENCE_ARCHETYPE_LABELS)
AUDIENCE_SCORE_BUCKETS = (
    (2.5, "low"),
    (5.0, "mid_low"),
    (7.5, "mid_high"),
    (10.0, "high"),
)
WIDER_AUDIENCE_NARROW_THRESHOLD = 3.5
WIDER_AUDIENCE_BROAD_THRESHOLD = 6.5
DIMENSION_WEIGHTS = {
    "stop_scrolling": 0.35,
    "share_likelihood": 0.25,
    "watch_to_end": 0.25,
    "comment_likelihood": 0.15,
}
DIMENSION_LABELS = {
    "stop_scrolling": "Stop scrolling",
    "watch_to_end": "Watch to end",
    "share_likelihood": "Share",
    "comment_likelihood": "Comment",
}


def demo_mode_enabled(environ=None):
    """Enable replay mode only for the exact VIRALITY_DEMO=1 value."""
    environ = os.environ if environ is None else environ
    return environ.get("VIRALITY_DEMO") == "1"

NICHE_CATEGORIES = (
    "education",
    "comedy",
    "beauty",
    "fitness",
    "product_marketing",
    "general",
)

NICHE_HINTS = {
    "Auto-detect": "Auto-detect",
    "comedy": "Pacing, relatability, and the payoff",
    "education": "Clarity, curiosity, and value",
    "fitness": "Energy, credibility, and transformation",
    "beauty": "Visual appeal, trust, and technique",
    "food": "Appetite, clarity, and craveability",
    "Food": "Appetite, clarity, and craveability",
    "product_marketing": "Benefits, differentiation, and the call to action",
    "general": "Audience fit, clarity, and memorability",
    "Other (describe)": "A general rubric, guided by your description",
}


def wider_audience_label(crossover_score):
    """Map a crossover score to a broad, qualitative audience label."""
    if crossover_score is None or isinstance(crossover_score, bool):
        return None
    try:
        score = float(crossover_score)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    if score < WIDER_AUDIENCE_NARROW_THRESHOLD:
        return "Narrow"
    if score < WIDER_AUDIENCE_BROAD_THRESHOLD:
        return "Moderate"
    return "Broad"


def weighted_dimension_score(dimension_means, wave="wave1", condition="sound_on"):
    """Calculate a weighted 0-10 score from saved per-wave dimension means."""
    wave_data = dimension_means.get(wave) if isinstance(dimension_means, dict) else None
    values = wave_data.get(condition) if isinstance(wave_data, dict) else None
    if not isinstance(values, dict):
        return None
    weighted = []
    for dimension, weight in DIMENSION_WEIGHTS.items():
        value = values.get(dimension)
        if isinstance(value, bool):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            weighted.append((score, weight))
    if len(weighted) != len(DIMENSION_WEIGHTS):
        return None
    return sum(score * weight for score, weight in weighted)


def sound_score_gap(result):
    """Return Wave 1 weighted sound-on minus sound-off points, when available."""
    means = result.get("dimension_means") if isinstance(result, dict) else None
    sound_on = weighted_dimension_score(means, "wave1", "sound_on")
    sound_off = weighted_dimension_score(means, "wave1", "sound_off")
    return None if sound_on is None or sound_off is None else sound_on - sound_off


def weakest_dimension(dimension_means, wave="wave1", condition="sound_on"):
    """Return the key of the lowest available dimension for a wave/condition."""
    wave_data = dimension_means.get(wave) if isinstance(dimension_means, dict) else None
    values = wave_data.get(condition) if isinstance(wave_data, dict) else None
    if not isinstance(values, dict):
        return None
    scores = []
    for key in DIMENSION_LABELS:
        value = values.get(key)
        if isinstance(value, bool):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            scores.append((score, key))
    return min(scores)[1] if scores else None


def selected_result_path(session_state, option_paths, pending_result=None):
    """Resolve explicit selection or a just-completed run; never auto-select history."""
    selected = pending_result if pending_result in option_paths else session_state.get("selected_result_path")
    if selected not in option_paths:
        selected = None
    previous = session_state.get("selected_result_path")
    if selected is None:
        session_state.pop("selected_result_path", None)
    else:
        session_state["selected_result_path"] = selected
    if selected is not None and selected != previous:
        session_state["selection_scroll_pending"] = True
    return selected


def select_history_result(session_state, result_path):
    """Store a tile selection and request a best-effort scroll to the analysis."""
    session_state["selected_result_path"] = result_path
    session_state["selection_scroll_pending"] = True


def niche_hint(choice):
    """Return the fixed one-line helper text for a supported niche choice."""
    return NICHE_HINTS.get(choice, "Choose the audience lens that fits your reel")


def sanitize_history_name(value):
    """Strip a history display name and cap it at 60 characters."""
    return str(value or "").strip()[:60]


def derive_history_title(result_filename, history_names=None, modified_at=None):
    """Derive a mapped or date-based title for an old result file."""
    filename = Path(str(result_filename)).name
    stem = Path(filename).stem
    history_names = history_names if isinstance(history_names, dict) else {}
    stem_folded = stem.casefold()
    matches = [
        (str(prefix), name)
        for prefix, name in history_names.items()
        if isinstance(prefix, str)
        and isinstance(name, str)
        and (stem_folded == prefix.casefold()
             or stem_folded.startswith(prefix.casefold() + "_")
             or stem_folded.startswith(prefix.casefold() + "-"))
    ]
    if matches:
        _, display_name = max(matches, key=lambda match: len(match[0]))
        return sanitize_history_name(display_name)

    exact_match = re.fullmatch(r"result(\d+)", stem, flags=re.IGNORECASE)
    if exact_match:
        return f"Test {exact_match.group(1)}"
    suffix_match = re.fullmatch(r"result(\d+)[_-](.+)", stem, flags=re.IGNORECASE)
    if suffix_match:
        return f"Test {suffix_match.group(1)} ({suffix_match.group(2)})"
    if isinstance(modified_at, datetime):
        date = modified_at.astimezone().strftime("%Y-%m-%d")
    else:
        try:
            timestamp = float(modified_at) if modified_at is not None else datetime.now().timestamp()
            date = datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d")
        except (TypeError, ValueError, OSError, OverflowError):
            date = datetime.now().strftime("%Y-%m-%d")
    return f"Run on {date}"


def history_display_name(result_filename, metadata=None, history_names=None, modified_at=None):
    """Prefer a custom metadata name, then map/fallback names for legacy files."""
    custom_name = sanitize_history_name((metadata or {}).get("custom_display_name"))
    return custom_name or derive_history_title(result_filename, history_names, modified_at)


def history_tile_widget_key(result_filename):
    """Create a CSS-safe unique widget key from a local result filename."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(result_filename)).strip("_")[:48] or "result"
    return f"history_tile_{slug}"


def history_items_newest_first(items):
    """Sort history records by their numeric mtime/created timestamp, newest first."""
    def timestamp(item):
        value = item.get("sort_timestamp")
        if isinstance(value, (int, float)):
            return float(value)
        created_at = item.get("created_at")
        if isinstance(created_at, str):
            try:
                return datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        return 0.0

    return sorted(items, key=timestamp, reverse=True)


def filter_history_items(items, show_incomplete=False):
    """Hide incomplete history entries unless the toggle is enabled."""
    return list(items) if show_incomplete else [item for item in items if not item.get("incomplete")]


def slice_history_items(items, show_all=False, limit=8):
    """Show only the newest history entries until Show all is enabled."""
    limit = max(0, int(limit))
    return list(items) if show_all else list(items[:limit])


def format_history_date(value):
    """Format an ISO history timestamp for a compact local display."""
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.astimezone().strftime("%b %d, %Y at %I:%M %p")
        except ValueError:
            return value
    return "Date unavailable"

RUN_STAGES = ("Reading reel", "Core audience", "Gate", "Wider audience", "Scoring")
WAIT_MESSAGES = (
    "The Gen Z scroller is deciding whether to stop",
    "The skeptic is rolling their eyes",
    "The niche fan is taking notes",
    "Counting who would share",
)
LONG_RUN_WARNING_SECONDS = 25 * 60
RUN_DURATION_CAPTION = "Usually 8 to 15 minutes. Keep this tab open."
LONG_RUN_WARNING = "This is taking longer than usual, probably Gemini is busy. You can keep waiting or cancel."


_PIPELINE_STAGE_KEYS = {
    "Reading reel": "reading_reel",
    "Core audience": "core_audience",
    "Gate": "gate",
    "Wider audience": "wider_audience",
    "Scoring": "scoring",
}


def stage_pill_states(current_stage=None, stage_statuses=None):
    """Render only explicitly completed stages as done; call markers are merely current."""
    stage_statuses = stage_statuses or {}
    states = []
    for label in RUN_STAGES:
        stage_key = _PIPELINE_STAGE_KEYS[label]
        status = stage_statuses.get(stage_key)
        active = label == current_stage and (
            status == "started" or stage_statuses.get("suggestions") == "started"
        )
        if active:
            state = "current"
        elif status == "completed":
            state = "done"
        elif status == "skipped":
            state = "skipped"
        elif status in {"incomplete", "failed"}:
            state = "failed"
        elif label == current_stage or status == "started":
            state = "current"
        else:
            state = "upcoming"
        states.append((label, state))
    return states


def stage_progress_percent(current_stage=None, stage_statuses=None):
    """Estimate progress from explicit stage results and the current stage marker."""
    states = stage_pill_states(current_stage, stage_statuses)
    terminal = sum(state in {"done", "failed", "skipped"} for _, state in states)
    current = sum(state == "current" for _, state in states)
    return min(100, terminal * 20 + current * 10)


def rotating_wait_message(elapsed_seconds):
    """Choose a stable waiting line for each six-second interval."""
    interval = max(0, int(float(elapsed_seconds)) // 6)
    return WAIT_MESSAGES[interval % len(WAIT_MESSAGES)]


def long_run_message(elapsed_seconds):
    """Return the calm long-run notice only after 25 minutes have elapsed."""
    return LONG_RUN_WARNING if float(elapsed_seconds) > LONG_RUN_WARNING_SECONDS else None


def run_progress_copy(current_stage, elapsed_seconds, fallback_stage="Reading reel, audience waves and scoring..."):
    """Return all text rendered in the active progress area, without a timer."""
    return {
        "waiting_message": rotating_wait_message(elapsed_seconds),
        "duration_caption": RUN_DURATION_CAPTION,
        "stage_caption": f"Current stage: {current_stage}" if current_stage else fallback_stage,
        "long_run_message": long_run_message(elapsed_seconds),
    }


def resolve_niche_choice(choice, custom_text=None):
    """Resolve a UI niche choice without connecting it to the pipeline."""
    if choice == "Auto-detect":
        return {"niche_override": None, "extra_context": None}
    if choice in NICHE_CATEGORIES:
        return {"niche_override": choice, "extra_context": None}
    if choice == "Food":
        return {
            "niche_override": "general",
            "extra_context": "Creator describes the niche as food.",
        }
    if choice == "Other (describe)":
        description = (custom_text or "").strip()[:60]
        if not description:
            return {"error": "Please describe your niche before continuing."}
        return {
            "niche_override": "general",
            "extra_context": f"Creator describes the niche as: {description}",
        }
    return {"error": "Choose Auto-detect or a listed niche."}


def archetype_averages(waves):
    """Average persona scores by the persona ID prefix used by archetype_report."""
    grouped = defaultdict(list)
    for wave_name in ("wave1", "wave2"):
        for persona in waves.get(wave_name) or []:
            if not isinstance(persona, dict) or not persona.get("persona_id") or persona.get("score") is None:
                continue
            archetype_id = str(persona["persona_id"]).rsplit("_", 2)[0]
            if archetype_id in ARCHETYPES:
                grouped[archetype_id].append(float(persona["score"]))
    return {
        key: (sum(grouped[key]) / len(grouped[key]) if grouped[key] else None)
        for key in ARCHETYPES
    }


def extract_audience_comments(result):
    """Collect persona reaction quotes and suggestion quotes from a saved result."""
    comments = []
    for wave_number, wave_key in ((1, "wave1"), (2, "wave2")):
        for persona in result.get(wave_key) or []:
            if not isinstance(persona, dict):
                continue
            persona_id = str(persona.get("persona_id") or "Unknown persona")
            archetype_key = persona_id.rsplit("_", 2)[0]
            initial = ARCHETYPES.get(archetype_key, "Audience member")[0].upper()
            for condition in ("sound_on", "sound_off"):
                reaction = persona.get(condition)
                quote = reaction.get("reason") if isinstance(reaction, dict) else None
                if isinstance(quote, str) and quote.strip():
                    comments.append({
                        "persona_id": persona_id,
                        "archetype_key": archetype_key if archetype_key in ARCHETYPES else "unknown",
                        "initial": initial,
                        "badge": f"Wave {wave_number} · {condition.replace('_', ' ')}",
                        "quote": quote.strip(),
                    })

    for suggestion in result.get("suggestions") or []:
        if not isinstance(suggestion, dict):
            continue
        quote = suggestion.get("quote")
        if isinstance(quote, str) and quote.strip():
            comments.append({
                "persona_id": str(suggestion.get("persona_id") or ""),
                "archetype_key": "suggestion",
                "initial": "S",
                "badge": "Suggestion",
                "quote": quote.strip(),
            })
    return comments


def split_audience_comments(comments, limit=12):
    """Split comments into the initially visible set and the expandable remainder."""
    limit = max(0, int(limit))
    return list(comments[:limit]), list(comments[limit:])


def _archetype_key(persona_id):
    """Resolve known archetype prefixes, including dynamic and break IDs."""
    persona_id = str(persona_id or "")
    for key in sorted(AUDIENCE_ARCHETYPE_LABELS, key=len, reverse=True):
        if persona_id == key or persona_id.startswith(key + "_"):
            return key
    prefix = persona_id.rsplit("_", 2)[0] if "_" in persona_id else persona_id
    return prefix if prefix in AUDIENCE_ARCHETYPE_LABELS else "unknown"


def collect_scored_personas(result):
    """Collect scored persona records from the result's two audience waves."""
    collected = []
    for wave_key in ("wave1", "wave2"):
        wave = result.get(wave_key)
        if not isinstance(wave, list):
            continue
        for item in wave:
            if not isinstance(item, dict) or not item.get("persona_id"):
                continue
            try:
                score = float(item.get("score"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            persona_id = str(item["persona_id"])
            collected.append({
                "persona_id": persona_id,
                "score": score,
                "archetype_key": _archetype_key(persona_id),
                "sound_on_probability": item.get("sound_on_probability", 1.0),
                "sound_on": item.get("sound_on") if isinstance(item.get("sound_on"), dict) else {},
                "sound_off": item.get("sound_off") if isinstance(item.get("sound_off"), dict) else {},
                "wave": wave_key,
            })
    return collected


def group_personas_by_archetype(personas):
    """Return known audience groups in display order, plus unknown IDs last."""
    grouped = defaultdict(list)
    for persona in personas:
        grouped[persona.get("archetype_key") or _archetype_key(persona.get("persona_id"))].append(persona)
    keys = [key for key in AUDIENCE_ARCHETYPE_ORDER if grouped.get(key)]
    if grouped.get("unknown"):
        keys.append("unknown")
    return [
        (key, AUDIENCE_ARCHETYPE_LABELS.get(key, "Other"), grouped[key])
        for key in keys
    ]


def score_bucket(score):
    """Map a 0-10 score into one of four visual intensity buckets."""
    value = min(10.0, max(0.0, float(score)))
    for upper_bound, bucket in AUDIENCE_SCORE_BUCKETS:
        if value <= upper_bound:
            return bucket
    return "high"


def select_extreme_personas(personas, limit=3):
    """Return stable top and low persona ID lists; score ties keep input order."""
    limit = max(0, int(limit))
    top = sorted(personas, key=lambda persona: float(persona["score"]), reverse=True)[:limit]
    low = sorted(personas, key=lambda persona: float(persona["score"]))[:limit]
    return (
        [str(persona["persona_id"]) for persona in top],
        [str(persona["persona_id"]) for persona in low],
    )


def audience_summary(personas, threshold=AUDIENCE_SCORE_THRESHOLD):
    """Count scores and weighted reaction dimensions using result JSON only."""
    def dimension_value(persona, dimension):
        on = persona.get("sound_on", {}).get(dimension)
        off = persona.get("sound_off", {}).get(dimension)
        try:
            on = float(on) if on is not None else None
            off = float(off) if off is not None else None
        except (TypeError, ValueError):
            return None
        if on is None:
            return off
        if off is None:
            return on
        try:
            probability = min(1.0, max(0.0, float(persona.get("sound_on_probability", 1.0))))
        except (TypeError, ValueError):
            probability = 1.0
        return probability * on + (1.0 - probability) * off

    return {
        "total": len(personas),
        "score_at_or_above": sum(persona["score"] >= threshold for persona in personas),
        "stop_scrolling": sum(
            (value := dimension_value(persona, "stop_scrolling")) is not None and value >= threshold
            for persona in personas
        ),
        "watch_to_end": sum(
            (value := dimension_value(persona, "watch_to_end")) is not None and value >= threshold
            for persona in personas
        ),
        "might_share": sum(
            (value := dimension_value(persona, "share_likelihood")) is not None and value >= threshold
            for persona in personas
        ),
    }


def shorten_first_sentence(text, max_chars=160):
    """Return the first sentence, shortened at sentence or word boundary."""
    text = str(text or "").strip()
    max_chars = max(1, int(max_chars))
    sentence = re.search(r"[.!?](?:[\"'\u2019\u201d\)\]]*)?(?=\s|$)", text)
    if sentence and sentence.end() <= max_chars:
        return text[:sentence.end()].strip()
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    boundary = head.rfind(" ")
    ellipsis = "..." if max_chars >= 3 else "." * max_chars
    if boundary <= 0:
        return head[:max(0, max_chars - len(ellipsis))].rstrip() + ellipsis
    return head[:min(boundary, max_chars - len(ellipsis))].rstrip() + ellipsis


def audience_spotlight_html(persona, sound_mode, quote):
    """Build an escaped spotlight card for the selected result persona."""
    archetype_key = persona.get("archetype_key", "unknown")
    label = AUDIENCE_ARCHETYPE_LABELS.get(archetype_key, "Audience member")
    initial = label[:1].upper()
    score = float(persona.get("score", 0.0))
    sound_label = "Sound on" if sound_mode == "sound_on" else "Sound off"
    safe_key = archetype_key if archetype_key in AUDIENCE_ARCHETYPE_LABELS else "unknown"
    return (
        f'<article class="vs-audience-spotlight"><div class="vs-spotlight-head">'
        f'<span class="vs-avatar vs-avatar-{safe_key}">{escape(initial, quote=True)}</span>'
        f'<div><div class="vs-spotlight-person">{escape(str(persona.get("persona_id", "")), quote=True)}</div>'
        f'<div class="vs-spotlight-archetype">{escape(label, quote=True)} | {escape(sound_label, quote=True)}</div></div>'
        f'<div class="vs-spotlight-score">{score:.1f}<small> / 10</small></div></div>'
        f'<p class="vs-spotlight-quote">{escape(str(quote), quote=True)}</p></article>'
    )


def audience_tile_widget_key(persona_id, index):
    """Make a stable, CSS-safe Streamlit key tied to one result persona."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(persona_id)).strip("_")[:40] or "persona"
    return f"audience_tile_{slug}_{int(index)}"


def audience_tile_style_html(widget_key, bucket, badges=(), selected=False):
    """Return safe CSS for one keyed Streamlit persona button."""
    if not re.fullmatch(r"audience_tile_[A-Za-z0-9_-]+", str(widget_key)):
        raise ValueError("Invalid audience tile widget key")
    colors = {
        "low": ("#F0EFF3", "#666574", "#D8D6DF"),
        "mid_low": ("#FCE5F2", "#9B376F", "#F2BED9"),
        "mid_high": ("#ED9BCB", "#59264F", "#E77BB9"),
        "high": ("linear-gradient(135deg,#E6007E,#A83CF0)", "#fff", "#A83CF0"),
    }
    background, foreground, border = colors.get(bucket, colors["low"])
    badge_values = [badge for badge in badges if badge in {"top", "low"}]
    badge_text = " / ".join(badge_values)
    badge_markup = ""
    selected_css = " box-shadow:0 0 0 2px #342043,0 5px 12px rgba(90,35,100,.16);" if selected else ""
    if badge_text:
        badge_markup = (
            f'<style>.st-key-{widget_key} button::after{{content:"{escape(badge_text, quote=True)}";'
            'position:absolute;top:-5px;right:-4px;padding:1px 4px;border-radius:999px;'
            'background:#fff;color:#6B5273;font-size:8px;font-weight:800;line-height:1.3;'
            'box-shadow:0 1px 5px rgba(40,20,50,.15)}}</style>'
        )
    return (
        f'<style>.st-key-{widget_key} button{{position:relative;background:{background} !important;'
        f'color:{foreground} !important;border:1px solid {border} !important;min-height:44px;'
        'padding:.32rem .3rem !important;font-size:.72rem !important;font-weight:800 !important;'
        f'{selected_css}'
        'white-space:nowrap;transition:transform .15s ease,box-shadow .15s ease}}'
        f'.st-key-{widget_key} button:hover{{transform:translateY(-2px);box-shadow:0 5px 12px rgba(90,35,100,.14)}}'
        f'</style>{badge_markup}'
    )
