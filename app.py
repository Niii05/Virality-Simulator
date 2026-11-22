"""Saved-result viewer and confirmed local pipeline runner."""
import json
import os
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

from game_html import game_html as build_game_html
from run_helpers import (
    FALLBACK_PROGRESS_TEXT,
    DemoRunner,
    build_pipeline_command,
    current_pipeline_stage,
    demo_stage_seconds_from_env,
    elapsed_run_seconds,
    generate_safe_filename,
    history_meta_path,
    last_log_lines,
    read_history_meta,
    read_log_text,
    run_is_active,
    start_subprocess,
    save_legacy_test_thumbnail,
    save_video_thumbnail,
    validate_upload,
    write_history_meta,
)
from ui_helpers import (
    AUDIENCE_SCORE_THRESHOLD,
    NICHE_CATEGORIES,
    archetype_averages,
    audience_summary,
    audience_spotlight_html,
    audience_tile_widget_key,
    collect_scored_personas,
    demo_mode_enabled,
    derive_history_title,
    filter_history_items,
    format_history_date,
    group_personas_by_archetype,
    history_display_name,
    history_items_newest_first,
    history_tile_widget_key,
    is_incomplete,
    run_progress_copy,
    niche_hint,
    resolve_niche_choice,
    score_bucket,
    select_extreme_personas,
    sanitize_history_name,
    slice_history_items,
    shorten_first_sentence,
    stage_pill_states,
    stage_progress_percent,
    wider_audience_label,
)
from ui_theme import (
    CSS,
    audience_tile_style_html,
    demo_banner_html,
    feature_card_html,
    hero_html,
    history_placeholder_html,
    history_tile_html,
    heat_bar_html,
    progress_html,
    result_hero_html,
    score_card_html,
    suggestion_card_html,
    wider_audience_card_html,
    wider_audience_raw_score_html,
)


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "tmp"
HISTORY_DIR = RESULT_DIR / "history"
HISTORY_NAMES_PATH = ROOT / "data" / "history_names.json"
DEMO_MODE = demo_mode_enabled()
GAME_HTML = build_game_html()
LIMITATIONS = [
    "Scores are not calibrated to real view counts.",
    "The judge reads a text brief of the video, so it does not see visual or design quality.",
    "Brand or celebrity fame is not visible unless added as context.",
    "Trending audio is not detected.",
    "Niche rubric guidance is partly extrapolated from ad guidance and creator advice.",
    "Results vary by about 0.8 between runs.",
]
NICHE_OPTIONS = ["Auto-detect", *NICHE_CATEGORIES, "Food", "Other (describe)"]


def _select_wall_persona(persona_id):
    st.session_state["selected_audience_persona_id"] = persona_id


def _select_history_result(result_path):
    st.session_state["selected_result_path"] = result_path


def _begin_history_rename(result_file, display_name):
    st.session_state["history_edit_result_file"] = result_file
    st.session_state["history_rename_value"] = display_name


st.set_page_config(page_title="Virality Simulator", page_icon="✨", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
if DEMO_MODE:
    st.markdown(demo_banner_html(), unsafe_allow_html=True)
st.markdown(hero_html(), unsafe_allow_html=True)
st.markdown('<div class="vs-how-title">How it works</div>', unsafe_allow_html=True)
feature_data = [
    ("Step 01", "🎬", "Upload", "Drop your reel."),
    ("Step 02", "👥", "Audience", "24 AI personas react in two waves."),
    ("Step 03", "✦", "Score", "A tier and a range, not a promise."),
    ("Step 04", "💬", "Fix", "3 suggestions with a quote."),
]
feature_columns = st.columns(4, wrap=True)
for index, column in enumerate(feature_columns):
    with column:
        label, icon, title, copy = feature_data[index]
        st.markdown(feature_card_html(label, icon, title, copy, dark=index == 3), unsafe_allow_html=True)

for key, default in (
    ("run_active", False),
    ("run_process", None),
    ("run_confirm_pending", False),
    ("last_run_failed", False),
    ("last_run_result_path", None),
    ("last_run_log_path", None),
    ("last_run_error_count", 0),
):
    if key not in st.session_state:
        st.session_state[key] = default

RESULT_DIR.mkdir(parents=True, exist_ok=True)
if not DEMO_MODE:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
try:
    history_names_value = json.loads(HISTORY_NAMES_PATH.read_text(encoding="utf-8"))
    history_names = history_names_value if isinstance(history_names_value, dict) else {}
except (OSError, json.JSONDecodeError):
    history_names = {}
result_files = sorted(RESULT_DIR.glob("result*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
pending_result = st.session_state.pop("pending_selected_result", None)

meta_by_result = {}
for meta_path in HISTORY_DIR.glob("*.meta.json"):
    meta = read_history_meta(meta_path)
    result_name = meta.get("result_file") if isinstance(meta, dict) else None
    if isinstance(result_name, str) and Path(result_name).name:
        meta_id = meta_path.name[:-len(".meta.json")]
        if meta_id and all(char.isalnum() or char in "_-" for char in meta_id):
            meta_by_result[Path(result_name).name] = (meta_id, meta)

history_items = []
for result_path in result_files:
    try:
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        result_data = loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        result_data = {}
    modified_timestamp = result_path.stat().st_mtime
    meta_id, meta = meta_by_result.get(result_path.name, (result_path.stem, {}))
    created_at = meta.get("created_at") if isinstance(meta.get("created_at"), str) else None
    if not created_at:
        created_at = datetime.fromtimestamp(modified_timestamp).astimezone().isoformat(timespec="seconds")
        sort_timestamp = modified_timestamp
    else:
        try:
            sort_timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            created_at = datetime.fromtimestamp(modified_timestamp).astimezone().isoformat(timespec="seconds")
            sort_timestamp = modified_timestamp
    brief = result_data.get("brief") if isinstance(result_data.get("brief"), dict) else {}
    niche_value = meta.get("niche_choice") or result_data.get("niche_category") or brief.get("niche_category") or "Niche not recorded"
    thumb_path = HISTORY_DIR / f"{meta_id}.jpg"
    if not thumb_path.is_file() and not DEMO_MODE:
        generated_thumb = save_legacy_test_thumbnail(result_path.name, ROOT, HISTORY_DIR)
        if generated_thumb is not None:
            thumb_path = generated_thumb
    history_items.append({
        "result_file": result_path.name,
        "result_path": str(result_path),
        "result": result_data,
        "meta_id": meta_id,
        "metadata": meta,
        "display_name": history_display_name(result_path.name, meta, history_names, modified_timestamp),
        "created_at": created_at,
        "date_label": format_history_date(created_at),
        "sort_timestamp": sort_timestamp,
        "niche": niche_value,
        "tier": result_data.get("reach_tier"),
        "core_score": result_data.get("core_resonance"),
        "incomplete": is_incomplete(result_data),
        "thumbnail_path": thumb_path if thumb_path.is_file() else None,
    })
history_items = history_items_newest_first(history_items)
option_paths = [item["result_path"] for item in history_items]

selected_result_path = pending_result if pending_result in option_paths else st.session_state.get("selected_result_path")
if selected_result_path not in option_paths:
    newest_complete = next((item for item in history_items if not item["incomplete"]), None)
    selected_result_path = newest_complete["result_path"] if newest_complete else None
if selected_result_path is None:
    st.session_state.pop("selected_result_path", None)
else:
    st.session_state["selected_result_path"] = selected_result_path

active_run = run_is_active(st.session_state)
upload_panel = st.container(key="upload-panel", border=True)
selected_file = Path(selected_result_path) if selected_result_path and not active_run else None
selected_history_item = next(
    (item for item in history_items if selected_file and item["result_path"] == str(selected_file)), None
)
result = selected_history_item.get("result") if selected_history_item else None
if selected_file is not None and not isinstance(result, dict):
    result = None

brief_data = result.get("brief") if isinstance(result, dict) and isinstance(result.get("brief"), dict) else {}
niche = (result or {}).get("niche_category") or brief_data.get("niche_category") or "Niche not recorded"
niche_reason = (result or {}).get("niche_reason") or brief_data.get("niche_reason", "")
default_niche_choice = niche if niche in NICHE_CATEGORIES else "Auto-detect"
selected_result_identity = str(selected_file) if selected_file else (
    st.session_state.get("selected_result_path", "__no_result__") if active_run else "__no_result__"
)
if st.session_state.get("niche_control_result") != selected_result_identity:
    st.session_state["niche_choice"] = default_niche_choice
    st.session_state["custom_niche_text"] = ""
    st.session_state["niche_control_result"] = selected_result_identity

with upload_panel:
    st.markdown(
        '<div class="vs-step-title"><span>1</span>Upload your reel</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload an MP4 or MOV reel",
        type=["mp4", "mov"],
        max_upload_size=100,
        disabled=active_run,
        label_visibility="collapsed",
    )
    if DEMO_MODE:
        st.caption("Demo mode replays a saved result locally. The uploaded file is not saved or sent.")
    else:
        st.caption("Only upload public or your own reels. Videos are processed by Google's Gemini on the free tier.")
    st.markdown(
        '<div class="vs-step-title vs-context-title"><span>2</span>Add context (optional)'
        '<small class="vs-step-optional">optional</small></div>',
        unsafe_allow_html=True,
    )
    user_context = st.text_area(
        "Add context (optional)",
        disabled=active_run,
        placeholder="e.g. brand collab, celebrity in the video, account size",
        height=68,
        max_chars=600,
        key="creator_context",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="vs-step-title"><span>3</span>Pick the niche</div>',
        unsafe_allow_html=True,
    )
    niche_choice = st.pills(
        "Niche",
        options=NICHE_OPTIONS,
        selection_mode="single",
        default=default_niche_choice,
        key="niche_choice",
        disabled=active_run,
        wrap=True,
    )
    st.markdown(
        f'<div class="vs-niche-hint">{escape(niche_hint(niche_choice), quote=True)}</div>',
        unsafe_allow_html=True,
    )
    custom_text = None
    if niche_choice == "Other (describe)":
        custom_text = st.text_input(
            "Describe your niche, e.g. food recipes, tech news",
            max_chars=60,
            key="custom_niche_text",
            disabled=active_run,
        )
        st.caption("No dedicated rubric for this niche yet. The general rubric is used and your description is passed as context.")
    niche_choice_result = resolve_niche_choice(niche_choice, custom_text)
    if "error" in niche_choice_result:
        st.caption(niche_choice_result["error"])

    upload_error = None
    if uploaded_file is not None:
        upload_error = validate_upload(uploaded_file.name, uploaded_file.size)
        if upload_error:
            st.error(upload_error)

    if not active_run and not st.session_state.get("run_confirm_pending"):
        if st.button(
            "Analyze a reel",
            type="primary",
            disabled=(uploaded_file is None or upload_error is not None or "error" in niche_choice_result),
            key="analyze_reel",
        ):
            st.session_state["run_confirm_pending"] = True

    if st.session_state.get("run_confirm_pending") and not active_run:
        if DEMO_MODE:
            st.info("This replays a saved result locally. No API calls or upload writes will occur.")
        else:
            st.info("This run uses about 53 Gemini calls. Free tier has a daily limit.")
        if st.button("Start", type="primary", key="confirm_start"):
            if run_is_active(st.session_state):
                st.warning("A run is already in progress.")
            elif uploaded_file is None or upload_error is not None or "error" in niche_choice_result:
                st.error(upload_error or niche_choice_result.get("error") or "Choose a valid reel before starting.")
                st.session_state["run_confirm_pending"] = False
            elif DEMO_MODE:
                demo = DemoRunner(
                    RESULT_DIR,
                    stage_seconds=demo_stage_seconds_from_env(),
                    mode=os.environ.get("VIRALITY_DEMO_MODE", "ok"),
                    result_name=os.environ.get("VIRALITY_DEMO_RESULT"),
                )
                try:
                    started = demo.start()
                except (OSError, ValueError) as exc:
                    st.error(f"Could not start demo replay: {escape(type(exc).__name__)}: {escape(str(exc))}")
                    started = False
                if started:
                    st.session_state["demo_runner"] = demo
                    st.session_state["run_process"] = demo
                    st.session_state["run_active"] = True
                    st.session_state["run_result_path"] = str(demo.result_path) if demo.result_path else None
                    st.session_state["last_run_result_path"] = None
                    st.session_state["last_run_log_path"] = None
                    st.session_state["last_run_failed"] = False
                    st.session_state["last_run_error_count"] = 0
                    st.session_state.pop("demo_log_tail", None)
                    st.session_state.pop("demo_result_payload", None)
                    st.session_state["run_confirm_pending"] = False
                    st.session_state.pop("pending_selected_result", None)
                    st.rerun()
            else:
                video_name = generate_safe_filename(uploaded_file.name)
                run_id = Path(video_name).stem
                upload_dir = RESULT_DIR / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                video_path = upload_dir / video_name
                video_path.write_bytes(uploaded_file.getvalue())

                context_parts = [str(user_context or "").strip(), niche_choice_result.get("extra_context") or ""]
                context_text = "\n\n".join(part for part in context_parts if part)
                context_file = RESULT_DIR / f"context_ui_{run_id}.txt"
                context_file.write_text(context_text, encoding="utf-8")
                output_path = RESULT_DIR / f"result_ui_{run_id}.json"
                log_path = RESULT_DIR / f"pipeline_ui_{run_id}.log"
                original_display_filename = Path(str(uploaded_file.name)).name
                try:
                    write_history_meta(
                        HISTORY_DIR,
                        run_id,
                        {
                            "original_upload_filename": original_display_filename,
                            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "niche_choice": niche_choice,
                            "custom_display_name": Path(original_display_filename).stem,
                            "result_file": output_path.name,
                        },
                    )
                except OSError as exc:
                    st.error(f"Could not save history metadata: {escape(type(exc).__name__)}: {escape(str(exc))}")
                    st.stop()
                save_video_thumbnail(video_path, HISTORY_DIR / f"{run_id}.jpg")
                command = build_pipeline_command(
                    sys.executable,
                    video_path,
                    output_path,
                    niche=niche_choice_result.get("niche_override"),
                    context_file=context_file,
                )
                st.session_state["run_result_path"] = str(output_path)
                st.session_state["last_run_result_path"] = str(output_path)
                st.session_state["last_run_log_path"] = str(log_path)
                st.session_state["last_run_failed"] = False
                st.session_state["last_run_error_count"] = 0
                st.session_state["run_confirm_pending"] = False
                try:
                    started = start_subprocess(st.session_state, command, log_path, ROOT)
                except OSError as exc:
                    st.error(f"Could not start the pipeline: {type(exc).__name__}: {exc}")
                    started = False
                if started:
                    st.rerun()


if not active_run:
    st.markdown(
        '<div class="vs-history-heading">Your history'
        f'<span class="vs-history-count">{escape(str(len(history_items)), quote=True)}</span></div>',
        unsafe_allow_html=True,
    )
    show_incomplete = st.toggle("Show incomplete runs", key="history_show_incomplete")
    visible_history = filter_history_items(history_items, show_incomplete)
    show_all = st.session_state.get("history_show_all", False)
    shown_history = slice_history_items(visible_history, show_all=show_all, limit=8)
    if len(visible_history) > 8:
        if st.button("Show fewer" if show_all else "Show all", key="history_show_all_button"):
            st.session_state["history_show_all"] = not show_all
            st.rerun()

    if shown_history:
        tile_columns = st.columns(3, gap="small")
        for index, item in enumerate(shown_history):
            with tile_columns[index % 3]:
                widget_key = history_tile_widget_key(item["result_file"])
                selected = item["result_path"] == st.session_state.get("selected_result_path")
                with st.container(key=widget_key, border=True):
                    thumbnail_path = item.get("thumbnail_path")
                    if thumbnail_path:
                        st.image(
                            str(thumbnail_path), width="stretch",
                            caption=None,
                        )
                    else:
                        st.markdown(history_placeholder_html(item["display_name"]), unsafe_allow_html=True)
                    st.markdown(history_tile_html(item, selected=selected), unsafe_allow_html=True)
                    open_key = widget_key.replace("history_tile_", "history_open_", 1)
                    if st.button("View result", key=open_key, on_click=_select_history_result,
                                 args=(item["result_path"],), use_container_width=True):
                        pass
                    rename_key = widget_key.replace("history_tile_", "history_rename_", 1)
                    if st.button("✎", key=rename_key, help="Rename this result"):
                        _begin_history_rename(item["result_file"], item["display_name"])
                    if st.session_state.get("history_edit_result_file") == item["result_file"]:
                        with st.form(f"history_rename_form_{widget_key}"):
                            renamed = st.text_input(
                                "Display name", key="history_rename_value", max_chars=60,
                            )
                            save_name, cancel_name = st.columns(2)
                            with save_name:
                                save_clicked = st.form_submit_button("Save name")
                            with cancel_name:
                                cancel_clicked = st.form_submit_button("Cancel")
                        if save_clicked:
                            clean_name = sanitize_history_name(renamed)
                            if not clean_name:
                                st.warning("Enter a name first.")
                            else:
                                rename_metadata = {"custom_display_name": clean_name}
                                if not history_meta_path(HISTORY_DIR, item["meta_id"]).is_file():
                                    rename_metadata["result_file"] = item["result_file"]
                                write_history_meta(
                                    HISTORY_DIR,
                                    item["meta_id"],
                                    rename_metadata,
                                )
                                st.session_state.pop("history_edit_result_file", None)
                                st.rerun()
                        if cancel_clicked:
                            st.session_state.pop("history_edit_result_file", None)
                            st.rerun()
    elif not history_items:
        st.caption("Your analyzed reels will appear here.")


def _monitor_demo(demo):
    status = demo.poll()
    stage = demo.current_stage()
    elapsed = demo.elapsed_seconds()
    progress_copy = run_progress_copy(stage, elapsed, fallback_stage=FALLBACK_PROGRESS_TEXT)
    st.markdown(progress_html(stage_pill_states(stage)), unsafe_allow_html=True)
    st.progress(demo.progress(), text=progress_copy["waiting_message"])
    st.caption(progress_copy["duration_caption"])
    if progress_copy["long_run_message"]:
        st.info(progress_copy["long_run_message"])
    st.caption(progress_copy["stage_caption"])

    if status is None:
        if st.button("Cancel", key="cancel_pipeline"):
            demo.cancel()
            st.session_state["run_active"] = False
            st.session_state["run_process"] = None
            st.session_state["last_run_failed"] = True
            st.session_state["last_run_result_path"] = None
            st.session_state["last_run_log_path"] = None
            st.session_state["last_run_error_count"] = 0
            st.session_state["demo_log_tail"] = demo.log_tail()
            st.rerun()
        return

    st.session_state["run_active"] = False
    st.session_state["run_process"] = None
    st.session_state["last_run_log_path"] = None
    st.session_state["last_run_result_path"] = None
    if demo.cancelled:
        st.session_state["last_run_failed"] = True
        st.session_state["last_run_error_count"] = 0
        st.session_state["demo_log_tail"] = demo.log_tail()
    elif demo.mode == "incomplete":
        st.session_state["last_run_failed"] = True
        st.session_state["last_run_error_count"] = 3
        st.session_state["demo_result_payload"] = demo.result()
        st.session_state["demo_log_tail"] = demo.log_tail()
    else:
        st.session_state["last_run_failed"] = False
        st.session_state["last_run_error_count"] = 0
        result_path = str(demo.result_path)
        st.session_state["last_run_result_path"] = result_path
        st.session_state["demo_replay_result_path"] = result_path
        st.session_state["pending_selected_result"] = result_path
        st.session_state.pop("demo_log_tail", None)
    st.rerun()


def _monitor_pipeline():
    if not st.session_state.get("run_active"):
        return
    process = st.session_state.get("run_process")
    if process is None:
        st.session_state["run_active"] = False
        return

    if isinstance(process, DemoRunner):
        _monitor_demo(process)
        return

    log_path = st.session_state.get("run_log_path", "")
    log_text = read_log_text(log_path)
    stage = current_pipeline_stage(log_text)
    st.markdown(progress_html(stage_pill_states(stage)), unsafe_allow_html=True)
    elapsed = elapsed_run_seconds(st.session_state)
    progress_pct = stage_progress_percent(stage)
    progress_copy = run_progress_copy(stage, elapsed, fallback_stage=FALLBACK_PROGRESS_TEXT)
    st.progress(progress_pct, text=progress_copy["waiting_message"])
    st.caption(progress_copy["duration_caption"])
    if progress_copy["long_run_message"]:
        st.info(progress_copy["long_run_message"])
    st.caption(progress_copy["stage_caption"])

    if process.poll() is None:
        if st.button("Cancel", key="cancel_pipeline"):
            try:
                process.terminate()
            except OSError:
                pass
            st.session_state["run_active"] = False
            st.session_state["run_process"] = None
            st.session_state["last_run_failed"] = True
            run_result_path = st.session_state.get("run_result_path")
            result_exists = bool(run_result_path and Path(run_result_path).is_file())
            st.session_state["last_run_result_path"] = run_result_path if result_exists else None
            st.session_state["last_run_log_path"] = str(log_path)
            st.session_state["last_run_error_count"] = 0
            if result_exists:
                st.session_state["pending_selected_result"] = run_result_path
            st.rerun()
        return

    output_path = Path(st.session_state.get("run_result_path", ""))
    process_failed = process.returncode != 0 or not output_path.is_file()
    error_count = 0
    if output_path.is_file():
        try:
            output_result = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(output_result, dict):
                errors = output_result.get("errors")
                error_count = len(errors) if isinstance(errors, list) else 0
                process_failed = process_failed or is_incomplete(output_result)
            else:
                process_failed = True
        except (OSError, json.JSONDecodeError):
            process_failed = True

    st.session_state["run_active"] = False
    st.session_state["run_process"] = None
    st.session_state["last_run_failed"] = process_failed
    st.session_state["last_run_error_count"] = error_count
    st.session_state["last_run_log_path"] = str(log_path)
    st.session_state["last_run_result_path"] = str(output_path) if output_path.is_file() else None
    if output_path.is_file():
        st.session_state["pending_selected_result"] = str(output_path)
    st.rerun()


if active_run:
    with st.expander("Play while you wait", expanded=True):
        st.components.v1.html(GAME_HTML, height=260, scrolling=False)

monitor = st.fragment(run_every=2 if active_run else None)(_monitor_pipeline)
monitor()

if not active_run and st.session_state.get("last_run_failed"):
    st.warning(f"Incomplete run, rerun needed. Errors: {st.session_state.get('last_run_error_count', 0)}")
    log_tail = st.session_state.get("demo_log_tail") or last_log_lines(
        st.session_state.get("last_run_log_path", ""), 3
    )
    if log_tail:
        st.code(log_tail)
    failed_result_path = st.session_state.get("last_run_result_path")
    if selected_file is None or failed_result_path is None or str(selected_file) == failed_result_path:
        st.stop()

if result is None:
    if not result_files:
        st.info("No saved result files found in tmp/. Upload a reel to create one.")
    st.stop()

if is_incomplete(result):
    errors = result.get("errors")
    error_count = len(errors) if isinstance(errors, list) else 0
    st.warning(f"Incomplete run, rerun needed. Errors: {error_count}")
    if str(selected_file) == st.session_state.get("last_run_result_path"):
        log_tail = last_log_lines(st.session_state.get("last_run_log_path", ""), 3)
        if log_tail:
            st.code(log_tail)
    st.stop()

result_columns = st.columns([1.55, 1], wrap=True)
with result_columns[0]:
    st.markdown(
        result_hero_html(
            result.get("reach_tier", "Tier unavailable"), niche,
            result.get("core_resonance"), niche_reason,
            demo_replay=(
                DEMO_MODE
                and str(selected_file) == st.session_state.get("demo_replay_result_path")
            ),
        ),
        unsafe_allow_html=True,
    )
with result_columns[1]:
    st.markdown(
        score_card_html("Core audience", result.get("core_resonance"),
                        "How strongly the core audience connects with the reel."),
        unsafe_allow_html=True,
    )
    st.markdown(
        wider_audience_card_html(wider_audience_label(result.get("crossover_score"))),
        unsafe_allow_html=True,
    )
    st.caption("Labels are rough buckets; small differences between reels are not meaningful.")
st.caption("Scores move by about 0.8 between identical runs.")

st.markdown('<div class="vs-section-label">Suggestions to strengthen the reel</div>', unsafe_allow_html=True)
suggestion_columns = st.columns(3, wrap=True)
suggestions = result.get("suggestions") or []
for index, column in enumerate(suggestion_columns):
    with column:
        if index < len(suggestions) and isinstance(suggestions[index], dict):
            item = suggestions[index]
            st.markdown(
                suggestion_card_html(
                    item.get("criterion", ""),
                    item.get("suggestion", "Suggestion unavailable"),
                    item.get("quote", ""),
                    item.get("persona_id", ""),
                ),
                unsafe_allow_html=True,
            )

st.markdown('<div class="vs-section-label">Audience Wall</div>', unsafe_allow_html=True)
audience_personas = collect_scored_personas(result)
if audience_personas:
    summary = audience_summary(audience_personas)
    st.markdown(
        '<div class="vs-audience-summary">'
        f'<div class="vs-audience-summary-lead">{escape(str(summary["score_at_or_above"]), quote=True)} '
        f'of {escape(str(summary["total"]), quote=True)} scored {AUDIENCE_SCORE_THRESHOLD} or above</div>'
        '<div class="vs-audience-counters">'
        f'<div class="vs-audience-counter"><strong>{escape(str(summary["stop_scrolling"]), quote=True)} of {escape(str(summary["total"]), quote=True)}</strong><span>Would stop scrolling</span></div>'
        f'<div class="vs-audience-counter"><strong>{escape(str(summary["watch_to_end"]), quote=True)} of {escape(str(summary["total"]), quote=True)}</strong><span>Would watch to the end</span></div>'
        f'<div class="vs-audience-counter"><strong>{escape(str(summary["might_share"]), quote=True)} of {escape(str(summary["total"]), quote=True)}</strong><span>Might share</span></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    top_ids, low_ids = select_extreme_personas(audience_personas)
    result_identity = str(selected_file) if selected_file else "__saved_result__"
    highest_id = max(audience_personas, key=lambda persona: persona["score"])["persona_id"]
    if (
        st.session_state.get("audience_wall_result_identity") != result_identity
        or st.session_state.get("selected_audience_persona_id") not in {
            persona["persona_id"] for persona in audience_personas
        }
    ):
        st.session_state["selected_audience_persona_id"] = highest_id
        st.session_state["audience_wall_result_identity"] = result_identity

    tile_index = 0
    for archetype_key, row_label, members in group_personas_by_archetype(audience_personas):
        st.markdown(
            f'<div class="vs-wall-row-label">{escape(row_label, quote=True)}</div>',
            unsafe_allow_html=True,
        )
        for offset in range(0, len(members), 8):
            tile_batch = members[offset:offset + 8]
            tile_columns = st.columns(len(tile_batch), wrap=True, gap="small")
            for column, persona in zip(tile_columns, tile_batch):
                persona_id = persona["persona_id"]
                widget_key = audience_tile_widget_key(persona_id, tile_index)
                bucket = score_bucket(persona["score"])
                badges = []
                if persona_id in top_ids:
                    badges.append("top")
                if persona_id in low_ids:
                    badges.append("low")
                selected = persona_id == st.session_state["selected_audience_persona_id"]
                with column:
                    st.markdown(
                        audience_tile_style_html(widget_key, bucket, badges, selected),
                        unsafe_allow_html=True,
                    )
                    st.button(
                        f'{persona.get("archetype_key", "unknown")[:1].upper()} | {persona["score"]:.1f}',
                        key=widget_key,
                        type="primary" if selected else "secondary",
                        width="stretch",
                        on_click=_select_wall_persona,
                        args=(persona_id,),
                    )
                tile_index += 1

    selected_persona = next(
        persona for persona in audience_personas
        if persona["persona_id"] == st.session_state["selected_audience_persona_id"]
    )
    sound_mode = st.radio(
        "Audio condition",
        options=("sound_on", "sound_off"),
        format_func=lambda value: "Sound on" if value == "sound_on" else "Sound off",
        horizontal=True,
        label_visibility="collapsed",
        key="audience_wall_sound_mode",
    )
    reaction = selected_persona.get(sound_mode) or {}
    full_quote = reaction.get("reason") if isinstance(reaction, dict) else ""
    full_quote = full_quote.strip() if isinstance(full_quote, str) else ""
    display_quote = shorten_first_sentence(full_quote) or "No quote is available for this setting."
    st.markdown(
        audience_spotlight_html(selected_persona, sound_mode, display_quote),
        unsafe_allow_html=True,
    )
    with st.expander("Read more"):
        st.markdown(escape(full_quote or "No full quote is available.", quote=True), unsafe_allow_html=True)
else:
    st.info("No scored audience personas are available in this result.")

averages = archetype_averages(result)
archetype_labels = {
    "genz": "Gen Z scroller",
    "casual": "Casual viewer",
    "niche_fan": "Niche fan",
    "skeptic": "Skeptic",
    "creator": "Creator peer",
    "marketer": "Brand marketer",
}
st.markdown('<div class="vs-section-label">Archetype heat strip</div>', unsafe_allow_html=True)
for archetype_key, archetype_label in archetype_labels.items():
    st.markdown(heat_bar_html(archetype_label, averages.get(archetype_key)), unsafe_allow_html=True)

with st.expander("Details", expanded=False):
    st.markdown(
        wider_audience_raw_score_html(result.get("crossover_score")),
        unsafe_allow_html=True,
    )
    st.markdown("**Dimension means**")
    st.json(result.get("dimension_means", {}))
    for wave_name in ("wave1", "wave2"):
        wave = result.get(wave_name)
        if not isinstance(wave, list):
            continue
        st.markdown(f"**{wave_name.replace('wave', 'Wave ')} scores**")
        st.dataframe(
            [{"persona_id": item.get("persona_id"), "score": item.get("score")}
             for item in wave if isinstance(item, dict)],
            hide_index=True,
            alt=f"{wave_name} persona scores",
        )
    st.markdown("**Per-persona scores**")
    st.json(result.get("per_persona_scores", {}))

with st.expander("Limitations"):
    for point in LIMITATIONS:
        st.markdown(f"- {point}")

st.markdown(
    '<footer class="vs-footer">Estimate of potential, not a prediction or a guarantee.</footer>',
    unsafe_allow_html=True,
)
