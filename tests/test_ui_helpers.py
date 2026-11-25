import json
import re
import shutil
import subprocess

import pytest
import ast
import importlib
from pathlib import Path

from ui_helpers import (
    AUDIENCE_SCORE_THRESHOLD,
    NICHE_CATEGORIES,
    RUN_STAGES,
    audience_spotlight_html,
    audience_summary,
    archetype_averages,
    collect_scored_personas,
    demo_mode_enabled,
    derive_history_title,
    extract_audience_comments,
    filter_history_items,
    history_items_newest_first,
    history_display_name,
    sanitize_history_name,
    slice_history_items,
    group_personas_by_archetype,
    is_incomplete,
    suggestions_error_count,
    resolve_niche_choice,
    rotating_wait_message,
    score_bucket,
    select_extreme_personas,
    shorten_first_sentence,
    stage_pill_states,
    stage_progress_percent,
    should_show_last_run_warning,
    wider_audience_label,
    sound_score_gap,
    weakest_dimension,
    split_audience_comments,
    selected_result_path,
    select_history_result,
)
from game_html import game_html
from run_helpers import elapsed_run_seconds, read_history_meta, save_legacy_test_thumbnail, write_history_meta
from ui_helpers import LONG_RUN_WARNING, RUN_DURATION_CAPTION, run_progress_copy
from ui_theme import (
    CSS,
    audience_tile_style_html,
    chat_bubble_html,
    dimension_bars_html,
    history_tile_html,
    result_hero_html,
    score_card_html,
    history_placeholder_html,
    heat_bar_html,
    suggestion_card_html,
    demo_banner_html,
    wider_audience_card_html,
    wider_audience_raw_score_html,
    sound_warning_html,
)

NODE = shutil.which("node")


def test_app_ui_imported_symbols_exist():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app_tree = ast.parse(app_path.read_text(encoding="utf-8"))
    for node in app_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in {"ui_theme", "ui_helpers"}:
            module = importlib.import_module(node.module)
            for imported in node.names:
                assert hasattr(module, imported.name), f"{node.module}.{imported.name} is missing"


def test_saved_result_view_has_no_stage_pills():
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    saved_result_source = app_source[app_source.index("result_columns = st.columns"):]
    assert "progress_html(" not in saved_result_source


def test_page_sections_are_in_required_order_and_limitations_are_last():
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    saved_result_source = app_source[app_source.index("result_columns = st.columns"):]
    markers = (
        "result_hero_html(",
        'score_card_html("Core audience"',
        "Why this score",
        "Suggestions to strengthen the reel",
        "Audience Wall",
        'with st.expander("Details"',
        "_render_history()",
        'with st.expander("Limitations"',
        "Estimate of potential, not a prediction or a guarantee.",
    )
    positions = [saved_result_source.index(marker) for marker in markers]
    assert positions == sorted(positions)
    footer_text_position = saved_result_source.index(markers[-1])
    footer_call_position = saved_result_source.rfind("st.markdown(", 0, footer_text_position)
    assert footer_call_position >= 0
    assert saved_result_source[footer_call_position:].count("st.") == 1


@pytest.mark.parametrize(
    "filename,title",
    [("result1.json", "Test 1"), ("result7.json", "Test 7")],
)
def test_history_title_derivation_for_legacy_files(filename, title):
    assert derive_history_title(filename) == title


def test_history_name_prefix_boundaries_prevent_result1_matching_result10_to_12():
    names = {
        "result1": "One",
        "result10": "Ten",
        "result11": "Eleven",
        "result12": "Twelve",
    }
    assert derive_history_title("result1.json", names) == "One"
    assert derive_history_title("result10.json", names) == "Ten"
    assert derive_history_title("result11.json", names) == "Eleven"
    assert derive_history_title("result12.json", names) == "Twelve"
    assert derive_history_title("result1_part.json", names) == "One"


def test_history_name_mapping_uses_longest_prefix_and_meta_custom_name_wins():
    names = {
        "result_ui": "General UI replay",
        "result_ui_4205f564": "Hash browns close-up",
    }
    assert derive_history_title("result_ui_4205f564-extra.json", names) == "Hash browns close-up"
    assert history_display_name(
        "result_ui_4205f564.json",
        {"custom_display_name": "Renamed by user"},
        names,
    ) == "Renamed by user"


def test_history_title_fallbacks_use_test_suffix_or_run_date():
    assert derive_history_title("result14.json", {}) == "Test 14"
    assert derive_history_title("result14_cut.json", {}) == "Test 14 (cut)"
    assert derive_history_title("result14-cut.json", {}) == "Test 14 (cut)"
    assert re.fullmatch(r"Run on \d{4}-\d{2}-\d{2}", derive_history_title("clip.json", {}, 0))
    assert "Earlier run" not in derive_history_title("clip.json", {}, 0)


def test_history_meta_read_write_and_name_sanitization(tmp_path):
    metadata = write_history_meta(tmp_path, "run_123", {
        "result_file": "result_ui_run_123.json",
        "custom_display_name": "  Reel name  ",
    })
    assert metadata["custom_display_name"] == "Reel name"
    assert read_history_meta(tmp_path / "run_123.meta.json") == metadata
    assert sanitize_history_name("  " + ("x" * 70) + "  ") == "x" * 60


def test_history_order_filter_and_latest_eight():
    items = [
        {"name": f"run{index}", "sort_timestamp": index, "incomplete": index == 10}
        for index in range(12)
    ]
    ordered = history_items_newest_first(items)
    assert ordered[0]["name"] == "run11"
    complete = filter_history_items(ordered)
    assert all(not item["incomplete"] for item in complete)
    assert len(filter_history_items(ordered, show_incomplete=True)) == 12
    assert len(slice_history_items(complete)) == 8
    assert len(slice_history_items(complete, show_all=True)) == 11


def test_history_tile_escapes_display_name_from_metadata():
    markup = history_tile_html({
        "display_name": '<img src=x onerror="alert(1)">',
        "niche": "comedy",
        "date_label": "today",
        "tier": "High",
        "core_score": 8,
        "incomplete": False,
    })
    assert "<img src=x" not in markup
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in markup


def test_history_tile_shows_missing_niche_and_placeholder_uses_display_initial():
    markup = history_tile_html({"display_name": "Apple iPhone ad"})
    assert "Niche not recorded" in markup
    assert "Unknown" not in markup
    assert ">A</div>" in history_placeholder_html("Apple iPhone ad")


def test_prefilled_legacy_history_names_are_editable_json_data():
    project_root = Path(__file__).resolve().parents[1]
    names = json.loads((project_root / "data" / "history_names.json").read_text(encoding="utf-8"))
    assert names["result6"] == "Apple x Raw Mango iPhone ad"
    assert names["result_ui_4205f564"] == "Hash browns close-up"
    display_name = history_display_name("result6.json", {}, names)
    escaped_tile = history_tile_html({"display_name": display_name})
    assert "Apple x Raw Mango iPhone ad" in escaped_tile
    unsafe_name = history_display_name("result1.json", {}, {"result1": "<script>bad</script>"})
    unsafe_tile = history_tile_html({"display_name": unsafe_name})
    assert "<script>" not in unsafe_tile
    assert "&lt;script&gt;bad&lt;/script&gt;" in unsafe_tile


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "Narrow"), (3.49, "Narrow"), (3.5, "Moderate"),
        (6.49, "Moderate"), (6.5, "Broad"), (10, "Broad"),
        (None, None), ("not a score", None),
    ],
)
def test_wider_audience_label_thresholds_and_invalid_values(score, expected):
    assert wider_audience_label(score) == expected


def test_wider_audience_card_uses_only_qualitative_label_and_copy():
    markup = wider_audience_card_html(wider_audience_label(7.2))
    assert "Broad" in markup
    assert "Likely to appeal beyond its closest audience." in markup
    assert "7.2" not in markup
    assert "+/-" not in markup
    assert "out of 10" not in markup


def test_wider_audience_raw_score_details_escape_file_text():
    markup = wider_audience_raw_score_html('<script>alert("x")</script>')
    assert "<script>" not in markup
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in markup
    assert "Wider audience (crossover) raw score" in markup


def test_demo_banner_is_present_only_when_enabled():
    enabled = demo_mode_enabled({"VIRALITY_DEMO": "1"})
    disabled = demo_mode_enabled({"VIRALITY_DEMO": "true"})
    default_off = demo_mode_enabled({})
    enabled_markup = demo_banner_html(enabled)
    assert "DEMO MODE: no API calls. Results shown are replays of saved runs." in enabled_markup
    assert disabled is False and default_off is False
    assert demo_banner_html(disabled) == ""


@pytest.mark.parametrize(
    "bucket,color",
    [
        ("low", "#F0EFF3"),
        ("mid_low", "#FCE5F2"),
        ("mid_high", "#ED9BCB"),
        ("high", "linear-gradient(135deg,#E6007E,#A83CF0)"),
    ],
)
def test_audience_tile_theme_has_four_score_buckets_and_top_low_badges(bucket, color):
    markup = audience_tile_style_html("audience_tile_creator_dyn_1_0", bucket, ["top", "low"])
    assert color in markup
    assert "top / low" in markup


def test_audience_summary_counts_threshold_and_weighted_reaction_dimensions():
    personas = collect_scored_personas({
        "wave1": [{
            "persona_id": "genz_dyn_1", "score": 6,
            "sound_on_probability": 0.5,
            "sound_on": {"stop_scrolling": 8, "watch_to_end": 8, "share_likelihood": 4},
            "sound_off": {"stop_scrolling": 4, "watch_to_end": 8, "share_likelihood": 8},
        }],
        "wave2": [{
            "persona_id": "skeptic_break_1", "score": 5.9,
            "sound_on_probability": 1,
            "sound_on": {"stop_scrolling": 6, "watch_to_end": 5, "share_likelihood": 6},
            "sound_off": {"stop_scrolling": 0, "watch_to_end": 0, "share_likelihood": 0},
        }],
    })
    assert AUDIENCE_SCORE_THRESHOLD == 6
    assert audience_summary(personas) == {
        "total": 2,
        "score_at_or_above": 1,
        "stop_scrolling": 2,
        "watch_to_end": 1,
        "might_share": 2,
    }


def test_sound_gap_uses_weighted_wave_one_dimensions_and_finds_weakest():
    result = {
        "dimension_means": {
            "wave1": {
                "sound_on": {
                    "stop_scrolling": 8,
                    "share_likelihood": 6,
                    "watch_to_end": 4,
                    "comment_likelihood": 2,
                },
                "sound_off": {
                    "stop_scrolling": 4,
                    "share_likelihood": 4,
                    "watch_to_end": 4,
                    "comment_likelihood": 4,
                },
            }
        }
    }
    assert sound_score_gap(result) == pytest.approx(1.6)
    assert weakest_dimension(result["dimension_means"], "wave1", "sound_on") == "comment_likelihood"
    assert sound_score_gap({"dimension_means": {}}) is None


def test_new_result_markup_escapes_json_strings_and_marks_weakest_dimension():
    dimensions = {
        "wave1": {"sound_on": {
            "stop_scrolling": 8,
            "share_likelihood": 6,
            "watch_to_end": 4,
            "comment_likelihood": 2,
        }}
    }
    bars = dimension_bars_html(dimensions, "wave1", "sound_on", weakest=True)
    assert bars.count("class=\"vs-weakest-tag\"") == 1
    assert "2.0" in bars and "8.0" in bars
    assert "Loses 1.2 points when muted." in sound_warning_html(1.24)
    assert sound_warning_html(0.99) == ""

def test_score_bar_html_renders_scaled_gradient_fills():
    dimensions = {
        "wave1": {"sound_on": {
            "stop_scrolling": 8.1,
            "share_likelihood": 4.0,
            "watch_to_end": 0,
            "comment_likelihood": 0,
        }}
    }
    bars = dimension_bars_html(dimensions, "wave1", "sound_on")
    assert "width:81%" in bars
    assert "width:40%" in bars
    assert "width:0%" in bars
    assert "background-image:linear-gradient(90deg,#E6007E,#A83CF0)" in bars
    assert "min-width:3px" in bars and "min-width:0" in bars


def test_selection_starts_empty_then_selects_completed_or_clicked_run():
    session = {}
    options = ["old", "new"]
    assert selected_result_path(session, options) is None
    assert "selected_result_path" not in session
    assert selected_result_path(session, options, pending_result="new") == "new"
    assert session["selected_result_path"] == "new"
    clicked_session = {"selected_result_path": "new"}
    select_history_result(clicked_session, "old")
    assert selected_result_path(clicked_session, options) == "old"
    assert clicked_session["selection_scroll_pending"] is True


def test_audience_groups_recognize_dynamic_persona_ids_in_display_order():
    personas = collect_scored_personas({
        "wave1": [
            {"persona_id": "creator_dyn_1", "score": 7},
            {"persona_id": "genz_dyn_1", "score": 6},
        ],
        "wave2": [
            {"persona_id": "skeptic_break_2", "score": 5},
            {"persona_id": "niche_fan_commute_2", "score": 4},
        ],
    })
    groups = group_personas_by_archetype(personas)
    assert [(key, label) for key, label, _ in groups] == [
        ("genz", "Gen Z"), ("skeptic", "Skeptic"),
        ("niche_fan", "Niche fan"), ("creator", "Creator"),
    ]
    assert groups[-1][2][0]["persona_id"] == "creator_dyn_1"


@pytest.mark.parametrize(
    "score,expected",
    [(0, "low"), (2.5, "low"), (2.6, "mid_low"), (5, "mid_low"),
     (5.1, "mid_high"), (7.5, "mid_high"), (7.6, "high"), (10, "high")],
)
def test_audience_score_bucket_boundaries(score, expected):
    assert score_bucket(score) == expected


def test_top_and_low_selection_is_stable_on_ties_and_handles_fewer_than_six():
    personas = [
        {"persona_id": "a", "score": 9},
        {"persona_id": "b", "score": 9},
        {"persona_id": "c", "score": 5},
        {"persona_id": "d", "score": 1},
    ]
    assert select_extreme_personas(personas) == (
        ["a", "b", "c"], ["d", "c", "a"]
    )


def test_first_sentence_shortening_uses_sentence_and_word_boundaries():
    assert shorten_first_sentence("First sentence! Second sentence.") == "First sentence!"
    long_quote = ("This is a long first sentence with many ordinary words " * 5) + ". More text."
    shortened = shorten_first_sentence(long_quote)
    assert len(shortened) <= 160
    assert shortened.endswith("...")
    assert not shortened[:-3].endswith(" ")


def test_audience_spotlight_escapes_quote_and_persona_id():
    markup = audience_spotlight_html(
        {"persona_id": '<img src=x onerror="alert(1)">', "archetype_key": "creator", "score": 7.5},
        "sound_on",
        '<script>alert("x")</script>',
    )
    assert "<img src=x" not in markup
    assert "<script>" not in markup
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in markup


@pytest.mark.parametrize("index", range(5))
def test_stage_pills_mark_done_current_and_upcoming_for_each_stage(index):
    keys = ("reading_reel", "core_audience", "gate", "wider_audience", "scoring")
    statuses = {key: "completed" for key in keys[:index]}
    statuses[keys[index]] = "started"
    states = stage_pill_states(RUN_STAGES[index], statuses)
    assert [state for _, state in states] == [
        "done" if step < index else "current" if step == index else "upcoming"
        for step in range(5)
    ]


@pytest.mark.parametrize("index", range(5))
def test_stage_progress_advances_in_twenty_point_steps_and_half_fills_current(index):
    keys = ("reading_reel", "core_audience", "gate", "wider_audience", "scoring")
    statuses = {key: "completed" for key in keys[:index]}
    statuses[keys[index]] = "started"
    assert stage_progress_percent(RUN_STAGES[index], statuses) == index * 20 + 10


def test_stage_pills_do_not_infer_completed_stages_from_api_marker():
    assert [state for _, state in stage_pill_states("Scoring") ] == [
        "upcoming", "upcoming", "upcoming", "upcoming", "current"
    ]


def test_last_run_warning_is_deduplicated_only_for_same_incomplete_result():
    result = {"complete": False, "final_score": 0.0}
    assert not should_show_last_run_warning(True, "same.json", "same.json", result)
    assert should_show_last_run_warning(True, "same.json", "other.json", result)
    assert should_show_last_run_warning(
        True, "same.json", "same.json", {"complete": True, "final_score": 7.0}
    )


@pytest.mark.parametrize(
    "elapsed,expected",
    [(0, 0), (5.99, 0), (6, 1), (12, 2), (18, 3), (24, 0)],
)
def test_wait_message_changes_deterministically_every_six_seconds(elapsed, expected):
    messages = [
        "The Gen Z scroller is deciding whether to stop",
        "The skeptic is rolling their eyes",
        "The niche fan is taking notes",
        "Counting who would share",
    ]
    assert rotating_wait_message(elapsed) == messages[expected]


def test_waiting_game_html_has_no_network_urls():
    assert "http" not in game_html().lower()


def test_waiting_game_contains_the_short_ignored_obstacle_labels():
    markup = game_html()
    labels_match = re.search(r"const LABELS=\[([^\]]+)\]", markup)
    assert labels_match
    labels = re.findall(r"'([^']+)'", labels_match.group(1))
    assert labels == ["Ad", "Spam", "Promo", "T&C", "Intro", "Sub", "Bio", "Poll"]
    assert all(len(label) <= 5 for label in labels)
    for old_label in ("Unskippable Ad", "Sponsored", "Link in bio", "Terms and Conditions", "Boring caption"):
        assert old_label not in markup


def test_waiting_game_uses_fixed_logical_canvas_resolution():
    markup = game_html()
    assert "const LOGICAL_WIDTH=900,LOGICAL_HEIGHT=240" in markup
    assert 'width="900" height="240"' in markup
    assert "canvas.width=Math.round(LOGICAL_WIDTH*dpr)" in markup
    assert "canvas.height=Math.round(LOGICAL_HEIGHT*dpr)" in markup
    assert "max-width:720px" in markup
    assert "width:100%" in markup
    assert "height:auto" in markup
    assert ".wrap{width:100%;height:auto;min-height:260px" in markup


def test_waiting_game_obstacles_have_fixed_30_unit_width():
    markup = game_html()
    assert "OBSTACLE_WIDTH=30" in markup
    assert "w:OBSTACLE_WIDTH,h:26+Math.random()*14" in markup
    assert "max-width:34" not in markup
    assert "let fontSize" not in markup
    assert "measureText" not in markup


def test_active_progress_copy_has_no_mm_ss_timer_and_keeps_required_messages():
    rendered_copy = run_progress_copy("Audience waves", 25 * 60 + 1)
    visible_text = " ".join(str(value) for value in rendered_copy.values() if value)
    assert not re.search(r"(?<!\d)\d{1,2}:\d{2}(?!\d)", visible_text)
    assert rendered_copy["duration_caption"] == RUN_DURATION_CAPTION
    assert rendered_copy["long_run_message"] == LONG_RUN_WARNING
    assert "Current stage: Audience waves" in visible_text
    assert "Elapsed:" not in visible_text


def test_long_run_message_appears_only_after_twenty_five_minutes():
    assert run_progress_copy("Reading reel", 25 * 60)["long_run_message"] is None
    assert run_progress_copy("Reading reel", 25 * 60 + 0.1)["long_run_message"] == LONG_RUN_WARNING


def test_progress_area_has_no_elapsed_timer_markup():
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "Elapsed:" not in app_source
    assert "_format_elapsed" not in app_source
    assert "vs-elapsed" not in app_source


def test_elapsed_run_time_remains_internal_and_uses_monotonic_start():
    assert elapsed_run_seconds({"run_started_at": 100.0}, now=135.5) == 35.5
    assert elapsed_run_seconds({"run_started_at": 100.0}, now=90.0) == 0.0
    assert elapsed_run_seconds({}) == 0.0


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_waiting_game_spawns_10000_obstacles_with_fair_nonoverlapping_gaps():
    markup = game_html()
    match = re.search(
        r"function nextGap\(speed,jumpAirtime,obstacleWidth,playerWidth,factor\)\s*\{[^{}]*\}",
        markup,
    )
    assert match, "game must expose its gap rule as nextGap"
    simulation = r"""
const assert = require('node:assert/strict');
const startSpeed = 330, capSpeed = startSpeed * 1.4;
const airtime = 0.55, playerWidth = 40;
let previousRight = 0, spawned = 0;
for (let i = 0; i < 10000; i++) {
  const speed = startSpeed + (capSpeed - startSpeed) * i / 9999;
  const obstacleWidth = 30;
  const gap = nextGap(speed, airtime, obstacleWidth, playerWidth);
  const minGap = Math.max(speed * airtime * 0.9, obstacleWidth + playerWidth) + speed * 0.35;
  assert.ok(gap >= minGap - 1e-9, `gap ${gap} below minimum ${minGap} at spawn ${i}`);
  assert.ok(gap <= minGap * 2.2 + 1e-9, `gap ${gap} above limit at spawn ${i}`);
  const nextX = previousRight + gap;
  assert.ok(nextX >= previousRight + minGap - 1e-9, `gap too short at spawn ${i}`);
  assert.ok(nextX >= previousRight, `obstacles overlap at spawn ${i}`);
  previousRight = nextX + obstacleWidth;
  spawned += 1;
}
assert.equal(spawned, 10000);
"""
    subprocess.run(
        [NODE, "-e", match.group(0) + "\n" + simulation],
        check=True,
        capture_output=True,
        text=True,
    )


def test_core_audience_card_displays_dynamic_range_without_scale_suffix():
    from core.scoring import score_interval

    first_scores = [2, 3, 4, 6, 7, 8, 5, 4]
    second_scores = [4, 4, 5, 5, 5, 5, 6, 6]
    first_range = score_interval(first_scores)
    second_range = score_interval(second_scores)
    hero = result_hero_html(
        "Medium", "comedy", 6.25, confidence="High", score_range=first_range
    )
    card = score_card_html(
        "Core audience", sum(first_scores) / len(first_scores), "Core connection",
        confidence="High", score_range=first_range,
    )
    assert f"{first_range[0]:.1f}–{first_range[1]:.1f}" in hero
    assert "6.2 / 10" not in hero and "/ 10" not in hero
    assert "Estimated score is out of 10" in hero
    assert "Confidence: High" in hero
    assert f"{first_range[0]:.1f}–{first_range[1]:.1f}" in card
    assert "/ 10" not in card
    assert first_range != second_range


def test_hero_shows_unavailable_range_and_scale_explanation():
    hero = result_hero_html(
        "Medium", "comedy", 6.25, confidence="High", score_range=None
    )
    assert "Unavailable" in hero
    assert "Estimated score is out of 10" in hero
    assert "Confidence: High" in hero
    assert "6.2 / 10" not in hero


def test_hero_estimate_note_and_confidence_have_readable_separate_placements():
    from core.scoring import score_interval

    score_range = score_interval([2, 3, 4, 6, 7, 8, 5, 4])
    hero = result_hero_html(
        "Medium", "comedy", 6.25, confidence="Medium", score_range=score_range
    )
    assert hero.index("vs-estimate-note") < hero.index("vs-estimate-confidence")
    assert "Estimated score is out of 10" in hero
    assert "Confidence: Medium" in hero
    assert ".vs-estimate-note {" in CSS and "color:#F5EFFA" in CSS
    assert ".vs-estimate-confidence { position:absolute; right:" in CSS
    assert "bottom:1.15rem" in CSS


def test_core_audience_card_shows_unavailable_when_range_is_missing():
    card = score_card_html(
        "Core audience", 5.0, "Core connection", confidence="High", score_range=None
    )
    assert "Unavailable" in card
    assert "requires at least 3 scored personas" in card


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"complete": False, "final_score": 4}, True),
        ({"complete": True, "final_score": None}, True),
        ({"complete": True, "final_score": 0}, False),
        ({"complete": None, "final_score": 4}, False),
    ],
)
def test_incomplete_detection(result, expected):
    assert is_incomplete(result) is expected


def test_suggestions_error_count_ignores_other_stages_and_malformed_items():
    assert suggestions_error_count({"errors": [
        {"stage": "suggestions", "error": "unavailable"},
        {"stage": "persona", "error": "failed"}, None,
        {"stage": "suggestions", "error": "quota"},
    ]}) == 2
    assert suggestions_error_count({"errors": "invalid"}) == 0


def test_legacy_history_score_without_confidence_still_renders():
    markup = history_tile_html({"display_name": "Old run", "core_score": 6.25})
    assert "6.2 / 10" in markup
    assert "Confidence: Unavailable" in markup


def test_archetype_averages_group_wave_scores_by_report_prefix():
    result = {
        "wave1": [
            {"persona_id": "genz_dyn_1", "score": 8},
            {"persona_id": "niche_fan_dyn_1", "score": 4},
        ],
        "wave2": [
            {"persona_id": "genz_break_4", "score": 6},
            {"persona_id": "niche_fan_commute_2", "score": 8},
            {"persona_id": "unknown_dyn_1", "score": 10},
        ],
    }
    averages = archetype_averages(result)
    assert averages["genz"] == 7
    assert averages["niche_fan"] == 6
    assert averages["casual"] is None


def test_suggestion_quote_is_html_escaped():
    markup = suggestion_card_html(
        "hook", "Shorten the intro", "<script>alert(1)</script>", "persona_1"
    )
    assert "<script>" not in markup
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in markup


@pytest.mark.parametrize("niche", NICHE_CATEGORIES)
def test_resolve_niche_choice_for_framework_niches(niche):
    assert resolve_niche_choice(niche) == {
        "niche_override": niche,
        "extra_context": None,
    }


def test_resolve_niche_choice_auto_detect():
    assert resolve_niche_choice("Auto-detect") == {
        "niche_override": None,
        "extra_context": None,
    }


def test_resolve_niche_choice_other_with_text():
    assert resolve_niche_choice("Other (describe)", "  food recipes  ") == {
        "niche_override": "general",
        "extra_context": "Creator describes the niche as: food recipes",
    }


def test_resolve_niche_choice_other_empty_text_returns_error():
    assert "error" in resolve_niche_choice("Other (describe)", "   ")


def test_resolve_niche_choice_other_caps_text_at_60_characters():
    result = resolve_niche_choice("Other (describe)", "x" * 75)
    assert result["extra_context"] == "Creator describes the niche as: " + "x" * 60


def test_resolve_niche_choice_preserves_html_as_context_text():
    result = resolve_niche_choice("Other (describe)", "<script>alert(1)</script>")
    assert result["extra_context"] == "Creator describes the niche as: <script>alert(1)</script>"


def test_extract_audience_comments_collects_wave_and_suggestion_quotes():
    result = {
        "wave1": [{
            "persona_id": "genz_dyn_1",
            "sound_on": {"reason": "Sound-on reaction"},
            "sound_off": {"reason": "Muted reaction"},
        }],
        "wave2": [{
            "persona_id": "skeptic_break_4",
            "sound_on": {"reason": "Wave two reaction"},
        }],
        "suggestions": [{"persona_id": "creator_dyn_1", "quote": "Suggestion quote"}],
    }
    comments = extract_audience_comments(result)
    assert [item["quote"] for item in comments] == [
        "Sound-on reaction", "Muted reaction", "Wave two reaction", "Suggestion quote"
    ]
    assert [item["badge"] for item in comments] == [
        "Wave 1 · sound on", "Wave 1 · sound off", "Wave 2 · sound on", "Suggestion"
    ]
    assert comments[-1]["persona_id"] == "creator_dyn_1"


def test_split_audience_comments_limits_initial_set_and_keeps_remainder():
    comments = [{"quote": f"comment {index}"} for index in range(15)]
    visible, remaining = split_audience_comments(comments)
    assert len(visible) == 12
    assert len(remaining) == 3
    assert visible[0] == comments[0]
    assert remaining[-1] == comments[-1]


def test_chat_bubble_escapes_every_result_text_field():
    comment = {
        "persona_id": '<img src=x onerror="alert(1)">',
        "initial": "G",
        "badge": "Wave <1>",
        "quote": "<script>alert(1)</script>",
        "archetype_key": "genz",
    }
    markup = chat_bubble_html(comment)
    assert '<img src=x' not in markup
    assert '<script>' not in markup
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in markup
    assert "Wave &lt;1&gt;" in markup
