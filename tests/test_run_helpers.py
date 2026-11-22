import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from run_helpers import (
    DemoRunner,
    MAX_UPLOAD_BYTES,
    build_pipeline_command,
    current_pipeline_stage,
    demo_stage_seconds_from_env,
    generate_safe_filename,
    redact_log_text,
    run_is_active,
    save_legacy_test_thumbnail,
    start_subprocess,
    validate_upload,
)
from ui_helpers import LONG_RUN_WARNING, run_progress_copy


def test_generated_filename_uses_uuid_and_original_extension_only():
    filename = generate_safe_filename(r"C:\private\folder\my reel.MOV")
    assert re.fullmatch(r"[0-9a-f]{32}\.mov", filename)
    assert "my reel" not in filename and "private" not in filename


@pytest.mark.parametrize(
    "filename,size,error_fragment",
    [
        ("clip.avi", 10, "Only MP4 and MOV"),
        ("clip.mp4", MAX_UPLOAD_BYTES + 1, "100 MB"),
        ("clip.MOV", MAX_UPLOAD_BYTES, None),
    ],
)
def test_upload_extension_and_size_validation(filename, size, error_fragment):
    error = validate_upload(filename, size)
    if error_fragment:
        assert error_fragment in error
    else:
        assert error is None


def test_pipeline_command_is_an_argv_list_with_optional_arguments():
    command = build_pipeline_command(
        r"C:\venv\python.exe", "tmp/uploads/id.mp4", "tmp/result.json",
        niche="product_marketing", context_file="tmp/context.txt",
    )
    assert command == [
        r"C:\venv\python.exe", "-m", "core.pipeline", "tmp/uploads/id.mp4",
        "--niche", "product_marketing", "--context-file", "tmp/context.txt",
        "--out", "tmp/result.json",
    ]


def test_redact_log_text_masks_gemini_keys_and_labeled_api_keys():
    redacted = redact_log_text(
        "provider returned AIzaSyExampleSecretValue API_KEY=another-secret"
    )
    assert "AIza" not in redacted
    assert "ExampleSecretValue" not in redacted
    assert "another-secret" not in redacted
    assert redacted.count("[REDACTED]") == 2


def test_current_stage_uses_existing_pipeline_stage_markers():
    assert current_pipeline_stage("Model call stage=vision_brief model=test") == "Reading reel"
    assert current_pipeline_stage("Model call stage=persona:p1:sound_on model=test") == "Audience waves"
    assert current_pipeline_stage("no stage markers") is None


def test_start_subprocess_guards_against_two_active_runs(tmp_path):
    class FakeProcess:
        def poll(self):
            return None

    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    state = {}
    log_path = tmp_path / "run.log"
    assert start_subprocess(state, ["python", "-m", "core.pipeline"], log_path, tmp_path, fake_popen)
    assert run_is_active(state)
    assert not start_subprocess(state, ["python", "-m", "core.pipeline"], log_path, tmp_path, fake_popen)
    assert len(calls) == 1
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["stderr"] is not None


def _write_demo_result(result_dir, name="result-demo.json", complete=True):
    result_path = result_dir / name
    result_path.write_text(json.dumps({
        "complete": complete,
        "final_score": 7.1 if complete else None,
        "reach_tier": "Medium",
    }), encoding="utf-8")
    return result_path


def test_demo_runner_advances_through_stages_on_fake_clock_without_history_writes(tmp_path):
    result_path = _write_demo_result(tmp_path)
    now = [100.0]
    runner = DemoRunner(tmp_path, stage_seconds=2, clock=lambda: now[0])
    assert runner.start()
    assert runner.result_path == result_path
    assert runner.current_stage() == "Reading reel"
    assert runner.progress() == 10
    for index, expected in enumerate(("Core audience", "Gate", "Wider audience", "Scoring"), start=1):
        now[0] = 100.0 + 2 * index
        assert runner.poll() is None
        assert runner.current_stage() == expected
    now[0] = 110.0
    assert runner.poll() == 0
    assert runner.finished()
    assert not (tmp_path / "history").exists()


@pytest.mark.parametrize("mode", ["ok", "cancel_test"])
def test_demo_runner_normal_modes_replay_saved_complete_result(tmp_path, mode):
    expected_path = _write_demo_result(tmp_path)
    now = [0.0]
    runner = DemoRunner(tmp_path, stage_seconds=1, mode=mode, clock=lambda: now[0])
    assert runner.start()
    now[0] = 5.0
    assert runner.poll() == 0
    assert runner.finished()
    assert runner.result_path == expected_path
    assert runner.result()["final_score"] == 7.1


def test_demo_runner_uses_newest_complete_or_explicit_saved_result(tmp_path):
    older = _write_demo_result(tmp_path, "result-old.json")
    incomplete_newer = _write_demo_result(tmp_path, "result-new.json", complete=False)
    os.utime(older, (10, 10))
    os.utime(incomplete_newer, (20, 20))
    default_runner = DemoRunner(tmp_path, clock=lambda: 0.0)
    assert default_runner.start()
    assert default_runner.result_path == older

    named_runner = DemoRunner(tmp_path, result_name=older.name, clock=lambda: 0.0)
    assert named_runner.start()
    assert named_runner.result_path == older
    with pytest.raises(ValueError, match="not a complete saved result"):
        DemoRunner(tmp_path, result_name=incomplete_newer.name).start()


def test_demo_runner_incomplete_mode_has_fake_three_line_log_and_no_score(tmp_path):
    now = [0.0]
    runner = DemoRunner(tmp_path, stage_seconds=1, mode="incomplete", clock=lambda: now[0])
    assert runner.start()
    now[0] = 5.0
    assert runner.poll() == 1
    assert runner.finished()
    assert runner.result()["final_score"] is None
    assert len(runner.log_tail().splitlines()) == 3
    assert all(line.startswith("DEMO") for line in runner.log_tail().splitlines())
    assert runner.result_path is None
    assert not (tmp_path / "history").exists()


def test_demo_runner_slow_mode_scales_time_and_stays_on_first_stage(tmp_path):
    _write_demo_result(tmp_path)
    now = [0.0]
    runner = DemoRunner(tmp_path, stage_seconds=6, mode="slow", clock=lambda: now[0])
    assert runner.start()
    now[0] = 1.6
    assert runner.elapsed_seconds() > 26 * 60
    assert runner.poll() is None
    assert runner.current_stage() == "Reading reel"
    assert runner.progress() == 10
    assert run_progress_copy(runner.current_stage(), runner.elapsed_seconds())["long_run_message"] == LONG_RUN_WARNING
    assert not runner.finished()


def test_demo_stage_seconds_environment_has_default_and_minimum():
    assert demo_stage_seconds_from_env({}) == 6
    assert demo_stage_seconds_from_env({"VIRALITY_DEMO_STAGE_SEC": "0"}) == 1
    assert demo_stage_seconds_from_env({"VIRALITY_DEMO_STAGE_SEC": "2.5"}) == 2.5


def test_demo_runner_does_not_import_core_llm_or_spawn_a_process(tmp_path):
    result_path = _write_demo_result(tmp_path)
    script = f'''\
import sys
from pathlib import Path
import run_helpers
assert "core.llm" not in sys.modules
def no_spawn(*args, **kwargs):
    raise AssertionError("DemoRunner attempted to spawn a subprocess")
run_helpers.subprocess.Popen = no_spawn
run_helpers.subprocess.run = no_spawn
runner = run_helpers.DemoRunner(Path({str(tmp_path)!r}), stage_seconds=1, clock=lambda: 10.0)
assert runner.start()
assert runner.poll() is None
assert runner.result_path == Path({str(result_path)!r})
assert "core.llm" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert not (tmp_path / "history").exists()


def test_legacy_test_thumbnail_uses_exact_result_number_and_skips_silently(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    history_dir = tmp_path / "tmp" / "history"
    project_root.mkdir()
    (project_root / "test10.mp4").write_bytes(b"demo fixture")
    calls = []

    def fake_thumbnail(source, target, **kwargs):
        calls.append((Path(source), Path(target)))
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"jpeg")
        return True

    monkeypatch.setattr("run_helpers.save_video_thumbnail", fake_thumbnail)
    assert save_legacy_test_thumbnail("result10.json", project_root, history_dir) == history_dir / "result10.jpg"
    assert save_legacy_test_thumbnail("result10_suffix.json", project_root, history_dir) is None
    assert save_legacy_test_thumbnail("result1.json", project_root, history_dir) is None
    assert len(calls) == 1
    assert calls[0][0] == project_root / "test10.mp4"
