"""Safe local helpers for launching and monitoring the pipeline subprocess."""

import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from ui_helpers import RUN_STAGES, is_incomplete, sanitize_history_name, stage_progress_percent


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov"}
FALLBACK_PROGRESS_TEXT = "Reading reel, audience waves and scoring..."
DEMO_TIME_SCALE = 1000
DEMO_INCOMPLETE_LOG = (
    "DEMO LOG (simulated)\n"
    "DEMO: audience replay stopped before scoring\n"
    "DEMO: no pipeline or model calls were made"
)


def demo_stage_seconds_from_env(environ=None):
    """Read demo stage duration, clamping valid values to at least one second."""
    environ = os.environ if environ is None else environ
    try:
        seconds = float(environ.get("VIRALITY_DEMO_STAGE_SEC", "6"))
    except (TypeError, ValueError):
        return 6.0
    return max(1.0, seconds) if math.isfinite(seconds) else 6.0


class DemoRunner:
    """Clock-driven replay runner; it reads saved results and never launches a process."""

    def __init__(self, result_dir, stage_seconds=6, mode="ok", result_name=None, clock=None):
        self.result_dir = Path(result_dir)
        self.stage_seconds = max(1.0, float(stage_seconds))
        self.mode = mode if mode in {"ok", "incomplete", "slow", "cancel_test"} else "ok"
        self.result_name = result_name
        self.clock = clock or time.monotonic
        self.started_at = None
        self.stage_index = 0
        self._active = False
        self._finished = False
        self.cancelled = False
        self.result_path = None

    def _choose_result(self):
        files = sorted(
            self.result_dir.glob("result*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if self.result_name:
            requested_name = Path(str(self.result_name)).name
            files = [path for path in files if path.name == requested_name]
        for path in files:
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(result, dict) and not is_incomplete(result):
                return path
            if self.result_name:
                raise ValueError(f"Demo result {requested_name} is not a complete saved result")
        if self.result_name:
            raise FileNotFoundError(f"No complete saved result named {requested_name}")
        raise FileNotFoundError("No complete tmp/result*.json file is available for demo replay")

    def start(self):
        """Begin the simulated run, selecting only an existing saved result."""
        if self._active:
            return False
        if self.mode != "incomplete":
            self.result_path = self._choose_result()
        self.started_at = float(self.clock())
        self.stage_index = 0
        self.cancelled = False
        self._finished = False
        self._active = True
        return True

    def elapsed_seconds(self):
        """Return simulated elapsed seconds, accelerated for the slow scenario."""
        if self.started_at is None:
            return 0.0
        scale = DEMO_TIME_SCALE if self.mode == "slow" else 1.0
        return max(0.0, (float(self.clock()) - self.started_at) * scale)

    def poll(self):
        """Advance stage state and return Popen-like None-while-running status."""
        if not self._active:
            return 1 if self.cancelled or self.mode == "incomplete" else 0
        elapsed = self.elapsed_seconds()
        if self.mode == "slow":
            self.stage_index = 0
            return None
        self.stage_index = min(len(RUN_STAGES) - 1, int(elapsed // self.stage_seconds))
        if elapsed >= self.stage_seconds * len(RUN_STAGES):
            self._active = False
            self._finished = True
            return 1 if self.mode == "incomplete" else 0
        return None

    def current_stage(self):
        self.poll()
        return RUN_STAGES[self.stage_index]

    def progress(self):
        return stage_progress_percent(self.current_stage())

    def cancel(self):
        self.cancelled = True
        self._active = False
        self._finished = True
        return True

    def finished(self):
        self.poll()
        return self._finished or self.cancelled

    @property
    def active(self):
        self.poll()
        return self._active

    def result(self):
        """Read the chosen replay payload or return a no-score incomplete fixture."""
        if self.mode == "incomplete":
            return {"complete": False, "final_score": None, "errors": ["demo"] * 3}
        if self.result_path is None:
            return None
        value = json.loads(self.result_path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None

    def log_tail(self):
        """Return a clearly marked simulated log tail for failure demos."""
        if self.mode == "incomplete" and self._finished:
            return DEMO_INCOMPLETE_LOG
        if self.cancelled:
            return "DEMO LOG (simulated)\nDEMO: replay cancelled\nDEMO: no subprocess was running"
        return ""


def validate_upload(filename, size_bytes):
    """Return a clear validation message, or None when the upload is accepted."""
    extension = Path(str(filename)).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return "Only MP4 and MOV videos are supported."
    if int(size_bytes) > MAX_UPLOAD_BYTES:
        return "Video is larger than the 100 MB upload limit."
    return None


def generate_safe_filename(original_filename):
    """Create a UUID filename preserving only the validated video extension."""
    error = validate_upload(original_filename, 0)
    if error:
        raise ValueError(error)
    extension = Path(str(original_filename)).suffix.lower()
    return f"{uuid.uuid4().hex}{extension}"


def build_pipeline_command(python_executable, video_path, out_path, niche=None, context_file=None,
                           retry_suggestions=False):
    """Build pipeline argv without shell parsing or interpolation."""
    command = [str(python_executable), "-m", "core.pipeline", str(video_path)]
    if niche is not None:
        command.extend(["--niche", str(niche)])
    if context_file is not None:
        command.extend(["--context-file", str(context_file)])
    if retry_suggestions:
        command.append("--retry-suggestions")
    command.extend(["--out", str(out_path)])
    return command


def run_is_active(state):
    """Return whether this session already owns a running pipeline process."""
    process = state.get("run_process")
    return bool(state.get("run_active") and process is not None)


def elapsed_run_seconds(state, now=None):
    """Return internal elapsed seconds for the current run, clamped at zero."""
    started_at = state.get("run_started_at")
    if started_at is None:
        return 0.0
    try:
        started_at = float(started_at)
        current_time = time.monotonic() if now is None else float(now)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, current_time - started_at)


def history_meta_path(history_dir, run_id):
    """Build a safe sidecar path from a run ID, never from an upload name."""
    run_id = str(run_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid history run ID")
    return Path(history_dir) / f"{run_id}.meta.json"


def read_history_meta(meta_path):
    """Read a JSON sidecar as a dict; ignore missing or malformed metadata."""
    try:
        value = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_history_meta(history_dir, run_id, metadata):
    """Merge and write one run sidecar, preserving fields outside the update."""
    meta_path = history_meta_path(history_dir, run_id)
    current = read_history_meta(meta_path) or {}
    updates = dict(metadata)
    if "custom_display_name" in updates:
        updates["custom_display_name"] = sanitize_history_name(updates["custom_display_name"])
    current.update(updates)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current


def save_video_thumbnail(video_path, thumbnail_path, ffmpeg_executable=None):
    """Best-effort first-frame extraction with ffmpeg, then optional OpenCV."""
    try:
        ffmpeg_executable = ffmpeg_executable or shutil.which("ffmpeg")
        thumbnail_path = Path(thumbnail_path)
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        if ffmpeg_executable:
            subprocess.run(
                [
                    str(ffmpeg_executable), "-y", "-ss", "0", "-i", str(video_path),
                    "-frames:v", "1", "-q:v", "3", str(thumbnail_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=8,
            )
            if thumbnail_path.is_file():
                return True
    except Exception:
        pass
    capture = None
    opencv_success = False
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        success, frame = capture.read()
        if success and frame is not None:
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            opencv_success = bool(cv2.imwrite(str(thumbnail_path), frame))
    except Exception:
        pass
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
    if opencv_success:
        return True
    try:
        thumbnail_path.unlink(missing_ok=True)
    except Exception:
        pass
    return False


def save_legacy_test_thumbnail(result_filename, project_root, history_dir):
    """Extract testN.mp4's first frame for an exact resultN.json legacy file."""
    match = re.fullmatch(r"result(\d+)\.json", Path(str(result_filename)).name, re.IGNORECASE)
    if not match:
        return None
    number = match.group(1)
    source = Path(project_root) / f"test{number}.mp4"
    if not source.is_file():
        return None
    target = Path(history_dir) / f"result{number}.jpg"
    if target.is_file() or save_video_thumbnail(source, target):
        return target
    return None


def start_subprocess(state, command, log_path, cwd, popen_factory=None):
    """Start one logged pipeline process, refusing a second active run."""
    if run_is_active(state):
        return False
    if popen_factory is None:
        popen_factory = subprocess.Popen
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = popen_factory(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(cwd),
            shell=False,
        )
    state["run_process"] = process
    state["run_active"] = True
    state["run_started_at"] = time.monotonic()
    state["run_log_path"] = str(log_path)
    state["run_command"] = list(command)
    return True


def redact_log_text(text):
    """Remove Google key-shaped strings, labeled API keys, and bearer tokens."""
    safe = str(text)
    safe = re.sub(r"\bAIza[^\s\"'<>]+", "[REDACTED]", safe)
    safe = re.sub(
        r"(?i)\b((?:[\w.-]*api[\w.-]*key|api\s*key|[\w.-]*token)\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
        r"\1[REDACTED]",
        safe,
    )
    safe = re.sub(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/-]+=*", r"\1[REDACTED]", safe)
    return safe


def read_log_text(log_path):
    try:
        return Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def last_log_lines(log_path, count=3):
    lines = read_log_text(log_path).splitlines()
    return redact_log_text("\n".join(lines[-max(0, int(count)):]))


def current_pipeline_stage(log_text):
    """Map the latest existing model-call stage marker to a safe UI label."""
    stages = re.findall(r"Model call stage=([A-Za-z0-9:_-]+)", str(log_text))
    if not stages:
        return None
    stage = stages[-1]
    if stage in {"audio_brief", "extract", "vision_brief"}:
        return "Reading reel"
    if stage in {"audience_personas", "audience_match"} or stage.startswith("persona:"):
        return "Audience waves"
    if stage == "suggestions":
        return "Scoring and suggestions"
    return None


_PIPELINE_STAGE_LABELS = {
    "reading_reel": "Reading reel",
    "core_audience": "Core audience",
    "gate": "Gate",
    "wider_audience": "Wider audience",
    "scoring": "Scoring",
    "suggestions": "Scoring",
}
_PIPELINE_STAGE_EVENT = re.compile(
    r"Pipeline stage event stage=([a-z_]+) status=(started|completed|incomplete|failed|skipped)"
)


def pipeline_stage_statuses(log_text):
    """Return the latest explicit status for each pipeline stage."""
    statuses = {}
    for stage, status in _PIPELINE_STAGE_EVENT.findall(str(log_text)):
        statuses[stage] = status
    return statuses


def pipeline_active_stage(log_text):
    """Find an explicitly active stage, using model-call logs only as a fallback."""
    active_stage = None
    for stage, status in _PIPELINE_STAGE_EVENT.findall(str(log_text)):
        if status == "started":
            active_stage = stage
        elif active_stage == stage:
            active_stage = None
    if active_stage:
        return _PIPELINE_STAGE_LABELS.get(active_stage)

    api_stage = current_pipeline_stage(log_text)
    if api_stage == "Reading reel":
        return (api_stage if pipeline_stage_statuses(log_text).get("reading_reel")
                not in {"completed", "failed", "skipped"} else None)
    if api_stage == "Audience waves":
        statuses = pipeline_stage_statuses(log_text)
        for key in ("core_audience", "gate", "wider_audience"):
            if statuses.get(key) not in {"completed", "incomplete", "failed", "skipped"}:
                return _PIPELINE_STAGE_LABELS[key]
        return None
    if api_stage == "Scoring and suggestions":
        if pipeline_stage_statuses(log_text).get("suggestions") in {
            "completed", "incomplete", "failed", "skipped"
        }:
            return None
        return "Scoring"
    return None


def pipeline_terminal_status(log_text):
    """Return the pipeline's result/suggestions status recorded at final persistence."""
    matches = re.findall(
        r"Pipeline terminal result_status=(complete|incomplete) "
        r"suggestions_status=(completed|failed|skipped|unknown) "
        r"process_status=(normal|error)", str(log_text),
    )
    if not matches:
        return None
    result_status, suggestions_status, process_status = matches[-1]
    return {"result": result_status, "suggestions": suggestions_status,
            "process": process_status}
