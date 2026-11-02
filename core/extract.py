"""Extract video metadata, sampled frames, and a local transcript."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from core import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "tmp" / "frames"
FFMPEG_HINT = "Install ffmpeg with 'winget install Gyan.FFmpeg', then restart VS Code."


class VideoError(RuntimeError):
    """Raised when a video cannot be inspected or processed."""


def check_media_tools():
    """Ensure ffmpeg and ffprobe are available before video work begins."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise VideoError(f"Missing {', '.join(missing)} on PATH. {FFMPEG_HINT}")


def _run_probe(args, video_path, description):
    try:
        return subprocess.run(
            ["ffprobe", "-v", "error", *args, str(video_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as error:
        raise VideoError(
            f"Could not {description} ({type(error).__name__}: {error}). "
            "Check that the file is a valid video."
        ) from error


def get_duration(video_path):
    """Return video duration in seconds, or raise VideoError for invalid media."""
    check_media_tools()
    result = _run_probe(
        [
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ],
        video_path,
        "read the video duration",
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        raise VideoError("Could not read the video duration. Check that the file is a valid video.") from None
    if duration <= 0:
        raise VideoError("The file has no playable video duration.")
    return duration


def has_audio_track(video_path):
    """Return whether ffprobe finds an audio stream in the video."""
    result = _run_probe(
        ["-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0"],
        video_path,
        "inspect the video's audio track",
    )
    return bool(result.stdout.strip())


def frame_timestamps(duration, n=8):
    """Sample the opening seconds densely and spread remaining samples out."""
    if duration <= 0 or n <= 0:
        return []
    early = [timestamp for timestamp in (0.0, 1.0, 2.0, 3.0) if timestamp < duration]
    remaining = max(0, n - len(early))
    rest = []
    if remaining and duration > 3:
        step = (duration - 3) / (remaining + 1)
        rest = [round(3 + step * (index + 1), 2) for index in range(remaining)]
    return early + rest


def extract_frames(video_path, n=8, out_dir=FRAMES_DIR):
    """Save sampled frames as 512-pixel-wide JPEGs."""
    check_media_tools()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in out_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    duration = get_duration(video_path)
    paths = []
    for index, timestamp in enumerate(frame_timestamps(duration, n)):
        out_path = out_dir / f"frame_{index:02d}_{timestamp:.1f}s.jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(timestamp),
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=512:-1",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, OSError):
            raise VideoError(
                f"Could not extract a frame at {timestamp:.1f} seconds. "
                "Check that the file is a valid video."
            ) from None
        paths.append(str(out_path))
    return paths


def transcribe(video_path, model_size=None):
    """Transcribe speech locally; return empty text if the video has no audio."""
    try:
        check_media_tools()
        if not has_audio_track(video_path):
            return "", ""

        decoded = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "f32le",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        import numpy

        audio = numpy.frombuffer(decoded.stdout, dtype=numpy.float32)
        if audio.size == 0 or not numpy.any(audio):
            return "", ""

        if model_size is None:
            model_size = os.getenv("WHISPER_MODEL", "base")

        from faster_whisper import WhisperModel

        from core import llm

        started = time.perf_counter()
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments, info = model.transcribe(audio)
        except Exception:
            llm.record_model_call(model_size, 1, time.perf_counter() - started, "OTHER")
            raise
        llm.record_model_call(model_size, 1, time.perf_counter() - started, "OK")
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        return transcript, info.language
    except Exception as error:
        raise VideoError(
            f"Could not transcribe this video's audio "
            f"({type(error).__name__}: {error})"
        ) from error


def _transcript_for_source(video_path, audio_result=None):
    source = config.transcript_source()
    if source == "whisper":
        return transcribe(video_path)

    if audio_result is None:
        from core.audio import audio_brief
        audio_result = audio_brief(video_path)

    gemini_text = ""
    gemini_language = ""
    if audio_result.get("available"):
        gemini_text = (audio_result.get("transcript_best_effort")
                       or audio_result.get("transcript") or "").strip()
        gemini_language = audio_result.get("language") or ""
        if source == "gemini":
            return gemini_text, gemini_language

    whisper_text, whisper_language = transcribe(video_path)
    if source == "gemini":
        return whisper_text, whisper_language
    if gemini_text and whisper_text:
        return f"Gemini: {gemini_text}\nWhisper: {whisper_text}", gemini_language or whisper_language
    return gemini_text or whisper_text, gemini_language or whisper_language


def extract_all(video_path, n_frames=8, audio_result=None):
    """Extract duration, frames, transcript, and speech metadata."""
    check_media_tools()
    video_path = Path(video_path)
    if not video_path.exists() or not video_path.is_file():
        raise VideoError(f"Video file not found: {video_path}")
    duration = get_duration(video_path)
    frames = extract_frames(video_path, n_frames)
    transcript, language = _transcript_for_source(video_path, audio_result=audio_result)
    return {
        "duration_sec": round(duration, 1),
        "frames": frames,
        "transcript": transcript,
        "language": language,
        "has_speech": bool(transcript),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python -m core.extract "path/to/reel.mp4"')
        sys.exit(1)
    try:
        result = extract_all(sys.argv[1])
    except VideoError as error:
        print(f"Video error: {error}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2, ensure_ascii=False))
