"""Audio extraction, signal statistics, and one-call audio brief generation."""

import json
import re
import subprocess
from pathlib import Path

from core import extract, llm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "tmp" / "audio"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "audio_brief.txt"


def extract_audio(video_path):
    """Extract mono 16 kHz WAV audio, returning None when the video is silent."""
    extract.check_media_tools()
    if not extract.has_audio_track(video_path):
        return None
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AUDIO_DIR / f"{Path(video_path).stem}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(video_path),
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_path)],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Audio extraction failed ({type(error).__name__}: {error})") from error
    return str(out_path)


def _duration(video_path):
    try:
        return extract.get_duration(video_path)
    except Exception:
        return None


def audio_stats(video_path):
    """Read volume and silence filters. Individual parse failures yield None."""
    result = {"mean_volume_db": None, "max_volume_db": None,
              "silence_ratio": None, "first_sound_sec": None}
    try:
        if not extract.has_audio_track(video_path):
            return result
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-i", str(video_path), "-af",
             "volumedetect,silencedetect=noise=-35dB:d=0.1", "-f", "null", "-"],
            capture_output=True, text=True, check=False,
        )
        stderr = proc.stderr or ""
    except Exception:
        return result
    for key, pattern in (
        ("mean_volume_db", r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB"),
        ("max_volume_db", r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB"),
    ):
        try:
            match = re.search(pattern, stderr)
            if match:
                result[key] = float(match.group(1))
        except (ValueError, TypeError):
            pass
    try:
        events = [(float(value), kind) for kind, value in re.findall(
            r"(silence_start|silence_end):\s*(-?\d+(?:\.\d+)?)", stderr
        )]
        duration = _duration(video_path)
        if duration and duration > 0:
            silenced = 0.0
            open_start = None
            intervals = []
            for value, kind in events:
                if kind == "silence_start":
                    open_start = max(0.0, min(duration, value))
                elif open_start is not None:
                    end = max(0.0, min(duration, value))
                    end = max(open_start, end)
                    intervals.append((open_start, end))
                    silenced += end - open_start
                    open_start = None
            if open_start is not None:
                silenced += max(0.0, duration - open_start)
                intervals.append((open_start, duration))
            result["silence_ratio"] = min(1.0, silenced / duration)
            leading_silence = (
                bool(events)
                and events[0][1] == "silence_start"
                and events[0][0] <= 0.02
            )
            result["first_sound_sec"] = intervals[0][1] if leading_silence and intervals else 0.0
    except (ValueError, TypeError, IndexError):
        pass
    return result


def audio_brief(video_path):
    """Generate an audio interpretation with a single Gemini request."""
    try:
        wav_path = extract_audio(video_path)
        if wav_path is None:
            return {"available": False, "reason": "No audio stream was found."}
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        result = llm.ask_json(prompt, audio_path=wav_path)
        result["available"] = True
        result.update(audio_stats(video_path))
        return result
    except Exception as error:
        return {"available": False,
                "reason": f"Audio brief failed ({type(error).__name__}: {error})"}
