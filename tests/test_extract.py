import sys
from types import ModuleType, SimpleNamespace

import numpy
import pytest

from core import extract


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (0.5, [0.0]),
        (2.5, [0.0, 1.0, 2.0]),
        (10, [0.0, 1.0, 2.0, 3.0, 4.4, 5.8, 7.2, 8.6]),
        (60, [0.0, 1.0, 2.0, 3.0, 14.4, 25.8, 37.2, 48.6]),
    ],
)
def test_frame_timestamps_for_short_and_long_videos(duration, expected):
    assert extract.frame_timestamps(duration) == expected


def test_frame_timestamps_empty_for_nonpositive_duration():
    assert extract.frame_timestamps(0) == []
    assert extract.frame_timestamps(10, n=0) == []


def set_audio_present(monkeypatch):
    monkeypatch.setattr(extract, "check_media_tools", lambda: None)
    monkeypatch.setattr(extract, "has_audio_track", lambda video_path: True)


def set_audio_output(monkeypatch, output):
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(extract.subprocess, "run", fake_run)
    return commands


def install_fake_whisper(monkeypatch, model_class):
    fake_module = ModuleType("faster_whisper")
    fake_module.WhisperModel = model_class
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)


@pytest.mark.parametrize(
    "decoded_audio",
    [b"", numpy.zeros(32, dtype=numpy.float32).tobytes()],
    ids=["empty", "all-zero"],
)
def test_empty_or_zero_audio_returns_empty_transcript_without_loading_whisper(
    monkeypatch, decoded_audio
):
    set_audio_present(monkeypatch)
    commands = set_audio_output(monkeypatch, decoded_audio)
    model_loads = []

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            model_loads.append((args, kwargs))

    install_fake_whisper(monkeypatch, FakeWhisperModel)

    assert extract.transcribe("sample.mp4") == ("", "")
    assert model_loads == []
    assert commands[0][0] == [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        "sample.mp4",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "-",
    ]
    assert commands[0][1]["stdout"] == extract.subprocess.PIPE


def test_valid_decoded_audio_is_passed_to_mock_whisper(monkeypatch):
    set_audio_present(monkeypatch)
    output = numpy.array([0.25, -0.5, 0.75], dtype=numpy.float32).tobytes()
    set_audio_output(monkeypatch, output)
    captured = {}

    class FakeWhisperModel:
        def __init__(self, model_size, device, compute_type):
            captured["model_args"] = (model_size, device, compute_type)

        def transcribe(self, audio):
            captured["audio"] = audio
            return [SimpleNamespace(text=" hello "), SimpleNamespace(text="world")], SimpleNamespace(
                language="en"
            )

    install_fake_whisper(monkeypatch, FakeWhisperModel)
    monkeypatch.setenv("WHISPER_MODEL", "tiny-test")

    assert extract.transcribe("sample.mp4") == ("hello world", "en")
    assert captured["model_args"] == ("tiny-test", "cpu", "int8")
    expected_audio = numpy.frombuffer(output, dtype=numpy.float32)
    assert captured["audio"].tolist() == expected_audio.tolist()


def test_no_audio_stream_skips_decode_and_whisper(monkeypatch):
    monkeypatch.setattr(extract, "check_media_tools", lambda: None)
    monkeypatch.setattr(extract, "has_audio_track", lambda video_path: False)
    monkeypatch.setattr(
        extract.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("ffmpeg must not run without an audio stream"),
    )
    model_loads = []

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            model_loads.append((args, kwargs))

    install_fake_whisper(monkeypatch, FakeWhisperModel)

    assert extract.transcribe("silent.mp4") == ("", "")
    assert model_loads == []


def test_transcription_error_preserves_original_type_message_and_cause(monkeypatch):
    set_audio_present(monkeypatch)
    original_error = TypeError("open() got an unexpected keyword argument 'metadata_errors'")
    set_audio_output(
        monkeypatch,
        numpy.array([0.25, -0.5], dtype=numpy.float32).tobytes(),
    )

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio):
            raise original_error

    install_fake_whisper(monkeypatch, FakeWhisperModel)

    with pytest.raises(extract.VideoError) as exc_info:
        extract.transcribe("sample.mp4")

    assert "TypeError" in str(exc_info.value)
    assert "metadata_errors" in str(exc_info.value)
    assert exc_info.value.__cause__ is original_error
