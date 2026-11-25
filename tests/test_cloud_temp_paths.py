from pathlib import Path

from core import extract, pipeline


def test_pipeline_uses_and_cleans_per_run_frame_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "extract_all", extract.extract_all)
    frames_root = tmp_path / "frames"
    monkeypatch.setattr(pipeline, "FRAMES_DIR", frames_root)
    monkeypatch.setattr(
        pipeline.audio,
        "audio_brief",
        lambda _video: {"available": False, "reason": "no audio"},
    )
    monkeypatch.setattr(
        pipeline,
        "simulate",
        lambda *_args, **_kwargs: {
            "wave1": [], "wave2": [], "wave1_selection": "fallback",
            "wave1_raw_ids": [], "wave1_personas": [], "dimension_means": {},
            "errors": [], "gate_passed": False, "waves_run": [1],
        },
    )
    monkeypatch.setattr(pipeline, "score_result", lambda _waves: {})
    monkeypatch.setattr(pipeline, "generate", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(pipeline.llm, "CALL_EVENTS", [])
    monkeypatch.setattr(pipeline, "_print_call_summary", lambda _events: None)
    monkeypatch.setattr(extract, "check_media_tools", lambda: None)
    monkeypatch.setattr(extract, "get_duration", lambda _video: 2.0)
    monkeypatch.setattr(extract, "_transcript_for_source", lambda *_args, **_kwargs: ("", ""))
    frame_directories = []

    def fake_extract_frames(_video, _n, out_dir):
        frame_directory = Path(out_dir)
        frame_directory.mkdir(parents=True, exist_ok=True)
        frame_path = frame_directory / "frame_00.jpg"
        frame_path.write_bytes(b"mock frame")
        frame_directories.append(frame_directory)
        return [str(frame_path)]

    monkeypatch.setattr(extract, "extract_frames", fake_extract_frames)

    def build_brief(extracted, **_kwargs):
        assert Path(extracted["frames"][0]).is_file()
        return {"niche_category": "general", "niche_reason": "mocked"}

    monkeypatch.setattr(pipeline.brief, "build_full_brief", build_brief)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mock video")

    pipeline.run(video, tmp_path / "result.json", use_cache=False)

    assert len(frame_directories) == 1
    assert frame_directories[0].parent == frames_root
    assert not frame_directories[0].exists()
