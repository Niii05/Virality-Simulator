"""Run Phase 2 briefs and a small persona comparison for one video."""

import json
import argparse
import logging
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import audio, brief, llm
from core.extract import extract_all
from core.personas import load_personas
from core.persona_react import react


def select_personas(personas, count, seed=None):
    """Choose a seeded, round-robin spread across available archetypes."""
    rng = random.Random(seed)
    grouped = {}
    for persona in personas:
        grouped.setdefault(persona["archetype"], []).append(persona)
    archetypes = list(grouped)
    rng.shuffle(archetypes)
    for group in grouped.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < min(count, len(personas)):
        made_progress = False
        for archetype in archetypes:
            if grouped[archetype]:
                selected.append(grouped[archetype].pop())
                made_progress = True
                if len(selected) == min(count, len(personas)):
                    break
        if not made_progress:
            break
    return selected


def _print_call_summary(events):
    print("MODEL CALL SUMMARY")
    print(f"Total model calls: {len(events)}")
    headers = ("stage", "model", "attempt", "seconds", "status")
    rows = [[str(event[field]) for field in ("stage", "model", "attempt", "seconds", "status")]
            for event in events]
    if rows:
        widths = [max(len(headers[index]), *(len(row[index]) for row in rows))
                  for index in range(len(headers))]
        print(" | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
        print("-+-".join("-" * width for width in widths))
        for row in rows:
            print(" | ".join(row[index].ljust(widths[index]) for index in range(len(headers))))
    extras = [event for event in events if event["reason"] in {"retry", "fallback"}]
    if extras:
        details = ", ".join(f"{event['reason']} at {event['stage']} ({event['model']})"
                             for event in extras)
        print(f"Extra call explanation: {details}.")
    else:
        print("Extra call explanation: none; no retry or fallback calls occurred.")


def main(video_path, persona_count=3, seed=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    event_start = len(llm.CALL_EVENTS)
    with llm.call_stage("audio_brief"):
        audio_result = audio.audio_brief(video_path)
    with llm.call_stage("extract"):
        extracted = extract_all(video_path, audio_result=audio_result)
    with llm.call_stage("vision_brief"):
        full = brief.build_full_brief(extracted, audio=audio_result)
    silent = brief.build_silent_brief(full)
    print("FULL BRIEF")
    print(json.dumps(full, ensure_ascii=False, indent=2))
    print("SILENT BRIEF")
    print(json.dumps(silent, ensure_ascii=False, indent=2))
    for persona in select_personas(load_personas(), persona_count, seed):
        print(f"REACTIONS: {persona['id']}")
        for condition in ("sound_on", "sound_off"):
            with llm.call_stage(f"persona:{persona['id']}:{condition}"):
                reaction = react(persona, full, condition)
            print(condition)
            print(json.dumps(reaction, ensure_ascii=False, indent=2))
    _print_call_summary(llm.CALL_EVENTS[event_start:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--personas", type=int, default=3, help="number of personas to sample (default: 3)")
    parser.add_argument("--seed", type=int, help="seed for repeatable persona sampling")
    arguments = parser.parse_args()
    if arguments.personas < 1:
        parser.error("--personas must be at least 1")
    main(arguments.video, persona_count=arguments.personas, seed=arguments.seed)
