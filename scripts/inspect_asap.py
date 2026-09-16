from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASAP_ROOT = PROJECT_ROOT.parent / "datasets" / "ASAP"

metadata = pd.read_csv(ASAP_ROOT / "metadata.csv")

row = metadata.iloc[0]

score_path = ASAP_ROOT / row["midi_score"]
performance_path = ASAP_ROOT / row["midi_performance"]

print("Score MIDI")
print(score_path)
print("exists:", score_path.exists())

print()

print("Performance MIDI")
print(performance_path)
print("exists:", performance_path.exists())


import pretty_midi

score = pretty_midi.PrettyMIDI(str(score_path))
performance = pretty_midi.PrettyMIDI(str(performance_path))

print("score instruments:", len(score.instruments))
print("performance instruments:", len(performance.instruments))

print("score duration:", score.get_end_time())
print("performance duration:", performance.get_end_time())

for instrument in performance.instruments:
    for note in instrument.notes[:10]:
        print(
            "pitch:", note.pitch,
            "start:", note.start,
            "end:", note.end,
            "velocity:", note.velocity,
        )

for i, instrument in enumerate(performance.instruments):
    print(
        "instrument:",
        i,
        "program:",
        instrument.program,
        "name:",
        instrument.name,
        "notes:",
        len(instrument.notes),
        "CC:",
        len(instrument.control_changes),
    )

for i, instrument in enumerate(score.instruments):
    print(
        "instrument:",
        i,
        "program:",
        instrument.program,
        "name:",
        instrument.name,
        "notes:",
        len(instrument.notes),
    )

for i, instrument in enumerate(score.instruments):
    print(
        "instrument:",
        i,
        "program:",
        instrument.program,
        "name:",
        instrument.name,
        "notes:",
        len(instrument.notes),
    )
for instrument in performance.instruments:
    pedal_events = [
        cc
        for cc in instrument.control_changes
        if cc.number == 64
    ]

    print("pedal events:", len(pedal_events))

    for cc in pedal_events[:20]:
        print(
            "time:", cc.time,
            "value:", cc.value,
        )

values = [
    cc.value
    for instrument in performance.instruments
    for cc in instrument.control_changes
    if cc.number == 64
]

print("min:", min(values))
print("max:", max(values))
print("unique:", len(set(values)))
print("first unique values:", sorted(set(values))[:20])

from collections import Counter

print(Counter(values).most_common(20))

import json


# ============================================================
# ASAP alignment inspection
# ============================================================

annotation_path = ASAP_ROOT / "asap_annotations.json"

with open(annotation_path, "r") as f:
    annotations = json.load(f)

performance_key = row["midi_performance"]

print("\n========== ASAP ALIGNMENT ==========")
print("performance key:", performance_key)

annotation = annotations[performance_key]

print("\nannotation keys:")
for key in annotation.keys():
    print("-", key)


# ------------------------------------------------------------
# 1. score-performance alignment 여부
# ------------------------------------------------------------

aligned = annotation["score_and_performance_aligned"]

print("\nscore_and_performance_aligned:", aligned)


# ------------------------------------------------------------
# 2. beat alignment
# ------------------------------------------------------------

score_beats = annotation["midi_score_beats"]
performance_beats = annotation["performance_beats"]

print("\nscore beats:", len(score_beats))
print("performance beats:", len(performance_beats))

print("\nfirst 10 beat pairs")

for i, (score_beat, perf_beat) in enumerate(
    zip(score_beats[:10], performance_beats[:10])
):
    print(
        f"beat {i:3d} | "
        f"score: {score_beat:8.3f}s | "
        f"performance: {perf_beat:8.3f}s"
    )


# ------------------------------------------------------------
# 3. downbeat alignment
# ------------------------------------------------------------

score_downbeats = annotation["midi_score_downbeats"]
performance_downbeats = annotation["performance_downbeats"]

print("\nscore downbeats:", len(score_downbeats))
print("performance downbeats:", len(performance_downbeats))

print("\nfirst 10 downbeat pairs")

for i, (score_db, perf_db) in enumerate(
    zip(score_downbeats[:10], performance_downbeats[:10])
):
    print(
        f"measure {i:3d} | "
        f"score: {score_db:8.3f}s | "
        f"performance: {perf_db:8.3f}s"
    )


# ------------------------------------------------------------
# 4. beat interval 확인
# ------------------------------------------------------------

print("\nfirst 10 beat intervals")

for i in range(min(10, len(score_beats) - 1)):
    score_interval = score_beats[i + 1] - score_beats[i]
    perf_interval = performance_beats[i + 1] - performance_beats[i]

    print(
        f"{i:3d} -> {i+1:3d} | "
        f"score interval: {score_interval:.3f}s | "
        f"performance interval: {perf_interval:.3f}s"
    )


# ------------------------------------------------------------
# 5. time signature
# ------------------------------------------------------------

print("\nperformance time signatures:")
print(annotation["perf_time_signatures"])

print("\nscore time signatures:")
print(annotation["midi_score_time_signatures"])