import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import pretty_midi

from features import (
    TempoInput,
    extract_articulation,
    extract_piece_tempo_features,
    summarize_articulation,
    summarize_tempo,
)
from preprocessing import ASAPLoader, AsapSample, MidiData, load_match, load_midi


class AsapPipelineIntegrationTest(unittest.TestCase):
    def test_asap_sample_paths_feed_midi_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            score_key = "piece/midi_score.mid"
            performance_key = "piece/take.mid"
            for key, velocity in [(score_key, 64), (performance_key, 96)]:
                path = root / key
                path.parent.mkdir(parents=True, exist_ok=True)
                midi = pretty_midi.PrettyMIDI()
                piano = pretty_midi.Instrument(program=0)
                piano.notes.append(
                    pretty_midi.Note(velocity=velocity, pitch=60, start=0.5, end=1.0)
                )
                midi.instruments.append(piano)
                midi.write(str(path))

            with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["midi_score", "midi_performance"])
                writer.writeheader()
                writer.writerow({"midi_score": score_key, "midi_performance": performance_key})
            annotation = {
                performance_key: {
                    "score_and_performance_aligned": True,
                    "midi_score_beats": [0.5],
                    "performance_beats": [0.5],
                    "midi_score_beats_type": {"0.5": "db"},
                    "performance_beats_type": {"0.5": "db"},
                    "midi_score_downbeats": [0.5],
                    "performance_downbeats": [0.5],
                    "midi_score_time_signatures": {"0.5": ["4/4", 4]},
                    "perf_time_signatures": {"0.5": ["4/4", 4]},
                }
            }
            (root / "asap_annotations.json").write_text(
                json.dumps(annotation), encoding="utf-8"
            )

            sample = ASAPLoader(root).get_sample(performance_key)
            score = load_midi(sample.score_path)
            performance = load_midi(sample.performance_path)

            self.assertIsInstance(sample, AsapSample)
            self.assertIsInstance(score, MidiData)
            self.assertIsInstance(performance, MidiData)
            self.assertTrue(sample.aligned)
            self.assertEqual(score.notes[0].velocity, 64)
            self.assertEqual(performance.notes[0].velocity, 96)

    def test_real_asap_sample_when_available(self) -> None:
        default_root = Path(__file__).resolve().parents[4] / "datasets" / "ASAP"
        root = Path(os.environ.get("ASAP_ROOT", default_root)).expanduser()
        if not (root / "metadata.csv").is_file():
            self.skipTest("ASAP dataset is not available")

        sample = ASAPLoader(root).get_sample("Bach/Fugue/bwv_846/Shi05M.mid")
        score = load_midi(sample.score_path)
        performance = load_midi(sample.performance_path)

        self.assertTrue(sample.aligned)
        self.assertEqual(len(sample.score_beats), len(sample.performance_beats))
        self.assertTrue(score.notes)
        self.assertTrue(performance.notes)

    def test_extracts_tempo_features_from_real_asap_piece_when_available(self) -> None:
        default_root = Path(__file__).resolve().parents[4] / "datasets" / "ASAP"
        root = Path(os.environ.get("ASAP_ROOT", default_root)).expanduser()
        if not (root / "metadata.csv").is_file():
            self.skipTest("ASAP dataset is not available")

        score_path = (root / "Bach/Fugue/bwv_848/midi_score.mid").resolve()
        samples = [
            sample
            for sample in ASAPLoader(root).iter_samples(aligned_only=True)
            if sample.score_path == score_path
        ]
        tempo_inputs = [
            TempoInput.from_asap_sample(sample)
            for sample in samples
        ]
        features = extract_piece_tempo_features(tempo_inputs)

        self.assertEqual(len(features), len(samples))
        for sample in samples:
            feature = features[sample.performance_key]
            self.assertEqual(len(feature.intervals), len(sample.score_beats) - 1)
            self.assertFalse(
                math.isnan(summarize_tempo(feature)["overall_individual_tempo"])
            )


    def test_extracts_articulation_from_real_note_alignment_when_available(self) -> None:
        datasets = Path(__file__).resolve().parents[4] / "datasets"
        root = Path(os.environ.get("ASAP_ROOT", datasets / "ASAP")).expanduser()
        nasap_root = Path(os.environ.get("NASAP_ROOT", datasets / "nASAP")).expanduser()
        if not (root / "metadata.csv").is_file() or not (nasap_root / "metadata.csv").is_file():
            self.skipTest("ASAP or (n)ASAP dataset is not available")

        loader = ASAPLoader(root, nasap_root)
        sample = loader.get_sample("Bach/Fugue/bwv_846/Shi05M.mid")
        self.assertIsNotNone(sample.note_alignment_path)
        self.assertTrue(sample.robust_note_alignment)

        alignment = load_match(sample.note_alignment_path)
        performance = load_midi(sample.performance_path)
        # match 파일의 연주 음은 연주 MIDI의 음과 음높이·시각이 같아야 한다.
        for _, note in alignment.matches:
            self.assertTrue(
                any(
                    other.pitch == note.pitch
                    and abs(other.start - note.onset) < 0.002
                    and abs(other.end - note.offset) < 0.002
                    for other in performance.notes
                ),
                note,
            )

        feature = extract_articulation(alignment, sample.performance_beats)
        self.assertEqual(len(feature.sequence), len(sample.performance_beats) - 1)
        self.assertGreater(feature.sequence.mask.mean(), 0.9)
        self.assertFalse(math.isnan(summarize_articulation(feature)["articulation_mean"]))

        moved = loader.get_sample("Beethoven/Piano_Sonatas/32-1/Park01.mid")
        self.assertEqual(
            moved.note_alignment_path,
            (nasap_root / "Beethoven/Piano_Sonatas/32-1_no_repeat/Park01.match").resolve(),
        )

if __name__ == "__main__":
    unittest.main()
