import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import pretty_midi

from features import TempoInput, extract_piece_tempo_features
from preprocessing import ASAPLoader, AsapSample, MidiData, load_midi


class PreprocessingIntegrationTest(unittest.TestCase):
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
        default_root = Path(__file__).resolve().parents[3] / "datasets" / "ASAP"
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
        default_root = Path(__file__).resolve().parents[3] / "datasets" / "ASAP"
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
            TempoInput(
                performance_key=sample.performance_key,
                score_beats=sample.score_beats,
                performance_beats=sample.performance_beats,
                score_beat_types=sample.score_beat_types,
                performance_beat_types=sample.performance_beat_types,
            )
            for sample in samples
        ]
        features = extract_piece_tempo_features(tempo_inputs)

        self.assertEqual(len(features), len(samples))
        for sample in samples:
            feature = features[sample.performance_key]
            self.assertEqual(len(feature.intervals), len(sample.score_beats) - 1)
            self.assertIsNotNone(feature.overall_individual_tempo)


if __name__ == "__main__":
    unittest.main()
