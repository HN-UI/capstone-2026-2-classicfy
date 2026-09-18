import csv
import json
import tempfile
import unittest
from pathlib import Path

from preprocessing.asap_loader import ASAPLoader


class ASAPLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        self.score_key = "Bach/Fugue/midi_score.mid"
        self.performance_keys = ["Bach/Fugue/first.mid", "Bach/Fugue/second.mid"]
        for key in [self.score_key, *self.performance_keys]:
            path = self.root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        with (self.root / "metadata.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=["composer", "title", "midi_score", "midi_performance"]
            )
            writer.writeheader()
            for key in self.performance_keys:
                writer.writerow(
                    {
                        "composer": "Bach",
                        "title": "Fugue",
                        "midi_score": self.score_key,
                        "midi_performance": key,
                    }
                )

        self.annotations = {
            self.performance_keys[0]: self._annotation(aligned=True),
            self.performance_keys[1]: self._annotation(aligned=False),
        }
        self.annotations[self.performance_keys[1]]["performance_beats"] = [1.1]
        self._write_annotations()

    @staticmethod
    def _annotation(aligned: bool) -> dict:
        return {
            "score_and_performance_aligned": aligned,
            "midi_score_beats": [0.5, 1.0],
            "performance_beats": [1.1, 2.2],
            "midi_score_downbeats": [0.5],
            "performance_downbeats": [1.1],
            "midi_score_time_signatures": {"0.5": ["4/4", 4]},
            "perf_time_signatures": {"1.1": ["4/4", 4]},
        }

    def _write_annotations(self) -> None:
        (self.root / "asap_annotations.json").write_text(
            json.dumps(self.annotations), encoding="utf-8"
        )

    def test_get_sample_and_filter_aligned_samples(self) -> None:
        loader = ASAPLoader(self.root)
        sample = loader.get_sample(self.performance_keys[0])

        self.assertEqual(sample.composer, "Bach")
        self.assertEqual(sample.title, "Fugue")
        self.assertEqual(sample.score_path, self.root / self.score_key)
        self.assertEqual(sample.performance_path, self.root / self.performance_keys[0])
        self.assertEqual(sample.score_beats, [0.5, 1.0])
        self.assertEqual(sample.performance_downbeats, [1.1])
        self.assertEqual(sample.score_time_signatures, {"0.5": ["4/4", 4]})
        self.assertEqual(len(list(loader.iter_samples())), 2)
        self.assertEqual(
            [item.performance_key for item in loader.iter_samples(aligned_only=True)],
            [self.performance_keys[0]],
        )
        self.assertFalse(loader.get_sample(self.performance_keys[1]).aligned)

    def test_missing_midi_file_raises(self) -> None:
        (self.root / self.performance_keys[0]).unlink()
        with self.assertRaises(FileNotFoundError):
            ASAPLoader(self.root).get_sample(self.performance_keys[0])

    def test_missing_annotation_raises(self) -> None:
        del self.annotations[self.performance_keys[0]]
        self._write_annotations()
        with self.assertRaisesRegex(KeyError, "Missing ASAP annotation"):
            ASAPLoader(self.root).get_sample(self.performance_keys[0])

    def test_aligned_sample_requires_matching_beat_counts(self) -> None:
        self.annotations[self.performance_keys[0]]["performance_beats"] = [1.1]
        self._write_annotations()
        with self.assertRaisesRegex(ValueError, "Mismatched aligned beats"):
            ASAPLoader(self.root).get_sample(self.performance_keys[0])

    def test_metadata_path_cannot_escape_dataset_root(self) -> None:
        metadata_path = self.root / "metadata.csv"
        metadata_path.write_text(
            "midi_score,midi_performance\n../outside.mid,Bach/Fugue/first.mid\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "outside ASAP root"):
            ASAPLoader(self.root).get_sample(self.performance_keys[0])


if __name__ == "__main__":
    unittest.main()
