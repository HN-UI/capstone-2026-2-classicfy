import csv
from dataclasses import replace
import json
import os
from pathlib import Path
from statistics import median
import tempfile
import unittest

import numpy as np
import pretty_midi

from features import (
    BeatSequence, PieceFeatureInput, extract_articulation, extract_dynamics,
    extract_pedaling, separate_piece_feature,
)
from preprocessing import ASAPLoader, load_match, load_midi


CHANNELS = ("dynamics", "pedal_depth", "pedal_down_ratio", "pedal_changes", "articulation")


def collect_inputs(samples, *, allow_non_robust=True):
    """정렬 품질 선택은 공통 제거 함수 밖에서 결정한다."""
    groups = {name: [] for name in CHANNELS}
    for sample in samples:
        midi = load_midi(sample.performance_path)
        dynamics = extract_dynamics(midi, sample.performance_beats)
        pedal = extract_pedaling(midi, sample.performance_beats)
        sequences = {
            "dynamics": dynamics.sequence, "pedal_depth": pedal.depth,
            "pedal_down_ratio": pedal.down_ratio, "pedal_changes": pedal.changes,
        }
        if allow_non_robust or sample.robust_note_alignment is True:
            if sample.note_alignment_path is not None:
                sequences["articulation"] = extract_articulation(
                    load_match(sample.note_alignment_path), sample.performance_beats
                ).sequence
            else:
                length = len(sample.performance_beats) - 1
                sequences["articulation"] = BeatSequence(
                    np.full(length, np.nan), np.zeros(length, dtype=bool)
                )
        for name, sequence in sequences.items():
            groups[name].append(PieceFeatureInput.from_asap_sample(sample, sequence))
    return groups


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class CommonPatternPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        asap_root, nasap_root = root / "ASAP", root / "nASAP"
        for path in (asap_root / "piece", nasap_root / "piece"):
            path.mkdir(parents=True)
        score_key = "piece/midi_score.mid"
        score = pretty_midi.PrettyMIDI()
        score.instruments.append(pretty_midi.Instrument(0))
        score.write(str(asap_root / score_key))
        asap_rows, nasap_rows, annotations = [], [], {}
        for key, factor, ratio, velocities in (
            ("a", 1, 1, {0: 40, 2: 100, 3: 64}),
            ("b", 2, .5, {0: 60, 1: 80, 2: 80, 3: 64}),
            ("c", 1, .25, {0: 100, 1: 100, 2: 60, 3: 64}),
        ):
            performance_key = f"piece/{key}.mid"
            match_key = f"piece/{key}.match"
            midi = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(0)
            lines = ["info(midiClockUnits,480).", "info(midiClockRate,500000)."]
            for beat, velocity in velocities.items():
                onset, offset = beat * factor, (beat + ratio) * factor
                piano.notes.append(pretty_midi.Note(velocity, 60, onset, offset))
                lines.append(
                    f"snote(s{beat},[C,n],4,1:1,0,1/4,{beat},{beat + 1},[])"
                    f"-note(p{beat},60,{int(onset * 960)},{int(offset * 960)},{velocity},0,0)."
                )
            if key != "a":
                value, release = (127, factor) if key == "b" else (64, 2 * factor)
                piano.control_changes.extend([
                    pretty_midi.ControlChange(64, value, 0),
                    pretty_midi.ControlChange(64, 0, release),
                ])
            midi.instruments.append(piano)
            midi.write(str(asap_root / performance_key))
            (nasap_root / match_key).write_text("\n".join(lines) + "\n", encoding="utf-8")
            score_beats = [0., 1., 2., 3.]
            performance_beats = [beat * factor for beat in score_beats]
            annotations[performance_key] = {
                "score_and_performance_aligned": True,
                "midi_score_beats": score_beats, "performance_beats": performance_beats,
                "midi_score_beats_type": {str(b): "b" for b in score_beats},
                "performance_beats_type": {str(b): "b" for b in performance_beats},
                "midi_score_downbeats": [], "performance_downbeats": [],
                "midi_score_time_signatures": {}, "perf_time_signatures": {},
            }
            asap_rows.append({"midi_score": score_key, "midi_performance": performance_key})
            nasap_rows.append({
                "midi_performance": performance_key, "match_file": match_key,
                "robust_note_alignment": "0.0" if key == "c" else "1.0",
            })
        write_csv(asap_root / "metadata.csv", ["midi_score", "midi_performance"], asap_rows)
        write_csv(nasap_root / "metadata.csv", ["midi_performance", "match_file", "robust_note_alignment"], nasap_rows)
        (asap_root / "asap_annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
        self.samples = list(ASAPLoader(asap_root, nasap_root).iter_samples(aligned_only=True))

    def test_all_feature_channels_from_midi_and_match_files(self) -> None:
        results = {name: separate_piece_feature(inputs) for name, inputs in collect_inputs(self.samples).items()}
        dynamics = results["dynamics"]["piece/a.mid"]
        np.testing.assert_allclose(dynamics.common.values, np.array([60, 90, 80]) / 127)
        np.testing.assert_allclose(dynamics.relative.values, np.array([-20, np.nan, 20]) / 127)
        np.testing.assert_array_equal(dynamics.common_support, [3, 2, 3])
        np.testing.assert_array_equal(dynamics.raw.mask, [True, False, True])
        art = results["articulation"]["piece/a.mid"]
        np.testing.assert_allclose(art.raw.values, [0, np.nan, 0])
        np.testing.assert_allclose(art.common.values, [-1, -1.5, -1])
        np.testing.assert_allclose(art.relative.values, [1, np.nan, 1])
        np.testing.assert_array_equal(art.common_support, [3, 2, 3])
        for name, common in (
            ("pedal_depth", [64 / 127, 0, 0]),
            ("pedal_down_ratio", [1, 0, 0]),
            ("pedal_changes", [1, 0, 0]),
        ):
            with self.subTest(channel=name):
                result = results[name]["piece/a.mid"]
                np.testing.assert_array_equal(result.raw.values, [0, 0, 0])
                np.testing.assert_allclose(result.common.values, common)
                np.testing.assert_allclose(result.relative.values, -np.array(common))
                np.testing.assert_array_equal(result.common_support, [3, 3, 3])
                self.assertTrue(result.relative.mask.all())

    def test_caller_can_exclude_non_robust_articulation(self) -> None:
        self.assertFalse(self.samples[2].robust_note_alignment)
        groups = collect_inputs(self.samples, allow_non_robust=False)
        self.assertEqual(len(groups["dynamics"]), 3)
        self.assertEqual(len(groups["articulation"]), 2)
        result = separate_piece_feature(groups["articulation"])["piece/a.mid"]
        np.testing.assert_allclose(result.common.values, [-.5, np.nan, -.5])
        np.testing.assert_array_equal(result.common_support, [2, 1, 2])

    def test_missing_alignment_remains_masked_and_does_not_reduce_other_channels(self) -> None:
        samples = [*self.samples[:2], replace(self.samples[2], note_alignment_path=None)]
        groups = collect_inputs(samples)
        result = separate_piece_feature(groups["articulation"])["piece/c.mid"]
        self.assertFalse(result.raw.mask.any())
        self.assertFalse(result.relative.mask.any())
        np.testing.assert_array_equal(result.common_support, [2, 1, 2])
        self.assertTrue(separate_piece_feature(groups["pedal_depth"])["piece/c.mid"].relative.mask.all())

    def test_sample_adapter_rejects_unaligned_and_repeated_structure_mismatch(self) -> None:
        sample = self.samples[0]
        sequence = extract_dynamics(load_midi(sample.performance_path), sample.performance_beats).sequence
        with self.assertRaisesRegex(ValueError, "not aligned"):
            PieceFeatureInput.from_asap_sample(replace(sample, aligned=False), sequence)
        with self.assertRaisesRegex(ValueError, "Mismatched aligned beats"):
            PieceFeatureInput.from_asap_sample(replace(sample, performance_beats=[0, 1]), sequence)
        altered = replace(sample, note_alignment_path=Path("/nasap/piece_no_repeat/a.match"))
        altered_input = PieceFeatureInput.from_asap_sample(altered, sequence)
        original_input = PieceFeatureInput.from_asap_sample(sample, sequence)
        self.assertNotEqual(altered_input.piece_key, original_input.piece_key)
        with self.assertRaisesRegex(ValueError, "same piece"):
            separate_piece_feature([original_input, replace(altered_input, performance_key="other")])
        override = PieceFeatureInput.from_asap_sample(sample, sequence, piece_key="explicit_piece")
        self.assertEqual(override.piece_key, "explicit_piece")


class RealCommonPatternPipelineTest(unittest.TestCase):
    def test_real_asap_and_nasap_piece_when_available(self) -> None:
        datasets = Path(__file__).resolve().parents[4] / "datasets"
        root = Path(os.environ.get("ASAP_ROOT", datasets / "ASAP")).expanduser()
        nasap_root = Path(os.environ.get("NASAP_ROOT", datasets / "nASAP")).expanduser()
        if not (root / "metadata.csv").is_file() or not (nasap_root / "metadata.csv").is_file():
            self.skipTest("ASAP or (n)ASAP dataset is not available")
        score_path = (root / "Bach/Fugue/bwv_848/midi_score.mid").resolve()
        samples = [s for s in ASAPLoader(root, nasap_root).iter_samples(True) if s.score_path == score_path]
        self.assertGreaterEqual(len(samples), 2)
        for inputs in collect_inputs(samples).values():
            results = list(separate_piece_feature(inputs).values())
            for beat in range(len(inputs[0].sequence)):
                raw = [
                    item.sequence.values[beat] for item in inputs
                    if item.sequence.mask[beat] and np.isfinite(item.sequence.values[beat])
                    and item.score_beats[beat + 1] > item.score_beats[beat]
                ]
                for result in results:
                    self.assertEqual(result.common_support[beat], len(raw))
                    np.testing.assert_allclose(result.common.values[beat], median(raw) if len(raw) >= 2 else np.nan)
                    if result.relative.mask[beat]:
                        self.assertAlmostEqual(result.relative.values[beat], result.raw.values[beat] - result.common.values[beat])
                    else:
                        self.assertTrue(np.isnan(result.relative.values[beat]))
            a, b = results[:2]
            overlap = a.relative.mask & b.relative.mask
            np.testing.assert_allclose(
                (a.relative.values - b.relative.values)[overlap],
                (a.raw.values - b.raw.values)[overlap], atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
