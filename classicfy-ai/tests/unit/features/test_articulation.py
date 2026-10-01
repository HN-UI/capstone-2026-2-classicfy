import unittest

import numpy as np

from features import (
    ArticulationFeature,
    BeatSequence,
    build_tempo_map,
    extract_articulation,
    extract_note_articulation,
    summarize_articulation,
)
from preprocessing import NoteAlignment, PerformedNote, ScoreNote


def make_alignment(
    notes: list[tuple[float, float, float, float]],
    attributes: dict[int, tuple[str, ...]] | None = None,
) -> NoteAlignment:
    """(악보 onset, 악보 offset, 연주 onset, 연주 offset) 목록으로 정렬을 만든다."""
    attributes = attributes or {}
    matches = tuple(
        (
            ScoreNote(
                note_id=f"s{index}",
                pitch=60,
                onset_beats=score_onset,
                offset_beats=score_offset,
                attributes=attributes.get(index, ()),
            ),
            PerformedNote(
                note_id=f"p{index}",
                pitch=60,
                onset=onset,
                offset=offset,
                velocity=64,
            ),
        )
        for index, (score_onset, score_offset, onset, offset) in enumerate(notes)
    )
    return NoteAlignment(matches=matches, deletions=(), insertions=())


# 1 beat = 0.5초인 일정한 빠르기. 마지막 음(4→5)은 tempo map 끝을 넘어서 제외된다.
STEADY = [(0, 1, 0.0, 0.5), (1, 2, 0.5, 0.75), (2, 3, 1.0, 2.0), (3, 4, 1.5, 2.0), (4, 5, 2.0, 2.5)]


class TempoMapTest(unittest.TestCase):
    def test_chord_uses_median_onset_and_skips_grace_notes(self) -> None:
        alignment = make_alignment(
            [(0, 1, 0.0, 0.5), (0, 1, 0.02, 0.5), (0, 1, 0.04, 0.5), (1, 1, 0.3, 0.35), (1, 2, 0.5, 1.0)],
            attributes={3: ("grace",)},
        )

        positions, times = build_tempo_map(alignment)

        self.assertEqual(positions.tolist(), [0.0, 1.0])
        np.testing.assert_allclose(times, [0.02, 0.5])

    def test_backward_position_from_misaligned_note_is_dropped(self) -> None:
        # beat 3이 beat 0과 1 사이 시각에 잘못 정렬되어 연주 시각이 거꾸로 간다.
        alignment = make_alignment(
            [(0, 1, 0.0, 0.4), (1, 2, 0.5, 0.9), (2, 3, 1.0, 1.4), (3, 4, 0.3, 0.4),
             (4, 5, 2.0, 2.4), (5, 6, 2.5, 2.9)]
        )

        positions, _ = build_tempo_map(alignment)

        self.assertEqual(positions.tolist(), [0.0, 1.0, 2.0, 4.0, 5.0])


class NoteArticulationTest(unittest.TestCase):
    def test_ratio_is_log2_of_held_time_over_notated_time(self) -> None:
        notes = extract_note_articulation(make_alignment(STEADY))

        np.testing.assert_allclose(notes.values, [0.0, -1.0, 1.0, 0.0])
        self.assertEqual(notes.match_indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(notes.excluded_counts["outside_tempo_map"], 1)

    def test_expected_duration_follows_local_tempo(self) -> None:
        # 두 번째 beat만 두 배로 느려졌다. 그 beat를 끝까지 누른 음은 악보 음가 그대로다.
        alignment = make_alignment(
            [(0, 1, 0.0, 0.5), (1, 2, 0.5, 1.5), (2, 3, 1.5, 2.0), (3, 4, 2.0, 2.5)]
        )

        notes = extract_note_articulation(alignment)

        np.testing.assert_allclose(notes.values, [0.0, 0.0, 0.0])

    def test_grace_and_ornament_notes_are_excluded(self) -> None:
        alignment = make_alignment(
            [*STEADY, (1, 1, 0.45, 0.5), (2, 3, 1.0, 1.1)],
            attributes={5: ("grace",), 6: ("trillmark",)},
        )

        notes = extract_note_articulation(alignment)

        self.assertEqual(len(notes.values), 4)
        self.assertEqual(notes.excluded_counts["grace"], 1)
        self.assertEqual(notes.excluded_counts["ornament"], 1)

    def test_single_score_position_gives_no_values(self) -> None:
        notes = extract_note_articulation(make_alignment([(0, 1, 0.0, 0.5)]))

        self.assertEqual(len(notes.values), 0)
        self.assertEqual(notes.excluded_counts["outside_tempo_map"], 1)


class ExtractArticulationTest(unittest.TestCase):
    def test_window_value_is_median_of_notes_starting_inside(self) -> None:
        feature = extract_articulation(make_alignment(STEADY), [0.0, 1.0, 2.0, 3.0])

        self.assertIsInstance(feature, ArticulationFeature)
        self.assertIsInstance(feature.sequence, BeatSequence)
        # 구간 0: 0.0과 -1.0의 중앙값, 구간 1: 1.0과 0.0의 중앙값, 구간 2: 음 없음
        np.testing.assert_allclose(feature.sequence.values[:2], [-0.5, 0.5])
        self.assertTrue(np.isnan(feature.sequence.values[2]))
        self.assertEqual(feature.sequence.mask.tolist(), [True, True, False])
        self.assertEqual(feature.note_counts.tolist(), [2, 2, 0])

    def test_summary_uses_valid_windows_only(self) -> None:
        feature = extract_articulation(make_alignment(STEADY), [0.0, 1.0, 2.0, 3.0])

        summary = summarize_articulation(feature)

        self.assertAlmostEqual(summary["articulation_mean"], 0.0)
        self.assertAlmostEqual(summary["articulation_range"], 0.9)

    def test_summary_without_values_is_nan(self) -> None:
        feature = extract_articulation(make_alignment([]), [0.0, 1.0])

        summary = summarize_articulation(feature)

        self.assertTrue(np.isnan(summary["articulation_mean"]))
        self.assertTrue(np.isnan(summary["articulation_range"]))


if __name__ == "__main__":
    unittest.main()
