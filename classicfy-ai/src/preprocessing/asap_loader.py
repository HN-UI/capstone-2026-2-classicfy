"""ASAP의 악보-연주 짝과 정렬 정보를 읽는다."""

import csv
import filecmp
import json
from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal


BeatType = Literal["b", "db", "bR"]


@dataclass
class AsapSample:
    """한 연주의 MIDI 경로와 ASAP 정렬 정보를 담는다."""

    performance_key: str
    composer: str
    title: str
    score_path: Path
    performance_path: Path
    aligned: bool
    score_beats: list[float]
    performance_beats: list[float]
    score_beat_types: list[BeatType]
    performance_beat_types: list[BeatType]
    score_downbeats: list[float]
    performance_downbeats: list[float]
    score_time_signatures: dict[str, list[str | int]]
    performance_time_signatures: dict[str, list[str | int]]
    # (n)ASAP 루트를 함께 준 경우에만 채운다. 정렬 파일이 없는 연주는 None이다.
    note_alignment_path: Path | None = None
    robust_note_alignment: bool | None = None


@dataclass(frozen=True)
class _NoteAlignmentEntry:
    key: str
    match_path: Path | None
    robust: bool | None


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")


def _read_times(annotation: dict, field: str, performance_key: str) -> list[float]:
    value = annotation.get(field)
    if not isinstance(value, list) or any(type(time) not in (int, float) for time in value):
        raise ValueError(f"Invalid {field} for {performance_key}")
    return [float(time) for time in value]


def _read_beat_types(
    annotation: dict,
    beats: list[float],
    field: str,
    performance_key: str,
) -> list[BeatType]:
    value = annotation.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {field} for {performance_key}")

    types_by_time: dict[float, BeatType] = {}
    for time, beat_type in value.items():
        if not isinstance(time, str) or beat_type not in {"b", "db", "bR"}:
            raise ValueError(f"Invalid {field} for {performance_key}")
        try:
            parsed_time = float(time)
        except ValueError as exc:
            raise ValueError(f"Invalid {field} for {performance_key}") from exc
        if not isfinite(parsed_time) or parsed_time in types_by_time:
            raise ValueError(f"Invalid {field} for {performance_key}")
        types_by_time[parsed_time] = beat_type

    beat_types = []
    for time in beats:
        if time not in types_by_time:
            raise ValueError(f"Missing {field} at {time} for {performance_key}")
        beat_types.append(types_by_time[time])
    return beat_types


def _read_time_signatures(
    annotation: dict, field: str, performance_key: str
) -> dict[str, list[str | int]]:
    value = annotation.get(field)
    if not isinstance(value, dict) or any(
        not isinstance(time, str)
        or not isinstance(signature, list)
        or len(signature) != 2
        or not isinstance(signature[0], str)
        or type(signature[1]) is not int
        for time, signature in value.items()
    ):
        raise ValueError(f"Invalid {field} for {performance_key}")
    return {time: signature.copy() for time, signature in value.items()}


def _require_directory(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    return path


def _parse_robust_flag(value: str | None, performance_key: str) -> bool | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid robust_note_alignment for {performance_key}") from exc
    if number not in (0.0, 1.0):
        raise ValueError(f"Invalid robust_note_alignment for {performance_key}")
    return number == 1.0


def _read_note_alignment_entries(root: Path) -> dict[str, _NoteAlignmentEntry]:
    """(n)ASAP metadata.csv에서 연주별 match 파일과 robust 표시를 읽는다."""
    metadata_path = root / "metadata.csv"
    _require_file(metadata_path)
    entries: dict[str, _NoteAlignmentEntry] = {}
    with metadata_path.open(encoding="utf-8-sig", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if not {"midi_performance", "match_file"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing note alignment columns in {metadata_path}")
        for row in reader:
            key = row["midi_performance"]
            if not key or key in entries:
                raise ValueError(f"Missing or duplicate performance key in {metadata_path}")

            match_path = None
            if row["match_file"]:
                relative_path = Path(row["match_file"])
                candidate = (root / relative_path).resolve()
                if relative_path.is_absolute() or not candidate.is_relative_to(root):
                    raise ValueError(f"Match path is outside (n)ASAP root: {row['match_file']}")
                # metadata에는 있지만 저장소에 파일이 없는 연주가 있다(3개).
                if candidate.is_file():
                    match_path = candidate
            entries[key] = _NoteAlignmentEntry(
                key=key,
                match_path=match_path,
                robust=_parse_robust_flag(row.get("robust_note_alignment"), key),
            )
    return entries


def _map_note_alignments(
    rows: dict[str, dict[str, str]], entries: dict[str, _NoteAlignmentEntry]
) -> dict[str, _NoteAlignmentEntry]:
    """ASAP 연주 키를 (n)ASAP 정렬 항목에 대응시킨다.

    (n)ASAP은 반복을 다르게 연주한 일부 연주를 ``<작품>_no_repeat``나
    ``<작품>_extra_repeat`` 같은 형제 폴더로 옮겼다. 같은 키가 없으면, 파일 이름이 같고
    폴더 이름이 ``<원래 폴더>_``로 시작하는 항목이 정확히 하나일 때만 대응시킨다.
    """
    by_parent_and_name: dict[tuple[str, str], list[_NoteAlignmentEntry]] = {}
    for key, entry in entries.items():
        path = Path(key)
        by_parent_and_name.setdefault(
            (path.parent.parent.as_posix(), path.name), []
        ).append(entry)

    mapping = {}
    for key in rows:
        if key in entries:
            mapping[key] = entries[key]
            continue
        path = Path(key)
        candidates = [
            entry
            for entry in by_parent_and_name.get((path.parent.parent.as_posix(), path.name), [])
            if Path(entry.key).parent.name.startswith(f"{path.parent.name}_")
        ]
        if len(candidates) == 1:
            mapping[key] = candidates[0]
    return mapping


class ASAPLoader:
    """메타데이터와 annotation을 한 번 읽고 샘플별 정보를 제공한다."""

    def __init__(
        self, root: str | Path, note_alignment_root: str | Path | None = None
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        if not self.root.is_dir():
            raise ValueError(f"Not an ASAP directory: {self.root}")

        metadata_path = self.root / "metadata.csv"
        annotations_path = self.root / "asap_annotations.json"
        _require_file(metadata_path)
        _require_file(annotations_path)

        with metadata_path.open(encoding="utf-8-sig", newline="") as metadata_file:
            reader = csv.DictReader(metadata_file)
            required_columns = {"midi_score", "midi_performance"}
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing MIDI columns in {metadata_path}")
            self._rows: dict[str, dict[str, str]] = {}
            for row in reader:
                performance_key = row["midi_performance"]
                if not performance_key or performance_key in self._rows:
                    raise ValueError(f"Missing or duplicate performance key in {metadata_path}")
                self._rows[performance_key] = row

        try:
            with annotations_path.open(encoding="utf-8") as annotations_file:
                self._annotations = json.load(annotations_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid ASAP annotations: {annotations_path}") from exc
        if not isinstance(self._annotations, dict):
            raise ValueError(f"Invalid ASAP annotations: {annotations_path}")

        self.note_alignment_root: Path | None = None
        self._note_alignments: dict[str, _NoteAlignmentEntry] = {}
        if note_alignment_root is not None:
            self.note_alignment_root = _require_directory(note_alignment_root)
            self._note_alignments = _map_note_alignments(
                self._rows, _read_note_alignment_entries(self.note_alignment_root)
            )

    def _resolve_midi_path(self, value: str | None) -> Path:
        if not value:
            raise ValueError("Missing MIDI path in metadata.csv")
        relative_path = Path(value)
        path = (self.root / relative_path).resolve()
        if relative_path.is_absolute() or not path.is_relative_to(self.root):
            raise ValueError(f"MIDI path is outside ASAP root: {value}")
        _require_file(path)
        return path

    def get_sample(self, performance_key: str) -> AsapSample:
        """metadata.csv의 연주 MIDI 경로로 샘플 하나를 찾는다."""
        if performance_key not in self._rows:
            raise KeyError(f"Unknown performance: {performance_key}")
        if performance_key not in self._annotations:
            raise KeyError(f"Missing ASAP annotation: {performance_key}")

        row = self._rows[performance_key]
        annotation = self._annotations[performance_key]
        if not isinstance(annotation, dict):
            raise ValueError(f"Invalid ASAP annotation: {performance_key}")
        aligned = annotation.get("score_and_performance_aligned")
        if not isinstance(aligned, bool):
            raise ValueError(f"Invalid alignment flag for {performance_key}")

        score_beats = _read_times(annotation, "midi_score_beats", performance_key)
        performance_beats = _read_times(annotation, "performance_beats", performance_key)
        score_beat_types = _read_beat_types(
            annotation, score_beats, "midi_score_beats_type", performance_key
        )
        performance_beat_types = _read_beat_types(
            annotation, performance_beats, "performance_beats_type", performance_key
        )
        score_downbeats = _read_times(annotation, "midi_score_downbeats", performance_key)
        performance_downbeats = _read_times(annotation, "performance_downbeats", performance_key)
        if aligned and (
            len(score_beats) != len(performance_beats)
            or len(score_downbeats) != len(performance_downbeats)
        ):
            raise ValueError(f"Mismatched aligned beats for {performance_key}")

        performance_path = self._resolve_midi_path(row["midi_performance"])
        entry = self._note_alignments.get(performance_key)
        if entry is not None:
            self._check_same_performance(performance_path, entry)

        # ASAP의 박자표 값은 시각 문자열을 키로 하는 원본 형식을 유지한다.
        return AsapSample(
            performance_key=performance_key,
            composer=row.get("composer") or "",
            title=row.get("title") or "",
            score_path=self._resolve_midi_path(row["midi_score"]),
            performance_path=performance_path,
            aligned=aligned,
            score_beats=score_beats,
            performance_beats=performance_beats,
            score_beat_types=score_beat_types,
            performance_beat_types=performance_beat_types,
            score_downbeats=score_downbeats,
            performance_downbeats=performance_downbeats,
            score_time_signatures=_read_time_signatures(
                annotation, "midi_score_time_signatures", performance_key
            ),
            performance_time_signatures=_read_time_signatures(
                annotation, "perf_time_signatures", performance_key
            ),
            note_alignment_path=entry.match_path if entry else None,
            robust_note_alignment=entry.robust if entry else None,
        )

    def _check_same_performance(self, performance_path: Path, entry: _NoteAlignmentEntry) -> None:
        """(n)ASAP 쪽에도 연주 MIDI가 있으면 같은 파일인지 확인한다."""
        assert self.note_alignment_root is not None
        other_path = self.note_alignment_root / entry.key
        if other_path.is_file() and not filecmp.cmp(performance_path, other_path, shallow=False):
            raise ValueError(
                f"Performance MIDI differs from (n)ASAP: {performance_path} != {other_path}"
            )

    def iter_samples(self, aligned_only: bool = False) -> Iterator[AsapSample]:
        """메타데이터 순서로 샘플을 순회하며 필요하면 미정렬 샘플을 제외한다."""
        for performance_key in self._rows:
            sample = self.get_sample(performance_key)
            if not aligned_only or sample.aligned:
                yield sample
