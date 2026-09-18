"""ASAP의 악보-연주 짝과 정렬 정보를 읽는다."""

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


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
    score_downbeats: list[float]
    performance_downbeats: list[float]
    score_time_signatures: dict[str, list[str | int]]
    performance_time_signatures: dict[str, list[str | int]]


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


class ASAPLoader:
    """메타데이터와 annotation을 한 번 읽고 샘플별 정보를 제공한다."""

    def __init__(self, root: str | Path) -> None:
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
        score_downbeats = _read_times(annotation, "midi_score_downbeats", performance_key)
        performance_downbeats = _read_times(annotation, "performance_downbeats", performance_key)
        if aligned and (
            len(score_beats) != len(performance_beats)
            or len(score_downbeats) != len(performance_downbeats)
        ):
            raise ValueError(f"Mismatched aligned beats for {performance_key}")

        # ASAP의 박자표 값은 시각 문자열을 키로 하는 원본 형식을 유지한다.
        return AsapSample(
            performance_key=performance_key,
            composer=row.get("composer") or "",
            title=row.get("title") or "",
            score_path=self._resolve_midi_path(row["midi_score"]),
            performance_path=self._resolve_midi_path(row["midi_performance"]),
            aligned=aligned,
            score_beats=score_beats,
            performance_beats=performance_beats,
            score_downbeats=score_downbeats,
            performance_downbeats=performance_downbeats,
            score_time_signatures=_read_time_signatures(
                annotation, "midi_score_time_signatures", performance_key
            ),
            performance_time_signatures=_read_time_signatures(
                annotation, "perf_time_signatures", performance_key
            ),
        )

    def iter_samples(self, aligned_only: bool = False) -> Iterator[AsapSample]:
        """메타데이터 순서로 샘플을 순회하며 필요하면 미정렬 샘플을 제외한다."""
        for performance_key in self._rows:
            sample = self.get_sample(performance_key)
            if not aligned_only or sample.aligned:
                yield sample
