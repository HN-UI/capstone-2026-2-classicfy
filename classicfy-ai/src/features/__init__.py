"""연주 해석 feature의 공개 인터페이스."""

from .articulation import (
    ArticulationFeature,
    NoteArticulation,
    build_tempo_map,
    extract_articulation,
    extract_note_articulation,
    summarize_articulation,
)
from .beat_grid import BeatGrid, as_beat_array, assign_windows, build_beat_grid
from .common_pattern import PieceFeatureInput, SeparatedFeature, separate_piece_feature
from .dynamics import DynamicsFeature, extract_dynamics, summarize_dynamics
from .models import BeatSequence
from .pedaling import PedalingFeature, extract_pedaling, summarize_pedaling
from .rubato import RubatoFeature, extract_piece_rubato_features, summarize_rubato
from .tempo import (
    TempoFeature,
    TempoInput,
    TempoInterval,
    extract_piece_tempo_features,
    summarize_tempo,
)

__all__ = [
    "ArticulationFeature",
    "BeatGrid",
    "BeatSequence",
    "DynamicsFeature",
    "NoteArticulation",
    "PedalingFeature",
    "PieceFeatureInput",
    "RubatoFeature",
    "SeparatedFeature",
    "TempoFeature",
    "TempoInput",
    "TempoInterval",
    "as_beat_array",
    "assign_windows",
    "build_beat_grid",
    "build_tempo_map",
    "extract_articulation",
    "extract_dynamics",
    "extract_note_articulation",
    "extract_pedaling",
    "extract_piece_rubato_features",
    "extract_piece_tempo_features",
    "separate_piece_feature",
    "summarize_articulation",
    "summarize_dynamics",
    "summarize_pedaling",
    "summarize_rubato",
    "summarize_tempo",
]
