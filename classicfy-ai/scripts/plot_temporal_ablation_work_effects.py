"""Show paired effects per held-out score so long works cannot hide differences."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_temporal_ablation import aggregate_metrics, paired_effects, LABELS
from analyze_temporal_failures import canvas, sha
from validate_tempo import setup_style, plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "analysis/temporal_ablation")
    args = parser.parse_args()
    source = pd.read_csv(args.out / "work_metrics.csv")
    rows = []
    for (seed, arm, split, piece), group in source.groupby(["seed", "arm", "split", "piece"]):
        rows.append(dict(seed=seed, arm=arm, split=split, piece=piece,
                         **aggregate_metrics(group.to_dict("records"))))
    table = pd.DataFrame(rows)
    effects = paired_effects(table, ["seed", "split", "piece"])
    table.to_csv(args.out / "work_summary.csv", index=False)
    effects.to_csv(args.out / "work_summary_effects.csv", index=False)
    macro = table.groupby(["seed", "arm", "split"])[["mse", "slope_mse", "step_amplitude_percent", "direction_percent"]].mean().reset_index()
    macro.to_csv(args.out / "equal_work_summary.csv", index=False)
    validation = effects[effects.split == "validation"]
    works = sorted(validation.piece.unique())
    arms = ("delta_loss", "wide_encoder")
    setup_style()
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 11})
    fig, axes = canvas("검증 작품별로도 개선이 일관적인가?",
                       "같은 작품·같은 seed에서 기존 모델과 비교 · 셀: 세 seed 평균 개선율 / 개선된 seed 수", 1, 2, (17, 10))
    names = [p.replace("/midi_score.mid", "").replace("Piano_Sonatas", "Sonata").replace("Etudes_op_", "Etude op.").replace("/", " · ") for p in works]
    for ax, metric, title in zip(axes.flat, ("value_gain_percent", "delta_gain_percent"), ("가린 값의 오차 감소", "인접 변화량의 오차 감소")):
        matrix = np.full((len(works), 2), np.nan)
        for wi, work in enumerate(works):
            for ai, arm in enumerate(arms):
                values = validation[(validation.piece == work) & (validation.arm == arm)][metric]
                matrix[wi, ai] = values.mean()
        im = ax.imshow(matrix, cmap="RdBu", vmin=-8, vmax=8, aspect="auto")
        ax.grid(False)
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=15)
        ax.set_xticks(np.arange(2), [LABELS[a] for a in arms])
        ax.set_yticks(np.arange(len(works)), names, fontsize=10)
        for wi, work in enumerate(works):
            for ai, arm in enumerate(arms):
                values = validation[(validation.piece == work) & (validation.arm == arm)][metric]
                mean = values.mean()
                ax.text(ai, wi, f"{mean:+.2f}%\n개선 {int((values > 0).sum())}/{len(values)} seed",
                        ha="center", va="center", fontsize=10,
                        color="white" if abs(mean) > 5 else "#243039")
    cax = fig.add_axes([.925, .17, .014, .66])
    fig.colorbar(im, cax=cax, label="기존 대비 MSE 감소율 (%)", extend="both")
    note = ("+는 기존 모델보다 오차 감소, −는 오차 증가. 색 범위를 벗어나도 실제 숫자를 병기. 작품 내부에서는 다섯 feature 동일 비중.\n"
           "이 표는 원래 split의 악보·악장 단위 8개이며 음악 작품 family 전체를 분리한 실험은 아님. 세 seed의 반복은 작품 수를 늘리지 않음.\n"
           "모든 작품에 같은 비중을 준 별도 집계: equal_work_summary.csv. 곡 길이가 긴 작품의 영향과 구분해 확인.")
    fig.text(.055, .035, note, fontsize=10.5, color="#56616A", va="bottom", linespacing=1.55)
    fig.subplots_adjust(left=.19, right=.88, top=.83, bottom=.18, wspace=1.05)
    path = args.out / "04_validation_work_effects"
    fig.savefig(path.with_suffix(".png"), dpi=165)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    metadata_path = args.out / "stats.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["work_comparison_script_sha256"] = sha(__file__)
    metadata["figure_files"] = sorted(str(p.relative_to(args.out)) for p in args.out.rglob("*.png"))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(macro[macro.split == "validation"].groupby("arm")[["mse", "slope_mse", "step_amplitude_percent", "direction_percent"]].mean().to_string())


if __name__ == "__main__":
    main()
