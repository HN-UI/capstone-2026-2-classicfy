"""Controlled value-loss / delta-loss / wider-encoder experiment.

The production model and checkpoints stay unchanged. Three paired random seeds,
identical works/windows/training masks, fixed evaluation masks, and final-epoch
comparison. Test data is never evaluated. Large checkpoints live in datasets/.
"""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding.data import CHANNELS, WindowDataset, make_hidden_mask, restore_sequences
from embedding.model import ModelConfig, TemporalAutoencoder, TemporalBlock, error_totals, feature_loss
from embedding.training import TrainingConfig, fixed_masks, loader
from embedding.evaluation import temporal_metrics, baseline_predictions
from validate_embedding import collect_groups
from validate_temporal_embedding import predictions, feature_average
from analyze_temporal_failures import (SHORT, NAMES, PATTERNS, PATTERN_NAMES, canvas, finish, sha,
                                      json_scalar, SPLIT_NAMES)
from validate_tempo import plt, setup_style


ARMS = ("baseline", "delta_loss", "wide_encoder")
LABELS = {"baseline": "기존 모델", "delta_loss": "변화량 오차 추가", "wide_encoder": "CNN 폭 2배"}
COLORS = {"baseline": "#687785", "delta_loss": "#267DC0", "wide_encoder": "#D78332"}
DELTA_WEIGHT = .5


class EncoderExperiment(TemporalAutoencoder):
    """Change encoder width without changing latent size or decoder architecture.

    The baseline constructor gives all arms the same initial decoder weights for
    a given seed. The two 64-wide arms are the exact production architecture.
    """

    def __init__(self, config, encoder_width=64):
        super().__init__(config)
        if encoder_width != config.hidden_size:
            self.input_projection = nn.Conv1d(21, encoder_width, 1)
            self.blocks = nn.Sequential(*(TemporalBlock(encoder_width, 2 ** i, config.dropout)
                                          for i in range(config.blocks)))
            self.bottleneck = nn.Sequential(nn.Flatten(), nn.Linear(encoder_width * config.window_size, config.embedding_size))
        self.encoder_width = encoder_width


def delta_totals(prediction, target, hidden):
    """Only genuinely adjacent pairs with both endpoints hidden contribute."""
    if prediction.shape != target.shape or hidden.shape != target.shape or hidden.dtype != torch.bool:
        raise ValueError("Matching prediction/target/boolean hidden tensors required")
    selected = hidden[..., 1:] & hidden[..., :-1]
    difference = torch.diff(prediction, dim=-1) - torch.diff(target, dim=-1)
    squared = torch.where(selected, difference, 0).square()
    return squared.sum(dim=(0, 2)), selected.sum(dim=(0, 2))


def training_loss(prediction, target, hidden, delta_weight):
    value = feature_loss(*error_totals(prediction, target, hidden))
    sums, counts = delta_totals(prediction, target, hidden)
    delta = feature_loss(sums, counts) if torch.any(counts) else prediction.sum() * 0
    # Normalization avoids simply multiplying the entire gradient scale.
    return (value + delta_weight * delta) / (1 + delta_weight)


def evaluate_loss(model, dataset, masks, config):
    model.eval()
    totals = [torch.zeros(7, dtype=torch.float64), torch.zeros(7, dtype=torch.int64),
              torch.zeros(7, dtype=torch.float64), torch.zeros(7, dtype=torch.int64)]
    with torch.no_grad():
        for values, valid, indices in loader(dataset, config):
            hidden = masks[indices]
            prediction, _ = model(values, valid, hidden)
            for i, part in enumerate((*error_totals(prediction, values, hidden), *delta_totals(prediction, values, hidden))):
                totals[i] += part.double() if i % 2 == 0 else part
    return {"value_mse": float(feature_loss(*totals[:2])), "delta_mse": float(feature_loss(*totals[2:]))}


def save(path, model, config, model_config, seed, arm, epoch, score):
    torch.save({"state_dict": model.state_dict(), "model_config": asdict(model_config),
                "training_config": asdict(config), "seed": seed, "arm": arm,
                "encoder_width": model.encoder_width, "epoch": epoch, "score": score}, path)


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = EncoderExperiment(ModelConfig(**checkpoint["model_config"]), checkpoint["encoder_width"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def train_arm(train, validation, masks, config, model_config, seed, arm, directory):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "training.json"
    if result_path.exists():
        print(f"Reuse completed {seed}/{arm}", flush=True)
        return json.loads(result_path.read_text())
    if any(directory.iterdir()):
        raise ValueError(f"Unfinished experiment directory; preserve it and choose a new --runs path: {directory}")
    torch.manual_seed(seed)
    model = EncoderExperiment(model_config, 128 if arm == "wide_encoder" else 64)
    delta_weight = DELTA_WEIGHT if arm == "delta_loss" else 0.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    shuffle_rng = torch.Generator().manual_seed(seed + 1)
    mask_rng = torch.Generator().manual_seed(seed + 3)
    initial = evaluate_loss(model, validation, masks, config)
    save(directory / "initial.pt", model, config, model_config, seed, arm, 0, initial)
    best, best_epoch = initial["value_mse"], 0
    save(directory / "best_value.pt", model, config, model_config, seed, arm, 0, initial)
    # Separate generators make batch order and training masks independent of model
    # size, dropout RNG consumption, and loss choice. Hash the actual schedule.
    schedule_hash = hashlib.sha256()
    fields = ("epoch", "train_objective", "validation_value_mse", "validation_delta_mse", "clip_fraction", "seconds")
    start_time = time.perf_counter()
    with (directory / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"epoch": 0, "validation_value_mse": initial["value_mse"], "validation_delta_mse": initial["delta_mse"]})
        for epoch in range(1, config.epochs + 1):
            tick = time.perf_counter()
            model.train()
            losses, clipped = [], 0
            for values, valid, indices in loader(train, config, shuffle=True, generator=shuffle_rng):
                hidden = make_hidden_mask(valid, fraction=config.mask_fraction,
                                          block_size=config.mask_block_size, generator=mask_rng)
                schedule_hash.update(indices.numpy().tobytes())
                schedule_hash.update(hidden.numpy().tobytes())
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(values, valid, hidden)
                loss = training_loss(prediction, values, hidden, delta_weight)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite experiment loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                clipped += int(norm > 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            score = evaluate_loss(model, validation, masks, config)
            seconds = time.perf_counter() - tick
            writer.writerow({"epoch": epoch, "train_objective": float(np.mean(losses)),
                             "validation_value_mse": score["value_mse"], "validation_delta_mse": score["delta_mse"],
                             "clip_fraction": clipped / len(losses), "seconds": seconds})
            handle.flush()
            if score["value_mse"] < best:
                best, best_epoch = score["value_mse"], epoch
                save(directory / "best_value.pt", model, config, model_config, seed, arm, epoch, score)
            print(f"{seed} {arm:12s} {epoch:02d}/{config.epochs}: value={score['value_mse']:.6f} delta={score['delta_mse']:.6f} ({seconds:.1f}s)", flush=True)
        save(directory / "final.pt", model, config, model_config, seed, arm, config.epochs, score)
    result = {"seed": seed, "arm": arm, "epochs": config.epochs, "encoder_width": model.encoder_width,
              "embedding_size": model_config.embedding_size, "decoder_width": model_config.hidden_size * 2,
              "parameters": sum(p.numel() for p in model.parameters()),
              "encoder_parameters": sum(p.numel() for n, p in model.named_parameters() if not n.startswith("decoder.")),
              "decoder_parameters": sum(p.numel() for p in model.decoder.parameters()),
              "delta_weight": delta_weight, "final_validation": score, "best_value_epoch": best_epoch,
              "best_validation_value_mse": best, "training_schedule_sha256": schedule_hash.hexdigest(),
              "seconds": time.perf_counter() - start_time, "final_sha256": sha(directory / "final.pt")}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def channel_metrics(target, prediction, hidden):
    rows = []
    for ci, channel in enumerate(CHANNELS):
        metrics = temporal_metrics(target[:, ci], prediction[:, ci], hidden[:, ci])
        pair = hidden[:, ci, 1:] & hidden[:, ci, :-1]
        actual_delta = np.diff(target[:, ci].astype(float), axis=-1)[pair]
        predicted_delta = np.diff(prediction[:, ci].astype(float), axis=-1)[pair]
        denominator = np.sum(actual_delta ** 2)
        metrics["step_amplitude_percent"] = float(100 * np.sqrt(np.sum(predicted_delta ** 2) / denominator)) if denominator > 1e-12 else None
        metrics["direction_percent"] = metrics["direction_accuracy"] * 100 if metrics["direction_accuracy"] is not None else None
        rows.append({"channel": channel, **metrics})
    return rows


def aggregate_metrics(rows):
    return {metric: feature_average(rows, metric) for metric in ("mse", "slope_mse", "step_amplitude_percent", "direction_percent")}


def paired_effects(table, keys):
    rows = []
    for values, group in table.groupby(keys):
        if not isinstance(values, tuple):
            values = (values,)
        identifiers = dict(zip(keys, values))
        reference = group[group.arm == "baseline"].iloc[0]
        for arm in ARMS[1:]:
            row = group[group.arm == arm].iloc[0]
            rows.append({**identifiers, "arm": arm,
                         "value_gain_percent": 100 * (1 - row.mse / reference.mse),
                         "delta_gain_percent": 100 * (1 - row.slope_mse / reference.slope_mse),
                         "amplitude_change_pp": row.step_amplitude_percent - reference.step_amplitude_percent,
                         "direction_change_pp": row.direction_percent - reference.direction_percent})
    return pd.DataFrame(rows)


def plot_overview(summary, effects, histories, channel_effects, out, seeds, epochs):
    specifications = (("mse", "숨긴 값의 오차", "값 MSE ↓", None),
                      ("slope_mse", "숨긴 인접 beat 변화량의 오차", "변화량 MSE ↓", None),
                      ("step_amplitude_percent", "실제 변화 크기의 몇 %를 예측했나?", "예측/실제 변화 RMS (%)", 100),
                      ("direction_percent", "상승·하강 방향을 얼마나 맞췄나?", "방향 일치 (%) ↑", 50))
    fig, axes = canvas("학습 목표와 CNN 용량, 무엇을 바꾸면 나아지는가?",
                       f"{len(seeds)}개 seed · 모두 {epochs} epoch 최종 모델 · 점=seed 평균 / 선=seed 최솟값~최댓값 · 다섯 feature 동일 비중", 2, 2, (17, 11))
    x = np.arange(3)
    for ax, (metric, title, ylabel, reference) in zip(axes.flat, specifications):
        for split, offset, color in (("train", -.10, "#2A8D78"), ("validation", .10, "#CF6A43")):
            sub = summary[summary.split == split]
            mean = sub.groupby("arm")[metric].mean().reindex(ARMS)
            low = sub.groupby("arm")[metric].min().reindex(ARMS)
            high = sub.groupby("arm")[metric].max().reindex(ARMS)
            ax.errorbar(x + offset, mean, yerr=np.array([mean-low, high-mean]), fmt="o-", color=color,
                        lw=2, ms=7, capsize=5, label=SPLIT_NAMES[split])
        ax.set_xticks(x, [LABELS[a] for a in ARMS])
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(frameon=False, fontsize=10)
        if reference is not None:
            ax.axhline(reference, color="#555", ls="--", lw=1)
            ax.set_ylim(0, 110 if metric == "step_amplitude_percent" else 100)
    finish(fig, out / "01_controlled_comparison",
           "기존: 값 오차만 학습. 목표 변경: (값 MSE + 0.5 × 인접 변화량 MSE) / 1.5. 용량 변경: CNN 폭 64→128, 임베딩 64D·복원기 동일.\n"
           "변화량은 인접한 두 beat가 모두 숨겨졌을 때만 평가. 방향 평가에서는 실제 변화 0을 제외. 아래 두 지표는 채널별 비율의 5-feature 평균.\n"
           "같은 작품 split·고정 평가 마스크를 사용. 범위는 신뢰구간이 아니며, 이 한 설정의 비교로 전체 원인을 확정할 수 없음.", top=.83, bottom=.18, hspace=.58)

    fig, axes = canvas("검증 데이터에서 기존 모델 대비 얼마나 개선됐나?",
                       "같은 seed끼리 비교한 뒤 평균 · +는 개선 · 선은 세 seed의 최솟값~최댓값", 2, 2, (17, 10))
    specs = (("value_gain_percent", "값 복원 개선", "기존 대비 값 MSE 감소율 (%)"),
             ("delta_gain_percent", "변화량 복원 개선", "기존 대비 변화량 MSE 감소율 (%)"),
             ("amplitude_change_pp", "예측 변화 크기 증가", "예측/실제 비율의 변화 (%p)"),
             ("direction_change_pp", "방향 일치 개선", "방향 일치율 변화 (%p)"))
    for ax, (metric, title, ylabel) in zip(axes.flat, specs):
        for arm, offset in (("delta_loss", -.18), ("wide_encoder", .18)):
            sub = channel_effects[(channel_effects.split == "validation") & (channel_effects.arm == arm)]
            grouped = sub.groupby("channel")[metric]
            mean, low, high = (getattr(grouped, stat)().reindex(CHANNELS) for stat in ("mean", "min", "max"))
            ax.bar(np.arange(7)+offset, mean, .34, color=COLORS[arm], label=LABELS[arm])
            ax.errorbar(np.arange(7)+offset, mean, yerr=np.array([mean-low, high-mean]), fmt="none", color="#33414B", capsize=3, lw=1)
        ax.axhline(0, color="#555", ls="--", lw=1)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=14)
        ax.set_xticks(np.arange(7), NAMES, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=10, frameon=False)
    finish(fig, out / "02_validation_feature_effects",
           "값·변화량 오차 감소와 변화 크기 증가는 별개. 더 크게 출렁여도 방향·변화량 오차가 좋아지지 않으면 흐름 복원 개선으로 판단하지 않음.\n"
           "학습 목표 변경은 λ=0.5 한 가지, CNN 용량 변경은 폭 2배 한 가지를 사전 고정한 탐색 실험.", top=.82, bottom=.16, hspace=.58)

    fig, axes = canvas("학습을 진행하면서 값과 변화량 오차가 어떻게 달라졌나?",
                       "검증 데이터 · 각 epoch의 seed 평균 / 음영은 seed 최솟값~최댓값 · 초기 모델은 epoch 0", 1, 2, (17, 7))
    for ax, metric, title in zip(axes.flat, ("validation_value_mse", "validation_delta_mse"), ("숨긴 값의 오차", "숨긴 인접 변화량의 오차")):
        for arm in ARMS:
            sub = histories[histories.arm == arm].groupby("epoch")[metric]
            mean, low, high = sub.mean(), sub.min(), sub.max()
            ax.plot(mean.index, mean, color=COLORS[arm], lw=2, label=LABELS[arm])
            ax.fill_between(mean.index, low, high, color=COLORS[arm], alpha=.12)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=15)
        ax.set_xlabel("학습 epoch")
        ax.set_ylabel("MSE ↓")
        ax.legend(fontsize=11, frameon=False)
    finish(fig, out / "03_learning_curves",
           "학습 목표가 달라져도 평가하는 값 MSE·변화량 MSE의 정의는 동일. 서로 다른 학습 목적함수 숫자를 직접 비교하지 않음.\n"
           "최선 epoch를 조건마다 고르면 선택 기준의 영향이 섞이므로 주 비교는 모두 20 epoch 최종 모델로 고정.", top=.80, bottom=.19)


def plot_examples(datasets, tensors, prediction_store, selections, out, seed):
    directory = out / "examples"
    directory.mkdir(exist_ok=True)
    records = []
    for ci, channel in enumerate(CHANNELS):
        fig, axes = canvas(f"{SHORT[ci]}: 학습 목표·용량을 바꾸면 곡선이 달라지는가?",
                           f"첫 seed {seed} · 이전 진단에서 고정한 동일 구간 · 왼쪽 학습 / 오른쪽 검증 · 회색은 숨긴 위치", 4, 2, (17, 15))
        for pi, pattern in enumerate(PATTERNS):
            for sj, split in enumerate(("train", "validation")):
                ax = axes[pi, sj]
                candidate = selections[(selections.split == split) & (selections.channel == channel) & (selections.pattern == pattern)]
                if candidate.empty:
                    ax.axis("off")
                    continue
                row = candidate.iloc[0]
                wi, start, stop = (int(row[k]) for k in ("window", "start", "stop"))
                dataset = datasets[split]
                si, offset = dataset.items[wi]
                sequence = dataset.sequences[si]
                if sequence.key != row.key or offset + start != row.beat:
                    raise ValueError("Example identity changed from previous diagnostics")
                target = tensors[split][0].numpy()
                beats = offset + np.arange(start-1, stop+1)
                ax.axvspan(offset+start-.45, offset+stop-.55, color="#E8ECEF", zorder=0)
                ax.plot(beats, target[wi,ci,start-1:stop+1], "o-", color="#26343D", lw=2, ms=4, label="실제 값")
                ax.scatter(beats[[0,-1]], target[wi,ci,[start-1,stop]], marker="s", color="#26343D", s=65, zorder=4)
                for arm in ARMS:
                    prediction = prediction_store[(seed,arm,split)]
                    ax.plot(offset+np.arange(start,stop), prediction[wi,ci,start:stop], "o-", color=COLORS[arm], lw=1.8, ms=4, label=LABELS[arm])
                ax.set_title(f"{PATTERN_NAMES[pi]} · {SPLIT_NAMES[split]}", fontsize=12, loc="left", fontweight="bold", pad=22)
                ax.text(0,1.025, sequence.key.replace("Piano_Sonatas/", "Sonata/"), transform=ax.transAxes, fontsize=8, color="#657078")
                ax.set_xticks(beats)
                ax.set_xlabel("원래 beat 구간 번호", fontsize=9)
                ax.set_ylabel("정규화된 상대 값", fontsize=9)
                for b in range(start,stop):
                    records.append({"seed":seed,"split":split,"channel":channel,"pattern":pattern,"key":sequence.key,
                                    "beat":offset+b,"target":float(target[wi,ci,b]),
                                    **{a:float(prediction_store[(seed,a,split)][wi,ci,b]) for a in ARMS}})
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.887),ncol=4,frameon=False,fontsize=11)
        finish(fig,directory/f"{ci+1:02d}_{channel}",
               "예측 성공·실패와 무관하게 이전 분석에서 실제 변화 특성이 중앙에 가까운 구간으로 선정. 세 모델에 동일한 입력과 가림을 적용.\n"
               "첫 seed의 사례이며 전체 결론은 세 seed·전체 구간 집계로 판단.",top=.82,bottom=.105,hspace=.88)
    pd.DataFrame(records).to_csv(out/"example_beats.csv",index=False)


def main():
    ai = Path(__file__).resolve().parents[1]
    data = ai.parent.parent / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=data/"temporal_embedding_cluster_run01")
    parser.add_argument("--cache",type=Path,default=data/"feature_normalization_cluster_raw.npz")
    parser.add_argument("--asap-root",type=Path,default=data/"ASAP")
    parser.add_argument("--nasap-root",type=Path,default=data/"nASAP")
    parser.add_argument("--runs",type=Path,default=data/"temporal_controlled_run01")
    parser.add_argument("--out",type=Path,default=ai/"analysis/temporal_ablation")
    parser.add_argument("--examples",type=Path,default=ai/"analysis/temporal_diagnostics/selected_examples.csv")
    parser.add_argument("--seeds",type=int,nargs="+",default=[20261006,20261007,20261008])
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--threads",type=int,default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive epochs/threads and unique seeds required")
    torch.set_num_threads(args.threads)
    stored=json.loads((args.reference/"dataset.json").read_text())
    info=json.loads((args.reference/"training.json").read_text())
    if sha(args.cache)!=stored["cache_sha256"]:
        raise ValueError("Cache differs from reference")
    source_files = [Path(__file__), ai/"src/embedding/model.py",ai/"src/embedding/data.py", ai/"src/embedding/training.py", ai/"src/embedding/evaluation.py"]
    protocol={"seeds":args.seeds,"epochs":args.epochs,"delta_weight":DELTA_WEIGHT,"arms":list(ARMS),
              "reference":str(args.reference.resolve()),"cache_sha256":sha(args.cache),"reference_dataset_sha256":sha(args.reference/"dataset.json"),
              "source_sha256":{str(p.relative_to(ai)):sha(p) for p in source_files},
              "primary_comparison":"final epoch for all arms; no early stopping; test untouched",
              "encoder_widths":{"baseline":64,"delta_loss":64,"wide_encoder":128},"embedding_size":64,"decoder_width":128,
              "evaluation_mask_seed":info["training_config"]["seed"]+2,
              "training_schedule":"separate torch generators: shuffle seed+1, masks seed+3; schedule hashes must match within seed",
              "delta_definition":"adjacent pairs with both beats hidden and valid; equal five-feature weighting",
              "objective":"(value_MSE + lambda*delta_MSE)/(1+lambda); lambda=.5 only in delta_loss arm",
              "limits":["One work split, three seeds, one lambda, one width, fixed 20 epoch budget.",
                        "Work identities follow original score/movement split; not guaranteed disjoint musical families.",
                        "Three-seed min/max ranges are not confidence intervals.",
                        "Reconstruction tests do not validate listening similarity or recommendation quality."]}
    args.runs.mkdir(parents=True,exist_ok=True)
    protocol_path=args.runs/"protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:
            raise ValueError("Existing experiment protocol differs; choose a new --runs directory")
    else:
        protocol_path.write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    print("Rebuilding and checking reference inputs",flush=True)
    groups,audits,scales,provenance=collect_groups(SimpleNamespace(asap_root=args.asap_root.resolve(),nasap_root=args.nasap_root.resolve(),cache=args.cache,**stored["cohort_settings"]))
    if provenance!=stored["source_provenance"] or scales!=stored["scales"]:
        raise ValueError("Reference feature provenance/scales changed")
    sequences=restore_sequences(groups,audits)
    datasets={split:WindowDataset([s for s in sequences if stored["split"][s.piece]==split],window_size=64,stride=stored["stride"]) for split in ("train","validation")}
    config=TrainingConfig(**dict(info["training_config"],epochs=args.epochs,patience=args.epochs))
    model_config=ModelConfig(**info["model_config"])
    tensors,masks={},{}
    for split,dataset in datasets.items():
        batches=list(DataLoader(dataset,batch_size=config.batch_size,shuffle=False))
        tensors[split]=(torch.cat([b[0] for b in batches]),torch.cat([b[1] for b in batches]))
        masks[split]=fixed_masks(dataset,config)
    print({s:len(d) for s,d in datasets.items()},flush=True)
    results=[]
    for seed in args.seeds:
        seed_results=[]
        for arm in ARMS:
            seed_results.append(train_arm(datasets["train"],datasets["validation"],masks["validation"],config,model_config,seed,arm,args.runs/str(seed)/arm))
        if len({r["training_schedule_sha256"] for r in seed_results})!=1:
            raise AssertionError("Training order/masks differ between arms")
        results.extend(seed_results)
    args.out.mkdir(parents=True,exist_ok=True)
    metrics,summary,work_metrics,histories=[],[],[],[]
    prediction_store={}
    for seed in args.seeds:
        for arm in ARMS:
            directory=args.runs/str(seed)/arm
            model,checkpoint=load(directory/"final.pt")
            if checkpoint["epoch"]!=args.epochs:
                raise ValueError("Wrong final-epoch checkpoint")
            histories.append(pd.read_csv(directory/"history.csv").assign(seed=seed,arm=arm))
            for split,dataset in datasets.items():
                values,valid=tensors[split]
                hidden=masks[split]
                prediction=predictions(model,values,valid,hidden,batch_size=config.batch_size)
                if seed==args.seeds[0]:
                    prediction_store[(seed,arm,split)]=prediction
                rows=channel_metrics(values.numpy(),prediction,hidden.numpy())
                metrics.extend(dict(seed=seed,arm=arm,split=split,**r) for r in rows)
                summary.append(dict(seed=seed,arm=arm,split=split,**aggregate_metrics(rows)))
                works=np.array([dataset.sequences[si].piece for si,_ in dataset.items])
                for piece in sorted(set(works)):
                    keep=works==piece
                    work_rows=channel_metrics(values.numpy()[keep],prediction[keep],hidden.numpy()[keep])
                    work_metrics.extend(dict(seed=seed,arm=arm,split=split,piece=piece,**r) for r in work_rows)
    metrics,summary,work_metrics,histories=(pd.DataFrame(metrics),pd.DataFrame(summary),pd.DataFrame(work_metrics),pd.concat(histories,ignore_index=True))
    effects=paired_effects(summary,["seed","split"])
    channel_effects=paired_effects(metrics,["seed","split","channel"])
    work_effects=paired_effects(work_metrics,["seed","split","piece","channel"])
    for name,table in (("channel_metrics",metrics),("summary",summary),("work_metrics",work_metrics),("history",histories),
                       ("paired_effects",effects),("channel_effects",channel_effects),("work_effects",work_effects),("training_runs",pd.DataFrame(results))):
        table.to_csv(args.out/f"{name}.csv",index=False)
    setup_style()
    plt.rcParams.update({"svg.fonttype":"path","font.size":11})
    plot_overview(summary,effects,histories,channel_effects,args.out,args.seeds,args.epochs)
    selections=pd.read_csv(args.examples)
    plot_examples(datasets,tensors,prediction_store,selections,args.out,args.seeds[0])
    final_manifest={**protocol,"source_provenance":provenance,"runs_path":str(args.runs.resolve()),
                    "run_results":results,"evaluation_masks_sha256":{s:hashlib.sha256(m.numpy().tobytes()).hexdigest() for s,m in masks.items()},
                    "examples_sha256":sha(args.examples),"counts":{s:{"windows":len(d),"performances":len(d.sequences),"works":len({v.piece for v in d.sequences})} for s,d in datasets.items()},
                    "figure_files":sorted(str(p.relative_to(args.out)) for p in args.out.rglob("*.png"))}
    (args.out/"stats.json").write_text(json.dumps(final_manifest,ensure_ascii=False,indent=2,default=json_scalar),encoding="utf-8")
    print("Final-epoch validation results:",flush=True)
    print(summary[summary.split=="validation"].to_string(index=False),flush=True)
    print("Paired effects:",flush=True)
    print(effects[effects.split=="validation"].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
