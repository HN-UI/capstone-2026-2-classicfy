"""Experiment 1 (up to 80 epochs) and 3 (BiLSTM 64/128/256D), validation only.

Preserves all prior 20-epoch artifacts. Shared schedules are computed once per
seed, checked against the original experiment, and actual epoch masks are hashed.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding.extended_experiments import ARMS, build_model, make_schedule, train, load, dump, sha
from embedding.evaluation import baseline_predictions
from compare_sequence_autoencoders import prepare, SEEDS, metric_rows
from train_temporal_ablation import evaluate_loss
from validate_temporal_embedding import predictions, select_examples
from embedding.data import CHANNELS

BUDGET = ("cnn64", "bilstm64", "transformer64")
BOTTLENECK = ("bilstm64", "bilstm128", "bilstm256")


def original_directory(args, arm, seed):
    return args.cnn_runs / str(seed) / "baseline" if arm == "cnn64" else args.original_runs / arm.removesuffix("64") / str(seed)


def evaluate(args, prepared, available):
    stored, mc, base_config, sequences, datasets, arrays, masks = prepared
    dataset = datasets["validation"]
    x, v = arrays["validation"]
    hidden = masks["validation"]
    works = np.array([dataset.sequences[si].piece for si,_ in dataset.items])
    summaries, channels, work_rows, histories, run_rows = [], [], [], [], []
    selections = select_examples(dataset)
    example_predictions = {}
    for baseline, pred in baseline_predictions(x.numpy(), v.numpy(), hidden.numpy()).items():
        summary, detail, by_work = metric_rows(baseline, -1, "validation", "baseline", x.numpy(), pred, hidden.numpy(), works)
        summaries.append(summary); channels.extend(detail); work_rows.extend(by_work)
        example_predictions[baseline] = pred[selections]
    for arm in available:
        for seed in SEEDS:
            directory = args.runs / arm / str(seed)
            result = json.loads((directory / "training.json").read_text())
            run_rows.append(result)
            histories.append(pd.read_csv(directory / "history.csv").assign(arm=arm, seed=seed))
            for condition, filename in (("initial", "initial"), ("best20", "best20"),
                                        ("epoch20", "epoch20"), ("best_extended", "best"), ("final", "final")):
                model, checkpoint = load(directory / f"{filename}.pt")
                pred = predictions(model, x, v, hidden, batch_size=base_config.batch_size)
                summary, detail, by_work = metric_rows(arm, seed, "validation", condition,
                    x.numpy(), pred, hidden.numpy(), works)
                summary["epoch"] = checkpoint["epoch"]
                summaries.append(summary); channels.extend(detail); work_rows.extend(by_work)
                if seed == SEEDS[0] and condition in ("best20", "best_extended"):
                    example_predictions[f"{arm}_{condition}"] = pred[selections]
    examples = []
    for number, index in enumerate(selections, 1):
        si, start = dataset.items[index]
        for ci, channel in enumerate(CHANNELS):
            for beat in range(mc.window_size):
                examples.append(dict(example=number, key=dataset.sequences[si].key, channel=channel,
                    beat=start+beat, valid=bool(v[index,ci,beat]), hidden=bool(hidden[index,ci,beat]),
                    target=float(x[index,ci,beat]),
                    **{name:float(pred[number-1,ci,beat]) for name,pred in example_predictions.items()}))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary",summaries), ("channel_metrics",channels), ("work_metrics",work_rows), ("example_beats",examples)):
        pd.DataFrame(rows).to_csv(args.out / f"{name}.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(args.out / "history.csv", index=False)
    dump(args.out / "training_runs.json", run_rows)
    from plot_extended_sequences import render
    render(args.out)
    print(pd.DataFrame(summaries).query("condition == 'best_extended'")[["model","seed","epoch","mse","shape_correlation","shape_amplitude_ratio","slope_mse","direction_percent"]].to_string(index=False), flush=True)


def main():
    ai = Path(__file__).resolve().parents[1]
    data = ai.parent.parent / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=data / "temporal_embedding_cluster_run01")
    parser.add_argument("--cnn-runs", type=Path, default=data / "temporal_controlled_run01")
    parser.add_argument("--original-runs", type=Path, default=data / "sequence_autoencoders_run01")
    parser.add_argument("--runs", type=Path, default=data / "extended_sequences_run01")
    parser.add_argument("--out", type=Path, default=ai / "analysis/sequence_followup")
    parser.add_argument("--cache", type=Path, default=data / "feature_normalization_cluster_raw.npz")
    parser.add_argument("--asap-root", type=Path, default=data / "ASAP")
    parser.add_argument("--nasap-root", type=Path, default=data / "nASAP")
    parser.add_argument("--stage", choices=("all", "budget", "bottleneck", "report"), default="all")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive threads required")
    torch.set_num_threads(args.threads)
    prepared = prepare(args)
    stored, mc, base_config, sequences, datasets, arrays, masks = prepared
    config = replace(base_config, epochs=80, patience=10)
    sources = [Path(__file__), ai/"src/embedding/extended_experiments.py",
        ai/"src/embedding/sequence_models.py", ai/"src/embedding/model.py", ai/"src/embedding/data.py",
        ai/"src/embedding/training.py", ai/"src/embedding/evaluation.py",
        ai/"scripts/compare_sequence_autoencoders.py", ai/"scripts/train_temporal_ablation.py",
        ai/"scripts/validate_temporal_embedding.py", ai/"scripts/validate_embedding.py",
        ai/"scripts/validate_feature_normalization.py"]
    protocol = {"experiments": [1,3], "arms": ARMS, "seeds":list(SEEDS), "training_config":asdict(config),
        "minimum_epochs":20, "stop_rule":"minimum 20; stop after 10 consecutive epochs without strictly lower validation value MSE; maximum 80",
        "selection":"best validation value MSE within first 20 vs best within actual extended budget on the same trajectory",
        "test_policy":"No test metrics, test figure selection or test-driven tuning; existing validation used for development",
        "initialization":"64D is the exact original model. 128/256D construct the same 64D BiLSTM then replace only dimension-dependent Linear layers with default initialization under seed+1000; encoder and decoder output layer start identically across sizes.",
        "dimension_effect":"Decoder hidden width stays 128. Bottleneck and decoder input parameter counts necessarily increase; not an equal-parameter experiment or a proof of information capacity alone.",
        "schedule":"Same ordered batches and hidden targets for each epoch/seed across arms, including first 20 epochs of original experiments. Precomputed schedules replace repeated Python generation only.",
        "reference_dataset_sha256":sha(args.reference/"dataset.json"), "cache_sha256":sha(args.cache),
        "sources":{str(p.relative_to(ai)):sha(p) for p in sources},
        "runtime":{"torch":str(torch.__version__), "device":"cpu", "threads":args.threads},
        "counts":{s:dict(works=len({seq.piece for seq in d.sequences}), performances=len(d.sequences),windows=len(d)) for s,d in datasets.items() if s != "test"},
        "validation_mask_sha256":__import__("hashlib").sha256(masks["validation"].numpy().tobytes()).hexdigest(),
        "limits":["One previously observed validation split; three seeds, not confidence intervals.",
                  "Maximum budget is 80, not proof of convergence when best is near the cap.",
                  "Bottleneck sizes change head initialization and parameter counts.",
                  "Early stopping yields different actual epoch counts and different extra compute.",
                  "Reconstruction metrics do not validate human listening similarity."]}
    args.runs.mkdir(parents=True,exist_ok=True)
    protocol_path=args.runs/"protocol.json"
    # JSON round-trip normalizes tuple-valued ARMS before comparison.
    protocol=json.loads(json.dumps(protocol))
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Experiment sources/settings changed; preserve runs and use a new --runs")
    dump(protocol_path,protocol)
    args.out.mkdir(parents=True,exist_ok=True)
    dump(args.out/"protocol.json",protocol)
    cached_validation=TensorDataset(*arrays["validation"],torch.arange(len(datasets["validation"])))
    stages = (BUDGET, ("bilstm128","bilstm256"))
    if args.stage != "report":
        for stage_number, arms in enumerate(stages,1):
            if args.stage == "budget" and stage_number == 2 or args.stage == "bottleneck" and stage_number == 1:
                continue
            for arm in arms:
                for seed in SEEDS:
                    expected=json.loads((args.cnn_runs/str(seed)/"baseline/training.json").read_text())["training_schedule_sha256"]
                    print(f"Preparing shared 80-epoch schedule for {seed}",flush=True)
                    schedule=make_schedule(*arrays["train"],replace(config,seed=seed),
                        args.runs/"schedules"/f"{seed}.npz",expected_first20=expected)
                    ref=None
                    if arm in BUDGET:
                        ref=pd.read_csv(original_directory(args,arm,seed)/"history.csv").to_dict("records")
                    train(arm,seed,*arrays["train"],cached_validation,masks["validation"],schedule,
                        replace(config,seed=seed),mc,args.runs/arm/str(seed),score_function=evaluate_loss,
                        minimum_epochs=20,reference_history=ref)
                    del schedule
            available = tuple(a for a in ARMS if all((args.runs/a/str(s)/"training.json").exists() for s in SEEDS))
            evaluate(args,prepared,available)
    else:
        available=tuple(a for a in ARMS if all((args.runs/a/str(s)/"training.json").exists() for s in SEEDS))
        evaluate(args,prepared,available)


if __name__ == "__main__":
    main()
