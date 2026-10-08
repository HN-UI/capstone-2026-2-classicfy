"""Audit actual schedules, early stopping, checkpoints and validation-only figures."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(out, runs):
    ai = Path(__file__).resolve().parents[1]
    data = ai.parent.parent / "datasets"
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol == json.loads((runs / "protocol.json").read_text())
    assert sha(data / "temporal_embedding_cluster_run01/dataset.json") == protocol["reference_dataset_sha256"]
    assert sha(data / "feature_normalization_cluster_raw.npz") == protocol["cache_sha256"]
    for name, expected in protocol["sources"].items():
        assert sha(ai / name) == expected, f"Training source changed: {name}"
    summary = pd.read_csv(out / "summary.csv")
    details = pd.read_csv(out / "channel_metrics.csv")
    works = pd.read_csv(out / "work_metrics.csv")
    assert set(summary.split) == set(details.split) == set(works.split) == {"validation"}
    assert not summary.duplicated(["model", "seed", "split", "condition"]).any()
    completed = json.loads((out / "training_runs.json").read_text())
    arms = protocol["arms"]
    assert len(completed) == len(arms) * len(protocol["seeds"]) == 15
    assert {(r["arm"], r["seed"]) for r in completed} == {
        (a, s) for a in arms for s in protocol["seeds"]}
    expected_conditions = {"initial", "best20", "epoch20", "best_extended", "final"}
    batch_size = protocol["training_config"]["batch_size"]
    for seed in protocol["seeds"]:
        # Independently hash the saved arrays with their actual minibatch boundaries.
        with np.load(runs / "schedules" / f"{seed}.npz", allow_pickle=False) as schedule:
            order, hidden = schedule["order"], schedule["hidden"]
            metadata = json.loads(str(schedule["metadata"]))
        assert order.shape == (80, 5803)
        assert hidden.shape == (80, 5803, 7, 64)
        prefix = hashlib.sha256()
        for epoch, (indices, mask) in enumerate(zip(order, hidden)):
            np.testing.assert_array_equal(np.sort(indices), np.arange(len(indices)))
            digest = hashlib.sha256()
            for start in range(0, len(indices), batch_size):
                for blob in (indices[start:start+batch_size].tobytes(), mask[start:start+batch_size].tobytes()):
                    digest.update(blob)
                    if epoch < 20:
                        prefix.update(blob)
            assert digest.hexdigest() == metadata["epoch_sha256"][epoch]
        assert prefix.hexdigest() == metadata["first20_sha256"]
        original = json.loads((data / "temporal_controlled_run01" / str(seed) / "baseline/training.json").read_text())
        assert prefix.hexdigest() == original["training_schedule_sha256"]
        for result in (r for r in completed if r["seed"] == seed):
            assert result["schedule_first20_sha256"] == prefix.hexdigest()
            assert result["schedule_epoch_sha256"] == metadata["epoch_sha256"][:result["completed_epochs"]]
    for result in completed:
        arm, seed = result["arm"], result["seed"]
        directory = runs / arm / str(seed)
        assert result == json.loads((directory / "training.json").read_text())
        history = pd.read_csv(directory / "history.csv")
        assert history.epoch.tolist() == list(range(result["completed_epochs"] + 1))
        assert 20 <= result["completed_epochs"] <= 80
        best_index = int(history.validation_value_mse.idxmin())
        assert int(history.loc[best_index, "epoch"]) == result["best_epoch"]
        np.testing.assert_allclose(history.loc[best_index, "validation_value_mse"], result["best_validation_mse"], rtol=0, atol=1e-14)
        # The first time the fixed stop rule applies must be the final epoch.
        stops = history[(history.epoch >= 20) & (history.stale_epochs >= 10)]
        if result["stop_reason"] == "patience":
            assert int(stops.epoch.iloc[0]) == result["completed_epochs"]
        else:
            assert result["completed_epochs"] == 80 and stops.empty
        if arm.endswith("64"):
            reference = (data / "temporal_controlled_run01" / str(seed) / "baseline" if arm == "cnn64"
                         else data / "sequence_autoencoders_run01" / arm.removesuffix("64") / str(seed))
            old = pd.read_csv(reference / "history.csv")
            delta = np.max(np.abs(history.iloc[:21].validation_value_mse.to_numpy() - old.iloc[:21].validation_value_mse.to_numpy()))
            assert delta <= 2e-6
            np.testing.assert_allclose(delta, result["first20_max_absolute_error_vs_original"], atol=1e-14, rtol=0)
        selected = summary[(summary.model == arm) & (summary.seed == seed)]
        assert set(selected.condition) == expected_conditions
        for condition, filename in (("initial", "initial"), ("best20", "best20"), ("epoch20", "epoch20"), ("best_extended", "best"), ("final", "final")):
            checkpoint_path = directory / f"{filename}.pt"
            assert sha(checkpoint_path) == result["checkpoint_sha256"][filename]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            row = selected[selected.condition == condition].iloc[0]
            assert row.epoch == checkpoint["epoch"]
            np.testing.assert_allclose(row.mse, checkpoint["score"]["value_mse"], atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(row.slope_mse, checkpoint["score"]["delta_mse"], atol=1e-7, rtol=1e-6)
            assert all(torch.isfinite(t).all() for t in checkpoint["state_dict"].values())
        assert result["parameters"] == sum(result[k] for k in ("encoder_parameters", "bottleneck_parameters", "decoder_parameters"))
    examples = pd.read_csv(out / "example_beats.csv")
    assert len(examples) == 3 * 7 * 64
    assert set(examples.example) == {1, 2, 3}
    assert not (examples.hidden & ~examples.valid).any()
    assert set(summary[summary.condition == "baseline"].model) == {"zero", "visible_mean", "linear_interpolation"}
    manifest = json.loads((out / "figure_manifest.json").read_text())
    assert manifest["plot_source_sha256"] == sha(ai / "scripts/plot_extended_sequences.py")
    assert len(manifest["figures"]) == 55
    for name, expected in manifest["figures"].items():
        path = out / name
        assert sha(path) == expected and path.with_suffix(".svg").exists()
        with Image.open(path) as image:
            assert image.width >= 1500 and image.height >= 900
    for name, expected in manifest["tables"].items():
        assert sha(out / name) == expected
    layout = json.loads((out / "layout_audit.json").read_text())
    assert layout["status"] == "passed" and layout["figures_checked"] == 55
    assert layout["plot_source_sha256"] == manifest["plot_source_sha256"]
    assert layout["figure_manifest_sha256"] == sha(out / "figure_manifest.json")
    for document in out.rglob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if not link.startswith(("https:", "http:", "#")):
                assert (document.parent / link.split("#")[0]).exists(), f"Broken link: {link}"
    result = {"status": "passed", "completed_runs": len(completed), "original_curves_reproduced": 9,
              "png_figures": len(manifest["figures"]), "svg_figures": len(list(out.rglob("*.svg"))),
              "evaluated_split": "validation only", "auditor_sha256": sha(Path(__file__)),
              "checks": ["frozen experiment sources", "all 80 schedule epochs independently hashed",
                         "first20 schedules match original", "nine original first20 curves reproduced",
                         "fixed minimum/patience/maximum stop rule", "75 checkpoint scores match tables",
                         "checkpoint hashes and finite parameters", "no test metrics",
                         "fixed examples contain no invalid hidden targets", "PNG/CSV hashes and SVG pairs",
                         "all figure text fits canvas", "report links resolve"]}
    (out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    ai = Path(__file__).resolve().parents[1]
    parser.add_argument("--out", type=Path, default=ai / "analysis/sequence_followup")
    parser.add_argument("--runs", type=Path, default=ai.parent.parent / "datasets/extended_sequences_run01")
    args = parser.parse_args()
    audit(args.out, args.runs)
