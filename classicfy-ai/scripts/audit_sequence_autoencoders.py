"""Check experiment/checkpoint agreement, retrieval exclusions and artifact hashes."""

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


def audit(out, runs_dir=None, cnn_runs=None):
    out = Path(out)
    ai = Path(__file__).resolve().parents[1]
    data = ai.parent.parent / "datasets"
    runs_dir = Path(runs_dir) if runs_dir else data / "sequence_autoencoders_run01"
    cnn_runs = Path(cnn_runs) if cnn_runs else data / "temporal_controlled_run01"
    protocol = json.loads((out / "protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        assert sha(ai / name) == expected, f"Experiment source changed: {name}"
    summary = pd.read_csv(out / "summary.csv")
    assert not summary.duplicated(["model", "seed", "split", "condition"]).any()
    runs = json.loads((out / "training_runs.json").read_text())
    assert len(runs) == 9
    for run in runs:
        directory = cnn_runs / str(run["seed"]) / "baseline" if run["model"] == "cnn" else runs_dir / run["model"] / str(run["seed"])
        if run["model"] == "cnn":
            assert sha(directory / "final.pt") == run["final_sha256"]
        else:
            for name, expected in run["checkpoint_sha256"].items():
                assert sha(directory / f"{name}.pt") == expected
        checkpoint = torch.load(directory / "best_value.pt", map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == run["best_value_epoch"]
        np.testing.assert_allclose(checkpoint["score"]["value_mse"], run["best_validation_value_mse"], rtol=0, atol=0)
        key = (summary.model == run["model"]) & (summary.seed == run["seed"]) & (summary.split == "validation")
        best = summary[key & (summary.condition == "best")].iloc[0]
        final = summary[key & (summary.condition == "final")].iloc[0]
        np.testing.assert_allclose(best.mse, run["best_validation_value_mse"], rtol=1e-6)
        np.testing.assert_allclose(final.mse, run["final_validation"]["value_mse"], rtol=1e-6)
        np.testing.assert_allclose(final.slope_mse, run["final_validation"]["delta_mse"], rtol=1e-6)
    for seed in protocol["seeds"]:
        assert len({r["training_schedule_sha256"] for r in runs if r["seed"] == seed}) == 1
    selected = summary[summary.condition == "best"]
    for model in ("cnn", "bilstm", "transformer"):
        for split in ("validation", "test"):
            assert len(selected[(selected.model == model) & (selected.split == split)]) == 3
    neighbors = pd.read_csv(out / "neighbors.csv")
    assert (neighbors.query_piece != neighbors.candidate_piece).all()
    assert set(neighbors["rank"]) == set(range(1, 6))
    assert (neighbors.groupby(["model", "seed", "query"]).size() == 5).all()
    assert not neighbors.duplicated(["model", "seed", "query", "candidate"]).any()
    retrieval = pd.read_csv(out / "retrieval_queries.csv")
    assert len(retrieval) == 3 * 3 * 82 * 3
    assert retrieval.closer_than_peer_share.between(0, 1).all()
    assert (retrieval.candidate_work_peers >= 4).all()
    examples = pd.read_csv(out / "example_beats.csv")
    assert set(examples.example) == {1, 2, 3}
    assert not (examples.hidden & ~examples.valid).any()
    assert len(examples) == 3 * 7 * 64
    for directory in (out / "bilstm_first_validation", out):
        manifest = json.loads((directory / "figure_manifest.json").read_text())
        assert manifest["plotting_source_sha256"] == sha(ai / "scripts/plot_sequence_autoencoders.py")
        for name, expected in manifest["figures"].items():
            path = directory / name
            assert sha(path) == expected, f"Figure changed: {path}"
            assert path.with_suffix(".svg").exists()
            with Image.open(path) as img:
                assert img.width >= 1500 and img.height >= 900
        for name, expected in manifest["tables"].items():
            assert sha(directory / name) == expected
    for path in out.rglob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if not link.startswith(("http:", "https:", "#")):
                assert (path.parent / link.split("#")[0]).exists(), f"Broken report link: {path}: {link}"
    result = {"status": "passed", "trained_sequence_runs": 6, "reused_cnn_runs": 3,
              "epochs_per_run": protocol["training_config"]["epochs"],
              "png_figures_including_intermediate_record": len(list(out.rglob("*.png"))),
              "retrieval_rows": len(retrieval), "test_queries_per_model_seed": 82,
              "auditor_source_sha256": sha(Path(__file__)),
              "checks": ["frozen source hashes", "checkpoint scores equal evaluation tables",
                         "checkpoint file hashes and selected epochs",
                         "same batch/mask schedules", "all test conditions have three seeds",
                         "same-work candidates excluded", "complete top-5 rows",
                         "fixed examples have no invalid targets", "PNG and CSV hashes",
                         "PNG/SVG pairs and resolution", "local report links"]}
    (out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--cnn-runs", type=Path)
    args = parser.parse_args()
    audit(args.out, args.runs, args.cnn_runs)
