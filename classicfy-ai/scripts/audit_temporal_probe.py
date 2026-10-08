"""Independently verify train-only probes, saved predictions and work aggregation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding.extended_experiments import load
from evaluate_temporal_order import encode


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(frame):
    counts = frame.groupby(['piece', 'key']).size()
    performances = frame.groupby('piece').key.nunique()
    return np.array([1 / (frame.piece.nunique() * performances[r.piece] * counts[r.piece, r.key])
                     for r in frame.itertuples(index=False)])


def mean_scale(x, w):
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-12] = 1
    return mean, scale


def pca_fit(x, w):
    mean, scale = mean_scale(x, w)
    z = (x - mean) / scale
    _, vectors = np.linalg.eigh((z * w[:, None]).T @ z)
    return mean, scale, vectors[:, -64:][:, ::-1]


def ridge_fit_predict(x, y, w, alpha, held):
    mean, scale = mean_scale(x, w)
    z, hz = (x - mean) / scale, (held - mean) / scale
    intercept = np.average(y.reshape(len(y), 28), axis=0, weights=w)
    centered = y.reshape(len(y), 28) - intercept
    coef = np.linalg.solve((z * w[:, None]).T @ z + alpha * np.eye(z.shape[1]),
                           (z * w[:, None]).T @ centered)
    return (hz @ coef + intercept).reshape(-1, 7, 4)


def feature_average(values):
    return np.mean([values[0], values[1], values[2], values[3], np.nanmean(values[4:])])


def metrics(y, pred, frame):
    # Grouping, rather than sample weighting, independently checks aggregation.
    def aggregate(values):
        data = pd.DataFrame(values)
        data['piece'], data['key'] = frame.piece.to_numpy(), frame.key.to_numpy()
        return data.groupby(['piece', 'key']).mean().groupby('piece').mean().mean().to_numpy()
    mse = aggregate(np.mean((y - pred) ** 2, axis=2))
    energy = aggregate(np.mean(y ** 2, axis=2))
    pe = aggregate(np.mean(pred ** 2, axis=2))
    product = aggregate(np.mean(y * pred, axis=2))
    with np.errstate(divide='ignore', invalid='ignore'):
        score = np.where(energy > 1e-12, 100 * (1 - mse / energy), np.nan)
        cosine = np.where(np.sqrt(energy * pe) > 1e-12, product / np.sqrt(energy * pe), np.nan)
        amplitude = np.where(energy > 1e-12, 100 * np.sqrt(pe / energy), np.nan)
    return dict(mse=mse, flat_mse=energy, score_percent=score,
                centered_cosine=cosine, amplitude_percent=amplitude)


def main():
    ai = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ai / 'analysis/temporal_probe')
    parser.add_argument('--runs', type=Path, default=ai.parent.parent / 'datasets/temporal_probe_run01')
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol = json.loads((args.out / 'protocol.json').read_text())
    assert protocol == json.loads((args.runs / 'protocol.json').read_text())
    generation = json.loads((args.out / 'report_generation.json').read_text())
    assert generation['plot_source_sha256'] == sha(ai / 'scripts/plot_temporal_probe.py')
    assert generation['protocol_sha256'] == sha(args.out / 'protocol.json')
    assert generation['summary_sha256'] == sha(args.out / 'summary.csv')
    for source, digest in protocol['sources'].items():
        if source != 'scripts/plot_temporal_probe.py':
            assert sha(ai / source) == digest, source
    for checkpoint in protocol['checkpoints']:
        assert sha(checkpoint['path']) == checkpoint['sha256']
    for path, digest in protocol['previous_embedding_files'].items():
        assert sha(path) == digest
    audit = json.loads((args.out / 'evaluation_audit.json').read_text())
    for name, digest in audit['csv_sha256'].items():
        assert sha(args.out / name) == digest
    selected = pd.read_csv(args.out / 'selected_windows.csv')
    frames, inputs, targets = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        frames[split] = selected[selected.split == split].reset_index(drop=True)
        with np.load(args.runs / f'{split}_inputs.npz') as stored:
            inputs[split], targets[split] = stored['values'], stored['targets']
        assert hashlib.sha256(inputs[split].tobytes()).hexdigest() == protocol['input_sha256'][split]
        x = inputs[split].astype(float)
        expected = np.stack([x[:, :, q * 16:(q + 1) * 16].mean(axis=2) for q in range(4)], axis=2)
        expected -= x.mean(axis=2, keepdims=True)
        np.testing.assert_allclose(expected, targets[split], atol=1e-14, rtol=1e-12)
        for _, performance in frames[split].groupby('key'):
            assert np.all(np.diff(np.sort(performance.start)) >= 64)
        assert len(x) == len(frames[split])
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        assert not set(frames[a].piece) & set(frames[b].piece)
    train = frames['train']
    w = weights(train)
    mapping = protocol['penalty_selection']['mapping']
    folds = train.piece.map(mapping).to_numpy()
    assert set(folds) == set(range(5)) and len(mapping) == train.piece.nunique()
    summary = pd.read_csv(args.out / 'summary.csv')
    channels = pd.read_csv(args.out / 'channel_metrics.csv')
    works = pd.read_csv(args.out / 'work_metrics.csv')
    cv = pd.read_csv(args.out / 'cv.csv')
    parameters = pd.read_csv(args.out / 'probe_parameters.csv')
    examples = pd.read_csv(args.out / 'example_quarters.csv')
    cv_checked = channel_checked = work_checked = example_checked = encodings_checked = 0
    for parameter in parameters.itertuples(index=False):
        method = parameter.method
        features = {s:np.load(args.runs / f'features_{method}_{s}.npz')['features'] for s in frames}
        if parameter.arm == 'summary14':
            for split, x in inputs.items():
                x = x.astype(float)
                mean = x.mean(axis=2)
                mean[:, 1] = np.median(abs(x[:, 1]), axis=1)
                width = np.percentile(x, 95, axis=2) - np.percentile(x, 5, axis=2)
                np.testing.assert_allclose(features[split], np.stack((mean, width), axis=2).reshape(len(x), 14), atol=1e-14)
        elif parameter.arm == 'pca64':
            for split, x in inputs.items():
                np.testing.assert_array_equal(features[split], x.reshape(len(x), -1).astype(float))
        else:
            checkpoint = next(c for c in protocol['checkpoints'] if c['arm'] == parameter.arm and
                              c['seed'] == parameter.seed and c['stage'] == parameter.stage)
            encoder, _ = load(checkpoint['path'])
            for split in frames:
                indices = np.array([0, len(inputs[split]) // 2, len(inputs[split]) - 1])
                actual = encode(encoder, inputs[split][indices])
                np.testing.assert_allclose(actual, features[split][indices], atol=1e-5, rtol=1e-4)
                encodings_checked += len(indices)
        # Recalculate every train-only fold and penalty independently.
        scores = {}
        for fold in range(5):
            keep, held = folds != fold, folds == fold
            assert not set(train[keep].piece) & set(train[held].piece)
            tw, hw = weights(train[keep]), weights(train[held])
            tx, hx = features['train'][keep], features['train'][held]
            if parameter.arm == 'pca64':
                mean, scale, components = pca_fit(tx, tw)
                tx, hx = ((tx - mean) / scale) @ components, ((hx - mean) / scale) @ components
            energy = np.average(np.mean(targets['train'][keep] ** 2, axis=2), axis=0, weights=tw)
            for alpha in protocol['penalty_selection']['alphas']:
                prediction = ridge_fit_predict(tx, targets['train'][keep], tw, alpha, hx)
                mse = np.average(np.mean((targets['train'][held] - prediction) ** 2, axis=2), axis=0, weights=hw)
                error = feature_average(mse / energy)
                row = cv[(cv.method == method) & (cv.fold == fold) & (cv.alpha == alpha)].iloc[0]
                np.testing.assert_allclose(error, row.normalized_mse, rtol=1e-8, atol=1e-10)
                assert row.held_works == train[held].piece.nunique()
                scores.setdefault(alpha, []).append((error, row.held_works))
                cv_checked += 1
        averages = {a:np.average([v[0] for v in vals], weights=[v[1] for v in vals]) for a, vals in scores.items()}
        assert parameter.alpha == min(averages, key=averages.get)
        np.testing.assert_allclose(parameter.cv_normalized_mse, averages[parameter.alpha], rtol=1e-8)
        probe = np.load(args.runs / f'probe_{method}.npz')
        fitted = features
        if parameter.arm == 'pca64':
            pca = np.load(args.runs / f'pca_{method}.npz')
            mean, scale = mean_scale(features['train'], w)
            np.testing.assert_allclose(mean, pca['mean'], atol=1e-12)
            np.testing.assert_allclose(scale, pca['scale'], atol=1e-12)
            z = (features['train'] - mean) / scale
            covariance = (z * w[:, None]).T @ z
            np.testing.assert_allclose(covariance @ pca['components'], pca['components'] * pca['eigenvalues'], atol=1e-10)
            np.testing.assert_allclose(pca['components'].T @ pca['components'], np.eye(64), atol=1e-12)
            fitted = {s:((x - mean) / scale) @ pca['components'] for s, x in features.items()}
        mean, scale = mean_scale(fitted['train'], w)
        np.testing.assert_allclose(probe['mean'], mean, atol=1e-12)
        np.testing.assert_allclose(probe['scale'], scale, atol=1e-12)
        z = (fitted['train'] - mean) / scale
        y = targets['train'].reshape(len(train), 28)
        intercept = np.average(y, axis=0, weights=w)
        np.testing.assert_allclose(probe['intercept'], intercept, atol=1e-12)
        residual = z @ probe['coefficients'] + intercept - y
        np.testing.assert_allclose((z * w[:, None]).T @ residual + parameter.alpha * probe['coefficients'], 0, atol=1e-10)
        for split in ('validation', 'test'):
            pred = np.load(args.runs / f'prediction_{method}_{split}.npz')['prediction']
            expected = (((fitted[split] - mean) / scale) @ probe['coefficients'] + intercept).reshape(-1, 7, 4)
            np.testing.assert_allclose(pred, expected, atol=1e-12)
        print(f'Checked train-only fit and CV: {method}', flush=True)
    train_mean = np.average(targets['train'], axis=0, weights=w)
    for row in summary.itertuples(index=False):
        split, method = row.split, row.method
        frame, y = frames[split], targets[split]
        pred = np.load(args.runs / f'prediction_{method}_{split}.npz')['prediction']
        if method in ('flat', 'train_mean', 'raw_reference'):
            expected = np.zeros_like(y) if method == 'flat' else np.broadcast_to(train_mean, y.shape) if method == 'train_mean' else y
            np.testing.assert_allclose(pred, expected, atol=1e-12)
        computed = metrics(y, pred, frame)
        stored = channels[(channels.method == method) & (channels.split == split)].sort_values('channel')
        for name, values in computed.items():
            np.testing.assert_allclose(stored[name], values, atol=1e-10, rtol=1e-9, equal_nan=True)
        np.testing.assert_allclose(row.dynamics_score_percent, computed['score_percent'][2], atol=1e-10)
        np.testing.assert_allclose(row.macro_score_percent, feature_average(computed['score_percent']), atol=1e-10)
        channel_checked += len(stored)
        for piece in frame.piece.unique():
            keep = (frame.piece == piece).to_numpy()
            current = metrics(y[keep], pred[keep], frame[keep])
            stored = works[(works.method == method) & (works.split == split) & (works.piece == piece)].sort_values('channel')
            for name, values in current.items():
                np.testing.assert_allclose(stored[name], values, atol=1e-10, rtol=1e-9, equal_nan=True)
            work_checked += len(stored)
        current_examples = examples[(examples.method == method) & (examples.split == split)]
        sorted_works = sorted(frame.piece.unique())
        for number, wi in enumerate((0, len(sorted_works) // 2, len(sorted_works) - 1), 1):
            keys = sorted(frame[frame.piece == sorted_works[wi]].key.unique())
            candidates = frame.index[frame.key == keys[len(keys) // 2]].to_numpy()
            index = candidates[len(candidates) // 2]
            stored = current_examples[current_examples.example == number].sort_values(['channel', 'quarter'])
            np.testing.assert_allclose(stored.target, y[index].ravel(), atol=1e-12)
            np.testing.assert_allclose(stored.prediction, pred[index].ravel(), atol=1e-12)
            assert set(stored.window) == {frame.iloc[index].window}
            example_checked += len(stored)
    manifest = json.loads((args.out / 'figure_manifest.json').read_text())
    for item in manifest:
        assert sha(args.out / item['file']) == item['sha256']
    assert json.loads((args.out / 'layout_audit.json').read_text())['status'] == 'passed'
    result = dict(status='passed', probes=len(parameters), independently_recalculated_cv_fits=cv_checked,
        channel_rows=channel_checked, work_rows=work_checked, example_rows=example_checked,
        encoder_forward_samples=encodings_checked, checkpoint_hashes=len(protocol['checkpoints']), figure_hashes=len(manifest),
        checks=['Frozen protocol, source, encoder and prior embedding integrity',
            'Disjoint works and nonoverlapping windows; original target recalculation',
            'All fold-normalized train-only CV fits and selected penalties recalculated',
            'Train-only PCA eigen equation, ridge normalization and normal equations',
            'All heldout predictions, channel/work aggregation and fixed examples recalculated',
            'Figure hashes and canvas-bound layout audit'],
        source_sha256=sha(Path(__file__)), protocol_sha256=sha(args.out / 'protocol.json'))
    assert cv_checked == len(cv) and channel_checked == len(channels)
    assert work_checked == len(works) and example_checked == len(examples)
    (args.out / 'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
