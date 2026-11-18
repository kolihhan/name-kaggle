"""Reproduce the preregistered five-fold, train-only prompt-column CE comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import logsumexp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

if __package__:
    from .audit_lora24_sibling_holdout import family
else:
    from audit_lora24_sibling_holdout import family


SEED = 20261118
STEPS = 200
L2 = 0.0005
BOOTSTRAPS = 10_000
ARMS = ("row", "matched", "shuffled")
TRAIN_HASH = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLD_HASH = "94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"


def load_snapshot(train_path, fold_path):
    for path, expected in [(train_path, TRAIN_HASH), (fold_path, FOLD_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input fingerprint differs from preregistration: {path.name}")
    text_data = pd.read_csv(train_path, dtype={"id": str, "prompt_id": str})
    folds = pd.read_csv(fold_path, dtype={"id": str, "prompt_id": str})
    keys = ["id", "prompt_id", "model"]
    if text_data.id.duplicated().any() or folds.id.duplicated().any():
        raise ValueError("Duplicate row IDs")
    frame = text_data.merge(folds, on=keys, how="outer", validate="one_to_one", indicator=True)
    if len(frame) != 2699 or (frame._merge != "both").any() or frame.isna().any().any():
        raise ValueError("Training rows, targets, prompts and fold manifest are misaligned")
    if frame.duplicated(["prompt_id", "model"]).any() or frame.model.nunique() != 18:
        raise ValueError("Invalid prompt and class grid")
    if frame.groupby("prompt_id").fold_id.nunique().ne(1).any():
        raise ValueError("A prompt crosses the fold boundary")
    if frame.groupby("fold_id").prompt_id.nunique().to_dict() != dict.fromkeys(range(5), 30):
        raise ValueError("Expected 30 disjoint validation prompts per fold")
    return frame.sort_values("id").reset_index(drop=True)


def encode_fold(train_texts, validation_texts):
    tfidf = TfidfVectorizer(
        analyzer="char", lowercase=False, ngram_range=(2, 5), min_df=2,
        max_features=24_000, sublinear_tf=True, smooth_idf=True, norm="l2",
    )
    svd = TruncatedSVD(n_components=256, n_iter=5, n_oversamples=10, random_state=SEED)
    train_sparse = tfidf.fit_transform(train_texts)
    train = svd.fit_transform(train_sparse)
    validation = svd.transform(tfidf.transform(validation_texts))
    return np.ascontiguousarray(normalize(train)), np.ascontiguousarray(normalize(validation))


def prompt_grid(prompts, labels, n_classes):
    """Index each real response by [prompt, true class]; use -1 for absent slots."""
    groups, inverse = np.unique(prompts, return_inverse=True)
    grid = np.full((len(groups), n_classes), -1, dtype=np.intp)
    grid[inverse, labels] = np.arange(len(labels))
    if np.count_nonzero(grid >= 0) != len(labels):
        raise ValueError("A prompt has multiple responses for the same target")
    return grid


def shuffled_grid(grid, fold):
    rng = np.random.default_rng(SEED + fold)
    shuffled = grid.copy()
    for column in shuffled.T:
        observed = column >= 0
        column[observed] = rng.permutation(column[observed])
    return shuffled


def row_loss_gradient(scores, labels):
    n_rows = len(labels)
    log_probs = scores - logsumexp(scores, axis=1, keepdims=True)
    rows = np.arange(n_rows)
    loss = -log_probs[rows, labels].mean()
    gradient = np.exp(log_probs)
    gradient[rows, labels] -= 1
    return loss, gradient / n_rows


def column_loss_gradient(scores, grid):
    """Average over present prompt/class targets. Response is the softmax axis."""
    present = grid >= 0
    grouped = np.where(present[:, :, None], scores[grid.clip(min=0)], -np.inf)
    log_probs = grouped - logsumexp(grouped, axis=1, keepdims=True)
    groups, targets = np.nonzero(present)
    n_targets = len(targets)
    loss = -log_probs[groups, targets, targets].mean()
    # A missing response contributes no negatives; an absent target has no loss at all.
    column_gradient = np.exp(log_probs) * present[:, None, :]
    column_gradient[groups, targets, targets] -= 1
    gradient = np.empty_like(scores)
    gradient[grid[present]] = column_gradient[present] / n_targets
    return loss, gradient


def fit_head(x, labels, n_classes, grid):
    # Appending a 1 permits one matrix product for weights and the free intercept.
    design = np.column_stack((x, np.ones(len(x))))
    parameters = np.zeros((design.shape[1], n_classes))
    moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    for step in range(1, STEPS + 1):
        scores = design @ parameters
        _, gradient = row_loss_gradient(scores, labels)
        if grid is not None:
            _, column_gradient = column_loss_gradient(scores, grid)
            gradient += column_gradient
        parameter_gradient = design.T @ gradient
        parameter_gradient[:-1] += L2 * parameters[:-1]
        moment = 0.9 * moment + 0.1 * parameter_gradient
        second_moment = 0.999 * second_moment + 0.001 * np.square(parameter_gradient)
        adjusted_moment = moment / (1 - 0.9**step)
        adjusted_scale = np.sqrt(second_moment / (1 - 0.999**step)) + 1e-8
        parameters -= 0.05 * adjusted_moment / adjusted_scale
    if not np.isfinite(parameters).all():
        raise ArithmeticError("Nonfinite fitted parameters")
    return parameters[:-1], parameters[-1]


def cross_validate(frame, labels, n_classes):
    predictions = {name: np.full(len(frame), -1, dtype=int) for name in ARMS}
    for fold in range(5):
        valid_mask = frame.fold_id.to_numpy() == fold
        train_mask = ~valid_mask
        x_train, x_valid = encode_fold(frame.text[train_mask], frame.text[valid_mask])
        train_y = labels[train_mask]
        grid = prompt_grid(frame.prompt_id.to_numpy()[train_mask], train_y, n_classes)
        groups = {"row": None, "matched": grid, "shuffled": shuffled_grid(grid, fold)}
        for name, group in groups.items():
            weights, bias = fit_head(x_train, train_y, n_classes, group)
            predictions[name][valid_mask] = (x_valid @ weights + bias).argmax(axis=1)
        print(f"Finished fixed fold {fold + 1}/5 ({valid_mask.sum()} held-out rows)", flush=True)
    if any((values < 0).any() for values in predictions.values()):
        raise ValueError("Incomplete out-of-fold predictions")
    return predictions


def paired_bootstrap(frame, labels, predictions, n_classes):
    """Draw whole prompts within fold and recompute each resampled class denominator."""
    rng = np.random.default_rng(SEED)
    totals = np.zeros((BOOTSTRAPS, n_classes), dtype=int)
    baselines = ("row", "shuffled")
    gains = {name: np.zeros_like(totals) for name in baselines}
    correct = {name: (pred == labels).astype(int) for name, pred in predictions.items()}
    for fold in range(5):
        mask = frame.fold_id.to_numpy() == fold
        grid = prompt_grid(frame.prompt_id.to_numpy()[mask], labels[mask], n_classes)
        present = grid >= 0
        n = len(grid)
        # A multinomial draw counts the multiplicity of each prompt in each replicate.
        counts = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAPS)
        totals += counts @ present.astype(int)
        for name in baselines:
            change = (correct["matched"] - correct[name])[mask]
            gains[name] += counts @ np.where(present, change[grid.clip(min=0)], 0)
    if (totals == 0).any():
        raise ValueError("Resampling removed an entire label")
    result = {}
    for name, net in gains.items():
        delta = (net / totals).mean(axis=1)
        result[f"matched_minus_{name}"] = np.quantile(delta, [0.025, 0.975]).tolist()
    return result


def paired_counts(before, after, mask):
    return {
        "rescued": int((mask & ~before & after).sum()),
        "regressed": int((mask & before & ~after).sum()),
    }


def summarize(frame, labels, classes, predictions):
    bas = {name: float(balanced_accuracy_score(labels, pred)) for name, pred in predictions.items()}
    correct = {name: pred == labels for name, pred in predictions.items()}
    row, matched = correct["row"], correct["matched"]
    all_rows = np.ones(len(frame), dtype=bool)
    intervals = paired_bootstrap(frame, labels, predictions, len(classes))
    comparisons = {}
    for name in ("row", "shuffled"):
        key = f"matched_minus_{name}"
        comparisons[key] = {
            "delta_ba": bas["matched"] - bas[name], "paired_prompt_ci95": intervals[key],
            **paired_counts(correct[name], matched, all_rows),
        }
    folds = []
    for fold in range(5):
        mask = frame.fold_id.to_numpy() == fold
        folds.append({
            "fold": fold, "prompts": int(frame.prompt_id[mask].nunique()), "rows": int(mask.sum()),
            "ba": {name: float(balanced_accuracy_score(labels[mask], pred[mask]))
                   for name, pred in predictions.items()},
            **paired_counts(row, matched, mask),
        })
    per_class = {}
    for i, name in enumerate(classes):
        mask = labels == i
        per_class[name] = {
            "n": int(mask.sum()), "row_recall": float(row[mask].mean()),
            "matched_recall": float(matched[mask].mean()), **paired_counts(row, matched, mask),
        }
    providers = np.array([family(name) for name in classes])
    true_provider = providers[labels]
    row_family = providers[predictions["row"]] == true_provider
    matched_family = providers[predictions["matched"]] == true_provider
    families = {}
    for name in np.unique(providers):
        mask = true_provider == name
        families[name] = {
            "n": int(mask.sum()), "exact_model": paired_counts(row, matched, mask),
            "sibling_rescues": int((mask & ~row & matched & row_family).sum()),
            "sibling_regressions": int((mask & row & ~matched & matched_family).sum()),
            "family_identity": paired_counts(row_family, matched_family, mask),
        }
    return {
        "stage": "development OOF on reused official prompt folds",
        "rows": len(frame), "prompts": int(frame.prompt_id.nunique()), "classes": len(classes),
        "train_sha256": TRAIN_HASH, "folds_sha256": FOLD_HASH,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__},
        "fit_count": 5 * len(ARMS), "updates_per_fit": STEPS, "bootstrap_replicates": BOOTSTRAPS,
        "pooled_ba": bas, "comparisons": comparisons, "by_fold": folds,
        "by_true_class": per_class, "by_true_family": families,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--local-predictions", type=Path, help="Optional row outputs outside the repository")
    args = parser.parse_args()
    if args.local_predictions and args.local_predictions.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Choose a local-predictions path outside the repository")
    frame = load_snapshot(args.train, args.folds)
    classes, labels = np.unique(frame.model.to_numpy(), return_inverse=True)
    with threadpool_limits(limits=2):
        predictions = cross_validate(frame, labels, len(classes))
        result = summarize(frame, labels, classes, predictions)
    if args.local_predictions:
        local = frame[["id", "prompt_id", "model", "fold_id"]].copy()
        for name, pred in predictions.items():
            local[name] = classes[pred]
        args.local_predictions.parent.mkdir(parents=True, exist_ok=True)
        local.to_csv(args.local_predictions, index=False)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("pooled_ba", "comparisons")}), flush=True)


if __name__ == "__main__":
    main()
