"""Fits layer-1 logistic regression on data/train and overwrites
src/classify/layer1-weights.json with real coefficients.

Trains on 2 s windows (windows.py), exactly what the app scores. Cross-
validation is grouped by source recording: overlapping windows from one
recording are near-duplicates, and letting them straddle a fold would
inflate the score. data/test is never touched here - benchmark.py owns it."""
import json
import pathlib
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from features import CLASSIFIER_FEATURES
from windows import feature_table

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train"
WEIGHTS = ROOT / "src" / "classify" / "layer1-weights.json"
# Strong-ish L2: 53 correlated features, a few thousand windows. Chosen in the
# Phase 3 experiments on grouped CV, not on any test set.
REGULARISATION = 0.3


def design(rows):
    X = np.array([[r[k] for k in CLASSIFIER_FEATURES] for r in rows])
    y = np.array([r["kind"] == "drone" for r in rows], dtype=int)
    return X, y


def fit(X, y):
    """Standardise, fit, then fold the scaler into raw-feature coefficients
    so the browser applies plain weights with no scaler of its own."""
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=REGULARISATION, class_weight="balanced",
                               max_iter=5000).fit(scaler.transform(X), y)
    coef = model.coef_[0] / scaler.scale_
    return coef, float(model.intercept_[0] - np.sum(coef * scaler.mean_))


def predict(coef, intercept, X):
    return 1 / (1 + np.exp(-(X @ coef + intercept)))


def grouped_cv(X, y, groups):
    """Out-of-fold probabilities, folds split by recording."""
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        p[te] = predict(*fit(X[tr], y[tr]), X[te])
    return p


def main():
    rows = feature_table(TRAIN)
    X, y = design(rows)
    groups = np.array([r["group"] for r in rows])
    n_pos, n_neg = len(set(groups[y == 1])), len(set(groups[y == 0]))
    if n_pos < 5 or n_neg < 5:
        print(f"Not enough data to train: {n_pos} drone and {n_neg} non-drone recordings "
              f"in {TRAIN}. Need at least 5 of each. Run train/datasets.py. Weights left untouched.")
        return

    cv = balanced_accuracy_score(y, grouped_cv(X, y, groups) >= 0.5)
    coef, intercept = fit(X, y)
    sources = Counter(f"{r['source']}:{r['label']}" for r in rows if r["kind"] == "drone")
    note = (f"Fitted on {int(y.sum())} drone windows ({n_pos} recordings: "
            f"{', '.join(sorted(sources))}) and {int((1 - y).sum())} confuser/background "
            f"windows ({n_neg} recordings) from data/train. Grouped 5-fold CV balanced "
            f"accuracy {cv:.3f} - a same-lab sanity check, NOT a result. Results, including "
            f"cross-lab numbers, come from train/benchmark.py.")
    WEIGHTS.write_text(json.dumps({
        "trained": True,
        # Phase 3: a detector that has never heard this microphone and site
        # finds almost nothing (benchmark "Where it stops working"). The app
        # says so until the author's own recordings are in training.
        "calibrated": any(r["source"] == "own" for r in rows),
        "note": note,
        "coefficients": {k: round(float(c), 8) for k, c in zip(CLASSIFIER_FEATURES, coef)},
        "intercept": round(intercept, 8),
    }, indent=2) + "\n")
    print(note)
    print(f"wrote {WEIGHTS}")


def _self_check():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 3)) * [1, 10, 0.1] + [0, 5, 2]
    y = (X[:, 0] + (X[:, 2] - 2) * 10 > 0).astype(int)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=REGULARISATION, class_weight="balanced", max_iter=5000).fit(scaler.transform(X), y)
    coef, intercept = fit(X, y)
    assert np.allclose(predict(coef, intercept, X), model.predict_proba(scaler.transform(X))[:, 1])
    print("train_layer1 self-check passed (scaler folds into raw weights exactly)")


if __name__ == "__main__":
    import sys
    _self_check() if "--self-check" in sys.argv else main()
