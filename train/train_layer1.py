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
from sklearn.model_selection import GroupKFold, cross_val_score
from features import FEATURE_ORDER
from windows import feature_table

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train"
WEIGHTS = ROOT / "src" / "classify" / "layer1-weights.json"


def main():
    rows = feature_table(TRAIN)
    y = np.array([r["kind"] == "drone" for r in rows], dtype=int)
    groups = np.array([r["group"] for r in rows])
    n_pos, n_neg = len(set(groups[y == 1])), len(set(groups[y == 0]))
    if n_pos < 5 or n_neg < 5:
        print(f"Not enough data to train: {n_pos} drone and {n_neg} non-drone recordings "
              f"in {TRAIN}. Need at least 5 of each. Run train/datasets.py. Weights left untouched.")
        return

    X = np.array([[r[k] for k in FEATURE_ORDER] for r in rows])
    # Balanced: negatives outnumber drones, and a missed drone matters as much
    # as a false alarm when choosing the boundary.
    make = lambda: LogisticRegression(class_weight="balanced", max_iter=1000)
    cv = cross_val_score(make(), X, y, groups=groups, cv=GroupKFold(n_splits=5),
                         scoring="balanced_accuracy")
    model = make().fit(X, y)

    sources = Counter(f"{r['source']}:{r['label']}" for r in rows if r["kind"] == "drone")
    note = (f"Fitted on {int(y.sum())} drone windows ({n_pos} recordings: "
            f"{', '.join(sorted(sources))}) and {int((1 - y).sum())} confuser/background "
            f"windows ({n_neg} recordings) from data/train. Grouped 5-fold CV balanced "
            f"accuracy {cv.mean():.3f} +/- {cv.std():.3f} - a training-set sanity check, "
            f"NOT a result. Results come from train/benchmark.py on held-out data.")
    WEIGHTS.write_text(json.dumps({
        "trained": True,
        "note": note,
        "coefficients": {k: round(float(c), 6) for k, c in zip(FEATURE_ORDER, model.coef_[0])},
        "intercept": round(float(model.intercept_[0]), 6),
    }, indent=2) + "\n")
    print(note)
    print(f"wrote {WEIGHTS}")


if __name__ == "__main__":
    main()
