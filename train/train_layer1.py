"""Fits layer-1 logistic regression and overwrites
src/classify/layer1-weights.json with real coefficients."""
import argparse
import json
import pathlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from features import extract_features
from benchmark import load_clips

FEATURE_ORDER = ["harmonicStack", "rotorBeat", "bandRatio", "spectralFlatness"]
ROOT = pathlib.Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "src" / "classify" / "layer1-weights.json"


def vectorise(clips, sample_rate):
    return np.array([
        [extract_features(audio, sample_rate)[k] for k in FEATURE_ORDER]
        for _, audio in clips
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", default="data/drones")
    ap.add_argument("--confusers", default="data/confusers")
    ap.add_argument("--sample-rate", type=int, default=48000)
    args = ap.parse_args()

    drones = load_clips(args.drones, args.sample_rate)
    confusers = load_clips(args.confusers, args.sample_rate)
    if len(drones) < 5 or len(confusers) < 5:
        print(f"Not enough data to train: {len(drones)} drone, {len(confusers)} confuser clips. "
              "Need at least 5 of each. Weights left untouched.")
        return

    X = np.vstack([vectorise(drones, args.sample_rate), vectorise(confusers, args.sample_rate)])
    y = np.concatenate([np.ones(len(drones)), np.zeros(len(confusers))])

    model = LogisticRegression(max_iter=1000).fit(X, y)
    folds = min(5, len(y) // 2)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=folds)

    WEIGHTS.write_text(json.dumps({
        "trained": True,
        "note": f"Fitted on {len(drones)} drone and {len(confusers)} confuser clips. "
                f"{folds}-fold CV accuracy {scores.mean():.4f} +/- {scores.std():.4f}.",
        "coefficients": {k: float(c) for k, c in zip(FEATURE_ORDER, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
    }, indent=2) + "\n")
    print(f"CV accuracy {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"wrote {WEIGHTS}")


if __name__ == "__main__":
    main()
