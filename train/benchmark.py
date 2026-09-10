"""Phase 3 evidence generator. Reports accuracy vs SNR and false-positive rate
per confuser class. Every number the video quotes comes from here."""
import argparse
import csv
import json
import pathlib
import sys
import numpy as np
import soundfile as sf
from features import extract_features
from mix import mix_at_snr

SNRS_DB = [20, 10, 5, 0, -5, -10]
ROOT = pathlib.Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "src" / "classify" / "layer1-weights.json"


def load_clips(directory, sample_rate):
    clips = []
    for path in sorted(pathlib.Path(directory).glob("*.wav")):
        audio, sr = sf.read(path, dtype="float64", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            print(f"  skipping {path.name}: {sr} Hz, expected {sample_rate}", file=sys.stderr)
            continue
        clips.append((path.name, audio))
    return clips


def probability(features, weights):
    z = weights["intercept"]
    for name, coefficient in weights["coefficients"].items():
        z += coefficient * features.get(name, 0.0)
    return 1 / (1 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", default="data/drones")
    ap.add_argument("--confusers", default="data/confusers")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--out", default="benchmark.csv")
    args = ap.parse_args()

    weights = json.loads(WEIGHTS.read_text())
    if not weights.get("trained"):
        print("WARNING: layer-1 weights are hand-tuned, not trained. "
              "These numbers are provisional and must not be quoted as results.",
              file=sys.stderr)

    drones = load_clips(args.drones, args.sample_rate)
    confusers = load_clips(args.confusers, args.sample_rate)
    if not drones or not confusers:
        print(f"No data: {len(drones)} drone clips, {len(confusers)} confuser clips. "
              f"Populate {args.drones} and {args.confusers} with {args.sample_rate} Hz wavs.")
        return

    rows = []
    for snr in SNRS_DB:
        hits = 0
        trials = 0
        for _, drone in drones:
            for _, confuser in confusers:
                mixed = mix_at_snr(drone, confuser, snr)
                p = probability(extract_features(mixed, args.sample_rate), weights)
                hits += p >= args.threshold
                trials += 1
        rows.append({"snr_db": snr, "class": "drone", "n": trials,
                     "detect_rate": round(hits / trials, 4)})

    for name, confuser in confusers:
        p = probability(extract_features(confuser, args.sample_rate), weights)
        rows.append({"snr_db": "", "class": f"confuser:{name}", "n": 1,
                     "detect_rate": round(float(p >= args.threshold), 4)})

    out = ROOT / args.out
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["snr_db", "class", "n", "detect_rate"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
