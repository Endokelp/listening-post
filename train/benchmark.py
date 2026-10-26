"""Benchmark. Scores ONLY data/test - recordings the
classifier never trained on - with the shipped layer-1 weights, and writes
docs/BENCHMARK.md. Every number the README quotes comes from there."""
import argparse
import datetime
import json
import pathlib
import sys
from collections import defaultdict
import numpy as np
from features import extract_features
from mix import mix_at_snr
from train_layer1 import TRAIN, design, fit, predict, grouped_cv
from windows import SAMPLE_RATE, feature_table, read_48k, recordings, windows

SNRS_DB = [20, 10, 5, 0, -5, -10]
THRESHOLDS = [0.5, 0.7, 0.9]
ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST = ROOT / "data" / "test"
WEIGHTS = ROOT / "src" / "classify" / "layer1-weights.json"
OUT = ROOT / "docs" / "BENCHMARK.md"


def probability(features, weights):
    """Mirrors classify() in src/classify/layer1.js."""
    z = weights["intercept"] + sum(c * features[k] for k, c in weights["coefficients"].items())
    return float(1 / (1 + np.exp(-z)))


def alerts(probs, threshold, needed=3, window=5):
    """Would the app's detector have fired at least once? Its idle ->
    detecting rule (src/detect/state-machine.js): `needed` of the last
    `window` probabilities at or above the threshold."""
    return any(sum(p >= threshold for p in probs[max(0, i - window + 1):i + 1]) >= needed
               for i in range(len(probs)))


def group_by(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return dict(sorted(out.items()))


def rate(hits, n):
    return f"{hits / n:.1%} ({hits}/{n})" if n else "n/a"


def md_table(header, lines):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in line) + " |" for line in lines])


def summary_lines(rows, threshold, keys, with_f0=False):
    lines = []
    for key, members in group_by(rows, *keys).items():
        recs = group_by(members, "group").values()
        hits = sum(r["p"] >= threshold for r in members)
        alerting = sum(alerts([r["p"] for r in rec], threshold) for rec in recs)
        line = [*key, len(recs), rate(hits, len(members)), rate(alerting, len(recs))]
        if with_f0:
            line.append(f"{np.median([r['f0'] for r in members]):.0f}")
        lines.append(line)
    return lines


def snr_sweep(weights, threshold):
    """Each held-out drone window mixed over a held-out background window at
    each SNR. The SNR is ADDED noise on top of whatever the recording already
    contains, so the true SNR is lower than the nominal one."""
    noises = [next(windows(read_48k(m["path"])), (0, None))[1]
              for m in recordings(TEST) if m["kind"] == "background"]
    noises = [n for n in noises if n is not None]
    if not noises:
        return None
    hits = defaultdict(int)
    count = defaultdict(int)
    i = 0
    for m in recordings(TEST):
        if m["kind"] != "drone":
            continue
        for _, w in windows(read_48k(m["path"])):
            noise = noises[i % len(noises)]
            i += 1
            for snr in SNRS_DB:
                p = probability(extract_features(mix_at_snr(w, noise, snr), SAMPLE_RATE), weights)
                hits[(m["source"], snr)] += p >= threshold
                count[(m["source"], snr)] += 1
    sources = sorted({s for s, _ in count})
    lines = [[f"{snr:+d} dB"] + [rate(hits[(s, snr)], count[(s, snr)]) for s in sources]
             for snr in SNRS_DB]
    return md_table(["Added-noise SNR"] + [f"{s} drones" for s in sources], lines), len(noises)


def cross_lab(threshold):
    """The honest limit. For each drone source in data/train: detection on its
    recordings when only THOSE recordings are unseen (grouped CV, the rest of
    the lab is in training) versus when the WHOLE lab is left out of training."""
    rows = feature_table(TRAIN)
    X, y = design(rows)
    src = np.array([r["source"] for r in rows])
    oof = grouped_cv(X, y, np.array([r["group"] for r in rows]))
    lines = []
    for s in sorted(set(src[y == 1])):
        pos = (src == s) & (y == 1)
        keep = src != s
        p = predict(*fit(X[keep], y[keep]), X[pos])
        lines.append([s, rate(int(np.sum(oof[pos] >= threshold)), int(pos.sum())),
                      rate(int(np.sum(p >= threshold)), int(pos.sum()))])
    neg = y == 0
    lines.append(["non-drone windows flagged", rate(int(np.sum(oof[neg] >= threshold)), int(neg.sum())), ""])
    return md_table(["Drone source (data/train)", "Its unseen recordings, lab in training",
                     "Whole lab left out of training"], lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.7, help="app enterThreshold")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()

    weights = json.loads(WEIGHTS.read_text())
    rows = feature_table(TEST)
    if not rows:
        print(f"No data in {TEST}. Run train/datasets.py first.")
        return
    for r in rows:
        r["p"] = probability(r, weights)
    thr = args.threshold
    drones = [r for r in rows if r["kind"] == "drone"]
    others = [r for r in rows if r["kind"] != "drone"]

    ops = []
    for t in THRESHOLDS:
        ops.append([f"{t:.1f}" + (" (app)" if t == thr else ""),
                    rate(sum(r["p"] >= t for r in drones), len(drones)),
                    rate(sum(r["p"] >= t for r in others), len(others))])
    sweep = snr_sweep(weights, thr)

    doc = [
        "# Benchmark: held-out test data",
        "",
        f"Generated by `train/benchmark.py` on {datetime.date.today()}. Do not edit by hand; "
        "rerun the script.",
        "",
        f"**Classifier:** {'trained' if weights.get('trained') else 'UNTRAINED - numbers are provisional'}. "
        f"{weights.get('note', '')}",
        "",
        f"**Test set:** `data/test`, recordings the classifier never trained on: "
        f"{len(group_by(rows, 'group'))} recordings, {len(rows)} windows. "
        "Sources and licences: the Data section of README.md.",
        "",
        f"**Method:** each recording is cut into the app's 2-second windows at a 1-second hop and "
        f"scored with the shipped layer-1 weights. A window counts as a detection at p >= {thr}, "
        "the app's alert threshold. *Recordings alerting* applies the app's 3-of-5 vote, i.e. "
        "whether the app would have raised at least one alert on that recording.",
        "",
        "## Drones, as recorded",
        "",
        md_table(["Source", "Airframe label", "Recordings", "Windows detected",
                  "Recordings alerting", "Median f0 (Hz)"],
                 summary_lines(drones, thr, ("source", "label"), with_f0=True)),
        "",
        "## False alarms on confusers and background",
        "",
        md_table(["Kind", "Source", "Class", "Recordings", "Windows flagged", "Recordings alerting"],
                 summary_lines(others, thr, ("kind", "source", "label"))),
        "",
        "## Operating points",
        "",
        md_table(["Threshold", "Drone windows detected", "Non-drone windows flagged"], ops),
        "",
        "## Drone detection vs added background noise",
        "",
    ]
    if sweep:
        table, n_noise = sweep
        doc += [f"Each held-out drone window mixed over one of {n_noise} held-out background "
                "windows. The SNR is noise added on top of the recording's own background, so "
                "the true SNR is lower than the nominal figure.", "", table, ""]
    else:
        doc += ["No background recordings in data/test.", ""]

    doc += ["## Where it stops working: labs it has never heard", "",
            "Training-data analysis. A detector that has heard some recordings from a lab "
            "(its microphone, its site) versus one that has heard nothing from that lab.", "",
            cross_lab(thr), ""]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(doc))
    print("\n".join(doc))
    print(f"\nwrote {OUT}")
    if not weights.get("trained"):
        print("WARNING: layer-1 weights are untrained. Do not quote these numbers.", file=sys.stderr)


def _self_check():
    assert not alerts([0.9, 0.1, 0.1, 0.1, 0.1], 0.7)        # one spike never fires
    assert alerts([0.9, 0.1, 0.9, 0.1, 0.9], 0.7)            # 3 of 5
    assert alerts([0.9, 0.9, 0.9], 0.7)                      # fires before 5 windows exist
    assert not alerts([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9], 0.7)  # never 3 inside any 5
    w = {"intercept": -4.0, "coefficients": {"rotorBeat": 5.0, "harmonicStack": 3.0}}
    assert abs(probability({"rotorBeat": 1.0, "harmonicStack": 0.5}, w) - 1 / (1 + np.exp(-2.5))) < 1e-12
    print("benchmark self-check passed")


if __name__ == "__main__":
    main()
