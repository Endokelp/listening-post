"""Figures and numbers for docs/FIELD-TEST.md: the author's DJI Mini 2 recorded
in a yard on 2026-10-25 through the app's own recorder (files in data/test,
named drone__own__dji-mini2-<5m|10m>-hover__*.wav, gitignored).

Run: python train/field_figures.py   -> docs/img/field-*.png (1920x1080)"""
import json
import pathlib
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram, welch
from features import extract_features, CLASSIFIER_FEATURES
from windows import windows, read_48k

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"
SURF, INK, INK2, MUTED, BLUE, GRAY = "#1a1a19", "#f0efec", "#c3c2b7", "#8a897f", "#3987e5", "#6e6d66"
CMAP = LinearSegmentedColormap.from_list("blue", [SURF, "#173a66", BLUE, "#dbe9fb"])
plt.rcParams.update({"font.family": "Segoe UI", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#3a3a37",
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 20})
FOOTER = ("DJI Mini 2 hovering 4.7 m up; Motorola phone on a 1.5 m tripod; "
          "Listening Post in-app recorder, 25 Oct 2026")


def spec(a, n, lo, hi):
    f, t, S = spectrogram(a, 48000, nperseg=n, noverlap=n - 1200)
    m = (f > lo) & (f < hi)
    return t, f[m], 10 * np.log10(S[m] + 1e-16)


def frame(fig, title, sub):
    fig.text(0.05, 0.93, title, fontsize=34, weight="bold")
    fig.text(0.05, 0.885, sub, fontsize=19, color=INK2)
    fig.text(0.05, 0.03, FOOTER, fontsize=14, color=MUTED)


def bare(ax):
    for s in ax.spines.values():
        s.set_visible(False)


def rotor_tones(c5):
    t, f, d = spec(c5, 16384, 240, 420)
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_axes([0.07, 0.12, 0.68, 0.72])
    ax.pcolormesh(t, f, d, shading="auto", cmap=CMAP, vmin=np.percentile(d, 35), vmax=np.percentile(d, 99.8))
    ax.set_xlabel("seconds")
    ax.set_ylabel("frequency (Hz)")
    bare(ax)
    frame(fig, "Two families of rotor tones",
          "My drone 5 m away (about 6 m in a straight line). Two families fit two sets of rotors "
          "at different speeds. Brighter = louder.")
    arrow = dict(arrowstyle="-|>", color=INK2, lw=1.5)
    for yv, txt in [(287, "287 Hz family"), (322, "322 Hz family")]:
        ax.annotate(txt, xy=(5.75, yv), xytext=(6.05, yv), va="center", fontsize=21, color=INK,
                    annotation_clip=False, arrowprops=arrow)
    ax.annotate("3.4 s: all the lines rise\ntogether, then settle back", xy=(4.05, 392),
                xytext=(6.05, 395), va="center", fontsize=21, color=INK, annotation_clip=False,
                arrowprops={**arrow, "connectionstyle": "arc3,rad=-0.15"})
    fig.savefig(OUT / "field-rotor-tones-5m.png", dpi=100)
    plt.close(fig)


def near_vs_far(c5, c10):
    fig = plt.figure(figsize=(19.2, 10.8))
    frame(fig, "At 5 m the rotor lines stand out. At 10 m they don't",
          "Same phone, same colour scale, 200 to 1,400 Hz. Brighter = louder.")
    specs = [spec(c5, 8192, 200, 1400), spec(c10, 8192, 200, 1400)]
    lo = np.percentile(np.concatenate([s[2].ravel() for s in specs]), 30)
    hi = np.percentile(specs[0][2], 99.8)
    titles = ["5 m out: rotor lines up to 23 dB above the rest", "10 m out: no steady rotor line"]
    for i, ((t, f, d), title) in enumerate(zip(specs, titles)):
        ax = fig.add_axes([0.07 + i * 0.465, 0.12, 0.42, 0.68])
        ax.pcolormesh(t, f, d, shading="auto", cmap=CMAP, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=22, loc="left", color=INK, pad=12)
        ax.set_xlabel("seconds")
        ax.set_ylabel("frequency (Hz)" if i == 0 else "")
        bare(ax)
    fig.savefig(OUT / "field-5m-vs-10m.png", dpi=100)
    plt.close(fig)


def why_it_said_no(c5):
    """Per-feature push (coefficient x value) of the shipped weights, averaged
    over windows: 25 random training-drone recordings vs the 5 m clip."""
    weights = json.loads((ROOT / "src" / "classify" / "layer1-weights.json").read_text())
    co = weights["coefficients"]

    def push(ws):
        rows = [extract_features(w, 48000) for w in ws]
        return {k: np.mean([co[k] * r[k] for r in rows]) for k in CLASSIFIER_FEATURES}

    random.seed(0)
    pub = random.sample(sorted((ROOT / "data" / "train").glob("drone__*.wav")), 25)
    a = push([w for p in pub for _, w in windows(read_48k(p))])
    b = push([w for _, w in windows(c5)])
    mel = [k for k in CLASSIFIER_FEATURES if k.startswith("mel")]
    groups = [("Harmonic stack\n(physics meter)", ["harmonicStack"]),
              ("Band energy ratio\n(physics meter)", ["bandRatio"]),
              ("Rotor beat\n(physics meter)", ["rotorBeat"]),
              ("Spectral fingerprint\n(48 learned numbers)", mel)]
    va = [sum(a[k] for k in ks) for _, ks in groups]
    vb = [sum(b[k] for k in ks) for _, ks in groups]

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_axes([0.25, 0.2, 0.68, 0.6])
    frame(fig, "The harmonic stack said drone. The learned fingerprint said no.",
          "How much each part of the shipped model pushed toward 'drone' (right) or 'not a drone' (left). "
          "Average 2 s window.")
    y = np.arange(len(groups))[::-1]
    ax.barh(y + 0.19, va, 0.36, color=GRAY, label="Training drones (other labs' microphones)")
    ax.barh(y - 0.19, vb, 0.36, color=BLUE, label="My DJI Mini 2 at 5 m (my phone)")
    for yy, pa, pb in zip(y, va, vb):
        for val, off in ((pa, 0.19), (pb, -0.19)):
            ax.text(val + (0.15 if val >= 0 else -0.15), yy + off, f"{val:+.1f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=18, color=INK)
    ax.axvline(0, color=INK2, lw=1.5)
    ax.set_yticks(y, [g for g, _ in groups], fontsize=20)
    ax.set_xlim(-13, 5)
    ax.set_xlabel("push on the score (log-odds)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=18)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#2e2e2b")
    ax.set_axisbelow(True)
    fig.text(0.25, 0.075, "Result on all 10 windows of my drone: probability 0.00. The app stays silent.",
             fontsize=20, color=INK)
    fig.savefig(OUT / "field-why-it-said-no.png", dpi=100)
    plt.close(fig)
    for (g, _), pa, pb in zip(groups, va, vb):
        print(f"{g.splitlines()[0]:22} training drones {pa:+.2f}   my drone at 5 m {pb:+.2f}")


def levels(name, a):
    f, P = welch(a, 48000, nperseg=16384)
    db = lambda lo, hi, op: 10 * np.log10(op(P[(f > lo) & (f < hi)]))
    print(f"{name}: {len(a) / 48000:.1f} s, rms {20 * np.log10(np.sqrt(np.mean(a ** 2))):.1f} dBFS, "
          f"strongest 300-340 Hz tone {db(300, 340, np.max):.1f} dB, "
          f"floor 200-1400 Hz {db(200, 1400, np.median):.1f} dB, floor 100-300 Hz {db(100, 300, np.median):.1f} dB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test = ROOT / "data" / "test"
    c5 = read_48k(next(test.glob("drone__own__dji-mini2-5m-*.wav")))
    c10 = read_48k(next(test.glob("drone__own__dji-mini2-10m-*.wav")))
    rotor_tones(c5)
    near_vs_far(c5, c10)
    why_it_said_no(c5)
    levels("5 m", c5)
    levels("10 m", c10)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
