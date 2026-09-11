"""Slices recordings into exactly the windows the app scores: 2 s long, 1 s
hop, 48 kHz mono. Scoring whole files would be a train/serve mismatch - the
app never sees a whole file.

Files are named {kind}__{source}__{label}__{recording}.wav (see datasets.py
and the in-app recorder), kind being drone, confuser or background."""
import math
import pathlib
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from features import extract_features, FEATURE_ORDER

SAMPLE_RATE = 48000
WINDOW = 2 * SAMPLE_RATE  # WINDOW_SEC in src/features/config.js
HOP = 1 * SAMPLE_RATE     # HOP_SEC
# ESC-50 pads short clips with digital zeros. Only true silence is dropped;
# a quiet real recording is a legitimate test window.
SILENT_RMS = 1e-5


def read_48k(path):
    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = math.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g)
    return audio


def windows(audio):
    """Same arithmetic as createWindower in src/audio/ring-buffer.js."""
    for start in range(0, len(audio) - WINDOW + 1, HOP):
        w = audio[start:start + WINDOW]
        if np.sqrt(np.mean(w ** 2)) >= SILENT_RMS:
            yield start / SAMPLE_RATE, w


def recordings(directory):
    for path in sorted(pathlib.Path(directory).glob("*.wav")):
        kind, source, label, recording = path.stem.split("__")
        yield {"kind": kind, "source": source, "label": label,
               "group": f"{source}/{recording}", "path": path}


def feature_table(directory):
    """One row per non-silent window: file metadata, start second, f0 and
    the layer-1 features. Samples are not kept - thousands of 2 s windows
    would not fit in memory."""
    rows = []
    for meta in recordings(directory):
        for start, w in windows(read_48k(meta["path"])):
            f = extract_features(w, SAMPLE_RATE)
            rows.append({**meta, "start": start, "f0": f["f0"],
                         **{k: f[k] for k in FEATURE_ORDER}})
    return rows


def _self_check():
    x = np.ones(5 * SAMPLE_RATE)
    starts = [s for s, _ in windows(x)]
    assert starts == [0, 1, 2, 3], starts  # 5 s -> 4 overlapping windows
    x[:3 * SAMPLE_RATE] = 0
    assert [s for s, _ in windows(x)] == [2, 3]  # all-zero windows dropped
    tone = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
    path = pathlib.Path(__file__).with_name("_selfcheck.wav")
    sf.write(path, np.stack([tone, tone], axis=1), 44100)
    try:
        assert len(read_48k(path)) == SAMPLE_RATE  # stereo 44.1k -> mono 48k
    finally:
        path.unlink()
    print("windows self-check passed")


if __name__ == "__main__":
    _self_check()
