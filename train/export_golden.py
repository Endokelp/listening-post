"""Generates tests/fixtures/golden.json - the parity fixture the JS test reads."""
import json
import pathlib
import numpy as np
from features import extract_features

SAMPLE_RATE = 48000
N = 96000  # 2 s


def synthetic_quadcopter():
    """Four rotors ~1.8% apart in RPM, each with two harmonics - the same
    shape proven to beat clearly under the current rotorBeatScore formula in
    tests/beat.test.js's multirotor() helper (quad score > 0.6 there)."""
    t = np.arange(N) / SAMPLE_RATE
    base = 200.0
    rotors = 4
    freqs = [base * (1 + i * 0.018) for i in range(rotors)]
    x = np.zeros(N)
    for f in freqs:
        for h in (1, 2, 3):
            x += np.sin(2 * np.pi * f * h * t) / (rotors * h)
    # Noise amplitude chosen so rotorBeat sits clearly above the "drone-like"
    # threshold without pinning at the 1.0 clamp - a saturated fixture value
    # would make the golden-vector drift check trivially pass at the ceiling
    # instead of actually exercising the unclamped computation.
    rng = np.random.default_rng(7)
    x += rng.standard_normal(N) * 0.5
    return x


def main():
    samples = synthetic_quadcopter()
    payload = {
        "sampleRate": SAMPLE_RATE,
        "note": "Synthetic quadcopter. Regenerate with: python train/export_golden.py",
        "samples": [round(float(v), 10) for v in samples],
        "expected": extract_features(samples, SAMPLE_RATE),
    }
    out = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(payload["expected"])


if __name__ == "__main__":
    main()
