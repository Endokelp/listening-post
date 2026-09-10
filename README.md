# Listening Post

**Acoustic drone detection that runs on a phone instead of a ten-thousand-dollar sensor.**

Try it: **https://endokelp.github.io/listening-post/** — tap *Start listening* and allow
the microphone. It works in a phone browser and can be added to the home screen.

A listening post is a forward position whose job is to hear a threat before it can be
seen. This app does that job for small multirotor drones, using nothing but a phone
microphone.

## The problem it takes on

Acoustic drone detection already works. What limits it is **cost per node**: one
expensive sensor covers one point, while the threat is spread out. A phone is a
microphone, a processor and a screen that people already carry.

**Scope is detection only.** Listening Post does not locate, track, jam or intercept
anything.

## The hard part

Hearing a drone in a quiet field is easy. The real problem is **not** raising an alarm
for a lawnmower, a leaf blower, a motorcycle, an air-conditioning unit or a distant
airliner. Those are all machines with strong engine tones.

The app separates a drone from them with two measurements that do different jobs:

| Measurement | What it separates | How |
|---|---|---|
| **Harmonic stack** | a machine from background noise | Finds the blade-pass tone and checks that its harmonics stand above the local noise floor |
| **Rotor beat** | a multirotor from a single engine | A quadcopter's four rotors spin at slightly different speeds, so their tones **beat** against each other at 1–30 Hz. A single engine has one tone and does not beat |

To score rotor beat, the app bandpasses around the blade-pass tone, takes the loudness
envelope, then looks for a few discrete lines in the envelope's spectrum. It requires
both low flatness and a peak that stands above the band.

One result runs against intuition: **a single engine scores *higher* on harmonic stack
than a drone does**, because a drone's slightly detuned rotors smear each harmonic.
Harmonic stack is not the drone detector. Rotor beat is.

Every 2-second window (one per second) goes through a small logistic-regression
classifier over those measurements. A 3-of-5 vote with hysteresis and a cooldown then
decides whether to raise an alert, so one drone-like second cannot fire on its own. The
four meters on screen show *why* the app decided what it did.

## Evidence

**Measured on synthetic signals** (the separation between signal types, not accuracy):

| Signal | Harmonic stack | Rotor beat |
|---|---|---|
| Quadcopter + noise | 0.27–0.40 | **0.89–1.00** |
| Hexacopter + noise | 0.38 | 0.30 |
| Single engine + noise | 0.51–0.66 | **0.006–0.05** |
| Broadband noise only | 0.04 | 0.38 |

**Real-audio results:** pending. They will come from public drone and confuser
recordings plus the author's own field recordings, benchmarked with `train/benchmark.py`.
No accuracy figure is claimed until then. The on-screen classifier currently says
*untrained*.

## Where it stops working (stated on purpose)

1. **An engine with strongly pulsing loudness can look like a drone.** An engine under
   changing load, or a leaf blower being swept, can pulse in the 1–30 Hz beat band. A test
   asserts this limitation so it stays visible.
2. **More rotors score weaker.** A hexacopter spreads its beating across more frequencies:
   0.30 against a quadcopter's 1.00.
3. **Broadband noise scores 0.38 on rotor beat.** The classifier rejects it because noise
   scores 0.04 on harmonic stack.
4. **One microphone gives no direction or range**, only presence.

## Built with

- **JavaScript** with the Web Audio API and an AudioWorklet. No frameworks, no runtime
  dependencies. The FFT, windowing and mel filterbank are written in the repo.
- **Python** (numpy, scipy, scikit-learn) for training and benchmarking. `train/features.py`
  mirrors the browser feature code, and a golden-vector test fails the build if the two
  drift by as little as 1%.
- **Node's built-in test runner**: `npm test`.

## Run it locally

```bash
npm test                                   # Node 22+
npm run serve                              # then open http://localhost:8000
pip install -r train/requirements.txt      # training and benchmark pipeline
```

The microphone needs `localhost` or HTTPS. Browser processing (echo cancellation, noise
suppression, auto gain) is switched off on purpose, because each one alters the exact
signature being measured.

## Repository map

```
src/features/   FFT, window, mel filterbank, harmonic stack, rotor beat, feature vector
src/audio/      microphone capture (AudioWorklet) and windowing
src/classify/   layer-1 classifier and its weights
src/detect/     3-of-5 vote, hysteresis, cooldown
src/ui/         spectrogram, feature meters, detection log
train/          Python mirror of the features, SNR mixing, training, benchmark
tests/          node:test suite, including the JS-Python parity test
docs/           design spec, build plan, build record
```
