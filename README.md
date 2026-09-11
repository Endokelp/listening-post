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

### The first idea, and how real audio broke it

A quadcopter's four rotors spin at slightly different speeds, so their tones **beat**
against each other at 1–30 Hz. A single engine has one tone. The idea was that a drone
beats and an engine doesn't. On synthetic signals the original version worked perfectly:
quadcopter 0.89–1.00, single engine 0.006–0.05.

Then it met real recordings, 428 windows of real Parrot drones and real ESC-50 engines:

- On the real drones the rotor-beat score was **0.000**, at the median and at the 90th
  percentile. Real flight controllers keep changing each rotor's speed, which smears the
  beat lines.
- Real engines **do** pulse in that band. Their combustion cycle modulates the loudness;
  one engine showed lines at 22.7, 45.4 and 68.1 Hz. Chainsaws and helicopters also
  scored higher than the drones.

The steady synthetic engine hid both facts. Rotor beat is still one of the meters on
screen, but it no longer decides anything alone.

### What decides now

Every 2-second window (one per second) is described by 53 numbers:

- **Four physical measurements:** harmonic stack, rotor beat, band energy and flatness.
  These are the meters on screen.
- **A 24-band spectral fingerprint** (100 Hz to 6.5 kHz). Each 43 ms frame is divided by
  its own loudness, so the fingerprint is the same for a quiet, distant source and a loud,
  close one. The app records both the shape and how much it moves over the window: engine
  firing pulses shuffle energy between bands, and steady rotor tones don't. A test proves
  the fingerprint doesn't change when the volume does.

A logistic-regression classifier combines them. A 3-of-5 vote with hysteresis and a
cooldown then decides whether to alert, so one drone-like second cannot fire on its own.

## Evidence (real audio)

All numbers come from `docs/BENCHMARK.md`, produced by `train/benchmark.py` on recordings
the classifier never trained on. Training used Parrot drones recorded indoors, outdoor
flyovers of four airframes (DroneNoise, University of Salford) and ESC-50 folds 1–4.
Sources and licences are in `docs/DATASETS.md`.

**The central finding: it works on microphones it has heard, and not on ones it hasn't.**

| Drone source | Unseen recordings, lab's other recordings in training | Whole lab never heard |
|---|---|---|
| Parrot (Al-Emadi, indoor phone mic) | **93.9%** of windows | 6.8% |
| DroneNoise (outdoor ground mics) | **80.2%** | 0.0% |

The same wall appears on a fresh test that was never used for any design decision: one
clip each of 32 airframes, recorded on a laptop microphone. The app would alert on **6 of
32**, all small toy-class drones (Tello, Syma, Hasakee) that sound like the Parrot Mambo it
trained on. It would alert on **none** of the DJI Mini, Mavic, Phantom or Air drones.

False alarms on held-out confusers and background: **4.8%** of 2-second windows before the
vote. Birds dropped from 94% (the first trained model) to 0%. The weak spots are wind
(37.5% of windows) and a vacuum cleaner (22%).

**What this means for the design:** an acoustic detector has to hear the microphone and
site it guards. A phone is cheap enough to calibrate on site, and a ten-thousand-dollar
sensor is not. The app therefore says *not yet calibrated to this microphone* until it has
trained on recordings made through that phone (`docs/FIELD-PROTOCOL.md`). Results on the
author's own DJI, trained on one day and tested on another, will be added here.

## Where it stops working (stated on purpose)

1. **A new microphone or site** it has never heard: detection falls below 10% (table above).
2. **Wind** is the worst natural false alarm: 37.5% of windows.
3. **Rotor beat alone cannot separate a drone from a real engine.** Engines pulse in the
   same band.
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
npm run benchmark                          # fetch public data, train, write docs/BENCHMARK.md
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
src/ui/         spectrogram, feature meters, detection log, field recorder
train/          Python mirror of the features, datasets, windowing, training, benchmark
tests/          node:test suite, including the JS-Python parity test
docs/           BENCHMARK (results), DATASETS, FIELD-PROTOCOL, design spec, build record
```
