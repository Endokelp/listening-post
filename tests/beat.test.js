import test from 'node:test';
import assert from 'node:assert/strict';
import { rotorBeatScore } from '../src/features/beat.js';

const SR = 48000;
const N = 96000; // 2 s

let seed = 7;
const rnd = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return (seed / 0x7fffffff) * 2 - 1;
};

// A multirotor: several rotors at slightly different RPM, each contributing a
// blade-pass fundamental plus harmonics.
function multirotor(base, rotors, noiseAmp = 0) {
  const freqs = Array.from({ length: rotors }, (_, i) => base * (1 + i * 0.018));
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let v = 0;
    for (const f of freqs) for (const h of [1, 2, 3]) v += Math.sin(2 * Math.PI * f * h * i / SR) / (rotors * h);
    x[i] = v + rnd() * noiseAmp;
  }
  return x;
}

// A single-engine confuser: one fundamental with harmonics. No beating.
function singleEngine(base, noiseAmp = 0, amHz = null) {
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let v = 0;
    for (const h of [1, 2, 3, 4]) v += Math.sin(2 * Math.PI * base * h * i / SR) / h;
    if (amHz) v *= 1 + 0.5 * Math.sin(2 * Math.PI * amHz * i / SR);
    x[i] = v + rnd() * noiseAmp;
  }
  return x;
}

// These tests describe SYNTHETIC signals - clean sine-wave rotor/engine
// models, not real recordings. The multirotor-vs-single-engine separation
// below holds for signals like these. It does NOT reliably hold on real
// audio: a real engine's combustion cycle amplitude-
// modulates its own output at 5-30 Hz, landing inside the same beat band a
// multirotor's difference tones occupy. That's why the classifier
// (extract.js) no longer relies on rotorBeat alone and also fingerprints
// spectral shape via the mel bank.
test('a multirotor beats and a single engine does not', () => {
  const quad = rotorBeatScore(multirotor(200, 4), 205, SR);
  assert.ok(quad > 0.6, `quad scored ${quad}, expected clear beating`);

  // An exactly-zero-noise tone is not a signal any real microphone
  // produces. Under the detrend+Hann envelope (beat.js), such a tone's own
  // truncation-edge leakage (it doesn't complete a whole number of periods
  // in the analysis window) concentrates right next to the excluded bin 1
  // and reads as a beat line: this went from 0.0 (old mean-subtract, no
  // window) to 1.0 with the new pipeline. Asserted here so the artifact
  // stays visible instead of silently vanishing from coverage.
  const engineClean = rotorBeatScore(singleEngine(200), 199, SR);
  assert.ok(engineClean > 0.9,
    `expected the known zero-noise leakage artifact to persist, got ${engineClean} - ` +
    `if this now scores low the artifact was fixed and this comment should say so`);

  // A trace of noise (1/50th the tone's amplitude - far less than any real
  // capture would have) is all it takes to break up that leakage and
  // restore separation.
  const engine = rotorBeatScore(singleEngine(200, 0.01), 199, SR);
  assert.ok(engine < 0.3, `single engine (trace noise) scored ${engine}, expected little beating`);
  assert.ok(quad - engine > 0.5, `separation was only ${quad - engine}`);
});

test('the separation survives added noise', () => {
  const quad = rotorBeatScore(multirotor(200, 4, 0.3), 205, SR);
  const engine = rotorBeatScore(singleEngine(200, 0.3), 199, SR);
  assert.ok(quad > 0.6, `noisy quad scored ${quad}`);
  assert.ok(engine < 0.20, `noisy single engine scored ${engine}`);
});

test('silence and sub-audible f0 score zero', () => {
  assert.equal(rotorBeatScore(new Float64Array(N), 200, SR), 0);
  assert.equal(rotorBeatScore(multirotor(200, 4), 10, SR), 0, 'f0 below F0_MIN_HZ');
});

test('score stays inside 0..1', () => {
  for (const s of [multirotor(200, 4), singleEngine(200, 1.0), new Float64Array(N)]) {
    const v = rotorBeatScore(s, 200, SR);
    assert.ok(v >= 0 && v <= 1, `got ${v}`);
  }
});

// KNOWN LIMITATION, asserted so it stays visible rather than being discovered
// on stage. A single engine whose output is strongly amplitude-modulated in
// the beat band (an engine under varying load, a blower being swept) mimics
// multirotor beating. Detection cannot currently reject it - the detrend+
// Hann change made this worse, not better (0.75 before, saturates at 1.0
// now), which is further evidence rotorBeat alone cannot carry rejection of
// this confuser and the mel-shape fingerprint has to do that work instead.
test('documented limitation: a strongly AM-modulated single engine reads as beating', () => {
  const throbbing = rotorBeatScore(singleEngine(200, 0.1, 12), 199, SR);
  assert.ok(throbbing > 0.5,
    `expected this known false positive to persist, got ${throbbing} - if this ` +
    `now passes below 0.5 the limitation was fixed and the docs should say so`);
});
