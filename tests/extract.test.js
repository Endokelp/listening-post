import test from 'node:test';
import assert from 'node:assert/strict';
import { extractFeatures, FEATURE_ORDER, CLASSIFIER_FEATURES } from '../src/features/extract.js';

const SR = 48000;
const N = 96000; // 2 s

test('extractFeatures returns every named feature in range', () => {
  const x = new Float64Array(N);
  for (const f of [200, 203.5, 207, 211, 400, 600]) {
    for (let i = 0; i < N; i++) x[i] += Math.sin(2 * Math.PI * f * i / SR) / 6;
  }
  const f = extractFeatures(x, SR);
  assert.ok(f.f0 > 150 && f.f0 < 260, `f0 was ${f.f0}`);
  for (const key of FEATURE_ORDER) {
    assert.ok(Number.isFinite(f[key]), `${key} not finite`);
    assert.ok(f[key] >= 0 && f[key] <= 1, `${key} out of range: ${f[key]}`);
  }
});

const lcg = () => { let s = 1; return () => ((s = (s * 16807) % 2147483647) / 2147483647) - 0.5; };

// A 16 kHz public recording has nothing above 8 kHz. If a feature could see
// that band, the classifier would learn the sample rate instead of the sound.
test('energy above FEATURE_MAX_HZ cannot move any spectral feature', () => {
  const rng = lcg();
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) x[i] = Math.sin(2 * Math.PI * 200 * i / SR) + rng() * 0.2;
  const withHiss = x.map((v, i) => v + Math.sin(2 * Math.PI * 12000 * i / SR));
  const a = extractFeatures(x, SR);
  const b = extractFeatures(withHiss, SR);
  // Uncapped, this tone moves bandRatio by ~0.5. What remains (~4e-6) is
  // leakage through averagedSpectrum's truncated last frame. The mel bank is
  // capped at the same FEATURE_MAX_HZ ceiling, so its keys get the same check.
  const keys = ['bandRatio', 'spectralFlatness', ...CLASSIFIER_FEATURES.filter(
    (k) => k.startsWith('melMean') || k.startsWith('melStd') || k === 'loudnessStd')];
  for (const key of keys) {
    assert.ok(Math.abs(a[key] - b[key]) < 1e-4, `${key}: ${a[key]} vs ${b[key]}`);
  }
});

test('silence produces finite, low features', () => {
  const f = extractFeatures(new Float64Array(N), SR);
  for (const key of FEATURE_ORDER) assert.ok(Number.isFinite(f[key]), `${key} not finite`);
});

test('every CLASSIFIER_FEATURES value is finite for a tone and for silence', () => {
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) x[i] = Math.sin(2 * Math.PI * 200 * i / SR);
  for (const samples of [x, new Float64Array(N)]) {
    const f = extractFeatures(samples, SR);
    for (const key of CLASSIFIER_FEATURES) assert.ok(Number.isFinite(f[key]), `${key} not finite`);
  }
});

// The mel-shape fingerprint subtracts each frame's own mean log level, which
// should make it blind to gain (a louder mic, a closer drone) - only the
// shape of the spectrum should matter, not its overall level. Uses a
// broadband signal (tone + noise) rather than pure discrete tones: a mel
// band with essentially no energy in it sits near logMelSpectrogram's 1e-10
// floor, and that fixed floor does NOT scale with gain, so a band that's
// silent except for FFT leakage can drift by several hundredths of a
// log-unit under a 3.7x gain change (measured up to ~0.2 in the
// highest-frequency bands for a pure 200/400/600 Hz tone with no noise).
// Every real signal has some broadband energy in every band, so this is not
// a practical concern - it's specific to synthetic near-silence, which is
// why the fixture below is broadband instead.
test('CLASSIFIER_FEATURES are gain-invariant', () => {
  const rng = lcg();
  const x = new Float64Array(N);
  for (const f of [200, 400, 600]) {
    for (let i = 0; i < N; i++) x[i] += Math.sin(2 * Math.PI * f * i / SR) / 3;
  }
  for (let i = 0; i < N; i++) x[i] += rng() * 0.05;
  const loud = x.map((v) => v * 3.7);
  const a = extractFeatures(x, SR);
  const b = extractFeatures(loud, SR);
  for (const key of CLASSIFIER_FEATURES) {
    const drift = Math.abs(a[key] - b[key]);
    assert.ok(drift < 1e-9, `${key} not gain-invariant: ${a[key]} vs ${b[key]} (${drift})`);
  }
});
