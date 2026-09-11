import test from 'node:test';
import assert from 'node:assert/strict';
import { extractFeatures, FEATURE_ORDER } from '../src/features/extract.js';

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
  assert.ok(f.logMel.length > 0);
});

// A 16 kHz public recording has nothing above 8 kHz. If a feature could see
// that band, the classifier would learn the sample rate instead of the sound.
test('energy above FEATURE_MAX_HZ cannot move any spectral feature', () => {
  const rng = (() => { let s = 1; return () => ((s = (s * 16807) % 2147483647) / 2147483647) - 0.5; })();
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) x[i] = Math.sin(2 * Math.PI * 200 * i / SR) + rng() * 0.2;
  const withHiss = x.map((v, i) => v + Math.sin(2 * Math.PI * 12000 * i / SR));
  const a = extractFeatures(x, SR);
  const b = extractFeatures(withHiss, SR);
  // Uncapped, this tone moves bandRatio by ~0.5. What remains (~4e-6) is
  // leakage through averagedSpectrum's truncated last frame.
  for (const key of ['bandRatio', 'spectralFlatness']) {
    assert.ok(Math.abs(a[key] - b[key]) < 1e-4, `${key}: ${a[key]} vs ${b[key]}`);
  }
});

test('silence produces finite, low features', () => {
  const f = extractFeatures(new Float64Array(N), SR);
  for (const key of FEATURE_ORDER) assert.ok(Number.isFinite(f[key]), `${key} not finite`);
});
