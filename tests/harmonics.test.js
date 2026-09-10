import test from 'node:test';
import assert from 'node:assert/strict';
import { averagedSpectrum, findF0, harmonicStackScore } from '../src/features/harmonics.js';
import { HARMONIC_FFT_SIZE } from '../src/features/config.js';

const SR = 48000;

function tone(n, sampleRate, freqs) {
  const x = new Float64Array(n);
  for (const f of freqs) {
    for (let i = 0; i < n; i++) x[i] += Math.sin(2 * Math.PI * f * i / sampleRate) / freqs.length;
  }
  return x;
}

function noise(n, seed = 1) {
  const x = new Float64Array(n);
  let s = seed;
  for (let i = 0; i < n; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    x[i] = (s / 0x7fffffff) * 2 - 1;
  }
  return x;
}

test('findF0 locates the fundamental of a harmonic stack', () => {
  const x = tone(HARMONIC_FFT_SIZE, SR, [200, 400, 600, 800]);
  const spec = averagedSpectrum(x);
  const f0 = findF0(spec, SR, HARMONIC_FFT_SIZE);
  assert.ok(Math.abs(f0 - 200) < 10, `expected ~200 Hz, got ${f0}`);
});

test('a harmonic stack scores higher than broadband noise', () => {
  const harmonic = tone(HARMONIC_FFT_SIZE, SR, [200, 400, 600, 800, 1000, 1200]);
  const hs = averagedSpectrum(harmonic);
  const harmonicScore = harmonicStackScore(hs, findF0(hs, SR, HARMONIC_FFT_SIZE), SR, HARMONIC_FFT_SIZE);

  const ns = averagedSpectrum(noise(HARMONIC_FFT_SIZE));
  const noiseScore = harmonicStackScore(ns, findF0(ns, SR, HARMONIC_FFT_SIZE), SR, HARMONIC_FFT_SIZE);

  assert.ok(harmonicScore > noiseScore, `harmonic ${harmonicScore} should beat noise ${noiseScore}`);
  assert.ok(harmonicScore > 0.5, `harmonic score ${harmonicScore} should be strong`);
});

test('scores stay inside 0..1', () => {
  const x = tone(HARMONIC_FFT_SIZE, SR, [200, 400]);
  const spec = averagedSpectrum(x);
  const s = harmonicStackScore(spec, findF0(spec, SR, HARMONIC_FFT_SIZE), SR, HARMONIC_FFT_SIZE);
  assert.ok(s >= 0 && s <= 1);
});
