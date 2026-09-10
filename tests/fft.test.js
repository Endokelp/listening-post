import test from 'node:test';
import assert from 'node:assert/strict';
import { magnitudeSpectrum } from '../src/features/fft.js';
import { hann } from '../src/features/window.js';

test('magnitudeSpectrum puts a pure tone in the right bin', () => {
  const n = 1024, sampleRate = 8000, freq = 500;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * freq * i / sampleRate);
  const mag = magnitudeSpectrum(x);
  assert.equal(mag.length, n / 2 + 1);
  let peak = 0;
  for (let i = 1; i < mag.length; i++) if (mag[i] > mag[peak]) peak = i;
  assert.equal(peak, Math.round(freq * n / sampleRate));
});

test('hann is symmetric and starts and ends at zero', () => {
  const w = hann(8);
  assert.ok(Math.abs(w[0]) < 1e-12);
  assert.ok(Math.abs(w[7]) < 1e-12);
  assert.ok(Math.abs(w[1] - w[6]) < 1e-12);
});
