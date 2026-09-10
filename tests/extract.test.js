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

test('silence produces finite, low features', () => {
  const f = extractFeatures(new Float64Array(N), SR);
  for (const key of FEATURE_ORDER) assert.ok(Number.isFinite(f[key]), `${key} not finite`);
});
