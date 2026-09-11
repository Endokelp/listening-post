import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { extractFeatures, CLASSIFIER_FEATURES } from '../src/features/extract.js';

const golden = JSON.parse(
  await readFile(new URL('./fixtures/golden.json', import.meta.url), 'utf8'),
);

test('golden fixture expects exactly f0 + CLASSIFIER_FEATURES', () => {
  const expectedKeys = new Set(['f0', ...CLASSIFIER_FEATURES]);
  const fixtureKeys = new Set(Object.keys(golden.expected));
  assert.deepEqual(fixtureKeys, expectedKeys,
    'the Python mirror is missing (or has extra) feature keys - regenerate ' +
    'with python train/export_golden.py after fixing features.py');
});

test('JS features match the Python mirror within tolerance', () => {
  const actual = extractFeatures(Float64Array.from(golden.samples), golden.sampleRate);

  assert.ok(Math.abs(actual.f0 - golden.expected.f0) < 1e-6,
    `f0: js ${actual.f0} vs py ${golden.expected.f0}`);

  for (const key of CLASSIFIER_FEATURES) {
    const drift = Math.abs(actual[key] - golden.expected[key]);
    assert.ok(drift < 1e-6,
      `${key} drifted: js ${actual[key]} vs py ${golden.expected[key]} (${drift})`);
  }
});

test('the golden fixture is a drone-like signal, not silence', () => {
  assert.ok(golden.expected.rotorBeat > 0.3,
    'fixture should exhibit rotor beating or it proves nothing');
});
