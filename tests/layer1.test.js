import test from 'node:test';
import assert from 'node:assert/strict';
import { classify, DEFAULT_WEIGHTS } from '../src/classify/layer1.js';
import { CLASSIFIER_FEATURES } from '../src/features/extract.js';

// These tests pin the classifier's mechanics, not the learned weight values -
// those change every time train/train_layer1.py runs. An earlier test here
// asserted "rotor beat must be what separates a drone from a mower"; real
// audio disproved that (real engines pulse in the beat band) and the trained
// model gives rotor beat a negative weight. See README "The hard part".

test('shipped weights cover every classifier feature', () => {
  for (const key of CLASSIFIER_FEATURES) {
    assert.ok(key in DEFAULT_WEIGHTS.coefficients, `missing coefficient for ${key}`);
  }
});

test('classify is the logistic of intercept plus weighted features', () => {
  const w = { intercept: -1, coefficients: { a: 2, b: -0.5 } };
  const expected = 1 / (1 + Math.exp(-(-1 + 2 * 0.75 - 0.5 * 2)));
  assert.ok(Math.abs(classify({ a: 0.75, b: 2 }, w) - expected) < 1e-12);
});

test('a feature missing from the input counts as zero', () => {
  const w = { intercept: 0.3, coefficients: { a: 5 } };
  assert.ok(Math.abs(classify({}, w) - 1 / (1 + Math.exp(-0.3))) < 1e-12);
});

test('output is always a probability', () => {
  for (const v of [0, 1, 50]) {
    const features = Object.fromEntries(CLASSIFIER_FEATURES.map((k) => [k, v]));
    const p = classify(features, DEFAULT_WEIGHTS);
    assert.ok(p >= 0 && p <= 1, `got ${p} for all features = ${v}`);
  }
});
