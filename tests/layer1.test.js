import test from 'node:test';
import assert from 'node:assert/strict';
import { classify, DEFAULT_WEIGHTS } from '../src/classify/layer1.js';
import { FEATURE_ORDER } from '../src/features/extract.js';

test('weights file covers every feature layer 1 consumes', () => {
  for (const key of FEATURE_ORDER) {
    assert.ok(key in DEFAULT_WEIGHTS.coefficients, `missing coefficient for ${key}`);
  }
});

test('strong harmonic stack plus strong beat scores higher than neither', () => {
  const drone = { harmonicStack: 0.9, rotorBeat: 0.8, bandRatio: 0.7, spectralFlatness: 0.1 };
  const quiet = { harmonicStack: 0.05, rotorBeat: 0.05, bandRatio: 0.2, spectralFlatness: 0.8 };
  assert.ok(classify(drone, DEFAULT_WEIGHTS) > classify(quiet, DEFAULT_WEIGHTS));
});

test('a single-engine confuser - harmonics but no beat - scores below a drone', () => {
  const drone = { harmonicStack: 0.9, rotorBeat: 0.8, bandRatio: 0.7, spectralFlatness: 0.1 };
  const mower = { harmonicStack: 0.9, rotorBeat: 0.1, bandRatio: 0.7, spectralFlatness: 0.1 };
  assert.ok(classify(mower, DEFAULT_WEIGHTS) < classify(drone, DEFAULT_WEIGHTS),
    'rotor beat must be what separates them');
});

test('output is always a probability', () => {
  const extreme = { harmonicStack: 1, rotorBeat: 1, bandRatio: 1, spectralFlatness: 1 };
  const p = classify(extreme, DEFAULT_WEIGHTS);
  assert.ok(p >= 0 && p <= 1);
});
