import test from 'node:test';
import assert from 'node:assert/strict';
import { createDetector } from '../src/detect/state-machine.js';

test('a single high frame does not fire', () => {
  const fired = [];
  const d = createDetector({ onAlert: (a) => fired.push(a) });
  assert.equal(d.push(0.95, 0), 'idle');
  assert.equal(fired.length, 0);
});

test('three of five high frames fire exactly one alert', () => {
  const fired = [];
  const d = createDetector({ onAlert: (a) => fired.push(a) });
  d.push(0.9, 0);
  d.push(0.2, 1000);
  d.push(0.9, 2000);
  assert.equal(d.push(0.9, 3000), 'detecting');
  assert.equal(fired.length, 1);
});

test('hysteresis keeps detection alive through a dip', () => {
  const d = createDetector({ onAlert: () => {} });
  d.push(0.9, 0); d.push(0.9, 1000); d.push(0.9, 2000);
  assert.equal(d.push(0.5, 3000), 'detecting', '0.5 is above exitThreshold');
});

test('sustained low frames return to idle', () => {
  const d = createDetector({ onAlert: () => {} });
  d.push(0.9, 0); d.push(0.9, 1000); d.push(0.9, 2000);
  for (let i = 0; i < 6; i++) d.push(0.05, 4000 + i * 1000);
  assert.equal(d.push(0.05, 12000), 'idle');
});

test('cooldown suppresses a second alert', () => {
  const fired = [];
  const d = createDetector({ cooldownMs: 10000, onAlert: (a) => fired.push(a) });
  d.push(0.9, 0); d.push(0.9, 1000); d.push(0.9, 2000);
  for (let i = 0; i < 6; i++) d.push(0.05, 3000 + i * 100);
  d.push(0.9, 4000); d.push(0.9, 4100); d.push(0.9, 4200);
  assert.equal(fired.length, 1, 'second episode inside cooldown must not alert');
});
