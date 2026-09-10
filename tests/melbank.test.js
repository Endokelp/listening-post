import test from 'node:test';
import assert from 'node:assert/strict';
import { hzToMel, melToHz, buildMelBank, logMelSpectrogram } from '../src/features/melbank.js';
import { N_MELS } from '../src/features/config.js';

test('HTK mel conversion round-trips', () => {
  for (const hz of [100, 440, 1000, 4000]) {
    assert.ok(Math.abs(melToHz(hzToMel(hz)) - hz) < 1e-6);
  }
});

test('HTK mel formula matches the reference value at 1000 Hz', () => {
  assert.ok(Math.abs(hzToMel(1000) - 2595 * Math.log10(1 + 1000 / 700)) < 1e-9);
});

test('mel bank has the requested shape and non-negative weights', () => {
  const bank = buildMelBank(48000, 2048, N_MELS);
  assert.equal(bank.length, N_MELS);
  assert.equal(bank[0].length, 2048 / 2 + 1);
  for (const f of bank) for (const w of f) assert.ok(w >= 0);
});

test('logMelSpectrogram returns frames of N_MELS bins', () => {
  const sampleRate = 48000;
  const x = new Float64Array(sampleRate);
  for (let i = 0; i < x.length; i++) x[i] = Math.sin(2 * Math.PI * 220 * i / sampleRate);
  const frames = logMelSpectrogram(x, sampleRate);
  assert.ok(frames.length > 0);
  assert.equal(frames[0].length, N_MELS);
});
