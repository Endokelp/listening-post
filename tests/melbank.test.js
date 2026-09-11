import test from 'node:test';
import assert from 'node:assert/strict';
import { hzToMel, melToHz, buildMelBank, logMelSpectrogram } from '../src/features/melbank.js';
import { MEL_BANDS, MEL_FMIN_HZ, FEATURE_MAX_HZ, STFT_SIZE } from '../src/features/config.js';

test('HTK mel conversion round-trips', () => {
  for (const hz of [100, 440, 1000, 4000]) {
    assert.ok(Math.abs(melToHz(hzToMel(hz)) - hz) < 1e-6);
  }
});

test('HTK mel formula matches the reference value at 1000 Hz', () => {
  assert.ok(Math.abs(hzToMel(1000) - 2595 * Math.log10(1 + 1000 / 700)) < 1e-9);
});

test('mel bank has the requested shape and non-negative weights', () => {
  const bank = buildMelBank(48000, STFT_SIZE, MEL_BANDS, MEL_FMIN_HZ, FEATURE_MAX_HZ);
  assert.equal(bank.length, MEL_BANDS);
  assert.equal(bank[0].length, STFT_SIZE / 2 + 1);
  for (const f of bank) for (const w of f) assert.ok(w >= 0);
});

// Every filter should carry some energy, and none should reach past the
// ceiling the rest of the feature set is capped at - otherwise the mel
// fingerprint would leak the same sample-rate tell FEATURE_MAX_HZ exists to
// hide (see config.js).
for (const sampleRate of [48000, 16000]) {
  test(`mel bank filters are non-empty and capped at FEATURE_MAX_HZ at ${sampleRate} Hz`, () => {
    const bank = buildMelBank(sampleRate, STFT_SIZE, MEL_BANDS, MEL_FMIN_HZ, FEATURE_MAX_HZ);
    const topBin = Math.floor((FEATURE_MAX_HZ * STFT_SIZE) / sampleRate);
    for (const filter of bank) {
      assert.ok(filter.some((w) => w > 0), 'filter has no weight at all');
      for (let k = topBin + 1; k < filter.length; k++) {
        assert.equal(filter[k], 0, `bin ${k} above FEATURE_MAX_HZ has weight ${filter[k]}`);
      }
    }
  });
}

test('logMelSpectrogram returns frames of MEL_BANDS bins', () => {
  const sampleRate = 48000;
  const x = new Float64Array(sampleRate);
  for (let i = 0; i < x.length; i++) x[i] = Math.sin(2 * Math.PI * 220 * i / sampleRate);
  const frames = logMelSpectrogram(x, sampleRate);
  assert.ok(frames.length > 0);
  assert.equal(frames[0].length, MEL_BANDS);
});
