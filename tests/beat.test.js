import test from 'node:test';
import assert from 'node:assert/strict';
import { rotorBeatScore } from '../src/features/beat.js';

const SR = 48000;
const N = 65536; // ~1.37 s, power of two

function mix(freqs) {
  const x = new Float64Array(N);
  for (const f of freqs) {
    for (let i = 0; i < N; i++) x[i] += Math.sin(2 * Math.PI * f * i / SR) / freqs.length;
  }
  return x;
}

test('four detuned rotors beat; one engine does not', () => {
  // Four rotors a few Hz apart - a quadcopter.
  const quad = mix([200, 203.5, 207, 211]);
  // One rotor with its harmonics - a single-engine confuser.
  const single = mix([200, 400, 600]);

  const quadScore = rotorBeatScore(quad, 200, SR);
  const singleScore = rotorBeatScore(single, 200, SR);

  assert.ok(quadScore > singleScore,
    `quad ${quadScore} should beat single-engine ${singleScore}`);
  assert.ok(quadScore > 0.4, `quad score ${quadScore} should be clearly present`);
  assert.ok(singleScore < 0.3, `single-engine score ${singleScore} should be low`);
});

test('score stays inside 0..1 and tolerates silence', () => {
  const s = rotorBeatScore(new Float64Array(N), 200, SR);
  assert.ok(s >= 0 && s <= 1, `got ${s}`);
});
