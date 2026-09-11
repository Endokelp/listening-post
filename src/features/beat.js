import { fft, magnitudeSpectrum } from './fft.js';
import { hann } from './window.js';
import { BEAT_MIN_HZ, BEAT_MAX_HZ, F0_MIN_HZ, BEAT_FLATNESS_REF } from './config.js';

const prevPow2 = (n) => 2 ** Math.floor(Math.log2(n));

// Zero every bin outside [lo, hi] Hz and transform back. Both halves of the
// spectrum are zeroed so the result stays real.
function bandpass(samples, loHz, hiHz, sampleRate) {
  const n = prevPow2(samples.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = samples[i];
  fft(re, im);
  const lo = Math.floor((loHz * n) / sampleRate);
  const hi = Math.ceil((hiHz * n) / sampleRate);
  for (let k = 0; k <= n / 2; k++) {
    if (k < lo || k > hi) {
      re[k] = 0; im[k] = 0;
      const mirror = (n - k) % n;
      re[mirror] = 0; im[mirror] = 0;
    }
  }
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = re[i] / n;
  return out;
}

const clamp01 = (x) => Math.min(1, Math.max(0, x));

// Four rotors at slightly different RPM beat against each other at a few Hz.
// A single-engine source has one fundamental and does not beat. This is the
// feature that separates a multirotor from a lawnmower.
export function rotorBeatScore(samples, f0, sampleRate) {
  if (!Number.isFinite(f0) || f0 < F0_MIN_HZ) return 0;

  // Wide enough to admit all four rotor tones and their spread.
  const band = Math.max(30, f0 * 0.25);
  const filtered = bandpass(samples, f0 - band, f0 + band, sampleRate);

  // Envelope: rectify, then remove trend so slow loudness drift (the mic
  // getting nearer/further, AGC settling) doesn't leak into the lowest
  // beat-band bins and swamp the beat peaks. A mean subtraction only
  // removes a flat DC offset - it leaves a linear ramp's energy smeared
  // across the low end of the spectrum, which is exactly where the beat
  // band starts looking. A least-squares linear detrend removes that ramp
  // too.
  const n = prevPow2(filtered.length);
  const env = new Float64Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) { env[i] = Math.abs(filtered[i]); mean += env[i]; }
  mean /= n;
  if (mean < 1e-12) return 0;

  const tMean = (n - 1) / 2;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const dt = i - tMean;
    num += dt * (env[i] - mean);
    den += dt * dt;
  }
  const slope = den > 0 ? num / den : 0;
  const w = hann(n);
  for (let i = 0; i < n; i++) env[i] = (env[i] - mean - slope * (i - tMean)) * w[i];

  const spec = magnitudeSpectrum(env);
  // Detrending is not perfect - residual trend leakage still concentrates in
  // bin 1, so the beat band now starts at bin 2 rather than bin 1.
  const lo = Math.max(2, Math.ceil((BEAT_MIN_HZ * n) / sampleRate));
  const hi = Math.min(spec.length - 1, Math.ceil((BEAT_MAX_HZ * n) / sampleRate));
  if (hi <= lo) return 0;

  let peak = 0, sum = 0, logSum = 0, sumSq = 0, bins = 0;
  for (let k = lo; k <= hi; k++) {
    const v = spec[k];
    if (v > peak) peak = v;
    sum += v;
    const p = v * v + 1e-20;
    logSum += Math.log(p);
    sumSq += p;
    bins++;
  }
  const bandMean = sum / bins + 1e-12;
  const flatness = Math.exp(logSum / bins) / (sumSq / bins + 1e-20);

  // Beating shows up as a FEW DISCRETE LINES in the envelope spectrum, so it
  // needs both a low in-band flatness and a peak standing above the band
  // mean. Broadband noise satisfies the second on its own - the tallest of
  // forty random bins always beats their mean - which is why requiring both
  // is what keeps noise from reading as a drone.
  const tonality = clamp01(Math.log10(BEAT_FLATNESS_REF / Math.max(flatness, 1e-9)));
  const prominence = clamp01(Math.log10(Math.max(peak / bandMean, 1)));

  // ponytail: rotor count is not estimated. More rotors spread the beat
  // energy across more difference frequencies and lower this score - a hex
  // reads weaker than a quad. Estimate the line count here if hexacopters
  // ever need equal footing.
  return tonality * prominence;
}
