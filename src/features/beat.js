import { fft, magnitudeSpectrum } from './fft.js';
import { BEAT_MIN_HZ, BEAT_MAX_HZ, F0_MIN_HZ } from './config.js';

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

export function rotorBeatScore(samples, f0, sampleRate) {
  if (!Number.isFinite(f0) || f0 < F0_MIN_HZ) return 0;

  // Wide enough to admit all four rotor tones and their spread.
  const band = Math.max(30, f0 * 0.25);
  const filtered = bandpass(samples, f0 - band, f0 + band, sampleRate);

  // Envelope: rectify, then remove the DC term so the beat peaks dominate.
  const n = prevPow2(filtered.length);
  const env = new Float64Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) { env[i] = Math.abs(filtered[i]); mean += env[i]; }
  mean /= n;
  if (mean < 1e-12) return 0;
  for (let i = 0; i < n; i++) env[i] -= mean;

  const spec = magnitudeSpectrum(env);
  const lo = Math.max(1, Math.floor((BEAT_MIN_HZ * n) / sampleRate));
  const hi = Math.min(spec.length - 1, Math.ceil((BEAT_MAX_HZ * n) / sampleRate));
  if (hi <= lo) return 0;

  let peak = 0, peakBin = lo;
  for (let k = lo; k <= hi; k++) {
    if (spec[k] > peak) { peak = spec[k]; peakBin = k; }
  }
  // Floor is the local median within the beat band itself (excluding bins
  // near the peak), not the median of the whole envelope spectrum. The
  // envelope's rectification harmonics (at 2*f0, 4*f0, ...) sit far outside
  // the beat band and make the global median numerically vanish for a
  // near-periodic single-tone envelope, which inflates any FFT leakage in
  // the beat band into a false high score. A broadband envelope still fails
  // to score high here: its in-band bins are flat, so peak barely exceeds
  // the local median.
  const rest = [];
  for (let k = lo; k <= hi; k++) if (Math.abs(k - peakBin) > 2) rest.push(spec[k]);
  rest.sort((a, b) => a - b);
  const floor = (rest.length ? rest[rest.length >> 1] : 0) + 1e-9;

  // One decade of prominence over the floor maps to a full score.
  return Math.min(1, Math.log10(Math.max(peak / floor, 1)) / 1.5);
}
