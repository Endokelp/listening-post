import { magnitudeSpectrum } from './fft.js';
import { hann } from './window.js';
import { HARMONIC_FFT_SIZE, F0_MIN_HZ, F0_MAX_HZ, N_HARMONICS } from './config.js';

export function averagedSpectrum(samples) {
  const chunk = new Float64Array(HARMONIC_FFT_SIZE);
  const n = Math.min(samples.length, HARMONIC_FFT_SIZE);
  const w = hann(HARMONIC_FFT_SIZE);
  for (let i = 0; i < n; i++) chunk[i] = samples[i] * w[i];
  return magnitudeSpectrum(chunk);
}

const binOf = (hz, sampleRate, fftSize) => Math.round((hz * fftSize) / sampleRate);

export function findF0(spectrum, sampleRate, fftSize) {
  const lo = Math.max(1, binOf(F0_MIN_HZ, sampleRate, fftSize));
  const hi = Math.min(spectrum.length - 1, binOf(F0_MAX_HZ, sampleRate, fftSize));
  let peak = lo;
  for (let k = lo; k <= hi; k++) if (spectrum[k] > spectrum[peak]) peak = k;
  return (peak * sampleRate) / fftSize;
}

// Median magnitude in a window around a bin, excluding the peak itself.
// Used as the local noise floor a harmonic must stand above.
function localMedian(spectrum, centre, halfWidth) {
  const lo = Math.max(0, centre - halfWidth);
  const hi = Math.min(spectrum.length - 1, centre + halfWidth);
  const vals = [];
  for (let k = lo; k <= hi; k++) if (Math.abs(k - centre) > 2) vals.push(spectrum[k]);
  if (vals.length === 0) return 1e-10;
  vals.sort((a, b) => a - b);
  return vals[vals.length >> 1] + 1e-10;
}

// Peak magnitude within +/-2 bins of the expected harmonic, so slight
// detuning does not zero the score.
function peakNear(spectrum, centre) {
  let best = 0;
  for (let k = Math.max(0, centre - 2); k <= Math.min(spectrum.length - 1, centre + 2); k++) {
    if (spectrum[k] > best) best = spectrum[k];
  }
  return best;
}

export function harmonicStackScore(spectrum, f0, sampleRate, fftSize) {
  if (!Number.isFinite(f0) || f0 < F0_MIN_HZ) return 0;
  const halfWidth = Math.max(8, binOf(f0 / 2, sampleRate, fftSize));
  let total = 0, counted = 0;
  for (let h = 1; h <= N_HARMONICS; h++) {
    const centre = binOf(f0 * h, sampleRate, fftSize);
    if (centre >= spectrum.length - 3) break;
    const prominence = peakNear(spectrum, centre) / localMedian(spectrum, centre, halfWidth);
    // A prominence of 1 means "indistinguishable from the local floor".
    total += Math.log10(Math.max(prominence, 1));
    counted++;
  }
  if (counted === 0) return 0;
  // Two decades of prominence maps to a full score.
  return Math.min(1, total / counted / 2);
}
