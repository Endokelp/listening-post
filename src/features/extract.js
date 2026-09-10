import { averagedSpectrum, findF0, harmonicStackScore } from './harmonics.js';
import { rotorBeatScore } from './beat.js';
import { logMelSpectrogram } from './melbank.js';
import { HARMONIC_FFT_SIZE } from './config.js';

export const FEATURE_ORDER = ['harmonicStack', 'rotorBeat', 'bandRatio', 'spectralFlatness'];

// Fraction of total energy in the band where multirotor blade-pass tones and
// their first harmonics live.
function bandRatio(spectrum, sampleRate, fftSize) {
  const lo = Math.round((100 * fftSize) / sampleRate);
  const hi = Math.round((2000 * fftSize) / sampleRate);
  let band = 0, total = 0;
  for (let k = 1; k < spectrum.length; k++) {
    const p = spectrum[k] * spectrum[k];
    total += p;
    if (k >= lo && k <= hi) band += p;
  }
  return total > 0 ? band / total : 0;
}

// Geometric mean over arithmetic mean. Near 1 for white noise, near 0 for tonal.
function spectralFlatness(spectrum) {
  let logSum = 0, sum = 0, n = 0;
  for (let k = 1; k < spectrum.length; k++) {
    const p = spectrum[k] * spectrum[k] + 1e-20;
    logSum += Math.log(p);
    sum += p;
    n++;
  }
  if (n === 0 || sum === 0) return 0;
  return Math.exp(logSum / n) / (sum / n);
}

export function extractFeatures(samples, sampleRate) {
  const spectrum = averagedSpectrum(samples);
  const f0 = findF0(spectrum, sampleRate, HARMONIC_FFT_SIZE);
  return {
    f0,
    harmonicStack: harmonicStackScore(spectrum, f0, sampleRate, HARMONIC_FFT_SIZE),
    rotorBeat: rotorBeatScore(samples, f0, sampleRate),
    bandRatio: bandRatio(spectrum, sampleRate, HARMONIC_FFT_SIZE),
    spectralFlatness: spectralFlatness(spectrum),
    logMel: logMelSpectrogram(samples, sampleRate),
  };
}
