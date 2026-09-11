import { magnitudeSpectrum } from './fft.js';
import { hann } from './window.js';
import { STFT_SIZE, STFT_HOP, MEL_BANDS, MEL_FMIN_HZ, FEATURE_MAX_HZ } from './config.js';

// HTK mel scale. Slaney's variant is deliberately not used - the Python
// mirror must match this exactly.
export function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

export function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

// Triangular filters, unnormalised. Normalisation is skipped so the Python
// mirror has one fewer thing to get wrong.
export function buildMelBank(sampleRate, fftSize, nMels, fMinHz, fMaxHz) {
  const nBins = fftSize / 2 + 1;
  const melMin = hzToMel(fMinHz);
  const melMax = hzToMel(fMaxHz);
  const points = new Float64Array(nMels + 2);
  for (let i = 0; i < points.length; i++) {
    const mel = melMin + ((melMax - melMin) * i) / (nMels + 1);
    points[i] = (melToHz(mel) * fftSize) / sampleRate;
  }
  const bank = [];
  for (let m = 1; m <= nMels; m++) {
    const filter = new Float64Array(nBins);
    const left = points[m - 1], centre = points[m], right = points[m + 1];
    for (let k = 0; k < nBins; k++) {
      if (k >= left && k <= centre && centre > left) filter[k] = (k - left) / (centre - left);
      else if (k > centre && k <= right && right > centre) filter[k] = (right - k) / (right - centre);
    }
    bank.push(filter);
  }
  return bank;
}

export function logMelSpectrogram(samples, sampleRate) {
  const bank = buildMelBank(sampleRate, STFT_SIZE, MEL_BANDS, MEL_FMIN_HZ, FEATURE_MAX_HZ);
  const w = hann(STFT_SIZE);
  const frames = [];
  for (let start = 0; start + STFT_SIZE <= samples.length; start += STFT_HOP) {
    const chunk = new Float64Array(STFT_SIZE);
    for (let i = 0; i < STFT_SIZE; i++) chunk[i] = samples[start + i] * w[i];
    const mag = magnitudeSpectrum(chunk);
    const power = new Float64Array(mag.length);
    for (let i = 0; i < mag.length; i++) power[i] = mag[i] * mag[i];
    const frame = new Float64Array(MEL_BANDS);
    for (let m = 0; m < MEL_BANDS; m++) {
      let sum = 0;
      const filter = bank[m];
      for (let k = 0; k < filter.length; k++) sum += filter[k] * power[k];
      frame[m] = Math.log(sum + 1e-10);
    }
    frames.push(frame);
  }
  return frames;
}

// Per-frame mean/std of the gain-invariant spectral shape, plus how much the
// overall loudness itself moves. Subtracting each frame's mean log level
// (loud) before averaging makes the fingerprint independent of gain and
// distance - a quiet drone and a loud one at the same range should produce
// the same shape. The std over frames captures how that shape moves from
// frame to frame: engine firing pulses shuffle energy between bands, a
// steady rotor tone does not.
export function melShapeStats(frames) {
  const nFrames = frames.length;
  const melMean = new Float64Array(MEL_BANDS);
  const melStd = new Float64Array(MEL_BANDS);
  let loudnessStd = 0;
  if (nFrames === 0) return { melMean, melStd, loudnessStd };

  const loud = new Float64Array(nFrames);
  const shapes = new Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    const frame = frames[f];
    let sum = 0;
    for (let m = 0; m < MEL_BANDS; m++) sum += frame[m];
    const l = sum / MEL_BANDS;
    loud[f] = l;
    const shape = new Float64Array(MEL_BANDS);
    for (let m = 0; m < MEL_BANDS; m++) shape[m] = frame[m] - l;
    shapes[f] = shape;
  }

  for (let m = 0; m < MEL_BANDS; m++) {
    let sum = 0;
    for (let f = 0; f < nFrames; f++) sum += shapes[f][m];
    melMean[m] = sum / nFrames;
  }
  for (let m = 0; m < MEL_BANDS; m++) {
    let sumSq = 0;
    for (let f = 0; f < nFrames; f++) {
      const d = shapes[f][m] - melMean[m];
      sumSq += d * d;
    }
    melStd[m] = Math.sqrt(sumSq / nFrames);
  }

  let loudMean = 0;
  for (let f = 0; f < nFrames; f++) loudMean += loud[f];
  loudMean /= nFrames;
  let sumSq = 0;
  for (let f = 0; f < nFrames; f++) {
    const d = loud[f] - loudMean;
    sumSq += d * d;
  }
  loudnessStd = Math.sqrt(sumSq / nFrames);

  return { melMean, melStd, loudnessStd };
}
