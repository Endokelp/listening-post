import { magnitudeSpectrum } from './fft.js';
import { hann } from './window.js';
import { STFT_SIZE, STFT_HOP, N_MELS } from './config.js';

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
export function buildMelBank(sampleRate, fftSize, nMels) {
  const nBins = fftSize / 2 + 1;
  const melMin = hzToMel(0);
  const melMax = hzToMel(sampleRate / 2);
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
  const bank = buildMelBank(sampleRate, STFT_SIZE, N_MELS);
  const w = hann(STFT_SIZE);
  const frames = [];
  for (let start = 0; start + STFT_SIZE <= samples.length; start += STFT_HOP) {
    const chunk = new Float64Array(STFT_SIZE);
    for (let i = 0; i < STFT_SIZE; i++) chunk[i] = samples[start + i] * w[i];
    const mag = magnitudeSpectrum(chunk);
    const power = new Float64Array(mag.length);
    for (let i = 0; i < mag.length; i++) power[i] = mag[i] * mag[i];
    const frame = new Float64Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      let sum = 0;
      const filter = bank[m];
      for (let k = 0; k < filter.length; k++) sum += filter[k] * power[k];
      frame[m] = Math.log(sum + 1e-10);
    }
    frames.push(frame);
  }
  return frames;
}
