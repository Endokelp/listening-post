// Iterative radix-2 Cooley-Tukey, in place. Owned rather than taken from
// AnalyserNode so the Python training pipeline can match it exactly.
export function fft(re, im) {
  const n = re.length;
  if ((n & (n - 1)) !== 0) throw new Error(`fft length must be a power of 2, got ${n}`);

  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cwr = 1, cwi = 0;
      for (let j = 0; j < half; j++) {
        const ur = re[i + j], ui = im[i + j];
        const xr = re[i + j + half], xi = im[i + j + half];
        const vr = xr * cwr - xi * cwi;
        const vi = xr * cwi + xi * cwr;
        re[i + j] = ur + vr; im[i + j] = ui + vi;
        re[i + j + half] = ur - vr; im[i + j + half] = ui - vi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr;
        cwr = nwr;
      }
    }
  }
}

// Magnitude spectrum of a real signal. Returns n/2 + 1 bins.
export function magnitudeSpectrum(samples) {
  const n = samples.length;
  const re = Float64Array.from(samples);
  const im = new Float64Array(n);
  fft(re, im);
  const out = new Float64Array(n / 2 + 1);
  for (let i = 0; i < out.length; i++) out[i] = Math.hypot(re[i], im[i]);
  return out;
}

// Inverse FFT of a spectrum given as separate re/im arrays. In place; returns real part.
export function ifftReal(re, im) {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = re[i] / n;
  return out;
}
