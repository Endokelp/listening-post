export const WINDOW_SEC = 2;
export const HOP_SEC = 1;
export const STFT_SIZE = 2048;
export const STFT_HOP = 512;
export const HARMONIC_FFT_SIZE = 8192;
export const N_MELS = 64;
export const F0_MIN_HZ = 80;
export const F0_MAX_HZ = 1000;
export const N_HARMONICS = 6;
export const BEAT_MIN_HZ = 1;
export const BEAT_MAX_HZ = 30;

// In-band envelope-spectrum flatness at or above which beating is considered
// absent. Broadband noise sits near 0.14, a clean single engine near 0.98,
// a quadcopter near 0.03. Calibrated against synthetic signals; re-check
// against real recordings once own-drone data exists.
export const BEAT_FLATNESS_REF = 0.3;
