export const WINDOW_SEC = 2;
export const HOP_SEC = 1;
export const STFT_SIZE = 2048;
export const STFT_HOP = 512;
export const HARMONIC_FFT_SIZE = 8192;
export const N_MELS = 64;
export const F0_MIN_HZ = 80;
export const F0_MAX_HZ = 1000;
export const N_HARMONICS = 6;
// No feature reads above this. Public drone datasets are 16 kHz: without a
// ceiling, full-band flatness drops from 0.96 to 0.0001 on 16 kHz-origin
// audio and the classifier learns the recording's sample rate, not the
// sound. The Al-Emadi drones' own roll-off is measured at -2 dB by 7.0 kHz
// and -11 dB by 7.5 kHz, so the ceiling sits below that knee. Also makes
// 44.1 and 48 kHz phones compute the same thing.
export const FEATURE_MAX_HZ = 6500;
export const BEAT_MIN_HZ = 1;
export const BEAT_MAX_HZ = 30;

// In-band envelope-spectrum flatness at or above which beating is considered
// absent. Broadband noise sits near 0.14, a clean single engine near 0.98,
// a quadcopter near 0.03. Calibrated against synthetic signals; re-check
// against real recordings once own-drone data exists.
export const BEAT_FLATNESS_REF = 0.3;
