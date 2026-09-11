"""Mirror of src/features/. Any change here needs the same change there.
tests/golden-vector.test.js enforces it.

Ported from the CURRENT JavaScript on disk (src/features/config.js, fft.js,
window.js, melbank.js, harmonics.js, beat.js, extract.js), not from any
earlier draft. In particular:
  - averaged_spectrum is a Welch-style power average across consecutive
    HARMONIC_FFT_SIZE frames spanning the whole input (harmonics.js).
  - rotor_beat_score is tonality * prominence, where tonality is in-band
    envelope-spectrum flatness measured against BEAT_FLATNESS_REF and
    prominence is peak-over-band-mean (beat.js). It is NOT peak-over-median.
"""
import numpy as np

FEATURE_ORDER = ["harmonicStack", "rotorBeat", "bandRatio", "spectralFlatness"]
STFT_SIZE = 2048
STFT_HOP = 512
HARMONIC_FFT_SIZE = 8192
MEL_BANDS = 24
MEL_FMIN_HZ = 100
F0_MIN_HZ = 80
F0_MAX_HZ = 1000
N_HARMONICS = 6
FEATURE_MAX_HZ = 6500
BEAT_MIN_HZ = 1
BEAT_MAX_HZ = 30
BEAT_FLATNESS_REF = 0.3

# The 4 meter features plus the gain-invariant mel-shape fingerprint, in the
# order the classifier consumes them. Mirrors src/features/extract.js.
CLASSIFIER_FEATURES = (
    FEATURE_ORDER
    + [f"melMean{i}" for i in range(MEL_BANDS)]
    + [f"melStd{i}" for i in range(MEL_BANDS)]
    + ["loudnessStd"]
)


# HTK mel scale. Slaney's variant is deliberately not used - the JS side
# must match this exactly. Mirrors src/features/melbank.js.
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def build_mel_bank(sample_rate, fft_size, n_mels, f_min_hz, f_max_hz):
    """Triangular filters, unnormalised. Mirrors melbank.js buildMelBank:
    mel points evenly spaced from hz_to_mel(f_min_hz) to hz_to_mel(f_max_hz)."""
    n_bins = fft_size // 2 + 1
    mel_min = hz_to_mel(f_min_hz)
    mel_max = hz_to_mel(f_max_hz)
    points = np.array([
        mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1)) * fft_size / sample_rate
        for i in range(n_mels + 2)
    ])
    bank = []
    k = np.arange(n_bins)
    for m in range(1, n_mels + 1):
        left, centre, right = points[m - 1], points[m], points[m + 1]
        filt = np.zeros(n_bins)
        if centre > left:
            mask = (k >= left) & (k <= centre)
            filt[mask] = (k[mask] - left) / (centre - left)
        if right > centre:
            mask = (k > centre) & (k <= right)
            filt[mask] = (right - k[mask]) / (right - centre)
        bank.append(filt)
    return bank


def log_mel_spectrogram(samples, sample_rate):
    """Mirrors melbank.js logMelSpectrogram."""
    bank = build_mel_bank(sample_rate, STFT_SIZE, MEL_BANDS, MEL_FMIN_HZ, FEATURE_MAX_HZ)
    w = np.hanning(STFT_SIZE)
    frames = []
    n = len(samples)
    start = 0
    while start + STFT_SIZE <= n:
        chunk = samples[start:start + STFT_SIZE] * w
        power = np.abs(np.fft.rfft(chunk)) ** 2
        frame = np.array([np.log(np.sum(filt * power) + 1e-10) for filt in bank])
        frames.append(frame)
        start += STFT_HOP
    return frames


def mel_shape_stats(frames):
    """Mirrors melbank.js melShapeStats. Subtracting each frame's mean log
    level makes the fingerprint independent of gain and distance; std over
    frames captures how the spectrum moves (engine firing pulses vs steady
    rotor tones)."""
    if len(frames) == 0:
        return np.zeros(MEL_BANDS), np.zeros(MEL_BANDS), 0.0
    frames = np.array(frames)
    loud = frames.mean(axis=1)
    shape = frames - loud[:, None]
    mel_mean = shape.mean(axis=0)
    mel_std = shape.std(axis=0)  # ddof=0, population std - matches JS
    loudness_std = float(loud.std())  # ddof=0
    return mel_mean, mel_std, loudness_std


def averaged_spectrum(samples):
    """Welch-style power average over consecutive HARMONIC_FFT_SIZE frames
    spanning all of `samples` (last frame zero-padded if it doesn't fill).
    Degenerates to a single windowed FFT when len(samples) <= HARMONIC_FFT_SIZE.
    Mirrors src/features/harmonics.js averagedSpectrum."""
    w = np.hanning(HARMONIC_FFT_SIZE)
    n = len(samples)
    sum_power = None
    frames = 0
    start = 0
    limit = max(n, 1)
    while start < limit:
        chunk = np.zeros(HARMONIC_FFT_SIZE)
        avail = max(0, min(HARMONIC_FFT_SIZE, n - start))
        if avail > 0:
            chunk[:avail] = samples[start:start + avail] * w[:avail]
        mag = np.abs(np.fft.rfft(chunk))
        if sum_power is None:
            sum_power = np.zeros(len(mag))
        sum_power += mag ** 2
        frames += 1
        start += HARMONIC_FFT_SIZE
    return np.sqrt(sum_power / frames)


def _bin_of(hz, sample_rate, fft_size):
    # Matches JS Math.round for non-negative inputs (round half away from
    # zero). Python's round() uses banker's rounding and would disagree on
    # exact .5 bins.
    return int(np.floor((hz * fft_size / sample_rate) + 0.5))


def find_f0(spectrum, sample_rate, fft_size):
    lo = max(1, _bin_of(F0_MIN_HZ, sample_rate, fft_size))
    hi = min(len(spectrum) - 1, _bin_of(F0_MAX_HZ, sample_rate, fft_size))
    peak = lo + int(np.argmax(spectrum[lo:hi + 1]))
    return peak * sample_rate / fft_size


def _local_median(spectrum, centre, half_width):
    # Matches JS localMedian exactly: sort ascending, take index
    # len >> 1 (integer floor division). This is NOT np.median - for an
    # even-length window np.median averages the two middle values, but the
    # JS picks a single element, and results can differ.
    lo = max(0, centre - half_width)
    hi = min(len(spectrum) - 1, centre + half_width)
    idx = np.arange(lo, hi + 1)
    vals = spectrum[idx[np.abs(idx - centre) > 2]]
    if vals.size == 0:
        return 1e-10
    sorted_vals = np.sort(vals)
    return float(sorted_vals[len(sorted_vals) // 2]) + 1e-10


def _peak_near(spectrum, centre):
    lo = max(0, centre - 2)
    hi = min(len(spectrum) - 1, centre + 2)
    return float(np.max(spectrum[lo:hi + 1]))


def harmonic_stack_score(spectrum, f0, sample_rate, fft_size):
    if not np.isfinite(f0) or f0 < F0_MIN_HZ:
        return 0.0
    half_width = max(8, _bin_of(f0 / 2, sample_rate, fft_size))
    total, counted = 0.0, 0
    for h in range(1, N_HARMONICS + 1):
        centre = _bin_of(f0 * h, sample_rate, fft_size)
        if centre >= len(spectrum) - 3:
            break
        prominence = _peak_near(spectrum, centre) / _local_median(spectrum, centre, half_width)
        total += np.log10(max(prominence, 1.0))
        counted += 1
    if counted == 0:
        return 0.0
    return float(min(1.0, total / counted / 2))


def _prev_pow2(n):
    return 2 ** int(np.floor(np.log2(n)))


def _bandpass(samples, lo_hz, hi_hz, sample_rate):
    n = _prev_pow2(len(samples))
    spec = np.fft.rfft(samples[:n])
    lo = int(np.floor(lo_hz * n / sample_rate))
    hi = int(np.ceil(hi_hz * n / sample_rate))
    mask = np.zeros(len(spec), dtype=bool)
    mask[max(lo, 0):min(hi + 1, len(spec))] = True
    return np.fft.irfft(spec * mask, n=n)


def _clamp01(x):
    return float(min(1.0, max(0.0, x)))


def rotor_beat_score(samples, f0, sample_rate):
    """Mirrors src/features/beat.js rotorBeatScore: score = tonality *
    prominence. tonality is in-band envelope-spectrum flatness measured
    against BEAT_FLATNESS_REF; prominence is peak-over-band-mean (mean of
    magnitudes in the beat band, not the median of the whole spectrum).

    The envelope is linearly detrended (closed-form least squares, not
    scipy.signal.detrend, to keep the arithmetic identical to the JS side)
    rather than just mean-subtracted, so slow loudness drift doesn't leak
    into the lowest beat-band bins, then Hann-windowed before the FFT."""
    if not np.isfinite(f0) or f0 < F0_MIN_HZ:
        return 0.0
    band = max(30.0, f0 * 0.25)
    filtered = _bandpass(np.asarray(samples, dtype=float), f0 - band, f0 + band, sample_rate)
    n = _prev_pow2(len(filtered))
    env = np.abs(filtered[:n])
    mean = env.mean()
    if mean < 1e-12:
        return 0.0

    t = np.arange(n)
    t_mean = (n - 1) / 2
    dt = t - t_mean
    slope = np.sum(dt * (env - mean)) / np.sum(dt * dt)
    env = (env - mean - slope * dt) * np.hanning(n)

    spec = np.abs(np.fft.rfft(env))
    # Detrending is not perfect - residual trend leakage still concentrates
    # in bin 1, so the beat band now starts at bin 2 rather than bin 1.
    lo = max(2, int(np.ceil(BEAT_MIN_HZ * n / sample_rate)))
    hi = min(len(spec) - 1, int(np.ceil(BEAT_MAX_HZ * n / sample_rate)))
    if hi <= lo:
        return 0.0

    band_vals = spec[lo:hi + 1]
    peak = float(np.max(band_vals))
    band_mean = float(np.mean(band_vals)) + 1e-12
    power = band_vals ** 2 + 1e-20
    flatness = float(np.exp(np.mean(np.log(power))) / (np.mean(power) + 1e-20))

    tonality = _clamp01(np.log10(BEAT_FLATNESS_REF / max(flatness, 1e-9)))
    prominence = _clamp01(np.log10(max(peak / band_mean, 1.0)))

    return float(tonality * prominence)


def _top_bin(spectrum, sample_rate, fft_size):
    return min(len(spectrum) - 1, _bin_of(FEATURE_MAX_HZ, sample_rate, fft_size))


def band_ratio(spectrum, sample_rate, fft_size):
    lo = _bin_of(100, sample_rate, fft_size)
    hi = _bin_of(2000, sample_rate, fft_size)
    top = _top_bin(spectrum, sample_rate, fft_size)
    power = spectrum[1:top + 1] ** 2
    total = power.sum()
    if total <= 0:
        return 0.0
    band = (spectrum[max(lo, 1):min(hi, top) + 1] ** 2).sum()
    return float(band / total)


def spectral_flatness(spectrum, sample_rate, fft_size):
    power = spectrum[1:_top_bin(spectrum, sample_rate, fft_size) + 1] ** 2 + 1e-20
    if power.size == 0 or power.sum() == 0:
        return 0.0
    return float(np.exp(np.log(power).mean()) / power.mean())


def extract_features(samples, sample_rate):
    samples = np.asarray(samples, dtype=float)
    spectrum = averaged_spectrum(samples)
    f0 = find_f0(spectrum, sample_rate, HARMONIC_FFT_SIZE)
    mel_mean, mel_std, loudness_std = mel_shape_stats(log_mel_spectrogram(samples, sample_rate))
    features = {
        "f0": float(f0),
        "harmonicStack": harmonic_stack_score(spectrum, f0, sample_rate, HARMONIC_FFT_SIZE),
        "rotorBeat": rotor_beat_score(samples, f0, sample_rate),
        "bandRatio": band_ratio(spectrum, sample_rate, HARMONIC_FFT_SIZE),
        "spectralFlatness": spectral_flatness(spectrum, sample_rate, HARMONIC_FFT_SIZE),
        "loudnessStd": loudness_std,
    }
    for m in range(MEL_BANDS):
        features[f"melMean{m}"] = float(mel_mean[m])
        features[f"melStd{m}"] = float(mel_std[m])
    return features
