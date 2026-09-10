"""Mixes a signal under background noise at a controlled SNR.
This is what produces the accuracy-vs-SNR curve - the honest 'here is where
it stops working' evidence."""
import numpy as np


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) or 1e-12


def mix_at_snr(signal, noise, snr_db):
    """Scale noise so signal sits snr_db above it, then sum. Lengths are
    matched by tiling or truncating the noise."""
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if noise.size < signal.size:
        noise = np.tile(noise, int(np.ceil(signal.size / noise.size)))
    noise = noise[:signal.size]

    target_noise_rms = rms(signal) / (10 ** (snr_db / 20))
    scaled = noise * (target_noise_rms / rms(noise))
    mixed = signal + scaled

    peak = np.max(np.abs(mixed))
    return mixed / peak * 0.99 if peak > 1.0 else mixed


def _self_check():
    rng = np.random.default_rng(3)
    t = np.arange(48000) / 48000
    sig = np.sin(2 * np.pi * 200 * t)
    noise = rng.standard_normal(48000)
    for target in (20, 10, 0, -10):
        mixed = mix_at_snr(sig, noise, target)
        # The >1.0 peak limiter applies a uniform gain to the whole mix
        # whenever the sine's own peak plus a noise excursion tips over 1.0 -
        # which happens on nearly every draw at every target SNR here.  That
        # gain does not change the signal/noise power ratio, but comparing
        # directly against the unscaled `sig` (mixed - sig) misattributes the
        # gain change to noise and reports a bogus low SNR. Recover the
        # actual signal component's scale by least-squares projection onto
        # the clean reference before measuring the residual.
        alpha = np.dot(mixed, sig) / np.dot(sig, sig)
        residual = mixed - alpha * sig
        measured = 20 * np.log10(rms(alpha * sig) / rms(residual))
        assert abs(measured - target) < 0.5, f"{target} dB requested, {measured:.2f} measured"
    print("mix_at_snr self-check passed")


if __name__ == "__main__":
    _self_check()
