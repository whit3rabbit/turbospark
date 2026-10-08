#!/usr/bin/env python3
"""Cheap objective distance between two 16-bit WAVs: waveform correlation and
a log-spaced band log-magnitude L1 in dB (a stand-in for mel-L1, no librosa).

    python3 -I spectral_distance.py REFERENCE.wav CANDIDATE.wav

It catches broken or drifted output; it is not a song-quality judgment.
"""
import sys, wave
import numpy as np


def load(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "16-bit PCM only"
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768
        return data.reshape(-1, ch), w.getframerate()


def bands(x, rate, n_fft=2048, hop=512, n_bands=80):
    win = np.hanning(n_fft)
    frames = [x[i:i + n_fft] * win for i in range(0, len(x) - n_fft, hop)]
    mag = np.abs(np.fft.rfft(np.array(frames), axis=1))
    edges = np.unique(np.geomspace(20, rate / 2, n_bands + 1).astype(float))
    freqs = np.fft.rfftfreq(n_fft, 1 / rate)
    idx = np.clip(np.searchsorted(freqs, edges), 0, mag.shape[1] - 1)
    pooled = np.stack([mag[:, a:max(b, a + 1)].mean(axis=1) for a, b in zip(idx[:-1], idx[1:])], 1)
    return 20 * np.log10(pooled + 1e-6)


def main(ref, cand):
    a, ra = load(ref)
    b, rb = load(cand)
    assert ra == rb
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    la, lb = bands(a.mean(1), ra), bands(b.mean(1), rb)
    print({"samples": n, "correlation": round(corr, 4),
           "log_band_l1_db": round(float(np.abs(la - lb).mean()), 3)})


if __name__ == "__main__":
    main(*sys.argv[1:3])
