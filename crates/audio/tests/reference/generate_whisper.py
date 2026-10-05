#!/usr/bin/env python3
"""Print independent Whisper fixture values for the Rust unit tests.

Run with NumPy installed:
    python3 crates/audio/tests/reference/generate_whisper.py
Or use a local official asset (its checksum is still verified):
    python3 crates/audio/tests/reference/generate_whisper.py --asset mel_filters.npz

Only the optional fixture-generation tool uses Python/NumPy. Cargo tests
consume literals and do not download files or add runtime dependencies.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np


ASSET_URL = (
    "https://raw.githubusercontent.com/openai/whisper/v20250625/"
    "whisper/assets/mel_filters.npz"
)
ASSET_SHA256 = "7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c"
N_FFT = 400
HOP = 160


def synthetic_second():
    # The Rust input performs each operation in f32 before sin/cos.
    t = np.arange(16000, dtype=np.float32)
    return (
        np.float32(0.5) * np.sin(t * np.float32(0.037))
        + np.float32(0.25) * np.cos(t * np.float32(0.11))
        + np.float32(0.1) * np.sin(t * np.float32(0.0017))
    )


def power_mel(samples, filters):
    # OpenAI's audio.py uses torch.stft(center=True, pad_mode="reflect")
    # with a periodic Hann window. NumPy supplies the independent FFT;
    # the official asset supplies weights, with no local mel construction.
    padded = np.pad(samples, N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(
        np.float32
    )
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    return power @ filters.T


def normalize(mel):
    # The reference slices away the last STFT frame before this reduction.
    logged = np.log10(np.maximum(mel, np.float32(1e-10)))
    return (
        np.maximum(logged, logged.max() - np.float32(8)) + np.float32(4)
    ) / np.float32(4)


def selected(frames, points):
    return [[frame, band, float(frames[frame, band])] for frame, band in points]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, help="local pinned mel_filters.npz")
    args = parser.parse_args()
    if args.asset:
        raw = args.asset.read_bytes()
    else:
        with urlopen(ASSET_URL, timeout=30) as response:
            raw = response.read()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != ASSET_SHA256:
        raise ValueError(f"asset SHA256 mismatch: {checksum}")

    report = {
        "asset_url": ASSET_URL,
        "asset_sha256": checksum,
        "numpy_version": np.__version__,
        "dtypes": {
            "pcm_window_power_filterbank_projection_log": "float32",
            "rfft": str(np.fft.rfft(np.zeros(N_FFT, dtype=np.float32)).dtype),
        },
        "n_fft": N_FFT,
        "hop": HOP,
        "sample_rate": 16000,
        "window": "periodic Hann, float32",
        "padding": "centered reflect",
        "fixtures": {},
    }
    with np.load(io.BytesIO(raw), allow_pickle=False) as asset:
        for n_mels in (80, 128):
            filters = asset[f"mel_{n_mels}"]
            assert filters.shape == (n_mels, N_FFT // 2 + 1)
            # Pick each selected band's nonzero maximum directly from the
            # official matrix, including its logarithmic high-frequency end.
            coefficients = [
                (band, int(np.argmax(filters[band])))
                for band in (0, n_mels // 2, n_mels - 1)
            ]
            one_second = normalize(power_mel(synthetic_second(), filters)[:-1])
            tail = np.zeros(3200, dtype=np.float32)
            tail[-1] = np.float32(16)
            tail_mel = power_mel(tail, filters)
            retained = normalize(tail_mel[:-1])
            incorrectly_normalized = normalize(tail_mel)[:-1]
            points = [(frame, band) for frame in (0, 50) for band in range(8)]
            points += [(0, n_mels // 2), (50, n_mels // 2), (0, n_mels - 1)]
            tail_points = [(0, 0), (0, n_mels - 1), (19, 0), (19, n_mels - 1)]
            report["fixtures"][str(n_mels)] = {
                "coefficients": [
                    [band, bin_index, float(filters[band, bin_index])]
                    for band, bin_index in coefficients
                ],
                "one_second_frames": len(one_second),
                "one_second": selected(one_second, points),
                "tail_input": "3200 zero samples, final sample 16.0",
                "tail_frames": len(retained),
                "tail": selected(retained, tail_points),
                "tail_wrong_order": selected(incorrectly_normalized, tail_points),
                "tail_retained_power_peak": float(tail_mel[:-1].max()),
                "tail_discarded_power_peak": float(tail_mel[-1].max()),
                "tail_order_max_error": float(
                    np.max(np.abs(retained - incorrectly_normalized))
                ),
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
