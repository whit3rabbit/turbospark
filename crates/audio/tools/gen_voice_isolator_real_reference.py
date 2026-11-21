"""Reference outputs of the released voice-isolator checkpoints for the Rust
real-checkpoint gate.

Loads `denoiser.safetensors` and `denoiser_realtime.safetensors` from MODEL_DIR
through the pinned inworld reference (`build_model` reading the embedded
config), renders one seeded 48 kHz clip with `VoiceIsolatorNetwork.enhance`
in float32 on the CPU, and writes into MODEL_DIR (not the repository; the
files are about 2 MB each):

    real_clip.npy            10 s: speech (or a harmonic tone) plus noise
    real_voice.npy           the offline render (two 8 s windows, crossfaded)
    real_voice_realtime.npy  the realtime file's render (causal, look-ahead 2)

    ../.venv-mlxaudio/bin/python crates/audio/tools/gen_voice_isolator_real_reference.py \
        /path/to/voice-isolator-<commit> ~/models/voice-isolator [speech16k.wav]
"""

import importlib.util
import pathlib
import sys
import types

import numpy as np
import torch

REFERENCE = pathlib.Path(sys.argv[1]).resolve()
MODEL_DIR = pathlib.Path(sys.argv[2]).expanduser().resolve()
sys.path.insert(0, str(REFERENCE / "src"))
for _missing in ("torchaudio", "soundfile"):
    if importlib.util.find_spec(_missing) is None:
        sys.modules[_missing] = types.ModuleType(_missing)

# pylint: disable=wrong-import-position
from inworld.voice_isolation import isolator  # noqa: E402

RATE = 48_000


def read_speech(path: str) -> np.ndarray:
    """A 16-bit mono WAV, linearly upsampled to 48 kHz (any input will do
    for parity; real speech keeps the voice the model is meant to keep)."""
    import wave  # pylint: disable=import-outside-toplevel
    with wave.open(path) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0
    t = np.arange(int(len(x) * RATE / rate)) * rate / RATE
    return np.interp(t, np.arange(len(x)), x)


def clip() -> np.ndarray:
    """10 s: speech (when a WAV is given) or a harmonic tone, plus white
    noise at about 0 dB SNR."""
    rng = np.random.default_rng(20261120)
    n = 10 * RATE
    if len(sys.argv) > 3:
        voice = np.resize(read_speech(sys.argv[3]), n)
    else:
        t = np.arange(n) / RATE
        phase = 2 * np.pi * np.cumsum(140 + 30 * np.sin(2 * np.pi * 0.7 * t)) / RATE
        voice = sum(0.12 / k * np.sin(k * phase) for k in range(1, 12))
    noise = rng.standard_normal(n)
    noise *= np.sqrt((voice**2).mean() / (noise**2).mean())
    return (voice + noise).astype(np.float32)


def main() -> None:
    torch.set_num_threads(8)
    audio = clip()
    np.save(MODEL_DIR / "real_clip.npy", audio)
    for name, out in (("denoiser", "real_voice.npy"),
                      ("denoiser_realtime", "real_voice_realtime.npy")):
        path = MODEL_DIR / f"{name}.safetensors"
        net = isolator.load_model(str(path), "cpu")
        with torch.inference_mode():
            voice = net.enhance(torch.from_numpy(audio)).numpy()
        np.save(MODEL_DIR / out, voice.astype(np.float32))
        print(name, "params", net.num_params, "rms in",
              float(np.sqrt((audio**2).mean())), "out",
              float(np.sqrt((voice**2).mean())))


if __name__ == "__main__":
    main()
