"""Reference outputs of a published Mel-Band RoFormer checkpoint for the Rust
real-checkpoint gate.

Loads MODEL_DIR (config.json plus model.safetensors, for example
mlx-community/mel-roformer-zfturbo-vocals-v1-mlx) through the pinned
mlx-audio `MelRoFormer.from_pretrained` (which applies `sanitize`), casts the
weights to float32, and runs on the MLX CPU device. The clip is a 44.1 kHz
stereo float or 16-bit WAV. Writes into MODEL_DIR (not the repository):

    real_clip.npy          the clip, planar [2, samples]
    real_chunk_vocals.npy  `model(clip[:, :chunk_size])`, one reference pass
    real_vocals.npy        the whole clip through the ZFTurbo MSS-Training
                           `demix` overlap-add below (the reference ships no
                           chunking helper; this is the training repo's)

Run from the pinned checkout so `mlx_audio` imports from it:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_mel_roformer_real_reference.py \
        ~/models/mel-roformer-zfturbo-vocals-v1-mlx clip.wav
"""

import pathlib
import sys
import wave

import mlx.core as mx
import numpy as np

from mlx_audio.sts.models.mel_roformer import MelRoFormer

MODEL_DIR = pathlib.Path(sys.argv[1]).expanduser().resolve()
CLIP = pathlib.Path(sys.argv[2]).expanduser().resolve()


def read_wav(path: pathlib.Path) -> np.ndarray:
    """Planar [2, samples] float32 from a 44.1 kHz stereo WAV (16-bit PCM
    or 32-bit float, the WAVE_FORMAT_EXTENSIBLE header included)."""
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", "not a WAV file"
    pos, fmt, body = 12, None, None
    while pos + 8 <= len(data):
        tag, size = data[pos : pos + 4], int.from_bytes(data[pos + 4 : pos + 8], "little")
        chunk = data[pos + 8 : pos + 8 + size]
        if tag == b"fmt ":
            fmt = chunk
        elif tag == b"data":
            body = chunk
        pos += 8 + size + (size & 1)
    assert fmt is not None and body is not None, "missing fmt or data chunk"
    channels = int.from_bytes(fmt[2:4], "little")
    rate = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")
    assert (channels, rate) == (2, 44100), f"expected 44.1 kHz stereo, got {channels} ch {rate} Hz"
    if bits == 32:
        samples = np.frombuffer(body, dtype="<f4")
    elif bits == 16:
        samples = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise SystemExit(f"unsupported bit depth {bits}")
    return np.ascontiguousarray(samples.reshape(-1, 2).T.astype(np.float32))


def window_array(size: int, fade: int) -> np.ndarray:
    window = np.ones(size, dtype=np.float32)
    window[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
    window[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
    return window


def demix(model: MelRoFormer, mix: np.ndarray) -> np.ndarray:
    """ZFTurbo `demix` (generic path) with batch size 1."""
    config = model.config
    chunk = config.chunk_size
    fade = chunk // 10
    step = chunk // config.num_overlap
    border = chunk - step
    length = mix.shape[-1]
    padded = length > 2 * border and border > 0
    if padded:
        mix = np.pad(mix, ((0, 0), (border, border)), mode="reflect")
    base = window_array(chunk, fade)
    result = np.zeros_like(mix)
    counter = np.zeros_like(mix)
    i = 0
    while i < mix.shape[1]:
        part = mix[:, i : i + chunk]
        seg = part.shape[-1]
        mode = "reflect" if seg > chunk // 2 else "constant"
        part = np.pad(part, ((0, 0), (0, chunk - seg)), mode=mode)
        out = np.array(model(mx.array(part)[None])[0])
        i += step
        window = base.copy()
        if i - step == 0:
            window[:fade] = 1
        elif i >= mix.shape[1]:
            window[-fade:] = 1
        start = i - step
        result[..., start : start + seg] += out[..., :seg] * window[:seg]
        counter[..., start : start + seg] += window[:seg]
    estimated = np.nan_to_num(result / counter, nan=0.0)
    if padded:
        estimated = estimated[..., border:-border]
    return estimated.astype(np.float32)


def main() -> None:
    mx.set_default_device(mx.cpu)
    model = MelRoFormer.from_pretrained(str(MODEL_DIR))
    model.set_dtype(mx.float32)
    model.eval()
    clip = read_wav(CLIP)
    chunk = clip[:, : model.config.chunk_size]
    chunk_vocals = np.array(model(mx.array(chunk)[None])[0]).astype(np.float32)
    vocals = demix(model, clip)
    np.save(MODEL_DIR / "real_clip.npy", clip)
    np.save(MODEL_DIR / "real_chunk_vocals.npy", chunk_vocals)
    np.save(MODEL_DIR / "real_vocals.npy", vocals)
    print("clip", clip.shape, "chunk", chunk_vocals.shape, "vocals", vocals.shape)
    print("vocals rms", float(np.sqrt(np.mean(vocals**2))), "clip rms", float(np.sqrt(np.mean(clip**2))))


if __name__ == "__main__":
    main()
