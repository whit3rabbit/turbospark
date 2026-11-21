"""Seeded tiny voice-isolator fixtures for the Rust port.

Builds small instances of the inworld reference network (normative source:
https://github.com/inworld-ai/voice-isolator at
c73b3fab81a870410cbac50d2f05f25edc4d10aa, `src/inworld/voice_isolation`),
randomizes every parameter (the reference zero-initializes the decoder heads
and FiLM conditions, which would hide their paths), and records per-stage
traces into `testdata/voice_isolator/<case>/`:

    model.safetensors   weights, with the architecture JSON in the header
                        metadata under "config" as the released files carry it
    wave.npy            input audio
    stft_re/im.npy      STFT [bins, frames]
    encode.npy          band tokens [bands, frames, width] after the encoders
    block_<i>.npy       tokens after block i
    raw.npy             decoder output [bins, frames, 4] (after read-ahead)
    forward.npy         one forward pass over the whole clip
    enhance.npy         the chunked render (`enhance`) over `long.npy`
    forward_la<n>.npy   one forward at each trained look-ahead (causal only)
    stream_la<n>.npy    the reference plain stream over the clip, pushed in
                        ragged pieces, at each look-ahead (causal only)
    traces.json         shapes and the case description
plus `manifest.json` (SHA-256 of every file) at the fixture root.

Run with a Python that has torch and safetensors, pointing at the reference:

    ../.venv-mlxaudio/bin/python crates/audio/tools/gen_voice_isolator_fixtures.py \
        /path/to/voice-isolator-c73b3fab81a870410cbac50d2f05f25edc4d10aa
"""

import hashlib
import json
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import save_file

REFERENCE = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REFERENCE / "src"))

# The package __init__ imports its file loaders (torchaudio, soundfile),
# which the network never touches; stand-ins let it import without them.
import importlib.util  # noqa: E402
import types  # noqa: E402

for _missing in ("torchaudio", "soundfile"):
    if importlib.util.find_spec(_missing) is None:
        sys.modules[_missing] = types.ModuleType(_missing)

# pylint: disable=wrong-import-position
from inworld.voice_isolation import model as modeling  # noqa: E402
from inworld.voice_isolation import streaming  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "testdata" / "voice_isolator"

BASE = dict(
    width=32,
    num_heads=2,
    n_fft=256,
    hop_length=64,
    sample_rate=16000,
    band_plan=[[500, 2000], [1000, 8000]],
    correction_bound=0.5,
    chunk_seconds=0.5,
    overlap_seconds=0.125,
    exclusive_self_attention=True,
    exclusive_axes="time",
    mamba_state=8,
    mamba_conv=4,
    mamba_expand=2,
    block_types=["transformer", "mamba", "transformer", "mamba"],
)

CASES = {
    # The released offline file's shape: two-sided attention and
    # bidirectional Mamba along time.
    "offline": dict(BASE, causal=False),
    # The released realtime file's shape: causal, trained look-ahead
    # positions with a learned embedding, a bounded attention window.
    "causal": dict(BASE,
                   causal=True,
                   lookahead_frames=2,
                   lookahead_frames_choices=[0, 2, 4],
                   attention_window_frames=8,
                   chunk_seconds=2.0),
}


def randomize(net: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith("A_log"):
                base = torch.log(torch.arange(1, p.shape[1] + 1, dtype=torch.float32))
                p.copy_(base.repeat(p.shape[0], 1) +
                        0.1 * torch.randn(p.shape, generator=g))
            elif "norm" in name and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                scale = 0.3 / max(1, p.shape[-1])**0.5 if p.ndim > 1 else 0.05
                p.copy_(torch.randn(p.shape, generator=g) * scale * (
                    3.0 if p.ndim > 1 else 1.0))


def save(path: pathlib.Path, array) -> None:
    np.save(path, np.ascontiguousarray(np.asarray(array, dtype=np.float32)))


def build(case: str, cfg: dict, seed: int) -> None:
    config = modeling.VoiceIsolatorConfig.from_dict(cfg)
    net = modeling.VoiceIsolatorNetwork(config).eval()
    randomize(net, seed)
    out = OUT / case
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in net.state_dict().items()}
    save_file(state,
              str(out / "model.safetensors"),
              metadata={
                  "config": json.dumps(cfg),
                  "format": "pt"
              })

    rng = np.random.default_rng(seed)
    sr = config.sample_rate
    t = np.arange(int(0.3 * sr)) / sr
    wave = (0.3 * np.sin(2 * np.pi * 440 * t) +
            0.2 * rng.standard_normal(t.shape)).astype(np.float32)
    long = (0.3 * np.sin(2 * np.pi * 330 * np.arange(int(1.3 * sr)) / sr) +
            0.2 * rng.standard_normal(int(1.3 * sr))).astype(np.float32)
    save(out / "wave.npy", wave)
    save(out / "long.npy", long)

    traces = {"case": case, "config": cfg, "seed": seed}
    with torch.inference_mode():
        audio = torch.from_numpy(wave)[None]
        spectrum = net.stft(audio)
        save(out / "stft_re.npy", spectrum.real[0])
        save(out / "stft_im.npy", spectrum.imag[0])
        causal = not net.offline
        features = net.encode(spectrum)
        save(out / "encode.npy", features[0])
        for index, block in enumerate(net.blocks):
            features = block(features, causal)
            save(out / f"block_{index}.npy", features[0])
        raw = net.denoise_decoder(features)
        if causal and net.lookahead_frames:
            raw = modeling.read_ahead(raw, net.lookahead_frames)
        save(out / "raw.npy", raw[0])
        save(out / "forward.npy", net(audio)[0])
        save(out / "enhance.npy", net.enhance(torch.from_numpy(long)))
        traces["bands"] = len(net.edges)
        traces["edges"] = net.edges
        traces["frames"] = int(spectrum.shape[-1])
        traces["lookahead_frames"] = net.lookahead_frames
        if case == "causal":
            # Each trained position over the same short clip, one forward,
            # and the reference plain stream (block 4, the cache at the
            # 8-frame attention window, so it compacts several times) fed
            # ragged pieces.
            for frames in config.lookahead_frames_choices:
                net.set_lookahead(frames)
                save(out / f"forward_la{frames}.npy", net(audio)[0])
                session = streaming.Stream(net, block_frames=4)
                pieces, at, sizes = [], 0, [37, 200, 513, 64]
                while at < len(wave):
                    size = sizes[len(pieces) % len(sizes)]
                    pieces.append(session.push(torch.from_numpy(wave[at:at + size])))
                    at += size
                pieces.append(session.flush())
                save(out / f"stream_la{frames}.npy", torch.cat(pieces))
            net.set_lookahead(config.lookahead_frames)
            traces["stream_pieces"] = [37, 200, 513, 64]
    (out / "traces.json").write_text(json.dumps(traces, indent=1) + "\n")


def manifest() -> None:
    entries = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            entries[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(
        json.dumps({
            "reference": "inworld-ai/voice-isolator@c73b3fab81a870410cbac50d2f05f25edc4d10aa",
            "torch": torch.__version__,
            "files": entries
        },
                   indent=1) + "\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    for i, (name, cfg) in enumerate(CASES.items()):
        build(name, cfg, 1234 + i)
    manifest()
    print("wrote", OUT)
