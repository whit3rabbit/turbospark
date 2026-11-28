#!/usr/bin/env python3
"""Regenerate the ACE-Step 1.5 Oobleck VAE Rust parity fixtures.

Normative reference: diffusers 0.37.1 `AutoencoderOobleck`, the class
ACE-Step 1.5 loads for its VAE (upstream ACE-Step/ACE-Step-1.5 at
ca1e85fe9430179831e6bc6be790c332190a3866,
acestep/core/generation/handler/init_service_loader_components.py).
PyTorch runs on CPU in float32.

Run with the ACE-Step venv (torch 2.15.0, diffusers 0.37.1 as pinned by
upstream uv.lock; 0.43.0 produced byte-identical fixtures):

    ../.venv-acestep/bin/python -I \
        crates/audio/tools/gen_ace_step15_vae_fixtures.py

Outputs into crates/audio/testdata/ace_step15_vae/:

- tiny_config.json + tiny_weights.safetensors
    A seeded tiny AutoencoderOobleck saved in the diffusers state_dict
    layout (weight_g / weight_v, Snake alpha / beta [1, C, 1]), which is
    the layout the real checkpoint uses.
- {a,b}_{audio,mean,std,noise,sample,decoded}.npy
    Encoder moments, a sample drawn with recorded noise, and the decode
    of that sample, for one hop-aligned and one unaligned input length.
- manifest.json
    File list with SHA-256 digests and generator metadata.

With `--real <checkpoint vae dir> <out dir>` it instead runs the real
VAE on a deterministic 2 s stereo signal and writes the same arrays to
<out dir> (kept outside the repository; the ignored Rust test reads it
through TURBOSPARK_ACE_STEP15_VAE_REF).

Deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderOobleck
from safetensors.torch import save_file

REFERENCE_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
OUT = Path(__file__).resolve().parents[1] / "testdata" / "ace_step15_vae"
TINY = dict(
    encoder_hidden_size=8,
    downsampling_ratios=[2, 4, 6],
    channel_multiples=[1, 2, 2],
    decoder_channels=8,
    decoder_input_channels=4,
    audio_channels=2,
    sampling_rate=4800,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(out: Path, name: str, arr) -> str:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    np.save(out / name, np.ascontiguousarray(arr, dtype=np.float32))
    return name


def run_case(vae, out: Path, tag: str, audio: np.ndarray, seed: int) -> list[str]:
    """Encode `audio` [C, T], sample with recorded noise, decode."""
    files = [save_npy(out, f"{tag}_audio.npy", audio)]
    with torch.inference_mode():
        x = torch.from_numpy(audio)[None]
        posterior = vae.encode(x).latent_dist
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(posterior.mean.shape, generator=gen, dtype=torch.float32)
        sample = posterior.mean + posterior.std * noise
        decoded = vae.decode(sample).sample
    files.append(save_npy(out, f"{tag}_mean.npy", posterior.mean[0]))
    files.append(save_npy(out, f"{tag}_std.npy", posterior.std[0]))
    files.append(save_npy(out, f"{tag}_noise.npy", noise[0]))
    files.append(save_npy(out, f"{tag}_sample.npy", sample[0]))
    files.append(save_npy(out, f"{tag}_decoded.npy", decoded[0]))
    return files


def tiny() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(48)
    vae = AutoencoderOobleck(**TINY).eval()
    gen = torch.Generator().manual_seed(53)
    state = {}
    for name, value in vae.state_dict().items():
        # Seeded weights at a scale that keeps activations O(1) through
        # the stack; Snake log-scales stay near zero (scale near one).
        scale = 0.1 if name.endswith((".alpha", ".beta")) else 0.3
        seeded = torch.randn(value.shape, generator=gen, dtype=torch.float32) * scale
        if name.endswith(".weight_g"):
            seeded = seeded.abs() + 0.5
        state[name] = seeded.contiguous()
    vae.load_state_dict(state)
    save_file(state, str(OUT / "tiny_weights.safetensors"))
    (OUT / "tiny_config.json").write_text(json.dumps(TINY, indent=2) + "\n")

    rng = np.random.default_rng(59)
    files = ["tiny_config.json", "tiny_weights.safetensors"]
    hop = int(np.prod(TINY["downsampling_ratios"]))
    for tag, n, seed in (("a", 4 * hop, 61), ("b", 4 * hop + 17, 67)):
        audio = (rng.standard_normal((2, n)) * 0.5).astype(np.float32)
        files += run_case(vae, OUT, tag, audio, seed)

    manifest = {
        "generator": "tools/gen_ace_step15_vae_fixtures.py",
        "reference": "diffusers AutoencoderOobleck",
        "diffusers_version": __import__("diffusers").__version__,
        "torch_version": torch.__version__,
        "ace_step_commit": REFERENCE_COMMIT,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(files) + 1} fixture files to {OUT}")


def real(vae_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderOobleck.from_pretrained(str(vae_dir), torch_dtype=torch.float32).eval()
    sr = int(vae.config.sampling_rate)
    t = np.arange(2 * sr, dtype=np.float64) / sr
    # A chord with a slow amplitude envelope and a little noise: enough
    # structure that a broken layer shows up in the decoded waveform.
    rng = np.random.default_rng(71)
    env = 0.5 * (1.0 - np.cos(2 * np.pi * t / 2.0))
    left = env * (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 277.18 * t))
    right = env * (0.3 * np.sin(2 * np.pi * 329.63 * t) + 0.2 * np.sin(2 * np.pi * 440 * t))
    audio = np.stack([left, right]) + 0.01 * rng.standard_normal((2, t.size))
    files = run_case(vae, out, "real", audio.astype(np.float32), 73)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "generator": "tools/gen_ace_step15_vae_fixtures.py --real",
                "vae_dir": str(vae_dir),
                "diffusers_version": __import__("diffusers").__version__,
                "torch_version": torch.__version__,
                "files": {name: digest(out / name) for name in files},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(files) + 1} reference files to {out}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--real":
        real(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) == 1:
        tiny()
    else:
        sys.exit("usage: gen_ace_step15_vae_fixtures.py [--real <vae dir> <out dir>]")
