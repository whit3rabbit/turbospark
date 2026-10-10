#!/usr/bin/env python3
"""Regenerate the WavVAE (MegaTTS3) Rust parity fixtures.

Normative reference: the unmodified MegaTTS3 model code
(`tts/modules/wavvae/{decoder,encoder}` from github.com/bytedance/MegaTTS3,
Apache-2.0; the Hugging Face Space mrfakename/MegaTTS3-Voice-Cloning carries the
same files at revision 8c2b5ab2ac67affda57a649fe0637425e8e1de58).

Needs torch + numpy + safetensors. Run in a scratch venv, not the repo:

    python gen_wavvae_fixtures.py --ref /path/to/dir/containing/tts \
        --out crates/audio/testdata/wavvae

Outputs into --out:
- tiny_config.json + tiny_weights.safetensors
    Seeded tiny WavVAE built from the reference classes. The weights go
    through convert_megatts3.convert_state_dict, the same path the real
    checkpoints take, so the fixture also covers the MLX layout swap.
- *.npy traces and manifest.json (sha256 of every file).

Optional, never committed: --real-src DIR --real-out DIR also writes golden
tensors for a short clip using the real converted-from checkpoint
(`DIR/wavvae/model_only_last.ckpt`), for the env-gated real-weights test.

The script is deterministic: fixed seeds, no network access.
"""

import argparse
import hashlib
import json
import os
import sys
import types
from argparse import Namespace

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import nn

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import convert_megatts3 as cv  # noqa: E402

TINY = dict(
    enc_filters=4,
    enc_ratios=[2, 2, 3],  # reference order; the encoder reverses it internally
    enc_dim=16,
    lstm_layers=2,
    z_dim=6,
    dec_in=12,  # proj_to_decoder width == latent_upsampler input
    latent_up=4,
    ngf=4,
    dec_ratios=[2, 3, 2],
    sample_rate=24000,
)


def stub_unneeded_imports():
    # hifigan_modules imports librosa only for Audio2Mel; conv.py imports
    # einops only for ConvLayerNorm. Neither is on the encode/decode path.
    librosa = types.ModuleType("librosa")
    filters = types.ModuleType("librosa.filters")
    filters.mel = lambda *a, **k: None
    librosa.filters = filters
    sys.modules.setdefault("librosa", librosa)
    sys.modules.setdefault("librosa.filters", filters)
    sys.modules.setdefault("einops", types.ModuleType("einops"))


def build_tiny(cfg):
    from tts.modules.wavvae.decoder.hifigan_modules import Generator, Upsample
    from tts.modules.wavvae.encoder.common_modules.seanet import SEANetEncoder

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.encoder = SEANetEncoder(
                causal=False, n_residual_layers=1, norm="weight_norm", pad_mode="reflect",
                lstm=cfg["lstm_layers"], dimension=cfg["enc_dim"], channels=1,
                n_filters=cfg["enc_filters"], ratios=cfg["enc_ratios"], activation="ELU",
                kernel_size=7, residual_kernel_size=3, last_kernel_size=7, dilation_base=2,
                true_skip=False, compress=2,
            )
            self.proj_to_z = nn.Linear(cfg["enc_dim"], 2 * cfg["z_dim"])
            self.proj_to_decoder = nn.Linear(cfg["z_dim"], cfg["dec_in"])
            self.latent_upsampler = Upsample(cfg["dec_in"], cfg["latent_up"])
            self.decoder = Generator(
                input_size_=cfg["dec_in"] // 2, ngf=cfg["ngf"], n_residual_layers=4, num_band=1,
                args=Namespace(frame_shift=240, use_tanh=True), ratios=cfg["dec_ratios"],
            )

    return Tiny()


def randomize(model, seed):
    # Default inits leave the final conv zeroed (weights_zero_init) and weight_g
    # at ||v||, which makes outputs trivial; draw everything from a seeded RNG.
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("weight_g"):
                p.copy_(0.4 + 0.4 * torch.rand(p.shape, generator=gen))
            elif name.endswith("weight_v"):
                p.copy_(torch.randn(p.shape, generator=gen))
            elif "lstm" in name:
                p.copy_(0.25 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(0.2 * torch.randn(p.shape, generator=gen))


def encode(model, wav, eps):
    """Mirror WavVAE_V3.encode / encode_latent, with the draw supplied by the caller."""
    from tts.modules.wavvae.decoder.diag_gaussian import DiagonalGaussianDistribution

    enc_out = model.encoder.encoder(wav.unsqueeze(1))  # [1, dim, T]
    x = model.proj_to_z(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
    post = DiagonalGaussianDistribution(x)
    latent = (post.mean + post.std * eps).permute(0, 2, 1)  # [1, T, z]
    return enc_out, post, latent


def decode(model, latent):
    x = model.proj_to_decoder(latent).permute(0, 2, 1)
    up = model.latent_upsampler(x)
    return up, model.decoder(up)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def save_npy(out, name, t):
    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    # Transposed views would otherwise be saved Fortran-ordered, which the Rust
    # reader (rightly) refuses.
    np.save(os.path.join(out, name), np.ascontiguousarray(arr, dtype="<f4"))


def run_clip(model, wav, z_dim, seed, out, tag):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        # eps shape needs T first, which comes from the encoder.
        enc_out, post, _ = encode(model, wav, torch.zeros(1))
        eps = torch.randn(post.mean.shape, generator=g)
        enc_out, post, latent = encode(model, wav, eps)
        up, audio = decode(model, latent)
        _, audio_mean = decode(model, post.mean.permute(0, 2, 1))
    save_npy(out, f"{tag}_wav.npy", wav[0])
    save_npy(out, f"{tag}_enc_out.npy", enc_out[0])  # [dim, T]
    save_npy(out, f"{tag}_mean.npy", post.mean[0].T)  # [T, z]
    save_npy(out, f"{tag}_logvar.npy", post.logvar[0].T)
    save_npy(out, f"{tag}_eps.npy", eps[0].T)
    save_npy(out, f"{tag}_latent.npy", latent[0])
    save_npy(out, f"{tag}_up.npy", up[0])  # [dec_in/2, 4T]
    save_npy(out, f"{tag}_audio.npy", audio[0, 0])
    save_npy(out, f"{tag}_audio_mean.npy", audio_mean[0, 0])
    return float(audio.abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-src")
    ap.add_argument("--real-out")
    args = ap.parse_args()
    sys.path.insert(0, args.ref)
    stub_unneeded_imports()
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    model = build_tiny(TINY).eval()
    randomize(model, seed=1234)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    converted = cv.convert_state_dict(sd)
    save_file({k: np.ascontiguousarray(v) for k, v in converted.items()},
              os.path.join(args.out, "tiny_weights.safetensors"))
    with open(os.path.join(args.out, "tiny_config.json"), "w") as fh:
        json.dump(TINY, fh, indent=2, sort_keys=True)

    hop = int(np.prod(TINY["enc_ratios"]))
    rng = np.random.default_rng(7)
    peaks = {}
    # 100 and 37 samples are not multiples of the hop (extra-padding path);
    # 37 is also shorter than the reflect pad of the widest conv.
    for tag, n in (("a", 100), ("b", 37)):
        wav = torch.from_numpy(rng.standard_normal((1, n)).astype("<f4") * 0.3)
        peaks[tag] = run_clip(model, wav, TINY["z_dim"], 99 + n, args.out, tag)
    assert all(p > 1e-3 for p in peaks.values()), f"degenerate fixture {peaks}"

    manifest = {
        "reference": "bytedance/MegaTTS3 tts/modules/wavvae (Apache-2.0)",
        "torch": torch.__version__,
        "hop": hop,
        "peaks": peaks,
        "files": {f: sha256(os.path.join(args.out, f)) for f in sorted(os.listdir(args.out))
                  if f != "manifest.json"},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print("wrote", args.out, peaks)

    if args.real_src:
        real(args, TINY)


def real(args, _tiny):
    from tts.modules.wavvae.decoder.wavvae_v3 import WavVAE_V3

    sd = cv.load_ckpt_state_dict(os.path.join(args.real_src, "wavvae/model_only_last.ckpt"), "model_gen")
    model = WavVAE_V3(hparams={"melgan_config": {"frame_shift": 240, "use_tanh": True}})
    missing = model.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()}, strict=True)
    print("real load:", missing)
    model.eval()
    os.makedirs(args.real_out, exist_ok=True)
    rng = np.random.default_rng(11)
    n = 24000 + 123  # ~1 s, not a multiple of the 960 hop
    t = np.arange(n) / 24000.0
    wav = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(n)).astype("<f4")[None]
    peak = run_clip(model, torch.from_numpy(wav), 32, 5, args.real_out, "r")
    print("real clip peak", peak)


if __name__ == "__main__":
    main()
