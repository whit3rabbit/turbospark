#!/usr/bin/env python3
"""Regenerate the MegaTTS3 DiT + sampler Rust parity fixtures.

Normative reference: the unmodified MegaTTS3 code
(`tts/modules/llm_dit/*`, `tts/modules/ar_dur/commons/*` from
github.com/bytedance/MegaTTS3, Apache-2.0; the Hugging Face Space
mrfakename/MegaTTS3-Voice-Cloning carries the same files at revision
8c2b5ab2ac67affda57a649fe0637425e8e1de58).

Needs torch + numpy + safetensors; run in a scratch venv, not the repo:

    python gen_megatts3_dit_fixtures.py --ref /path/to/dir/containing/tts \
        --out crates/audio/testdata/megatts3_dit

Outputs into --out:
- tiny_config.json + tiny_weights.safetensors
    A seeded tiny `Diffusion` (a subclass of the reference class that reuses
    its own `inference`, `_forward` and `forward_ling_encoder`). The weights go
    through convert_megatts3.convert_state_dict, the path the real checkpoint
    takes.
- per case `<tag>_*.npy`: phones/tones/mel2ph/prompt, the phone encoder output,
  `x_ling` after the 4x expansion, the first CFG velocity, the final
  latent, and every Gaussian draw `torch.randn` produced (draw 0 is the initial
  latent, then one per step). The Rust test replays these draws.
- manifest.json (sha256 of every file).

Optional, never committed: --real-src DIR --real-out DIR writes golden
tensors for the real checkpoint (`DIR/diffusion_transformer/model_only_last.ckpt`)
on a short sequence, for the env-gated real-weights test.

Deterministic: fixed seeds, no network access.
"""

import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import nn

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import convert_megatts3 as cv  # noqa: E402

TINY = dict(
    dim=32,
    layers=2,
    heads=2,
    channels=8,
    local_cond_dim=24,
    ctx_mask_dim=4,
    time_freq_dim=16,
    phone_vocab=20,
    tone_vocab=8,
    ph_layers=2,
    ph_heads=2,
    ph_filter=64,
    ph_kernel=3,
)
# (tag, phones N, latent frames T, prompt frames P, steps, p_w, t_w)
# B has fewer phones than the relative-attention window + 1, taking the
# short-sequence branch of `_get_relative_embeddings`.
CASES = [("a", 12, 10, 4, 4, 1.4, 3.0), ("b", 3, 6, 2, 3, 2.0, 2.5)]


def ffn_hidden(dim):
    # FeedForward(hidden_dim=2*dim, multiple_of=256) in the reference block.
    return 256 * ((2 * dim + 255) // 256)


def import_reference():
    sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda x, *a, **k: x))
    from tts.modules.ar_dur.commons.layers import Embedding
    from tts.modules.ar_dur.commons.nar_tts_modules import PosEmb
    from tts.modules.ar_dur.commons.rel_transformer import RelTransformerEncoder
    from tts.modules.llm_dit.cfm import ConditionalFlowMatcher
    from tts.modules.llm_dit.dit import Diffusion
    from tts.modules.llm_dit.time_embedding import TimestepEmbedding
    from tts.modules.llm_dit.transformer import Transformer

    class TinyDiffusion(Diffusion):
        def __init__(self, c):
            nn.Module.__init__(self)
            d = c["dim"]
            self.out_channels = self.in_channels = c["channels"]
            self.ctx_mask_proj = nn.Linear(1, c["ctx_mask_dim"])
            self.local_cond_project = nn.Linear(c["channels"] + c["ctx_mask_dim"], c["local_cond_dim"])
            self.encoder = Transformer(c["layers"], d, c["heads"], 256)
            self.x_prenet = nn.Linear(c["channels"], d)
            self.prenet = nn.Linear(c["local_cond_dim"], d)
            self.postnet = nn.Linear(d, c["channels"])
            self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
            self.f5_time_embed = TimestepEmbedding(d, freq_embed_dim=c["time_freq_dim"])
            self.ph_encoder = RelTransformerEncoder(
                c["phone_vocab"], d, d, c["ph_filter"], c["ph_heads"], c["ph_layers"],
                c["ph_kernel"], 0.0, prenet=True, pre_ln=True,
            )
            self.tone_embed = Embedding(c["tone_vocab"], d, padding_idx=0)
            self.ph_pos_embed = PosEmb(d)
            self.ling_pre_net = nn.Sequential(
                *[nn.Conv1d(d, d, kernel_size=s * 2, stride=s, padding=s // 2) for s in (2, 2)]
            )

    return Diffusion, TinyDiffusion


def randomize(model, seed):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("gamma"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
            elif "emb" in name and "rel" not in name:
                p.copy_(0.3 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(0.15 * torch.randn(p.shape, generator=gen))


def save_npy(out, name, t):
    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    # Never save transposed views Fortran-ordered; the Rust reader refuses them.
    np.save(os.path.join(out, name), np.ascontiguousarray(arr, dtype=arr.dtype if arr.dtype.kind == "i" else "<f4"))


def make_case(rng, n, t, p, mask_phone, mask_tone, ch):
    phone = rng.integers(1, mask_phone, size=n)
    tone = rng.integers(1, mask_tone, size=n)
    total = t * 4
    dur = np.ones(n, dtype=np.int64)
    for _ in range(total - n):
        dur[rng.integers(0, n)] += 1
    mel2ph = np.repeat(np.arange(1, n + 1), dur)
    prompt = rng.standard_normal((p, ch)).astype("<f4") * 0.8
    return phone.astype("<i4"), tone.astype("<i4"), mel2ph.astype("<i4"), prompt


def run_case(model, c, case, out, tag, seed):
    _, n, t, p, steps, p_w, t_w = case
    rng = np.random.default_rng(seed)
    mask_phone, mask_tone = c["phone_vocab"] - 1, c["tone_vocab"] - 1
    phone, tone, mel2ph, prompt = make_case(rng, n, t, p, mask_phone, mask_tone, c["channels"])
    ph, tn, m2p, pl = (torch.from_numpy(a.astype("i8" if a.dtype.kind == "i" else "<f4")) for a in (phone, tone, mel2ph, prompt))

    # prepare_inputs_for_dit: three CFG rows, row 0 carries the prompt.
    ph_seq = torch.stack([ph, ph, torch.full_like(ph, mask_phone)])
    tone_seq = torch.stack([tn, tn, torch.full_like(tn, mask_tone)])
    lat = torch.zeros(3, t, c["channels"])
    lat[:, :p] = pl[None]
    ctx_mask = torch.zeros(3, t, 1)
    ctx_mask[:, :p] = 1
    lat[1:] = 0
    inputs = {
        "phone": ph_seq, "tone": tone_seq, "lat_ctx": lat * ctx_mask,
        "ctx_mask": ctx_mask, "dur": m2p[None].repeat(3, 1),
    }

    with torch.no_grad():
        ph_enc = model.forward_ling_encoder(ph_seq[:1], tone_seq[:1])  # [1, N, dim]
        from tts.modules.ar_dur.ar_dur_predictor import expand_states

        x_ling = model.ling_pre_net(expand_states(ph_enc, inputs["dur"][:1]).transpose(1, 2)).transpose(1, 2)

    draws, vels = [], []
    orig_randn, orig_forward = torch.randn, model._forward

    def rec_randn(*a, **k):
        r = orig_randn(*a, **k)
        draws.append(r.clone())
        return r

    def rec_forward(*a, **k):
        r = orig_forward(*a, **k)
        vels.append(r.clone())
        return r

    torch.manual_seed(seed)
    torch.randn, model._forward = rec_randn, rec_forward
    try:
        x = model.inference(inputs, timesteps=steps, seq_cfg_w=[p_w, t_w])
    finally:
        torch.randn, model._forward = orig_randn, orig_forward
    assert len(draws) == steps + 1, (len(draws), steps)

    save_npy(out, f"{tag}_phone.npy", phone)
    save_npy(out, f"{tag}_tone.npy", tone)
    save_npy(out, f"{tag}_mel2ph.npy", mel2ph)
    save_npy(out, f"{tag}_prompt.npy", prompt)
    save_npy(out, f"{tag}_ph_enc.npy", ph_enc[0])
    save_npy(out, f"{tag}_x_ling.npy", x_ling[0])
    save_npy(out, f"{tag}_vel_first.npy", vels[0][0])
    save_npy(out, f"{tag}_x_final.npy", x[0])
    for i, d in enumerate(draws):
        save_npy(out, f"{tag}_draw_{i:02d}.npy", d[0])
    return {"steps": steps, "p_w": p_w, "t_w": t_w, "n": n, "frames": t, "prompt_frames": p,
            "peak": float(x.abs().max())}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-src")
    ap.add_argument("--real-out")
    args = ap.parse_args()
    sys.path.insert(0, args.ref)
    Diffusion, TinyDiffusion = import_reference()
    os.makedirs(args.out, exist_ok=True)

    model = TinyDiffusion(TINY).eval()
    randomize(model, 4321)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    save_file(
        {k: np.ascontiguousarray(v) for k, v in cv.convert_state_dict(sd).items()},
        os.path.join(args.out, "tiny_weights.safetensors"),
    )
    cfg = dict(TINY, ffn_hidden=ffn_hidden(TINY["dim"]))
    with open(os.path.join(args.out, "tiny_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)

    info = {}
    for i, case in enumerate(CASES):
        info[case[0]] = run_case(model, TINY, case, args.out, case[0], seed=100 + i)
    assert all(v["peak"] > 1e-3 and np.isfinite(v["peak"]) for v in info.values()), info
    manifest = {
        "reference": "bytedance/MegaTTS3 tts/modules/llm_dit (Apache-2.0)",
        "torch": torch.__version__,
        "cases": info,
        "files": {f: sha256(os.path.join(args.out, f)) for f in sorted(os.listdir(args.out)) if f != "manifest.json"},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print("wrote", args.out, {k: round(v["peak"], 3) for k, v in info.items()})

    if args.real_src:
        real(args, Diffusion)


def real(args, Diffusion):
    model = Diffusion().eval()
    sd = cv.load_ckpt_state_dict(os.path.join(args.real_src, "diffusion_transformer/model_only_last.ckpt"), "dit")
    res = model.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()}, strict=False)
    print("real load: missing", res.missing_keys, "unexpected", res.unexpected_keys)
    assert not res.missing_keys
    os.makedirs(args.real_out, exist_ok=True)
    c = dict(TINY, dim=1024, channels=32, phone_vocab=302, tone_vocab=32)
    info = run_case(model, c, ("r", 10, 16, 6, 3, 1.4, 3.0), args.real_out, "r", seed=7)
    print("real case", info)


if __name__ == "__main__":
    main()
