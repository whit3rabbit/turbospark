#!/usr/bin/env python3
"""Regenerate the MegaTTS3 duration-LM Rust parity fixtures.

Normative reference: the unmodified MegaTTS3 code
(`tts/modules/ar_dur/*`, `tts/frontend_function.py` from
github.com/bytedance/MegaTTS3, Apache-2.0; the Hugging Face Space
mrfakename/MegaTTS3-Voice-Cloning carries the same files at revision
8c2b5ab2ac67affda57a649fe0637425e8e1de58).

The cases call the reference's own `make_dur_prompt` and `dur_pred` with a
stub `self`, so the whole chain (teacher-forced prompt priming, per-chunk
decoding from the prompt state, jitter, silence floors, padding, length
regulation) is what the Rust port is compared against.

Needs torch + numpy + safetensors; run in a scratch venv, not the repo:

    python gen_megatts3_duration_fixtures.py --ref /path/to/dir/containing/tts \
        --out crates/audio/testdata/megatts3_duration

Outputs into --out:
- tiny_config.json + tiny_weights.safetensors: a seeded tiny `ARDurPredictor`
  (including the unused `char_encoder`, like the real checkpoint), converted
  through convert_megatts3.convert_state_dict.
- `<tag>_*.npy` per chunk case plus `prompt_*.npy`: the prompt alignment, the
  duration tokens, each chunk's raw codes and unrounded regression outputs,
  the two `torch.rand_like` draws, and the final `mel2ph`.
- manifest.json (sha256 of every file).

Optional, never committed: --real-src DIR --real-out DIR writes goldens for the
real checkpoint (`DIR/duration_lm/model_only_last.ckpt`).

Deterministic: fixed seeds, no network access.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import types

import numpy as np
import torch
from safetensors.numpy import save_file

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import convert_megatts3 as cv  # noqa: E402

TINY = dict(
    hidden=32,
    layers=2,
    heads=8,  # fixed upstream: RotTransformerDecoderLayer's default
    code_size=20,
    phone_vocab=170,  # must exceed the pause phone ids 145..166
    tone_vocab=10,
    enc_filter=64,
    enc_heads=2,
    enc_layers=2,
    enc_kernel=3,
    char_dict_size=50,
)
# (tag, phones M, is_first, is_final, dur_disturb, dur_alpha)
CHUNKS = [("c0", 7, True, False, 0.1, 1.0), ("c1", 11, False, True, 0.1, 1.0), ("c2", 5, True, True, 0.2, 1.2)]


def stub_imports():
    for name in ("whisper", "librosa", "einops"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda x, *a, **k: x))


def make_model(c, hp_extra=None):
    from tts.modules.ar_dur.ar_dur_predictor import ARDurPredictor

    hp = dict(
        char_dict_size=c["char_dict_size"], lm_use_enc=True, encoder_type="rel_fft",
        ffn_hidden_size=c["enc_filter"], num_heads=c["enc_heads"], enc_layers=c["enc_layers"],
        enc_ffn_kernel_size=c["enc_kernel"], dropout=0.0, enc_prenet=True, enc_pre_ln=True,
        use_ph_pos_embed=True, use_bert_input=False, ling_labels=["tone"],
        ling_label_dict_size=[c["tone_vocab"] - 3], use_spk_embed=False, use_spk_id=False,
        use_spk_enc=False, use_pos_embed=False, use_post_ln=False, dur_model_type="ar_mse",
        max_tokens=6000,
    )
    hp.update(hp_extra or {})
    return ARDurPredictor(hp, c["hidden"], c["hidden"], c["layers"], c["phone_vocab"], c["code_size"],
                          use_rot_embed=True).eval()


def randomize(model, seed):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name == "project_out_dim.0.bias":
                p.fill_(4.0)  # durations around a handful of frames, well inside the code range
            elif name.endswith("gamma") or (".layer_norm" in name and name.endswith("weight")) or name == "layer_norm.weight":
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
            elif "emb" in name and "rel" not in name:
                p.copy_(0.4 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(0.15 * torch.randn(p.shape, generator=gen))


def save_npy(out, name, arr):
    arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    dt = "<i4" if arr.dtype.kind == "i" else "<f4"
    np.save(os.path.join(out, name), np.ascontiguousarray(arr, dtype=dt))


def make_prompt(rng, n, c, specials):
    ph = rng.integers(1, c["phone_vocab"], size=n)
    tone = rng.integers(1, c["tone_vocab"], size=n)
    dur = rng.integers(2, 10, size=n)
    return ph, tone, np.repeat(np.arange(1, n + 1), dur)


def run(model, c, out, seed, n_prompt, chunks, real=False):
    from tts.frontend_function import dur_pred, make_dur_prompt
    from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator

    rng = np.random.default_rng(seed)
    me = types.SimpleNamespace(
        dur_model=model, hp_dur_model={"dur_code_size": c["code_size"]}, fm=8,
        length_regulator=LengthRegulator(), device="cpu", precision=torch.float32,
    )
    ph, tone, mel2ph = make_prompt(rng, n_prompt, c, None)
    ph_t, tone_t, m2p_t = (torch.from_numpy(a.astype("i8"))[None] for a in (ph, tone, mel2ph))
    with torch.no_grad():
        state, ctx = make_dur_prompt(me, m2p_t, ph_t, tone_t)
    save_npy(out, "prompt_phone.npy", ph)
    save_npy(out, "prompt_tone.npy", tone)
    save_npy(out, "prompt_mel2ph.npy", mel2ph)
    save_npy(out, "prompt_tokens.npy", ctx[0])
    info = {}
    # silence/pause ids so the floors are exercised (they exist only in the tiny vocab)
    special = [148, 153, 166, 145, 163, 165] if c["phone_vocab"] > 166 else []
    for tag, m, first, final, disturb, alpha in chunks:
        p = rng.integers(1, c["phone_vocab"], size=m)
        for i, sp in enumerate(special):
            if i < m:
                p[(2 * i + 1) % m] = sp
        t = rng.integers(1, c["tone_vocab"], size=m)
        p_t, t_t = (torch.from_numpy(a.astype("i8"))[None] for a in (p, t))

        # Unrounded regression outputs, from a copy of the prompt state.
        pos = torch.LongTensor([list(range(ctx.shape[1], ctx.shape[1] + m))])
        with torch.no_grad():
            raw_codes, probs = model.infer(
                p_t, {"tone": t_t}, None, None, None, incremental_state=copy.deepcopy(state),
                first_decoder_inp=ctx[:, -1:], spk_pos_ids_flat=pos, return_probs=True)

        draws = []
        orig = torch.rand_like

        def rec(x, *a, **k):
            r = orig(x, *a, **k)
            draws.append(r.clone())
            return r

        torch.manual_seed(seed + len(info))
        torch.rand_like = rec
        try:
            with torch.no_grad():
                mel2ph_pred = dur_pred(me, ctx, state, p_t, t_t, 0, disturb, alpha, is_first=first, is_final=final)
        finally:
            torch.rand_like = orig
        assert len(draws) == 2, len(draws)
        save_npy(out, f"{tag}_phone.npy", p)
        save_npy(out, f"{tag}_tone.npy", t)
        save_npy(out, f"{tag}_raw_codes.npy", raw_codes[0])
        save_npy(out, f"{tag}_raw_reg.npy", probs[0, :, 0])
        save_npy(out, f"{tag}_rand_0.npy", draws[0][0])
        save_npy(out, f"{tag}_rand_1.npy", draws[1][0])
        save_npy(out, f"{tag}_mel2ph.npy", mel2ph_pred[0])
        info[tag] = {"m": m, "is_first": first, "is_final": final, "disturb": disturb, "alpha": alpha,
                     "frames": int(mel2ph_pred.shape[1]), "raw_codes": raw_codes[0].tolist()}
    return info


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
    stub_imports()
    os.makedirs(args.out, exist_ok=True)

    model = make_model(TINY)
    randomize(model, 2468)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    save_file({k: np.ascontiguousarray(v) for k, v in cv.convert_state_dict(sd).items()},
              os.path.join(args.out, "tiny_weights.safetensors"))
    cfg = dict(TINY, ffn_hidden=4 * TINY["hidden"])
    cfg.pop("char_dict_size")
    cfg["tone_vocab"] = TINY["tone_vocab"]
    with open(os.path.join(args.out, "tiny_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)

    info = run(model, TINY, args.out, 31, 9, CHUNKS)
    manifest = {
        "reference": "bytedance/MegaTTS3 tts/modules/ar_dur + tts/frontend_function.py (Apache-2.0)",
        "torch": torch.__version__, "chunks": info,
        "files": {f: sha256(os.path.join(args.out, f)) for f in sorted(os.listdir(args.out)) if f != "manifest.json"},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print("wrote", args.out, {k: (v["m"], v["frames"], max(v["raw_codes"])) for k, v in info.items()})

    if args.real_src:
        real(args)


def real(args):
    real_c = dict(hidden=512, layers=8, heads=8, code_size=128, phone_vocab=174, tone_vocab=23,
                  enc_filter=1024, enc_heads=2, enc_layers=4, enc_kernel=3, char_dict_size=15000)
    model = make_model(real_c, {"ling_label_dict_size": [20, 4, 5, 2, 3, 3, 3, 6, 15]})
    sd = cv.load_ckpt_state_dict(os.path.join(args.real_src, "duration_lm/model_only_last.ckpt"), "dur_model")
    res = model.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()}, strict=True)
    print("real load:", res)
    os.makedirs(args.real_out, exist_ok=True)
    chunks = [("r0", 10, False, True, 0.1, 1.0)]
    info = run(model, real_c, args.real_out, 77, 14, chunks, real=True)
    print("real chunk", {k: (v["frames"], v["raw_codes"]) for k, v in info.items()})


if __name__ == "__main__":
    main()
