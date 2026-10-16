#!/usr/bin/env python3
"""Refresh the optional chatterbox parity-bisection traces (dbg_*.npy).

The two `debug_*_stages` tests in
`crates/audio/src/tts/chatterbox/tests.rs` compare the Rust flow decoder
and HiFT vocoder stage by stage against these traces. They are bisection
scratch, not shipped fixtures: the tests skip when the files are absent,
and a fresh checkout does not need them.

The script rebuilds the tiny fixture exactly like
`gen_chatterbox_fixtures.py`, overwrites every parameter with the frozen
checkpoint values, and dumps one npy per pipeline stage. The HiFT stages
replay the frozen sine draws (`hift/phase.npy`, `hift/sine_noise.npy`)
so every dumped stage aligns with what the Rust tests consume.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/dbg_chatterbox_traces.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_chatterbox_fixtures as G

OUT = Path(__file__).resolve().parent.parent / "testdata" / "tts_chatterbox"


def save(name: str, arr) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(np.asarray(arr, dtype=np.float32)))


def flow_traces(s3gen) -> None:
    """Flow decoder stages at t = 0 plus the per-step Euler states."""
    flow = s3gen.flow
    decoder = flow.decoder
    est = decoder.estimator

    prompt_tokens, gen_tokens, prompt_feat, embedding = G.flow_inputs(G.FLOW_SEED + 2)
    emb = mx.array(embedding)[None]
    norm = mx.linalg.norm(emb, axis=1, keepdims=True)
    spk = flow.spk_embed_affine_layer(emb / (norm + 1e-8))
    save("dbg_spk.npy", spk)

    token = mx.concatenate(
        [mx.array(prompt_tokens)[None], mx.array(gen_tokens)[None]], axis=1
    )
    token_len = mx.array([G.PROMPT_TOKENS + G.GEN_TOKENS], dtype=mx.int32)
    num_emb = flow.input_embedding.weight.shape[0]
    h, _ = flow.encoder(flow.input_embedding(mx.clip(token, 0, num_emb - 1)), token_len, streaming=False)
    mu = flow.encoder_proj(h)
    save("dbg_mu.npy", mx.transpose(mu, [0, 2, 1]))
    mu_cm = mx.transpose(mu, [0, 2, 1])

    mel_len1 = prompt_feat.shape[0]
    total = mu.shape[1]
    conds = mx.zeros([1, total, 80])
    conds[:, :mel_len1] = mx.array(prompt_feat)[None]
    conds = mx.transpose(conds, [0, 2, 1])
    mask = mx.ones([1, 1, total])

    T = mu_cm.shape[2]
    z = decoder.rand_noise[:, :, :T]
    t_span = 1 - mx.cos(mx.linspace(0, 1, 11) * 0.5 * np.pi)
    t0 = mx.expand_dims(t_span[0], 0)
    t_emb = est.time_mlp(est.time_embeddings(t0))
    save("dbg_t_emb.npy", t_emb)

    x_in = mx.concatenate([z, mu_cm], axis=1)
    spks_e = mx.broadcast_to(mx.expand_dims(spk, -1), (1, spk.shape[1], T))
    x_in = mx.concatenate([x_in, spks_e, conds], axis=1)
    save("dbg_dec_in.npy", x_in)

    def transformers(block, x_cm):
        x_t = mx.swapaxes(x_cm, 1, 2)
        attn_bias = mx.zeros((1, T, T))
        for tb in block.transformer_blocks:
            x_t = tb(x_t, attention_mask=attn_bias, timestep=t_emb)
        return mx.swapaxes(x_t, 1, 2)

    down = est.down_blocks[0]
    d_res = down.resnet(x_in, mask, t_emb)
    save("dbg_dec_down_resnet.npy", d_res)
    d_tf = transformers(down, d_res)
    save("dbg_dec_down_transformers.npy", d_tf)
    d_tail = down.downsample(d_tf)
    save("dbg_dec_down_tail.npy", d_tail)

    x = d_tail
    for i, mid in enumerate(est.mid_blocks):
        m_res = mid.resnet(x, mask, t_emb)
        save(f"dbg_dec_mid{i}_resnet.npy", m_res)
        x = transformers(mid, m_res)
        save(f"dbg_dec_mid{i}.npy", x)

    up = est.up_blocks[0]
    up_in = mx.concatenate([x[:, :, : d_tf.shape[-1]], d_tf], axis=1)
    save("dbg_dec_up_in.npy", up_in)
    u_res = up.resnet(up_in, mask, t_emb)
    save("dbg_dec_up_resnet.npy", u_res)
    u_tf = transformers(up, u_res)
    save("dbg_dec_up_transformers.npy", u_tf)
    u_tail = up.upsample(u_tf)
    save("dbg_dec_up_tail.npy", u_tail)
    fb = est.final_block(u_tail, mask)
    save("dbg_dec_final_block.npy", fb)
    save("dbg_dec_final.npy", mx.swapaxes(est.final_proj(mx.swapaxes(fb, 1, 2)), 1, 2))

    xx = z
    t = t0
    dt = t_span[1] - t_span[0]
    zeros_mu = mx.zeros_like(mu_cm)
    zeros_spk = mx.zeros_like(spk)
    zeros_cond = mx.zeros_like(conds)
    for step in range(1, len(t_span)):
        t_emb_s = est.time_mlp(est.time_embeddings(t))
        dphi_c = est(xx, mask, mu_cm, t, spk, conds)
        dphi_u = est(xx, mask, zeros_mu, t, zeros_spk, zeros_cond)
        save(f"dbg_step{step}_dphi_c.npy", dphi_c)
        save(f"dbg_step{step}_dphi_u.npy", dphi_u)
        xx = xx + dt * (
            (1.0 + decoder.inference_cfg_rate) * dphi_c
            - decoder.inference_cfg_rate * dphi_u
        )
        save(f"dbg_step{step}_x.npy", xx)
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t


def hift_traces(s3gen) -> None:
    """HiFT vocoder stages with the frozen sine draws replayed."""
    import mlx.nn as nn

    hift = s3gen.mel2wav
    prompt_tokens, gen_tokens, prompt_feat, embedding = G.flow_inputs(G.FLOW_SEED + 2)
    feat, _ = s3gen.flow.inference(
        token=mx.array(gen_tokens)[None],
        token_len=mx.array([G.GEN_TOKENS], dtype=mx.int32),
        finalize=True,
        prompt_token=mx.array(prompt_tokens)[None],
        prompt_token_len=mx.array([G.PROMPT_TOKENS], dtype=mx.int32),
        prompt_feat=mx.array(prompt_feat)[None],
        prompt_feat_len=mx.array([G.PROMPT_TOKENS * 2], dtype=mx.int32),
        embedding=mx.array(embedding)[None],
    )
    mel_h = feat  # (1, 80, gen frames)
    f0 = hift.f0_predictor(mel_h)
    save("dbg_hift_f0.npy", f0)

    frozen_phase = np.load(OUT / "hift" / "phase.npy").reshape(1, -1, 1)
    frozen_noise = np.load(OUT / "hift" / "sine_noise.npy")[None]
    frozen_src = np.zeros((1, 1, frozen_noise.shape[-1]), dtype=np.float32)
    replay = [frozen_phase, frozen_noise, frozen_src]
    with G.DrawPatch(replay=replay):
        wav, _ = hift.inference(speech_feat=mel_h, cache_source=mx.zeros((1, 1, 0)))
        save("dbg_hift_wav.npy", wav)
    with G.DrawPatch(replay=replay):
        f0_up = mx.swapaxes(
            mx.repeat(mx.expand_dims(f0, 1), hift.f0_upsample_scale, axis=2), 1, 2
        )
        sine_wavs, uv, _ = hift.m_source.l_sin_gen(mx.swapaxes(f0_up, 1, 2))
        save("dbg_hift_sine.npy", sine_wavs)
        save("dbg_hift_uv.npy", uv)
        sine_merge = mx.swapaxes(
            mx.tanh(hift.m_source.l_linear(mx.swapaxes(sine_wavs, 1, 2))), 1, 2
        )
        save("dbg_hift_source.npy", sine_merge)

        s_real, s_imag = hift._stft(sine_merge.squeeze(1))
        s_stft = mx.concatenate([s_real, s_imag], axis=1)
        save("dbg_hift_sstft.npy", s_stft)

        x = mx.swapaxes(mel_h, 1, 2)
        x = hift.conv_pre(x)
        x = mx.swapaxes(x, 1, 2)
        save("dbg_hift_conv_pre.npy", x)
        for i in range(3):
            x = nn.leaky_relu(x, 0.1)
            x = mx.swapaxes(x, 1, 2)
            x = hift.ups[i](x)
            x = mx.swapaxes(x, 1, 2)
            if i == 2:
                x = mx.concatenate([x[:, :, 1:2], x], axis=2)
                save("dbg_hift_up2_padded.npy", x)
            else:
                save(f"dbg_hift_up{i}_pre.npy", x)
            si = mx.swapaxes(s_stft, 1, 2)
            si = hift.source_downs[i](si)
            si = mx.swapaxes(si, 1, 2)
            save(f"dbg_hift_up{i}_sidown.npy", si)
            for d in range(len(hift.source_resblocks[i].convs1)):
                xt = hift.source_resblocks[i].activations1[d](si)
                xt = hift.source_resblocks[i].convs1[d](mx.swapaxes(xt, 1, 2))
                xt = hift.source_resblocks[i].activations2[d](
                    mx.swapaxes(xt, 1, 2)
                )
                xt = hift.source_resblocks[i].convs2[d](mx.swapaxes(xt, 1, 2))
                si = mx.swapaxes(xt, 1, 2) + si
                save(f"dbg_hift_up{i}_si_it{d}.npy", si)
            save(f"dbg_hift_up{i}_si.npy", si)
            x = x + si
            start = i * 3
            x = mx.mean(
                mx.stack([hift.resblocks[start + j](x) for j in range(3)], axis=0),
                axis=0,
            )
            save(f"dbg_hift_up{i}_post.npy", x)
        x = nn.leaky_relu(x, 0.1)
        save("dbg_hift_xpost.npy", x)
        x = hift.conv_post(mx.swapaxes(x, 1, 2))
        x = mx.swapaxes(x, 1, 2)
        save("dbg_hift_melphase.npy", x)
        n_half = 9
        mag = mx.exp(x[:, :n_half, :])
        ph = mx.sin(x[:, n_half:, :])
        save("dbg_hift_wav_preclip.npy", hift._istft(mag, ph))


def main() -> None:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    G.apply_tiny_patches()
    s3gen = G.build_s3gen(
        G.FLOW_SEED, G.FLOW_SEED + 10, total_frames=2 * (G.PROMPT_TOKENS + G.GEN_TOKENS)
    )
    weights = mx.load(str(OUT / "model.safetensors"))
    flow_flat = {k[len("flow."):]: v for k, v in weights.items() if k.startswith("flow.")}
    s3gen.flow.update(tree_unflatten(flow_flat))
    s3gen.eval()
    flow_traces(s3gen)
    hift_traces(s3gen)
    print(f"refreshed bisection traces in {OUT}")
    mx.set_default_device(previous)


if __name__ == "__main__":
    main()
