#!/usr/bin/env python3
"""Golden tensors for the whole MegaTTS3 pipeline on the real checkpoints.

Never committed (real weights, real audio): the Rust test is env-gated.

Composes `MegaTTS3DiTInfer.preprocess` and `.forward` from the reference's own
functions (`align`, `make_dur_prompt`, `g2p`, `dur_pred`, `prepare_inputs_for_dit`,
`Diffusion.inference`, `WavVAE_V3`, pyloudnorm, `combine_audio_segments`),
skipping only what cannot run here: the WeTextProcessing normalizer (pynini) and
`langdetect`, so the input text must already be normalized English. Every
`torch.randn` and `torch.rand_like` draw is recorded in order so the Rust
pipeline can replay them: one normal draw for the encoder, then per chunk two
uniform draws (duration jitter) and `steps + 1` normal draws (sampler).

    python gen_megatts3_e2e_fixtures.py --ref DIR --whisper-pkg DIR \
        --src DIR_WITH_CKPTS --g2p DIR --clip speech.wav --out OUT

Reference: github.com/bytedance/MegaTTS3 (Apache-2.0), Space revision
8c2b5ab2ac67affda57a649fe0637425e8e1de58.
"""

import argparse
import json
import os
import sys
import types

import numpy as np
import torch

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import convert_megatts3 as cv  # noqa: E402

# (text, steps, p_w, t_w, max chunk bytes)
CASES = [
    ("Hello world, this is a short test.", 6, 1.4, 3.0, 130),
    ("Hello world, this is a short test.", 4, 2.0, 2.5, 20),
]
DUR_DISTURB, DUR_ALPHA = 0.1, 1.0


def save(out, name, arr):
    arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    dt = "<i4" if arr.dtype.kind == "i" else "<f4"
    np.save(os.path.join(out, name), np.ascontiguousarray(arr, dtype=dt))


def main():
    ap = argparse.ArgumentParser()
    for a in ("--ref", "--whisper-pkg", "--src", "--g2p", "--clip", "--out"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.ref)
    sys.path.insert(0, args.whisper_pkg)
    sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda x, *a, **k: x))
    sys.modules.setdefault("einops", types.ModuleType("einops"))  # only ConvLayerNorm uses it
    import librosa
    import pyloudnorm as pyln
    import whisper
    import whisper.audio as wa

    whisper.log_mel_spectrogram = wa.log_mel_spectrogram
    sys.modules.setdefault("pydub", types.SimpleNamespace(AudioSegment=None))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tts.frontend_function import align, dur_pred, g2p, make_dur_prompt, prepare_inputs_for_dit
    from tts.modules.aligner.whisper_small import Whisper
    from tts.modules.ar_dur.ar_dur_predictor import ARDurPredictor
    from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator
    from tts.modules.llm_dit.dit import Diffusion
    from tts.modules.wavvae.decoder.wavvae_v3 import WavVAE_V3
    from tts.utils.audio_utils.io import combine_audio_segments
    from tts.utils.text_utils.split_text import chunk_text_english

    sys.path.insert(0, TOOLS)
    from gen_megatts3_duration_fixtures import make_model as make_dur

    os.makedirs(args.out, exist_ok=True)

    def ckpt(rel, key):
        return {k: torch.from_numpy(v) for k, v in cv.load_ckpt_state_dict(os.path.join(args.src, rel), key).items()}

    aligner = Whisper().eval()
    aligner.load_state_dict(ckpt("aligner_lm/model_only_last.ckpt", "model"), strict=True)
    real_c = dict(hidden=512, layers=8, heads=8, code_size=128, phone_vocab=174, tone_vocab=23,
                  enc_filter=1024, enc_heads=2, enc_layers=4, enc_kernel=3, char_dict_size=15000)
    dur_model = make_dur(real_c, {"ling_label_dict_size": [20, 4, 5, 2, 3, 3, 3, 6, 15]})
    dur_model.load_state_dict(ckpt("duration_lm/model_only_last.ckpt", "dur_model"), strict=True)
    dit = Diffusion().eval()
    dit.load_state_dict(ckpt("diffusion_transformer/model_only_last.ckpt", "dit"), strict=False)
    wavvae = WavVAE_V3(hparams={"melgan_config": {"frame_shift": 240, "use_tanh": True}}).eval()
    wavvae.load_state_dict(ckpt("wavvae/model_only_last.ckpt", "model_gen"), strict=True)
    tok = AutoTokenizer.from_pretrained(args.g2p, padding_side="right")
    g2p_model = AutoModelForCausalLM.from_pretrained(args.g2p, dtype=torch.float32).eval()

    me = types.SimpleNamespace(
        sr=24000, fm=8, device="cpu", precision=torch.float32, vae_stride=4, aligner_lm=aligner,
        dur_model=dur_model, hp_dur_model={"dur_code_size": 128}, length_regulator=LengthRegulator(),
        g2p_tokenizer=tok, g2p_model=g2p_model, speech_start_idx=tok.encode("<Reserved_TTS_0>")[0],
        cfg_mask_token_phone=302 - 1, cfg_mask_token_tone=32 - 1,
    )

    # Draw recording.
    events = []
    orig_randn, orig_rand_like = torch.randn, torch.rand_like

    def rec_randn(*a, **k):
        r = orig_randn(*a, **k)
        events.append(("n", r.clone()))
        return r

    def rec_rand_like(x, *a, **k):
        r = orig_rand_like(x, *a, **k)
        events.append(("u", r.clone()))
        return r

    # The clip: 16 kHz speech -> 24 kHz prompt (what a user would supply).
    clip, _ = librosa.load(args.clip, sr=16000)
    wav24 = librosa.resample(clip, orig_sr=16000, target_sr=24000).astype("float32")
    save(args.out, "e_wav24.npy", wav24)

    # ---- preprocess ----
    wav = wav24
    if len(wav) % 1200 < 1199:
        wav = np.pad(wav, (0, 1199 - len(wav) % 1200), mode="constant").astype(np.float32)
    wav = np.pad(wav, (0, 12000), mode="constant").astype(np.float32)
    meter = pyln.Meter(24000)
    loudness_prompt = meter.integrated_loudness(wav.astype(float))
    save(args.out, "e_wav16.npy", librosa.resample(wav, orig_sr=24000, target_sr=16000))

    torch.randn, torch.rand_like = rec_randn, rec_rand_like
    try:
        with torch.inference_mode():
            ph_ref, tone_ref, mel2ph_ref = align(me, wav)
            wav_t = torch.FloatTensor(wav)[None]
            vae_latent = wavvae.encode_latent(wav_t)
            vae_latent = vae_latent[:, : mel2ph_ref.size(1) // 4]
            state, ctx = make_dur_prompt(me, mel2ph_ref, ph_ref, tone_ref)
    finally:
        torch.randn, torch.rand_like = orig_randn, orig_rand_like
    save(args.out, "e_prompt_phone.npy", ph_ref[0].long())
    save(args.out, "e_prompt_tone.npy", tone_ref[0].long())
    save(args.out, "e_prompt_mel2ph.npy", mel2ph_ref[0])
    save(args.out, "e_prompt_latent.npy", vae_latent[0])
    info = {"loudness_prompt": float(loudness_prompt), "prompt_phones": int(ph_ref.shape[1]),
            "prompt_latent_frames": int(vae_latent.shape[1]), "cases": []}

    # ---- forward, one case after another (the draw stream continues) ----
    for ci, (text, steps, p_w, t_w, max_chars) in enumerate(CASES):
        segs = chunk_text_english(text, max_chars=max_chars)
        out_segs = []
        torch.randn, torch.rand_like = rec_randn, rec_rand_like
        try:
            with torch.inference_mode():
                for seg_i, seg in enumerate(segs):
                    ph_pred, tone_pred = g2p(me, seg)
                    mel2ph_pred = dur_pred(me, ctx, state, ph_pred, tone_pred, seg_i, DUR_DISTURB, DUR_ALPHA,
                                           is_first=seg_i == 0, is_final=seg_i == len(segs) - 1)
                    inputs = prepare_inputs_for_dit(me, mel2ph_ref, mel2ph_pred, ph_ref, tone_ref, ph_pred, tone_pred, vae_latent)
                    x = dit.inference(inputs, timesteps=steps, seq_cfg_w=[p_w, t_w]).float()
                    x[:, : vae_latent.size(1)] = vae_latent
                    wav_pred = wavvae.decode(x)[0, 0].to(torch.float32)
                    wav_pred = wav_pred[vae_latent.size(1) * 4 * 240:].cpu().numpy()
                    loud = meter.integrated_loudness(wav_pred.astype(float))
                    wav_pred = pyln.normalize.loudness(wav_pred, loud, loudness_prompt)
                    if np.abs(wav_pred).max() >= 1:
                        wav_pred = wav_pred / np.abs(wav_pred).max() * 0.95
                    out_segs.append(wav_pred)
        finally:
            torch.randn, torch.rand_like = orig_randn, orig_rand_like
        audio = combine_audio_segments(out_segs, sr=24000).astype(float)
        save(args.out, f"e_audio_{ci}.npy", audio.astype("float32"))
        info["cases"].append({"text": text, "steps": steps, "p_w": p_w, "t_w": t_w, "max_chars": max_chars,
                              "chunks": segs, "samples": int(len(audio)), "peak": float(np.abs(audio).max())})
        print("case", ci, segs, "->", len(audio), "samples")

    # Draws in order; layout converted to what the Rust pipeline requests.
    normals = [e for e in events if e[0] == "n"]
    uniforms = [e for e in events if e[0] == "u"]
    for i, (_, t) in enumerate(normals):
        t = t[0]  # drop batch
        if i == 0:  # encoder: [32, T] -> time-major [T, 32]
            t = t.T
        save(args.out, f"n_{i:03d}.npy", t)
    for i, (_, t) in enumerate(uniforms):
        save(args.out, f"u_{i:03d}.npy", t[0])
    info["normals"], info["uniforms"] = len(normals), len(uniforms)
    with open(os.path.join(args.out, "info.json"), "w") as fh:
        json.dump(info, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in info.items() if k != "cases"}))


if __name__ == "__main__":
    main()
