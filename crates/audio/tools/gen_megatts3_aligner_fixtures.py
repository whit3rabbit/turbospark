#!/usr/bin/env python3
"""Regenerate the MegaTTS3 aligner Rust parity fixtures.

Normative reference: the unmodified MegaTTS3 code
(`tts/modules/aligner/whisper_small.py`, `align()` in `tts/frontend_function.py`
from github.com/bytedance/MegaTTS3, Apache-2.0; the Hugging Face Space
mrfakename/MegaTTS3-Voice-Cloning carries the same files at revision
8c2b5ab2ac67affda57a649fe0637425e8e1de58) and OpenAI Whisper v20240930
(`whisper/audio.py` and its `mel_filters.npz`, MIT).

Four kinds of fixture:
- model: a seeded tiny Whisper built from the reference's own `AudioEncoder`
  and `TextDecoder` (encoder output, teacher-forced logits for a token
  sequence, a short greedy decode);
- mel: OpenAI's real `log_mel_spectrogram` on a synthetic 16 kHz waveform;
- parse: scripted token streams pushed through the reference's real `align()`
  with a mock model, so the token parser, the duration fix-up (including the
  negative-duration path) and the length regulator are what is compared;
- real (optional, never committed): the real checkpoint on a real speech clip,
  resampled by the reference's librosa/soxr path.

Needs torch, numpy, safetensors, librosa (soxr) in a scratch venv, plus
`--whisper-pkg DIR` holding `whisper/audio.py`, `whisper/utils.py` and
`whisper/assets/mel_filters.npz` extracted from the openai-whisper sdist.

    python gen_megatts3_aligner_fixtures.py --ref DIR --whisper-pkg DIR \
        --out crates/audio/testdata/megatts3_aligner

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

TINY = dict(n_mels=80, state=32, heads=2, enc_layers=2, dec_layers=2, enc_ctx=64, dec_ctx=800, vocab=810)
GREEDY_STEPS = 24
VOCAB_REAL = 6800


def setup_imports(ref, whisper_pkg):
    sys.path.insert(0, ref)
    sys.path.insert(0, whisper_pkg)
    sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda x, *a, **k: x))
    import whisper  # namespace package: only audio.py/utils.py are present
    import whisper.audio as wa

    whisper.log_mel_spectrogram = wa.log_mel_spectrogram
    return wa


def tiny_model():
    from tts.modules.aligner.whisper_small import AudioEncoder, TextDecoder, Whisper

    class TinyWhisper(Whisper):
        def __init__(self, c):
            nn.Module.__init__(self)
            self.n_vocab, self.n_text_layer, self.n_text_head, self.n_text_ctx = (
                c["vocab"], c["dec_layers"], c["heads"], c["dec_ctx"])
            self.encoder = AudioEncoder(c["n_mels"], c["enc_ctx"], c["state"], c["heads"], c["enc_layers"])
            self.decoder = TextDecoder(c["vocab"], c["dec_ctx"], c["state"], c["heads"], c["dec_layers"])

    return TinyWhisper(TINY).eval()


def randomize(model, seed):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("attn_ln.weight", "mlp_ln.weight", "ln.weight", "ln_post.weight")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
            elif "embedding" in name:
                p.copy_(0.4 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(0.12 * torch.randn(p.shape, generator=gen))


def save_npy(out, name, arr):
    arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    dt = "<i4" if arr.dtype.kind == "i" else "<f4"
    np.save(os.path.join(out, name), np.ascontiguousarray(arr, dtype=dt))


def model_fixtures(out):
    model = tiny_model()
    randomize(model, 99)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    save_file({k: np.ascontiguousarray(v) for k, v in cv.convert_state_dict(sd).items()},
              os.path.join(out, "tiny_weights.safetensors"))
    with open(os.path.join(out, "tiny_config.json"), "w") as fh:
        json.dump(TINY, fh, indent=2, sort_keys=True)
    rng = np.random.default_rng(5)
    mel = (0.4 * rng.standard_normal((80, 40))).astype("<f4")
    with torch.no_grad():
        feats = model.embed_audio(torch.from_numpy(mel)[None])
        tokens = torch.from_numpy(np.concatenate([[798], rng.integers(0, TINY["vocab"], 9)]).astype("i8"))[None]
        tf_logits = model.logits(tokens, feats, None)
        # The reference decode loop, with the full-sequence recompute it uses.
        tok = torch.LongTensor([[798]])
        min_margin = 1e9
        for _ in range(GREEDY_STEPS):
            logits = model.logits(tok, feats, None)[:, -1]
            top2 = torch.topk(logits, 2).values[0]
            min_margin = min(min_margin, float(top2[0] - top2[1]))
            pred = torch.argmax(torch.softmax(logits, dim=-1), 1)[None]
            tok = torch.cat([tok, pred], dim=1)
    assert min_margin > 1e-3, f"greedy decode too close to a tie ({min_margin}); change the seed"
    save_npy(out, "tiny_mel.npy", mel)
    save_npy(out, "tiny_feats.npy", feats[0])
    save_npy(out, "tiny_tf_tokens.npy", tokens[0])
    save_npy(out, "tiny_tf_logits.npy", tf_logits[0])
    save_npy(out, "tiny_greedy.npy", tok[0])
    return {"greedy_min_margin": min_margin, "greedy_steps": GREEDY_STEPS}


def mel_fixtures(out, wa):
    rng = np.random.default_rng(11)
    n = int(3.37 * 16000)
    t = np.arange(n) / 16000.0
    wav = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1350 * t * (1 + 0.1 * t))
           + 0.05 * rng.standard_normal(n)).astype("<f4")
    mel = wa.log_mel_spectrogram(torch.from_numpy(wav), 80)  # [80, frames]
    save_npy(out, "mel_wav.npy", wav)
    save_npy(out, "mel_ref.npy", mel)
    return {"mel_frames": int(mel.shape[1]), "pmf": int(mel.shape[1] // 8 * 8)}


def stub_self(model_or_mock, sr):
    from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator

    return types.SimpleNamespace(aligner_lm=model_or_mock, sr=sr, fm=8, device="cpu",
                                 precision=torch.float32, length_regulator=LengthRegulator())


class ScriptedAligner:
    """Stands in for the Whisper model: always predicts the next scripted token."""

    def __init__(self, stream):
        self.stream = stream

    def embed_audio(self, mel):
        return None

    def logits(self, token, audio_features, kv_cache):
        n = token.shape[1]
        out = torch.full((1, n, VOCAB_REAL), -10.0)
        out[0, -1, self.stream[n]] = 10.0
        return out


def stream_of(pairs):
    s = [798]
    for ph, end in pairs:
        s += [ph, 800 + end]
    return s + [799]


# 2.0 s of audio -> 201 STFT frames -> 200 after the dropped frame (pmf 200).
PARSE_CASES = {
    # Total 171 < 200: the last phone absorbs the 29 missing frames.
    "s1": [(150, 20), (300, 55), (160, 80), (205, 120), (77, 150), (399, 171)],
    # Durations 30,1,40,25,2,60,55 sum to 213: 13 frames come off phone by phone,
    # driving the 1-frame phone negative. Chinese edge tokens 200 and 788.
    "s2": [(200, 30), (788, 31), (150, 71), (306, 96), (145, 98), (545, 158), (148, 213)],
    # Total exactly 200.
    "s3": [(10, 50), (403, 100), (99, 150), (163, 200)],
    # Zero-length phone in the middle, short total.
    "s4": [(60, 40), (207, 40), (61, 90)],
}


def parse_fixtures(out):
    from tts.frontend_function import align

    rng = np.random.default_rng(3)
    wav = (0.1 * rng.standard_normal(32000)).astype("<f4")  # 2.0 s
    info = {}
    for tag, pairs in PARSE_CASES.items():
        stream = stream_of(pairs)
        me = stub_self(ScriptedAligner(stream), 16000)
        ph, tone, mel2ph = align(me, wav)
        save_npy(out, f"{tag}_stream.npy", np.array(stream))
        save_npy(out, f"{tag}_ph.npy", ph[0].long())
        save_npy(out, f"{tag}_tone.npy", tone[0].long())
        save_npy(out, f"{tag}_mel2ph.npy", mel2ph[0])
        info[tag] = {"phones": len(pairs), "mel2ph_len": int(mel2ph.shape[1])}
    return info


def real_fixtures(args, out):
    import librosa
    from tts.frontend_function import align
    from tts.modules.aligner.whisper_small import Whisper

    model = Whisper().eval()
    sd = cv.load_ckpt_state_dict(os.path.join(args.real_src, "aligner_lm/model_only_last.ckpt"), "model")
    print("real load:", model.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()}, strict=True))
    clip, _ = librosa.load(args.clip, sr=16000)
    wav24 = librosa.resample(clip, orig_sr=16000, target_sr=24000).astype("<f4")
    # `preprocess`: pad to ws - 1 modulo 1200, then 12000 zeros.
    w = wav24
    if len(w) % 1200 < 1199:
        w = np.pad(w, (0, 1199 - len(w) % 1200), mode="constant").astype(np.float32)
    w = np.pad(w, (0, 12000), mode="constant").astype(np.float32)
    me = stub_self(model, 24000)
    # Record the reference's own 16 kHz audio and mel for the resampler comparison.
    wav16 = librosa.resample(w, orig_sr=24000, target_sr=16000)
    import whisper

    mel = whisper.log_mel_spectrogram(wav16)
    toks = []
    orig = torch.cat

    def rec_cat(ts, *a, **k):
        r = orig(ts, *a, **k)
        if isinstance(ts, (list, tuple)) and len(ts) == 2 and r.dtype == torch.long and r.dim() == 2 and r.shape[0] == 1:
            toks.append(r)
        return r

    torch.cat = rec_cat
    try:
        ph, tone, mel2ph = align(me, w)
    finally:
        torch.cat = orig
    stream = toks[-1][0]
    os.makedirs(out, exist_ok=True)
    save_npy(out, "r_wav24.npy", wav24)
    save_npy(out, "r_wav24_padded_len.npy", np.array([len(w)]))
    save_npy(out, "r_wav16.npy", wav16)
    save_npy(out, "r_mel.npy", mel)
    save_npy(out, "r_stream.npy", stream)
    save_npy(out, "r_ph.npy", ph[0].long())
    save_npy(out, "r_tone.npy", tone[0].long())
    save_npy(out, "r_mel2ph.npy", mel2ph[0])
    print("real: tokens", len(stream), "phones", ph.shape[1], "mel2ph", mel2ph.shape[1])


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--whisper-pkg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-src")
    ap.add_argument("--clip")
    ap.add_argument("--real-out")
    args = ap.parse_args()
    wa = setup_imports(args.ref, args.whisper_pkg)
    os.makedirs(args.out, exist_ok=True)
    info = {"model": model_fixtures(args.out), "mel": mel_fixtures(args.out, wa), "parse": parse_fixtures(args.out)}
    manifest = {
        "reference": "bytedance/MegaTTS3 aligner + frontend_function.align; openai-whisper 20240930 audio.py",
        "torch": torch.__version__, "info": info,
        "files": {f: sha256(os.path.join(args.out, f)) for f in sorted(os.listdir(args.out)) if f != "manifest.json"},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print("wrote", args.out, json.dumps(info))
    if args.real_src:
        real_fixtures(args, args.real_out)


if __name__ == "__main__":
    main()
