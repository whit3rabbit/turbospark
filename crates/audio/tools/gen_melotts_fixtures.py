#!/usr/bin/env python3
"""Regenerate the melotts family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/tts/models/melotts, mlx 0.32.3). The reference modules run
directly against a tiny seeded model:

- text.py `text_normalize` (lowercase, abbreviations, number expansion)
  and the pure post-G2P pipeline (`_refine_ph`, `_post_replace_ph`,
  `_distribute_phone`, `cleaned_text_to_sequence`, the `add_blank`
  interleave) on pinned inputs (the Rust port pins the same string and
  mapping pipeline);
- the G2P itself (g2p_en + cmudict + the bert-base-uncased wordpiece
  tokenizer) is NOT runnable in this venv (the packages are not
  installed and the HF cache has no vocab), so hand-written ARPAbet
  phone lists stand in for its output and the reference post-processing
  runs on them;
- the tiny BERT (`bert.py`) on pinned wordpiece-style ids, feeding
  `get_bert_features` (the word2ph repeat) for the ja_bert features the
  shipped generate path consumes;
- the end-to-end assembly mirrors `Model._prepare_inputs` + `infer`
  (the reference `generate` cannot run: it tokenizes internally), with
  the tiny BERT attached and the z-path stochastic draw pinned by
  `mx.random.seed`.

Weights are saved in the raw MLX parameter tree layout (exactly what
`tree_flatten` lists under `enc_p.*`, `dec.*`, `flow_layers.*`, `dp.*`,
`emb_g.*`, `bert.*`); conv weights are MLX order and the Rust loader
transposes. The training-only modules (`enc_q`, `sdp`) are saved too and
ignored by the Rust loader.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_melotts_fixtures.py

Outputs into crates/audio/testdata/tts_melotts/.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLX_NUM_THREADS", "1")

import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.tts.models.melotts import text as melo_text
from mlx_audio.tts.models.melotts.bert import BertConfig, BertModel
from mlx_audio.tts.models.melotts.melotts import Model as MeloModel
from mlx_audio.tts.models.melotts.modules import Flip
from mlx_audio.tts.models.melotts.melotts import ModelConfig

OUT = Path("../turbospark/crates/audio/testdata/tts_melotts")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

MODEL_SEED = 11
E2E_SEED = 4321

# The reference's default English fallback symbol table (overwritten from
# config.json at runtime); n_vocab follows it.
SYMBOLS = list(melo_text.symbols)
PAD = melo_text.pad

CONFIG = {
    "model_type": "melotts",
    "sampling_rate": 44100,
    "filter_length": 64,
    "hop_length": 16,
    "segment_size": 8192,
    "add_blank": True,
    "n_speakers": 4,
    "spk2id": {"EN-Default": 0, "EN-US": 1},
    "inter_channels": 32,
    "hidden_channels": 32,
    "filter_channels": 64,
    "n_heads": 2,
    "n_layers": 2,
    "n_layers_trans_flow": 2,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 5, 7],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [4, 4, 2],
    "upsample_initial_channel": 32,
    "upsample_kernel_sizes": [8, 8, 4],
    "n_layers_q": 2,
    "use_spectral_norm": False,
    "gin_channels": 16,
    "use_spk_conditioned_encoder": True,
    "use_noise_scaled_mas": True,
    "use_transformer_flow": True,
    "num_tones": 16,
    "num_languages": 10,
    "n_vocab": len(SYMBOLS),
    "bert_hidden_size": 1024,
    "symbols": SYMBOLS,
    # Fixture-only extension: the reference hard-codes the bert-base
    # dimensions (768 / 12 layers / 12 heads / 3072); the Rust loader
    # defaults to those and reads this section when present.
    "bert": {
        "vocab_size": 100,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-12,
    },
}

BERT_FEATURE_DIM = 768


def floats(arr) -> list:
    # Seven decimals keep every f32 value within one ULP of the stored
    # magnitude while roughly halving the JSON size.
    values = np.asarray(arr, dtype=np.float32).ravel().tolist()
    return [round(v, 7) for v in values]


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def build_model():
    config = ModelConfig(**{
        k: v for k, v in CONFIG.items() if k not in ("model_type", "symbols", "bert")
    })
    model = MeloModel(config)
    model.eval()
    mx.random.seed(MODEL_SEED)

    def fill(weight):
        return mx.random.uniform(low=-0.06, high=0.06, shape=weight.shape)

    flat = dict(mlx.utils.tree_flatten(model.parameters()))
    for name, weight in list(flat.items()):
        if weight.dtype not in (mx.float32, mx.float16):
            continue
        if name.endswith("norm.weight") or "LayerNorm" in name:
            flat[name] = mx.ones(weight.shape)
        elif name.endswith("norm.bias"):
            flat[name] = mx.zeros(weight.shape)
        else:
            flat[name] = fill(weight)
    model.load_weights(list(flat.items()), strict=True)

    # The tiny BERT's hidden size stands in for the 768-dim ja_bert input
    # the reference hardcodes; rebuild that projection so the fixture is
    # self-consistent (the Rust reads the ja width from the config, with
    # 768 as the default).
    from mlx_audio.tts.models.melotts.hifigan import Conv1dPT

    ja_dim = CONFIG["bert"]["hidden_size"]
    model.enc_p.ja_bert_proj = Conv1dPT(ja_dim, CONFIG["hidden_channels"], 1)
    mx.random.seed(MODEL_SEED + 5)
    flat = dict(mlx.utils.tree_flatten(model.parameters()))
    for name, weight in list(flat.items()):
        if name.startswith("enc_p.ja_bert_proj") and weight.dtype in (
            mx.float32,
            mx.float16,
        ):
            flat[name] = mx.random.uniform(
                low=-0.06, high=0.06, shape=weight.shape
            )
    model.load_weights(list(flat.items()), strict=True)
    return model, flat


def warmup(model):
    """One discarded call: MLX lazy init makes the first call differ from
    the steady state, so every pinned run goes through this first."""
    x = mx.array([[1, 2, 3, 1]], dtype=mx.int32)
    mx.random.seed(99)
    model.infer(
        x,
        mx.array([4]),
        mx.array([0]),
        mx.array([[7] * 4], dtype=mx.int32),
        mx.array([[2] * 4], dtype=mx.int32),
        mx.zeros((1, 1024, 4)),
        ja_bert=mx.zeros((1, CONFIG["bert"]["hidden_size"], 4)),
        noise_scale=0.1,
    )


def build_bert():
    """The tiny seeded BERT (fixture-only dimensions)."""
    config = BertConfig(**CONFIG["bert"])
    bert = BertModel(config)
    bert.eval()
    mx.random.seed(MODEL_SEED + 2)

    def fill(weight):
        return mx.random.uniform(low=-0.08, high=0.08, shape=weight.shape)

    flat = dict(mlx.utils.tree_flatten(bert.parameters()))
    for name, weight in list(flat.items()):
        if weight.dtype not in (mx.float32, mx.float16):
            continue
        if name.endswith("norm.weight") or "LayerNorm" in name:
            flat[name] = mx.ones(weight.shape)
        elif name.endswith("norm.bias"):
            flat[name] = mx.zeros(weight.shape)
        else:
            flat[name] = fill(weight)
    bert.load_weights(list(flat.items()), strict=True)
    return bert, flat


class PinnedTokenizer:
    """Stands in for the bert-base-uncased wordpiece tokenizer: the ids
    are pinned per case (the real tokenizer is not available offline)."""

    def __init__(self, ids, mask):
        self.ids = ids
        self.mask = mask

    def __call__(self, text, return_tensors=None):
        return {
            "input_ids": self.ids,
            "token_type_ids": np.zeros_like(self.ids),
            "attention_mask": self.mask,
        }

    def tokenize(self, text):
        return ["pinned"]


def bert_features_for(word2ph, ids, mask, bert, add_blank=False):
    """The reference `get_bert_features` with the tokenizer pinned.

    Returns the phone-level features plus the intermediate traces
    (per-layer hidden states and the pooled CLS vector) the fixture
    records for stage-by-stage bisection.
    """
    melo_text._tokenizer = PinnedTokenizer(np.array(ids, dtype=np.int64), np.array(mask, dtype=np.int64))
    try:
        features = melo_text.get_bert_features("ignored", list(word2ph), bert, add_blank=add_blank)
    finally:
        melo_text._tokenizer = None
    input_ids = mx.array(ids)
    token_type_ids = mx.zeros_like(input_ids)
    attention_mask = mx.array(mask)
    _, pooled, hidden_states = bert(
        input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    traces = {
        "features": features,
        "hidden_states": [h for h in hidden_states],
        "pooled": pooled[0],
    }
    return traces


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, flat = build_model()
    warmup(model)
    bert, bert_flat = build_bert()
    model.bert = bert

    # ---- frontend traces --------------------------------------------------
    normalize_cases = [
        {"name": name, "input": text, "output": melo_text.text_normalize(text)}
        for name, text in NORMALIZE_CASES
    ]
    g2p_cases = []
    for name, norm_text, word_phones in G2P_CASES:
        # The reference g2p pads with "_" and distributes phones over
        # wordpiece groups; here each word's ARPAbet list stands in for
        # one word group, so word2ph is the per-word phone counts.
        phones: list[str] = []
        tones: list[int] = []
        word2ph: list[int] = []
        for word_phone_list in word_phones:
            count = 0
            for ph in word_phone_list:
                p, t = melo_text._refine_ph(ph)
                phones.append(p)
                tones.append(t)
                count += 1
            word2ph.append(count)
        phones = [melo_text._post_replace_ph(p) for p in phones]
        phones = [PAD] + phones + [PAD]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
        phone_ids, tone_ids, lang_ids = melo_text.cleaned_text_to_sequence(
            phones, tones, "EN"
        )
        g2p_cases.append(
            {
                "name": name,
                "norm_text": norm_text,
                "word_phones": word_phones,
                "phones": phones,
                "tones": tones,
                "word2ph": word2ph,
                "phone_ids": phone_ids,
                "tone_ids": tone_ids,
                "lang_ids": lang_ids,
            }
        )

    # The add_blank interleave over the first case.
    case = g2p_cases[0]
    phones_b = [PAD]
    tones_b = [0]
    for p, t in zip(case["phones"], case["tones"]):
        phones_b.extend([p, PAD])
        tones_b.extend([t, 0])
    ids_b, tones_mapped, lang_b = melo_text.cleaned_text_to_sequence(
        phones_b, tones_b, "EN"
    )
    add_blank_case = {
        "input_phones": list(case["phones"]),
        "input_tones": list(case["tones"]),
        "phones": phones_b,
        "tones": tones_b,
        "phone_ids": ids_b,
        "tone_ids": tones_mapped,
        "lang_ids": lang_b,
    }

    frontend = {
        "reference_commit": REFERENCE_COMMIT,
        "symbols": SYMBOLS,
        "normalize_cases": normalize_cases,
        "g2p_cases": g2p_cases,
        "add_blank_case": add_blank_case,
    }
    frontend_dir = OUT / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    (frontend_dir / "traces.json").write_text(json.dumps(frontend, indent=1))

    # ---- BERT features on pinned wordpiece-style ids ----------------------
    bert_feature_cases = []
    for name, ids, mask, word2ph in [
        ("short", [[1, 5, 9, 2]], [[1, 1, 1, 1]], [1, 2, 1, 1]),
        (
            "mask_tail",
            [[1, 7, 3, 9, 2, 0]],
            [[1, 1, 1, 1, 1, 0]],
            [2, 1, 1, 1, 1, 1],
        ),
    ]:
        trace = bert_features_for(word2ph, ids, mask, model.bert, add_blank=False)
        bert_feature_cases.append(
            {
                "name": name,
                "input_ids": ints(ids[0]),
                "attention_mask": ints(mask[0]),
                "word2ph": word2ph,
                # Per-layer hidden states (the embedding output first),
                # the pooled CLS vector, and the phone-level features.
                "hidden_states": [floats(h) for h in trace["hidden_states"]],
                "pooled": floats(trace["pooled"]),
                "features": floats(trace["features"]),
            }
        )
    bert_dir = OUT / "bert"
    bert_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(bert_dir / "model.safetensors"), dict(bert_flat))
    (bert_dir / "config.json").write_text(json.dumps(CONFIG["bert"], indent=2) + "\n")
    (bert_dir / "traces.json").write_text(
        json.dumps({"cases": bert_feature_cases}, indent=1)
    )

    # ---- end-to-end scenarios (the assembled _prepare_inputs + infer) -----
    scenarios = []
    for name, text, word_phones in E2E_CASES:
        norm_text = melo_text.text_normalize(text)
        # The pinned G2P stand-in: one ARPAbet list per whitespace word of
        # the normalized text.
        phones: list[str] = []
        tones: list[int] = []
        word2ph: list[int] = []
        for word_phone_list in word_phones:
            count = 0
            for ph in word_phone_list:
                p, t = melo_text._refine_ph(ph)
                phones.append(p)
                tones.append(t)
                count += 1
            word2ph.append(count)
        phones = [melo_text._post_replace_ph(p) for p in phones]
        phones = [PAD] + phones + [PAD]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
        phone_ids, tone_ids, lang_ids = melo_text.cleaned_text_to_sequence(
            phones, tones, "EN"
        )
        # add_blank interleave (the reference process_text with add_blank).
        phones_blank = [PAD]
        tones_blank = [0]
        for p, t in zip(phones, tones):
            phones_blank.extend([p, PAD])
            tones_blank.extend([t, 0])
        ids_blank, tones_blank, lang_blank = melo_text.cleaned_text_to_sequence(
            phones_blank, tones_blank, "EN"
        )
        n_phones = len(ids_blank)
        # ja_bert features (the 1024-dim bert input stays zeros, exactly
        # like the shipped _prepare_inputs).
        w2ph = [p * 2 for p in word2ph]
        w2ph[0] += 1
        # One BERT token per word2ph entry (CLS + one per word + SEP).
        n_words = len(word_phones)
        bert_ids = [[1] + [5] * n_words + [2]]
        bert_mask = [[1] * (n_words + 2)]
        features = bert_features_for(
            w2ph, bert_ids, bert_mask, model.bert, add_blank=False
        )["features"]
        # The reference pads/truncates the features to the phone count.
        # The tiny BERT's hidden size stands in for the 768-dim input the
        # reference's ja_bert_proj takes; the Rust reads the ja width from
        # the config (default 768).
        ja_dim = CONFIG["bert"]["hidden_size"]
        ja = np.zeros((ja_dim, n_phones), dtype=np.float32)
        width = min(features.shape[1], n_phones)
        ja[:, :width] = np.array(features)[:, :width]

        sid = 1
        mx.random.seed(E2E_SEED)
        audio = model.infer(
            mx.array([ids_blank], dtype=mx.int32),
            mx.array([n_phones]),
            mx.array([sid]),
            mx.array([tones_blank], dtype=mx.int32),
            mx.array([lang_blank], dtype=mx.int32),
            mx.zeros((1, 1024, n_phones)),
            ja_bert=mx.array(ja[None, :, :]),
            noise_scale=0.1,
            length_scale=1.0,
        )
        scenarios.append(
            {
                "name": name,
                "text": text,
                "norm_text": norm_text,
                "word_phones": word_phones,
                "phones": phones,
                "tones": tones,
                "word2ph": word2ph,
                "phones_blank": phones_blank,
                "phone_ids": ids_blank,
                "tone_ids": tones_blank,
                "lang_ids": lang_blank,
                "sid": sid,
                "ja_bert": floats(ja),
                "audio": floats(audio),
            }
        )

    generate_dir = OUT / "generate"
    generate_dir.mkdir(parents=True, exist_ok=True)
    (generate_dir / "traces.json").write_text(
        json.dumps({"scenarios": scenarios}, indent=1)
    )
    (generate_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    # ---- stage traces (the infer internals for the base scenario) ---------
    scenario = scenarios[0]
    ids = scenario["phone_ids"]
    n_phones = len(ids)
    sid = mx.array([scenario["sid"]])
    x_lengths = mx.array([n_phones])
    tone = mx.array([scenario["tone_ids"]], dtype=mx.int32)
    language = mx.array([scenario["lang_ids"]], dtype=mx.int32)
    ja_dim = CONFIG["bert"]["hidden_size"]
    ja = mx.array(
        np.array(scenario["ja_bert"], dtype=np.float32).reshape(ja_dim, n_phones)[None, :, :]
    )
    g = mx.expand_dims(model.emb_g(sid), -1)
    x, m_p, logs_p, x_mask = model.enc_p(
        mx.array([ids], dtype=mx.int32), x_lengths, tone, language,
        mx.zeros((1, 1024, n_phones)), ja_bert=ja, g=g,
    )
    # Intermediate encoder traces: the pre-encoder embedding sum and the
    # output after every transformer layer.
    enc = model.enc_p.encoder
    x_emb = model.enc_p.emb(mx.array([ids], dtype=mx.int32)).transpose(0, 2, 1)
    tone_emb = model.enc_p.tone_emb(tone).transpose(0, 2, 1)
    lang_emb = model.enc_p.language_emb(language).transpose(0, 2, 1)
    bert_emb = model.enc_p.bert_proj(mx.zeros((1, 1024, n_phones)))
    bert_emb = bert_emb + model.enc_p.ja_bert_proj(ja)
    enc_in = x_emb + tone_emb + lang_emb + bert_emb
    attn_mask_enc = mx.expand_dims(x_mask, 2) * mx.expand_dims(x_mask, -1)
    enc_layers = []
    cur = enc_in * x_mask
    for i in range(enc.n_layers):
        if i == enc.cond_layer_idx and enc.gin_channels > 0:
            g_proj = enc.spk_emb_linear(g.transpose(0, 2, 1)).transpose(0, 2, 1)
            cur = (cur + g_proj) * x_mask
        y = enc.attn_layers[i](cur, cur, attn_mask_enc)
        cur = enc.norm_layers_1[i](cur + y)
        y = enc.ffn_layers[i](cur, x_mask)
        cur = enc.norm_layers_2[i](cur + y)
        enc_layers.append(cur)
    # Intermediate duration-predictor traces (the two conv+relu+norm
    # stages).
    dp = model.dp
    h = x + dp.cond(g)
    h = dp.conv_1(h * x_mask)
    h = nn.relu(h)
    h = h.transpose(0, 2, 1)
    h = dp.norm_1(h)
    h = h.transpose(0, 2, 1)
    dp_h1 = h
    h = dp.conv_2(h * x_mask)
    h = nn.relu(h)
    h = h.transpose(0, 2, 1)
    h = dp.norm_2(h)
    h = h.transpose(0, 2, 1)
    dp_h2 = h
    logw_dp = model.dp(x, x_mask, g=g)
    w = mx.exp(logw_dp) * x_mask
    w_ceil = mx.ceil(w)
    y_lengths = mx.clip(mx.sum(w_ceil, axis=(1, 2)), a_min=1, a_max=None).astype(mx.int32)
    y_mask = (mx.arange(int(y_lengths.max()))[None, :] < y_lengths[:, None]).astype(mx.float32)[:, None, :]
    attn_mask = x_mask[:, :, :, None] * y_mask[:, :, None, :]

    # _generate_path
    dur = w_ceil.squeeze(1)
    cum_dur = mx.cumsum(dur, axis=-1)
    cum_dur_shifted = mx.pad(cum_dur[:, :-1], [(0, 0), (1, 0)])
    t_y = attn_mask.shape[-1]
    y_pos = mx.arange(t_y)[None, :]
    start = cum_dur_shifted[:, :, None]
    end = cum_dur[:, :, None]
    path = ((y_pos[None, :, :] >= start) & (y_pos[None, :, :] < end)).astype(mx.float32)
    path = path[:, None, :, :]
    attn = path * attn_mask

    m_p_aligned = mx.matmul(m_p, attn.squeeze(1))
    logs_p_aligned = mx.matmul(logs_p, attn.squeeze(1))
    mx.random.seed(E2E_SEED)
    z_p = m_p_aligned + mx.random.normal(m_p_aligned.shape) * mx.exp(logs_p_aligned) * 0.1
    z = z_p
    # The latent after every reverse coupling step (the Flip entries in
    # the stack produce no separate trace).
    flow_z = []
    for layer in reversed(model.flow_layers):
        z = layer(z, y_mask, g=g, reverse=True)
        if not isinstance(layer, Flip):
            flow_z.append(z)
    # Re-run the first reverse step (flip into the last coupling layer)
    # with internals for bisection.
    first_coupling = None
    for layer in reversed(model.flow_layers):
        if not isinstance(layer, Flip):
            first_coupling = layer
            break
    step_flip = z_p[:, ::-1, :]
    x0_step = step_flip[:, : first_coupling.half_channels, :]
    h_step = first_coupling.pre(x0_step) * y_mask
    h_step = first_coupling.enc(h_step, y_mask, g=g)
    m_step = first_coupling.post(h_step) * y_mask
    mx.random.seed(E2E_SEED)
    audio = model.dec(z * y_mask, g=g)

    stage_case = {
        "name": scenario["name"],
        "input_ids": ids,
        "sid": scenario["sid"],
        "tone_ids": scenario["tone_ids"],
        "lang_ids": scenario["lang_ids"],
        "ja_bert": scenario["ja_bert"],
        "g": floats(g),
        "enc_in": floats(enc_in),
        "enc_layers": [floats(l) for l in enc_layers],
        "x": floats(x),
        "m_p": floats(m_p),
        "logs_p": floats(logs_p),
        "dp_h1": floats(dp_h1),
        "dp_h2": floats(dp_h2),
        "logw_dp": floats(logw_dp),
        "w_ceil": floats(w_ceil),
        "y_lengths": ints(y_lengths),
        "attn": floats(attn),
        "m_p_aligned": floats(m_p_aligned),
        "logs_p_aligned": floats(logs_p_aligned),
        "z_p": floats(z_p),
        "flow_z": [floats(v) for v in flow_z],
        "flow_step_flip": floats(step_flip),
        "flow_step_x0": floats(x0_step),
        "flow_step_h": floats(h_step),
        "flow_step_m": floats(m_step),
        "z": floats(z),
        "audio": floats(audio),
    }
    stages_dir = OUT / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(stages_dir / "model.safetensors"), dict(flat))
    (stages_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (stages_dir / "traces.json").write_text(json.dumps({"cases": [stage_case]}, indent=1))

    # ---- manifest ---------------------------------------------------------
    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_melotts_fixtures.py",
        "model_seed": MODEL_SEED,
        "e2e_seed": E2E_SEED,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


NORMALIZE_CASES = [
    ("plain", "Hello World."),
    ("mr", "Mr. Smith met Mrs. Jones and Dr. Who."),
    ("titles", "St. Peter followed Capt. Cook and Sgt. Pepper."),
    ("company", "They worked at Co. Ltd and Ltd companies."),
    ("military", "Maj. Gen. Smith met Lt. Col. Brown and Hon. Esq."),
    ("reverend", "Rev. Green spoke to the group."),
    ("fort", "The fort at Ft. Knox is well guarded."),
    ("comma number", "The population is 7,900,000 people."),
    ("decimal", "Pi is approximately 3.14."),
    ("integer", "There are 1200 students and 42 teachers."),
    ("zero", "There were 0 errors."),
    ("big", "It costs 1234567 dollars."),
    ("negative", "The value is -5 degrees."),
]

# Hand-written ARPAbet stand-ins for the g2p output (g2p_en and the CMU
# dict are not installed in this venv; the reference post-processing
# above runs for real on these phones).
G2P_CASES = [
    ("hello world", "hello world", [["HH", "AH0", "L", "OW1"], ["W", "ER0", "L", "D"]]),
    ("tts", "tts", [["T", "IY1", "T", "IY1", "EH1", "S"]]),
    (
        "cats and dogs",
        "cats and dogs",
        [["K", "AE1", "T", "S"], ["AE1", "N", "D"], ["D", "AO1", "G", "Z"]],
    ),
    (
        "hello.",
        "hello.",
        [["HH", "AH0", "L", "OW1"]],
    ),
]

E2E_CASES = [
    (
        "hello",
        "Hello world.",
        [["HH", "AH0", "L", "OW1"], ["W", "ER0", "L", "D"]],
    ),
    (
        "tts speaks",
        "The tts speaks.",
        [["DH", "AH0"], ["T", "IY1", "T", "IY1", "EH1", "S"], ["S", "P", "IY1", "K", "S"]],
    ),
]


if __name__ == "__main__":
    main()
