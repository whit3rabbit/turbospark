#!/usr/bin/env python3
"""Regenerate the kitten_tts family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/tts/models/kitten_tts, mlx 0.32.3). The reference modules run
directly against a tiny seeded model:

- preprocess.py `TextPreprocessor` (default flags), kitten_tts.py
  `ensure_punctuation` / `chunk_text` / `basic_english_tokenize` /
  `TextCleaner` on fixed inputs (the Rust port pins the same string
  pipeline);
- the espeak phonemizer (`phonemizer-fork` EspeakBackend en-us,
  preserve_punctuation, with_stress) on the fixed texts, with the espeak
  library located on disk (no network, no downloads);
- per-stage traces of the reference `Model.__call__` (ALBERT, duration
  predictor, alignment, prosody, text encoder, decoder, iSTFTNet
  generator) on pinned ids;
- end-to-end `Model.generate` scenarios (single chunk and the chunked
  crossfade path) including the tail treatment, with the vocoder's
  stochastic terms pinned by the MLX key sequence (`mx.random.seed`).

Weights are saved in the raw MLX parameter tree layout under the keys
`bert.*`, `bert_encoder.*`, `predictor.*`, `text_encoder.*`,
`decoder.*` (exactly what `tree_flatten` lists); every conv weight is
MLX order `[out, kernel, in]` and the Rust loader transposes.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_kitten_tts_fixtures.py

Outputs into crates/audio/testdata/tts_kitten_tts/.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLX_NUM_THREADS", "1")

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.utils
import numpy as np

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.tts.models.kitten_tts import istftnet
from mlx_audio.tts.models.kitten_tts.kitten_tts import (
    Model as KittenModel,
    ModelConfig,
    TextCleaner,
    basic_english_tokenize,
    chunk_text,
    ensure_punctuation,
)
from mlx_audio.tts.models.kitten_tts.preprocess import TextPreprocessor
from mlx_audio.tts.models.kitten_tts.quant import maybe_fake_quant

OUT = Path("../turbospark/crates/audio/testdata/tts_kitten_tts")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

MODEL_SEED = 11
E2E_SEED = 1234
VOICE_ROWS = 8
VOICE = "test-voice"

# Tiny geometry. style_dim stays 128: the reference Model.__call__ and
# decoder hard-code the 128-dim split of the 256-dim voice style vector.
CONFIG = {
    "model_type": "kitten_tts",
    "hidden_dim": 32,
    "max_conv_dim": 32,
    "max_dur": 8,
    "n_layer": 2,
    "n_mels": 80,
    "n_token": 120,
    "style_dim": 128,
    "text_encoder_kernel_size": 5,
    "asr_res_dim": 4,
    "decoder_out_dim": None,
    "sample_rate": 24000,
    "voices_path": "voices.safetensors",
    "speed_priors": {"test-voice": 1.25},
    "voice_aliases": {"kv": "test-voice"},
    "plbert": {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 24,
        "intermediate_size": 32,
        "max_position_embeddings": 128,
        "embedding_size": 8,
        "inner_group_num": 1,
        "num_hidden_groups": 1,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-12,
    },
    "istftnet": {
        "resblock_kernel_sizes": [3, 5, 7],
        "upsample_rates": [4, 4],
        "upsample_initial_channel": 32,
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_kernel_sizes": [8, 8],
        "gen_istft_n_fft": 32,
        "gen_istft_hop_size": 4,
    },
}

QUANT_MODULES = [
    "bert_encoder",
    "predictor.duration_proj",
    "decoder.generator.conv_post",
]


def floats(arr) -> list:
    # Seven decimals keep every f32 value within one ULP of the stored
    # magnitude while roughly halving the JSON size.
    values = np.asarray(arr, dtype=np.float32).ravel().tolist()
    return [round(v, 7) for v in values]


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def find_espeak_library() -> str:
    import ctypes.util

    for candidate in [
        ctypes.util.find_library("espeak-ng"),
        ctypes.util.find_library("espeak"),
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/usr/local/lib/libespeak-ng.dylib",
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/x86_64-linux-gnu/libespeak.so.1",
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit(
        "espeak-ng library not found; install espeak-ng or extend the search list"
    )


def get_phonemizer():
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(find_espeak_library())
    return EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)


def build_model():
    model_config = {k: v for k, v in CONFIG.items() if k != "model_type"}
    config = ModelConfig(**model_config)
    model = KittenModel(config)
    model.eval()
    mx.random.seed(MODEL_SEED)

    def fill(weight):
        return mx.random.uniform(low=-0.03, high=0.03, shape=weight.shape)

    flat = dict(mlx.utils.tree_flatten(model.parameters()))
    for name, weight in list(flat.items()):
        if weight.dtype not in (mx.float32, mx.float16):
            continue
        if name.endswith("weight_g"):
            flat[name] = mx.ones(weight.shape)
        elif re.search(r"LayerNorm\.weight$", name) or re.search(
            r"\.cnn\.\d+\.1\.weight$", name
        ):
            flat[name] = mx.ones(weight.shape)
        elif re.search(r"LayerNorm\.bias$", name) or re.search(
            r"\.cnn\.\d+\.1\.bias$", name
        ):
            flat[name] = mx.zeros(weight.shape)
        elif re.search(r"alpha1_\d$", name) or re.search(r"alpha2_\d$", name):
            flat[name] = mx.ones(weight.shape)
        else:
            flat[name] = fill(weight)
    model.load_weights(list(flat.items()), strict=True)
    return model, flat


def warmup(model):
    """One discarded call: MLX lazy init makes the first call differ from
    the steady state, so every pinned run goes through this first."""
    ids = mx.array([[0, 5, 9, 0]], dtype=mx.int32)
    mx.random.seed(99)
    model(ids, mx.zeros((1, 256)), 1.0)


def build_voices() -> np.ndarray:
    rng = np.random.RandomState(MODEL_SEED + 1)
    return (rng.standard_normal((VOICE_ROWS, 256)) * 0.4).astype(np.float32)


# ---------------------------------------------------------------------------
# Frontend cases
# ---------------------------------------------------------------------------

PREPROCESS_CASES = [
    ("plain", "Hello world."),
    ("integer", "There are 1200 students and 42 teachers."),
    ("comma number", "The population is 7,900,000 people."),
    ("negative", "Temperature dropped to -5 degrees overnight."),
    ("float", "Pi is approximately 3.14159."),
    ("float trailing zero", "The voltage is 1.50 volts."),
    ("leading decimal", "Add .5 teaspoons of salt and .25 cup of milk."),
    ("negative leading decimal", "A -.05 correction was applied."),
    ("zero", "There were 0 errors and 0.0 warnings."),
    ("scientific", "Learning rate is 1e-4, weight decay 1e-5."),
    ("scientific capital", "Avogadro's number is 6.022E23."),
    ("scale suffix B", "We trained a 7B parameter model and a 13B variant."),
    ("scale suffix M", "The 340M model beat the 7B on MMLU."),
    ("percent", "Inflation rose by 3.5% last quarter."),
    ("negative percent", "Stocks fell -2% today."),
    ("ordinal", "She finished 1st, he came 2nd, I was 3rd."),
    ("ordinal 21st", "It's the 21st century and the 100th anniversary."),
    ("ordinal 42nd", "He ran his 42nd marathon."),
    ("fraction half", "Cut the recipe in 1/2."),
    ("fraction quarters", "Add 3/4 cup of sugar and 1/4 teaspoon of salt."),
    ("fraction thirds", "The team completed 2/3 of the project."),
    ("time 12h", "The meeting starts at 3:30pm."),
    ("time 24h", "Departure at 14:00."),
    ("time oh", "Alarm set for 9:05 AM."),
    ("decade", "The 80s music scene was iconic. She loved 1990s grunge."),
    ("range", "Read pages 10-20 for homework. The war lasted 2020-2024."),
    ("model name", "gpt-3 is pretty sick. They upgraded to GPT-3.5 last month."),
    ("model version", "Requires Python-3.10 or higher. Both CUDA-11 and CUDA-12 work."),
    ("unit distance", "The trail is 42km long."),
    ("unit temperature", "Water boils at 100C today."),
    ("unit data", "Download the 2.5GB model file. The CPU runs at 3.6GHz."),
    ("unit latency", "Average latency is 12ms."),
    ("currency dollar", "A coffee costs $4.99 here. It cost $2 last year."),
    ("currency euro", "Rent is 1,200 euros per month."),
    ("currency scale", "The salary was $85K per year."),
    ("html", "<b>Hello</b> World! It's a great day."),
    ("url", "Visit https://example.com/x or email hello@example.com."),
    ("contractions", "I don't know, won't you help? They've already left."),
    ("contractions 2", "Ain't no mountain high enough. Let's go!"),
    ("score", "The final score was 3:0."),
    ("hyphen vs minus", "On a scale of -10 to 10, she rated it 8."),
    ("mixed", "We raised $2.5M in seed funding and are growing 20% monthly."),
    ("whitespace", "  multiple   spaces\tand newlines\nbecome one  "),
    ("unicode punctuation", "Wait -- what? Really!! OK."),
]

CHUNK_CASES = [
    ("short", "Hello world"),
    ("no punctuation", "Hello world"),
    ("question", "What is this"),
    ("two sentences", "First one here. Second one follows! Third?"),
    ("long sentence", " ".join(["word"] * 150)),
]

TOKENIZE_CASES = [
    "həlˈoʊ wˈɜːld, ðɪs ɪz ɐ tˈɛst.",
    "ðɛɹˌɑːɹ wˈʌn θˈaʊzənd tˈuːhˈʌndɹɪd stˈuːdənts.",
    "aɪ lˈʌv kˈoʊfsi.",  # includes characters outside the symbol table
]

E2E_TEXTS = [
    ("hello", "Hello world, this is a test."),
    ("numbers", "There are 1200 students, and it costs $4.99."),
]

CHUNKED_TEXT = (
    "Kitten TTS is a small model. It runs fast on a laptop! "
    "The decoder uses an inverse short time Fourier transform. "
    "Weights are quantized from an ONNX export, which keeps the model "
    "compact, and the voices ship as a packed style table. "
    "This sentence pushes the text past the four hundred character chunk "
    "limit, so the reference generate loop splits it into pieces, "
    "crossfades the audio boundaries, and fades the tail out. "
    "The end result should still sound smooth."
)


# ---------------------------------------------------------------------------
# Stage traces (instrumented copy of Model.__call__)
# ---------------------------------------------------------------------------


def stage_traces(model, input_ids, ref_s, speed, seed, strict=True):
    seq = input_ids.shape[-1]
    input_lengths = mx.array([seq])
    text_mask = mx.arange(seq)[None, ...]
    text_mask = mx.repeat(text_mask, input_lengths.shape[0], axis=0).astype(
        input_lengths.dtype
    )
    text_mask = text_mask + 1 > input_lengths[:, None]
    attention = (~text_mask).astype(mx.int32)
    bert_out, _ = model.bert(input_ids, attention_mask=attention)
    d_en = model.bert_encoder(bert_out).transpose(0, 2, 1)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration_summed = mx.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = mx.clip(mx.round(duration_summed), a_min=1, a_max=None).astype(mx.int32)[0]
    indices = mx.concatenate(
        [mx.repeat(mx.array(i), int(n)) for i, n in enumerate(pred_dur)]
    )
    pred_aln_trg = mx.zeros((input_ids.shape[1], indices.shape[0]))
    pred_aln_trg[indices, mx.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg[None, :]
    en = d.transpose(0, 2, 1) @ pred_aln_trg
    f0_pred, n_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg

    # Decoder internals (mirrors KittenDecoder.__call__). The instrumented
    # path is the only path here: calling model.decoder separately would
    # draw the harmonic-source randoms twice and the two audios would be
    # pinned to different draws.
    dec = model.decoder
    f0_c = dec.F0_conv(f0_pred[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    n_c = dec.N_conv(n_pred[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    xd = mx.concatenate([asr, f0_c, n_c], axis=1)
    encode_out = dec.encode(xd, ref_s[:, :128])
    asr_res = dec.asr_res[0](asr.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    blocks = []
    res = True
    cur = encode_out
    for block in dec.decode:
        if res:
            cur = mx.concatenate([cur, asr_res, f0_c, n_c], axis=1)
        cur = block(cur, ref_s[:, :128])
        blocks.append(cur)
        if getattr(block, "upsample_type", "none") != "none":
            res = False

    # Generator internals (mirrors Generator.__call__; the harmonic STFT
    # feeds every noise conv, so it is computed once up front).
    gen = dec.generator
    f0_up = gen.f0_upsamp(f0_pred[:, None].transpose(0, 2, 1))
    har_source, _, _ = gen.m_source(f0_up)
    har_s = mx.squeeze(har_source.transpose(0, 2, 1), axis=1)
    mag, phase = gen.stft.transform(har_s)
    har = mx.concatenate([mag, phase], axis=1).swapaxes(2, 1)
    xg = cur
    ups = []
    for i in range(gen.num_upsamples):
        xg = istftnet.leaky_relu(xg, negative_slope=0.1)
        x_source = gen.noise_convs[i](har)
        x_source = gen.noise_res[i](x_source.swapaxes(2, 1), ref_s[:, :128])
        xg = gen.ups[i](xg.swapaxes(2, 1), mx.conv_transpose1d).swapaxes(2, 1)
        ups.append(xg)
        if i == gen.num_upsamples - 1:
            xg = gen.reflection_pad(xg)
        xg = xg + x_source
        xs = None
        for j in range(gen.num_kernels):
            blk = gen.resblocks[i * gen.num_kernels + j](xg, ref_s[:, :128])
            xs = blk if xs is None else xs + blk
        xg = xs / gen.num_kernels
    xg = istftnet.leaky_relu(xg, negative_slope=0.01)
    post = gen.conv_post(xg.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    spec = mx.exp(post[:, : gen.post_n_fft // 2 + 1, :])
    phase_pre = post[:, gen.post_n_fft // 2 + 1 :, :]
    audio_from_stages = gen.stft.inverse(spec, mx.sin(phase_pre))

    # The reference call consumes the same random stream; replay it.
    mx.random.seed(seed)
    reference_audio = model(input_ids, ref_s, speed)
    # MLX fuses lazily, so the two call patterns round differently at the
    # ULP level; the parity gate is the Rust fixture comparison.
    scale = max(1.0, float(np.abs(np.array(reference_audio)).max()))
    diff = float(
        np.abs(
            np.array(audio_from_stages).reshape(-1) - np.array(reference_audio).reshape(-1)
        ).max()
    )
    if strict:
        assert diff <= 1e-4 * scale, f"instrumented stages diverge: {diff} (scale {scale})"

    return {
        "input_ids": ints(input_ids),
        "ref_s": floats(ref_s),
        "speed": float(speed),
        "bert_out": floats(bert_out),
        "d_en": floats(d_en),
        "duration_logits": floats(duration),
        "durations_raw": floats(duration_summed),
        "pred_dur": ints(pred_dur),
        "en": floats(en),
        "f0": floats(f0_pred),
        "n": floats(n_pred),
        "t_en": floats(t_en),
        "asr": floats(asr),
        "decoder": {
            "f0_conv": floats(f0_c),
            "n_conv": floats(n_c),
            "encode": floats(encode_out),
            "asr_res": floats(asr_res),
            "blocks": [floats(b) for b in blocks],
            "gen_x": floats(cur),
            "f0_up": floats(f0_up),
            "har_source": floats(har_source),
            "stft_mag": floats(mag),
            "stft_phase": floats(phase),
            "ups": [floats(u) for u in ups],
            "conv_post": floats(post),
            "spec": floats(spec),
            "phase_pre_sin": floats(phase_pre),
            "audio": floats(audio_from_stages),
        },
        "audio": floats(reference_audio),
    }


def phoneme_inputs(phonemizer, cleaner, chunk, voices):
    """The reference `_prepare_inputs` text side, pinned per chunk."""
    tokens = " ".join(basic_english_tokenize(phonemizer.phonemize([chunk])[0]))
    ids = [0] + cleaner(tokens) + [0]
    voice_row = min(len(chunk), voices.shape[0] - 1)
    return {
        "chunk": chunk,
        "phonemes": phonemizer.phonemize([chunk])[0],
        "token_string": tokens,
        "input_ids": ids,
        "voice_row": int(voice_row),
        "ref_s": floats(voices[voice_row]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, flat = build_model()
    warmup(model)
    voices = build_voices()
    model.voices = {VOICE: voices}
    phonemizer = get_phonemizer()
    cleaner = TextCleaner()
    preprocessor = TextPreprocessor()

    # ---- frontend traces --------------------------------------------------
    symbols = list(cleaner.word_index_dictionary.keys())
    phonemize_cases = []
    for _, text in E2E_TEXTS:
        phonemize_cases.append({"text": text, "phonemes": phonemizer.phonemize([text])[0]})
    cleaner_cases = []
    for phonemes in [c["phonemes"] for c in phonemize_cases] + TOKENIZE_CASES:
        tokens = " ".join(basic_english_tokenize(phonemes))
        cleaner_cases.append(
            {"phonemes": phonemes, "token_string": tokens, "ids": cleaner(tokens)}
        )
    frontend = {
        "reference_commit": REFERENCE_COMMIT,
        "symbols": symbols,
        "phonemize_cases": phonemize_cases,
        "cleaner_cases": cleaner_cases,
        "preprocess_cases": [
            {"name": name, "input": text, "output": preprocessor(text)}
            for name, text in PREPROCESS_CASES
        ],
        "chunk_cases": [
            {
                "name": name,
                "input": text,
                "punctuated": ensure_punctuation(text),
                "chunks": chunk_text(text, max_len=400),
            }
            for name, text in CHUNK_CASES
        ],
        "tokenize_cases": [
            {"input": text, "tokens": basic_english_tokenize(text)}
            for text in TOKENIZE_CASES
        ],
    }
    frontend_dir = OUT / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    (frontend_dir / "traces.json").write_text(json.dumps(frontend, indent=1))

    # ---- stage traces -----------------------------------------------------
    ids = mx.array([[0, 5, 9, 3, 42, 7, 88, 0]], dtype=mx.int32)
    cases = []
    for label, case_ref_s, speed in [
        ("base", mx.array(voices[3:4]), 1.0),
        ("speed", mx.array(voices[3:4]), 1.4),
        ("style", mx.array(voices[7:8]), 1.0),
    ]:
        mx.random.seed(E2E_SEED)
        cases.append(stage_traces(model, ids, case_ref_s, speed, E2E_SEED))
        cases[-1]["case"] = label

    stages_dir = OUT / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(stages_dir / "model.safetensors"), dict(flat))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "voices.safetensors"
        mx.save_safetensors(str(path), {"voices": mx.array(voices)})
        (stages_dir / "voices.safetensors").write_bytes(path.read_bytes())
    (stages_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (stages_dir / "traces.json").write_text(json.dumps({"cases": cases}, indent=1))

    # ---- quant parity -----------------------------------------------------
    rng = np.random.RandomState(5)
    quant_in = (rng.standard_normal(64) * 2.5).astype(np.float32)
    quant_in[0] = 0.0
    quant_in[1] = -0.0
    quant_in[2:6] = 0.0
    quant_in[7] = 127.5
    quant_in[8] = -300.0
    quant_in[9] = 300.0
    quant = {
        "input": floats(quant_in),
        "output": floats(maybe_fake_quant(mx.array(quant_in), True)),
    }

    # ---- end-to-end scenarios --------------------------------------------
    scenarios = []
    for name, text in E2E_TEXTS:
        # The numbers scenario goes through the "kv" alias; the config's
        # speed prior for test-voice applies in both paths.
        voice = "kv" if name == "numbers" else VOICE
        effective_speed = 1.0 * CONFIG["speed_priors"]["test-voice"]
        mx.random.seed(E2E_SEED)
        results = list(model.generate(text, voice=voice, speed=1.0))
        assert len(results) == 1
        mx.random.seed(E2E_SEED)
        chunk = ensure_punctuation(text.strip())
        prepared = phoneme_inputs(phonemizer, cleaner, chunk, voices)
        ids_mx = mx.array([prepared["input_ids"]], dtype=mx.int32)
        ref_mx = mx.array(voices[prepared["voice_row"] : prepared["voice_row"] + 1])
        raw = model(ids_mx, ref_mx, effective_speed)
        pred_dur = model(ids_mx, ref_mx, effective_speed, return_output=True).pred_dur
        scenarios.append(
            {
                "name": name,
                "text": text,
                "voice": voice,
                **prepared,
                "pred_dur": ints(pred_dur),
                "raw_audio": floats(raw),
                "audio": floats(results[0].audio),
            }
        )

    # Chunked crossfade scenario through the reference generate loop.
    # Yields N results for N chunks: the first N-1 are the crossfaded
    # merges of chunk i into i+1, the last is the tail-treated final chunk.
    mx.random.seed(E2E_SEED)
    results = list(model.generate(CHUNKED_TEXT, voice=VOICE, speed=1.0))
    chunks = chunk_text(CHUNKED_TEXT.strip(), max_len=400)
    assert len(results) == len(chunks) >= 3
    segments = []
    for chunk_index, chunk in enumerate(chunks):
        prepared = phoneme_inputs(phonemizer, cleaner, chunk, voices)
        ids_mx = mx.array([prepared["input_ids"]], dtype=mx.int32)
        ref_mx = mx.array(voices[prepared["voice_row"] : prepared["voice_row"] + 1])
        pred_dur = model(ids_mx, ref_mx, 1.0, return_output=True).pred_dur
        segments.append(
            {**prepared, "pred_dur": ints(pred_dur)}
        )
    scenarios.append(
        {
            "name": "chunked",
            "text": CHUNKED_TEXT,
            "chunks": segments,
            "crossfaded_segments": [floats(r.audio) for r in results[:-1]],
            "final_tail_audio": floats(results[-1].audio),
        }
    )

    generate_dir = OUT / "generate"
    generate_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "voices.safetensors"
        mx.save_safetensors(str(path), {"voices": mx.array(voices)})
        (generate_dir / "voices.safetensors").write_bytes(path.read_bytes())
    (generate_dir / "traces.json").write_text(
        json.dumps({"scenarios": scenarios, "quant": quant}, indent=1)
    )
    (generate_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

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
        "generator": "crates/audio/tools/gen_kitten_tts_fixtures.py",
        "model_seed": MODEL_SEED,
        "e2e_seed": E2E_SEED,
        "espeak_library": find_espeak_library(),
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
