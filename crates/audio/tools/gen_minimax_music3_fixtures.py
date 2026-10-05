#!/usr/bin/env python3
"""Regenerate the MiniMax Music 3 Rust parity fixtures.

Normative reference: mlx-audio at commit
feb25a37b07923bae556e59111995071d66afa0d (mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_minimax_music3_fixtures.py

Outputs into crates/speech/testdata/minimax_music3/:

- tiny_weights.safetensors + tiny_config.json
    A seeded `ModelConfig.tiny()` parameter tree in the converted MLX
    layout (conv weights [out, K, in]) plus its config.json.
- rng_fixtures.json
    threefry key/split chains, uniform draws (bit-exact targets), a
    global KeySequence chain, categorical samples with the uniform draw
    and a decision margin, and normal draws (erfinv quality targets).
- prompt_fixtures.json
    clean_caption / normalize_lyrics / assemble_prompt cases and tiny
    text-pair token ids.
- ar_trace_short.json
    Frame-by-frame AR replay for duration 0.08 (2 frames, seed 7):
    logits, guided masks, sampled codes, depth decoder step traces,
    feedback embeddings, frame hiddens. Cross-checked in-script against
    the real `generate_frame_hiddens`.
- ar_trace_long.json (+ ar_frame_hiddens_long.npy)
    201-frame AR run: per-frame codebooks (exact token-parity targets)
    and the stacked frame hiddens.
- flow_trace_short.json
    Per-chunk condition / noise / latents / vocoder wave and the final
    stereo waveform for the short scenario, with an in-script
    cross-check against `Model._run_flow`.
- flow_trace_long.json (+ long_*.npy)
    Two-chunk flow: per-chunk condition / noise / latents and waveform
    statistics plus 4096 evenly spaced spot samples of the final stereo
    waveform (the crop-and-stitch parity target).
- dit_op.json (+ dit_*.npy)
    DiT forward velocities on a 689-frame condition at fixed sigmas.
- official_tiny/, converted_plain/, converted_q8/
    An official-format modular tree plus mlx_audio.convert outputs
    (plain and affine 8-bit) for loader tests.
- loaded_plain.json / loaded_plain_wave.npy
    End-to-end waveform of the converted tree loaded through the real
    `mlx_audio.music.load` path.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. Large
tensors are little-endian float32 .npy files; small ones stay in JSON.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.music.models.minimax_music3.ar import (
    generate_frame_hiddens,
    lm_logits,
    qwen3_hidden,
)
from mlx_audio.music.models.minimax_music3.config import (
    AR_CFG_SCALE,
    AR_CFG_TOP_K,
    AR_SAMPLING_TOP_K,
    CHUNK_FRAMES,
    CHUNK_HOP,
    CROP_LEFT_LATENT,
    CROP_RIGHT_LATENT,
    DIT_CFG_SCALE,
    ModelConfig,
    OVERLAP_LATENT_LENGTH,
)
from mlx_audio.music.models.minimax_music3.minimax_music3 import (
    Model,
    _chunk_starts,
    _crop_waveform,
    _encode_tiny_text_pair,
)
from mlx_audio.music.models.minimax_music3.prompt import (
    assemble_prompt,
    clean_caption,
    normalize_lyrics,
)
from mlx_audio.music.models.minimax_music3.sampling import sample_top_k

OUT = Path(__file__).resolve().parents[1] / "testdata" / "minimax_music3"
WEIGHT_SEED = 1234
SHORT_SEED = 7
SHORT_TEXT = "Genre: acoustic pop. BPM: 96. Warm female vocal."
SHORT_LYRICS = "[verse]\nMorning light\n[chorus]\nSing with me"
SHORT_FRAMES = 2
SHORT_STEPS = 2
LONG_FRAMES = 201
LONG_STEPS = 2


def f32_list(array: mx.array) -> list[float]:
    return [float(v) for v in np.asarray(array.astype(mx.float32)).ravel()]


def ints(array: mx.array) -> list[int]:
    return [int(v) for v in np.asarray(array).ravel()]


def save_npy(path: Path, array: mx.array) -> dict:
    data = np.ascontiguousarray(np.asarray(array.astype(mx.float32)))
    with path.open("wb") as file:
        np.save(file, data, allow_pickle=False)
    return {"file": path.name, "shape": list(data.shape)}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def all_files() -> list[Path]:
    seen: set[Path] = set()
    for path in OUT.rglob("*"):
        if path.is_file():
            seen.add(path)
    # The manifest cannot contain its own digest; listing it anyway (it
    # happens when a previous run's manifest is still in the tree)
    # records a stale self-entry that never reproduces.
    seen.discard(OUT / "manifest.json")
    return sorted(seen)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = str(mx.default_device())
    print(f"device: {device}")

    # ------------------------------------------------------------------
    # Tiny model with a deterministic init.
    # ------------------------------------------------------------------
    mx.random.seed(WEIGHT_SEED)
    config = ModelConfig.tiny()
    model = Model(config)
    params = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), params)
    write_json(OUT / "tiny_config.json", config.to_dict())
    print(f"tiny params: {len(params)} tensors")

    record_rng_fixtures()
    record_prompt_fixtures(config)

    short_ar = record_ar_trace(model, config, SHORT_SEED, SHORT_FRAMES, full=True)
    write_json(OUT / "ar_trace_short.json", short_ar)

    long_ar = record_ar_trace(model, config, SHORT_SEED, LONG_FRAMES, full=False)
    write_json(OUT / "ar_trace_long.json", long_ar)

    fused = config.num_codebooks * config.hidden_size
    short_hiddens = mx.reshape(
        mx.array(short_ar["frame_hiddens_stacked"], dtype=mx.float32),
        [1, SHORT_FRAMES, fused],
    )
    short_flow = record_flow(model, config, short_hiddens, SHORT_STEPS, SHORT_SEED, "short")
    write_json(OUT / "flow_trace_short.json", short_flow)

    # The long AR run legitimately hits the end token early (27 frames
    # with these tiny weights), so anchor the multi-chunk flow fixture
    # on a deterministic input instead: the short-trace hiddens tiled
    # to 202 frames. The flow stage is sampling-free, so this input
    # pins two-chunk scheduling, overlap carry, and crop stitching.
    short_hiddens = mx.reshape(
        mx.array(short_ar["frame_hiddens_stacked"], dtype=mx.float32),
        [1, SHORT_FRAMES, fused],
    )
    tiled = mx.tile(short_hiddens, [1, (LONG_FRAMES + SHORT_FRAMES - 1) // SHORT_FRAMES, 1])
    long_hiddens = tiled[:, :LONG_FRAMES, :]
    save_npy(OUT / "long_flow_hiddens.npy", long_hiddens)
    long_flow = record_flow(model, config, long_hiddens, LONG_STEPS, SHORT_SEED, "long")
    write_json(OUT / "flow_trace_long.json", long_flow)

    write_json(OUT / "dit_op.json", record_dit(model, config, LONG_FRAMES))

    write_official_tree(OUT / "official_tiny", model, config)
    from mlx_audio.convert import convert

    convert(str(OUT / "official_tiny"), str(OUT / "converted_plain"), quantize=False)
    convert(
        str(OUT / "official_tiny"),
        str(OUT / "converted_q8"),
        quantize=True,
        q_bits=8,
        q_mode="affine",
    )
    # prepare_config keeps the official token constants, which do not
    # fit the tiny 512-entry vocab (embedding an out-of-range CFG id is
    # undefined in the reference). Re-align the converted tiny trees
    # with the tiny constants so the trees stay self-consistent; real
    # checkpoints already satisfy vocab > offset.
    for name in ("converted_plain", "converted_q8"):
        path = OUT / name / "config.json"
        value = json.loads(path.read_text(encoding="ascii"))
        value["audio_code_offset"] = config.audio_code_offset
        value["audio_end_token_id"] = config.audio_end_token_id
        value["audio_cfg_token_id"] = config.audio_cfg_token_id
        value["semantic_vocab_size"] = config.semantic_vocab_size
        path.write_text(json.dumps(value, indent=1, sort_keys=True), encoding="ascii")

    from mlx_audio.music import load

    loaded = load(str(OUT / "converted_plain"))
    result = next(
        loaded.generate(
            text="Warm acoustic pop",
            lyrics="[verse]\nMorning light",
            duration=SHORT_FRAMES / config.frame_rate,
            steps=SHORT_STEPS,
            seed=SHORT_SEED,
        )
    )
    wave = np.asarray(result.audio, dtype=np.float32)
    npy_info = save_npy(OUT / "loaded_plain_wave.npy", mx.array(wave))
    write_json(
        OUT / "loaded_plain.json",
        {
            "wave": npy_info,
            "samples": int(result.samples),
            "token_count": int(result.token_count),
            "sample_rate": int(result.sample_rate),
        },
    )

    write_json(
        OUT / "manifest.json",
        {
            "generator": Path(__file__).name,
            "mlx_audio_commit": "feb25a37b07923bae556e59111995071d66afa0d",
            "mlx_version": mx.__version__,
            "device": device,
            "weight_seed": WEIGHT_SEED,
            "short": {"seed": SHORT_SEED, "frames": SHORT_FRAMES, "steps": SHORT_STEPS},
            "long": {"seed": SHORT_SEED, "frames": LONG_FRAMES, "steps": LONG_STEPS},
            "files": [
                {
                    "path": p.relative_to(OUT).as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": sha256(p),
                }
                for p in all_files()
            ],
        },
    )
    print("fixtures written:", len(all_files()), "files")


def record_rng_fixtures() -> None:
    rng: dict[str, object] = {"seeds": [0, 1, 7, 42, 1234]}
    keys: dict[str, object] = {}
    chains: dict[str, object] = {}
    chain_uniforms: dict[str, object] = {}
    for seed in rng["seeds"]:  # type: ignore[union-attr]
        key = mx.random.key(seed)
        keys[str(seed)] = ints(key)
        current = key
        chain = []
        for _ in range(3):
            pair = mx.random.split(current)
            chain.append([ints(pair[0]), ints(pair[1])])
            chain_uniforms[f"chain_{seed}_{len(chain)}"] = {
                "first": f32_list(mx.random.uniform(0.0, 1.0, (4,), key=pair[0])),
                "second": f32_list(mx.random.uniform(0.0, 1.0, (4,), key=pair[1])),
            }
            current = pair[0]
        keys[f"{seed}_chain"] = chain
        chains[str(seed)] = chain
    rng["keys"] = keys
    rng["split_chains"] = chains
    rng["chain_uniforms"] = chain_uniforms

    flat: dict[str, object] = {}
    for index, shape in enumerate([(1,), (2,), (3,), (4,), (5,), (8,), (7, 3), (1, 16, 6)]):
        key = mx.random.key(1000 + index)
        flat[str(list(shape))] = {
            "uniform": f32_list(mx.random.uniform(0.0, 1.0, shape, key=key)),
            "normal": f32_list(mx.random.normal(shape, key=key)),
        }
    rng["flat"] = flat

    sequence: dict[str, object] = {}
    for seed in (7, 8, 1234):
        mx.random.seed(seed)
        sequence[str(seed)] = {
            "uniform": [f32_list(mx.random.uniform(0.0, 1.0, (4,))) for _ in range(3)],
            "normal": [f32_list(mx.random.normal((4,))) for _ in range(3)],
        }
    rng["global_sequence"] = sequence

    # Categorical draws with the exact uniform and a decision margin.
    categorical = []
    patterns = {
        "uniform32": [0.0] * 32,
        "spike512": [(-1e9 if (i % 7) else float(i)) for i in range(512)],
        "masked32": [(-1e9 if i < 20 else 0.1 * (i - 20)) for i in range(32)],
    }
    top_k = 8
    for name, logits in patterns.items():
        for seed in (0, 7, 42):
            key = mx.random.key(seed)
            draws = []
            for _ in range(16):
                values_np = np.asarray(
                    mx.where(
                        mx.isnan(mx.array([logits], dtype=mx.float32)),
                        mx.array(-1e9, dtype=mx.float32),
                        mx.array([logits], dtype=mx.float32),
                    ).astype(mx.float32)
                )[0]
                k = min(top_k, values_np.shape[-1])
                threshold = np.sort(values_np)[-k]
                masked = np.where(values_np < threshold, -1e9, values_np)
                weight = np.exp(masked.astype(np.float64) - masked.max())
                cdf = np.concatenate([[0.0], np.cumsum(weight)[:-1]])
                total = weight.sum()
                # Pure function of the key: the draw categorical consumes.
                u = float(
                    np.asarray(mx.random.uniform(0.0, 1.0, (1,), key=key))[0]
                )
                drawn, key = sample_top_k(
                    mx.array([logits], dtype=mx.float32), key, top_k
                )
                picked = int(np.asarray(drawn)[0])
                lower = cdf[picked]
                upper = cdf[picked + 1] if picked + 1 < len(cdf) else total
                scaled = u * total
                margin = min(scaled - lower, upper - scaled)
                draws.append(
                    {
                        "token": picked,
                        "uniform": u,
                        "margin": margin,
                        "total": total,
                    }
                )
            categorical.append(
                {
                    "pattern": name,
                    "logits": [float(v) for v in logits],
                    "seed": seed,
                    "top_k": top_k,
                    "draws": draws,
                }
            )
    rng["categorical"] = categorical
    write_json(OUT / "rng_fixtures.json", rng)


def record_prompt_fixtures(config: ModelConfig) -> None:
    caption_cases = [
        "Genre: acoustic pop. BPM: 96. Warm female vocal.",
        "<|genre: rock|> **Loud** guitars\n- drums\n\n\n### tempo",
        "A <|long special|> tag and a <tag>",
        "Line one\n\n\nLine two    with spaces\nbullet - point\n___",
    ]
    lyrics_cases = [
        "[verse]\nMorning light\n[chorus]\nSing with me",
        "[Verse 1] Hello ] there [ Chorus ] ^ tail",
    ]
    tiny_cases = []
    for text, lyrics in [
        (SHORT_TEXT, SHORT_LYRICS),
        ("ab", lyrics_cases[0]),
    ]:
        assembled = assemble_prompt(text, lyrics)
        pair = _encode_tiny_text_pair(assembled, config)
        tiny_cases.append(
            {
                "text": text,
                "lyrics": lyrics,
                "assembled": assembled,
                "conditional": ints(pair[0]),
                "unconditional": ints(pair[1]),
            }
        )
    write_json(
        OUT / "prompt_fixtures.json",
        {
            "captions": [
                {"input": case, "clean": clean_caption(case)} for case in caption_cases
            ],
            "lyrics": [
                {"input": case, "normalized": normalize_lyrics(case)}
                for case in lyrics_cases
            ],
            "tiny_text": {
                "audio_cfg_token_id": config.audio_cfg_token_id,
                "cases": tiny_cases,
            },
        },
    )


def guided_logits(logits: mx.array, allowed: mx.array) -> mx.array:
    conditional, unconditional = logits[:1], logits[1:2]
    guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
    k = min(AR_CFG_TOP_K, conditional.shape[-1])
    threshold = mx.min(mx.topk(conditional, k, axis=-1), axis=-1, keepdims=True)
    guided = mx.where(conditional < threshold, -1e9, guided)
    guided = mx.where(allowed[None, :], guided, -1e9)
    return guided


def record_ar_trace(
    model: Model, config: ModelConfig, seed: int, max_frames: int, full: bool
) -> dict:
    """Replay `generate_frame_hiddens` with per-step recording.

    The replay ends with an in-script assertion that its frame hiddens
    equal the reference function's output, so the trace is the real
    computation, not a transcription.
    """
    language_model = model.language_model
    depth = model.rvq_depth_decoder
    text_ids = _encode_tiny_text_pair(
        assemble_prompt(SHORT_TEXT, SHORT_LYRICS), config
    )

    mx.random.seed(seed)
    key = mx.random.key(seed)
    embeddings = language_model.model.embed_tokens(text_ids)
    hidden, cache = qwen3_hidden(language_model, embeddings)
    last_hidden = hidden[:, -1]

    trace: dict[str, object] = {
        "seed": seed,
        "max_frames": max_frames,
        "text_ids": [ints(row) for row in text_ids],
        "prefill": {
            "hidden": f32_list(hidden),
            "hidden_shape": list(hidden.shape),
            "logits": f32_list(lm_logits(language_model, last_hidden).astype(mx.float32)),
        },
        "frames": [],
        "ended": False,
    }
    token_ids = mx.arange(config.vocab_size)
    allowed = mx.logical_or(
        mx.logical_and(
            token_ids >= config.audio_code_offset,
            token_ids < config.audio_code_offset + config.semantic_vocab_size,
        ),
        token_ids == config.audio_end_token_id,
    )
    trace["allowed_hint"] = {
        "offset": config.audio_code_offset,
        "end": config.audio_end_token_id,
        "semantic": config.semantic_vocab_size,
    }

    frames = []
    depth_frames_recorded = 0
    for frame_index in range(max_frames + 1):
        key, subkey = mx.random.split(key)
        logits = lm_logits(language_model, last_hidden).astype(mx.float32)
        masked = mx.where(allowed[None, :], logits, -1e9)
        guided = guided_logits(masked, allowed)
        sampled, key_after = sample_top_k(guided, subkey, AR_SAMPLING_TOP_K)
        sampled_id = int(np.asarray(sampled)[0])

        record: dict[str, object] = {
            "frame_index": frame_index,
            "sampled_id": sampled_id,
        }
        if full:
            record["logits"] = f32_list(logits)
            record["guided"] = f32_list(guided)
            record["last_hidden"] = f32_list(last_hidden)
        if sampled_id == config.audio_end_token_id:
            record["ended"] = True
            trace["frames"].append(record)
            trace["ended"] = True
            break

        semantic_code = mx.concatenate([sampled, sampled], axis=0) - config.audio_code_offset
        sequence = [depth.projection(last_hidden)[:, None, :]]
        code_embedding = language_model.model.embed_tokens(
            semantic_code + config.audio_code_offset
        )
        sequence.append(depth.projection(code_embedding)[:, None, :])
        codes = [semantic_code]
        hidden_parts = []
        depth_steps = []
        rng_key = key_after
        for index in range(1, config.num_codebooks):
            depth_in = mx.concatenate(sequence, axis=1)
            depth_out = depth(depth_in)
            hidden_last = depth_out[:, -1]
            logits_d = depth.audio_heads[index - 1](hidden_last)
            conditional_d = logits_d[:1].astype(mx.float32)
            unconditional_d = logits_d[1:2].astype(mx.float32)
            guided_d = unconditional_d + (conditional_d - unconditional_d) * AR_CFG_SCALE
            sampled_d, rng_key = sample_top_k(guided_d, rng_key, AR_SAMPLING_TOP_K)
            if full and depth_frames_recorded < 2:
                depth_steps.append(
                    {
                        "index": index,
                        "input": f32_list(depth_in),
                        "input_shape": list(depth_in.shape),
                        "output": f32_list(depth_out),
                        "output_shape": list(depth_out.shape),
                        "head_logits": f32_list(logits_d.astype(mx.float32)),
                        "sampled": int(np.asarray(sampled_d)[0]),
                    }
                )
            code = mx.concatenate([sampled_d, sampled_d], axis=0)
            codes.append(code)
            hidden_parts.append(hidden_last[:1])
            if index < config.num_codebooks - 1:
                embedding = depth.audio_embeddings(
                    code + (index - 1) * config.audio_vocab_size
                )
                sequence.append(depth.projection(embedding)[:, None, :])
        depth_frames_recorded += 1
        if depth_steps:
            record["depth_steps"] = depth_steps

        frame_codes = mx.stack(codes, axis=1)
        depth_hidden = mx.concatenate(hidden_parts, axis=-1)
        frame_hidden = (
            mx.concatenate([last_hidden[:1], depth_hidden], axis=-1)
            if frame_index > 0
            else None
        )
        embeddings_f = language_model.model.embed_tokens(
            frame_codes[:, :1] + config.audio_code_offset
        )
        offsets = mx.arange(config.residual_codebooks) * config.audio_vocab_size
        residual = depth.audio_embeddings(frame_codes[:, 1:] + offsets[None, :]).sum(
            axis=1, keepdims=True
        )
        feedback = (embeddings_f + residual.astype(embeddings_f.dtype)) * (
            config.num_codebooks**-0.5
        )
        hidden, cache = qwen3_hidden(language_model, feedback, cache)
        last_hidden = hidden[:, -1]
        record["frame_codes"] = ints(frame_codes[0])
        if full:
            record["feedback"] = f32_list(feedback)
            if frame_hidden is not None:
                record["frame_hidden"] = f32_list(frame_hidden)
        trace["frames"].append(record)
        if frame_hidden is not None:
            frames.append(frame_hidden)
        if len(frames) >= max_frames:
            break

    stacked = mx.stack(frames, axis=1)
    if full:
        trace["frame_hiddens_stacked"] = f32_list(stacked)
        trace["frame_hiddens_shape"] = list(stacked.shape)
    else:
        trace["frame_hiddens_npy"] = save_npy(
            OUT / "ar_frame_hiddens_long.npy", stacked
        )

    reference = generate_frame_hiddens(
        model.language_model,
        model.rvq_depth_decoder,
        config,
        text_ids,
        max_frames=max_frames,
        seed=seed,
    )
    assert np.array_equal(
        np.asarray(stacked), np.asarray(reference)
    ), "instrumented AR replay diverged from generate_frame_hiddens"
    return trace


def record_flow(
    model: Model,
    config: ModelConfig,
    frame_hiddens: mx.array,
    steps: int,
    seed: int,
    tag: str,
) -> dict:
    """Replay `Model._run_flow` with per-chunk recording."""
    from mlx_audio.music.models.minimax_music3.euler import (
        denoise_chunk,
        make_sigma_schedule,
    )

    starts = _chunk_starts(frame_hiddens.shape[1])
    waves = []
    previous_latent = None
    previous_condition = None
    mx.random.seed(seed + 7)
    chunks = []
    for chunk_index, start in enumerate(starts):
        end = min(start + CHUNK_FRAMES, frame_hiddens.shape[1])
        condition = model.condition_encoder(frame_hiddens[:, start:end])
        noise = mx.random.normal((1, config.dit_in_channels, condition.shape[1])).astype(
            condition.dtype
        )
        chunk: dict[str, object] = {
            "index": chunk_index,
            "start": start,
            "end": end,
            "latent_length": int(condition.shape[1]),
        }
        if tag == "short":
            chunk["condition"] = f32_list(condition)
            chunk["noise"] = f32_list(noise)
        else:
            chunk["condition_npy"] = save_npy(
                OUT / f"long_condition_{chunk_index}.npy", condition
            )
            chunk["noise_npy"] = save_npy(
                OUT / f"long_noise_{chunk_index}.npy", noise
            )
        latents, cond_out = denoise_chunk(
            model.transformer,
            noise,
            condition,
            num_inference_steps=steps,
            guidance_scale=DIT_CFG_SCALE,
            previous_latent=previous_latent,
            previous_condition=previous_condition,
        )
        if tag == "short":
            chunk["latents"] = f32_list(latents)
            chunk["cond_out"] = f32_list(cond_out)
        else:
            chunk["latents_npy"] = save_npy(
                OUT / f"long_latents_{chunk_index}.npy", latents
            )
        carry_start = max(0, latents.shape[-1] - 2 * OVERLAP_LATENT_LENGTH)
        carry_end = max(carry_start, latents.shape[-1] - OVERLAP_LATENT_LENGTH)
        previous_latent = latents[..., carry_start:carry_end]
        previous_condition = cond_out[:, carry_start:carry_end]
        chunk["carry"] = {"start": int(carry_start), "end": int(carry_end)}
        wave = model.vocoder(latents)
        if tag == "short":
            chunk["wave"] = f32_list(wave)
        else:
            chunk["wave_stats"] = waveform_stats(wave)
        chunks.append(chunk)
        waves.append(wave)

    cropped = [_crop_waveform(wave, index, len(waves)) for index, wave in enumerate(waves)]
    audio = mx.concatenate(cropped, axis=-1)
    waveform = mx.clip(audio[0].transpose(1, 0).astype(mx.float32), -1.0, 1.0)
    result: dict[str, object] = {
        "steps": steps,
        "seed": seed,
        "sigma_schedule": [float(v) for v in make_sigma_schedule(steps)],
        "chunk_starts": [int(s) for s in starts],
        "crop_latents": {"left": CROP_LEFT_LATENT, "right": CROP_RIGHT_LATENT},
        "chunks": chunks,
        "waveform_shape": list(waveform.shape),
    }
    if tag == "short":
        result["waveform"] = f32_list(waveform)
    else:
        result["waveform_stats"] = waveform_stats(waveform)
        result["waveform_spots"] = {
            "count": 4096,
            "values": f32_list(
                waveform[:: max(1, waveform.shape[0] // 4096)]
            ),
            "stride": max(1, waveform.shape[0] // 4096),
        }

    # Cross-check against the reference pipeline.
    reference = model._run_flow(frame_hiddens, steps, seed)
    assert np.array_equal(
        np.asarray(audio), np.asarray(reference)
    ), "instrumented flow replay diverged from Model._run_flow"
    return result


def waveform_stats(wave: mx.array) -> dict[str, float]:
    data = np.asarray(wave.astype(mx.float32), dtype=np.float64)
    return {
        "count": int(data.size),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
    }


def record_dit(model: Model, config: ModelConfig, long_frames: int) -> dict:
    """DiT forward velocities at fixed sigma on a 689-frame condition."""
    # Rebuild the long chunk-0 condition from its fixture.
    condition = mx.array(
        np.load(OUT / "long_condition_0.npy", allow_pickle=False)
    )
    noise = mx.random.normal((1, config.dit_in_channels, condition.shape[1]))
    info: dict[str, object] = {
        "input_npy": save_npy(OUT / "dit_input_noise.npy", noise),
        "condition_npy": save_npy(OUT / "dit_condition.npy", condition),
    }
    for name, sigma in (("sigma_0", 0.0), ("sigma_half", 0.5), ("sigma_1", 1.0)):
        timestep = mx.full((1,), sigma, dtype=mx.float32)
        info[f"{name}_velocity_npy"] = save_npy(
            OUT / f"dit_velocity_{name}.npy", model.transformer(noise, timestep, condition)
        )
        info[f"{name}_uncond_velocity_npy"] = save_npy(
            OUT / f"dit_velocity_{name}_uncond.npy",
            model.transformer(noise, timestep, mx.zeros_like(condition)),
        )
    return info


def write_official_tree(source: Path, model: Model, config: ModelConfig) -> None:
    """Mirror the upstream test's official-tree writer."""
    source.mkdir(parents=True, exist_ok=True)
    (source / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxMusic3ModularPipeline",
                "_diffusers_version": "0.40.0.dev0",
            }
        ),
        encoding="ascii",
    )
    modules = {
        "language_model": model.language_model,
        "rvq_depth_decoder": model.rvq_depth_decoder,
        "condition_encoder": model.condition_encoder,
        "transformer": model.transformer,
        "vocoder": model.vocoder,
    }
    for name, module in modules.items():
        folder = source / name
        folder.mkdir(exist_ok=True)
        (folder / "config.json").write_text(
            json.dumps(component_configs(config)[name]), encoding="ascii"
        )
        official = {}
        for key, value in tree_flatten(module.parameters()):
            official_key = key
            if name == "transformer" and ".to_out.0." in official_key:
                official_key = official_key.replace(".to_out.0.", ".to_out.")
            official[official_key] = to_official_tensor(key, value)
        if name == "vocoder":
            weight = official.pop("conv_in.weight")
            norm = mx.sqrt(mx.sum(weight.astype(mx.float32) ** 2, axis=(1, 2)))
            official["conv_in.weight_v"] = weight
            official["conv_in.weight_g"] = norm.reshape(-1, 1, 1)
        mx.save_safetensors(str(folder / "diffusion_pytorch_model.safetensors"), official)


def component_configs(config: ModelConfig) -> dict[str, dict]:
    return {
        "language_model": {
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "max_position_embeddings": config.max_position_embeddings,
            "rms_norm_eps": config.rms_norm_eps,
            "tie_word_embeddings": config.tie_word_embeddings,
            "rope_parameters": {"rope_theta": config.rope_theta},
            # Extra beyond the real official config: the tiny token
            # contract, without which the dataclass defaults (151675...)
            # do not fit the 512-entry embedding table.
            "audio_code_offset": config.audio_code_offset,
            "audio_end_token_id": config.audio_end_token_id,
            "audio_cfg_token_id": config.audio_cfg_token_id,
            "semantic_vocab_size": config.semantic_vocab_size,
        },
        "rvq_depth_decoder": {
            "hidden_size": config.hidden_size,
            "num_layers": config.depth_num_layers,
            "num_attention_heads": config.depth_num_heads,
            "intermediate_size": config.depth_intermediate_size,
            "audio_vocab_size": config.audio_vocab_size,
            "num_codebooks": config.num_codebooks,
        },
        "condition_encoder": {
            "condition_hidden_dim": config.hidden_size,
            "num_condition_layers": config.num_condition_layers,
            "out_dim": config.condition_out_dim,
            "input_sampling_rate": config.input_sampling_rate,
            "input_hop_length": config.input_hop_length,
            "output_sampling_rate": config.output_sampling_rate,
            "output_hop_length": config.output_hop_length,
        },
        "transformer": {
            "in_channels": config.dit_in_channels,
            "condition_dim": config.condition_out_dim,
            "num_layers": config.dit_num_layers,
            "num_attention_heads": config.dit_num_heads,
            "attention_head_dim": config.dit_head_dim,
            "ff_inner_dim": config.dit_ff_inner_dim,
            "rotary_dim": config.dit_rotary_dim,
            "fourier_embedding_dim": config.dit_fourier_dim,
        },
        "vocoder": {
            "latent_channels": config.dit_in_channels,
            "decoder_input_dim": config.vocoder_input_dim,
            "decoder_hidden_dim": config.vocoder_hidden_dim,
            "upsampling_ratios": list(config.vocoder_upsampling_ratios),
            "sampling_rate": config.sample_rate,
        },
    }


def to_official_tensor(key: str, value: mx.array) -> mx.array:
    if value.ndim != 3 or not key.endswith(".weight"):
        return value
    if "conv_t" in key:
        return value.transpose(2, 0, 1)
    return value.transpose(0, 2, 1)


if __name__ == "__main__":
    main()
