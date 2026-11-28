#!/usr/bin/env python3
"""Regenerate the ACE-Step 1.5 condition encoder and DiT parity fixtures.

Normative reference: `modeling_acestep_v15_turbo.py` and
`configuration_acestep_v15.py` from the checkpoint `ACE-Step/Ace-Step1.5`
at revision 19671f406d603126926c1b7e2adc169acbcade22 (the code the
checkpoint loads with trust_remote_code), run on the CPU in float32 with
the default SDPA attention.

The FSQ tokenizer uses `vector_quantize_pytorch.ResidualFSQ` 1.27.20
(upstream uv.lock).

Run with the ACE-Step venv pinned to upstream's uv.lock (transformers
4.57.6, diffusers 0.37.1, huggingface-hub 0.36.0, vector-quantize-pytorch
1.27.20; torch 2.15.0 CPU).
Upstream requires transformers <4.58: under 5.x `from_pretrained`
leaves every `Qwen3RotaryEmbedding.inv_freq` buffer at zero, which
silently disables RoPE in the reference. `check_reference` refuses that.

    ../.venv-acestep/bin/python -I \
        crates/audio/tools/gen_ace_step15_dit_fixtures.py <checkpoint>/acestep-v15-turbo

Outputs into crates/audio/testdata/ace_step15_dit/:

- tiny_config.json + tiny_weights.safetensors
    A seeded tiny AceStepConditionGenerationModel, weights stored in
    bf16 under the real checkpoint's tensor names. The reference runs on
    exactly those bf16 values widened to f32.
- inputs: text.npy, lyric.npy, refer.npy, x.npy, context.npy,
  src.npy, chunk.npy, silence.npy, text_nc.npy, noise.npy
- outputs: encoder.npy (packed encoder states), velocity.npy (one DiT
  forward at t=0.75), latents.npy (8-step turbo text2music, shift 3),
  tokens.npy + indices.npy (5 Hz FSQ tokenize of src, padded from
  silence), hints.npy (detokenized 25 Hz cover hints), encoder_nc.npy
  (packed states with the non-cover text), latents_cover.npy (8-step
  cover at strength 0.5: 4 steps on hints, 4 on silence context)
- manifest.json with SHA-256 digests.

With `--real <dit dir> <out dir>` it runs the real bf16 checkpoint in
f32 on short deterministic inputs and writes the same arrays to
<out dir>, outside the repository.

Deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
OUT = Path(__file__).resolve().parents[1] / "testdata" / "ace_step15_dit"
TINY = dict(
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    num_lyric_encoder_hidden_layers=2,
    num_timbre_encoder_hidden_layers=2,
    num_attention_pooler_hidden_layers=2,
    text_hidden_dim=16,
    timbre_hidden_dim=8,
    audio_acoustic_hidden_dim=8,
    in_channels=24,
    # A window shorter than the tiny sequences so sliding layers differ
    # from full layers.
    sliding_window=3,
    fsq_dim=32,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    patch_size=2,
    pool_window_size=5,
)
SEED = 1234
T_STEP = 0.75
COVER_STRENGTH = 0.5
VQ_VERSION = "1.27.20"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_reference(dit_dir: Path):
    from importlib.metadata import version

    if version("vector-quantize-pytorch") != VQ_VERSION:
        sys.exit(f"vector-quantize-pytorch {version('vector-quantize-pytorch')} != {VQ_VERSION}")
    cfg = load_module("configuration_acestep_v15", dit_dir / "configuration_acestep_v15.py")
    mod = load_module("modeling_acestep_v15_turbo", dit_dir / "modeling_acestep_v15_turbo.py")
    return cfg, mod


def check_reference(model) -> None:
    """Refuse a reference environment outside upstream's supported range
    or one whose rotary buffers were not initialized."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) >= (4, 58) or (major, minor) < (4, 51):
        sys.exit(f"transformers {transformers.__version__} is outside upstream's >=4.51,<4.58")
    for name, module in model.named_modules():
        inv = getattr(module, "inv_freq", None)
        if inv is not None and not bool((inv[1:] != 0).all()):
            sys.exit(f"{name}.inv_freq has zero entries: RoPE would be disabled")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(out: Path, name: str, t) -> str:
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy()
    np.save(out / f"{name}.npy", np.ascontiguousarray(t, dtype=np.float32))
    return f"{name}.npy"


def run(model, out: Path, inputs: dict) -> list[str]:
    """Encoder, one DiT forward, and the 8-step turbo loop."""
    files = [save(out, k, v[0]) for k, v in inputs.items()]
    with torch.inference_mode():
        enc, enc_mask = model.encoder(
            text_hidden_states=inputs["text"],
            text_attention_mask=torch.ones(inputs["text"].shape[:2]),
            lyric_hidden_states=inputs["lyric"],
            lyric_attention_mask=torch.ones(inputs["lyric"].shape[:2]),
            refer_audio_acoustic_hidden_states_packed=inputs["refer"],
            refer_audio_order_mask=torch.zeros(1, dtype=torch.long),
        )
        assert bool(enc_mask.all()), "fixtures must not contain padding"
        files.append(save(out, "encoder", enc[0]))
        velocity = model.decoder(
            hidden_states=inputs["x"],
            timestep=torch.tensor([T_STEP]),
            timestep_r=torch.tensor([T_STEP]),
            attention_mask=None,
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_mask,
            context_latents=inputs["context"],
            use_cache=False,
        )[0]
        files.append(save(out, "velocity", velocity[0]))

        src, chunk = inputs["src"], inputs["chunk"]
        noise = model.prepare_noise(torch.cat([src, chunk], dim=-1), SEED)
        files.append(save(out, "noise", noise[0]))
        result = model.generate_audio(
            text_hidden_states=inputs["text"],
            text_attention_mask=torch.ones(inputs["text"].shape[:2]),
            lyric_hidden_states=inputs["lyric"],
            lyric_attention_mask=torch.ones(inputs["lyric"].shape[:2]),
            refer_audio_acoustic_hidden_states_packed=inputs["refer"],
            refer_audio_order_mask=torch.zeros(1, dtype=torch.long),
            src_latents=src,
            chunk_masks=chunk,
            is_covers=torch.zeros(1, dtype=torch.long),
            silence_latent=src,
            seed=SEED,
            shift=3.0,
            infer_method="ode",
            audio_cover_strength=1.0,
            # Supplying hints skips the FSQ tokenizer; with is_covers=0
            # they are never selected, so the context is src + chunk.
            precomputed_lm_hints_25Hz=src,
        )
        files.append(save(out, "latents", result["target_latents"][0]))

        # Cover: FSQ round trip of the source (prepare_condition with
        # is_covers=1), then a strength-0.5 run that switches to the
        # silence context and the non-cover text halfway.
        frames = src.shape[1]
        silence = inputs["silence"]
        tokens, indices, _ = model.tokenize(src, silence, torch.ones(1, frames))
        files.append(save(out, "tokens", tokens[0]))
        np.save(out / "indices.npy", indices[0, :, 0].numpy().astype(np.int32))
        files.append("indices.npy")
        hints = model.detokenize(tokens)[:, :frames, :]
        files.append(save(out, "hints", hints[0]))
        enc_nc, _ = model.encoder(
            text_hidden_states=inputs["text_nc"],
            text_attention_mask=torch.ones(inputs["text_nc"].shape[:2]),
            lyric_hidden_states=inputs["lyric"],
            lyric_attention_mask=torch.ones(inputs["lyric"].shape[:2]),
            refer_audio_acoustic_hidden_states_packed=inputs["refer"],
            refer_audio_order_mask=torch.zeros(1, dtype=torch.long),
        )
        files.append(save(out, "encoder_nc", enc_nc[0]))
        cover = model.generate_audio(
            text_hidden_states=inputs["text"],
            text_attention_mask=torch.ones(inputs["text"].shape[:2]),
            lyric_hidden_states=inputs["lyric"],
            lyric_attention_mask=torch.ones(inputs["lyric"].shape[:2]),
            refer_audio_acoustic_hidden_states_packed=inputs["refer"],
            refer_audio_order_mask=torch.zeros(1, dtype=torch.long),
            src_latents=src,
            chunk_masks=chunk,
            is_covers=torch.ones(1, dtype=torch.long),
            silence_latent=silence,
            seed=SEED,
            shift=3.0,
            infer_method="ode",
            audio_cover_strength=COVER_STRENGTH,
            non_cover_text_hidden_states=inputs["text_nc"],
            non_cover_text_attention_mask=torch.ones(inputs["text_nc"].shape[:2]),
        )
        files.append(save(out, "latents_cover", cover["target_latents"][0]))
    return files


def seeded_state(model, gen: torch.Generator) -> dict:
    state = {}
    for name, value in model.state_dict().items():
        shape = value.shape
        if name.endswith("norm.weight"):
            t = 1.0 + 0.1 * torch.randn(shape, generator=gen)
        elif value.dim() >= 2 and not name.endswith(("scale_shift_table", "special_token", "special_tokens", "null_condition_emb")):
            fan_in = math.prod(shape[1:])
            t = torch.randn(shape, generator=gen) / math.sqrt(fan_in)
        elif value.dim() == 1:
            t = 0.1 * torch.randn(shape, generator=gen)
        else:
            t = 0.5 * torch.randn(shape, generator=gen)
        state[name] = t.to(torch.bfloat16).contiguous()
    return state


def tiny(dit_dir: Path) -> None:
    from safetensors.torch import save_file

    cfgm, mod = import_reference(dit_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = cfgm.AceStepConfig(**TINY)
    torch.manual_seed(7)
    model = mod.AceStepConditionGenerationModel(cfg).eval()
    check_reference(model)
    state = seeded_state(model, torch.Generator().manual_seed(11))
    model.load_state_dict({k: v.float() for k, v in state.items()})
    save_file(state, str(OUT / "tiny_weights.safetensors"))
    config = dict(
        TINY,
        layer_types=list(cfg.layer_types),
        use_sliding_window=True,
        fsq_input_levels=list(cfg.fsq_input_levels),
        fsq_input_num_quantizers=cfg.fsq_input_num_quantizers,
    )
    (OUT / "tiny_config.json").write_text(json.dumps(config, indent=2) + "\n")

    g = torch.Generator().manual_seed(13)
    a = TINY["audio_acoustic_hidden_dim"]
    # 13 frames: odd, so the patchify pad and the output crop are
    # exercised, and longer than the sliding window.
    inputs = {
        "text": torch.randn(1, 7, TINY["text_hidden_dim"], generator=g),
        "lyric": torch.randn(1, 11, TINY["text_hidden_dim"], generator=g),
        "refer": torch.randn(1, 9, TINY["timbre_hidden_dim"], generator=g),
        "x": torch.randn(1, 13, a, generator=g),
        "context": torch.randn(1, 13, 2 * a, generator=g),
        "src": torch.randn(1, 13, a, generator=g),
        "chunk": torch.ones(1, 13, a),
        # Longer than src: tokenize pads 13 -> 15 frames from its head.
        "silence": torch.randn(1, 20, a, generator=g),
        "text_nc": torch.randn(1, 5, TINY["text_hidden_dim"], generator=g),
    }
    files = ["tiny_config.json", "tiny_weights.safetensors"] + run(model, OUT, inputs)
    manifest = {
        "generator": "tools/gen_ace_step15_dit_fixtures.py",
        "reference": f"ACE-Step/Ace-Step1.5@{CHECKPOINT_REVISION} acestep-v15-turbo modeling code",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "attn_implementation": model.config._attn_implementation,
        "t_step": T_STEP,
        "seed": SEED,
        "cover_strength": COVER_STRENGTH,
        "vector_quantize_pytorch_version": VQ_VERSION,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(files) + 1} fixture files to {OUT}")


def real(dit_dir: Path, out: Path) -> None:
    cfgm, mod = import_reference(dit_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = mod.AceStepConditionGenerationModel.from_pretrained(str(dit_dir), torch_dtype=torch.float32).eval()
    check_reference(model)
    cfg = model.config
    silence = torch.load(dit_dir / "silence_latent.pt", map_location="cpu").float()
    if silence.shape[1] == cfg.audio_acoustic_hidden_dim:
        silence = silence.transpose(1, 2)
    g = torch.Generator().manual_seed(17)
    frames = 48  # about 2 s at 25 Hz; not a multiple of 5
    a = cfg.audio_acoustic_hidden_dim
    inputs = {
        "text": torch.randn(1, 12, cfg.text_hidden_dim, generator=g) * 0.5,
        "lyric": torch.randn(1, 20, cfg.text_hidden_dim, generator=g) * 0.05,
        "refer": silence[:, : cfg.timbre_fix_frame, :].clone(),
        "x": torch.randn(1, frames, a, generator=g),
        "context": torch.cat([silence[:, :frames, :], torch.ones(1, frames, a)], dim=-1),
        "src": silence[:, :frames, :].clone(),
        "chunk": torch.ones(1, frames, a),
        "silence": silence[:, : frames + 10, :].clone(),
        "text_nc": torch.randn(1, 9, cfg.text_hidden_dim, generator=g) * 0.5,
    }
    # Cover needs a source that is not silence so the quantizer sees
    # varied input; 48 frames also forces the 5-frame pad.
    inputs["src"] = torch.randn(1, frames, a, generator=g)
    files = run(model, out, inputs)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "generator": "tools/gen_ace_step15_dit_fixtures.py --real",
                "dit_dir": str(dit_dir),
                "torch_version": torch.__version__,
                "transformers_version": __import__("transformers").__version__,
                "t_step": T_STEP,
                "seed": SEED,
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
    elif len(sys.argv) == 2:
        tiny(Path(sys.argv[1]))
    else:
        sys.exit("usage: gen_ace_step15_dit_fixtures.py <dit dir> | --real <dit dir> <out dir>")
