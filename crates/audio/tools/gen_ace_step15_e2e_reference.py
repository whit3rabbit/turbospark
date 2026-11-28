#!/usr/bin/env python3
"""End-to-end text2music reference for the ACE-Step 1.5 Rust pipeline.

Reproduces upstream's one-request text2music path at the pinned commit on
the real checkpoint, CPU float32, from the same public pieces the Rust
port gates one by one:

1. prompts: upstream's own helpers (SFT_GEN_PROMPT, _build_metadata_dict,
   _format_lyrics), tokenized by the checkpoint tokenizer as upstream calls
   it (truncation at 256 / 2048, one request so no padding);
2. text: Qwen3-Embedding `last_hidden_state` for the caption, its
   `embed_tokens` for the lyrics;
3. conditioning: reference timbre `silence[:750]`, source latents
   `silence[:frames]`, all-ones chunk mask, frames
   `max(128, int(duration * 48000) // 1920)`;
4. sampling: the checkpoint's `generate_audio` (turbo, shift 3, ODE), its
   noise drawn by `prepare_noise(seed)` and saved for replay;
5. decode: diffusers AutoencoderOobleck untiled (bit-identical to the
   512 / 64 tiled decode), then divide by the peak when it exceeds 1.

    ../.venv-acestep/bin/python -I crates/audio/tools/gen_ace_step15_e2e_reference.py \
        ../ACE-Step-1.5 <checkpoint root> <out dir> [--dcw]

Without `--dcw` the sampler is the checkpoint's own modeling file (no
DCW). With `--dcw` it is upstream's acestep/models/turbo file, which
upstream's loader copies over the checkpoint's at load, run with its
default DCW (double, 0.05 / 0.02, haar; needs pytorch-wavelets 1.3.0).
No LM planner either way. Writes outside the repository.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REQUEST = dict(
    caption="orchestral strings, solo cello, slow and warm, classical",
    lyrics="[Instrumental]",
    language="en",
    bpm=72,
    key_scale="D minor",
    time_signature="4",
    duration=10.0,
    seed=1234,
    shift=3.0,
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def upstream_turbo(upstream: Path):
    """Upstream's own turbo modeling module (with DCW), import-wired."""
    dcw = load("gen_dcw", HERE / "gen_ace_step15_dcw_fixtures.py")
    dcw.load_dcw(upstream)
    common = upstream / "acestep" / "models" / "common"
    cfg = load("acestep.models.common.configuration_acestep_v15", common / "configuration_acestep_v15.py")
    sys.modules["configuration_acestep_v15"] = cfg
    stub = __import__("types").ModuleType("vector_quantize_pytorch")
    from vector_quantize_pytorch import ResidualFSQ  # noqa: F401  (real package, locked version)
    del stub
    return load("upstream_turbo_modeling", upstream / "acestep" / "models" / "turbo" / "modeling_acestep_v15_turbo.py")


def main(upstream: Path, root: Path, out: Path, use_dcw: bool) -> None:
    prompts = load("gen_prompt", HERE / "gen_ace_step15_prompt_fixtures.py")
    prompts.stub_modules(upstream)
    constants = sys.modules["acestep.constants"]
    handler = upstream / "acestep" / "core" / "generation" / "handler"
    meta = load("ace_metadata_utils", handler / "metadata_utils.py")
    prompt = load("ace_prompt_utils", handler / "prompt_utils.py")

    class H(meta.MetadataMixin, prompt.PromptMixin):
        pass

    h = H()
    r = REQUEST
    metas = h._parse_metas([h._build_metadata_dict(r["bpm"], r["key_scale"], r["time_signature"], r["duration"])])[0]
    text_prompt = constants.SFT_GEN_PROMPT.format(
        h._format_instruction(constants.TASK_INSTRUCTIONS["text2music"]), r["caption"], metas
    )
    lyric_prompt = h._format_lyrics(r["lyrics"], r["language"])

    from transformers import AutoModel, AutoTokenizer

    text_dir = root / "Qwen3-Embedding-0.6B"
    tok = AutoTokenizer.from_pretrained(str(text_dir))
    enc_model = AutoModel.from_pretrained(str(text_dir), torch_dtype=torch.float32).eval()
    cap = tok(text_prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt")
    lyr = tok(lyric_prompt, padding="longest", truncation=True, max_length=2048, return_tensors="pt")
    with torch.inference_mode():
        text_hidden = enc_model(input_ids=cap.input_ids).last_hidden_state
        lyric_hidden = enc_model.embed_tokens(lyr.input_ids)
    del enc_model

    dit_mod = load("gen_dit", HERE / "gen_ace_step15_dit_fixtures.py")
    dit_dir = root / "acestep-v15-turbo"
    if use_dcw:
        mod = upstream_turbo(upstream)
    else:
        _, mod = dit_mod.import_reference(dit_dir)
    model = mod.AceStepConditionGenerationModel.from_pretrained(str(dit_dir), torch_dtype=torch.float32).eval()
    dit_mod.check_reference(model)
    silence = torch.load(dit_dir / "silence_latent.pt", map_location="cpu", weights_only=True).transpose(1, 2).float()
    frames = max(128, int(r["duration"] * 48000) // 1920)
    src = silence[:, :frames, :]
    chunk = torch.ones(1, frames, src.shape[-1])
    with torch.inference_mode():
        noise = model.prepare_noise(torch.cat([src, chunk], dim=-1), r["seed"])
        result = model.generate_audio(
            text_hidden_states=text_hidden,
            text_attention_mask=cap.attention_mask.bool(),
            lyric_hidden_states=lyric_hidden,
            lyric_attention_mask=lyr.attention_mask.bool(),
            refer_audio_acoustic_hidden_states_packed=silence[:, :750, :],
            refer_audio_order_mask=torch.zeros(1, dtype=torch.long),
            src_latents=src,
            chunk_masks=chunk,
            is_covers=torch.zeros(1, dtype=torch.long),
            silence_latent=silence,
            seed=r["seed"],
            shift=r["shift"],
            infer_method="ode",
            audio_cover_strength=1.0,
            precomputed_lm_hints_25Hz=src,
            **(
                dict(dcw_enabled=True, dcw_mode="double", dcw_scaler=0.05, dcw_high_scaler=0.02, dcw_wavelet="haar")
                if use_dcw
                else {}
            ),
        )
    latents = result["target_latents"]
    del model

    from diffusers import AutoencoderOobleck

    vae = AutoencoderOobleck.from_pretrained(str(root / "vae"), torch_dtype=torch.float32).eval()
    with torch.inference_mode():
        wav = vae.decode(latents.transpose(1, 2).contiguous()).sample
    peak = wav.abs().amax(dim=[1, 2], keepdim=True)
    if torch.any(peak > 1.0):
        wav = wav / peak.clamp(min=1.0)

    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "caption_ids": cap.input_ids[0].numpy().astype(np.int32),
        "lyric_ids": lyr.input_ids[0].numpy().astype(np.int32),
        "noise": noise[0].numpy().astype(np.float32),
        "latents": latents[0].numpy().astype(np.float32),
        "audio": wav[0].numpy().astype(np.float32),
    }
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", np.ascontiguousarray(v))
    (out / "request.json").write_text(
        json.dumps(
            dict(
                r,
                frames=frames,
                caption_prompt=text_prompt,
                lyric_prompt=lyric_prompt,
                peak=float(peak.max()),
                dcw=use_dcw,
                files={f"{k}.npy": hashlib.sha256((out / f"{k}.npy").read_bytes()).hexdigest() for k in arrays},
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"frames {frames}, audio {tuple(arrays['audio'].shape)}, peak {float(peak.max()):.3f} -> {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dcw"]
    if len(args) != 3:
        sys.exit("usage: gen_ace_step15_e2e_reference.py <ACE-Step-1.5> <checkpoint root> <out dir> [--dcw]")
    main(Path(args[0]).resolve(), Path(args[1]), Path(args[2]), "--dcw" in sys.argv)
