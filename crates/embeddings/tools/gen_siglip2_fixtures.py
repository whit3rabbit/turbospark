#!/usr/bin/env python3
"""Regenerate the SigLIP2 NaFlex Rust parity fixtures.

Normative reference: Hugging Face transformers 4.57.1 (`Siglip2Model`,
`Siglip2Processor`, `Siglip2ImageProcessorFast`) with torch 2.14.1 and
torchvision 0.29.1. `AutoProcessor.from_pretrained` instantiates the Fast
(torchvision) image processor for this checkpoint because its
`preprocessor_config.json` names `Siglip2ImageProcessorFast`; the script
asserts that, so a run in an environment without torchvision (which falls
back to the PIL processor) fails instead of writing different goldens.

Checkpoint: google/siglip2-base-patch16-naflex at
b53b807d3a2d5e2b3911292f2d69e5341cdc064c, read from a local directory.

Outputs into crates/embeddings/testdata/siglip2/:

- tiny/: a seeded random-init Siglip2Model (2+2 layers, small dims, vocab
  128, 4x4 position grid, patch 4, max_num_patches 24) saved as
  config.json + model.safetensors + preprocessor_config.json, three
  synthetic source PNGs, and per-image / per-text goldens (pixel values,
  resized position embeddings, embeddings output, last hidden state,
  pooled output, normalized embeddings) plus the logits matrix.
- real/: synthetic source PNGs, the exact uint8 resize output of the real
  processor config as PNG, sampled pixel_values rows, spatial shapes and
  masks, and the real checkpoint's normalized image/text embeddings,
  token ids, and logits.
- manifest.json: generator, package versions, pinned revision, measured
  Fast-vs-slow processor divergence, and SHA-256 of every fixture file.

Usage (deterministic, no network):

    <venv>/bin/python -I crates/embeddings/tools/gen_siglip2_fixtures.py \
        --model-dir ~/models/embeddings/siglip2-base-patch16-naflex
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torchvision
import transformers
from PIL import Image
from torchvision.transforms.v2 import functional as TF
from transformers import (
    AutoProcessor,
    Siglip2Config,
    Siglip2ImageProcessor,
    Siglip2ImageProcessorFast,
    Siglip2Model,
)

REPO = "google/siglip2-base-patch16-naflex"
REVISION = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
OUT = Path(__file__).resolve().parent.parent / "testdata" / "siglip2"

# Texts for the real checkpoint. Mixed case on purpose: the reference
# processor does not lowercase, and the fixture pins that.
REAL_TEXTS = [
    "a smooth color gradient",
    "a black and white checkerboard",
    "A photo of a cat",
    "diagonal stripes " * 30,  # overlength: pins truncation to 63 + EOS
]


def synth_image(width: int, height: int, seed: int) -> np.ndarray:
    """Deterministic RGB8 pattern: gradients plus stripes and a checker so
    the antialiased resize has high frequencies to work on."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = 255.0 * x / max(width - 1, 1)
    g = 255.0 * y / max(height - 1, 1)
    period = 3 + seed % 5
    stripes = ((x + 2 * y) // period) % 2
    checker = ((x // (7 + seed)) + (y // (5 + seed))) % 2
    b = 40.0 + 150.0 * stripes + 60.0 * checker
    b += 20.0 * np.sin((x * 0.37 + y * 0.11) * (1 + seed))
    img = np.stack([r, g, np.clip(b, 0, 255)], axis=-1)
    return np.round(img).clip(0, 255).astype(np.uint8)


def save_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr, mode="RGB").save(path, format="PNG", optimize=True)


def f32(path: Path, arr) -> None:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    np.save(path, np.ascontiguousarray(arr, dtype="<f4"))


def i32(path: Path, arr) -> None:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    np.save(path, np.ascontiguousarray(arr, dtype="<i4"))


def patchify(img_chw: torch.Tensor, patch: int) -> torch.Tensor:
    """The Fast processor's own path: fused rescale+normalize then
    convert_image_to_patches. Used to assert the stored resize output
    reproduces the processor's pixel_values exactly."""
    c, h, w = img_chw.shape
    x = img_chw.to(torch.float32)
    mean = torch.tensor([0.5] * 3) * (1.0 / 0.00392156862745098)
    std = torch.tensor([0.5] * 3) * (1.0 / 0.00392156862745098)
    x = TF.normalize(x, mean.tolist(), std.tolist())
    x = x.reshape(c, h // patch, patch, w // patch, patch).permute(1, 3, 2, 4, 0)
    return x.reshape((h // patch) * (w // patch), -1)


def randomize(model: torch.nn.Module, seed: int) -> None:
    """Overwrites every parameter with seeded noise. HF init leaves
    LayerNorms at identity and biases at zero, which would hide a dropped
    bias or affine term."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            noise = torch.randn(p.shape, generator=gen, dtype=torch.float32)
            if "layer_norm" in name or "layernorm" in name:
                p.copy_((1.0 if name.endswith("weight") else 0.0) + 0.1 * noise)
            elif name == "logit_scale":
                p.fill_(1.3)
            elif name == "logit_bias":
                p.fill_(-2.5)
            elif ("embedding" in name and "patch_embedding" not in name) or name.endswith("probe"):
                p.copy_(0.5 * noise)
            else:
                fan_in = p.shape[-1] if p.ndim > 1 else 16
                p.copy_(noise / np.sqrt(fan_in) * (1.0 if p.ndim > 1 else 0.2))


def tiny(out: Path) -> dict:
    d = out / "tiny"
    d.mkdir(parents=True, exist_ok=True)
    config = Siglip2Config(
        text_config=dict(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=16,
            projection_size=48,
        ),
        vision_config=dict(
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_patches=16,
            patch_size=4,
        ),
    )
    model = Siglip2Model(config).eval()
    randomize(model, 1234)
    model.save_pretrained(d, safe_serialization=True)
    proc = Siglip2ImageProcessorFast(patch_size=4, max_num_patches=24)
    proc.save_pretrained(d)

    # (width, height): square (identity pos grid), tall, and very wide
    # (pos grid height downsampled 4 -> 1, width upsampled 4 -> 24).
    sizes = [("square", 16, 16), ("tall", 23, 37), ("wide", 200, 8)]
    meta = {"images": [], "texts": []}
    image_embeds = []
    for k, (name, w, h) in enumerate(sizes):
        arr = synth_image(w, h, k)
        save_png(d / f"{name}.png", arr)
        inputs = proc(images=[Image.fromarray(arr)], return_tensors="pt")
        pv = inputs["pixel_values"]
        mask = inputs["pixel_attention_mask"]
        shapes = inputs["spatial_shapes"]
        n = int(mask.sum())
        with torch.no_grad():
            vm = model.vision_model
            pos = vm.embeddings.position_embedding.weight.reshape(4, 4, -1)
            pos_resized = vm.embeddings.resize_positional_embeddings(pos, shapes, pv.shape[1])
            emb = vm.embeddings(pv, shapes)
            vo = vm(pixel_values=pv, attention_mask=mask, spatial_shapes=shapes)
        pooled = vo.pooler_output[0]
        image_embeds.append(pooled / pooled.norm())
        f32(d / f"{name}_pixel_values.npy", pv[0, :n])
        f32(d / f"{name}_pos_resized.npy", pos_resized[0, :n])
        f32(d / f"{name}_embeddings.npy", emb[0, :n])
        f32(d / f"{name}_last_hidden.npy", vo.last_hidden_state[0, :n])
        f32(d / f"{name}_pooled.npy", pooled)
        meta["images"].append(
            {
                "name": name,
                "width": w,
                "height": h,
                "spatial_shape": [int(shapes[0, 0]), int(shapes[0, 1])],
                "valid_patches": n,
                "padded_to": int(pv.shape[1]),
            }
        )

    # Ids: short with pad tail (pad 0, eos 1), full length, and one whose
    # last slot is a content token (pins last-position pooling).
    seqs = [
        [17, 42, 99, 5, 1] + [0] * 11,
        list(range(20, 35)) + [1],
        [3, 127, 64, 8, 8, 8, 21, 77, 90, 2, 11, 13, 50, 60, 70, 80],
    ]
    ids = torch.tensor(seqs, dtype=torch.long)
    with torch.no_grad():
        to = model.text_model(input_ids=ids)
        emb = model.text_model.embeddings(input_ids=ids)
        out = model(
            input_ids=ids,
            **proc(
                images=[Image.open(d / f"{n}.png").convert("RGB") for n, _, _ in sizes],
                return_tensors="pt",
            ),
        )
    i32(d / "text_ids.npy", ids)
    f32(d / "text_embeddings.npy", emb)
    f32(d / "text_last_hidden.npy", to.last_hidden_state)
    f32(d / "text_pooled.npy", to.pooler_output)
    f32(d / "text_embeds.npy", out.text_embeds)
    f32(d / "image_embeds.npy", out.image_embeds)
    f32(d / "logits_per_image.npy", out.logits_per_image)
    # Batched (padded, masked) run must equal the per-image runs.
    batched_vs_single = float((out.image_embeds - torch.stack(image_embeds)).abs().max())
    assert batched_vs_single < 1e-5, batched_vs_single
    meta["batched_vs_single_image_embeds_max_abs"] = batched_vs_single
    meta["texts"] = seqs
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def real(out: Path, model_dir: Path) -> dict:
    d = out / "real"
    d.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_dir)
    ip = processor.image_processor
    assert type(ip).__name__ == "Siglip2ImageProcessorFast", type(ip).__name__
    slow = Siglip2ImageProcessor.from_pretrained(model_dir)
    model = Siglip2Model.from_pretrained(model_dir, dtype=torch.float32).eval()
    attn_impl = model.config._attn_implementation

    # (name, width, height): square upscale 224 -> 256, landscape, tall,
    # tiny upscale, and an extreme wide strip (pos grid height 16 -> 5).
    sizes = [
        ("square224", 224, 224),
        ("land640x480", 640, 480),
        ("tall180x500", 180, 500),
        ("small50x37", 50, 37),
        ("wide1000x120", 1000, 120),
    ]
    meta = {"images": [], "texts": REAL_TEXTS, "attn_implementation": attn_impl}
    pils = []
    max_pixel_diff_slow = 0.0
    for k, (name, w, h) in enumerate(sizes):
        arr = synth_image(w, h, k + 3)
        save_png(d / f"{name}.png", arr)
        pil = Image.open(d / f"{name}.png").convert("RGB")
        pils.append(pil)
        inputs = processor(images=[pil], return_tensors="pt")
        pv, mask, shapes = (
            inputs["pixel_values"],
            inputs["pixel_attention_mask"],
            inputs["spatial_shapes"],
        )
        th, tw = int(shapes[0, 0]) * 16, int(shapes[0, 1]) * 16
        chw = torch.from_numpy(np.array(pil)).permute(2, 0, 1).contiguous()
        resized = TF.resize(
            chw, [th, tw], interpolation=TF.InterpolationMode.BILINEAR, antialias=True
        )
        n = int(mask.sum())
        assert torch.equal(patchify(resized, 16), pv[0, :n]), name
        save_png(d / f"{name}_resized.png", resized.permute(1, 2, 0).numpy())
        rows = sorted({0, 1, n // 2, n - 1})
        f32(d / f"{name}_pixel_rows.npy", pv[0, rows])
        sv = slow(images=[pil], return_tensors="pt")["pixel_values"]
        max_pixel_diff_slow = max(max_pixel_diff_slow, float((sv - pv).abs().max()))
        meta["images"].append(
            {
                "name": name,
                "width": w,
                "height": h,
                "target_hw": [th, tw],
                "spatial_shape": [int(shapes[0, 0]), int(shapes[0, 1])],
                "valid_patches": n,
                "mask_sum": int(mask.sum()),
                "pixel_rows": rows,
                "pixel_values_sum_f64": float(pv[0, :n].double().sum()),
            }
        )

    inputs = processor(text=REAL_TEXTS, images=pils, return_tensors="pt")
    assert "attention_mask" not in inputs, "processor now returns a text mask"
    with torch.no_grad():
        o = model(**inputs)
        slow_in = slow(images=pils, return_tensors="pt")
        so = model.get_image_features(**slow_in)
    slow_embeds = so / so.norm(dim=-1, keepdim=True)
    cos_slow = (slow_embeds * o.image_embeds).sum(-1)
    i32(d / "text_ids.npy", inputs["input_ids"])
    f32(d / "image_embeds.npy", o.image_embeds)
    f32(d / "text_embeds.npy", o.text_embeds)
    f32(d / "logits_per_image.npy", o.logits_per_image)
    meta["logit_scale"] = model.logit_scale.item()
    meta["logit_bias"] = model.logit_bias.item()
    meta["slow_vs_fast"] = {
        "pixel_values_max_abs": max_pixel_diff_slow,
        "image_embed_cosine_min": float(cos_slow.min()),
        "image_embed_max_abs": float((slow_embeds - o.image_embeds).abs().max()),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    tiny_meta = tiny(args.out)
    real_meta = real(args.out, args.model_dir.expanduser())
    files = sorted(
        p for p in args.out.rglob("*") if p.is_file() and p.name != "manifest.json"
    )
    manifest = {
        "generator": "crates/embeddings/tools/gen_siglip2_fixtures.py",
        "checkpoint": {"repo": REPO, "revision": REVISION},
        "packages": {
            "python": sys.version.split()[0],
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "safetensors": safetensors.__version__,
        },
        "image_processor": "Siglip2ImageProcessorFast (torchvision uint8 antialiased bilinear)",
        "tiny_batched_vs_single_image_embeds_max_abs": tiny_meta[
            "batched_vs_single_image_embeds_max_abs"
        ],
        "real_attn_implementation": real_meta["attn_implementation"],
        "slow_vs_fast": real_meta["slow_vs_fast"],
        "files": {
            str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(p.stat().st_size for p in files)
    print(f"wrote {len(files)} files, {total} bytes")
    print(json.dumps(real_meta["slow_vs_fast"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
