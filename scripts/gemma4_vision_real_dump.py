#!/usr/bin/env python3
"""Run mlx-vlm's Gemma 4 vision tower on the REAL checkpoint and dump what the
Metal tower's parity test needs (`docs/GEMMA4_VISION_PHASE0.md`).

Downloads only the 358 vision tensors (about 1.1 GB) by HTTP range read from
the pinned checkpoint, never the 15 GB model. Then, on one deterministic
image:

- runs the tower and projector in FLOAT32 (the ground truth) and in BF16
  (how the reference normally runs), capturing every stage;
- writes the patch rows and positions in this port's layout, every weight as
  FP16 (BF16 widened, then range-checked; the projector dequantized by
  `mx.dequantize`), and the stage outputs as raw little-endian F32.

Run from the repository root:

    uv run --python 3.12 --with mlx --with mlx-vlm --with numpy --with pillow \\
      scripts/gemma4_vision_real_dump.py OUT_DIR

then the ignored test, which reads `OUT_DIR`:

    TURBOSPARK_GEMMA4_VISION_DUMP=OUT_DIR cargo test -p turbospark-runtime --lib \\
      vision::gemma4::tests::the_real_tower_matches_mlx_vlm -- --ignored --nocapture
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import struct
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
REVISION = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
BASE = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"
SHARD = "model-00003-of-00003.safetensors"
PREFIXES = ("vision_tower.", "embed_vision.")
# The stages compared, and which block outputs are kept (all 27 would be
# 300 MB of F32 for one image).
KEEP_BLOCKS = (0, 13, 26)


def fetch(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    if len(data) != end - start:
        raise SystemExit(f"short read {len(data)} of {end - start} at {start}")
    return data


def download_vision(out: Path) -> Path:
    """The vision tensors as one local safetensors file, cached in `out`."""
    dst = out / "vision.safetensors"
    if dst.exists():
        return dst
    url = f"{BASE}/{SHARD}"
    n = struct.unpack("<Q", fetch(url, 0, 8))[0]
    header = json.loads(fetch(url, 8, 8 + n))
    base = 8 + n
    names = sorted(k for k in header if k.startswith(PREFIXES))
    if len(names) != 358:
        raise SystemExit(f"expected 358 vision tensors, found {len(names)}")

    def one(name: str):
        a, b = header[name]["data_offsets"]
        return name, fetch(url, base + a, base + b)

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        blobs = dict(pool.map(one, names))
    new_header = {}
    cursor = 0
    for name in names:
        size = len(blobs[name])
        new_header[name] = {
            "dtype": header[name]["dtype"],
            "shape": header[name]["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    hj = json.dumps(new_header).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    tmp = dst.with_suffix(".part")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for name in names:
            f.write(blobs[name])
    tmp.rename(dst)
    print(f"wrote {dst} ({dst.stat().st_size / 2**20:.0f} MiB)")
    return dst


def synth_image(height: int, width: int):
    import numpy as np

    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = 128 + 100 * np.sin(x / 9.0 + y / 23.0)
    g = 128 + 100 * np.cos(y / 13.0 - x / 31.0)
    b = 128 + 100 * np.sin((x + 2 * y) / 17.0)
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--mlx-vlm",
        default=str(REPO_ROOT.parent / "mlx-v" / "mlx-vlm"),
        help="the vendored mlx-vlm checkout to import from",
    )
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(args.mlx_vlm).resolve()))

    import importlib

    import mlx.core as mx
    import numpy as np

    vision_mod = importlib.import_module("mlx_vlm.models.gemma4.vision")
    config_mod = importlib.import_module("mlx_vlm.models.gemma4.config")
    gemma_mod = importlib.import_module("mlx_vlm.models.gemma4.gemma4")
    proc_mod = importlib.import_module("mlx_vlm.models.gemma4.processing_gemma4")

    config = json.loads(fetch(f"{BASE}/config.json", 0, 12180))
    vc = dict(config["vision_config"])
    fields = config_mod.VisionConfig.__dataclass_fields__
    cfg = config_mod.VisionConfig(**{k: v for k, v in vc.items() if k in fields})
    text_hidden = config["text_config"]["hidden_size"]

    weights = mx.load(str(download_vision(out)))

    # One deterministic image through the reference processor.
    src = synth_image(480, 640)
    proc = proc_mod.Gemma4ImageProcessor(patch_size=16, pooling_kernel_size=3)
    data, ntok = proc.preprocess([src], max_soft_tokens=280)
    pv = data["pixel_values"]
    pv = pv[0] if isinstance(pv, list) else pv[0]  # (C, H, W) in [0, 1]
    c, h, w = pv.shape
    p = cfg.patch_size
    gh, gw = h // p, w // p
    # This port's patch rows: (py, px) row-major, each [16][16][RGB].
    rows = pv.reshape(c, gh, p, gw, p).transpose(1, 3, 2, 4, 0).reshape(gh * gw, p * p * c)
    gx, gy = np.meshgrid(np.arange(gw), np.arange(gh), indexing="xy")
    positions = np.stack([gx.reshape(-1), gy.reshape(-1)], -1).astype(np.int32)

    def build(dtype):
        model = vision_mod.VisionModel(cfg)
        embed = gemma_mod.MultimodalEmbedder(cfg.hidden_size, text_hidden, eps=cfg.rms_norm_eps)
        tower = [
            (k[len("vision_tower."):], v.astype(dtype))
            for k, v in weights.items()
            if k.startswith("vision_tower.")
        ]
        model.load_weights(tower)
        proj = mx.dequantize(
            weights["embed_vision.embedding_projection.weight"],
            weights["embed_vision.embedding_projection.scales"],
            weights["embed_vision.embedding_projection.biases"],
            group_size=64,
            bits=4,
        )
        embed.embedding_projection.weight = proj.astype(dtype)
        return model, embed, proj

    def run(dtype):
        model, embed, proj = build(dtype)
        px = mx.array(rows, dtype=dtype)[None]
        pos = mx.array(positions)[None]
        pad = mx.zeros((1, gh * gw), dtype=mx.bool_)
        stages = {}
        hs = model.patch_embedder.embed_patches(px, pos, pad)
        stages["patch_embed"] = hs
        for i, layer in enumerate(model.encoder.layers):
            hs = layer(hs, pos, None)
            if i in KEEP_BLOCKS:
                stages[f"block_{i}"] = hs
        tower = model(px, pos)
        stages["tower"] = tower
        stages["embedded"] = embed(tower)
        mx.eval(stages)
        return {k: np.asarray(v[0].astype(mx.float32)) for k, v in stages.items()}, proj

    f32, proj = run(mx.float32)
    bf16, _ = run(mx.bfloat16)

    def raw(path: Path, array, dtype):
        np.ascontiguousarray(array, dtype=dtype).tofile(path)

    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "grid": [gw, gh],
        "soft_tokens": int(ntok[0]),
        "text_hidden": text_hidden,
        "keep_blocks": list(KEEP_BLOCKS),
        "lossy": {},
    }
    raw(out / "pixels.f32", rows, np.float32)
    raw(out / "positions.u32", positions, np.uint32)
    wdir = out / "weights"
    wdir.mkdir(exist_ok=True)
    for k, v in weights.items():
        if not k.startswith("vision_tower."):
            continue
        a = np.asarray(v.astype(mx.float32))
        if not np.isfinite(a).all() or np.abs(a).max() > 65504:
            raise SystemExit(f"{k} does not fit FP16")
        h16 = a.astype(np.float16)
        lossy = int((h16.astype(np.float32) != a).sum())
        if lossy:
            manifest["lossy"][k] = lossy
        raw(wdir / (k[len("vision_tower."):] + ".f16"), h16, np.float16)
    pa = np.asarray(proj.astype(mx.float32))
    raw(wdir / "embedding_projection.weight.f16", pa.astype(np.float16), np.float16)
    for name, stages in (("f32", f32), ("bf16", bf16)):
        sdir = out / name
        sdir.mkdir(exist_ok=True)
        for k, v in stages.items():
            raw(sdir / f"{k}.f32", v, np.float32)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    def cos(a, b):
        a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    print(f"grid {gw}x{gh}, {gh * gw} patches, {ntok[0]} soft tokens")
    for k in f32:
        print(f"bf16 vs f32 {k}: cosine {cos(bf16[k], f32[k]):.8f}, absmax {np.abs(f32[k]).max():.1f}")
    print(f"lossy FP16 conversions in {len(manifest['lossy'])} tensors")


if __name__ == "__main__":
    main()
