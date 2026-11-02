#!/usr/bin/env python3
"""Dump the LLaVA-1.5 CLIP tower's reference stages for the parity gate.

Reads the vision tower and projector straight out of a `llava-hf` checkpoint
(no language model is loaded), runs the `transformers` CLIP tower and LLaVA
projector on the pixels the Hugging Face `CLIPImageProcessor` produces, and
writes raw little-endian f32 files the Rust test
`crates/runtime/tests/clip_vision_parity.rs` reads:

  pixels.f32        [3, 336, 336]   the processor's pixel_values (NCHW, N=1)
  hs0.f32           [577, 1024]     hidden_states[0]  (after ln_pre)
  hs1.f32           [577, 1024]     hidden_states[1]  (after block 0)
  hs_last.f32       [577, 1024]     hidden_states[-2] (after block 22)
  features.f32      [576, 4096]     projector(hs_last[1:])
  meta.json                         shapes and the image path

Run:

  uv run --python 3.12 --with torch --with numpy --with pillow \\
    --with transformers --with safetensors -- \\
    python scripts/llava_vision_probe.py \\
    --checkpoint /Volumes/T7/hf-models/llava-1.5-7b-hf \\
    --image /Volumes/T7/vision-pages/shapes.png \\
    --out-dir /Volumes/T7/vision-pages/llava-dump-shapes

The reference is `transformers`' own `CLIPVisionModel` and
`LlavaMultiModalProjector`, in float32 on the CPU, so the only difference
from the Rust port is the port's FP16 storage. The feature layer and the
select strategy are read from the config, not restated here.
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from safetensors import safe_open
    from transformers import CLIPImageProcessor, CLIPVisionModel, LlavaConfig
    from transformers.models.llava.modeling_llava import LlavaMultiModalProjector

    cfg = LlavaConfig.from_pretrained(args.checkpoint)
    layer = cfg.vision_feature_layer
    assert cfg.vision_feature_select_strategy == "default"

    proc = CLIPImageProcessor.from_pretrained(args.checkpoint)
    img = Image.open(args.image).convert("RGB")
    pix = proc(images=img, return_tensors="np")["pixel_values"].astype(np.float32)
    assert pix.shape == (1, 3, 336, 336), pix.shape

    index = json.load(open(os.path.join(args.checkpoint, "model.safetensors.index.json")))
    wanted = {
        k: v
        for k, v in index["weight_map"].items()
        if k.startswith("vision_tower.") or k.startswith("multi_modal_projector.")
    }
    tower_sd, proj_sd = {}, {}
    for shard in sorted(set(wanted.values())):
        with safe_open(os.path.join(args.checkpoint, shard), framework="pt") as f:
            for k in f.keys():
                if k not in wanted:
                    continue
                t = f.get_tensor(k).to(torch.float32)
                if k.startswith("vision_tower."):
                    tower_sd[k[len("vision_tower."):]] = t
                else:
                    proj_sd[k[len("multi_modal_projector."):]] = t

    tower = CLIPVisionModel(cfg.vision_config).eval()
    # Newer transformers drop the `vision_model.` level the checkpoint keeps.
    own = set(tower.state_dict().keys())
    if not any(k.startswith("vision_model.") for k in own):
        tower_sd = {
            (k[len("vision_model."):] if k.startswith("vision_model.") else k): v
            for k, v in tower_sd.items()
        }
    missing, unexpected = tower.load_state_dict(tower_sd, strict=False)
    missing = [m for m in missing if "position_ids" not in m]
    assert not missing, f"{len(missing)} missing, e.g. {missing[:3]}"
    assert not unexpected, f"{len(unexpected)} unexpected, e.g. {list(unexpected)[:3]}"
    proj = LlavaMultiModalProjector(cfg).eval()
    proj.load_state_dict(proj_sd)

    with torch.no_grad():
        out = tower(torch.from_numpy(pix), output_hidden_states=True)
        hs = out.hidden_states
        last = hs[layer]
        feat = proj(last[:, 1:])

    # The noise floor: the SAME reference run with FP16 weights and
    # activations, as this port stores them. A port-versus-float32 gap no
    # bigger than this one is storage precision, not a defect.
    import copy

    tower_h = copy.deepcopy(tower).half()
    with torch.no_grad():
        out_h = tower_h(torch.from_numpy(pix).half(), output_hidden_states=True)
        last_h = out_h.hidden_states[layer]

    os.makedirs(args.out_dir, exist_ok=True)

    def dump(name, a):
        np.asarray(a.detach().numpy() if hasattr(a, "detach") else a, dtype=np.float32).reshape(-1).tofile(
            os.path.join(args.out_dir, name)
        )

    dump("pixels.f32", pix[0])
    dump("hs0.f32", hs[0][0])
    dump("hs1.f32", hs[1][0])
    dump("hs_last.f32", last[0])
    dump("hs_last_f16.f32", last_h[0].float())
    dump("features.f32", feat[0])
    meta = {
        "image": os.path.abspath(args.image),
        "checkpoint": os.path.abspath(args.checkpoint),
        "feature_layer": layer,
        "hidden_states": len(hs),
        "shapes": {
            "pixels": [3, 336, 336],
            "hs": list(hs[0].shape[1:]),
            "features": list(feat.shape[1:]),
        },
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    sys.exit(main())
