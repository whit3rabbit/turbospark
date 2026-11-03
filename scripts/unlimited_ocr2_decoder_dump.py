#!/usr/bin/env python3
"""Reference run of Unlimited-OCR-2 through mlx-vlm, for the real-model gates.

Dumps, for one rendered page: the prompt ids, the fused image rows the trunk
reads, the greedy tokens with their first logits, and the tower's intermediate
stages. `crates/runtime/tests/unlimited_ocr2_real.rs` feeds the ids and rows to
the decoder, so a decoder failure cannot be a tower failure.

    python scripts/unlimited_ocr2_decoder_dump.py MODEL_DIR OUT_DIR [--tokens N]
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.deepseekocr import sam as mlx_sam
from PIL import Image, ImageDraw, ImageFont

LINES = [
    "Quarterly Report on Sparse Attention",
    "",
    "The decoder keeps every key and reads only 512 of them per token.",
    "Remote blocks hold 32 tokens each, and twelve of them are routed.",
    "A local window of 128 tokens is always attended, without any bias.",
    "",
    "Results",
    "",
    "Pages parsed: 1,284. Mean latency: 41.7 ms. Errors: 3.",
    "The final column of table 2 was reviewed on 14 March 2026.",
]


def render_page(path):
    page = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    y = 90
    for line in LINES:
        draw.text((80, y), line, fill="black", font=font)
        y += 62
    page.save(path)
    return page


def save_f16(path, array):
    np.asarray(array.astype(mx.float32), dtype=np.float32).astype(np.float16).tofile(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--tokens", type=int, default=200)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, processor = load(args.model)
    page = render_page(out / "page.png")
    inputs = processor.process_one(prompt="Multi page parsing.", images=[page])
    ids = inputs["input_ids"]
    crops, base = inputs["images"]
    features = model.get_input_embeddings(
        ids,
        [crops, base],
        images_spatial_crop=inputs["images_spatial_crop"],
        images_seq_mask=inputs["images_seq_mask"],
    )
    embeds = features.inputs_embeds
    mx.eval(embeds)
    id_list = ids[0].tolist()
    image_id = model.config.image_token_index
    positions = [i for i, t in enumerate(id_list) if t == image_id]
    save_f16(out / "image_rows.f16", embeds[0, mx.array(positions)])
    # NHWC, which is what both towers and the Rust preprocessing use.
    save_f16(out / "pixels_base.f16", base.transpose(0, 2, 3, 1))
    save_f16(out / "pixels_crops.f16", crops.transpose(0, 2, 3, 1))

    # The tower's stages, per view size, so a failure names its stage.
    for label, view in (("base", base[:1]), ("crop", crops[:1])):
        nhwc = view.transpose(0, 2, 3, 1)
        sam_model = model.sam_model
        patch = sam_model.patch_embed(nhwc) + mlx_sam.get_abs_pos_sam(
            sam_model.pos_embed, nhwc.shape[1] // 16
        )
        sam = sam_model(nhwc)
        clip = model.vision_model(nhwc, sam)
        fused = mx.concatenate(
            (clip[:, 1:], sam.flatten(start_axis=1, end_axis=2)), axis=-1
        )
        projected = model.projector(fused)
        mx.eval(patch, sam, clip, projected)
        save_f16(out / f"stage_{label}_patch.f16", patch)
        save_f16(out / f"stage_{label}_sam.f16", sam)
        save_f16(out / f"stage_{label}_clip.f16", clip)
        save_f16(out / f"stage_{label}_projected.f16", projected)

    # Prefill in the chunks upstream's own generate uses.
    language = model.language_model
    cache = language.make_cache()
    logits = None
    for start in range(0, len(id_list), 1024):
        end = min(start + 1024, len(id_list))
        logits = language(
            ids[:, start:end], inputs_embeds=embeds[:, start:end], cache=cache
        ).logits
        mx.eval(logits)
    first = logits[0, -1].astype(mx.float32)
    tokens, per_step = [], []
    eos = 1
    for _ in range(args.tokens):
        last = logits[0, -1].astype(mx.float32)
        top = mx.argsort(-last)[:5]
        mx.eval(top, last)
        per_step.append(
            {"ids": top.tolist(), "logits": last[top].tolist()}
        )
        token = int(top[0].item())
        tokens.append(token)
        if token == eos:
            break
        logits = language(mx.array([[token]]), cache=cache).logits
        mx.eval(logits)

    tokenizer = processor.tokenizer
    text = tokenizer.decode(tokens)
    (out / "reference.json").write_text(
        json.dumps(
            {
                "ids": id_list,
                "spatial_crop": inputs["images_spatial_crop"].tolist(),
                "tokens": tokens,
                "top": per_step,
                "text": text,
                "first_top": mx.argsort(-first)[:5].tolist(),
            }
        )
    )
    print(f"prompt tokens {len(id_list)}, image rows {len(positions)}")
    print(f"crops {crops.shape}, base {base.shape}, grid {inputs['images_spatial_crop'].tolist()}")
    print(text)


if __name__ == "__main__":
    main()
