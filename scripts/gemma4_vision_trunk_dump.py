#!/usr/bin/env python3
"""Run mlx-vlm's full Gemma 4 model on one image prompt and dump what the
runtime's trunk gate compares against (`docs/GEMMA4_VISION_PHASE0.md`).

Needs a local copy of the pinned checkpoint (`hf download
mlx-community/gemma-4-26b-a4b-it-4bit --revision 0d77464e... --local-dir DIR`).

The image is rendered here (black text on white, a red disc) so the right
answer is known: the model should read the text. Its raw RGB bytes are
written out, and the runtime preprocesses the SAME bytes with
`turbospark_vision_io::gemma4_preprocess`, whose parity with this processor
is already held (`crates/vision-io/tests/gemma4_preprocess_parity.rs`).

TWO RUNS, because the two references disagree on the mask. transformers
(the pinned `modeling_gemma4.py`) makes only SLIDING layers bidirectional
inside an image span; mlx-vlm overlays every layer, full ones included. The
runtime implements the transformers rule, so the run named `hf_mask` patches
mlx-vlm's mask to that rule, and `mlx_mask` is mlx-vlm as shipped.

Each run writes the last prompt position's logits (F32) and a greedy
continuation. Run from the repository root:

    uv run --python 3.12 --with mlx --with mlx-vlm --with numpy --with pillow \\
      scripts/gemma4_vision_trunk_dump.py MODEL_DIR OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTION = "What text is written in this image? Answer with the text only."
IMAGE_TEXT = "TURBOSPARK 42"
GREEDY_TOKENS = 24


def render_image():
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=72)
    except TypeError:
        raise SystemExit("Pillow >= 10.1 is needed for a sized default font")
    draw.text((40, 170), IMAGE_TEXT, fill=(0, 0, 0), font=font)
    draw.ellipse((480, 40, 600, 160), fill=(220, 30, 30))
    return img


def rule_make_masks(lm_class, overlay_on):
    """`_make_masks` with the image overlay applied only to the layer types in
    `overlay_on`. Everything else is mlx-vlm's own code path."""
    original = lm_class._make_masks
    overlay = lm_class._apply_blockwise_bidirectional_overlay

    def patched(self, h, cache, mm_token_type_ids=None):
        lm_class._apply_blockwise_bidirectional_overlay = lambda s, m, t: m
        try:
            causal = original(self, h, cache, mm_token_type_ids)
        finally:
            lm_class._apply_blockwise_bidirectional_overlay = overlay
        both = original(self, h, cache, mm_token_type_ids)
        return [
            b if layer.layer_type in overlay_on else c
            for layer, b, c in zip(self.layers, both, causal)
        ]

    return original, patched


# The rules compared. `hf_mask` is transformers' (sliding only) and
# `mlx_mask` is mlx-vlm as shipped (every layer); the other two locate which
# half a disagreement comes from.
RULES = {
    "hf_mask": {"sliding_attention"},
    "mlx_mask": None,
    "causal_mask": set(),
    "full_only_mask": {"full_attention"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--mlx-vlm", default=str(REPO_ROOT.parent / "mlx-v" / "mlx-vlm"), help="vendored mlx-vlm"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(args.mlx_vlm).resolve()))

    import mlx.core as mx
    import numpy as np
    from mlx_vlm import load
    from mlx_vlm.models.gemma4 import language as lang

    model, processor = load(str(args.model))
    img = render_image()
    np.asarray(img, dtype=np.uint8).tofile(args.out / "image.rgb")
    # The same image as PNG, for the server and FFI tests, which take a file.
    img.save(args.out / "image.png")

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=[img], return_tensors="np")
    input_ids = np.asarray(inputs["input_ids"])[0].astype(np.int64)
    mm = np.asarray(inputs["mm_token_type_ids"]) if "mm_token_type_ids" in inputs else None
    pixel_values = inputs["pixel_values"]
    if isinstance(pixel_values, list):
        pixel_values = [mx.array(p) for p in pixel_values]
    else:
        pixel_values = mx.array(pixel_values)
    image_id = model.config.image_token_id
    image_tokens = int((input_ids == image_id).sum())

    lm_class = type(model.language_model.model) if hasattr(model.language_model, "model") else None
    if lm_class is None or not hasattr(lm_class, "_make_masks"):
        lm_class = next(
            c for c in vars(lang).values() if isinstance(c, type) and hasattr(c, "_make_masks")
        )
    def prefill(name: str):
        rule = RULES[name]
        original, patched = rule_make_masks(lm_class, rule or set())
        lm_class._make_masks = original if rule is None else patched
        cache = model.language_model.make_cache()
        kwargs = {"cache": cache}
        if mm is not None:
            kwargs["mm_token_type_ids"] = mx.array(mm)
        out = model(mx.array(input_ids)[None], pixel_values, **kwargs)
        logits = out.logits if hasattr(out, "logits") else out
        last = logits[0, -1].astype(mx.float32)
        mx.eval(last)
        lm_class._make_masks = original
        return cache, last

    def step(cache, token):
        out = model.language_model(inputs=mx.array([[token]]), cache=cache)
        return (out.logits if hasattr(out, "logits") else out)[0, -1]

    def free_run(name: str):
        cache, last = prefill(name)
        generated = []
        token = int(mx.argmax(last).item())
        for _ in range(GREEDY_TOKENS):
            generated.append(token)
            token = int(mx.argmax(step(cache, token)).item())
        np.asarray(last).astype(np.float32).tofile(args.out / f"{name}.logits.f32")
        text = processor.tokenizer.decode(generated)
        print(f"[{name}] {text!r}")
        return generated, text

    def forced_run(name: str, tokens):
        # Row i: the logits after feeding tokens[i]. Every rule is forced on
        # the SAME tokens, so their rows are comparable position by position.
        cache, _ = prefill(name)
        rows = [np.asarray(step(cache, t).astype(mx.float32)) for t in tokens]
        np.stack(rows).astype(np.float32).tofile(args.out / f"{name}.steps.f32")

    results = {name: free_run(name) for name in RULES}
    common = results["mlx_mask"][0]
    for name in RULES:
        forced_run(name, common)
    manifest = {
        "question": QUESTION,
        "image_text": IMAGE_TEXT,
        "image_hw": [480, 640],
        "input_ids": input_ids.tolist(),
        "image_token_id": int(image_id),
        "image_tokens": image_tokens,
        "vocab": int(np.fromfile(args.out / "hf_mask.logits.f32", dtype=np.float32).size),
        "runs": {k: {"greedy": v[0], "text": v[1]} for k, v in results.items()},
        "forced_tokens": common,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    a = np.fromfile(args.out / "hf_mask.logits.f32", dtype=np.float32)
    b = np.fromfile(args.out / "mlx_mask.logits.f32", dtype=np.float32)
    print(f"prompt {len(input_ids)} tokens, {image_tokens} image tokens")
    print(f"hf vs mlx mask: top1 {a.argmax()} vs {b.argmax()}, max |dlogit| {np.abs(a - b).max():.3f}")


if __name__ == "__main__":
    main()
