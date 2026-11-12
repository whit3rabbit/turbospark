#!/usr/bin/env python3
"""Greedy LLaVA-1.5 generation with `transformers`, the reference for the
end-to-end content gate (`docs/LLAVA_VISION_PHASE0.md`).

Loads the full `llava-hf` checkpoint in FP16 and answers one image + question
with greedy decoding, through the checkpoint's own chat template and
processor. Prints only the generated text on stdout.

  uv run --python 3.12 --with torch --with numpy --with pillow \\
    --with transformers --with safetensors --with accelerate -- \\
    python scripts/llava_reference_generate.py \\
    --checkpoint /Volumes/T7/hf-models/llava-1.5-7b-hf \\
    --image /Volumes/T7/vision-pages/shapes.png \\
    --prompt "Describe the shapes in this image and say what color each one is."

The port runs the Q4_K_M GGUF trunk, so the wording will differ from this
FP16 reference; what must agree is WHAT THE IMAGE CONTAINS.
"""
import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument(
        "--prompt",
        required=True,
        action="append",
        help="repeat to ask several questions of the same image with one model load",
    )
    ap.add_argument("--max-new", type=int, default=80)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.float16
    ).to(device)
    image = Image.open(args.image).convert("RGB")
    for prompt in args.prompt:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
        new = out[0][inputs["input_ids"].shape[1]:]
        print(f"Q: {prompt}\nA: {processor.decode(new, skip_special_tokens=True).strip()}")


if __name__ == "__main__":
    main()
