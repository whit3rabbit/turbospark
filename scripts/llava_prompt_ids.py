#!/usr/bin/env python3
"""Print the Hugging Face processor's prompt ids for one LLaVA image turn, with
the 576-long `<image>` run collapsed, so they can be diffed against the ids
`turbospark-check` prefills (`TURBOSPARK_DEBUG_PROMPT_IDS=1`).

  uv run --python 3.12 --with torch --with numpy --with pillow \\
    --with transformers -- python scripts/llava_prompt_ids.py \\
    --checkpoint /Volumes/T7/hf-models/llava-1.5-7b-hf \\
    --image /Volumes/T7/vision-pages/shapes.png \\
    --prompt "Describe the shapes in this image and say what color each one is."

Output: `head=[...] run=<n> of <id> tail=[...]` where `run` is the length of
the contiguous image-marker run.
"""
import argparse

from PIL import Image
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": args.prompt}],
        }
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    image = Image.open(args.image).convert("RGB")
    ids = processor(images=image, text=text, return_tensors="np")["input_ids"][0].tolist()
    marker = 32000
    first = ids.index(marker)
    last = len(ids) - 1 - ids[::-1].index(marker)
    run = last - first + 1
    assert all(i == marker for i in ids[first : last + 1]), "the run is not contiguous"
    print(f"total={len(ids)} head={ids[:first]} run={run} of {marker} tail={ids[last + 1:]}")
    print("rendered text:", repr(text))


if __name__ == "__main__":
    main()
