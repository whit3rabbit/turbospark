#!/usr/bin/env python3
"""Render a deterministic dense-text page for the vision activation probe.

Companion to `vision_tower_probe.py`. Exists so the activation numbers in
`docs/VISION_PHASE0.md` can be reproduced rather than merely quoted: the
probe's answer depends on the INPUT as much as on the tower, and "a dense
OCR page" is not a reproducible specification.

The default 4064x4064 is the size that closed Phase 0 item 3's open action
item -- 16,516,096 px, just under `preprocessor_config.json`'s 16,777,216
ceiling, which resizes to a 254x254 patch grid (64,516 patches). That is
essentially the largest input the processor accepts, and it is what says
the block-26 activation peak is a fixed outlier feature rather than
something that scales with sequence length.

DENSE SMALL TEXT ON PURPOSE. A mostly-blank page at this size exercises the
sequence length while leaving the outlier features idle, so it would report
a comfortable peak for the wrong reason.

Usage:
    uv run --python 3.12 --with pillow -- \
      scripts/make_vision_test_page.py /tmp/vision-probe/imgs/extreme_page.png
    ... --size 1024 1280      # the smaller page the original probe used
"""

import argparse
import random

from PIL import Image, ImageDraw, ImageFont

WORDS = (
    "throughput decode prefill expert router kernel tensor quantize "
    "footprint checkpoint attention residual embedding tokenizer "
    "streaming manifest oracle parity fixture gradient"
).split()


def render(width: int, height: int, seed: int, point_size: int) -> Image.Image:
    random.seed(seed)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Courier New.ttf", point_size
        )
    except OSError:
        # Pillow's built-in bitmap font. Different glyphs, so the absolute
        # activation numbers will not reproduce the committed ones; the shape
        # of the result (a fixed peak rather than a length-scaled one) will.
        font = ImageFont.load_default()

    y = point_size - 3
    step = point_size + 3
    # `- point_size - 5` rather than `- point_size`: the bottom margin has to
    # leave room for the glyphs' descenders, and this bound is what the page
    # the committed activation numbers were measured on actually used. One
    # extra line at the bottom is 0.019% of the pixels and reproduces none of
    # the committed bytes.
    while y < height - point_size - 5:
        count = random.randint(9, 14)
        body = " ".join(random.choice(WORDS) for _ in range(count))
        line = f"{y:05d} | {body} | {random.randint(1000, 9999)}.{random.randint(10, 99)}"
        draw.text((10, y), line, font=font, fill="black")
        y += step
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", help="where to write the PNG")
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=(4064, 4064),
        metavar=("WIDTH", "HEIGHT"),
        help="default 4064x4064, just under the processor's pixel ceiling",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--point-size", type=int, default=15)
    args = parser.parse_args()

    width, height = args.size
    img = render(width, height, args.seed, args.point_size)
    img.save(args.out)
    print(f"wrote {args.out}: {width}x{height}, {width * height:,} px")


if __name__ == "__main__":
    main()
