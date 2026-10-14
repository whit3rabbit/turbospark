"""Render the known-text pages for the Core ML OCR smoke gate.

Two letter-shaped pages of plain prose in a system font, black on white,
written to crates/runtime/testdata/coreml_ocr/ with the exact text in
pages.json. The gate compares the model's output to that text by
character error rate.

    uv run --python 3.12 --with pillow==12.3.0 python -I crates/runtime/tools/gen_ocr_pages.py

Needs macOS (the font is /System/Library/Fonts/Supplemental/Arial.ttf).
"""

import hashlib
import json
import os

from PIL import Image, ImageDraw, ImageFont, __version__ as pil_version

OUT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata", "coreml_ocr")
)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"

PAGES = [
    [
        "Quarterly Report",
        "Revenue grew eleven percent compared with the same",
        "period last year, driven by strong demand in Europe.",
        "Operating costs were flat, and the team hired twelve",
        "new engineers to support the product roadmap.",
    ],
    [
        "Meeting Notes",
        "The committee approved the new library budget.",
        "Construction will begin in the spring and should",
        "finish before the end of next year.",
        "Volunteers are welcome to join the planning group.",
    ],
]


def render(lines, path):
    img = Image.new("RGB", (850, 1100), "white")
    draw = ImageDraw.Draw(img)
    title = ImageFont.truetype(FONT, 40)
    body = ImageFont.truetype(FONT, 26)
    y = 90
    for i, line in enumerate(lines):
        font = title if i == 0 else body
        draw.text((70, y), line, fill="black", font=font)
        y += 70 if i == 0 else 42
    img.save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {"generator": "crates/runtime/tools/gen_ocr_pages.py", "pillow": pil_version, "pages": []}
    for i, lines in enumerate(PAGES):
        name = f"page{i + 1}.png"
        path = os.path.join(OUT, name)
        render(lines, path)
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        meta["pages"].append({"file": name, "sha256": digest, "lines": lines})
    with open(os.path.join(OUT, "pages.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
