#!/usr/bin/env python3
"""Score an OCR run's output against the dense test page used by the vision gates.

The page (`/Volumes/T7/vision-pages/page.png` and its crop `page-small.png`) is
a monospace text page whose lines are `NNNNN | word word ...` with NNNNN
running 00012, 00030, 00048, ... (step 18). The numbers and the words exist
ONLY in the image, so output that carries them proves the image reached the
model; a no-image control must carry none.

`GROUND_TRUTH` is the first 24 lines' number and first three words, read off
the page image by eye (the generator needs Pillow and the exact font, so the
page cannot be regenerated here). That is what the match rate is against; the
line-number progression is checked over every line the run produced.

Usage:
    python3 -I scripts/check_ocr_page_output.py RUN.out [--expect-none]

`--expect-none` is the control: it passes only when NO ground-truth number
appears. Exit status 0 on pass.
"""
import re
import sys

GROUND_TRUTH = [
    ("00012", "parity streaming streaming"),
    ("00030", "prefill parity decode"),
    ("00048", "decode tensor quantize"),
    ("00066", "checkpoint manifest throughput"),
    ("00084", "checkpoint throughput prefill"),
    ("00102", "embedding embedding tokenizer"),
    ("00120", "quantize expert throughput"),
    ("00138", "router tokenizer embedding"),
    ("00156", "fixture expert decode"),
    ("00174", "quantize throughput gradient"),
    ("00192", "checkpoint quantize gradient"),
    ("00210", "expert expert decode"),
    ("00228", "kernel prefill router"),
    ("00246", "streaming quantize decode"),
    ("00264", "oracle fixture router"),
    ("00282", "checkpoint attention router"),
    ("00300", "prefill manifest prefill"),
    ("00318", "prefill prefill expert"),
    ("00336", "prefill oracle oracle"),
    ("00354", "residual checkpoint router"),
    ("00372", "embedding footprint tensor"),
    ("00390", "residual decode prefill"),
    ("00408", "attention gradient streaming"),
    ("00426", "embedding streaming tensor"),
]


def generated_text(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    marker = "generating (real forward pass, chat template applied):"
    body = text.split(marker, 1)[1] if marker in text else text
    # The footer lines (`context:`, `expert ...`, `[stop=...]`) follow the text.
    return body.split("\ncontext:", 1)[0]


def main():
    path = sys.argv[1]
    expect_none = "--expect-none" in sys.argv
    body = generated_text(path)
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    rows = []
    for line in lines:
        # The pipe is optional: on the 1024 x 1280 page the model drops it.
        m = re.match(r"^(\d{5})\s*\|?\s*(.*)$", line)
        if m:
            rows.append((m.group(1), m.group(2)))
    numbers = [n for n, _ in rows]
    truth_numbers = {n for n, _ in GROUND_TRUTH}
    hits = [n for n in numbers if n in truth_numbers]
    print(f"output lines {len(lines)}, numbered rows {len(rows)}")
    if expect_none:
        ok = not hits
        print(f"control: ground-truth numbers present: {hits[:5]} -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    # Progression: every numbered row is the previous plus 18, starting at 12.
    want = [f"{12 + 18 * i:05d}" for i in range(len(numbers))]
    progression_ok = numbers == want
    first_bad = next((i for i, (a, b) in enumerate(zip(numbers, want)) if a != b), None)
    matched = 0
    checked = 0
    for number, words in rows:
        truth = dict(GROUND_TRUTH).get(number)
        if truth is None:
            continue
        checked += 1
        got = " ".join(words.split()[:3])
        if got == truth:
            matched += 1
        else:
            print(f"  line {number}: got {got!r}, want {truth!r}")
    print(
        f"line-number progression 00012 step 18 over {len(numbers)} rows: "
        f"{'OK' if progression_ok else f'BROKEN at row {first_bad}'}"
    )
    print(f"first-three-words match: {matched}/{checked} of the transcribed lines")
    # The strict default (every transcribed line, an unbroken progression) is
    # for the 448 px crop. The dense 1024 x 1280 page is padded to 1024 x 1024
    # (scale 0.8, ~9 px glyphs), where a rate is the honest measure: pass it
    # with --min-match 0.75, and the progression is then reported, not gated.
    min_match = 1.0
    if "--min-match" in sys.argv:
        min_match = float(sys.argv[sys.argv.index("--min-match") + 1])
    ok = checked > 0 and matched >= min_match * len(GROUND_TRUTH)
    if min_match >= 1.0:
        ok = ok and progression_ok
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
