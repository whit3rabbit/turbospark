#!/usr/bin/env python3
"""Regenerate the OmniVoice duration-estimator Rust parity fixture.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/tts/models/omnivoice/duration.py).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    PYTHONPATH=$PWD ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_omnivoice_duration_fixture.py

Writes crates/audio/testdata/tts_omnivoice/duration_fixture.json: the
reference `RuleDurationEstimator` over mixed-script texts and boundary
codepoints, with the target weights and the estimate for each case.
Deterministic; no network.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from mlx_audio.tts.models.omnivoice.duration import RuleDurationEstimator

OUT = Path(
    "../turbospark/crates/audio/testdata/tts_omnivoice/duration_fixture.json"
)
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# Mixed scripts, punctuation, digits, combining marks, the tatweel, and
# the range-table boundary codepoints (last Latin row, first modifier,
# last CJK row, first Yi row, CJK Ext B, fullwidth forms).
CASES = [
    "Hello world.",
    "Nice to meet you.",
    "The quick brown fox jumps over the lazy dog!",
    "中文字符串与English混合。",
    "日本語のテキスト、です。123",
    "한국어 텍스트",
    "Привет, мир!",
    "مرحبا بالعالم",
    "שלום עולם",
    "สวัสดีครับ",
    "नमस्ते दुनिया",
    "አማርኛ ጽሑፍ",
    "café naïve résumé",
    "\u0627\u0644\u0639\u0631\u0627\u0628\u064a\u0651\u0629 \u0640\u0640\u0640",
    "a\u0301e\u0303 combining marks",
    "\u02af\u02b0 boundary",
    "\u9fff\ua000 boundary",
    "\U00020000\U0002a700 upper planes",
    "\uff21\uff32 fullwidth latin",
    "\uff10\uff19 fullwidth digits",
    "123 456.789",
    "[laughter] really? yes!",
]


def main() -> None:
    estimator = RuleDurationEstimator()
    cases = []
    for text in CASES:
        # The generate() fallback calibrates against this reference
        # pair; the fixture also exercises other pairs and thresholds.
        variants = [
            ("Nice to meet you.", 25.0, 50, 3),
            ("Hello world.", 4.0, 50, 3),
            ("Hello world.", 4.0, None, 3),
            ("Hello world.", 4.0, 5.0, 2),
        ]
        for ref, ref_duration, low, boost in variants:
            weight = sum(estimator._get_char_weight(c) for c in text)
            estimate = estimator.estimate_duration(
                text, ref, ref_duration, low_threshold=low, boost_strength=boost
            )
            # serde_json's default float parser is not correctly rounded,
            # so the exact f64s travel as bit patterns; the readable
            # floats stay for review only.
            cases.append(
                {
                    "target": text,
                    "ref": ref,
                    "ref_duration": ref_duration,
                    "low_threshold": low,
                    "boost_strength": boost,
                    "target_weight": weight,
                    "target_weight_bits": struct.unpack("<Q", struct.pack("<d", weight))[0],
                    "estimate": estimate,
                    "estimate_bits": struct.unpack("<Q", struct.pack("<d", estimate))[0],
                }
            )
    document = {
        "generator": "tools/gen_omnivoice_duration_fixture.py",
        "reference_commit": REFERENCE_COMMIT,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=2, ensure_ascii=True))
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
