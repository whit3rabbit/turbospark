#!/usr/bin/env python3
"""Regenerate the MegaTTS3 post-processing Rust parity fixtures.

Normative reference: pyloudnorm 0.1.1 (`Meter.integrated_loudness`,
`normalize.loudness`) and `combine_audio_segments` from
`tts/utils/audio_utils/io.py` of github.com/bytedance/MegaTTS3 (Apache-2.0),
with the peak guard from `MegaTTS3DiTInfer.forward`.

Needs numpy, scipy and pyloudnorm==0.1.1 in a scratch venv:

    python gen_megatts3_post_fixtures.py --ref DIR --out crates/audio/testdata/megatts3_post

Deterministic: fixed seeds, no network access.
"""

import argparse
import json
import os
import sys
import types

import numpy as np
import pyloudnorm as pyln

RATE = 24000


def signals(rng):
    t = lambda s: np.arange(int(s * RATE)) / RATE
    sig = {}
    sig["sine"] = (0.3 * np.sin(2 * np.pi * 440 * t(1.0))).astype("<f4")
    noisy = 0.2 * np.sin(2 * np.pi * 220 * t(3.7)) + 0.05 * rng.standard_normal(len(t(3.7)))
    noisy[int(1.2 * RATE):int(2.7 * RATE)] = 0.0  # a silent stretch the absolute gate must drop
    sig["gappy"] = noisy.astype("<f4")
    # Loud and quiet sections 30 dB apart: the relative gate drops the quiet one.
    env = np.concatenate([np.full(int(1.5 * RATE), 0.4), np.full(int(1.5 * RATE), 0.4 * 10 ** (-30 / 20)), np.full(int(1.0 * RATE), 0.4)])
    sig["levels"] = (env * rng.standard_normal(len(env)) * 0.5).astype("<f4")
    sig["short"] = (0.25 * rng.standard_normal(int(0.45 * RATE))).astype("<f4")
    sig["block"] = (0.25 * rng.standard_normal(int(0.4 * RATE))).astype("<f4")
    sig["half_second"] = (0.25 * rng.standard_normal(int(0.5 * RATE))).astype("<f4")
    sig["odd_length"] = (0.1 * rng.standard_normal(int(2.3333 * RATE) + 7)).astype("<f4")
    sig["silence"] = np.zeros(RATE, dtype="<f4")
    sig["quiet_noise"] = (1e-6 * rng.standard_normal(RATE)).astype("<f4")
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.ref)
    # io.py imports pydub only for file conversion helpers.
    sys.modules.setdefault("pydub", types.SimpleNamespace(AudioSegment=None))
    from tts.utils.audio_utils.io import combine_audio_segments

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(2024)
    meter = pyln.Meter(RATE)
    loud = {}
    for name, wav in signals(rng).items():
        np.save(os.path.join(args.out, f"sig_{name}.npy"), wav)
        if len(wav) < 0.4 * RATE:
            loud[name] = "too_short"
            continue
        v = meter.integrated_loudness(wav.astype(float))
        loud[name] = None if not np.isfinite(v) else float(v)

    # Loudness matching with the peak guard (as in forward()).
    match = []
    base = (0.3 * rng.standard_normal(RATE)).astype("<f4")
    for target in (-18.0, -30.0, -6.0, 3.0):
        inp = meter.integrated_loudness(base.astype(float))
        out = pyln.normalize.loudness(base.astype(float), inp, target)
        if np.abs(out).max() >= 1:
            out = out / np.abs(out).max() * 0.95
        np.save(os.path.join(args.out, f"match_{len(match)}.npy"), out.astype("<f4"))
        match.append({"input": float(inp), "target": target, "peak": float(np.abs(out).max())})
    np.save(os.path.join(args.out, "match_base.npy"), base)

    # Crossfade joins.
    combos = []
    for k, (sr, sizes) in enumerate([(24000, [9000, 12000, 8000]), (1000, [400, 300, 500, 161]), (24000, [5000])]):
        segs = [rng.standard_normal(n).astype("<f4") for n in sizes]
        for i, s in enumerate(segs):
            np.save(os.path.join(args.out, f"comb{k}_seg{i}.npy"), s)
        joined = combine_audio_segments([s.astype(np.float32) for s in segs], sr=sr)
        np.save(os.path.join(args.out, f"comb{k}_out.npy"), joined.astype("<f4"))
        combos.append({"rate": sr, "sizes": sizes, "out_len": int(len(joined))})
    with open(os.path.join(args.out, "cases.json"), "w") as fh:
        json.dump({"loudness": loud, "match": match, "combine": combos, "pyloudnorm": "0.1.1"}, fh, indent=1, sort_keys=True)
    print("wrote", args.out, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in loud.items()})


if __name__ == "__main__":
    main()
