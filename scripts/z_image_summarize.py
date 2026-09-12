#!/usr/bin/env python3
"""Validate completed captures and summarize quality and resource observations."""

import argparse
import json
from pathlib import Path

import numpy as np

from z_image_compare import errors
from z_image_evidence import sha256, validate_capture

CASES = ["lighting", "composition", "typography", "detail"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("target/ig0/runs"))
    p.add_argument("--output", type=Path, default=Path("docs/verification/z-image-ig0-real.json"))
    args = p.parse_args()
    report = {"scope": "four prompts, one explicit noise seed, BF16 reference and INT4 emulation",
              "performance_baseline": False, "captures": {}, "comparisons": {}}
    for case in CASES:
        results = []
        for suffix in ("", "-int4"):
            run = args.runs / (case + suffix)
            stages = {stage: validate_capture(run / (stage + ".json")) for stage in ("encode", "denoise", "decode")}
            rows = {}
            for stage, evidence in stages.items():
                resource = evidence["resources"]
                samples = resource["samples"]
                rows[stage] = {"capture_sha256": sha256(run / (stage + ".json")),
                    "instrumented_elapsed_s": resource["elapsed_s"],
                    "sampled_max_phys_footprint_bytes": max(s.get("phys_footprint_bytes", 0) for s in samples),
                    "sampled_max_mps_live_bytes": max(s.get("mps_live_bytes", 0) for s in samples),
                    "sampled_max_mps_driver_bytes": max(s.get("mps_driver_bytes", 0) for s in samples),
                    "final_mps_live_bytes": samples[-1].get("mps_live_bytes"),
                    "final_mps_driver_bytes": samples[-1].get("mps_driver_bytes"),
                    "swap_start": samples[0]["swap"], "swap_end": samples[-1]["swap"],
                    "physical_read_bytes_during_capture": samples[-1].get("disk_read_bytes", 0) - samples[0].get("disk_read_bytes", 0),
                    "quiet_ac_throughout": resource["quiet_ac_throughout"],
                    "benchmark_eligible": evidence.get("benchmark_eligible", False)}
                if "quantization" in evidence:
                    rows[stage]["quantized_tensor_count"] = len(evidence["quantization"]["tensors"])
            report["captures"][case + suffix] = {"stages": rows, "settings": stages["denoise"]["settings"],
                                                  "actual_forwards": stages["denoise"]["actual_forwards"],
                                                  "png_sha256": stages["decode"]["png_sha256"]}
            results.append(run)
        base, quant = results
        for name in ("initial_noise", "token_ids", "attention_mask", "timesteps", "sigmas"):
            if not np.array_equal(np.load(base / (name + ".npy")), np.load(quant / (name + ".npy"))):
                raise ValueError(f"uncontrolled comparison input: {case}/{name}")
        report["comparisons"][case] = {name: errors(np.load(base / (name + ".npy")),
                                                    np.load(quant / (name + ".npy")))
                                               for name in ("conditioning", "final_latents", "decoded_pixels")}
    report["edge_encodes"] = {}
    for case in ("empty", "unicode", "overlong"):
        path = args.runs / case / "encode.json"
        if path.exists():
            evidence = validate_capture(path)
            report["edge_encodes"][case] = {"capture_sha256": sha256(path),
                "conditioning": evidence["arrays"]["conditioning"]}
    repeat = args.runs / "lighting-repeat"
    if (repeat / "decode.json").exists():
        rows = {}
        for stage in ("encode", "denoise", "decode"):
            original = validate_capture(args.runs / "lighting" / (stage + ".json"))
            repeated = validate_capture(repeat / (stage + ".json"))
            if original["arrays"].keys() != repeated["arrays"].keys():
                raise ValueError("repeat fixture inventory differs")
            rows[stage] = {"capture_sha256": sha256(repeat / (stage + ".json")),
                "identical_arrays": {name: row["sha256"] == repeated["arrays"][name]["sha256"]
                                     for name, row in original["arrays"].items()}}
        report["repeat"] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    cards = []
    for case in CASES:
        images = "".join(
            f'<figure><a href="{args.runs.name}/{case}{suffix}/image.png">'
            f'<img loading="lazy" src="{args.runs.name}/{case}{suffix}/image.png"></a>'
            f'<figcaption>{label}</figcaption></figure>'
            for suffix, label in (("", "BF16 reference"), ("-int4", "INT4 emulation")))
        cards.append(f'<section><h2>{case.title()}</h2><div class="pair">{images}</div></section>')
    page = '''<!doctype html><meta charset="utf-8"><title>Z-Image IG0 comparisons</title>
<style>body{font:16px system-ui;margin:32px auto;max-width:1300px;padding:0 24px;background:#17191c;color:#eee}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%;border-radius:6px}
figcaption{padding:8px 0;color:#bbb}h2{margin-top:40px}a{color:inherit}</style>
<h1>Z-Image IG0: fixed-seed reference comparisons</h1>
<p>1024 x 1024, seed 42, identical initial noise, nine transformer forwards.
INT4 is dequantized reference emulation for quality assessment, not a packed runtime benchmark.
Click an image to inspect the original PNG.</p>'''
    (args.runs.parent / "review.html").write_text(page + "".join(cards))
    print(args.output)


if __name__ == "__main__":
    main()
