#!/usr/bin/env python3
"""Download immutable reference weights and verify every component payload."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from z_image_capture import verify_weights
from z_image_evidence import sha256
from z_image_probe import COMPONENTS, MODEL, REVISION


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("target/ig0/model"))
    p.add_argument("--inventory", type=Path, default=Path("docs/verification/z-image-ig0-inputs.json"))
    p.add_argument("--output", type=Path, default=Path("docs/verification/z-image-ig0-download.json"))
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if not args.verify_only:
        snapshot_download(MODEL, revision=REVISION, local_dir=args.model, max_workers=3,
                          allow_patterns=["*.json", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*", "scheduler/*"])
    hashes = {}
    for component in sorted(COMPONENTS):
        hashes.update(verify_weights(args.model, args.inventory, component))
        print(component + " payload verified", flush=True)
    report = {"model": MODEL, "revision": REVISION, "inventory_sha256": sha256(args.inventory),
              "verified_payload_sha256": hashes, "all_component_payloads_verified": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
