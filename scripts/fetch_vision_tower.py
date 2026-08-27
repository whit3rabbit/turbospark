#!/usr/bin/env python3
"""Fetch ONLY the `vision_tower.*` tensors out of a real HF safetensors
checkpoint, by ranged HTTP, without downloading the (multi-GiB) text trunk.

Used for the vision-support Phase 0 fact-finding (`docs/VISION_PHASE0.md`)
and by `vision_tower_probe.py`. Reads the checkpoint's own
`model.safetensors.index.json` if present (multi-shard installs) or falls
back to a single `model.safetensors` file, fetches that shard's JSON header
(a small ranged read: 8-byte length prefix + the header itself), then issues
ONE ranged GET spanning the byte offsets of every `vision_tower.*` tensor
(they are contiguous in every checkpoint measured so far -- if a future
checkpoint interleaves them with trunk tensors this will over-fetch rather
than miss data, since it spans min..max offset).

Writes two files to --out-dir:
    vision_tower.bin           raw tensor bytes, re-based to offset 0
    vision_tower_header.json   {name: {dtype, shape, data_offsets}} with
                                offsets relative to vision_tower.bin

Usage:
    uv run --python 3.12 --with requests \
      scripts/fetch_vision_tower.py prism-ml/Bonsai-27B-mlx-1bit /tmp/vision-probe
"""

import json
import struct
import sys
import urllib.request
from pathlib import Path


def _ranged_get(url: str, start: int, end_inclusive: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
    return urllib.request.urlopen(req).read()


def _read_header(shard_url: str) -> dict:
    n = struct.unpack("<Q", _ranged_get(shard_url, 0, 7))[0]
    return json.loads(_ranged_get(shard_url, 8, 7 + n)), 8 + n


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    repo, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    base = f"https://huggingface.co/{repo}/resolve/main"
    try:
        index = json.loads(urllib.request.urlopen(f"{base}/model.safetensors.index.json").read())
        weight_map = index["weight_map"]
        shards = sorted({v for k, v in weight_map.items() if k.startswith("vision_tower.")})
    except urllib.error.HTTPError:
        shards = ["model.safetensors"]

    if not shards:
        print("no vision_tower.* tensors found in the index", file=sys.stderr)
        sys.exit(1)

    all_vis: dict = {}
    all_bytes = bytearray()
    for shard in shards:
        shard_url = f"{base}/{shard}"
        hdr, data_base = _read_header(shard_url)
        vis = {k: v for k, v in hdr.items() if k.startswith("vision_tower.")}
        if not vis:
            continue
        lo = min(v["data_offsets"][0] for v in vis.values())
        hi = max(v["data_offsets"][1] for v in vis.values())
        print(f"{shard}: {len(vis)} vision tensors, {(hi - lo) / 2**20:.1f} MiB")
        blob = _ranged_get(shard_url, data_base + lo, data_base + hi - 1)
        rebase = len(all_bytes)
        for name, meta in vis.items():
            s, e = meta["data_offsets"]
            all_vis[name] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [s - lo + rebase, e - lo + rebase],
            }
        all_bytes.extend(blob)

    (out_dir / "vision_tower.bin").write_bytes(all_bytes)
    json.dump(all_vis, open(out_dir / "vision_tower_header.json", "w"))
    print(f"wrote {len(all_vis)} tensors, {len(all_bytes) / 2**20:.1f} MiB total, to {out_dir}")


if __name__ == "__main__":
    main()
