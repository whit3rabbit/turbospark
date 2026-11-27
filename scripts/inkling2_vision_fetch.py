#!/usr/bin/env python3
"""Range-read the Inkling 2 Mini vision tensors (and nothing else) from the
pinned Hugging Face shard into a small safetensors file, original names and
BF16 bytes unchanged, for `scripts/inkling2_vision_reference.py`.

The checkpoint is one 18 GB shard; the 422 `model.visual.*` tensors are about
636 MB of it. This reads the safetensors header with two Range requests, then
fetches each run of adjacent vision tensors with one Range request each.

Integrity: Hugging Face publishes a sha256 for the whole shard only, so a
partial read cannot be checked against it. The script checks every tensor's
byte length against its dtype and shape, and records the revision and the
header's own byte count in `__metadata__`.

Usage:
    python3 -I scripts/inkling2_vision_fetch.py --out ~/models/inkling2-vision/vision.safetensors
"""

import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

REPO = "thinkingmachines/Inkling-2-Mini"
REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
SHARD = "model-00001-of-00001.safetensors"
PREFIX = "model.visual."
EXPECTED_TENSORS = 422
GAP = 1 << 20  # merge runs separated by less than this many bytes
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def url(repo, revision, name):
    return f"https://huggingface.co/{repo}/resolve/{revision}/{name}"


def get_range(target, start, end_inclusive):
    req = urllib.request.Request(
        target,
        headers={"Range": f"bytes={start}-{end_inclusive}", "User-Agent": "turbospark-vision-fetch"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 206:
            raise RuntimeError(f"expected 206 Partial Content, got {resp.status}")
        data = resp.read()
    want = end_inclusive - start + 1
    if len(data) != want:
        raise RuntimeError(f"short read: {len(data)} of {want} bytes at {start}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args()

    target = url(args.repo, args.revision, SHARD)
    (n,) = struct.unpack("<Q", get_range(target, 0, 7))
    header = json.loads(get_range(target, 8, 8 + n - 1))
    base = 8 + n

    names = sorted(
        (k for k in header if k.startswith(PREFIX)),
        key=lambda k: header[k]["data_offsets"][0],
    )
    if len(names) != EXPECTED_TENSORS:
        sys.exit(f"{len(names)} tensors under {PREFIX}, expected {EXPECTED_TENSORS}")
    for k in names:
        meta = header[k]
        count = 1
        for d in meta["shape"]:
            count *= d
        s, e = meta["data_offsets"]
        if meta["dtype"] not in DTYPE_BYTES or e - s != count * DTYPE_BYTES[meta["dtype"]]:
            sys.exit(f"{k}: {meta['dtype']} {meta['shape']} spans {e - s} bytes")

    runs = []
    for k in names:
        s, e = header[k]["data_offsets"]
        if runs and s - runs[-1][1] < GAP:
            runs[-1][1] = max(runs[-1][1], e)
            runs[-1][2].append(k)
        else:
            runs.append([s, e, [k]])

    blobs = {}
    total = sum(e - s for s, e, _ in runs)
    done = 0
    for s, e, keys in runs:
        data = get_range(target, base + s, base + e - 1)
        for k in keys:
            ks, ke = header[k]["data_offsets"]
            blobs[k] = data[ks - s : ke - s]
        done += e - s
        print(f"\r{done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)

    out_header = {
        "__metadata__": {
            "source": f"{args.repo}@{args.revision}/{SHARD}",
            "source_header_bytes": str(n),
        }
    }
    offset = 0
    for k in names:
        size = len(blobs[k])
        out_header[k] = {
            "dtype": header[k]["dtype"],
            "shape": header[k]["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(out_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for k in names:
            fh.write(blobs[k])
    tmp.replace(out)
    print(f"{out}: {len(names)} tensors, {offset / 1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
