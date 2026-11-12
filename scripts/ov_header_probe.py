#!/usr/bin/env python3
"""List the non-language tensors of a sharded HF safetensors repo by ranged
header reads (no weights downloaded).

Usage: python3 -I scripts/ov_header_probe.py REPO REVISION NUM_SHARDS
"""
import collections
import json
import struct
import sys
import urllib.request


def rng(url, a, b):
    r = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(r, timeout=60).read()


def main():
    repo, rev, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    base = f"https://huggingface.co/{repo}/resolve/{rev}/"
    total = 0
    for i in range(1, count + 1):
        name = f"model-{i:05d}-of-{count:05d}.safetensors"
        (hl,) = struct.unpack("<Q", rng(base + name, 0, 7))
        h = json.loads(rng(base + name, 8, 8 + hl - 1))
        keys = [k for k in h if k != "__metadata__"]
        non = [k for k in keys if not k.startswith("language_model.")]
        size = sum(h[k]["data_offsets"][1] - h[k]["data_offsets"][0] for k in non)
        total += size
        c = collections.Counter(".".join(k.split(".")[:2]) for k in non)
        print(name, len(keys), "tensors; non-LM", len(non), round(size / 1e6, 1), "MB", dict(c))
        for k in non:
            if "layers" not in k or ".layers.0." in k:
                print("   ", k, h[k]["dtype"], h[k]["shape"])
    print("tower bytes", total)


main()
