#!/usr/bin/env python3
"""Fetch only `vision_tower.*` tensors from an HF safetensors checkpoint.

The remote index, headers, and tensor metadata are untrusted. Downloads are
bounded and each tensor is streamed separately so intervening text-model data
is never fetched or retained in memory.

Usage:
    python3 scripts/fetch_vision_tower.py REPO OUT_DIR [REVISION]
"""

import json
import re
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path


INDEX_LIMIT = 16 * 2**20
HEADER_LIMIT = 64 * 2**20
TOWER_LIMIT = 2 * 2**30
READ_CHUNK = 1024 * 1024
TIMEOUT = 60
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def _read_limited(response, limit: int, description: str) -> bytes:
    data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{description} exceeds {limit} bytes")
    return data


def _open_range(url: str, start: int, end_inclusive: int):
    if start < 0 or end_inclusive < start:
        raise ValueError("invalid HTTP byte range")
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end_inclusive}"}
    )
    response = urllib.request.urlopen(request, timeout=TIMEOUT)
    status = getattr(response, "status", response.getcode())
    match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
    if status != 206 or match is None:
        response.close()
        raise ValueError("server did not honor the requested byte range")
    returned_start, returned_end, total = map(int, match.groups())
    if (returned_start, returned_end) != (start, end_inclusive):
        response.close()
        raise ValueError("server returned a different byte range")
    if total <= returned_end:
        response.close()
        raise ValueError("invalid Content-Range file length")
    return response, total


def _ranged_bytes(url: str, start: int, end_inclusive: int, limit: int) -> tuple[bytes, int]:
    size = end_inclusive - start + 1
    if size > limit:
        raise ValueError(f"requested range exceeds {limit} bytes")
    response, total = _open_range(url, start, end_inclusive)
    with response:
        data = _read_limited(response, size, "ranged response")
    if len(data) != size:
        raise ValueError("truncated ranged response")
    return data, total


def _read_header(shard_url: str) -> tuple[dict, int, int]:
    prefix, shard_size = _ranged_bytes(shard_url, 0, 7, 8)
    header_size = struct.unpack("<Q", prefix)[0]
    if header_size == 0 or header_size > HEADER_LIMIT:
        raise ValueError(f"safetensors header size {header_size} is not allowed")
    data_base = 8 + header_size
    if data_base > shard_size:
        raise ValueError("safetensors header extends beyond the shard")
    raw, confirmed_size = _ranged_bytes(shard_url, 8, data_base - 1, HEADER_LIMIT)
    if confirmed_size != shard_size:
        raise ValueError("shard length changed between ranged requests")
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    return header, data_base, shard_size


def _tensor_range(meta: object, data_size: int) -> tuple[int, int]:
    if not isinstance(meta, dict):
        raise ValueError("tensor metadata must be an object")
    offsets = meta.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("tensor data_offsets must contain two integers")
    start, end = offsets
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
        or end > data_size
    ):
        raise ValueError("tensor data_offsets are outside the shard")
    return start, end


def _stream_range(
    url: str, start: int, end_inclusive: int, expected_total: int, output
) -> None:
    response, total = _open_range(url, start, end_inclusive)
    if total != expected_total:
        response.close()
        raise ValueError("shard length changed between ranged requests")
    remaining = end_inclusive - start + 1
    with response:
        while remaining:
            chunk = response.read(min(READ_CHUNK, remaining))
            if not chunk:
                raise ValueError("truncated ranged response")
            output.write(chunk)
            remaining -= len(chunk)
        if response.read(1):
            raise ValueError("ranged response exceeded its declared length")


def _read_index(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        raw = _read_limited(response, INDEX_LIMIT, "safetensors index")
    index = json.loads(raw)
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("safetensors index has no weight_map object")
    return index


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    repo, out_dir = sys.argv[1], Path(sys.argv[2])
    revision = sys.argv[3] if len(sys.argv) == 4 else "main"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = f"https://huggingface.co/{repo}/resolve/{revision}"
    try:
        index = _read_index(f"{base}/model.safetensors.index.json")
        shards = set()
        for name, shard in index["weight_map"].items():
            if not isinstance(name, str):
                raise ValueError("safetensors index contains a non-string tensor name")
            if name.startswith("vision_tower."):
                if not isinstance(shard, str):
                    raise ValueError("safetensors index contains a non-string shard name")
                shards.add(shard)
        shards = sorted(shards)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        shards = ["model.safetensors"]
    else:
        # THE INDEX IS A CLAIM, NOT A FACT (docs/QWEN3VL_PHASE0.md section
        # 0): mlx-community/Qwen3-VL-4B-Instruct-4bit's own index names two
        # shards from an earlier upload while the repo carries one
        # consolidated model.safetensors, so every index-derived name 404s.
        # Cross-check against the repository's real file list, drop the
        # stale names, and fall back to the single-file convention when
        # nothing survives -- the same order the catalog's stream path runs.
        with urllib.request.urlopen(
            f"https://huggingface.co/api/models/{repo}", timeout=60
        ) as listing:
            files = {s["rfilename"] for s in json.load(listing).get("siblings", [])}
        shards = [s for s in shards if s in files]
        if not shards:
            shards = ["model.safetensors"]

    if not shards:
        raise ValueError("no vision_tower.* tensors found in the index")

    all_vis: dict = {}
    total_bytes = 0
    output_path = out_dir / "vision_tower.bin"
    temporary_path = output_path.with_suffix(".bin.part")
    try:
        with temporary_path.open("wb") as output:
            for shard in shards:
                shard_url = f"{base}/{shard}"
                header, data_base, shard_size = _read_header(shard_url)
                vision = {
                    name: meta
                    for name, meta in header.items()
                    if name.startswith("vision_tower.")
                }
                ranges = []
                for name, meta in vision.items():
                    start, end = _tensor_range(meta, shard_size - data_base)
                    ranges.append((start, end, name, meta))
                ranges.sort()
                shard_bytes = sum(end - start for start, end, _, _ in ranges)
                if total_bytes + shard_bytes > TOWER_LIMIT:
                    raise ValueError(f"vision tower exceeds {TOWER_LIMIT} bytes")
                print(
                    f"{shard}: {len(ranges)} vision tensors, "
                    f"{shard_bytes / 2**20:.1f} MiB"
                )

                for start, end, name, meta in ranges:
                    rebase = total_bytes
                    _stream_range(
                        shard_url,
                        data_base + start,
                        data_base + end - 1,
                        shard_size,
                        output,
                    )
                    total_bytes += end - start
                    all_vis[name] = {
                        "dtype": meta["dtype"],
                        "shape": meta["shape"],
                        "data_offsets": [rebase, total_bytes],
                    }
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    with (out_dir / "vision_tower_header.json").open("w") as header_file:
        json.dump(all_vis, header_file)
    print(
        f"wrote {len(all_vis)} tensors, {total_bytes / 2**20:.1f} MiB "
        f"total, to {out_dir}"
    )


if __name__ == "__main__":
    main()
