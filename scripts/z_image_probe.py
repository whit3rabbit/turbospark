#!/usr/bin/env python3
"""Pin and inventory Z-Image inputs without downloading weight payloads.

Run with --output docs/verification/z-image-ig0-inputs.json. Model and source
revisions are explicit constants; updating them is an evidence refresh.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request

MODEL = "Tongyi-MAI/Z-Image-Turbo"
REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
REFERENCES = {
    "diffusers": ("huggingface/diffusers", "a71e62e0d226c284b86abf518791a5ffbba064bf"),
    "mflux": ("mflux-community/mflux", "051ba9ff25c9a8a8703356053a012a0dfad3fe39"),
}
COMPONENTS = {"text_encoder", "transformer", "vae"}
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
               "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, start=None, length=None):
    headers = {"User-Agent": "turbospark-ig0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{start + length - 1}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        if start is not None:
            if r.status != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                raise ValueError("server did not honor bounded range request")
            data = r.read(length + 1)
            if len(data) != length:
                raise ValueError("incomplete or oversized range response")
            return data
        return r.read()


def validate_header(header):
    tensors = {}
    end = 0
    for name, tensor in sorted(header.items(), key=lambda kv: kv[1].get("data_offsets", [0])[0]):
        if name == "__metadata__":
            continue
        shape, dtype, offsets = tensor["shape"], tensor["dtype"], tensor["data_offsets"]
        if dtype not in DTYPE_BYTES or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"invalid shape/dtype: {name}")
        size = math.prod(shape) * DTYPE_BYTES[dtype]
        if offsets != [end, end + size]:
            raise ValueError(f"noncontiguous or incorrect tensor offsets: {name}")
        end += size
        tensors[name] = {"shape": shape, "dtype": dtype, "bytes": size, "data_offsets": offsets}
    return tensors, end


def validate_inventory(report):
    if report["model"]["revision"] != REVISION:
        raise ValueError("unexpected checkpoint revision")
    required = {"model_index.json", "scheduler/scheduler_config.json", "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json"} | {c + "/config.json" for c in COMPONENTS}
    if not required.issubset(report["files"]):
        raise ValueError("missing component configuration or tokenizer")
    for label, (_, revision) in REFERENCES.items():
        if report["references"].get(label, {}).get("revision") != revision:
            raise ValueError("unexpected reference revision")
    seen = set()
    for component in COMPONENTS:
        shards = [v for k, v in report["weights"].items() if k.startswith(component + "/")]
        if not shards:
            raise ValueError(f"missing component: {component}")
        for shard in shards:
            tensors, size = validate_header(shard["tensors"])
            if size != shard["tensor_bytes"] or not tensors:
                raise ValueError("invalid tensor byte total")
            for name in tensors:
                identity = (component, name)
                if identity in seen:
                    raise ValueError(f"duplicate tensor: {identity}")
                seen.add(identity)
        index_path = component + ("/model.safetensors.index.json" if component == "text_encoder"
                                  else "/diffusion_pytorch_model.safetensors.index.json")
        if index_path in report["files"]:
            weight_map = report["files"][index_path]["json"]["weight_map"]
            actual = {name: path.split("/")[-1] for path, shard in report["weights"].items()
                      if path.startswith(component + "/") for name in shard["tensors"]}
            if weight_map != actual:
                raise ValueError(f"index/header mismatch: {component}")


def probe(cache):
    cache.mkdir(parents=True, exist_ok=True)
    api = json.loads(fetch(f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"))
    if api["sha"] != REVISION:
        raise ValueError("model API returned a different revision")
    report = {"schema": 1, "model": {"repo": MODEL, "revision": REVISION,
              "license": api.get("cardData", {}).get("license")}, "files": {}, "weights": {}, "references": {}}
    for item in api["siblings"]:
        path = item["rfilename"]
        if path.startswith("assets/") or path == ".gitattributes":
            continue
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{path}"
        if path.endswith(".safetensors"):
            prefix = fetch(url, 0, 8)
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= 32 * 1024 * 1024:
                raise ValueError("unbounded safetensors header")
            # A distinct query avoids caches reusing the first range response.
            raw = fetch(url + "?header=1", 8, length)
            tensors, size = validate_header(json.loads(raw))
            if item["size"] != 8 + length + size:
                raise ValueError(f"file size/header mismatch: {path}")
            report["weights"][path] = {"file_bytes": item["size"], "sha256": item["lfs"]["sha256"],
                "payload_hash_verified": False, "header_sha256": digest(raw),
                "header_bytes": length, "tensor_bytes": size, "tensors": tensors}
        else:
            raw = fetch(url)
            dest = cache / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            row = {"bytes": len(raw), "sha256": digest(raw)}
            if path.endswith(".json") and "tokenizer" not in path:
                row["json"] = json.loads(raw)
            report["files"][path] = row
        print(path, flush=True)
    for label, (repo, revision) in REFERENCES.items():
        tree = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"))
        paths = [x["path"] for x in tree["tree"] if x["type"] == "blob" and (
            x["path"] in {"LICENSE", "pyproject.toml"} or
            ("/z_image/" in x["path"] and x["path"].endswith(".py")) or
            x["path"].endswith(("transformer_z_image.py", "scheduling_flow_match_euler_discrete.py")))]
        files = {}
        for path in paths:
            raw = fetch(f"https://raw.githubusercontent.com/{repo}/{revision}/{path}")
            dest = cache / label / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            files[path] = {"bytes": len(raw), "sha256": digest(raw)}
        report["references"][label] = {"repo": repo, "revision": revision, "files": files}
    validate_inventory(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("target/ig0/inputs"))
    args = parser.parse_args()
    result = probe(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
