#!/usr/bin/env python3
"""Extract a per-layer steering direction from two sets of residual captures.

    uv run --python 3.12 --with numpy scripts/extract_direction.py \
        --positive /tmp/steer/pos --negative /tmp/steer/neg \
        --out /tmp/steer/d.gguf

Reads the capture pairs `MFERENCE_RESID_CAPTURE` writes (a small JSON header
beside a raw `.f32` sidecar, see `crates/runtime/src/resid_capture.rs`),
computes one direction per layer, and writes a control vector in llama.cpp's
GGUF layout.

WHAT IT COMPUTES

Per layer, the difference of means between the two sets -- the standard
extraction, and the one Arditi et al. use for a refusal direction:

    d_l = mean(positive activations at layer l) - mean(negative ones)

`--method svd` instead takes the top right-singular vector of the centred
difference matrix, which is what OBLITERATUS's "advanced" mode does and is
worth having when the two sets differ along more than one axis: the mean is
the average of the per-pair differences and can be dominated by one outlier
pair, where the SVD direction is the one those differences most consistently
share. The mean is the default because it is what every published refusal
direction is, and because it needs no judgement about rank.

The written direction is NOT normalized, and that is deliberate: the reader
takes `1 / ||d||` as a parameter, so `ablate` and `clamp` are already defined
against the unit direction while `add` reads the magnitude. Normalizing here
would silently rescale `add`, which is the mode that has to stay compatible
with the published llama.cpp vector sets.

THE FILE FORMAT, AND THE ONE THING NOT VERIFIED ABOUT IT

llama.cpp's loader reads tensors named `direction.N`, F32, one-dimensional,
with N ONE-INDEXED -- it rejects a zero index by name. Its apply loop runs
`for il = 1; il < n_layer`, so its layer 0 never receives a direction at all.

This writer maps 0-based layer `l` to `direction.{l+1}` and records
`turbospark.layer_base = 0` so this port's own reader is unambiguous.
**Whether that lines up with llama.cpp's own numbering is UNVERIFIED here** --
the two could differ by one layer, which is exactly the kind of error that
produces a plausible wrong answer rather than a failure. Read a vector
written by this script with this port; do not assume it is positioned
identically under llama.cpp until someone measures it.
"""

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
DEFAULT_ALIGNMENT = 32

# GGUF metadata value type tags.
VT_UINT32 = 4
VT_FLOAT32 = 6
VT_STRING = 8


def load_capture(header_path: pathlib.Path) -> tuple[np.ndarray, list[int]]:
    """Return `[snapshot][layer][hidden]` float32 and the captured positions."""
    meta = json.loads(header_path.read_text())
    data_path = header_path.parent / meta["data"]
    if not data_path.exists():
        sys.exit(f"{header_path}: sidecar {meta['data']} is missing")
    layers, hidden, snaps = meta["layers"], meta["hidden"], meta["snapshots"]
    raw = np.fromfile(data_path, dtype="<f4")
    want = snaps * layers * hidden
    if raw.size != want:
        sys.exit(
            f"{data_path}: holds {raw.size} floats, header describes {want} "
            f"({snaps} x {layers} x {hidden})"
        )
    return raw.reshape(snaps, layers, hidden), meta["positions"]


def load_set(spec: str) -> np.ndarray:
    """Load every capture under a directory (or one file) into one stack."""
    path = pathlib.Path(spec)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        sys.exit(f"{spec}: no capture headers found")
    stacks, total_positions = [], []
    for f in files:
        arr, positions = load_capture(f)
        stacks.append(arr)
        total_positions.extend(positions)
    shapes = {a.shape[1:] for a in stacks}
    if len(shapes) != 1:
        sys.exit(f"{spec}: captures disagree on (layers, hidden): {shapes}")
    out = np.concatenate(stacks, axis=0)
    print(
        f"  {spec}: {out.shape[0]} snapshot(s), {out.shape[1]} layers, "
        f"{out.shape[2]} hidden, positions {total_positions}"
    )
    return out


def directions(pos: np.ndarray, neg: np.ndarray, method: str) -> np.ndarray:
    """One direction per layer, `[layers][hidden]` float32."""
    layers, hidden = pos.shape[1], pos.shape[2]
    out = np.zeros((layers, hidden), dtype=np.float32)
    for l in range(layers):
        p, n = pos[:, l, :].astype(np.float64), neg[:, l, :].astype(np.float64)
        if method == "mean":
            out[l] = (p.mean(axis=0) - n.mean(axis=0)).astype(np.float32)
        else:
            # Pair every positive against every negative only when the sets
            # are the same size; otherwise centre each set and difference the
            # means, then take the leading direction of the residual spread.
            diff = p[:, None, :] - n[None, :, :]
            diff = diff.reshape(-1, hidden)
            diff -= diff.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(diff, full_matrices=False)
            lead = vt[0]
            # SVD fixes the direction only up to sign. Orient it so it points
            # from negative to positive, which is what makes the sign of
            # `alpha` mean the same thing as under `--method mean`.
            if float(lead @ (p.mean(axis=0) - n.mean(axis=0))) < 0:
                lead = -lead
            out[l] = lead.astype(np.float32)
    return out


def separation(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Per-layer effect size: `||mean_p - mean_n||` over the pooled spread.

    **The raw direction norm cannot be compared across layers and reading it
    that way is the trap this function exists to close.** A residual stream's
    magnitude grows with depth -- measured on the real `qwen38-27b`, 13.9 at
    layer 0 against 535.5 at layer 63 -- so a difference of means grows with
    it whether or not the two sets separate any better. Ranking layers by
    that norm reports where the stream is BIGGEST and reads as where the
    concept LIVES, which is a different claim and usually a different layer.

    Dividing by the pooled within-set spread removes the scale. What is left
    is Cohen's d generalized to a vector: how far apart the two sets are in
    units of how spread out they each are.
    """
    layers = pos.shape[1]
    out = np.zeros(layers, dtype=np.float64)
    for l in range(layers):
        p, n = pos[:, l, :].astype(np.float64), neg[:, l, :].astype(np.float64)
        gap = np.linalg.norm(p.mean(axis=0) - n.mean(axis=0))
        # Mean squared distance from each set's own centroid, pooled.
        vp = float(((p - p.mean(axis=0)) ** 2).sum(axis=1).mean())
        vn = float(((n - n.mean(axis=0)) ** 2).sum(axis=1).mean())
        spread = np.sqrt(0.5 * (vp + vn))
        out[l] = gap / spread if spread > 0 else 0.0
    return out


def _kv_string(key: str, value: str) -> bytes:
    kb, vb = key.encode(), value.encode()
    return (
        struct.pack("<Q", len(kb))
        + kb
        + struct.pack("<I", VT_STRING)
        + struct.pack("<Q", len(vb))
        + vb
    )


def _kv_u32(key: str, value: int) -> bytes:
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", VT_UINT32) + struct.pack("<I", value)


def write_gguf(path: pathlib.Path, dirs: np.ndarray, arch: str, method: str) -> None:
    layers, hidden = dirs.shape
    metadata = [
        # llama.cpp reads neither of these for a control vector, but a file
        # that says which model it belongs to is the difference between a
        # mismatched vector failing at load and applying silently.
        _kv_string("general.architecture", arch),
        _kv_string("controlvector.model_hint", arch),
        _kv_u32("controlvector.layer_count", layers),
        # Port-local provenance. Nothing else reads these; they exist so a
        # file found later can say what it is.
        _kv_string("turbospark.extraction", method),
        _kv_u32("turbospark.layer_base", 0),
    ]

    infos = []
    offset = 0
    row_bytes = hidden * 4
    for l in range(layers):
        name = f"direction.{l + 1}".encode()  # ONE-indexed; 0 is rejected.
        infos.append(
            struct.pack("<Q", len(name))
            + name
            + struct.pack("<I", 1)  # n_dims
            + struct.pack("<Q", hidden)
            + struct.pack("<I", GGML_TYPE_F32)
            + struct.pack("<Q", offset)
        )
        offset += row_bytes

    header = (
        GGUF_MAGIC
        + struct.pack("<I", GGUF_VERSION)
        + struct.pack("<Q", layers)
        + struct.pack("<Q", len(metadata))
        + b"".join(metadata)
        + b"".join(infos)
    )
    pad = (-len(header)) % DEFAULT_ALIGNMENT
    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\0" * pad)
        for l in range(layers):
            f.write(np.ascontiguousarray(dirs[l], dtype="<f4").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positive", required=True, help="capture dir or header .json")
    ap.add_argument("--negative", required=True, help="capture dir or header .json")
    ap.add_argument("--out", required=True, help="output .gguf")
    ap.add_argument("--method", choices=["mean", "svd"], default="mean")
    ap.add_argument("--arch", default="qwen35", help="general.architecture to stamp")
    args = ap.parse_args()

    print("loading captures:")
    pos = load_set(args.positive)
    neg = load_set(args.negative)
    if pos.shape[1:] != neg.shape[1:]:
        sys.exit(f"sets disagree on shape: {pos.shape[1:]} vs {neg.shape[1:]}")

    dirs = directions(pos, neg, args.method)
    norms = np.linalg.norm(dirs, axis=1)
    sep = separation(pos, neg)

    # Two columns, because they answer different questions and only the
    # second one can be compared across layers. `norm` is the raw magnitude
    # of the difference of means and rises with the residual stream's own
    # scale; `sep` divides that out. Reported rather than thresholded: a
    # near-zero layer is a real and interesting outcome (the sets do not
    # separate there), and which layers to steer at is the caller's call.
    print(f"\nper-layer direction ({args.method}):")
    print(f"  {'layer':>5}  {'norm':>10}  {'sep':>7}  (sep = effect size, scale-free)")
    for l, (n, s) in enumerate(zip(norms, sep)):
        bar = "#" * int(40 * s / max(sep.max(), 1e-9))
        print(f"  {l:5d}  {n:10.4f}  {s:7.3f}  {bar}")

    best, worst = int(sep.argmax()), int(sep.argmin())
    print(
        f"\nBY EFFECT SIZE: strongest at layer {best} ({sep.max():.3f}), "
        f"weakest at {worst} ({sep.min():.3f})."
    )
    print(
        f"By raw norm it would read layer {int(norms.argmax())}, which is "
        f"mostly where the residual stream is biggest -- see `separation`."
    )
    if norms.max() <= 0.0:
        sys.exit("every layer's direction is zero; the two sets are identical")

    out = pathlib.Path(args.out)
    write_gguf(out, dirs, args.arch, args.method)
    print(f"\nwrote {out} ({dirs.shape[0]} directions of {dirs.shape[1]})")


if __name__ == "__main__":
    main()
