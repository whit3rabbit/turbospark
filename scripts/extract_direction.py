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


def stream_share(pos: np.ndarray, neg: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Per layer, `||d_l|| / ||x_l||`: the share of the residual an ablation
    at `alpha = 1` removes.

    **This is the quantity that predicts the COLLAPSE, and it is the third
    distinct thing `||d_l||` can be divided by in this file.** `separation`
    divides by the within-set spread to ask where the concept lives; this
    divides by the stream's own magnitude to ask what removing it costs. They
    rank layers differently and neither substitutes for the other.

    The mechanism is recorded in `docs/OBLITERATION.md`: the late-layer
    residual IS the output head's input, so ablating a direction that is a
    large share of it damages what the head reads. Measured on the real
    `qwen38-27b` ocean/mountain corpus, layer 63's direction norm is 116.34
    against a mean row norm of 457.9 -- 25.4% -- and `ablate` at `alpha = 1`
    over all 64 layers made the model emit end-of-turn immediately, while
    0.3 stayed coherent. The hottest layer is 59 at 27.9%, not 63.

    `x_l` is the mean activation over BOTH sets, which is the stream the edit
    will actually meet at that layer: an operator steers arbitrary prompts,
    not the extraction corpus, and the two sets' shared magnitude is the best
    estimate available offline.
    """
    layers = pos.shape[1]
    out = np.zeros(layers, dtype=np.float64)
    for l in range(layers):
        both = np.concatenate(
            (pos[:, l, :].astype(np.float64), neg[:, l, :].astype(np.float64))
        )
        stream = float(np.linalg.norm(both, axis=1).mean())
        out[l] = float(np.linalg.norm(dirs[l])) / stream if stream > 0 else 0.0
    return out


def suggested_alpha(share: np.ndarray, budget: float) -> float:
    """The largest `ablate` alpha keeping the worst layer's removal under
    `budget` of the stream.

    **A CEILING, NOT A RECOMMENDATION, and the difference matters.** It says
    where the edit starts damaging what the head reads; it says nothing about
    where the edit starts WORKING, which is a property of the direction and
    the concept and is not computable from norms. An alpha under this that
    steers nothing is an ordinary outcome.

    It is derived per model AND per direction rather than looked up. The
    obvious alternative -- a table of known-good alphas per model family, or
    a tier by parameter count -- cannot work here, because what collapses the
    turn is this ratio and a parameter count does not predict it: two
    directions extracted from one checkpoint can differ severalfold in it.

    The budget is the one number here that is a judgement rather than a
    measurement, and it is deliberately loose. It is set from the two
    measured points named in `stream_share` (22% collapsed, 6.5% did not) and
    a third of the way between them is not a threshold anyone has bracketed.
    Widen or ignore it; it is printed with its own reasoning for that reason.
    """
    worst = float(share.max())
    return budget / worst if worst > 0 else float("inf")


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
    ap.add_argument(
        "--alpha-budget",
        type=float,
        default=0.10,
        help=(
            "share of the residual an ablation may remove at the hottest "
            "layer, used only to print a suggested alpha ceiling (default "
            "0.10). A judgement, not a measured threshold -- see "
            "`suggested_alpha`."
        ),
    )
    args = ap.parse_args()

    print("loading captures:")
    pos = load_set(args.positive)
    neg = load_set(args.negative)
    if pos.shape[1:] != neg.shape[1:]:
        sys.exit(f"sets disagree on shape: {pos.shape[1:]} vs {neg.shape[1:]}")

    dirs = directions(pos, neg, args.method)
    norms = np.linalg.norm(dirs, axis=1)
    sep = separation(pos, neg)
    share = stream_share(pos, neg, dirs)

    # THREE columns, and each divides `norm` by something different because
    # each answers a different question. `norm` is the raw magnitude and
    # rises with the residual stream's own scale, so it can be compared
    # across layers only by accident. `sep` divides by the within-set spread:
    # where does the CONCEPT live. `share` divides by the stream's own
    # magnitude: what does removing it COST. A layer can rank high on one and
    # low on another, which is the whole reason all three are printed.
    #
    # Reported rather than thresholded: a near-zero layer is a real and
    # interesting outcome (the sets do not separate there), and which layers
    # to steer at is the caller's call.
    print(f"\nper-layer direction ({args.method}):")
    print(
        f"  {'layer':>5}  {'norm':>10}  {'sep':>7}  {'share':>7}   "
        f"(sep = effect size, scale-free; share = ||d||/||x||)"
    )
    for l, (n, s, sh) in enumerate(zip(norms, sep, share)):
        bar = "#" * int(40 * s / max(sep.max(), 1e-9))
        print(f"  {l:5d}  {n:10.4f}  {s:7.3f}  {sh:6.1%}  {bar}")

    best, worst = int(sep.argmax()), int(sep.argmin())
    print(
        f"\nBY EFFECT SIZE: strongest at layer {best} ({sep.max():.3f}), "
        f"weakest at {worst} ({sep.min():.3f})."
    )
    print(
        f"By raw norm it would read layer {int(norms.argmax())}, which is "
        f"mostly where the residual stream is biggest -- see `separation`."
    )

    # THE ALPHA CEILING, derived rather than looked up. See `suggested_alpha`
    # on why a per-model table cannot do this job: the collapse is a function
    # of THIS direction against THIS stream, and two directions off one
    # checkpoint differ in it.
    hottest = int(share.argmax())
    budget = args.alpha_budget
    cap = suggested_alpha(share, budget)
    print(
        f"\nBY STREAM SHARE: layer {hottest} carries the largest share at "
        f"{share[hottest]:.1%}; ablating there at alpha 1 removes that much "
        f"of the residual."
    )
    if cap >= 1.0:
        # NOT "full ablation is fine". A budget loose enough to admit alpha
        # 1.0 has stopped bounding anything, and on the corpus this was
        # written against a 30% budget admits exactly the operating point
        # measured to COLLAPSE the turn. Report that the budget implied no
        # ceiling and say which number was the judgement.
        print(
            f"  No ceiling implied AT THIS BUDGET: the hottest layer's "
            f"{share[hottest]:.1%} is already inside {budget:.0%}."
        )
        print(
            f"  That is a statement about --alpha-budget, not about safety. "
            f"On this page's own\n  corpus a 30% budget admits alpha 1.0, "
            f"which is the arm measured to collapse the turn."
        )
    else:
        print(
            f"  SUGGESTED ABLATE CEILING: alpha <= {cap:.2f} keeps every "
            f"layer's removal under {budget:.0%} of the stream."
        )
    print(
        "  A CEILING, NOT A RECOMMENDATION: it says where the edit starts "
        "damaging what\n  the output head reads, not where it starts working. "
        "Restricting to a layer band\n  (--steering-layers) raises the usable "
        "alpha by dropping the hot layers entirely."
    )
    if norms.max() <= 0.0:
        sys.exit("every layer's direction is zero; the two sets are identical")

    out = pathlib.Path(args.out)
    write_gguf(out, dirs, args.arch, args.method)
    print(f"\nwrote {out} ({dirs.shape[0]} directions of {dirs.shape[1]})")


if __name__ == "__main__":
    main()
