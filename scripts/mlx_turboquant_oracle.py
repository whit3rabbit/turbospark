#!/usr/bin/env python3
"""Ports mlx-vlm's `_TurboQuantMSECodec` (turboquant.py) into a Rust oracle.

Runs the REAL mlx-vlm codec -- not a reimplementation -- on a fixed,
dependency-free LCG input at four (dim, bits) shapes this port's real
head_dims can reach: (64, 2), (64, 3), (128, 4), (256, 3). Prints the sign
vector, the Lloyd-Max codebook, the raw input rows, and the quantized
norms/packed-words as Rust consts into
crates/compute/tests/generated/kv_quant_oracle.rs.

Also asserts every rotated coordinate sits at least 6e-5 from every
midpoint (comfortably above the ~5.3e-5 max separation observed between
this port's f64 Lloyd-Max loop and mlx-vlm's own f32 one -- see
`kv_quant.rs`'s `1e-4` codebook tolerance), so this port's OWN codebook can
never flip which bucket a coordinate lands in. Retries are PER ROW rather
than per shape: the codebook (a 32768-point, 100-iteration Lloyd-Max fit)
is built ONCE per shape and reused, so a rejected row costs one cheap
rotate-and-compare rather than a full codebook rebuild, which is what
makes a four-figure retry budget per row affordable even in a slow
environment. Densely-packed boundaries near a peaked density's center
(e.g. dim=128 bits=4, 15 midpoints over 128 coordinates) make hitting SOME
boundary within a tiny margin common by chance -- that is what the retry
loop is for, not a sign either codebook is wrong.

Run with:
    uv run --with mlx --with numpy python3 scripts/mlx_turboquant_oracle.py

`turboquant.py` is loaded directly from the local checkout (see
CLAUDE.local.md) by FILE PATH rather than via `import mlx_vlm` -- that
package's own `__init__.py` pulls in transformers/pillow/huggingface_hub
(and, observed once, hangs on a network call from inside that closure with
no network available). Nothing this script calls
(`_TurboQuantMSECodec`, `_rht_sign_vector`, `_codebook`, `_pack_lowbit`)
reaches `.models.cache`, so that relative import is satisfied with a tiny
stub registered in `sys.modules` ahead of time -- it is never executed.

**`--numpy-fallback`: a documented DEGRADED mode, not the canonical path.**
`mx.array` CONSTRUCTION works in a headless/sandboxed shell with no working
Metal device, but any EVALUATION of one (`mx.eval`, `np.array(mx_array)`,
printing, `.item()`) hangs indefinitely -- observed in the session that
first wrote this script: no error, no timeout, on a trivial 3-element
multiply, with the sandbox both on and off. `--numpy-fallback` skips every
`mx` evaluation and instead runs a line-by-line NUMPY transliteration of
the same functions (`_beta_pdf`, `_codebook`, `_rht_sign_vector`, and the
codec's `quantize`/`_quantize_unit`/`_pack_lowbit` fallback path, all of
which are pure NumPy internally in the real file and only wrap their result
in `mx.array(...)` at the very end). This is NOT a run of mlx-vlm's own
code path and cannot be: on a machine where `mx.eval` actually completes,
drop `--numpy-fallback` and this script calls the genuine
`_TurboQuantMSECodec` instead, and the two should agree to full precision.
The generated file's header records which path produced it.
"""

import argparse
import math
import sys
import types
import importlib.util
from pathlib import Path

MLX_VLM_CHECKOUT = Path("/Users/whit3rabbit/Documents/GitHub/mlx-v/mlx-vlm")

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "crates/compute/tests/generated/kv_quant_oracle.rs"
)

SHAPES = [(64, 2), (64, 3), (128, 4), (256, 3)]
ROWS_PER_SHAPE = 4
SEED_BASE = 20260906
DEFAULT_SEED = 0  # mlx-vlm's `DEFAULT_TURBOQUANT_SEED`
MIN_MIDPOINT_GAP = 6e-5
MAX_ROW_ATTEMPTS = 500


def lcg_row(dim: int, seed: int):
    """A fixed, dependency-free deterministic input row via a 64-bit LCG
    (Numerical Recipes constants) mapped to roughly [-3, 3]. Any fixed
    generator works here -- the fixture only needs to be reproducible, not
    statistically clean -- and this one is deliberately NOT
    `np.random.default_rng` so it can never share a stream with
    `_rht_sign_vector`'s own PCG64 draw.
    """
    import numpy as np

    state = seed & 0xFFFFFFFFFFFFFFFF
    vals = []
    for _ in range(dim):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        vals.append(((state >> 11) / float(1 << 53)) * 2.0 - 1.0)
    return np.array(vals, dtype=np.float32) * 3.0


def rust_f32_array(name: str, values) -> str:
    body = ", ".join(f"{float(v):.9e}f32" for v in values)
    return f"pub const {name}: [f32; {len(values)}] = [{body}];\n"


def rust_u32_array(name: str, values) -> str:
    body = ", ".join(str(int(v)) for v in values)
    return f"pub const {name}: [u32; {len(values)}] = [{body}];\n"


# ---------------------------------------------------------------------------
# Canonical path: the real mlx-vlm codec.
# ---------------------------------------------------------------------------


def _load_turboquant_standalone():
    """Loads `turboquant.py` as `mlx_vlm.turboquant` without importing the
    real `mlx_vlm` package: registers empty `mlx_vlm` / `mlx_vlm.models`
    stub packages and a `mlx_vlm.models.cache` stub carrying only the three
    names `turboquant.py` imports from it, then execs the real file under
    that package name so its `from .models.cache import ...` resolves
    against the stub instead of recursing into the real package's
    `__init__.py`.
    """
    pkg = types.ModuleType("mlx_vlm")
    pkg.__path__ = [str(MLX_VLM_CHECKOUT / "mlx_vlm")]
    models_pkg = types.ModuleType("mlx_vlm.models")
    models_pkg.__path__ = [str(MLX_VLM_CHECKOUT / "mlx_vlm" / "models")]
    cache_stub = types.ModuleType("mlx_vlm.models.cache")
    cache_stub._BaseCache = object
    cache_stub.create_attention_mask = lambda *a, **k: None
    cache_stub.create_causal_mask = lambda *a, **k: None
    sys.modules["mlx_vlm"] = pkg
    sys.modules["mlx_vlm.models"] = models_pkg
    sys.modules["mlx_vlm.models.cache"] = cache_stub

    spec = importlib.util.spec_from_file_location(
        "mlx_vlm.turboquant", MLX_VLM_CHECKOUT / "mlx_vlm" / "turboquant.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["mlx_vlm.turboquant"] = module
    spec.loader.exec_module(module)
    return module


def build_tables_mlx(dim: int, bits: int):
    """Builds the codec once: signs, codebook, midpoints. The expensive
    part (the 32768-point Lloyd-Max fit) happens here, ONCE per shape.
    """
    import numpy as np

    tq = _load_turboquant_standalone()
    codec = tq._TurboQuantMSECodec(dim=dim, bits=bits, seed=DEFAULT_SEED)
    assert codec.signs is not None, f"dim {dim} did not take the RHT path"
    signs_np = np.array(codec.signs)
    codebook_np = np.array(codec.codebook)
    midpoints_np = np.array(codec._midpoints)
    return codec, signs_np, codebook_np, midpoints_np


def quantize_row_mlx(codec, tq, row):
    """The cheap per-row part: rotate, quantize, pack ONE row. Needs a
    working `mx.eval` -- see the module docstring for what happens
    without one.
    """
    import numpy as np
    import mlx.core as mx

    row_mx = mx.array(row[None, :])
    state = codec.quantize(row_mx)
    norms_mx = state.norms.astype(mx.float32)
    mx.eval(norms_mx, state.indices)
    norm = float(np.array(norms_mx)[0])
    packed = np.array(state.indices)[0].astype(np.uint32)

    unit = row / max(float(np.linalg.norm(row)), tq._EPS)
    rotated = np.array(codec._rotate_forward(mx.array(unit[None, :])))[0]
    return norm, packed, rotated


# ---------------------------------------------------------------------------
# `--numpy-fallback`: a line-by-line transliteration of the same functions,
# with every `mx` call replaced by its `np` equivalent. See the module
# docstring for why this exists and when to stop using it.
# ---------------------------------------------------------------------------


def _rht_sign_vector_np(dim: int, seed: int):
    """Transliterated from `_rht_sign_vector`: identical RNG call, so this
    produces the SAME bytes mlx-vlm's own function would (that function is
    pure NumPy up to its final `mx.array(...)` wrap).
    """
    import numpy as np

    rng = np.random.default_rng(seed + dim * 7919)
    return rng.choice([-1.0, 1.0], size=dim).astype(np.float32)


def _beta_pdf_np(grid, dim: int):
    """Transliterated from `_beta_pdf`, verbatim."""
    import numpy as np

    if dim <= 1:
        pdf = np.ones_like(grid)
    else:
        log_coeff = math.lgamma(dim / 2) - 0.5 * math.log(math.pi) - math.lgamma((dim - 1) / 2)
        log_pdf = log_coeff + ((dim - 3) / 2) * np.log(np.clip(1.0 - grid**2, 1e-30, None))
        pdf = np.exp(log_pdf - np.max(log_pdf))
    pdf_sum = pdf.sum()
    if pdf_sum == 0:
        return np.full_like(grid, 1.0 / len(grid))
    return pdf / pdf_sum


def _codebook_np(dim: int, bits: int):
    """Transliterated from `_codebook`, verbatim (grid, quantile init,
    100-iteration Lloyd-Max, right-inclusive last bucket)."""
    import numpy as np

    if bits <= 0:
        return np.zeros((0,), dtype=np.float32)
    levels = 1 << bits
    if dim <= 1:
        return np.linspace(-1.0, 1.0, levels, dtype=np.float32)

    grid = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, 32768, dtype=np.float32)
    weights = _beta_pdf_np(grid, dim)
    cdf = np.cumsum(weights)
    quantiles = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)

    for _ in range(100):
        boundaries = np.empty(levels + 1, dtype=np.float32)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        new_centroids = centroids.copy()
        for i in range(levels):
            if i == levels - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            else:
                mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            bucket_weights = weights[mask]
            if bucket_weights.size == 0:
                continue
            total_weight = bucket_weights.sum()
            if total_weight > 0:
                new_centroids[i] = np.sum(bucket_weights * grid[mask]) / total_weight
        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids.astype(np.float32)


def _hadamard_transform_np(y, scale: float):
    """A normalized Walsh-Hadamard transform matching `mx.hadamard_transform`'s
    convention (orthogonal, `scale` applied once to the whole transform
    rather than per stage). `y.shape[-1]` must be a power of two -- true of
    every shape this script uses, so mlx-vlm's own pad-to-next-power-of-two
    branch is not needed here either.
    """
    import numpy as np

    n = y.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0
    out = y.astype(np.float64).copy()
    h = 1
    while h < n:
        out = out.reshape(*out.shape[:-1], n // (2 * h), 2, h)
        a = out[..., 0, :]
        b = out[..., 1, :]
        lo = a + b
        hi = a - b
        out = np.stack([lo, hi], axis=-2).reshape(*out.shape[:-3], n)
        h *= 2
    return (out * scale).astype(np.float32)


def _rht_forward_np(x, signs):
    d = signs.shape[0]
    return _hadamard_transform_np(x * signs, 1.0 / math.sqrt(d))


def _pack_lowbit_np(indices, bits: int):
    """Transliterated from `_pack_lowbit`'s pure-Python fallback loop (the
    branch taken when no Metal custom kernel is available), with `mx`
    arrays replaced by `np` ones.
    """
    import numpy as np

    length = indices.shape[-1]
    packed_width = (length * bits + 31) // 32
    flat = indices.reshape(-1, length).astype(np.uint32)
    packed = np.zeros((flat.shape[0], packed_width), dtype=np.uint32)
    for idx in range(length):
        bit_offset = idx * bits
        word_idx = bit_offset // 32
        offset = bit_offset % 32
        packed[:, word_idx] |= flat[:, idx] << np.uint32(offset)
        spill = offset + bits - 32
        if spill > 0:
            packed[:, word_idx + 1] |= flat[:, idx] >> np.uint32(bits - spill)
    return packed.reshape(*indices.shape[:-1], packed_width)


def build_tables_numpy(dim: int, bits: int):
    """The expensive part of the numpy fallback: builds signs, codebook,
    midpoints ONCE per shape."""
    signs_np = _rht_sign_vector_np(dim, DEFAULT_SEED)
    codebook_np = _codebook_np(dim, bits)
    midpoints_np = 0.5 * (codebook_np[:-1] + codebook_np[1:])
    return signs_np, codebook_np, midpoints_np


def quantize_row_numpy(signs_np, midpoints_np, bits: int, row):
    """The cheap per-row part of the numpy fallback."""
    import numpy as np

    eps = 1e-6
    norm = float(np.linalg.norm(row.astype(np.float32)))
    unit = row / max(norm, eps)
    rotated = _rht_forward_np(unit, signs_np)

    indices = np.zeros(rotated.shape, dtype=np.uint32)
    for m in midpoints_np:
        indices = indices + (rotated > m).astype(np.uint32)
    packed = _pack_lowbit_np(indices, bits)
    return norm, packed, rotated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--numpy-fallback",
        action="store_true",
        help="Skip mlx entirely; see the module docstring's DEGRADED MODE note.",
    )
    args = parser.parse_args()

    import numpy as np

    provenance = (
        "numpy-fallback transliteration (mx.eval was not usable when this "
        "was generated -- see the module docstring in "
        "scripts/mlx_turboquant_oracle.py; RE-RUN WITHOUT --numpy-fallback "
        "on a machine with a working mlx backend before trusting this "
        "fixture for anything beyond structural review)"
        if args.numpy_fallback
        else "the real mlx-vlm `_TurboQuantMSECodec`"
    )

    out_lines = [
        "// GENERATED by scripts/mlx_turboquant_oracle.py, from",
        f"// {provenance}.",
        "// Do not hand-edit; regenerate with the command in that script's docstring.",
        "//",
        "// `include!`d into a test module body, so no inner (`#![...]`) attribute",
        "// belongs here -- it would annotate whatever const follows it, not the file.",
        "",
    ]

    if args.numpy_fallback:
        tq = None
    else:
        tq = _load_turboquant_standalone()

    for shape_idx, (dim, bits) in enumerate(SHAPES):
        if args.numpy_fallback:
            signs_np, codebook_np, midpoints_np = build_tables_numpy(dim, bits)
            codec = None
        else:
            codec, signs_np, codebook_np, midpoints_np = build_tables_mlx(dim, bits)

        rows = []
        norms = []
        packed_rows = []
        worst_gap_overall = float("inf")

        for r in range(ROWS_PER_SHAPE):
            best_gap = -1.0
            best = None
            for attempt in range(MAX_ROW_ATTEMPTS):
                row = lcg_row(dim, SEED_BASE + shape_idx * 1000 + r * 10000 + attempt)

                if args.numpy_fallback:
                    norm, packed, rotated = quantize_row_numpy(signs_np, midpoints_np, bits, row)
                else:
                    norm, packed, rotated = quantize_row_mlx(codec, tq, row)

                gap = float(np.min(np.abs(rotated[:, None] - midpoints_np[None, :])))
                if gap > best_gap:
                    best_gap = gap
                    best = (row, norm, packed)
                if gap >= MIN_MIDPOINT_GAP:
                    break
            else:
                raise AssertionError(
                    f"dim={dim} bits={bits} row={r}: could not clear "
                    f"{MIN_MIDPOINT_GAP} in {MAX_ROW_ATTEMPTS} attempts "
                    f"(best gap={best_gap})"
                )

            row, norm, packed = best
            rows.append(row)
            norms.append(norm)
            packed_rows.append(packed)
            worst_gap_overall = min(worst_gap_overall, best_gap)

        prefix = f"D{dim}B{bits}"
        out_lines.append(
            f"// dim={dim} bits={bits}, seed={DEFAULT_SEED}, "
            f"{ROWS_PER_SHAPE} rows, min midpoint gap {worst_gap_overall:.6e}"
        )
        out_lines.append(rust_f32_array(f"{prefix}_SIGNS", signs_np.tolist()))
        out_lines.append(rust_f32_array(f"{prefix}_CODEBOOK", codebook_np.tolist()))
        out_lines.append(
            rust_f32_array(
                f"{prefix}_INPUT_ROWS", np.concatenate(rows).astype(np.float32).tolist()
            )
        )
        out_lines.append(rust_f32_array(f"{prefix}_NORMS", norms))
        out_lines.append(
            rust_u32_array(f"{prefix}_PACKED", np.concatenate(packed_rows).tolist())
        )
        out_lines.append(f"pub const {prefix}_PACKED_WIDTH: usize = {packed_rows[0].shape[-1]};\n")
        out_lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(out_lines))
    print(f"wrote {OUT_PATH} ({provenance})")


if __name__ == "__main__":
    main()
