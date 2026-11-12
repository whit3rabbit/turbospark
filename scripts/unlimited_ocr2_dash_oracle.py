#!/usr/bin/env python3
"""Generate `crates/compute/tests/generated/dash_oracle.rs` from mlx-vlm.

Unlimited-OCR-2's router, page bookkeeping and page2d RoPE are host arithmetic
in this port, and a wrong route reads as fluent, slightly wrong OCR. So the CPU
reference is pinned against upstream rather than against a re-derivation.

Needs no checkpoint: the inputs are seeded and the parameters are the shipped
`sparse_attention_config`. Run with a venv that has mlx-vlm >= 0.7.11:

    python scripts/unlimited_ocr2_dash_oracle.py \
        crates/compute/tests/generated/dash_oracle.rs
"""

import sys

import mlx.core as mx
import numpy as np
from mlx_vlm.models.unlimited_ocr import dash

SPARSE = {
    "dash_floor_per_layer": [1e-4] * 12,
    "dash_prior_floor": 1e-4,
    "dash_image_only": True,
    "dash_scale_per_layer": [128.0] * 12,
    "dash_scaling_factor": 128.0,
    "dash_sigma": 1.0,
    "dash_stage0": "head_cls",
    "dash_topk_blocks": 12,
    "image_token_id": 128815,
    "page_token_id": 128827,
    "per_page_image_tokens": 273,
    "sparse_attention_freq": [1] * 12,
    "sparse_block_size": 32,
    "sparse_index_dim": 128,
    "sparse_local_block": 4,
    "sparse_main_rope_mode": "page2d",
    "sparse_main_rope_tok_freq_mode": "resample",
    "sparse_rope_page_mod": 2,
    "sparse_rope_page_pairs": 1,
    "sparse_rope_page_theta": 256.0,
    "sparse_rope_tok_theta": 10000.0,
    "training_stage": "dash",
    "use_sparse_attention": True,
}
IMAGE = 128815
PAGE = 128827
BOS = 0


def floats(a):
    """Bit-exact f32 round trip: JSON carries the IEEE-754 bits, not a decimal."""
    arr = np.asarray(a, dtype=np.float32).reshape(-1)
    return arr.view(np.uint32).tolist()


def ints(a):
    return np.asarray(a).astype(np.int64).reshape(-1).tolist()


def prompt(pages, per_page=273, suffix=19):
    ids = [BOS]
    for _ in range(pages):
        ids.append(PAGE)
        ids.extend([IMAGE] * per_page)
    ids.extend(range(1000, 1000 + suffix))
    return ids


def pages_case(ids, p, generated):
    pp = dash.PromptPages(np.array(ids), p)
    state = dash.DashState(p)
    state.start(np.array(ids))
    rope = dash.DashRope(p, 128, _RopeConfig())
    steps = []
    offset = len(ids)
    for token in generated:
        ctx = dash.build_context(
            state, rope, mx.array([[token]]), offset, 1, mx.float32
        )
        mx.eval(ctx.cos, ctx.candidates, state.markers)
        steps.append(
            {
                "token": token,
                "tok": int(state.tok.item()) if state.tok is not None else -1,
                "page": int(state.page.item()) if state.page is not None else -1,
                "candidates": ints(ctx.candidates),
                "cos": floats(ctx.cos),
                "sin": floats(ctx.sin),
            }
        )
        offset += 1
    tok, page = pp.coords if pp.coords is not None else ([], [])
    positions = np.arange(len(ids))
    nblk = -(-len(ids) // p.block_size)
    cand = pp.prefill_candidates(positions, nblk)
    return {
        "ids": ids,
        "tok": ints(tok),
        "page": ints(page),
        "query_pages": ints(pp.query_pages),
        "blk_min": ints(pp.blk_min),
        "blk_max": ints(pp.blk_max),
        "dec_blk_min": ints(pp.dec_blk_min),
        "dec_blk_max": ints(pp.dec_blk_max),
        "dec_prompt_pages": int(pp.dec_prompt_pages),
        "marker_count": int(pp.marker_count),
        "n_pages": int(getattr(pp, "n_pages", 0)),
        "pf_cnt": ints(pp.pf_cnt) if pp.coords is not None else [],
        "page_aware": bool(pp.page_aware),
        "nblk": nblk,
        "prefill_candidates": ints(cand) if cand is not None else [],
        "steps": steps,
    }


class _RopeConfig:
    rope_theta = 500000.0
    rope_scaling = {
        "attention_factor": 1.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "rope_type": "yarn",
    }
    max_position_embeddings = 131072


def select_case(rng, p, nblk, pos, cand_pattern):
    d = p.index_dim
    q = rng.standard_normal((1, d)).astype(np.float32)
    # Scaled like real router output: the RMS norm leaves unit-RMS rows.
    summaries = rng.standard_normal((dash.block_capacity(nblk), d)).astype(np.float32)
    summaries[nblk:] = 0
    cap = summaries.shape[0]
    if cand_pattern == "all":
        cand = np.ones((1, cap), dtype=bool)
    elif cand_pattern == "none":
        cand = np.zeros((1, cap), dtype=bool)
    else:
        cand = np.zeros((1, cap), dtype=bool)
        cand[0, : nblk // 2] = True
    scale = 128.0 / np.sqrt(d)
    idx, mask = dash._decode_select(
        mx.array(q),
        mx.array(summaries),
        mx.array(cand),
        mx.array(pos, dtype=mx.int32),
        mx.array(scale, dtype=mx.float32),
        mx.array(1e-4, dtype=mx.float32),
        p,
    )
    mx.eval(idx, mask)
    return {
        "q": floats(q),
        "summaries": floats(summaries[:nblk]),
        "capacity": cap,
        "nblk": nblk,
        "pos": pos,
        "candidates": ints(cand),
        "idx": ints(idx),
        "mask": floats(mask),
    }


def main(out):
    p = dash.DashParams.from_config(SPARSE, 12, 128)
    inv_tok, inv_page = dash.page2d_inv_freq(p, 128)
    rng = np.random.default_rng(20261102)

    entmax_cases = []
    for m in (1, 2, 5, 16):
        for spread in (0.1, 1.0, 40.0):
            x = -np.sort(-rng.standard_normal((1, m)) * spread).astype(np.float32)
            entmax_cases.append(
                {"x": floats(x), "y": floats(dash.entmax15(mx.array(x)))}
            )
    # The padded case the router really sees: fewer real candidates than topm.
    pad = np.full((1, 16), dash._ENTMAX_NEG, dtype=np.float32)
    pad[0, :3] = [50.0, 49.5, 12.0]
    entmax_cases.append({"x": floats(pad), "y": floats(dash.entmax15(mx.array(pad)))})

    blocks = rng.standard_normal((5, p.block_size, p.index_dim)).astype(np.float32)
    head_cls = rng.standard_normal((p.index_dim,)).astype(np.float32)
    summaries = dash._summarize_blocks(mx.array(blocks), mx.array(head_cls))

    selects = [
        select_case(rng, p, 4, 100, "all"),  # no remote block yet
        select_case(rng, p, 6, 170, "all"),  # fewer remotes than topk
        select_case(rng, p, 40, 1279, "all"),
        select_case(rng, p, 40, 1279, "half"),
        select_case(rng, p, 40, 1279, "none"),  # falls back to every remote
        select_case(rng, p, 90, 2880, "half"),  # capacity 128
    ]

    two_pages = prompt(2)
    generated = [500, PAGE, 501, 502, PAGE, 503, PAGE, 504]
    data = {
        "params": {
            "block_size": p.block_size,
            "local_blocks": p.local_blocks,
            "index_dim": p.index_dim,
            "topk": p.topk,
            "topm": p.topm,
            "sigma": p.sigma,
            "page_mod": p.rope_page_mod,
            "per_page_image_tokens": p.per_page_image_tokens,
            "image_token_id": p.image_token_id,
            "page_token_id": p.page_token_id,
            "idx_scale": floats(p.idx_scale),
            "floor": floats(p.floor),
        },
        "inv_tok": floats(inv_tok),
        "inv_page": floats(inv_page),
        "entmax": entmax_cases,
        "summaries": {
            "blocks": floats(blocks),
            "head_cls": floats(head_cls),
            "out": floats(summaries),
        },
        "selects": selects,
        "pages": [
            pages_case(prompt(1), p, [500, 501, PAGE, 502]),
            pages_case(two_pages, p, generated),
            # A prompt with no page marker: running-count pages, 1-D rope.
            pages_case([BOS] + [IMAGE] * 546 + list(range(1000, 1010)), p, [500, PAGE]),
            # Text only: nothing is page aware.
            pages_case([BOS] + list(range(1000, 1100)), p, [500, 501]),
        ],
    }
    with open(out, "w", encoding="ascii") as f:
        f.write("// @generated by scripts/unlimited_ocr2_dash_oracle.py. Do not edit.\n")
        f.write("// f32 values are IEEE-754 bit patterns, so no decimal round trip.\n")
        f.write("pub const ORACLE: Oracle = ")
        f.write(rust(data, "Oracle"))
        f.write(";\n")


# Field name -> element struct, for the generated literal. Everything else is a
# scalar or a flat slice, so the test file's struct definitions stay the schema.
STRUCTS = {
    "params": "Params",
    "entmax": "Entmax",
    "summaries": "Summaries",
    "selects": "Select",
    "pages": "Pages",
    "steps": "Step",
}
SIGNED = {"tok", "page", "query_pages", "blk_min", "blk_max", "dec_blk_min", "dec_blk_max"}
SIGNED |= {"pf_cnt", "ids", "idx"}
FLOATS = {"sigma"}


def rust(value, name=None, key=None):
    if isinstance(value, dict):
        fields = ",".join(f"{k}:{rust(v, STRUCTS.get(k), k)}" for k, v in value.items())
        return f"{name}{{{fields}}}"
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return "&[" + ",".join(rust(v, name, key) for v in value) + "]"
        if not value:
            return "&[]"
        return "&[" + ",".join(str(int(v)) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if key in FLOATS:
        return f"{float(value)}f32"
    return str(int(value))


if __name__ == "__main__":
    main(sys.argv[1])
