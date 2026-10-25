#!/usr/bin/env python3
"""Cross-engine KL for Flint: this port against mlx-vlm on the identical MLX
bytes of `mlx-community/flint-mixed-3-4bit`.

    TURBOSPARK_FLINT_INSTALL_DIR=~/models/flint-mixed-3-4bit.gturbo \
    TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/flint-mixed-warm \
      cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
    uv run --python 3.12 --with 'mlx-vlm==0.7.10' --with numpy \
      scripts/kld_flint_mlx_vlm.py /tmp/kld/flint-mixed-warm ~/models/flint-mixed-3-4bit

The second argument is a LOCAL copy of the checkpoint (`hf download
mlx-community/flint-mixed-3-4bit --revision b1c30585...`). It is checked
against the pin below before anything runs, because pointing this at the INT4
build passes every shape check and reads as a kernel bug.

WHY MLX-VLM AND NOT `kld_mlx_affine.py`. mlx-lm has no released Flint module
(its PR is unmerged); mlx-vlm 0.7.10 ships `mlx_vlm/models/flint`. The KL and
perplexity helpers are imported from `kld.py` rather than restated.

WHAT THE TWO ENGINES DO DIFFERENTLY ON PURPOSE, so a nonzero headline is read
against them rather than as a bug:
  - the router: mlx-vlm keeps the published BF16 gate and takes the sigmoid in
    FP32; this port force-quantizes it to INT8 at intake (97.3% identical top-6
    on the INT4 build, every difference a near-tie: docs/FLINT_PHASE0.md);
  - the routed sum: mlx-vlm sums the weighted experts in FP32; this port
    reduces in FP32 per output and stores FP16 activations;
  - the SwiGLU: this port's install carries the exact 2^6 up/down rebalance.

THE FLOOR IS mlx-vlm AGAINST ITSELF, cached against batched: same engine and
bytes, a different forward shape, which on an MoE moves near-tie routes. The
headline is only small or large relative to it (bench crate Gotcha 8). No
backend (CPU) floor: mlx's CPU path is unaffordable on a 40B MoE.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kld import divergences, perplexity  # noqa: E402

REPO = "mlx-community/flint-mixed-3-4bit"
REVISION = "b1c305855ae7c3464813815be4dbd544b22c36aa"
MLX_VLM = "0.7.10"


def require_the_checkpoint(path: pathlib.Path) -> None:
    """The local copy is the pinned mixed build: 3-bit routed experts, 8-bit
    embedding, everything else 4-bit, all four shards complete."""
    config = json.loads((path / "config.json").read_text())
    export = config.get("mlx_export", {})
    if config.get("model_type") != "flint" or export.get("routed_expert_bits") != 3:
        sys.exit(f"{path} is not {REPO}: model_type/routed_expert_bits disagree")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [s for s in shards if not (path / s).is_file()]
    if missing:
        sys.exit(f"{path} is missing shards {missing}")
    routed = [k for k, v in config["quantization"].items() if isinstance(v, dict) and v.get("bits") == 3]
    if len(routed) != 141:
        sys.exit(f"{path} declares {len(routed)} 3-bit tensors, the pinned build has 141")


def mlx_logits(model, token_ids: list[int], cached: bool) -> np.ndarray:
    """mlx-vlm's next-token logits per position as float32 [rows, vocab]; row
    i follows `token_ids[i]`, as `logit_dump.rs` writes them."""
    import mlx.core as mx

    if not cached:
        out = model(mx.array([token_ids])).logits
        mx.eval(out)
        return np.array(out[0, :-1, :].astype(mx.float32))
    cache = model.make_cache()
    rows = []
    for token in token_ids[:-1]:
        out = model(mx.array([[token]]), cache=cache).logits
        mx.eval(out)
        rows.append(np.array(out[0, -1, :].astype(mx.float32)))
    return np.stack(rows)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <dump-dir> <local-checkpoint-dir>")
    dump = pathlib.Path(sys.argv[1])
    checkpoint = pathlib.Path(sys.argv[2]).expanduser()
    require_the_checkpoint(checkpoint)

    import mlx_vlm
    from mlx_vlm.utils import load_model

    if mlx_vlm.__version__ != MLX_VLM:
        sys.exit(f"mlx-vlm {mlx_vlm.__version__}, this reference is pinned to {MLX_VLM}")

    meta = json.loads((dump / "meta.json").read_text())
    rows, vocab, ids = meta["rows"], meta["vocab_size"], meta["token_ids"]
    port = np.fromfile(dump / "logits.f16", dtype=np.float16)
    if port.size != rows * vocab:
        sys.exit(f"{dump}/logits.f16 holds {port.size} values, expected {rows * vocab}")
    port = port.reshape(rows, vocab).astype(np.float32)

    model = load_model(checkpoint)
    cached = mlx_logits(model, ids, cached=True)
    batched = mlx_logits(model, ids, cached=False)
    for name, arr in (("cached", cached), ("batched", batched)):
        if arr.shape != port.shape:
            sys.exit(f"mlx-vlm {name} returned {arr.shape}, this port dumped {port.shape}")

    first = meta["first_scored_position"]
    report = {
        "rows": rows,
        "reference": f"{REPO}@{REVISION[:8]} via mlx-vlm {MLX_VLM}",
        # This port against mlx-vlm in the same (one token at a time) shape.
        "kl_vs_mlx_vlm": divergences(cached, port),
        # mlx-vlm against itself across its two shapes: the floor.
        "kl_mlx_vlm_self": divergences(batched, cached),
        "perplexity": {
            "turbospark": perplexity(port, ids, first),
            "mlx_vlm_cached": perplexity(cached, ids, first),
            "mlx_vlm_batched": perplexity(batched, ids, first),
        },
        "max_abs_logit": {
            "turbospark": float(np.abs(port).max()),
            "mlx_vlm_cached": float(np.abs(cached).max()),
        },
        "install": meta["install"],
        "cache_state": meta["cache_state"],
        "prompt_len": meta["prompt_len"],
    }
    print(json.dumps(report, indent=2))
    (dump / "kld_flint_mlx_vlm.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
