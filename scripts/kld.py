"""Token-level KL divergence between this port and mlx-lm (ROADMAP Phase Q).

Run `cargo test -p mrefrust-bench --test logit_dump` first; it writes the
id sequence and this port's full-vocabulary logits. This script replays the
SAME IDS through mlx-lm, on the same quantized checkpoint the install was
repacked from, and reports how far apart the two distributions are.

    MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
    MREFRUST_LOGIT_DUMP_DIR=/tmp/kld/mrefrust \
      cargo test -p mrefrust-bench --test logit_dump --release -- --ignored --nocapture

    uv run --python 3.12 --with mlx-lm --with numpy \
      scripts/kld.py /tmp/kld/mrefrust

WHY IDS AND NOT PROSE. A tokenizer or chat-template difference between the
two engines would show up as a divergence and read as a numerics gap. The
ids come out of `meta.json` and are fed to mlx-lm directly; its tokenizer
is never asked to encode anything.

WHY THE HEADS ARE COMPARABLE. Both sides return `softcap * tanh(z /
softcap)` with softcap 30 and neither normalizes: this port by
`utility.metal`'s `logit_softcap_fp16` (AGENTS.md Gotcha 16), mlx-lm at
`mlx_lm/models/gemma4_text.py`'s `Model.__call__`, which applies
`logit_softcap` and returns. That was read rather than assumed; a softcap
on one side only would dominate every number below.

`kl_mlx_self` IS THE NUMBER THAT MAKES `kl_vs_mlx` READABLE, and it is the
whole reason this script runs mlx-lm twice. A KL between two engines has no
natural scale: 0.02 nats means nothing until something says what agreement
even looks like at 4 bits. So mlx-lm is run against ITSELF in the two
forward shapes it supports -- one batched pass over the sequence, and the
same sequence stepped token by token through a cache -- and the divergence
between those is a floor built from arithmetic alone, since the weights,
the kernels, and the engine are identical across it. Measured here, that
floor came out LARGER than the cross-engine number.

BOTH ENGINES ARE STEPPED THROUGH A CACHE for `kl_vs_mlx`, because that is
the shape this port runs in and shape turns out to matter more than
implementation. Comparing this port's cached pass against mlx's batched one
would fold the floor above into the headline.

THERE IS NO f16 STORAGE FLOOR TO SUBTRACT, which is worth writing down
because it looks like there should be. This port dumps IEEE-754 binary16;
mlx-lm returns bfloat16, whose 8 mantissa bits are strictly coarser than
f16's 10, and every logit here is softcapped into +/-30 where the f16
exponent range is not a constraint. So the round trip through this port's
storage width is exactly lossless: measured directly, mlx's own logits
rounded to f16 diverge from themselves by 3.5e-22 nats. mlx is the
lower-precision side, not this port.
"""

import json
import pathlib
import sys

import numpy as np

# Pinned in `crates/repack/tests/gemma4_checkpoint_network.rs`: the exact
# repo and commit `~/models/gemma4.gturbo` was repacked from. Same
# quantized bytes on both sides is the whole point, so a different
# revision here silently turns a kernel comparison into a checkpoint
# comparison.
REPO = "mlx-community/gemma-4-26b-a4b-it-4bit"
REVISION = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"


def snapshot_dir() -> pathlib.Path:
    path = (
        pathlib.Path.home()
        / ".cache/huggingface/hub"
        / f"models--{REPO.replace('/', '--')}"
        / "snapshots"
        / REVISION
    )
    if not (path / "config.json").exists():
        sys.exit(
            f"reference checkpoint not found at {path}\n"
            f"  hf download {REPO} --revision {REVISION}"
        )
    return path


def mlx_logits(token_ids: list[int], cached: bool) -> np.ndarray:
    """mlx-lm's next-token logits for every position, as float32 [rows, vocab].

    Row i is the logits after consuming `token_ids[i]`, which is the layout
    `logit_dump.rs` writes, so the last position is dropped: nothing is fed
    the final id.

    `cached` picks the forward SHAPE, and the two are not interchangeable
    (see the module doc). True steps one token at a time through a prompt
    cache, matching this port; False runs a single batched pass, which is
    what mlx-lm does in normal use and is roughly ten times faster here.
    """
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, _ = load(str(snapshot_dir()))
    if not cached:
        out = model(mx.array([token_ids]))
        mx.eval(out)
        return np.array(out[0, :-1, :].astype(mx.float32))

    cache = make_prompt_cache(model)
    rows = []
    for token in token_ids[:-1]:
        out = model(mx.array([[token]]), cache=cache)
        mx.eval(out)
        rows.append(np.array(out[0, -1, :].astype(mx.float32)))
    return np.stack(rows)


def divergences(p_logits: np.ndarray, q_logits: np.ndarray) -> dict:
    """Per-position KL between two logit matrices of the same shape.

    Accumulated in float64 and max-subtracted, the same discipline
    `quality_common::negative_log_prob` uses for the perplexity: 262,144
    exponentials per row overflow float32 and lose the tail in it.
    """

    def log_softmax(row: np.ndarray) -> np.ndarray:
        row = row.astype(np.float64)
        row -= row.max()
        return row - np.log(np.exp(row).sum())

    forward, reverse, agree = [], [], 0
    for i in range(p_logits.shape[0]):
        lp = log_softmax(p_logits[i])
        lq = log_softmax(q_logits[i])
        forward.append(float((np.exp(lp) * (lp - lq)).sum()))
        reverse.append(float((np.exp(lq) * (lq - lp)).sum()))
        agree += int(lp.argmax() == lq.argmax())

    def stats(values: list[float]) -> dict:
        a = np.asarray(values)
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }

    return {
        "forward": stats(forward),
        "reverse": stats(reverse),
        "top1_agreement": agree / p_logits.shape[0],
    }


def perplexity(logits: np.ndarray, token_ids: list[int], first: int) -> float:
    """Teacher-forced perplexity over the assistant slot only.

    The one number that is directly comparable to what `quality_gate`
    prints, so it cross-validates the whole pipeline: a tokenization or
    alignment mistake anywhere above shows up here as a wild value rather
    than as a plausible-looking KL. UNLIKE the KL, this one does need the
    assistant-slot restriction -- an instruction-tuned checkpoint was never
    trained to predict prompt tokens (`quality_common`'s module doc).
    """
    nll = 0.0
    for i in range(first, logits.shape[0]):
        row = logits[i].astype(np.float64)
        row -= row.max()
        nll += float(np.log(np.exp(row).sum()) - row[token_ids[i + 1]])
    return float(np.exp(nll / (logits.shape[0] - first)))


def main() -> None:
    dump = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/kld/mrefrust")
    meta = json.loads((dump / "meta.json").read_text())
    rows, vocab, ids = meta["rows"], meta["vocab_size"], meta["token_ids"]

    port = np.fromfile(dump / "logits.f16", dtype=np.float16)
    if port.size != rows * vocab:
        sys.exit(f"{dump}/logits.f16 holds {port.size} values, expected {rows * vocab}")
    port = port.reshape(rows, vocab).astype(np.float32)

    cached = mlx_logits(ids, cached=True)
    batched = mlx_logits(ids, cached=False)
    for name, arr in (("cached", cached), ("batched", batched)):
        if arr.shape != port.shape:
            sys.exit(f"mlx-lm {name} returned {arr.shape}, this port dumped {port.shape}")

    first = meta["first_scored_position"]
    report = {
        "rows": rows,
        # This port against mlx-lm in the SAME forward shape. The headline.
        "kl_vs_mlx": divergences(cached, port),
        # mlx-lm against itself across its two shapes. The floor that says
        # whether the line above is small (see the module doc).
        "kl_mlx_self": divergences(batched, cached),
        "perplexity": {
            "mrefrust": perplexity(port, ids, first),
            "mlx_cached": perplexity(cached, ids, first),
            "mlx_batched": perplexity(batched, ids, first),
        },
        "reference": f"{REPO}@{REVISION[:8]}",
        "install": meta["install"],
        "expert_cache_slots": meta["expert_cache_slots"],
        "cache_state": meta["cache_state"],
        "prompt_len": meta["prompt_len"],
    }
    print(json.dumps(report, indent=2))
    (dump / "kld.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
