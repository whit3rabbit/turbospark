"""Token-level KL divergence between this port and mlx-lm (ROADMAP Phase Q).

Run `cargo test -p turbospark-bench --test logit_dump` first; it writes the
id sequence and this port's full-vocabulary logits. This script replays the
SAME IDS through mlx-lm, on the same quantized checkpoint the install was
repacked from, and reports how far apart the two distributions are.

    TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
    TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/turbospark \
      cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture

    uv run --python 3.12 --with mlx-lm --with numpy \
      scripts/kld.py /tmp/kld/turbospark gemma4

**THE CHECKPOINT NAME IS REQUIRED, NOT DEFAULTED.** This file carried
`REPO = "mlx-community/gemma-4-26b-a4b-it-4bit"` as a module constant for
its whole life, which was defensible while its only caller was Gemma and is
exactly the failure AGENTS.md Gotcha 38 names: a model-specific constant in
a measurement script is a wrong ANSWER waiting for its second caller. The
failure mode is the quiet one -- a dump paired with the wrong reference
still has the right shape, so every check below passes and the only symptom
is a divergence that reads as a kernel bug. Same shape as
`kld_llamacpp.py`'s cache-name bug and the reason
`kld_mlx_affine.py`'s `CHECKPOINTS` table is keyed and required; see
`CHECKPOINTS` below.

WHY IDS AND NOT PROSE. A tokenizer or chat-template difference between the
two engines would show up as a divergence and read as a numerics gap. The
ids come out of `meta.json` and are fed to mlx-lm directly; its tokenizer
is never asked to encode anything.

WHY THE HEADS ARE COMPARABLE, and it is READ off the install rather than
recalled. Neither side normalizes (AGENTS.md Gotcha 16). Where the family
declares a `finalLogitSoftcap` both sides return `softcap * tanh(z /
softcap)` -- this port by `utility.metal`'s `logit_softcap_fp16`, mlx-lm at
`mlx_lm/models/gemma4_text.py`'s `Model.__call__`, which applies
`logit_softcap` and returns -- and `check_heads` ASSERTS against the
declared bound, because a softcap on one side only would dominate every
number below. Where the family declares none there is no transform to
mismatch, so both maxima are REPORTED instead of being checked against an
invented tolerance. THE BOUND USED TO BE THE LITERAL 30 IN THIS DOCSTRING,
which is Gemma's value; `softcap_of` reads it from the dump's own install,
which is the fix Gotcha 38 required of `kld_llamacpp.py` and which this
file never received.

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

# Each entry is pinned in the matching `crates/repack/tests/*_checkpoint_
# network.rs`: the exact repo and commit that install was repacked from.
# Same quantized bytes on both sides is the whole point, so a different
# revision here silently turns a kernel comparison into a checkpoint
# comparison.
#
# WHAT MAKES A CHECKPOINT ELIGIBLE FOR THIS DRIVER rather than for
# `kld_mlx_affine.py`: upstream `mlx` must be able to load it. That split is
# the ENVIRONMENT and not the family -- upstream refuses `bits=1` at the API
# level, so the 1-bit reference needs a fork built from source and therefore
# needs a script that is handed an interpreter. Everything upstream can load
# belongs here, under an ordinary `uv run --with mlx-lm`.
CHECKPOINTS = {
    "gemma4": {
        "repo": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "revision": "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        "install_var": "TURBOSPARK_GEMMA4_INSTALL_DIR",
    },
    "qwen25": {
        "repo": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "revision": "c8e9187488f846965507bfc2b3957d59fd0d5a27",
        "install_var": "TURBOSPARK_QWEN2_DENSE_INSTALL_DIR",
    },
    # `qwen36` IS NOT HERE, AND THAT IS A DECISION RATHER THAN AN OMISSION.
    # `docs/BENCHMARKS.md` has recorded since Phase Q that Qwen 3.6 has no
    # cross-engine number because "`kld.py`'s reference is pinned to the
    # Gemma repo", which reads as though killing the pin above is the whole
    # job. It is not: that checkpoint is an MoE, and THIS driver has no
    # reference guard at all. `kld_mlx_affine.py` does, and its guard exists
    # for exactly this shape -- mlx packs routed experts into a
    # `QuantizedSwitchLinear`, a type an `isinstance` list misses, so an MoE
    # reference silently counts short. The row is in that file's
    # `CHECKPOINTS` instead. Adding it here would put an MoE reference in
    # the one driver that cannot check it loaded quantized.
}


def snapshot_dir(spec) -> pathlib.Path:
    repo, revision = spec["repo"], spec["revision"]
    path = (
        pathlib.Path.home()
        / ".cache/huggingface/hub"
        / f"models--{repo.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    if not (path / "config.json").exists():
        sys.exit(
            f"reference checkpoint not found at {path}\n"
            f"  hf download {repo} --revision {revision}"
        )
    return path


def softcap_of(install: str) -> float:
    """The install's declared `final_logit_softcapping`, or 0.0 for none.

    Read the property, do not recall it (AGENTS.md Gotcha 38). Shared with
    `kld_llamacpp.py`, which is where it was written and which imports it
    from here rather than keeping a second copy: the question "what
    transform does this install's head apply" is about the install and not
    about which engine it is being compared against.

    Falls back to "unknown" when the install is gone: the dump is a frozen
    artifact and outlives the install dir it names (the Gemma GGUF arms in
    /tmp/kld are the standing example -- that install was deleted and its
    logits cannot be regenerated).
    """
    try:
        manifest = json.loads((pathlib.Path(install) / "manifest.json").read_text())
    except OSError:
        return float("nan")
    return float(manifest.get("arch", {}).get("finalLogitSoftcap", 0.0))


def check_heads(cached: np.ndarray, port: np.ndarray, softcap: float,
                reference: str) -> dict:
    """Are the two engines' output heads the same function?

    Every divergence below is meaningless if they are not, and the failure
    is not hypothetical: a reference that skips a saturating nonlinearity
    the port applies disagrees by a factor, not by a rounding error.

    Where the family softcaps, the bound is exact and declared, so this
    asserts against it (1.001 for f32 rounding at the asymptote). Where it
    does not, there is no transform to mismatch and nothing to assert -- so
    both maxima are REPORTED instead of being checked against an invented
    tolerance. A gross mismatch is visible in the two numbers.

    `reference` names the other engine, and is the key the maxima are
    reported under, so one implementation serves both drivers.
    """
    reference_max = float(np.abs(cached).max())
    port_max = float(np.abs(port).max())
    if softcap > 0.0 and reference_max > softcap * 1.001:
        sys.exit(
            f"{reference}'s max |logit| is {reference_max:.4f}, over the {softcap} "
            "softcap this install declares: the two heads are not the same "
            "function, so no divergence below would be about the weights"
        )
    return {
        "declared_softcap": softcap,
        f"max_abs_logit_{reference}": reference_max,
        "max_abs_logit_port": port_max,
    }


def mlx_logits(token_ids: list[int], cached: bool, spec) -> np.ndarray:
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

    model, _ = load(str(snapshot_dir(spec)))
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
    if len(sys.argv) != 3 or sys.argv[2] not in CHECKPOINTS:
        sys.exit(
            f"usage: {sys.argv[0]} <dump-dir> <{'|'.join(CHECKPOINTS)}>\n"
            "  the checkpoint name is REQUIRED; see the module doc for why"
        )
    dump = pathlib.Path(sys.argv[1])
    name = sys.argv[2]
    spec = CHECKPOINTS[name]
    meta = json.loads((dump / "meta.json").read_text())
    rows, vocab, ids = meta["rows"], meta["vocab_size"], meta["token_ids"]

    port = np.fromfile(dump / "logits.f16", dtype=np.float16)
    if port.size != rows * vocab:
        sys.exit(f"{dump}/logits.f16 holds {port.size} values, expected {rows * vocab}")
    port = port.reshape(rows, vocab).astype(np.float32)

    cached = mlx_logits(ids, cached=True, spec=spec)
    batched = mlx_logits(ids, cached=False, spec=spec)
    for shape, arr in (("cached", cached), ("batched", batched)):
        if arr.shape != port.shape:
            sys.exit(f"mlx-lm {shape} returned {arr.shape}, this port dumped {port.shape}")

    first = meta["first_scored_position"]
    report = {
        "rows": rows,
        # Read this first. It is the precondition for everything under it.
        "heads": check_heads(cached, port, softcap_of(meta["install"]), "mlx"),
        # This port against mlx-lm in the SAME forward shape. The headline.
        "kl_vs_mlx": divergences(cached, port),
        # mlx-lm against itself across its two shapes. The floor that says
        # whether the line above is small (see the module doc).
        "kl_mlx_self": divergences(batched, cached),
        "perplexity": {
            "turbospark": perplexity(port, ids, first),
            "mlx_cached": perplexity(cached, ids, first),
            "mlx_batched": perplexity(batched, ids, first),
        },
        "reference": f"{spec['repo']}@{spec['revision'][:8]}",
        "install": meta["install"],
        "expert_cache_slots": meta["expert_cache_slots"],
        "cache_state": meta["cache_state"],
        "prompt_len": meta["prompt_len"],
    }
    print(json.dumps(report, indent=2))
    # Named per artifact, for `kld_llamacpp.py`'s reason: every arm of every
    # model is the same shape, so a shared filename lets one run's report be
    # read as another's. The bare `kld.json` this used to write was safe for
    # exactly as long as there was one entry above it.
    (dump / f"kld-{name}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
