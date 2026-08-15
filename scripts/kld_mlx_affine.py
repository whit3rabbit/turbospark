#!/usr/bin/env python3
"""Cross-engine KL for the SUB-4-BIT MLX families: this port against MLX on
the identical bytes (ROADMAP's 1-bit entry step 5, and its ternary entry).

    TURBOSPARK_QWEN35_INSTALL_DIR=~/models/bonsai27b.gturbo \
    TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/bonsai-warm \
      cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
    /tmp/prism-venv/bin/python scripts/kld_mlx_affine.py /tmp/kld/bonsai-warm bonsai-1bit

    TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
    TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/ternary-warm \
      cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
    uv run --python 3.12 --with 'mlx-lm==0.31.2' --with numpy \
      scripts/kld_mlx_affine.py /tmp/kld/ternary-warm ternary-2bit

**THE CHECKPOINT NAME IS REQUIRED, NOT DEFAULTED, and that is the whole
reason this file is parameterized rather than copied.** A model-specific
constant in a measurement script is a wrong ANSWER waiting for its second
caller (AGENTS.md Gotcha 38), and this one has a worse failure than most:
point the ternary dump at the 1-bit reference and every check below still
passes -- the reference really is 1-bit throughout, the shapes agree, and the
only symptom is a big divergence that reads as a kernel bug. That is the same
shape as `kld_llamacpp.py`'s cache-name bug, which silently compared one
model's logits against another's because every arm is the same size.

**THE ENVIRONMENT DIFFERS PER WIDTH AND IT IS MEASURED, NOT ASSUMED.**
Upstream `mlx` refuses `bits=1` at the API LEVEL -- not merely on Metal -- so
`mx.quantize` reports "The supported bits are 2, 3, 4, 5, 6 and 8" and the
1-bit reference has to be `github.com/PrismML-Eng/mlx@prism` built from
source. TWO bits is in that list, so the ternary arm runs under an ordinary
`uv run --with mlx-lm` ephemeral env like `kld.py`'s. `require_the_mlx_build`
probes for the width it is about to use rather than checking a version
string, so either environment is accepted when it can actually do the work.

**THIS IS A SEPARATE DRIVER FROM `kld.py` AND NOT A WIDENING OF IT**, which
is a decision rather than duplication. `kld.py` runs mlx-lm out of a `uv run
--with mlx-lm` ephemeral environment, and that environment cannot serve this
model at all: upstream `mlx` refuses `bits=1` at the API level -- not merely
on Metal -- so `mx.quantize` reports "The supported bits are 2, 3, 4, 5, 6
and 8" and a forward pass is impossible on any device. The reference here is
`github.com/PrismML-Eng/mlx@prism`, which the checkpoint's own README names,
BUILT FROM SOURCE into a venv. A script that had to be handed an interpreter
is a different script from one that provisions its own.

Everything that is not the environment is IMPORTED from `kld.py` rather than
restated -- `divergences`, `perplexity` -- exactly as `kld_llamacpp.py` does.
The three things this file owns are the per-artifact pins, the environment
note above, and the reading below.

TWO FLOORS, NOT ONE, and this family can only have one of them (bench crate
Gotcha 8, AGENTS.md Gotcha 34). The SHAPE floor is mlx-lm against itself,
cached against batched, which holds weights and kernels and engine fixed and
varies only the forward shape; it is measured here. The BACKEND floor -- the
same engine's CPU against its Metal -- is NOT: this model is DENSE, so every
one of its 24.8B backbone weights is read per token, where the families that
have a cheap CPU arm (`qwen3moe`, `gpt-oss`) activate 3B. Its absence is
reported rather than skipped silently, because a missing floor is a missing
scale and the number above it cannot be read as small without one.

AND THE PORT NARROWS SOMETHING MLX DOES NOT (AGENTS.md Gotcha 45): every
unquantized tensor in BOTH checkpoints is F16 and this port stores them BF16,
which loses 19.5% of the norm values at up to 2^-8 relative. That is a real
difference between the two engines' weights, it is on the port's side of this
comparison, and it is the first thing to suspect if the headline lands high.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kld import divergences, perplexity  # noqa: E402

# Pinned in the matching `crates/repack/tests/*_checkpoint_network.rs`: the
# exact repo and commit each install was streamed from. Same quantized bytes
# on both sides is the whole point.
#
# `modules` is the checkpoint header's count of tensors carrying a `.scales`
# companion, and it is a CROSS-CHECK rather than decoration; see
# `assert_reference_matches`.
CHECKPOINTS = {
    "bonsai-1bit": {
        "repo": "prism-ml/Bonsai-27B-mlx-1bit",
        "revision": "ef22f239c670078e1507f9769bcaa66657332b96",
        "bits": 1,
        "group_size": 128,
        "modules": 498,
    },
    "ternary-2bit": {
        "repo": "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
        "revision": "70f75f3ad081ab840a42f3304c02c27e7f89bfb7",
        "bits": 2,
        "group_size": 128,
        "modules": 498,
    },
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


def require_the_mlx_build(spec) -> str:
    """Refuse to run under an mlx that cannot represent this checkpoint.

    At ONE bit upstream mlx does not merely lack a Metal kernel, it rejects
    the argument outright, so the failure without this check is an exception
    thrown somewhere inside model loading -- far from the cause. At TWO bits
    upstream is fine and this passes on any recent build. Checked by ASKING
    mlx to do the thing rather than by comparing a version string: the fork's
    version is `0.31.2.dev...`, which is neither newer nor distinguishable
    from upstream's by ordering.
    """
    import mlx.core as mx

    bits, group = spec["bits"], spec["group_size"]
    try:
        w = mx.random.normal((256, 256))
        q, s, b = mx.quantize(w, group_size=group, bits=bits)
        mx.eval(mx.quantized_matmul(mx.random.normal((1, 256)), q, s, b,
                                    transpose=True, group_size=group, bits=bits))
    except Exception as exc:  # noqa: BLE001 -- the message IS the diagnosis
        sys.exit(
            f"this mlx cannot do bits={bits} ({type(exc).__name__}: {exc})\n"
            "  at one bit that needs github.com/PrismML-Eng/mlx@prism, built from source:\n"
            "    git clone -b prism https://github.com/PrismML-Eng/mlx.git\n"
            "    uv venv /tmp/prism-venv --python 3.12\n"
            "    uv pip install --python /tmp/prism-venv/bin/python cmake ninja setuptools nanobind\n"
            "    uv pip install --python /tmp/prism-venv/bin/python -e mlx/ --no-build-isolation\n"
            "    uv pip install --python /tmp/prism-venv/bin/python 'mlx-lm==0.31.2' transformers numpy"
        )
    return f"{mx.__version__} ({mx.__file__})"


def assert_reference_matches(model, spec) -> dict:
    """The reference must be running the PACKED weights, not a widened copy,
    and it must be running the WIDTH this dump was taken at.

    Without this the comparison could silently become "this port's 1-bit
    kernels against MLX's fp16 kernels on dequantized weights", which is a
    different question with the same shape of answer -- and the tell would be
    a suspiciously SMALL divergence, i.e. the direction nobody investigates.

    Counted rather than spot-checked, and the count is a cross-check: each
    checkpoint's safetensors header carries `modules` tensors with a
    `.scales` companion, so that many quantized modules is the whole model
    and one fewer is a layer that quietly did not quantize.

    The BITS check is the other half, and it is what catches a dump paired
    with the wrong reference -- the two published checkpoints of this
    architecture have the same module count and the same shapes, so nothing
    else here would notice.
    """
    import mlx.nn as nn

    want_bits, want_modules = spec["bits"], spec["modules"]
    seen: dict[tuple, int] = {}
    for _, module in model.named_modules():
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            key = (type(module).__name__, module.bits, module.group_size)
            seen[key] = seen.get(key, 0) + 1
    if not seen or any(bits != want_bits for (_, bits, _) in seen):
        sys.exit(
            f"reference model is not {want_bits}-bit throughout: "
            f"{seen or 'no quantized modules'}"
        )
    total = sum(seen.values())
    if total != want_modules:
        sys.exit(
            f"reference has {total} quantized modules, the checkpoint header says {want_modules}"
        )
    return {f"{name}(bits={b},group={g})": n for (name, b, g), n in sorted(seen.items(), key=str)}


def mlx_logits(token_ids: list[int], cached: bool, spec) -> tuple[np.ndarray, dict]:
    """mlx-lm's next-token logits per position, as float32 [rows, vocab].

    Row i is the logits after consuming `token_ids[i]`, matching what
    `logit_dump.rs` writes, so the last position is dropped. `cached` picks
    the forward SHAPE: True steps one token at a time through a prompt cache
    (this port's shape, and the headline), False runs one batched pass.
    """
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, _ = load(str(snapshot_dir(spec)))
    quantization = assert_reference_matches(model, spec)
    if not cached:
        out = model(mx.array([token_ids]))
        mx.eval(out)
        return np.array(out[0, :-1, :].astype(mx.float32)), quantization

    cache = make_prompt_cache(model)
    rows = []
    for token in token_ids[:-1]:
        out = model(mx.array([[token]]), cache=cache)
        mx.eval(out)
        rows.append(np.array(out[0, -1, :].astype(mx.float32)))
    return np.stack(rows), quantization


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

    mlx_build = require_the_mlx_build(spec)

    port = np.fromfile(dump / "logits.f16", dtype=np.float16)
    if port.size != rows * vocab:
        sys.exit(f"{dump}/logits.f16 holds {port.size} values, expected {rows * vocab}")
    port = port.reshape(rows, vocab).astype(np.float32)

    cached, quantization = mlx_logits(ids, cached=True, spec=spec)
    batched, _ = mlx_logits(ids, cached=False, spec=spec)
    for name, arr in (("cached", cached), ("batched", batched)):
        if arr.shape != port.shape:
            sys.exit(f"mlx {name} returned {arr.shape}, this port dumped {port.shape}")

    # REPORTED, never checked against an invented bound. This family declares
    # no logit softcap, so there is no transform for the two heads to
    # disagree about -- AGENTS.md Gotcha 38's lesson from the first
    # non-Gemma caller of `kld_llamacpp.py`, which aborted a correct run on a
    # bound that was Gemma's `final_logit_softcapping` all along.
    maxima = {
        "turbospark": float(np.abs(port).max()),
        "mlx_cached": float(np.abs(cached).max()),
    }

    first = meta["first_scored_position"]
    report = {
        "rows": rows,
        "mlx_build": mlx_build,
        "reference_quantization": quantization,
        # This port against MLX in the SAME forward shape. The headline.
        "kl_vs_mlx": divergences(cached, port),
        # MLX against itself across its two shapes: the SHAPE floor.
        "kl_mlx_self": divergences(batched, cached),
        # Named rather than omitted: see the module doc.
        "backend_floor": "not measured (dense 27B; a CPU arm is not affordable here)",
        "perplexity": {
            "turbospark": perplexity(port, ids, first),
            "mlx_cached": perplexity(cached, ids, first),
            "mlx_batched": perplexity(batched, ids, first),
        },
        "max_abs_logit": maxima,
        "reference": f"{spec['repo']}@{spec['revision'][:8]}",
        "install": meta["install"],
        "cache_state": meta["cache_state"],
        "prompt_len": meta["prompt_len"],
    }
    print(json.dumps(report, indent=2))
    # Named per artifact, for `kld_llamacpp.py`'s reason: every arm of every
    # model is the same shape, so a shared filename lets one run's report be
    # read as another's.
    (dump / f"kld_mlx_affine-{name}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
