#!/usr/bin/env python3
"""Prefill (prompt-processing) throughput for mlx-lm, on the SAME machine.

The cross-engine PREFILL counterpart to `kld.py` and `kld_llamacpp.py`, which
compare LOGITS. This one compares only wall-clock prompt processing, because
that is where this port's published gap against the community MLX builds
lives (`docs/BENCHMARKS.md`, the Qwen3.8-27B external reference points).

**WHY IT EXISTS.** That section used to lean on an uncontrolled community
reading of 210.3 tok/s -- different silicon, unknown prompt length, a build
named "Optimized-Speed" -- which is exactly the shape of number AGENTS.md
Gotcha 62 says to re-derive rather than quote. Running the reference engine
HERE removes every confound at once: same chip, same checkpoint, same
quantization, same prompt, same thermal and contention conditions. When it
was first run (2026-08-29) it reproduced the community figure to within 4 to
7%, which is what turned "that bar looks unrealistic" into a measured 4.94x
deficit that belongs to this port.

**IT IS A THROUGHPUT MEASUREMENT, SO THE MACHINE MATTERS** (Gotchas 22, 28
and 43). Discard the warmup, which this does, and prefer a quiet machine on
AC; a contended one depresses tok/s rather than inflating it, so a number
taken on a busy machine is conservative for BOTH engines and the ratio
survives better than either absolute does.

**THE CHUNK WIDTH IS THE POINT OF THE COMPARISON, NOT AN INCIDENTAL SETTING.**
This uses mlx-lm's own default `prefill_step_size` of 512, which is the
number to read against this port's `MAX_PREFILL_BATCH` of 16. The two engines
therefore re-read the weight set a very different number of times over one
prompt, and that arithmetic is most of the gap.

Usage (the env is ephemeral, nothing is installed into this workspace):

    uv run --python 3.12 --with mlx --with mlx-lm --with transformers -- \
      python scripts/mlx_prefill.py <snapshot-dir> <prompt-file> [trials]

`<snapshot-dir>` is a local HF snapshot, for example
`~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/<rev>`.
`<prompt-file>` is raw text; the checkpoint's own chat template is applied
when it has one, so the token count matches what a chat caller really sends.
"""

import pathlib
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# mlx-lm `generate`'s own default. Named rather than inlined because it is
# the term being compared, not a knob to tune.
PREFILL_STEP = 512
WARMUP_TOKENS = 64


def prefill(model, ids, step):
    """One full prompt through the model, timed the way generation does it.

    `mx.eval` on the CACHE STATE per chunk rather than on the logits alone:
    MLX is lazy, so evaluating only the final logits would let chunk
    boundaries collapse and would measure a different program than the one
    `generate` runs.
    """
    cache = make_prompt_cache(model)
    x = mx.array(ids)[None]
    out = None
    for off in range(0, len(ids), step):
        out = model(x[:, off : off + step], cache=cache)
        mx.eval([c.state for c in cache])
    mx.eval(out)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    snapshot, prompt_file = sys.argv[1], sys.argv[2]
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    model, tokenizer = load(snapshot)
    text = pathlib.Path(prompt_file).read_text()
    try:
        ids = list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True
            )
        )
        framing = "chat template applied"
    except Exception:
        # A base checkpoint with no template: the raw text IS the prompt, and
        # saying so beats silently reporting a different token count than a
        # chat caller would see.
        ids = list(tokenizer.encode(text))
        framing = "no chat template; raw text"
    print(f"{snapshot}\n{len(ids)} prompt tokens ({framing}), step {PREFILL_STEP}")

    # Discarded: the first pass pays lazy weight loading and kernel
    # compilation, which on a cold GPU is worth more than any effect being
    # measured (AGENTS.md Gotcha 20).
    prefill(model, ids[:WARMUP_TOKENS], PREFILL_STEP)

    rates = []
    for trial in range(1, trials + 1):
        started = time.perf_counter()
        prefill(model, ids, PREFILL_STEP)
        elapsed = time.perf_counter() - started
        rates.append(len(ids) / elapsed)
        print(f"trial {trial}: {elapsed:.2f}s -> {rates[-1]:.1f} tok/s prefill")

    if len(rates) > 1:
        spread = (max(rates) - min(rates)) / min(rates) * 100.0
        # Dispersion is the contamination tell (Gotcha 43); a few percent is
        # normal, and a large spread means re-run rather than publish.
        print(f"spread {spread:.1f}% across {len(rates)} trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
