---
title: 'Python Reference: Cross-Engine KL and MLX Benchmarks'
description: 'Public API of the scripts/ measurement drivers that compare this port against mlx-lm, llama.cpp and mlx-vlm, and price MLX prefill and qmm performance.'
diataxisType: reference
---

<!-- generated: python lane, signal: scripts/*.py (20 tracked files; no pyproject.toml/setup.py/setup.cfg) -->

These are standalone measurement drivers under `scripts/`, documented by their
module docstrings. There is no Python package in this repository: no
`pyproject.toml`, `setup.py` or `setup.cfg`, and every script runs under an
ephemeral environment (`uv run --python 3.12 --with mlx-lm --with numpy ...`).
Heavy dependencies (mlx-lm, numpy) are imported at the top of each script, so
the docstrings below are extracted from source rather than from an import.

Public surface rule applied: no script defines `__all__`, so every module-level
name without a leading underscore is public. Functions marked **(no
docstring)** are public but undocumented.

## `kld.py` — KL divergence vs mlx-lm

Token-level KL divergence between this port and mlx-lm (ROADMAP Phase Q). Run
`cargo test -p turbospark-bench --test logit_dump` first; it writes the id
sequence and this port's full-vocabulary logits. This script replays the SAME
IDS through mlx-lm, on the same quantized checkpoint the install was repacked
from, and reports how far apart the two distributions are.

| Constant | Value |
|---|---|
| `CHECKPOINTS` | keyed table of reference checkpoints (repo, revision, ...) |

### Functions

```python
def snapshot_dir(spec) -> pathlib.Path
```

**(no docstring)**

```python
def softcap_of(install: str) -> float
```

> The install's declared `final_logit_softcapping`, or 0.0 for none.
>
> Read the property, do not recall it (AGENTS.md Gotcha 38). Shared with
> `kld_llamacpp.py`, which is where it was written and which imports it from
> here rather than keeping a second copy: the question "what transform does
> this install's head apply" is about the install and not about which engine
> it is being compared against.
>
> Falls back to "unknown" when the install is gone: the dump is a frozen
> artifact and outlives the install dir it names.

```python
def check_heads(cached: np.ndarray, port: np.ndarray, softcap: float, reference: str) -> dict
```

> Are the two engines' output heads the same function?
>
> Every divergence below is meaningless if they are not, and the failure is
> not hypothetical: a reference that skips a saturating nonlinearity the port
> applies disagrees by a factor, not by a rounding error.
>
> Where the family softcaps, the bound is exact and declared, so this asserts
> against it (1.001 for f32 rounding at the asymptote). Where it does not,
> there is no transform to mismatch and nothing to assert -- so both maxima
> are REPORTED instead of being checked against an invented tolerance.
>
> `reference` names the other engine, and is the key the maxima are reported
> under, so one implementation serves both drivers.

```python
def mlx_logits(token_ids: list[int], cached: bool, spec) -> np.ndarray
```

> mlx-lm's next-token logits for every position, as float32 [rows, vocab].
>
> Row i is the logits after consuming `token_ids[i]`, which is the layout
> `logit_dump.rs` writes, so the last position is dropped: nothing is fed
> the final id.
>
> `cached` picks the forward SHAPE, and the two are not interchangeable.
> True steps one token at a time through a prompt cache, matching this port;
> False runs a single batched pass, which is roughly ten times faster.

```python
def divergences(p_logits: np.ndarray, q_logits: np.ndarray) -> dict
```

> Per-position KL between two logit matrices of the same shape.
>
> Accumulated in float64 and max-subtracted, the same discipline
> `quality_common::negative_log_prob` uses for the perplexity: 262,144
> exponentials per row overflow float32 and lose the tail in it.

```python
def perplexity(logits: np.ndarray, token_ids: list[int], first: int) -> float
```

> Teacher-forced perplexity over the assistant slot only.
>
> The one number that is directly comparable to what `quality_gate` prints,
> so it cross-validates the whole pipeline: a tokenization or alignment
> mistake anywhere above shows up here as a wild value rather than as a
> plausible-looking KL. UNLIKE the KL, this one does need the assistant-slot
> restriction -- an instruction-tuned checkpoint was never trained to predict
> prompt tokens.

```python
def main() -> None
```

**(no docstring)**

## `kld_llamacpp.py` — KL divergence vs llama.cpp

Token-level KL divergence between this port and llama.cpp, ON THE SAME GGUF
BYTES (ROADMAP Phase G's last open gate clause). Replays the same id sequence
through llama.cpp on the published GGUF the install was streamed from.

| Constant | Value |
|---|---|
| `HARNESS_SRC` | `scripts/llamacpp_logits.c` (beside this script) |
| `HARNESS_BIN` | `/tmp/llamacpp_logits` |

### Functions

```python
def build_harness() -> None
```

**(no docstring)**

```python
def llamacpp_logits(model: pathlib.Path, ids_path: pathlib.Path, out: pathlib.Path, mode: str, rows: int, vocab: int, n_gpu_layers: int) -> np.ndarray
```

> llama.cpp's next-token logits for every position, as float32
> [rows, vocab].
>
> Cached results are reused: each shape costs minutes and 577 MiB, and the
> inputs are fixed files, so a re-run of the analysis should not re-run the
> model. Delete the .f32 to force a fresh pass.

```python
def load_port_dump(dump: pathlib.Path) -> tuple[np.ndarray, dict]
```

**(no docstring)**

```python
def main() -> None
```

**(no docstring)**

## `kld_mlx_affine.py` — KL for MLX-affine (sub-4-bit) families

Cross-engine KL for the MLX-AFFINE families: this port against MLX on the
identical bytes (ROADMAP's 1-bit entry step 5, and its ternary entry). Nothing
in the driver is narrower than "MLX affine at some width"; `ornith-35b-4bit`
is 4- and 8-bit mixed.

| Constant | Value |
|---|---|
| `CHECKPOINTS` | keyed table (bonsai-1bit, ...): repo, revision, ... |

### Functions

```python
def snapshot_dir(spec) -> pathlib.Path
```

**(no docstring)**

```python
def require_the_mlx_build(spec) -> str
```

> Refuse to run under an mlx that cannot represent this checkpoint.
>
> At ONE bit upstream mlx does not merely lack a Metal kernel, it rejects
> the argument outright, so the failure without this check is an exception
> thrown somewhere inside model loading -- far from the cause. At TWO bits
> upstream is fine and this passes on any recent build. Checked by ASKING
> mlx to do the thing rather than by comparing a version string: the fork's
> version is `0.31.2.dev...`, which is neither newer nor distinguishable
> from upstream's by ordering.

```python
def assert_reference_matches(model, spec) -> dict
```

> The reference must be running the PACKED weights, not a widened copy, and
> it must be running the WIDTH this dump was taken at.
>
> Without this the comparison could silently become "this port's 1-bit
> kernels against MLX's fp16 kernels on dequantized weights", which is a
> different question with the same shape of answer -- and the tell would be
> a suspiciously SMALL divergence, i.e. the direction nobody investigates.
>
> Counted rather than spot-checked, and the COMPOSITION is the cross-check:
> the checkpoint's safetensors header says how many tensors carry a
> `.scales` companion at each width. The map is keyed on
> `(bits, group_size)` and NOT on the module type; modules are found by duck
> typing (any module carrying a `(bits, group_size)` pair) rather than a
> class list, because mlx packs routed experts into a
> `QuantizedSwitchLinear` a fixed class list silently misses on MoE
> checkpoints.

```python
def mlx_logits(token_ids: list[int], cached: bool, spec) -> tuple[np.ndarray, dict]
```

> mlx-lm's next-token logits per position, as float32 [rows, vocab].
>
> Row i is the logits after consuming `token_ids[i]`, matching what
> `logit_dump.rs` writes, so the last position is dropped. `cached` picks
> the forward SHAPE: True steps one token at a time through a prompt cache
> (this port's shape, and the headline), False runs one batched pass.

```python
def main() -> None
```

**(no docstring)**

## `kld_mlx_vlm.py` — KL for text+image prompts vs mlx-vlm

Cross-engine KL for a TEXT+IMAGE prompt: this port against mlx-vlm on the
identical checkpoint (ROADMAP M-V5, stage 2). The only instrument here that
puts a picture in front of the model, so the only one that can see which rows
land at which positions, which rope angle each position gets, and whether the
mRoPE selector agrees with the reference's.

| Constant | Value |
|---|---|
| `CHECKPOINTS` | keyed table (qwen38-27b-vision, ...): repo, revision, ... |
| `DEFAULT_QUESTION` | `'Transcribe the text in this image.'` |

### Functions

```python
def snapshot_dir(spec) -> pathlib.Path
```

**(no docstring)**

```python
def log_softmax(row: np.ndarray) -> np.ndarray
```

> Max-subtracted and accumulated in float64.
>
> 248,320 exponentials per row overflow float32 and lose the tail in it;
> `quality_common::negative_log_prob` applies the same discipline for the
> same reason.

```python
def divergences(p_logits: np.ndarray, q_logits: np.ndarray) -> dict
```

> Per-position KL between two logit matrices of the same shape.
>
> A restatement of `kld.py`'s function rather than an import, because that
> module executes an mlx-lm dependency chain at import time and this driver
> runs its `compare` step under numpy alone.

```python
def softcap_of(install: str) -> float
```

> The install's declared `final_logit_softcapping`, or 0.0 for none.
>
> Read the property, never recall it (AGENTS.md Gotcha 38). This family
> declares none, so there is no transform to mismatch and both maxima are
> REPORTED rather than checked against an invented tolerance.

```python
def prepare(args) -> None
```

**(no docstring)**

```python
def shape_floor(args) -> None
```

> The reference token-by-token through a cache, for `compare` to divide by.
>
> WITHOUT THIS THE HEADLINE NUMBER HAS NO SCALE (`crates/bench` Gotcha 8).
> This port walks a prompt one token at a time through a KV cache;
> `prepare` runs ONE batched pass over every position. That is a different
> reduce shape on identical weights, and on the MoE families it alone costs
> 0.0352 mean nats and 4% of the argmaxes.
>
> AND AN IMAGE PROMPT IS NOT A TEXT PROMPT ON THIS AXIS: a batched pass over
> 1,280 image positions attends over a span nothing in the text corpus
> reaches, so the floor has to be measured on THIS prompt rather than quoted
> from the text one. Slow by construction -- one forward per position, no
> batching -- which is why it is a separate mode and not part of `prepare`.

```python
def generate(args) -> None
```

> The reference's own greedy continuation, and this port's beside it.
>
> The divergence numbers are the instrument; this is the thing a reader
> wants to know. It is also the only arm that exercises the DECODE side of
> the position rule -- past the prompt `rope_position` resolves to
> `position + rope_delta`, which no prompt position reaches.

```python
def compare(args) -> None
```

**(no docstring)**

```python
def main() -> None
```

**(no docstring)**

## `mlx_prefill.py` — MLX prefill throughput

Prefill (prompt-processing) throughput for mlx-lm, on the SAME machine. The
cross-engine PREFILL counterpart to `kld.py` and `kld_llamacpp.py`, which
compare LOGITS. Compares wall-clock prompt processing only, because that is
where this port's published gap against the community MLX builds lives.

| Constant | Value |
|---|---|
| `PREFILL_STEP` | `512` |
| `WARMUP_TOKENS` | `64` |

### Functions

```python
def prefill(model, ids, step)
```

> One full prompt through the model, timed the way generation does it.
>
> `mx.eval` on the CACHE STATE per chunk rather than on the logits alone:
> MLX is lazy, so evaluating only the final logits would let chunk
> boundaries collapse and would measure a different program than the one
> `generate` runs.

```python
def main() -> int
```

**(no docstring)**

## `mlx_qmm_reference.py` — MLX quantized-matmul kernel reference

`c(M)` for MLX's own INT4 quantized matmul, at THIS port's matrix shapes. The
KERNEL-QUALITY counterpart to `mlx_prefill.py`: it asks which half of the
prefill deficit belongs to the kernel by running the reference engine's GEMM
on the same matrices, on the same machine, against the same yardstick
`crates/gpu/tests/gemv_bandwidth_bench.rs` uses.

| Constant | Value |
|---|---|
| `QWEN38_SHAPES` | labelled (rows, cols) shape table for the qwen38 family |
| `BITS` | `4` |
| `GROUP_SIZE` | `64` |
| `BATCH_SIZES` | `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]` |
| `TARGET_SECONDS` | `0.1` |
| `MAX_OUTPUT_BYTES` | `1 << 30` |

### Functions

```python
def machine_header() -> str
```

> Power source and load, the two contamination tells that need no sudo.

```python
def build_shape(rows: int, cols: int)
```

> One quantized matrix, allocated ONCE per shape and reused.
>
> Allocating inside a timed region measures the allocation and the GPU's
> first-touch faults rather than the kernel.

```python
def timed(w_q, scales, biases, x, reps: int) -> float
```

> `reps` back-to-back matmuls in one lazy graph, then one eval.
>
> The outputs are kept in a list rather than reduced: a `sum` or an `add`
> per rep would put an elementwise pass inside the region being timed.
> `mx.synchronize` after the eval because `eval` returning is not the GPU
> being done.

```python
def calibrate(w_q, scales, biases, x, rows: int, m: int) -> int
```

> Reps for a ~TARGET_SECONDS region, bounded by the output slab.

```python
def main() -> int
```

**(no docstring)**
