---
title: 'Python Reference: Capture Analysis and Steering Tools'
description: 'Public API of the scripts/ that analyze runtime captures (routing, sparsity, residual snapshots), extract steering vectors, probe agent state loops, and generate app icons.'
diataxisType: reference
---

<!-- generated: python lane, signal: scripts/*.py (20 tracked files; no pyproject.toml/setup.py/setup.cfg) -->

These scripts read JSON/raw sidecar captures that `turbospark-check` writes
under `TURBOSPARK_*` env vars and answer go/no-go questions offline, plus one
asset generator. No script defines `__all__`; every module-level name without
a leading underscore is public. Functions marked **(no docstring)** are public
but undocumented.

## `extract_direction.py` — steering-direction extraction

Extracts a per-layer steering direction from two sets of residual captures.
Reads the capture pairs `TURBOSPARK_RESID_CAPTURE` writes (a small JSON
header beside a raw `.f32` sidecar, see
`crates/runtime/src/resid_capture.rs`), computes one direction per layer, and
writes a control vector in llama.cpp's GGUF layout. Per layer, the difference
of means between the two sets -- the standard extraction, and the one Arditi
et al. use.

GGUF constants: `GGUF_MAGIC = b'GGUF'`, `GGUF_VERSION = 3`,
`GGML_TYPE_F32 = 0`, `DEFAULT_ALIGNMENT = 32`, and value types `VT_UINT32`,
`VT_FLOAT32`, `VT_STRING`.

### Functions

```python
def load_capture(header_path: pathlib.Path) -> tuple[np.ndarray, list[int]]
```

> Return `[snapshot][layer][hidden]` float32 and the captured positions.

```python
def load_set(spec: str) -> np.ndarray
```

> Load every capture under a directory (or one file) into one stack.

```python
def directions(pos: np.ndarray, neg: np.ndarray, method: str) -> np.ndarray
```

> One direction per layer, `[layers][hidden]` float32.

```python
def separation(pos: np.ndarray, neg: np.ndarray) -> np.ndarray
```

> Per-layer effect size: `||mean_p - mean_n||` over the pooled spread.
>
> The raw direction norm cannot be compared across layers and reading it
> that way is the trap this function exists to close. A residual stream's
> magnitude grows with depth, so ranking layers by that norm reports where
> the stream is BIGGEST and reads as where the concept LIVES, which is a
> different claim and usually a different layer. Dividing by the pooled
> within-set spread removes the scale: Cohen's d generalized to a vector.

```python
def stream_share(pos: np.ndarray, neg: np.ndarray, dirs: np.ndarray) -> np.ndarray
```

> Per layer, `||d_l|| / ||x_l||`: the share of the residual an ablation at
> `alpha = 1` removes.
>
> This is the quantity that predicts the COLLAPSE. `separation` divides by
> the within-set spread to ask where the concept lives; this divides by the
> stream's own magnitude to ask what removing it costs. They rank layers
> differently and neither substitutes for the other. `x_l` is the mean
> activation over BOTH sets, which is the stream the edit will actually
> meet at that layer. BUT THIS IS NOT THE FRACTION `ablate` REMOVES -- see
> `removed_share`. Kept because it RANKS layers and because published tables
> cite it.

```python
def removed_share(pos: np.ndarray, neg: np.ndarray, dirs: np.ndarray) -> np.ndarray
```

> Per layer, `|c_hat| / ||x_l||`: the fraction of the row an `ablate` at
> `alpha = 1` ACTUALLY removes.
>
> `stream_share` divides by `||d||`, which is what the direction is; this
> divides by `|c_hat| = |d . x| / ||d||`, which is what the kernel
> subtracts. It is the cosine between the row and the direction, times the
> row. Through the deep layers the two are a factor of exactly 2.0 apart
> (structural: `d` is a difference of means); at layer 0 they are 180x
> apart and the sign of the conclusion flips with them.

```python
def suggested_alpha(share: np.ndarray, budget: float) -> float
```

> The largest `ablate` alpha keeping the worst layer's removal under
> `budget` of the stream.
>
> THIS IS A DIAGNOSTIC AND NOT A PREDICTOR OF THE USABLE BAND. It was
> published as a validated ceiling and that claim is REFUTED
> (2026-08-23, `docs/OBLITERATION.md`): measured against
> `steering_sweep.rs`, the relationship is INVERTED, not merely mis-scaled
> (the register direction carries 3.8x the stream share and tolerates 2x
> MORE alpha). What survives is the ORDERING: both quantities say where
> ablating costs most, which is what `--steering-layers` is chosen from.
> The alpha question is answered by generating and scoring
> (`steering_sweep.rs`); there is no offline substitute.

```python
def write_gguf(path: pathlib.Path, dirs: np.ndarray, arch: str, method: str) -> None
```

**(no docstring)**

```python
def main() -> None
```

**(no docstring)**

## `ffn_sparsity.py` — FFN activation sparsity analysis

Activation-sparsity analysis for `TURBOSPARK_FFN_HIST` captures. Each
argument is a comma-separated group of census JSONs (one per
`turbospark-check` run) summed into one corpus. Reports per layer and as
means: n50/n90/n95/n99 neurons covering that fraction of cumulative |act|
mass (out of `inter`); the coverage curve that decides whether a hot-neuron
working set exists at all; `<t` drops (fraction of activation MASS below a
few magnitude thresholds, the CATS-style operating curve); and ov512..ov4096
overload rates.

| Constant | Value |
|---|---|
| `NON_FFN_GB` | `4.8` |
| `FFN_GB` | `10.9` |
| `THRESHOLDS` | `[0.001, 0.01, 0.1]` |

### Functions

```python
def load_group(arg)
```

**(no docstring)**

```python
def n_frac(mass_row, frac)
```

> Smallest neuron count covering `frac` of the layer's |act| mass.

```python
def mass_below(cap, layer, threshold)
```

> Fraction of the layer's activation mass in buckets under `threshold`.
>
> Bucket b covers [2^(b-off), 2^(b-off+1)); a bucket counts as "below" only
> if its whole range is, so this is a slight UNDER-estimate of the droppable
> mass -- the conservative direction for a go/no-go.

```python
def main()
```

**(no docstring)**

## `router_hist.py` — routing concentration analysis

Concentration and overlap analysis for `TURBOSPARK_ROUTER_HIST` captures.
With one group it reports per-layer routing concentration (how many experts
cover 95% / 99% of the routed mass, and how many were never routed). With
two groups it adds the Jaccard overlap of the two corpora's 95%-mass "hot"
expert sets, the go/no-go number for a domain-pruned or pre-warmed expert
set.

### Functions

```python
def load_group(arg)
```

**(no docstring)**

```python
def hot_set(row, frac)
```

> Smallest expert set covering `frac` of the layer's routed mass.

```python
def main()
```

**(no docstring)**

## `router_window.py` — expert-union cost of batched verify

Expert-union cost of a batched verify, from `TURBOSPARK_ROUTER_TRACE`
captures. `capture.json` is a run of turbospark-check under
`TURBOSPARK_ROUTER_HIST=capture.json TURBOSPARK_ROUTER_TRACE=1`, which
records each layer's top-k expert ids in pass order. `skip` drops that many
leading passes (how prefill is excluded; pass the PROMPT TOKEN COUNT here,
default 0). `M` values default to 2,4,8,16.

### Functions

```python
def windows(row, top_k, m, skip)
```

> Distinct expert count over each window of `m` consecutive passes.

```python
def main()
```

**(no docstring)**

## `pilot_ceiling.py` — expert-prefetch ceiling analysis

What is the CEILING on a one-layer-ahead expert prefetcher? Reads a
`TURBOSPARK_ROUTER_HIST` capture taken with `TURBOSPARK_ROUTER_TRACE=1` and
answers, offline, the questions that decide whether a router-lookahead
prefetcher (colibri's PILOT) is worth building here -- WITHOUT building a
predictor first. A prefetcher can only ever convert a MISS into a hit, so
the miss count is the ceiling on every predictor, perfect or not.

### Functions

```python
def passes(trace_row, top_k)
```

> The flat row split back into one group of `top_k` ids per pass.

```python
def analyse(capture, slot_counts, skip)
```

**(no docstring)**

```python
def pilot_k_sweep(routed, predicted, num_experts, slots, npass, skip, top_k)
```

> colibri's `PILOT_K`: prefetch only the top k of the prediction.
>
> The head of the router's ranking is more reliable than its tail, so a
> narrower prefetch trades coverage for bandwidth. This is the knob that
> decides the whole question here, because the full-width prefetcher reads
> MORE bytes than it saves. `net` is the ratio of total expert reads under
> prefetch to the baseline's demand reads. Below 1.00 the prefetcher moves
> fewer bytes AND hides latency; above it, it is buying overlap with
> bandwidth.

```python
def main()
```

**(no docstring)**

### Classes

```python
class ExpertCache:
```

> Port of `crates/streaming/src/expert_cache.rs`'s LFU policy.
>
> Faithful on the two details that change the answer. Eviction order is
> computed from the counts BEFORE this request's increment (the Rust sorts
> `evictable` and only then bumps `expert_use_count`), and empty slots sort
> ahead of occupied ones regardless of count.

  ```python
  def plan(...)
  ```

  > Returns (hits, misses) as lists of expert ids, and commits.

## `skill_state_probe.py` — SKILL.state agent-loop probe

SKILL.state go/no-go probe: can a LOCAL install drive a bounded-state agent
loop? Measures the two numbers `docs/SKILL_STATE.md`'s "The measurement
that decides" section names: the valid-patch rate (first try, and after one
retry) and the state accuracy at T steps, against the append-only baseline
on the same events. Two arms over one deterministic warehouse task:
`state` ([fixed spec P][current state Sigma][one event] -> JSON Merge
Patch) and `history` ([fixed spec P][every event and reply so far] ->
complete state). Both arms emit a state at every step, so scoring is
identical and the ONLY variable is the prompt shape.

Task constants: `SHELVES` (shelf-1..8), `ITEMS` (sku-01..14),
`SCHEMA_TEXT`, `RULES`, `SPEC_STATE`, `SPEC_HISTORY`, `NOISE`; unit-test
tables `VALIDATOR_CASES`, `MERGE_CASES`, `EXTRACTOR_CASES`.

### Functions

```python
def empty_state()
```

**(no docstring)**

```python
def locate(state, item)
```

> Where the state believes `item` is: a shelf id, 'shipped', or None.

```python
def build_task(seed, steps, noise_every)
```

> Deterministic event stream plus the ground-truth state after each event.
>
> Conservation invariant, asserted here rather than trusted: after every
> event each item is in exactly one place (a shelf, shipped, or unplaced).

```python
def merge_patch(target, patch)
```

> RFC 7386 JSON Merge Patch. Arrays replace; null deletes.

```python
def validate_patch(patch)
```

> Schema errors in a proposed patch. Empty list means valid.

```python
def item_accuracy(state, truth)
```

> Fraction of the 14 items whose location matches ground truth.

```python
def extract_json(text)
```

> First balanced JSON object in `text`, mirroring the server's rescue.
>
> Returns (object, was_whole_reply) or (None, False).
> `was_whole_reply` distinguishes a model that emitted clean JSON from one
> that needed rescuing out of prose, which is the difference between the
> strict and rescued validity rates this probe reports separately.

```python
def run(agent, events, truths, arm, retries = 1)
```

**(no docstring)**

```python
def summarize(steps, truths, tokens, label, arm)
```

**(no docstring)**

```python
def check_units()
```

> Direct cases for the two functions the control agents cannot reach.

```python
def selfcheck(steps_n, seed)
```

> The harness must award 1.00 to a correct agent and less to wrong ones.

```python
def main()
```

**(no docstring)**

### Classes

```python
class LocalAgent:
```

> Server-free controls. These are what make the harness falsifiable.

  ```python
  def reply(...)
  ```

  **(no docstring)**

```python
class ServerAgent:
```

**(no docstring)** -- the real-model arm; drives a `turbospark-server`
process. Its `reply(...)` method is also undocumented.

## `generate_app_icons.py` — macOS app icon generator

Generates Apple macOS compliant app icons (all resolutions, iconset, and
`.icns`) from a source master image, with clean alpha transparency and
proper padding. This is the one script here that is asset tooling rather
than a measurement instrument.

### Functions

```python
def extract_and_generate_icons(source_path: str, output_dirs: list[str])
```

**(no docstring)**
