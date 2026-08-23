# Live directional steering: runtime obliteration, with measurement

The question this page answers: can this engine apply an abliteration-style
directional edit at RUNTIME -- turned on and off between two generations in
one process, with the install untouched -- rather than baking it into weights
at repack time, and can the edited model be measured against the unedited one
without holding two copies?

The answer, so far, is yes on both counts, and the runtime form turns out to
be the better one here rather than a compromise. This is a WORKING page: it
records what is built, what is measured, and what is still a claim. Read the
status table before quoting anything under it.

`ROADMAP.md` item 9 ("Directional Weight Steering") scopes this as a
repack-time weight orthogonalization and names its unmet prerequisite --
"requires activation-capture surface and measurable domain eval". That
prerequisite is what Phase 0 built, so this work unblocks the roadmap item as
well as replacing it. `docs/EXPERT_ROUTING.md` already states the correct
mental model and is worth reading first: abliteration finds one DIRECTION in
the residual stream and does not localize a capability into a prunable region.

## What this is, and what it is not

The engine loads a direction file and applies one of three edits to the
residual stream. It is agnostic to what the direction encodes: the same code
path serves concept steering, style vectors, interpretability probes, and
refusal-direction work. Direction PROVENANCE is the operator's, and extraction
is a separate offline step (`scripts/extract_direction.py`).

This is representation engineering, not pruning and not quantization. Nothing
here makes a model smaller or faster; it changes behaviour and costs a little
throughput.

## Status

| phase | what | state |
|---|---|---|
| 1 | the kernel, its CPU reference, parity tests | **LANDED**, verified |
| 0 | residual capture + offline extraction | **LANDED**, verified on the real install |
| 2 | direction loading, per-family dispatch, CLI/server flags | **LANDED**, working end to end |
| 3 | the A/B probe, coefficient trace, KL against unsteered | not started |

**It works.** On the real `qwen38-27b`, a direction extracted by this engine
from its own activations, applied at runtime with no weight byte modified,
steers generation and reverses under a negative scale:

```
OFF                      From the high-rise window, the city skyline stretches out in a
                         glittering mosaic of lights against the deep blue night.

--steering-mode add      From the hotel window, I can see a vast, shimmering expanse of
--steering-scale 0.5     the ocean stretching to the horizon...
--steering-layers 20:45

--steering-scale -0.5    From the third-floor window, the jagged peaks of the Alps pierced
                         the morning mist, their snow-capped summits glowing...
```

Target install for verification: `~/models/qwen38-27b.gturbo` (dense
`qwen3_5`, hidden 5120, 64 layers), chosen because it carries both a frozen
quality gate and a memory oracle, so a steered-vs-unsteered number is readable
against known baselines.

## The decision: runtime, not repack-time

Four reasons, and the first is the one that makes the rest safe to act on.

**They are the same operation.** Arditi et al. state the weight edit
`W' = W - r_hat r_hat^T W` and the activation edit `x' = x - r_hat r_hat^T x`
are identical in effect; the weight form just precomputes it. So the runtime
version gives up no fidelity, and the paper's own measurements of the
inference-time intervention characterize the weight one exactly.

**The weight edit is actively WORSE on a quantized install.** Every install
here is INT4 / Q8_0 / sub-4-bit. Orthogonalizing weights means dequantize,
edit, requantize -- and `crates/bench/tests/quality_sensitivity.rs` measured
that damaging 0.0122% of routed-expert bytes moves perplexity +10.5%. The
runtime edit runs in FP32 on the accumulator and writes no weight byte.

**It costs a sidecar, not a re-stream.** A full per-layer direction set for
this target is 64 x 5120 floats, 1.25 MB against a 14 GB install. A repack is
~20 minutes and a second copy of the model.

**It makes the A/B possible in one process**, which is the whole measurement
story. A baked model cannot be compared against itself, and every KL floor
this repo trusts (`crates/bench/CLAUDE.md` Gotcha 8) is built by running one
engine two ways.

## The edit

One kernel, three modes, one dot product. Writing `c = d . x` for the raw dot
product and `c_hat = c / ||d||` for the coefficient along the unit direction:

| mode | operation | what it is |
|---|---|---|
| `ablate` | `x -= alpha * c_hat * d_hat` | abliteration; equals the weight edit at `alpha = 1` |
| `add` | `x += alpha * d` | llama.cpp control vectors, ActAdd |
| `clamp` | `x += (target - c_hat) * d_hat` | feature clamping (the Golden Gate shape) |

`c_hat` falls out of every mode for free and is written to a coefficient
buffer. **That is the measurement**: it says how much of the direction the
stream carried at each layer, which is the steered-vs-unsteered signal without
a second model to compare against.

Contract: `turbospark_compute::steering`. Kernel: `steer_direction_fp16` in
`crates/gpu/src/shaders/utility.metal`. Mode enum: `foundation::SteeringMode`,
which lives in the leaf crate because `crates/gpu` carries
`turbospark-compute` as a DEV-dependency only, so an enum declared in
`compute` would be unnameable from the dispatch module that selects on it.

## Measured

Everything in this section was run; nothing is projected.

### Phase 1, the kernel

Parity against the CPU reference at each real hidden size (2048, 2816, 5120),
plus a discriminating guard and a null control. 18 tests in
`crates/gpu/tests/utility_and_pass.rs`, 7 in `crates/compute/src/steering.rs`.

Seven mutations, each reddening only its own cases: ablate's scale, the shader
mode codes, the coefficient's normalization, the gate, the row stride, the
cross-simdgroup merge, clamp's `inv_norm`.

### Phase 0, the capture (2026-08-23, real `qwen38-27b`)

| check | result |
|---|---|
| greedy A/B, capture off vs on | md5 `f4654068...` IDENTICAL |
| sampled A/B (T=0.2, top-k 64, top-p 0.95) | md5 `2edd8bd7...` IDENTICAL, coherent |
| `qwen38_quality_gate` | perplexity 4.9432 exact, both frozen digests exact, 8-slot digest equal |
| workspace suite | 229 targets, 1142 passed, 0 failed |
| capture shape | 64 x 5120 f32, all finite, no zero layer |
| residual norm across depth | 13.9 at layer 0, 535.5 at layer 63 |
| `--max-new` independence | same prompt captured position 21 at both 120 and 200 |
| GGUF container | `direction.1`..`direction.64`, 1-D F32, no zero index, layer 0 round-trips bit-exactly |

The byte-identity result is STRUCTURAL as well as measured: the capture is a
COPY, and a copy cannot change what it copies. That is why it needed no
quality-gate justification beyond running one.

### Phase 2, the edit applied (2026-08-23, real `qwen38-27b`)

Greedy, `--max-new 100`, one prompt, md5 over the generated text alone:

| arm | md5 | reading |
|---|---|---|
| steering off | `485301ec...` | the reference |
| `--steering-scale 0.0` | `485301ec...` | **IDENTICAL**, with the kernel dispatched 64x per token |
| `--steering-mode ablate --steering-scale 1.0` | `89969e1e...` | the edit applies |

**The null control is the load-bearing arm.** At `alpha = 0` the kernel runs
at every layer -- it reduces, it reports a coefficient, it writes back -- and
the output is bit-identical to not running it at all. One comparison covers a
wrong reduction, a wrong buffer offset and a wrong row stride at once, on the
real model rather than a fixture.

Behaviour under the three shapes, all `stop=` recorded:

| arm | result |
|---|---|
| ablate `alpha=1`, all 64 layers | **collapses**: immediate `EndOfTurn`, no text |
| ablate `alpha=1`, layers 25:40 | coherent, wording shifts |
| ablate `alpha=0.3`, all 64 layers | coherent, wording shifts |
| add `alpha=+/-0.5`, layers 20:45 | coherent, concept steered both ways |

### Full ablation at every layer destroys the turn

Ablating at `alpha = 1` across all 64 layers made the model emit its
end-of-turn token immediately. That is not a bug and it is worth stating
plainly, because the natural first command to type is exactly that one.

The mechanism: the residual stream at the late layers IS the output head's
input, and this direction carries a norm of 116 there against the stream's
535. Removing the whole component at every layer damages what the head reads.
Bounding the edit -- either to a layer band or to a fractional alpha -- keeps
the model coherent, which is the empirical form of Arditi et al.'s note that
ablating at 2-3 middle layers is often as effective as ablating everywhere.
`--steering-layers` exists for this and is not an optimization.

### A demonstration, not a result

A 6-prompt-vs-6-prompt ocean/mountain corpus extracts a direction with
per-layer effect sizes of 0.19 to 0.51. That exercises the pipeline end to
end and says nothing about how well the method works: 6+6 is a plumbing
fixture, and the concept pair is deliberately mild.

## Lessons

### A degenerate parameter point can hide a swapped mode

`ablate` at `alpha = 1` and `clamp` at `target = 0` are THE SAME FUNCTION --
both drive the coefficient to zero, and both reduce to a scale of
`-c * inv_norm^2`. The ablate parity case was written at exactly those
parameters, so a mutation swapping the shader's two mode codes PASSED it.

Found by mutation, not by review. Fixed by running that case at a fractional
alpha and a non-zero target, and the identity is now pinned as its own test
(`full_ablation_and_a_zero_clamp_are_the_same_edit`) rather than left as a
trap. This is AGENTS.md Gotcha 48's "assert the fixture discriminates" rule
arriving one level deeper than usual: the file already HAD a discrimination
guard, and that guard was fine -- it was the individual parity case whose
parameters sat on a degenerate point.

### The raw direction norm ranks layers by the residual's scale, not by the concept

A residual stream's magnitude grows with depth: 13.9 to 535.5 here, a factor
of 39. So a difference of means grows with it whether or not the two sets
separate any better, and ranking layers by `||d_l||` reports where the STREAM
is biggest while reading as where the CONCEPT lives.

Measured on the ocean/mountain corpus: by raw norm the answer is layer 63 and
the profile is monotone over a 360x range; by a pooled-spread effect size the
answer is layer 62 and the profile is roughly FLAT at 0.19-0.26 through the
middle before rising to 0.51 near the output. The second is the one that can
be compared across layers. `scripts/extract_direction.py` prints both columns
and says which is which.

The general form is this repo's recurring one (Gotchas 30, 57, 59): an
instrument returning a plausible number for a degenerate or scale-dominated
input, where nobody investigates a confident-looking answer.

### The capture must key on a transition, not on "the last pass"

The obvious implementation overwrites on every forward pass and keeps
whatever ran last. That silently captures a GENERATED token's activation the
moment anyone runs with `--max-new` above 1, and a corpus half-captured at
the wrong positions yields a direction that is a plausible vector and the
wrong one.

Keying on the first pass with `skip_head` false gives the last PROMPT token
regardless of the generation budget, because `run_raw_completion` runs every
earlier prompt token through `produce_prefill`. Measured: position 21 at both
`--max-new 120` and `--max-new 200`.

### A family guard on a capture is not defensive tidiness

A family whose flow contains no copy would write a file of ZEROS. That
extracts to a zero direction, which `steering::inv_norm` then makes inert --
so the whole pipeline would run and steer nothing, with no error anywhere.
`ffn_hist.rs` makes the same argument for the same reason.

## Open, and stated as open

- **llama.cpp interop of the layer indexing is UNVERIFIED.** Their loader is
  1-indexed and rejects `direction.0`; their apply loop runs
  `for il = 1; il < n_layer`, so their layer 0 never receives a direction.
  This writer maps 0-based layer `l` to `direction.{l+1}` and stamps
  `turbospark.layer_base = 0`. Whether that aligns with their numbering is
  untested, and an off-by-one layer is exactly the kind of error that produces
  a plausible wrong answer. Do not assume a vector written here is positioned
  identically under llama.cpp until someone measures it.
- **The qwen family only.** Other families disable the CAPTURE with a
  diagnostic and REFUSE a direction set at open by name -- a set that loaded,
  reported itself on the startup line and changed nothing would be the exact
  silent no-op this whole surface is built to avoid.
- **No throughput number yet.** The kernel adds one dispatch per steered
  layer per token. Nothing has measured what that costs; the layer band is
  the lever if it matters.
- **The batched path does not steer, and steering plus speculation is
  REFUSED at open because of it.** `produce_batched` (the speculative verify,
  and the chunked-prefill driver) has no hook, so the drafted-and-verified
  tokens -- which are the ones COMMITTED -- would come from the unsteered
  model while sequential-fallback tokens came from the steered one. That is a
  silent mixture of two models, coherent and wrong. Refused by name until the
  batched path carries the edit; the kernel already takes `rows` and
  `row_stride` for exactly that.
- **No integration test for the capture**, matching how `ffn_hist` and
  `router_hist` are treated: an env-gated capture needs a process-global
  write, which races other tests in the same binary. The real-model A/B is
  the gate.
- **No alpha cap.** `ablate` cannot overflow (it only removes a component of
  `x`), but `add` and `clamp` can push an FP16 stream past 65,504, arriving as
  `inf` and then NaN -- which reads as a PERFECT score on any rank instrument
  (Gotcha 59). A numeric cap was declined as a fabricated threshold (Gotcha
  38's rule); the finiteness assertion belongs to Phase 3's probe, at the
  point a measurement is taken.
- **No throughput number.** The kernel adds one dispatch per hooked layer per
  token -- 64 on this model against a decode step's several hundred -- and the
  bytes are negligible against a projection's weight read. That is an
  arithmetic expectation, not a measurement, and it is Phase 3's to take.

## Reproducing

```sh
# 1. Capture, one run per prompt. `--max-new 1` is enough; the hook keys on
#    the prefill/decode transition, so a larger budget captures the same row.
MFERENCE_RESID_CAPTURE=/tmp/steer/pos/p1.json \
  ./target/release/turbospark-check --model ~/models/qwen38-27b.gturbo \
  --messages-file /tmp/prompt.json --max-new 1 --temperature 0.0001 --top-k 1
```

```sh
# 2. Extract. Reads a DIRECTORY of captures per set; prints the per-layer
#    effect size, which is the column to read when picking layers.
uv run --python 3.12 --with numpy scripts/extract_direction.py \
  --positive /tmp/steer/pos --negative /tmp/steer/neg --out /tmp/steer/d.gguf
```

```sh
# 3. Steer. Start with a LAYER BAND and a fractional scale: ablating at 1.0
#    over every layer collapses the turn (see above).
./target/release/turbospark-check --model ~/models/qwen38-27b.gturbo \
  --messages-file /tmp/prompt.json \
  --steering /tmp/steer/d.gguf --steering-mode add \
  --steering-scale 0.5 --steering-layers 20:45
```

```sh
# The null control, which is what to run first after touching the kernel:
# with the edit dispatched at every layer and alpha zero, the output must be
# byte-identical to steering off.
./target/release/turbospark-check --model ~/models/qwen38-27b.gturbo \
  --messages-file /tmp/prompt.json --steering /tmp/steer/d.gguf --steering-scale 0.0
```

```sh
# Any control vector on disk, including a published repeng one, reported
# without loading a model.
TURBOSPARK_CONTROL_VECTOR=/tmp/steer/d.gguf \
  cargo test -p turbospark-repack --test control_vector_file -- --ignored --nocapture
```

The server takes the same flags, resolved once at startup; unlike
speculation there is no per-request half, so a server started with
`--steering` steers every request it serves.

## Reading list

- Arditi et al., "Refusal in Language Models Is Mediated by a Single
  Direction" (NeurIPS 2024) -- the ablation operation, the weight/activation
  equivalence, and the finding that 2-3 middle layers are often as effective
  as all of them.
- Turner et al. (2023), activation addition -- the `add` mode.
- llama.cpp `--control-vector-scaled` and vgel's `repeng` -- the file format
  this port reads, and the published vector sets it makes reachable.
- `elder-plinius/OBLITERATUS` -- the toolkit this question came from. Its
  weight-projection half is what ROADMAP item 9 scoped; its steering-vector
  half is what this page builds.
