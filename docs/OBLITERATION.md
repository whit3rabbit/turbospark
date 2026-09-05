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

The engine loads a direction file and applies one of four edits to the
residual stream. It is agnostic to what the direction encodes: the same code
path serves concept steering, style vectors, interpretability probes, and
refusal-direction work. Direction PROVENANCE is the operator's, and extraction
is a separate offline step (`scripts/extract_direction.py`).

This is representation engineering, not pruning and not quantization. Nothing
here makes a model smaller or faster: it changes behaviour and costs
throughput, measured below at 1.72% of decode with all 64 layers steered and
0.75% at 26 of them.

## Status

| phase | what | state |
|---|---|---|
| 1 | the kernel, its CPU reference, parity tests | **LANDED**, verified |
| 0 | residual capture + offline extraction | **LANDED**, verified on the real install |
| 2 | direction loading, per-family dispatch, CLI/server flags | **LANDED**, working end to end |
| 3 | the A/B probe, coefficient trace, KL against unsteered | **LANDED**, four arms green |
| 4 | the alpha sweep, the `renorm` mode, the layer-band axis | **LANDED**; both of the sweep's predictions REFUTED, which is the result |
| 5 | a SECOND direction and a second prompt | **LANDED**; the two refutations REPLICATE, and the derived alpha ceiling does not survive |
| 6 | the BATCHED path, and steering beside speculation | **LANDED**; lossless on both drafters, and acceptance barely moves |
| 7 | the throughput cost | **LANDED**; -1.72% at all 64 layers, -0.75% at 26, `renorm` free |
| 8 | llama.cpp interop | **LANDED**; the numbering was OFF BY ONE and is corrected, no measurement here moves |
| 9 | a fifth family (Gemma 4) and its chunked prefill driver | **LANDED**; found `encode_steering`/`encode_resid_capture` hardcoded the edited row at offset 0, fixed with an `x_off` parameter, mutation-checked on the real chunked path |
| 10 | the sixth and seventh flows (`gpt-oss`, `muse_glimmer`) | **LANDED** on synthetic fixtures, mutation-checked, and **BOTH NOW MEASURED ON A REAL INSTALL** since 2026-08-25 (below). `muse_glimmer`: null control byte-identical, memory oracle clean, but at the time of that measurement the probe's single-position divergence check did not clear its (`qwen3_5`-borrowed) floor at the prompts tried, despite real coefficients and CLI-visible divergence over a generation. `gpt-oss`: the same pattern, one step sharper -- the null control passed on both instruments, the probe's single-position check missed for a now-EXACT reason (the position it measured is Harmony's near-fixed `<|channel|>` token, decoded and confirmed rather than guessed), and the edit was visibly real once generation ran past that token: coherent, differently-worded output at alpha 0.3, and a DIFFERENT failure mode from `qwen38-27b`'s at alpha 1.0 (an unresolved reasoning loop rather than an immediate collapse). **The single-position check itself is now FIXED, generally** (arm 2 asserts on a windowed teacher-forced KL trace rather than one position -- see "Open, and stated as open" below and `crates/bench/CLAUDE.md` Gotcha 25); re-captured with a fresh direction on each real install the same day, arm 2 now PASSES on both: museGlimmer's window max reads 11440x its floor (against 0x at the old single position), gpt-oss's reads 33x (against 0x at a position confirmed to decode to `<|channel|>` on both engines) |
| 11 | Swift bindings and demo GUI integration | **LANDED**; full steering options in `turbospark-ffi` wire types / `turbospark.h`, `swift/TurboSpark` (`OpenOptions`, `SessionInfo.Steering`), and `swift/TurboSparkApp` status footer |
| 12 | a CAPABILITY report and a preset UI | **LANDED**; `steering.supported`/`reason` over the ABI from the same predicate the open refuses with, `ts_control_vector_info_json` for a pre-open shape check, and named presets with a compatibility check plus a reload prompt in `swift/TurboSparkApp` |

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

## CLI and Interfaces

`--steering <path.gguf>` on BOTH `turbospark-check` and `turbospark-server` --
same flag, no separate binary, no rebuild. A direction is a llama.cpp-layout
control vector (`.gguf`): one this port extracted with
`scripts/extract_direction.py`, or a foreign one such as `repeng`'s or
`jukofyork`'s published sets, provided it declares the checkpoint's own
`hidden_size` (`SteeringSet::validate` checks the shape and nothing about the
concept, so a vector for the wrong model opens and steers something, silently
-- see "Running someone else's vector" under Reproducing).

| Flag | Takes | Default | Meaning |
| --- | --- | --- | --- |
| `--steering` | path | none (off) | the control vector; every other flag below is REFUSED at parse time without this one |
| `--steering-mode` | `ablate` \| `add` \| `clamp` \| `renorm` | `ablate`, or whatever the file declares | which edit |
| `--steering-scale` | float | `1.0` | strength; `0.0` is the exact identity in every mode |
| `--steering-layers` | `START:END` | every layer the vector covers | 0-based, inclusive layer band |
| `--steering-target` | float | `0.0` | the coefficient `clamp` pins to; ignored by `ablate`/`add` |
| `--steering-gate` | float | `0.0` (always fires) | only steer where the direction's own coefficient reaches this magnitude |

```sh
turbospark-check --model qwen38-27b --messages-file /tmp/p.json \
  --steering /tmp/steer/d.gguf --steering-mode add \
  --steering-scale 0.5 --steering-layers 20:45
```

`turbospark-server --model ... --steering ...` takes the same flags, resolved
ONCE at startup: there is no per-request override the way there is for
`reasoning_effort`, so a server started with `--steering` steers every
request it serves for the life of the process.

### Swift bindings and native apps

Native applications configure steering via `OpenOptions` when initializing a
`TurboSparkSession`. Like the CLI and server, the vector is validated and
loaded at open time:

```swift
var options = OpenOptions()
options.steering = "/path/to/vector.gguf"
options.steeringMode = .add             // .ablate, .add, .clamp, .renorm
options.steeringScale = 0.5
options.steeringLayers = "20:45"
options.steeringTarget = 0.0            // for .clamp mode
options.steeringGate = 0.0              // threshold to trigger edit

let session = try await TurboSparkSession(modelPath: "qwen38-27b", options: options)
```

**`steering.supported` IS A DIFFERENT QUESTION FROM `steering.active`, AND IT
IS THE ONE A UI GATES ON.** `active` says a vector is running; `supported`
says one COULD be, from `family_dispatches_steering`. An open is REFUSED on a
family that answers false, so an app offering the knob there offers one whose
only outcome is a failed load, and an unsteered session on a family that
steers answers `active: false, supported: true` -- gating on `active` would
disable the control for every model that is not already steering, i.e. all of
them at first open.

```swift
if session.info.steering.supported {
    print(session.info.steering.summary ?? "off")
} else {
    print(session.info.steering.reason!)   // the open's own refusal wording
}
```

`ts_control_vector_info_json(path)` reads a vector's header with no model, no
session and no network, returning `hidden`, `coveredLayers`, `minLayer`,
`maxLayer`, `spannedLayers`, `declaredMode` and `declaredArch`. It exists so a
host can say "this file is 4096 wide and your model is 5120" instead of
failing minutes into a load, and it reads the same parser the open reads.
`TurboSparkCatalog.controlVectorInfo(path:)` is the Swift form. Two things it
does NOT tell you: **a shape match is not a semantic match** (see "Running
someone else's vector"), and `declaredArch` is advisory because nothing
validates against it. `minLayer` is normally 1, which is the interop
convention working rather than a gap.

The C ABI (`turbospark.h`) exposes `steering`, `steeringMode`, `steeringScale`,
`steeringLayers`, `steeringTarget`, `steeringGate` in `ts_session_open` JSON
and reports
`{ "active": bool, "supported": bool, "reason": string?, "mode": string?, "scale": number?, "summary": string? }`
in `ts_session_info_json`.

**`swift/TurboSparkApp` SURFACES THIS AS NAMED PRESETS, NOT AS SIX RAW
KNOBS.** Settings > Safety and Steering registers directions (name, vector,
mode, strength, layer band, notes), checks each one's shape against the
selected install's `manifest.json`, and has one switch. The Inspector keeps
the raw fields as the expert surface and what is set there becomes an implicit
"Custom" preset. Three things the pane is deliberate about:

- **It states that nothing ships a direction** and points at
  `scripts/extract_direction.py`. A control labelled as though a behaviour
  shipped would claim work that does not exist.
- **The default strength is 0.3, not 1.0**, because 1.0 over every layer is a
  documented collapse (below) and a default landing on a documented failure
  mode is worse than no default. Past 0.8 the editor warns with the measured
  bands.
- **It says when the loaded model is not running the current setting.**
  Steering resolves once at OPEN, so a changed preset does nothing until a
  reload, and a switch that silently did nothing is the failure this whole
  page is about. The Server pane reports each attached model's
  `info.steering` read-only for the same reason: that server serves
  already-open sessions, so steering is a property of the load rather than of
  the server.

**Wired today (seven of eight families)**: the qwen flow (both halves,
per-token and batched-verify), `families/llama/` (Mixtral, `qwen3moe`, and
the dense Mistral / Llama 2/3.x half), `families/gemma4/` (sequential decode
and its chunked prefill driver), `families/gptoss/` (its one call site, the
routed-MoE tail's raw residual add), and `families/museglimmer/` (its one
call site, the FFN-half sandwich tail's residual add). Both `gpt-oss` and
`museglimmer` are now measured on real installs (below, both since
2026-08-25): clean null controls and clean memory oracles on both, though
the probe's single-position divergence check does not clear its floor at
the prompts tried on either -- for `museglimmer` an unverified borrowed
floor, for `gpt-oss` an EXACT mechanism (the position it measures is
Harmony's near-fixed `<|channel|>` token), with the edit visible in
generated text past that token on both.
Requesting `--steering` against the one remaining unwired family
(DeepSeek-V4-Flash, whose compressed-attention kernels are unported, so
there is no decode flow at all to hook) is refused at open BY NAME rather
than silently loaded and ignored -- see Open, below.

## The decision: runtime, not repack-time

Four reasons, and the first is the one that makes the rest safe to act on.

**They are the same operation.** Arditi et al. state the weight edit
`W' = W - r_hat r_hat^T W` and the activation edit `x' = x - r_hat r_hat^T x`
are identical in effect: the weight form just precomputes it. So the runtime
version gives up no fidelity, and the paper's own measurements of the
inference-time intervention characterize the weight one exactly.

**The weight edit is actively WORSE on a quantized install.** Every install
here is INT4 / Q8_0 / sub-4-bit. Orthogonalizing weights means dequantize,
edit, and requantize -- and `crates/bench/tests/quality_sensitivity.rs` measured
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

One kernel, four modes, one reduction pass. Writing `c = d . x` for the raw
dot product and `c_hat = c / ||d||` for the coefficient along the unit
direction:

| mode | operation | what it is |
|---|---|---|
| `ablate` | `x -= alpha * c_hat * d_hat` | abliteration; equals the weight edit at `alpha = 1` |
| `add` | `x += alpha * d` | llama.cpp control vectors, ActAdd |
| `clamp` | `x += (target - c_hat) * d_hat` | feature clamping (the Golden Gate shape) |
| `renorm` | `ablate`, then scale the row back to `\|\|x\|\|` | norm-preserving projection |

`renorm` is the newest and is the one that needed an argument rather than a
formula, so it has its own section below. The short version: it costs no
second reduction and no second pass, and it leaves the other three
bit-identical.

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

Everything in this section was run. Nothing is projected.

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

That container row is a PHASE 0 reading and the numbering under it was
corrected on 2026-08-24: a file written now carries `direction.1`..
`direction.63` for blocks 1..63, and block 0 is not written at all. The
vectors this phase produced still read as blocks 0..63, because they declare
`turbospark.layer_base = 0` and the reader honours it. See the interop
section below.

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

### Phase 3, the probe (2026-08-23, real `qwen38-27b`)

`crates/bench/tests/steering_probe.rs`. Three opens of ONE install, ~10 s of
measurement each, because the edit is rank-1 and reversible and therefore
needs no second model to compare against. Four arms:

| arm | at `ablate` alpha 0.3, all 64 layers |
|---|---|
| 1, null control at alpha 0 | **0 of 248,320 logits differ**, KL exactly `0.000e0`, 24 greedy tokens identical |
| 2, steered vs unsteered | KL **0.10142 nats**, 13,705x the 7.4e-6 dense shape floor |
| 3, coefficient trace | 64 layers reporting, `|c|` from 0.0484 to 20.9949 |
| 4, determinism | two 24-token runs from one checkpoint agree |

The steered turn stays coherent and its wording shifts, parting from the
unsteered continuation at token 6:

```
unsteered: " I am an AI, I don't have eyes to see a specific body of water
             in front of me. However,"
steered:   " I am an AI, I do not have physical senses and cannot see or
             touch water directly. However, I can describe"
```

**That arm 3 row was corrected on 2026-08-23 and is worth a sentence.** It
read `0.0491 to 29.9005` and reproduces from nothing: three separate vector
files on this disk all give `0.0484 to 20.9949`, and so does the kernel as it
stood BEFORE the `renorm` change (checked by reverting the shader and
re-running, because a 30% move in a reported number is not something to ship
past). Arms 1, 2, and 4 reproduce to the digit, so nothing regressed -- the
figure was simply wrong when written. The reason it survived is the reason to
record it: **arm 3 REPORTS where the others ASSERT**, so no test could ever
have reddened on it. A number in a table that nothing checks is a number
nobody re-runs.

**Arm 1 is the load-bearing one and it is the only assertion of the four
that is about the KERNEL.** At alpha 0 the dispatch runs at every layer, it
reduces, it reports a real coefficient, it writes back -- and the result is
bit-identical to not running it at all, 24 tokens deep. One comparison covers
a wrong reduction, a wrong buffer offset and a wrong row stride. Mutation-
checked: pointing that arm at a non-zero alpha reddens it at 248,088
differing logits.

**Arm 2 is the one arm in this repo where a LARGE divergence is the success
signal**, and the floor is what says it is the edit rather than noise. That
inversion is worth stating, because every other divergence measurement here
(the cross-engine KLs, the batched-forward probe) wants a small number.

### A divergence number cannot tell a steered model from a destroyed one

**The probe's first run defaulted to `ablate` at alpha 1.0 over all 64
layers, read 21.818 nats -- 2,948,321x the floor -- and passed.** That is the
operating point the section below already records as COLLAPSING the turn. The
model emitted its end-of-turn token immediately and generated nothing, and
"the distribution moved enormously" is exactly what that looks like in a KL.

A model that stops generating diverges hugely from one that does not, so the
number was real and meant nothing about steering. This is the repo's
characteristic failure -- an instrument reading a plausible value on
degenerate input (Gotchas 30, 57, 59) -- arriving on a new axis, and it was
caught only because the probe prints the two continuations side by side.

Two things changed as a result, both cheap:

- **The default scale is 0.3, not 1.0.** A probe whose natural default lands
  on a documented failure mode and reports it as a success is worse than no
  probe.
- **A collapse detector runs after the text.** Fluency is a judgement no test
  can make, but the documented collapse mode has an objective proxy: the
  turn's first token is end-of-turn, or the whole continuation is one or two
  distinct tokens. It REPORTS rather than asserts -- deliberately measuring
  the collapse point is legitimate -- and it names the operating point in its
  message. Verified to fire at alpha 1.0 ("3 distinct tokens, starting with
  end-of-turn").

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

### The stream share, and an alpha ceiling derived rather than looked up

The section above explains the collapse with a ratio. That ratio is
computable offline, per layer, from the captures already taken, so
`scripts/extract_direction.py` prints it as a third column and turns the
prose into a number handed over before anything is run.

`share = ||d_l|| / ||x_l||`, where `x_l` is the MEAN ROW NORM over both sets:
the fraction of the residual an `ablate` at `alpha = 1` removes.

That denominator is why this reads 25.4% at layer 63 where the section above
divides 116 by 535 and gets 22%. Over this corpus the layer 63 row norms run
392.7 to 526.9 with a mean of 457.9, and `116.34 / 457.9` is the 25.4%. The
535.5 comes from the Phase 0 capture check on a DIFFERENT prompt and sits
just outside this corpus's range, which is ordinary prompt-to-prompt
variation rather than a discrepancy.

Both are fine. They are different
quantities, and a reader who divides one of these tables by the other will
reproduce neither.

| layer band | norm | sep (effect size) | share |
|---|---|---|---|
| 0 | 0.049 | 0.453 | 0.4% |
| 3-46 | 0.24 - 10.2 | 0.185 - 0.304 | 0.9% - 9.2% |
| 51-58 | 28.8 - 62.1 | 0.35 - 0.44 | 19.0% - 24.8% |
| 59-63 | 74.6 - 116.3 | 0.48 - 0.51 | 25.4% - 27.9% |

**The three columns rank three different layers, which is the whole reason
all three are printed**: raw norm says 63, effect size says 62, stream share
says 59. Each divides `||d_l||` by something different and answers a
different question -- by nothing, by the within-set spread, by the stream's
own magnitude.

The derived ceiling on this corpus is **alpha <= 0.36** at a 10% budget.
That is a real prediction rather than a fit: it was computed from the
captures, and the two operating points already measured by hand sit either
side of it -- 0.3 coherent, 1.0 collapsed. `--alpha-budget` moves it.

**IT DOES NOT TRANSFER TO A SECOND DIRECTION, and the section on that is
below.** On a register direction off the same checkpoint it predicts 0.09
against a measured 0.8, and the relationship is inverted rather than
mis-scaled. Read this whole subsection as a description of one corpus and the
layer RANKING as the part that survives: the numbers below are correct and the
prediction built on them is not.

**It is a CEILING and not a RECOMMENDATION**, and the distinction is the
whole honesty of the number. It says where the edit starts damaging what the
output head reads. It says nothing about where the edit starts WORKING,
which is a property of the direction and the concept and is not computable
from norms. An alpha under the ceiling that steers nothing is an ordinary
outcome.

One row deserves suspicion rather than use: layer 0 reads an effect size of
0.453, second highest in the model, at a share of 0.4%. A layer where the
concept separates strongly and ablation is nearly free would be an excellent
place to steer. It is more likely an artifact -- the layer 0 residual is
essentially the token embedding, so a corpus of similarly-shaped prompts has
a tiny within-set spread, and `sep` divides by that.

Worth measuring before
it is believed. Llama.cpp never applies a direction at layer 0 anyway.

**MEASURED, AND THE "nearly free" HALF IS AN ARTIFACT OF THE FORMULA RATHER
THAN OF THE CORPUS.** `share` divides by `||d||`, but `ablate` removes
`|c_hat|`, and at layer 0 those differ by **180x**: the true fraction removed
is 72.1%, making it the most expensive layer in the model to ablate rather
than the cheapest. The guessed mechanism was right and applies to both
columns -- a near-embedding residual sits close to a low-dimensional subspace,
which shrinks the within-set spread `sep` divides by AND raises its cosine
with any direction extracted from it.

### The alpha sweep, and the derived ceiling checked against it (2026-08-23)

`crates/bench/tests/steering_sweep.rs`. Six opens, ~2 min. Generate greedily
under a steered engine at each alpha, then teacher-force those exact ids
through the UNSTEERED engine. One install is both models, which is the whole
reason this measurement is affordable.

The anchor is the unsteered model on the frozen reference answer: **4.9432**,
which is what ordinary human prose costs it. It reproduces
`qwen38_quality_gate`'s frozen number to the last digit, so this target and
that gate provably walk the same ids.

| alpha | ppl under unsteered | x anchor | distinct | diverges at |
|---|---|---|---|---|
| 0.0 | 1.3187 | 0.3x | 34 | none |
| 0.2 | 1.5434 | 0.3x | 35 | 7 |
| 0.4 | 1.7158 | 0.3x | 36 | 8 |
| 0.6 | 5.1736 | 1.0x | 13 | 0 |
| 0.8 | 313.99 | 63.5x | 2 | 0 |
| 1.0 | 280.31 | 56.7x | 5 | 0 |

**USABLE BAND: up to alpha 0.4, against a ceiling of 0.36 derived from the
captures alone.** Two instruments sharing no code and no inputs beyond the
same corpus -- one arithmetic on activations with no generation at all, one
generated text scored under the unedited model -- landing one sweep step
apart.

**THAT WAS WRITTEN AS "the derived ceiling validated rather than merely
plausible" AND IT IS WITHDRAWN.** One agreement on one corpus is not a
validation, and the second direction measured on this page reads 0.09 derived
against 0.8 measured, inverting the relationship rather than scaling it. Two
instruments agreeing once is worth exactly one datapoint, and the reason this
one read as more than that is that it was the only pair anyone had. The
measured band in the table below stands. The ceiling beside it does not.

At 0.6 the model emits template markup (`assistant\n<think>\n\n</think>`
repeating) at 13 distinct tokens. At 0.8 it says `" contains"` and stops.

### The steepest step is the wrong criterion, and it is inside the wreckage

The sweep's first verdict rule was the largest multiplicative jump between
neighbours, chosen to avoid a fabricated threshold. **It gives the wrong
answer on this data**: it picks 0.6 -> 0.8 at 60.7x and concludes everything
at or below 0.6 is usable, when 0.6 is already broken. Once output is
degenerate the number keeps climbing, so the biggest jump lands well past the
point anyone cares about.

The criterion is the ANCHOR CROSSING instead, and it is not a fabricated
constant either. The anchor is measured, and the argument for comparing to it
is structural: a model's own GREEDY output is the argmax path, so it should
be far MORE predictable to that model than human writing is. The fluent arms
sit at 0.3x. An arm whose own greedy output is as surprising as human prose
has stopped producing its own distribution's typical text.

Two things the table shows that no single value can. The column **stops being
monotone past the crossing** -- 0.8 scores higher than 1.0 -- so ordering
broken outputs by perplexity means nothing. And `distinct` collapses from
34-36 to 13/2/5 exactly where perplexity explodes, which is two independent
signals agreeing on where the break is.

### The fourth mode, and the prediction it was built to test (2026-08-23)

`renorm` is `ablate` followed by a rescale of the row back to its original
`||x||`. The OBLITERATUS review below listed it as the one remaining idea with
a measurable prediction attached: it attacks the collapse from the opposite
side to the derived ceiling -- the ceiling AVOIDS the damage, this REPAIRS it
-- so **its usable band should reach alpha 1.0 where the other modes stop
near 0.4**.

**That prediction is REFUTED.** Both modes break by alpha 0.6.

| alpha | `ablate` ppl | distinct | `renorm` ppl | distinct |
|---|---|---|---|---|
| 0 | 1.3187 | 34 | 1.3187 | 34 |
| 0.4 | 1.7158 | 36 | 1.7176 | 35 |
| 0.45 | 1.7883 | 37 | 1.8331 | 33 |
| 0.5 | 2.8168 | 34 | 2.8797 | 37 |
| 0.55 | **2.1309** | **18** | 3.3624 | 36 |
| 0.6 | 5.1736 | 13 | 5.2974 | 26 |

**What it DOES buy is real and much smaller than predicted.** Both modes score
the same anchor crossing (0.55), and only one of them is telling the truth at
the top of that band. At 0.55 `ablate` has already degenerated -- 18 distinct
tokens, one sentence repeated with template markup between the copies -- while
`renorm` reads "Here is what I notice about the water: 1. **It is clear and
transparent**...", fluent at 36. At 0.6 it holds 26 distinct against 13.

So the honest claim is that norm preservation degrades more GRACEFULLY through
the knee and buys roughly one grid step of genuinely usable strength, not the
jump to full ablation the idea promised. On this direction and this prompt.

Two things worth not re-deriving. The mechanism the prediction rested on is
narrower than it looks: the final RMS norm is scale-invariant, so if the
collapse were simply "the head reads a smaller vector" the model's own norm
would already undo it and this mode would change nothing at all. The only
thing it can repair is the RESIDUAL ADD, which is not scale-invariant --
shrinking `x` at every layer amplifies each subsequent sublayer's relative
contribution. That mechanism is real, and it is evidently not what dominates
the collapse.

And at alpha 1.0 `renorm` scores WORSE than `ablate` (8927
against 280). Both are word salad there, and ordering broken outputs by
perplexity means nothing, so that row is reported rather than read.

It costs no second reduction and no second pass, which is the part that made
it cheap enough to test: `||x||^2` fuses into the loop that already computes
the coefficient, and the post-edit norm is analytic
(`||x'||^2 = ||x||^2 - alpha*(2 - alpha)*c_hat^2`, exactly, since the
projection is orthogonal -- Pythagoras at `alpha = 1`). Its rescale factor is
exactly `1.0` in the other three modes and `1.0 * v == v` in IEEE-754, so one
write loop serves all four and the older three are unmoved to the BIT. That is
measured rather than argued: the `ablate` column above reproduces this page's
frozen sweep table to the last digit, and the GPU's two bit-identity cases
stayed green through the change.

### The layer band is not the lever the stream share suggested (2026-08-23)

The stream-share column says the damage concentrates in layers 51-63, so a
band excluding them should carry a higher usable alpha. `--steering-layers` is
the knob and nothing had measured it.

**Also refuted, and more flatly.** Sweeping `all` against `0:50` (51 of the
vector's 64 layers, so every high-share layer dropped):

| band | usable alpha, `ablate` | usable alpha, `renorm` |
|---|---|---|
| all | 0.4 | 0.4 |
| 0:50 | 0.4 | 0.4 |

On the coarse grid the two bands are not merely equal -- under `ablate` they
are **byte-identical through alpha 0.6**, four arms agreeing to the last digit
of perplexity and the last token of text. The restriction is doing something
(the arms differ at 0.8 and 1.0), it just does not touch the greedy path at
any strength anyone would use.

The reading that survives: the stream share explains WHY full ablation
collapses, and it does not follow that excluding the high-share layers
recovers headroom. Those layers carry the largest share of the direction and
apparently contribute little to the argmax path until the model is already
broken.

### The anchor crossing under-calls damage, and `distinct` is what catches it

Third instance on this page of a verdict rule being wrong, after the steepest
step and the probe's alpha-1.0 default -- and this one was invisible on the
coarse grid, because both modes happened to land on the same answer there.

At alpha 0.55 `ablate` reads 2.1309, which is 0.4x the anchor and comfortably
inside the usable band by the stated criterion. Its output is one sentence
repeated with template markup between the copies. The perplexity is not
lying -- a short degenerate repetition IS predictable to the unsteered model --
it is answering a different question from the one being asked of it.

`distinct` is the independent signal, and this page already noted the two
"agreeing on where the break is" as what made the original table readable.
The fine grid is where they come apart: `ablate`'s vocabulary collapses
34 -> 18 at 0.5 -> 0.55, AT the crossing, while `renorm`'s falls 36 -> 26 at
0.55 -> 0.6, ABOVE it.

`steering_sweep.rs` reports that disagreement now. Deliberately REPORTS: there
is no measured basis for a "distinct must stay above N" line, so inventing one
to resolve the conflict would be the fabricated threshold Gotcha 38 warns
against. It prints the largest consecutive drop and says when it lands at or
before the crossing, and an operator choosing a strength gets both numbers
rather than one silently preferred. Its own discrimination check is the pair
above: the warning fires for `ablate` and stays quiet for `renorm`, on the
same grid, which is what says it is reading the difference and not the noise.

### The second direction: what replicated, and the one thing that did not (2026-08-23)

Every number above this point came from ONE 6+6 ocean/mountain corpus on one
prompt. Two predictions had been refuted on it, and nothing in the repo could
say whether those were facts about steering on this engine or facts about that
direction. This is the answer, and it is different for each claim.

The second corpus is a REGISTER pair -- `"<topic>. Use formal academic
language."` against `"<topic>. Use casual conversational language."`, six
matched pairs holding the topic constant inside each pair, so the only thing
varying is the register clause. It was chosen to be a different KIND of
direction (a style, not a topic) and it came out far stronger:

| | ocean/mountain | register |
|---|---|---|
| effect size (`sep`) | 0.185 - 0.511 | **1.087 - 4.182** |
| peak stream share | 27.9% (layer 59) | **105.7% (layer 57)** |
| derived alpha ceiling | 0.36 | **0.09** |
| MEASURED usable band | 0.4 | **0.8** |

`steering_probe` is green on it end to end, which is worth stating because
arm 1 is the only assertion on this page that is about the KERNEL: at alpha 0
the dispatch runs at all 64 layers and the output is bit-identical to steering
off, 24 tokens deep, on a direction and a prompt the kernel had never seen.
Arm 2 reads 3.6063e-3 nats at alpha 0.1, 487x the shape floor.

The qualitative check lands too, and it is the sign the arithmetic cannot
give: ablating a FORMALITY direction should make the model less formal, and
the one visible change at alpha 0.1 is `"a physical phenomenon known as"`
becoming `"a physical phenomenon called"`.

**BOTH REFUTATIONS REPLICATE.** `renorm` again fails to extend the usable band
and again degrades far more gracefully past it -- at alpha 1.0 both modes are
degenerate at 3 distinct tokens while `ablate` reads perplexity 4727.1131 and
`renorm` 3.8793, a thousandfold in graceful-failure terms and nothing at all
in usable strength.

And the layer band again fails to buy headroom: `all` and
`0:50` both stop at 0.8. That second one replicates by a DIFFERENT mechanism,
which strengthens it -- on the ocean direction the restriction left the greedy
path byte-identical through 0.6, i.e. it did nothing. Here it visibly moves
arms (1.3268 against 1.2753 at 0.2) and still does not move the band.

**THE DERIVED ALPHA CEILING DOES NOT SURVIVE, AND THE FAILURE IS NOT A
MIS-SCALING.** It predicted 0.09 against a measured 0.8. The section above
this one called it "validated rather than merely plausible" on the strength of
0.36 against 0.4, and that agreement was a coincidence of one corpus. Worse
than the 8.9x: **the relationship is inverted.** The register direction
carries 3.8x the stream share and tolerates 2x MORE alpha, where the formula
says tolerable alpha falls as share rises. No budget constant fixes a sign.

Three things were checked before that was written down, in the order that made
each one cheap.

**Was it the direction or the prompt?** Both changed at once, so neither was
attributable. Crossing them is two more sweeps:

| | water prompt | sky prompt |
|---|---|---|
| ocean direction | 0.4 | 0.4 |
| register direction | 0.8 | 0.8 |

The band is a property of the DIRECTION and the prompt does not move it. That
also reproduces this page's frozen ocean band on a prompt it was never
measured on.

**Is the share formula measuring the wrong thing?** Yes, and this part is a
real correction rather than a caveat. `ablate` removes `alpha * c_hat * d_hat`,
whose length is `alpha * |c_hat|`: `share` divides by `||d||`, which is what
the direction IS rather than what the stream carries of it. Measured on both
corpora the two sit a factor of exactly **2.0** apart through the deep layers,
and that is structural: `d` is a difference of means, so where the direction
dominates what separates the sets, a positive row sits near `+||d||/2` along it
and a negative one near `-||d||/2`. A constant factor is precisely why `share`
ranks layers usefully and calibrates badly.

**It does not rescue the prediction** -- corrected ceilings are 0.14 and 0.19
against measured 0.4 and 0.8, conservative on both and still not proportional.

**But it inverts the layer-0 anomaly this page flagged as "worth measuring
before it is believed".** Layer 0 reads a 0.4% share, the cheapest place in the
model to ablate, beside an effect size second only to layer 62 -- which the
page correctly called suspicious. By the quantity actually removed it is
**72.1%**, the single most expensive layer in the model, off by 180x. The
mechanism is the one the page guessed for the effect size: the layer 0
residual is essentially the token embedding, so it lies near a low-dimensional
subspace and its cosine with a direction extracted from it is large.
`scripts/extract_direction.py` prints both columns now.

**Was the gap the corpus-versus-prompt difference?** A natural explanation:
the share is computed on extraction rows chosen to separate along `d`, while
the alpha is applied to a neutral prompt that should carry less of it.
**Refuted** -- the same ratio on the two sweep prompts, which are in neither
corpus, comes out 1.0x and 1.1x of the corpus figure.

So the honest standing claim is that both columns RANK layers and neither
predicts a strength. `steering_sweep.rs` is the only instrument that answers
"what alpha is usable", and there is no offline substitute for generating and
scoring. The script says so where it used to print a ceiling.

One instrument result rode along. Gotcha 20's `distinct`-versus-crossing
disagreement fires on this direction too, and on the arm it should: `renorm`
at alpha 1.0 scores 3.8793, a comfortable 0.8x of the anchor, at **3 distinct
tokens**. The crossing calls that usable and the vocabulary collapse says it
is not. A reporting feature no test can redden on has now discriminated on two
independent directions.

### The batched path steers, and the drafter turns out to be half-steered already (2026-08-24)

`produce_batched` -- the speculative verify -- had no steering hook, so
steering and speculation were REFUSED together at open. That was not
conservatism: the verify is what COMMITS a speculative token, so a steered
sequential path beside an unsteered batched one emits a run of tokens drawn
from two models, coherent and wrong, and no losslessness check could see it
(both the run and its speculative reference would carry the same mixture).

`families/qwen/batched.rs` carries the edit now, through the SAME
`encode_steering` the per-token path calls, at the same boundary, with `rows`
set to the block instead of 1. The kernel was row-parallel from the start (one
threadgroup per row, `coeff[row]`), so a verify pays one dispatch per steered
layer, exactly as a single token does.

**THE END-TO-END GATE IS AN md5, AND IT IS STRONGER THAN IT LOOKS.** On the
real `qwen38-27b-mtp` install, greedy, 200 tokens:

| arm | md5 | accepted/round | per-position |
|---|---|---|---|
| sequential, unsteered | `a79fc953` | | |
| sequential, steered | `b16817e4` | | |
| speculative, unsteered | `a79fc953` | 1.30 | 0.80 0.62 |
| speculative, steered | **`b16817e4`** | 1.31 | 0.83 0.59 |

A speculative steered run is byte-identical to a sequential steered one, so
speculation stays lossless UNDER the edit. Had the batched path not been
steered, that arm would have matched neither row -- the drafted tokens would
have come from the unsteered model and the fallback ones from the steered one
-- so a single equality covers the whole failure mode. `qwen38-27b-dflash2`
replicates it exactly (`b16817e4` again, 1.33 -> 1.29 accepted per round).

**ACCEPTANCE BARELY MOVES, WHICH REFUTES THE PREDICTION THIS WAS BUILT
AROUND.** The expectation, written into the code comment that lifted the
refusal, was that a steered target verified against an UNSTEERED drafter would
collapse acceptance: a direction set covers trunk layers, and neither the MTP
head nor the DFlash2 drafter is a trunk layer. Measured, the change is inside
the noise of one prompt -- MTP 1.30 to 1.31, DFlash2 1.33 to 1.29, rollbacks
44 and 41 in both arms.

The mechanism is that **the drafter is already half-steered through its
INPUT.** The edit is applied at every layer's output including the last, so
`scratch.x` is steered by the time the head reads `h_t` out of it. Its WEIGHTS
are untouched and its input is not, which is evidently enough to keep it
predicting the edited model about as well as it predicted the unedited one.
That is a property of where these drafters read from rather than a general
result: a drafter taking its input from anywhere upstream of the last steered
layer would not inherit the edit this way.

**THE ORDER OF THE EDIT AND THE DRAFTER'S CAPTURE IS NOW A DECISION.** Both
paths steer FIRST and capture second, so the drafter sees the residual the
trunk actually committed. It was the other way round in the per-token path and
unobservable, because the two features could not both be on. With steering off
the two orders are the same program. `the_batched_capture_agrees_with_the_per_
token_hook_under_steering` is the guard, and flipping the batched path alone
reddens it and nothing else -- the pre-existing unsteered capture case cannot
see it, by construction.

One structural note worth not re-deriving. The coefficient buffer was one FP32
slot per LAYER, which a block of M rows overruns -- at the last layer, off the
end of the buffer entirely. It is `num_layers * MAX_STEER_ROWS` now, and
`MAX_STEER_ROWS` is `gpu::MAX_BATCH_ROWS` rather than a second 16 that happens
to agree: the batched INT4 GEMM caps the block for its own reason (a
per-thread register array), so the steering refusal is a BACKSTOP that cannot
currently fire, and if the kernel's cap ever rose the buffer would follow it
instead of silently falling short. Neither the trace nor the logits can see a
wrong stride -- later layers overwrite the spill and a short GPU overrun lands
in page slack -- so the invariant is asserted as arithmetic
(`the_coefficient_blocks_partition_the_buffer`), which is the only place it is
visible at all.

### What it costs: 1.72% of decode at every layer, and it scales with the band (2026-08-24)

Owed for five sessions, blocked on machine conditions rather than on code, and
finally taken. Interleaved rounds on the real `qwen38-27b`, greedy, 400 tokens
per arm, warmup discarded. Every arm generated 400 tokens and stopped on
`maxTokens`, so the arms did equal work.

| arm | r1 | r2 | r3 | mean | vs off |
|---|---|---|---|---|---|
| steering off | 22.376 | 22.261 | 22.232 | 22.290 | |
| `ablate`, all 64 layers | 21.937 | 21.901 | 21.883 | 21.907 | **-1.72%** |
| `ablate`, layers 20:45 (26 of 64) | 22.138 | 22.132 | 22.095 | 22.122 | **-0.75%** |
| `renorm`, all 64 layers | 21.925 | 21.909 | 21.892 | 21.909 | -1.71% |

**THE COST IS PER STEERED LAYER AND VERY NEARLY PROPORTIONAL.** 26 layers of
64 is 40.6% of the coverage and costs 43.6% of the throughput, so a band's
price is about 0.027% per layer and an operator can price any band by counting
it. That is the arithmetic expectation confirmed rather than a surprise: the
edit is one dispatch per covered layer per token, one reduction over `hidden`
against a decode step's ~810 dispatches, most of which are GEMVs over matrices
five thousand times larger.

**`renorm` IS FREE RELATIVE TO `ablate`**, 21.909 against 21.907, which is
0.01% and far inside the run-to-run spread. Its second reduction fuses into
the loop that already computes the coefficient -- one fma per element, no extra
memory traffic -- so this confirms the prediction rather than testing it.

**WHY THIS WAS MEASURABLE AT LAST, and it was not that the machine went
quiet.** `spotlightknowledged` was pegging a full core throughout, indexing the
several hundred test binaries this session had just built. The check that
licensed the run anyway is three IDENTICAL arms taken before it: 22.227 /
22.254 / 22.233, a spread of **0.12%** against an expected effect of a few
percent. Spotlight is CPU-bound and decode here is GPU-bound, which is exactly
what `crates/bench/CLAUDE.md` Gotcha 43 says survives that kind of
contamination. Measure the reference arm's spread and decide from it; four
earlier sessions declined this run on the load average alone, and the load
average was answering a different question.

One caveat in the numbers themselves: the `off` arm drifts down across the
three rounds (22.376 / 22.261 / 22.232) where the steered arms are flat, so
the PAIRED deltas are the honest reading and they run -1.96% / -1.62% /
-1.57%. Interleaving is what makes that drift harmless rather than a bias.

### This number is not a coherence score

It rises for two unrelated reasons: the edit WORKING (a steered model is
supposed to say things the unsteered one would not) and the edit doing
DAMAGE. Nothing in one value separates them. Only the shape across a sweep
does, which is why items 1 and 2 of the plan were built together and why a
single-alpha version of this would have been misleading rather than partial.

### A demonstration, not a result

A 6-prompt-vs-6-prompt ocean/mountain corpus extracts a direction with
per-layer effect sizes of 0.19 to 0.51. That exercises the pipeline end to
end and says nothing about how well the method works: 6+6 is a plumbing
fixture, and the concept pair is deliberately mild.

There are TWO such fixtures now, and the second is 8x the first on effect size
(1.09 to 4.18) -- which is what makes the pair useful rather than either one
alone. Two is still two: everything on this page is one checkpoint, one
family, and two concept pairs of six prompts each.

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
the profile is monotone over a 360x range. By a pooled-spread effect size the
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

## Prior art reviewed: OBLITERATUS (2026-08-23)

Reviewed at commit `bdcb5dd4`, README and two files, not the whole source.

**IT IS AGPL-3.0** (dual-licensed with a commercial option) and this
workspace is MIT, so **nothing may be copied or adapted from it**. Reading it
for ideas and reimplementing independently is the only route, and is what
AGENTS.md asks for with prior art the docs name. Everything below is an idea,
never a line.

### What was hoped for and is not there

`obliteratus/model_profile.py` sounds like a per-model table of known-good
settings and is not one. It is a frozen dataclass (`model`, `source`,
`total_params`, `num_layers`, `hidden_size`, `intermediate_size`,
`vocab_size`, `model_type`, `dtype`) populated entirely by deriving from
`config.json`, with GQA/MQA read off `num_key_value_heads`, MoE off
`num_local_experts` / `num_experts_per_tok`, and a search of `text_config`
for Qwen-style nested dimensions. No hardcoded value for any family.

This port already derives all of that into `ArchConfig`, including the
`text_config`-before-root rule that the multimodal wrappers force (AGENTS.md
Gotcha 55). Nothing to take.

What it does carry is a tiering by parameter count -- 3 directions and 0.30
regularization above 20B, 2 and 0.25 below 7B. Those are knobs on their SVD
weight-projection pipeline, and **the tiering axis does not transfer**: what
collapses a turn here is the stream share, and a parameter count does not
predict it. Two directions extracted from one checkpoint can differ
severalfold in that ratio. The derived ceiling above is the same idea done
per direction instead of per model.

### Their one published number is on this page's exact model

`docs/executive_research_summary.md` reports Qwen3.8-27B at MMLU 85.3% stock,
81.4% after an aggressive SVD projection, and 86.3% after a 60/40 blend, with
refusal at 0% in both edited arms.

**Do not quote it.** Their own document flags it: 570 questions of a 14k
MMLU, no raw outputs, no statistical validation, and the 0.60 blend not
established as an optimum. It also has no data on which layers mediate
refusal or on optimal ranges, and lists cross-architecture replication as
pending. It is a comparison point if weight-projection is ever built here,
and nothing today.

### Taken

- **The stream-share ratio and the derived alpha ceiling** (landed above).
  Prompted by their strength-sweep interface, which trades coherence against
  effect. The ratio form is this port's, because the mechanism was already
  measured here.
- **Coherence as a measured quantity**, and **an alpha sweep** to read it
  across. Landed together as `steering_sweep.rs`, because a single-alpha
  version of the first would have been misleading rather than partial.
- **Norm-preserving projection**, as the fourth `SteeringMode` (`renorm`).
  **Built, measured, and its headline prediction REFUTED** -- see the section
  above. Recorded here as taken rather than moved to Declined, because the
  idea was worth the cost: it was the only one on their list with a
  falsifiable claim attached, the claim was cheap to test against an
  instrument that already existed, and the answer is a real result either way.
  What it buys is one grid step of graceful degradation, not the usable full
  ablation it promised.

### Worth taking, not yet built

Nothing from this review is left in this state. The three ideas above are
built. The rest are under Declined.

### Declined

- **Telemetry**, the "community-powered research" surface. It collects model
  name, method, aggregate scores, hardware and timestamps -- explicitly not
  prompts or outputs -- on by default in their HF Space and opt-in locally.
  Reasonable for them. It cannot come here: `docs/FORGE_GUARDRAILS.md` makes
  a load-bearing "there are no outbound sockets" claim with a one-command
  verification beside it, and an opt-in flag still means shipping the socket.
- **Their MMLU-style capability check.** This repo's frozen quality gates,
  perplexity and output digests are the stronger instrument, and their own
  document calls theirs unvalidated.
- **COSMIC layer selection** (pick layers by lowest cosine similarity between
  the two representation sets). A different scale-free metric answering the
  question `separation` already answers. Not obviously better, and no reason to
  hold two.

### Noted, and it bounds what this design can do

They use up to 8 directions and a "concept cone" that distinguishes single
from multiple refusal mechanisms. A `SteeringSet` here is ONE direction per
layer. If a behaviour is mediated by several directions, a rank-1 edit
cannot remove it however the alpha is tuned, and no amount of measurement on
this page would reveal that -- every arm would read as a working but weak
steer. That is a limit of the shape, not of the tuning.

## The llama.cpp interop off-by-one (2026-08-24)

`direction.N` names llama.cpp's 0-based block `N`. This port read it as block
`N - 1`, so every vector it wrote was positioned one block early under
llama.cpp, and every foreign vector it read was applied one block early here.

**No measurement on this page moves.** The writer emitted `direction.{l+1}`
and the reader mapped back to `l`, so the round trip was self-consistent and
every frozen row was taken on the block it says it was. What was wrong is the
INTEROP claim alone, which is why this closed as a correction rather than a
re-freeze.

**BUT A REGENERATED VECTOR IS NOT THE SAME VECTOR, and anyone reproducing
from scratch has to know it.** The frozen rows were measured on files that
carry a direction for block 0; `extract_direction.py` can no longer write
one. So re-running the extraction and then the sweep is a different
experiment, not a reproduction, and the difference is not small in the place
it lands: the layer-band section above measures block 0's TRUE removed
fraction at 72.1%, the highest in the model, against the 0.4% its `share`
column reports. Reproduce a frozen row against the vector it was taken on and treat a
regenerated one as a new direction that happens to share a corpus. **Those
two vectors LIVE OUTSIDE `/tmp` since 2026-08-24**, at
`~/models/steering-vectors/{ocean,register}-legacy-layerbase0.gguf`, because
a file that cannot be regenerated has no business in a directory whose whole
job is to be wiped -- this page has twice recorded losing an artifact to a
disk cleanup, and both times the note saying it was cheap to regenerate had
gone stale before anyone read it. 1.3 MB each, and the name now carries the
convention they declare.

### Settled by reading, at no download

Three independent sources, none of them a measurement:

| source | what it says |
|---|---|
| `common.cpp`, `common_control_vector_load_one` | writes `direction.N` to buffer offset `n_embd * (N - 1)` |
| `llama-adapter.cpp` | block `il` reads offset `n_embd * (il - 1)`, looping from `il = 1` |
| `llama.h`, in a comment | the buffer "should point to an n_embd x n_layers buffer starting from layer 1" |

The first two compose to `il = N`. The third says the same thing
independently and was already on this machine, in
`/opt/homebrew/include/llama.h`. `repeng`'s own exporter is a fourth: it skips
layer 0 and names tensors `direction.{layer}`.

The APPLY SITE was already right and is unchanged. llama.cpp calls
`build_cvec` between the FFN residual add and `l_out`, which is the boundary
`families/qwen/produce.rs` uses. One axis was wrong, not two.

### Then confirmed empirically, which is what the published vector bought

`jukofyork/creative-writing-control-vectors-v3.0`,
`Meta-Llama-3-8B-Instruct/llama-3:8b-optimism_vs_nihilism__optimism.gguf`
(509 kB), read by `control_vector_file`'s foreign arm:

| | value |
|---|---|
| hidden | 4096 (Llama-3-8B's) |
| blocks | **31 covered of 32 spanned** |
| under the OLD reader | 31 covered of 31 spanned |

Llama-3-8B has 32 blocks. 31 of 32 is blocks 1..31, exactly llama.cpp's
`for il = 1; il < n_layer`. The old reader put the same bytes on blocks 0..30:
steering a block llama.cpp never steers and leaving the last block unsteered.

### The fix reads both conventions and writes one

`turbospark.layer_base` was stamped from the first commit and READ BY NOTHING
-- AGENTS.md Gotcha 45's shape exactly, a tag no reader honours. Giving it a
consumer is what let the correction land without reinterpreting the vectors
already on disk.

- `1`, or ABSENT: `direction.N` is block `N`. llama.cpp's convention, every
  foreign vector, and everything this port writes now.
- `0`: `direction.N` is block `N - 1`. Files written before the correction,
  which keep meaning what they meant when the frozen rows were measured on
  them. Verified: both legacy vectors
  (`~/models/steering-vectors/*-legacy-layerbase0.gguf`) still read 64
  covered of 64 spanned, at the same norms.
- Anything else is REFUSED rather than clamped.

**BLOCK 0 IS NO LONGER EXPRESSIBLE** in a file this port writes, and that
costs nothing anyone wants: llama.cpp never applies a direction there, and the
layer-band section above already records block 0's row as an artifact of the
`share` formula rather than a place to steer. `extract_direction.py` announces
the drop rather than making it quietly.

### The lesson, which is not "read the reference"

The module knew every relevant fact about llama.cpp and still got it wrong. It
recorded, correctly, that llama.cpp rejects `direction.0` by name and that its
apply loop never reaches block 0 -- and then INFERRED from those that the
indices must shift down by one. They do not. Block 0 goes unsteered precisely
BECAUSE the lowest direction lands on block 1.

The tell was available the whole time and was internal: under the old mapping
`direction.1` steered block 0, which contradicts the very invariant the doc
cited two sentences earlier. **A true fact about a reference is not a reading
of it.** When a convention is derived from a fact rather than from the line
that implements it, check the derivation against its own premises before
writing UNVERIFIED beside it -- the honest hedge made this look measured-open
rather than reasoned-and-wrong, and it survived five sessions on that.

## A second family, and someone else's direction (2026-08-24)

`families/llama/` steers, which makes this the second flow to carry the edit
and the first that can run a vector this port did not extract. It covers
Mixtral, `qwen3moe`, and the dense Mistral / Llama 2 / 3.x half -- both
branches of the flow, which are two call sites and not one.

### What was wired, and the one predicate that keeps it honest

`encode_steering` moved out of `families/qwen/produce.rs` and into
`steering.rs` beside the state it reads, and `encode_resid_capture` moved into
`resid_capture.rs` the same way. Neither is a tidy-up: with three call sites
and then four, N copies would have to agree on the mode, the alpha, the
direction offset, the row stride and the coefficient block, and a
disagreement in any one of them is a fluent model that is not the one asked
for.

**The family gate is now ONE predicate, `family_dispatches_steering`, read by
both the steering refusal and the capture guard.** They were two lists that
happened to agree. The edit and the capture land on the same boundary by
construction, so a family wired for one and not the other extracts a direction
from a place nothing steers, or steers where nothing was measured -- and two
lists is exactly how that drift happens. One `match`, no default arm, so a new
family answers `false` and fails loudly rather than writing a file of zeros.

**THE ENCODE HALF IS NOT THE WHOLE HOOK, and wiring only it is a mistake this
session made.** The per-layer copies fill a buffer; a separate readback after
the command buffer is waited on is what keeps a snapshot. With the copies in
and the readback missing, the first real capture printed `no non-prefill pass
ran; wrote nothing` -- loud, which is the good failure mode, and still a
family half-wired. Grep for `record_pass` when adding a family, not just for
the encode.

### The numbering correction, confirmed on a real model rather than read

The interop section above settled `direction.N = block N` by READING
llama.cpp and confirmed it against a published vector's coverage. This is the
first time a foreign vector has been APPLIED here, and the startup line reads

```
steering: ablate at alpha 0.5 over 31 of 32 layers
```

**31 of 32, block 0 unsteered**, which is exactly llama.cpp's
`for il = 1; il < n_layer`. Under the pre-correction mapping the same bytes
would have steered block 0 and left block 31 alone.

### The alpha band does NOT transfer, and the layer band DOES

Item 8 asked whether either band survives a direction extracted by someone
else's method. Measured on `mistral7b`
(`Mistral-7B-Instruct-v0.3` Q4_K_M, dense `llama`, hidden 4096, 32 layers)
with `jukofyork/creative-writing-control-vectors-v3.0`'s
`mistral-0.3:7b-honesty_vs_machiavellianism__machiavellianism.gguf` --
the vector set's own build for this exact checkpoint, so the model and the
direction match and only the METHOD is foreign.

The anchor is the unsteered model's perplexity on the frozen reference
answer, **9.3971**: that is what ordinary fluent prose costs it. Read the
column against that, never against 1.0.

| alpha | ppl | x anchor | distinct tokens |
|---|---|---|---|
| 0 | 1.3756 | 0.1x | 35 |
| 0.5 | 1.6097 | 0.2x | 34 |
| 1 | 1.7602 | 0.2x | 35 |
| 2 | 2.7324 | 0.3x | 35 |
| 4 | 40.3439 | 4.3x | 32 |
| 8 | 11.0857 | 1.2x | **3** |

**Usable to alpha 2**, against 0.3 for this port's ocean direction on
`qwen38-27b` and 0.8 for its register one. So the band is two-and-a-half to
six times wider, and the direction it moves is the one the norms predict:
this vector's per-layer norms run **0.0017 to 0.4268** where this port's own
ocean direction runs 0.0492 to 116.3376 -- a 270x span on the top end. **An
alpha read off one direction is not an operating point for another**, which is
the third independent time this page has had to say so.

(An earlier draft of this paragraph quoted 0.0052 to 2.5433 here. That is the
LLAMA-3 vector's range, carried across from the interop section, and it is the
wrong artifact -- the two are different files for different models from the
same publisher. Caught by reading the norms back off the file rather than off
the note about it, which is the only reason it is a correction and not a
published number.)

Note the ppl column FALLS from alpha 4 to 8 while the distinct-token count
collapses 32 to 3. That is degenerate repetition, which the unsteered model
finds very predictable -- the reason the harness reads the distinct column as
the collapse criterion and not the peak.

The layer band is the opposite result. Steering `all` and steering `1:25`
(dropping the last six layers) are IDENTICAL at every usable alpha -- 1.3756
and 2.7324, 35 distinct tokens, both arms:

| band | alpha 2 | alpha 4 | alpha 8 | usable |
|---|---|---|---|---|
| all (31 layers) | 2.7324 | 40.3439 | 11.0857 (3 distinct) | 2 |
| 1:25 (25 layers) | 2.7324 | 64.5058 | 27.7245 (4 distinct) | 2 |
| 8:23 (16 layers) | 2.1517 | 7.7828 | 99.9566 | 4 |

**That REPLICATES the page's late-layer refutation on a different model, a
different family and a foreign direction**, where excluding layers 51-63 of
64 on `qwen38-27b` left the greedy path byte-identical at every usable alpha.
Two of the page's claims have now split cleanly: the layer-band result is a
property of this ENGINE, the alpha ceiling is a property of a DIRECTION.

The middle-only band tolerating alpha 4 is reported and not claimed: it
steers 16 layers against 31, so a weaker edit is the ordinary explanation and
nothing here isolates the band from the count.

### What is not measured

No throughput number for this family. The machine was running a test suite
throughout, and a timing row taken under that is Gotcha 43 rather than a
measurement. The page's existing cost result (per steered layer, very nearly
proportional to the band) has no reason to be family-specific, but it has not
been checked here.

## museGlimmer, measured on a real install (2026-08-25)

`families/museglimmer/` steers and captures on a real 30B install
(`~/models/museglimmer-30b.gturbo`, hidden 6656, 52 layers), closing the
"NOT YET measured on a real install" caveat for this family. `gpt-oss`'s
install was deleted in an earlier disk cleanup at the time this section was
written; its own real-install measurement is the next section below, taken
the same day once the install was re-streamed.

### The direction: a fresh 4-pair corpus, this family's own captures

No foreign vector exists for this checkpoint (it is a niche MLX finetune), so
the direction is self-extracted: 4 matched ocean/mountain prompt pairs (the
page's own established corpus shape), captured via `TURBOSPARK_RESID_CAPTURE`
on this install and reduced with `scripts/extract_direction.py`'s
diff-of-means. Every capture is finite with no zero layer, per-layer norms
rising 89 to 600 through the 52 layers -- the same sanity shape every other
family's captures show.

| | value |
|---|---|
| effect size (`sep`), strongest / weakest | layer 2 (0.363) / layer 17 (0.127) |
| stream share, peak | layer 50, 20.8% |
| removal (`\|c_hat\|/\|\|x\|\|` at alpha 1), peak | layer 20, 20.3% |
| covered layers | 51 of 52 (block 0 dropped, the interop convention) |

### The null control -- clean

`--steering-scale 0.0` with the kernel dispatched at all 51 covered layers is
byte-identical to steering off: `diff` against the two runs' full stderr
shows only the informational `steering: ablate at alpha 0 over 51 of 52
layers` startup line and the non-deterministic wall-clock footer differing --
every token of generated text agrees. This is the load-bearing check on this
page, and it is the one this session ran first.

### Full ablation does not collapse the turn here

At `alpha = 1.0` over all 51 covered layers museGlimmer stays COHERENT, a
different outcome from qwen38-27b's documented collapse at the exact same
operating point (immediate `EndOfTurn`, no text). Wording shifts noticeably
from the unsteered continuation within the first several tokens; at
`alpha = 0.3` the shift is smaller. Neither run degenerates into repetition
or template markup. Whether this holds beyond one prompt and one corpus is
not established here -- reported as measured, not generalised.

### The probe: the null control passes, the single-position divergence check does not

```sh
TURBOSPARK_PROBE_INSTALL_DIR=~/models/museglimmer-30b.gturbo \
TURBOSPARK_STEERING_VECTOR=/tmp/steer-museglimmer/d.gguf \
  cargo test -p turbospark-bench --test steering_probe --release -- --ignored --nocapture
```

Arm 1 passes cleanly on the probe's own default prompt: 0 of 202,048 logits
differ from steering off, KL exactly 0.0, a 24-token greedy continuation
identical. Arm 3's coefficient trace reads real, sign-varying values at every
layer (0.31 to 27.87 at alpha 0.3) -- the direction is read and dispatched,
not zero.

**Arm 2's assertion fails, and it is worth separating from the null control
rather than reading as one result.** Three combinations were tried:

| prompt | alpha | KL (nats) | x the 7.4e-6 floor | argmax |
|---|---|---|---|---|
| probe default ("...the water.") | 0.3 | 1.7574e-10 | 0.00002x | 328 vs 328 |
| probe default | 1.0 | 1.5236e-9 | 0.0002x | 328 vs 328 |
| "Explain how coastal wetlands..." | 0.3 | 2.0189e-6 | 0.27x | same |

Every combination reads below the dense-family shape floor (borrowed from
`qwen3_5`'s own cross-engine measurement, since museGlimmer has no
cross-engine reference of its own -- reasonable for a dense architecture but
not independently verified for this one), and the single-position KL swings
over three orders of magnitude between the two prompts at the same alpha.
Because arm 2 panics on assertion failure, arm 4 (determinism) never ran in
any of these three combinations.

**This does not read as a broken hook.** The null control (the actual
kernel-correctness assertion on this page) is clean; the coefficient trace
shows the direction reaching every layer with real magnitude; and the CLI
runs above show the generated text visibly, coherently diverging within a
handful of tokens on both prompts, at both alphas -- the same "parts a few
tokens in" shape this page documents for other families. The most consistent
reading is that this family's very first generated token, for these prompts,
sits at an unusually confident point in the distribution, so a metric
anchored at exactly that one position under-reads an effect that is visible
once the model has generated a few tokens. `steering_sweep.rs` (which scores
a multi-token continuation rather than one position) was not run this
session and would be the next instrument to reach for.

### Regression gates: the memory oracle is clean; the quality gate's golden was stale, and NOT for the reason first guessed

```sh
TURBOSPARK_MUSEGLIMMER_INSTALL_DIR=~/models/museglimmer-30b.gturbo \
  cargo test -p turbospark-bench --test museglimmer_memory_oracle --release -- --ignored --nocapture
```

Passes: session peak 531 MiB against a 650 MiB ceiling (8,192 context),
replay growth +0.00 MiB, all three protocol cases stop `endOfTurn`, decode
17.643 / 15.805 / 14.947 tok/s against a floor of 9.

`museglimmer_quality_gate` failed against its 2026-08-15 golden, and the
failure PRE-DATES this session's diff -- confirmed by `git stash push --
crates/runtime/src/families/museglimmer/mod.rs`, since the greedy digest
reads the identical `e11b7013...` both with and without this diff's steering
hook in the file. **The first guess at a cause ("the already-committed
`d78f6b3` muse-decoder work") was WRONG and is corrected here rather than
left standing.** A full `git bisect` from the golden's own commit
(`ea77279`) to HEAD landed on every single commit reproducing the SAME new
digest -- including `ea77279` itself, checked out and rebuilt fresh: the
exact commit that wrote the frozen golden cannot reproduce its own recorded
values on this machine ten days later. That rules out every commit in
between by construction; this is environmental drift, not a code
regression. What was checked and found unchanged: the install's own files
(mtimes from 2026-08-15), macOS (26.5.2, build 25F84) and Xcode (26.6),
both matching this repo's other notes from around that period. The exact
mechanism is not established. The new values are stable -- identical across
roughly ten independent runs this session, plus the two fresh-process
agreement runs the gate itself requires -- and the current generation reads
as coherent, on-topic prose (spot-checked via a manual CLI run of the
gate's own reference prompt). Re-frozen in `museglimmer_quality_gate.rs`
with this investigation recorded in its comments; the gate is green again.

### Standard smokes

Both coherent, both `EndOfTurn`: greedy 954 new tokens at 17.973 tok/s,
sampled (CLI defaults) 860 new tokens at 18.070 tok/s. The family needs a
budget in the 900-1,600 range to reach `EndOfTurn` on an ordinary question
(per the oracle's own case data) -- the usual 400-token smoke budget is too
short for this family and stops mid-reasoning on `MaxTokens`.

## gpt-oss, measured on a real install (2026-08-25)

`families/gptoss/` steers and captures on a real 20B install
(`~/.turbospark/models/gptoss-20b.gturbo`, hidden 2880, 24 layers, MXFP4
routed experts), closing the last "NOT YET measured on a real install"
caveat for the family the previous entry left open. The install had been
deleted in an earlier disk cleanup; this machine now has 89 GiB free, so it
was re-streamed via `turbospark-model pull gptoss-20b` (~9 min this
session, 11.3 GiB).

### The pre-existing gates reproduce, unmoved

Before touching steering at all: the frozen Phase M5 gates
(2026-08-11/12), which predate this work entirely, both hold against the
fresh stream. `gptoss_memory_oracle`: session peak 5,419 MiB against the
5,700 MiB ceiling, replay +0.00 MiB, all three protocol cases `endOfTurn`.
`gptoss_quality_gate`: reference-answer perplexity **12.0801** exact,
greedy and sampled digests both matching their frozen goldens, 8-slot
digest equal to 16-slot at 0.86x. So the stream is byte-for-byte the
checkpoint this repo has measured before -- no Gotcha-24-style
investigation was needed here, unlike museGlimmer's stale golden the same
week.

### The direction: a fresh 4-pair corpus, this family's own captures

No foreign vector exists for this checkpoint either, so the direction is
self-extracted: 4 matched ocean/mountain prompt pairs (the same corpus
shape used for `qwen38-27b` and `museGlimmer`), captured via
`TURBOSPARK_RESID_CAPTURE` on this install and reduced with
`scripts/extract_direction.py`'s diff-of-means. Every capture is finite
with no zero layer, per-layer norms rising steeply through the 24 layers
(171 at layer 0 to a peak of 24,100 mid-stack, easing to 20,459 at the
last layer -- the same "not monotone at the very top" shape `qwen38-27b`'s
own capture showed).

| | value |
|---|---|
| effect size (`sep`), strongest / weakest | layer 16 (0.599) / layer 0 (0.248) |
| stream share, peak | layer 23, 4.7% |
| removal (`\|c_hat\|/\|\|x\|\|` at alpha 1), peak | layer 5, 37.4% |
| covered layers | 23 of 24 (block 0 dropped, the interop convention) |

The removal-vs-share gap at layer 5 (37.4% actually removed against a
1.4% `share` reading) is the same 2.0x-to-180x factor `docs/OBLITERATION.md`
already records for the other families' directions -- confirmed on a
fourth checkpoint rather than re-derived.

### The null control -- clean, on both instruments

`--steering-scale 0.0` with the kernel dispatched at all 23 covered layers
is byte-identical to steering off at the CLI (`diff` clean, matching md5s
over a 120-token generation). The probe's arm 1 -- the load-bearing
kernel-correctness assertion -- passes the same way: 0 of 201,088 logits
differ from steering off, KL exactly `0.000e0`, a 24-token greedy
continuation identical.

### The probe's single-position check does not clear its borrowed floor, and this time the mechanism is exact rather than guessed

```sh
TURBOSPARK_PROBE_INSTALL_DIR=~/.turbospark/models/gptoss-20b.gturbo \
TURBOSPARK_STEERING_VECTOR=/tmp/steer-gptoss/d.gguf \
  cargo test -p turbospark-bench --test steering_probe --release -- --ignored --nocapture
```

Arm 1 passes as above. Arm 2 -- the divergence check against
`steering_probe.rs`'s hardcoded `DENSE_SHAPE_FLOOR_NATS = 7.4e-6`, a
`qwen3_5` (DENSE) shape floor with no per-family branching anywhere in
that file -- fails, and does so on every combination tried:

| prompt | alpha | KL (nats) | x the 7.4e-6 floor | argmax |
|---|---|---|---|---|
| probe default ("...the water.") | 0.3 | 2.5865e-11 | 0.0000x | 200005 vs 200005 |
| probe default | 1.0 | 1.5727e-9 | 0.0002x | 200005 vs 200005 |
| "Explain how coastal wetlands..." | 0.3 | 8.2510e-12 | 0.0000x | 200005 vs 200005 |

**This is a WEAKER result than museGlimmer's, and the floor comparison is
even less appropriate here than the doc already flags it as being.**
museGlimmer is dense, so the borrowed dense floor was at least the right
kind of quantity, merely unverified for that checkpoint. `gpt-oss` is
MoE (32 experts, top-4), and per `crates/bench/CLAUDE.md` Gotcha 8 an MoE
shape floor runs orders of magnitude above a dense one (`qwen3moe`'s own
is ~0.00135). So the comparison here is not just unverified, it is very
likely the wrong order of magnitude for this architecture -- meaning even
a genuine, working edit could plausibly fail to clear this particular
floor, and clearing it would have been correspondingly weak evidence that
the edit worked. Neither direction of that asymmetry was tested against a
real MoE-specific floor this session; that instrumentation does not exist
yet and building it was out of scope here.

**Built the following session and it was NOT the cause.**
`steering_probe.rs` now resolves a real `MOE_SHAPE_FLOOR_NATS = 0.00135`
(`qwen3moe`, llama.cpp batched vs cached) whenever `ArchConfig.num_experts >
0`, rather than always checking against the dense `qwen3_5` number
(`crates/bench/CLAUDE.md` Gotcha 25). Re-run against `gpt-oss` with the fix
in place, the probe correctly prints `MoE shape floor 1.35e-3 nats` and the
steered KL still reads ~1.6e-9 nats -- `0x` even the wider, correct floor.
So the floor genuinely was wrong (a threshold off by over two orders of
magnitude, and worth fixing on its own terms for any future MoE family
whose direction is merely weak rather than measured at a pinned position),
and fixing it changes nothing about this reading: the cause is the
`<|channel|>` position below, not the threshold. The two are independent
problems and only one is closed.

**But the argmax identity across all three combinations is not
ambiguous, and it has an exact cause.** Token id 200005 of the 201,088-token
vocabulary -- identical in every row above, regardless of prompt, alpha, or
whether the edit ran at all -- decodes (checked directly against
`gptoss-20b.gturbo/tokenizer.json`'s `added_tokens`) to `<|channel|>`:
Harmony's channel-opening special token. The rendered assistant turn
always opens `<|start|>assistant<|channel|>...`, so the single token
immediately after ANY user prompt is close to certain under this
checkpoint's own chat template, independent of content. Arm 2 measures
divergence at exactly that one position, so it is reading a near-fixed
point of the FORMAT rather than of the concept the direction encodes --
a sharper, checkable version of what `docs/OBLITERATION.md` could only
guess at for museGlimmer's own single-position shortfall ("this family's
very first generated token... sits at an unusually confident point").
Unlike that guess, this one is a specific, cheaply-verified token id
rather than a hypothesis.

### CLI, over a generation: the edit is visibly real past the forced token

The coefficient trace is not in question -- `|c|` ranges 28.52 to 3646.97
across the 23 covered layers at alpha 0.3, real and sign-varying, so the
edit dispatches and moves the residual by a large amount; only the
single-position readout misses it. Generating past the forced `<|channel|>`
token settles whether that movement shows up in what the model actually
says.

At **alpha 0.3**, same seed, same 75-token prompt, unsteered and steered
reasoning-channel text diverge in wording within the first two sentences
(both open "The user says: ... 'Describe what you notice about the
water.'" and then part ways immediately: unsteered continues "There's no
context. Likely they want..." against steered's "This is ambiguous. It
could be a prompt for a creative writing exercise..."). Both runs reach
`EndOfTurn` with a coherent final answer (unsteered: "Could you tell me a
bit more about the water you're thinking of?..."; steered: "I'm not sure
which water you're referring to--whether it's a calm lake, a rushing
river..." -- different phrasing, same underlying request for
clarification, which is an ordinary outcome for a direction extracted
from an unrelated ocean/mountain concept pair rather than from anything
about ambiguous-prompt handling).

At **alpha 1.0** over all 23 layers, the turn does **not** collapse the
way `qwen38-27b`'s does at the same operating point (no immediate
end-of-turn, no immediate degenerate repeat) -- reasoning stays legible
and on-topic for several hundred tokens. It does eventually fail to
converge: past roughly 900 tokens of reasoning it settles into a four-
sentence cycle ("I'll mention how water can be both a physical and
emotional element. I'll mention how water can be both a catalyst for
life...") repeating to the 1,000-token cap without reaching a final
answer. **This is a real but DIFFERENT failure mode from `qwen38-27b`'s
documented collapse**, not the same one on a new family: rather than an
immediate stop, full-strength ablation here produces an extended, coherent-
sounding deliberation that never resolves. Reported as observed, on one
prompt; whether it is a general property of full ablation on this family
is not established.

### Regression gates: both clean, unlike museGlimmer's stale golden

Unlike the previous section, no environmental-drift investigation was
needed: both `gptoss_memory_oracle` and `gptoss_quality_gate` reproduced
their frozen 2026-08-12 goldens exactly on the first run (see above), and
neither steering commit nor this session's captures touch anything either
gate reads.

### Standard smokes

Both stop `MaxTokens` at the usual 400-token budget (greedy: coherent,
on-topic reasoning about wetland flood mitigation; sampled, CLI defaults:
the same, different wording) -- consistent with the family's own
2,153-token `medium-review` case from the memory oracle and with
museGlimmer's identical note above: 400 tokens is well under what
Harmony's reasoning-then-answer structure needs to reach `EndOfTurn` on an
ordinary question for this family.

## Open, and stated as open

- ~~**llama.cpp interop of the layer indexing is UNVERIFIED**~~ --
  **CLOSED 2026-08-24, AND THE ANSWER IT WAS CARRYING WAS WRONG.** See the
  section below: `direction.N` is llama.cpp's block `N`, this port read it as
  block `N - 1`, and the two were off by one for the life of the surface.
- ~~**The qwen family only**~~ -- **THREE FLOWS SINCE 2026-08-24**, the qwen
  one (both halves, per-token and batched), `families/llama/` (Mixtral,
  `qwen3moe`, and the dense Mistral / Llama 2 / 3.x half), and
  `families/gemma4/`. Five families of eight. The remaining three still
  disable the CAPTURE with a diagnostic and REFUSE a direction set at open by
  name -- a set that loaded, reported itself on the startup line and changed
  nothing would be the exact silent no-op this whole surface is built to
  avoid. Both gates still read ONE predicate
  (`steering::family_dispatches_steering`) rather than two lists that agreed.
  Gemma 4 WAS the awkward one and is now CLOSED (see below): it has a CHUNKED
  prefill driver as well as a per-token path, so wiring it needed three call
  sites -- `mod.rs`'s sequential decode, `prefill.rs`'s per-token routed loop,
  and `moe_batch.rs`'s batched-routed tail -- and a fourth thing neither qwen
  nor llama ever needed: `encode_steering` and `encode_resid_capture`
  hardcoded the edited row's offset at 0, because every earlier caller had
  exactly one row and it always sat there. Gemma's chunk driver packs several
  tokens into one `scratch.x` buffer at their own slot offsets
  (`token * hidden * 2`), so both functions gained an `x_off` parameter and
  every pre-existing call site was updated to pass `0` explicitly (a
  behaviour-preserving change, confirmed by `qwen38_quality_gate` and the
  gemma4 `quality_gate` reproducing their frozen perplexity and every digest
  exactly). Mutation-checked: hardcoding the new parameter back to `0` in
  either `prefill.rs`'s or `moe_batch.rs`'s hook reddens ONLY the test
  covering that call site (`real_forward_gemma4_steered.rs`'s two
  chunked-vs-sequential byte-identity cases), which is what says a caller
  copying the qwen/llama call verbatim -- steer/capture row 0 regardless of
  which token is being processed -- would have shipped a family that is
  fluent, finite, and wrong for every token past the first of a micro-batch.
- ~~**The remaining three families (gpt-oss, museGlimmer, DeepSeek-V4-Flash)
  are unwired.**~~ **TWO OF THREE LANDED**, gpt-oss and museGlimmer. Seven
  families of eight now dispatch the edit; DeepSeek-V4-Flash stays refused
  because its compressed-attention kernels are unported, so there is no
  decode flow at all for a hook to sit in -- not a wiring gap, a missing
  flow. Both new families have exactly ONE call site each (neither has a
  dense/MoE split the way `llama` does, nor a chunked-prefill driver the way
  Gemma 4 does): gpt-oss's sits right after `encode_gpt_oss_layer_moe`'s raw
  residual add, on the post-mid-layer-commit "routed cb" pass (the router's
  host-side top-k forces the same kind of commit `families/llama/`'s MoE
  branch needs); museGlimmer's sits after the FFN-half sandwich tail's
  residual add, and since that family has no router at all the whole token
  runs on one pass, so there is no routed-vs-cb1 boundary to get wrong the
  way Gemma's and gpt-oss's have. Two new regression files
  (`real_forward_gptoss_steered.rs`, `real_forward_museglimmer_steered.rs`)
  follow the shape `real_forward_llama_steered.rs` set: the null control at
  alpha 0, a full-strength dispatch, and a one-layer-isolation case, on
  SYNTHETIC fixtures only. Mutation-checked: deleting either family's hook
  reddens exactly its own two tests and nothing else in the suite.
  ~~**NOT YET MEASURED ON A REAL INSTALL**~~ -- **BOTH CLOSED, museGlimmer
  2026-08-25 and gpt-oss the same day** (see "museGlimmer, measured on a
  real install" and "gpt-oss, measured on a real install" above). Both:
  a real self-extracted direction, a clean null control on both instruments,
  coherent generation at a moderate alpha, and a clean memory oracle. Both
  also shared the probe's one honestly-reported shortfall -- the
  single-position divergence check did not clear its borrowed floor at the
  prompts tried, despite real coefficients and CLI-visible divergence over a
  generation -- and `gpt-oss`'s write-up narrowed WHY: the position that
  check measured decodes to Harmony's `<|channel|>` token, near-fixed by the
  chat template regardless of content, so the check was reading the format's
  own near-determinism rather than the direction's absence. Full ablation on
  `gpt-oss` also does NOT collapse the way `qwen38-27b`'s does at alpha 1 --
  it fails a different way, settling into an unresolved reasoning loop past
  ~900 tokens rather than stopping immediately.
  ~~**The probe measures divergence at exactly one position, which a chat
  template can pin -- fixing that is a bigger instrument change than a
  floor constant, and was left open.**~~ **CLOSED.** `steering_probe.rs`'s
  arm 2 no longer relies on the single prompt-final position at all: it
  teacher-forces the unsteered engine's own greedy continuation through the
  steered engine, position by position, and asserts on the MAXIMUM KL over
  that whole window rather than on one entry. A template-pinned position
  (Harmony's `<|channel|>` included) still contributes a near-zero entry --
  the fix does not detect pinning, it refuses to trust any single entry, so
  the position where real content diverges is what the max finds instead,
  with no per-dialect branch anywhere in the new code
  (`crates/bench/CLAUDE.md` Gotcha 25). Both the prompt-final KL and the
  window's max (with its position and the decoded argmax token on both
  sides) are printed, so a template-pinned entry stays visible for
  diagnosis even though it no longer decides the verdict alone. Backward-safe
  by construction: the window contains the old single position, so a
  checkpoint that already passed cannot newly fail.

  **CONFIRMED ON BOTH REAL INSTALLS THE SAME DAY, WITH A FRESH SELF-EXTRACTED
  DIRECTION ON EACH.** No published vector exists for either checkpoint, so
  a new 4-pair ocean/mountain corpus was captured and reduced per family
  (the same shape as every other direction on this page), rather than
  reusing `qwen38-27b`'s vectors, which do not fit either model's hidden
  size.

  `museGlimmer` (`~/models/museglimmer-30b.gturbo`, alpha 0.3, 51 covered
  layers): prompt-final KL `8.3235e-10` nats, `0x` the `7.4e-6` dense floor
  -- the exact false negative this fix targets. Window max: `8.4657e-2`
  nats, **11440x the floor**, at window position 21 (prompt token index
  85), argmax `" provided"` on both sides (the steered continuation departs
  from the unsteered one earlier in the window than the position where the
  max KL lands, since divergence in probability mass need not coincide with
  a change in the argmax). Arm 2 now PASSES.

  `gpt-oss` (`~/.turbospark/models/gptoss-20b.gturbo`, alpha 0.3, 23
  covered layers): prompt-final argmax decodes to `<|channel|>` on both
  engines, exactly as the mechanism above predicts, KL `1.6498e-12` nats,
  `0x` the `1.4e-3` MoE floor. Window max: `4.4824e-2` nats, **33x the
  floor**, at window position 23, argmax `" want"` on both sides. Arm 2 now
  PASSES, and the generated text confirms it is real content diverging, not
  noise: unsteered continues `"...Likely they want a"`, steered
  `"...This is ambiguous. We need context. The"`.

  A second, smaller bug surfaced verifying this: two of arm 2's own
  diagnostic `println!` arguments were swapped on the first pass (the
  `Nx the floor` ratio and the decoded window position printed in each
  other's slots) -- caught by the printed numbers not matching their own
  labels (11440x showing up next to "prompt token index" instead of next
  to "x the floor"), fixed, and confirmed on both installs above. The
  assertion itself was unaffected throughout, since it compares the
  unformatted `f64`s directly; only the human-readable line was wrong.
- ~~**No Llama-3 chat dialect, so no Llama-3 checkpoint runs here at all.**~~
  **LANDED 2026-08-24, `ChatDialect::Llama3`** (`crates/tokenizer/CLAUDE.md`
  Gotcha 10). `detect_dialect` used to fall through to Gemma for a table
  carrying `<|begin_of_text|>` / `<|start_header_id|>` / `<|eot_id|>` and
  nothing else, and `resolve_gemma` then failed on a missing `<pad>` --
  unrelated to steering and found by walking into it. Fixed by keying
  detection on this family's OWN frame markers (`<|start_header_id|>` /
  `<|eot_id|>`) rather than the `<|begin_of_text|>` / `<|end_of_text|>` pair it
  shares with `muse_glimmer` (Gotcha 6's trap on a third pair), and by giving
  it a fallback renderer (`chat_template/llama3.rs`) on Mistral's reasoning
  rather than Harmony's refusal -- the format is a handful of markers around
  the content, not a large second implementation of something complex.

  **CONFIRMED ON THE REAL, GATED CHECKPOINT, NOT JUST A FIXTURE.**
  `bartowski/Meta-Llama-3-8B-Instruct-GGUF` (Q4_K_M, 4.6 GiB) streamed and
  loaded; both real-model smokes (greedy and sampled) are coherent and stop
  `EndOfTurn`. **ONE THING TO KNOW BEFORE TRUSTING A `probe` RUN AGAINST A
  GATED SIDECAR REPO WITHOUT `HF_TOKEN` SET: it silently reports `template
  NONE FOUND`** where the authenticated fetch reports
  `tokenizer_config.json:chat_template` -- the unauthenticated GET 401s and
  the probe treats a failed sidecar fetch the same as an absent one, which
  reads exactly like "this checkpoint ships no template" and would have been
  a false claim in this file had it not been re-checked with a token before
  writing it down. And running the vector Phase 8 fetched (rather than
  Mistral-7B-v0.3's, which needed no new dialect and only ever stood in for
  it) closes that Phase's other open thread too:
  `steering: ablate at alpha 1 over 31 of 32 layers` on the real install,
  matching `docs/OBLITERATION.md`'s and `control_vector_file`'s own
  `31 covered of 32 spanned` exactly.
- ~~**No throughput number**~~ -- **MEASURED 2026-08-24**, see the section
  above: -1.72% of decode with all 64 layers steered, -0.75% at 26, and
  `renorm` free relative to `ablate`. The cost is per steered layer and very
  nearly proportional to the band, so `--steering-layers` prices itself. Four
  sessions declined the run on the machine's load average; what settled it was
  measuring the REFERENCE arm's spread instead (0.12% across three identical
  arms, with a core pegged by Spotlight the whole time), because the
  contention was CPU-bound and decode here is not.

- ~~**The batched path does not steer**~~ -- **CLOSED 2026-08-24**, see the
  section above. It carries the edit through the same `encode_steering` the
  per-token path calls, the refusal is lifted, and a speculative steered run
  is byte-identical to a sequential steered one on both drafters.
- ~~**The chunked-prefill driver does not steer**~~ -- **CLOSED 2026-08-24**,
  same day Gemma 4 was wired (see above). `prefill_chunk_real_gemma4` is a
  DIFFERENT function from the batched-verify path the item above closes --
  the two used to be named together in older notes on the strength of both
  being "batched", and they are not the same code: one is speculative
  VERIFY's M-row forward on the qwen flow, the other is Gemma's OWN
  micro-batched prefill driver, unrelated to speculation. Its per-token
  routed loop and its batched-routed tail (`TURBOSPARK_ROUTED_BATCH`) are two
  further call sites beyond Gemma's sequential decode, both proven to steer
  the RIGHT token's row rather than always row 0 by mutation-checked
  byte-identity against a steered sequential run
  (`real_forward_gemma4_steered.rs`).
- **No integration test for the capture**, matching how `ffn_hist` and
  `router_hist` are treated: an env-gated capture needs a process-global
  write, which races other tests in the same binary. The real-model A/B is
  the gate.
- **No alpha cap.** `ablate` cannot overflow (it only removes a component of
  `x`), but `add` and `clamp` can push an FP16 stream past 65,504, arriving as
  `inf` and then NaN -- which reads as a PERFECT score on any rank instrument
  (Gotcha 59). A numeric cap was declined as a fabricated threshold (Gotcha
  38's rule). The finiteness assertion belongs to Phase 3's probe, at the
  point a measurement is taken.
## Next, in order

Ranked by value per cost.

1. ~~Coherence as a measured quantity~~ and ~~an alpha sweep~~ -- **LANDED**
   as `steering_sweep.rs`.
2. ~~Norm-preserving projection~~ -- **LANDED as `renorm`, and its prediction
   REFUTED.** See above. It buys graceful degradation through the knee rather
   than a usable full ablation.
3. ~~A layer-band sweep~~ -- **LANDED as the sweep's second axis, and that
   prediction refuted too.** Excluding layers 51-63 leaves the greedy path
   byte-identical at every usable alpha.
4. ~~A throughput number~~ -- **MEASURED 2026-08-24.** -1.72% of decode at
   all 64 layers, -0.75% at 26, and `renorm` free relative to `ablate`; the
   cost is per steered layer and nearly proportional to the band. Four
   sessions declined it on the machine's load average, and the thing that
   settled it was measuring the reference arm's SPREAD instead (0.12% across
   three identical arms, with a core pegged by Spotlight throughout) -- the
   contention was CPU-bound and this workload is not.
5. ~~A second direction and a second prompt~~ -- **LANDED, and it split the
   page's claims in two.** Both sweep refutations REPLICATE on a register
   direction 8x stronger than the first, so those are facts about this
   engine. The derived alpha ceiling does NOT: it reads 0.09 against a
   measured 0.8, with the relationship inverted. The band is a property of
   the direction and not of the prompt (a 2x2 says so).
6. ~~The batched path~~ -- **LANDED.** Steering and speculation are no longer
   mutually exclusive: a speculative steered run is byte-identical to a
   sequential steered one on both drafters, and acceptance barely moves,
   which refutes the prediction the work was built around.
7. ~~llama.cpp interop~~ -- **LANDED 2026-08-24**, and it found a real
   off-by-one rather than confirming the mapping. See the section below.
8. ~~A THIRD direction, and a direction someone else extracted~~ --
   **LANDED 2026-08-24, and it split the page's two band claims in two.** See
   the section above. `families/llama/` steers, `mistral7b` is installed, and
   a `jukofyork` vector built for that exact checkpoint runs on it. The ALPHA
   band does not transfer (usable to 2 here against 0.3 and 0.8 for this
   port's own directions, in the direction the 46x norm difference predicts);
   the LAYER band does (dropping the last six of 32 layers is a no-op at every
   usable alpha, exactly as dropping 51-63 of 64 was). So the layer result is
   a property of this engine and the alpha ceiling is a property of a
   direction. ~~The Llama-3-8B vector item 7 downloaded is still unapplied,
   now for a smaller reason: no Llama-3 chat dialect (see Open).~~ **APPLIED
   2026-08-24**, once the dialect landed -- see Open's now-closed Llama-3
   dialect item for the run and its `31 of 32 layers` result.
9. ~~A fifth family, and the chunked prefill driver with it~~ -- **LANDED
   2026-08-24.** `families/gemma4/` steers on all three call sites its flow
   needs (sequential decode, the chunk driver's per-token routed loop, and
   its batched-routed tail), retiring the last "steering does not reach the
   chunked path" caveat. The item turned out to be more than a third call
   site: `encode_steering` and `encode_resid_capture` had hardcoded the
   edited row at offset 0 in `scratch.x` since the qwen flow, because every
   caller before this one had exactly one row and it always sat there.
   Gemma's chunk driver packs several prompt tokens into `scratch.x` at once,
   each at its own slot offset, so both functions gained an `x_off`
   parameter. Every existing call site (qwen's two, the batched verify's one,
   llama's two) was updated to pass `0` explicitly -- a change proven to move
   no bytes by `qwen38_quality_gate` reproducing its frozen perplexity and
   both digests exactly. The new threading is proven necessary rather than
   merely plausible: reverting either of Gemma's two chunked-path calls back
   to a hardcoded 0 reddens exactly the test covering that call site and
   nothing else (`real_forward_gemma4_steered.rs`), which is what a caller
   that copied the qwen/llama call verbatim -- silently steering token 0's
   row for every token in a micro-batch -- would have shipped: fluent,
   finite, and wrong past the first token of every chunk.

## Reproducing

```sh
# 1. Capture, one run per prompt. `--max-new 1` is enough; the hook keys on
#    the prefill/decode transition, so a larger budget captures the same row.
TURBOSPARK_RESID_CAPTURE=/tmp/steer/pos/p1.json \
  ./target/release/turbospark-check --model ~/models/qwen38-27b.gturbo \
  --messages-file /tmp/prompt.json --max-new 1 --temperature 0.0001 --top-k 1
```

```sh
# 2. Extract. Reads a DIRECTORY of captures per set and prints FOUR columns
#    that rank layers differently on purpose: `sep` (effect size) is the one
#    to read when picking layers, `share` (||d||/||x||) and `removed`
#    (|c_hat|/||x||) are what ablating COSTS -- the second being the fraction
#    the kernel actually subtracts, which differs from the first by 2.0x at
#    depth and 180x at layer 0 -- and `norm` is raw and comparable across
#    layers only by accident. The derived ceiling it ends with is a LAYER
#    RANKING diagnostic and NOT a usable alpha: measured on two directions it
#    under-called the band by 1.1x and 8.9x with the relationship inverted,
#    so run the sweep below for that.
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
# The same null control as an ASSERTION, plus the divergence, the
# coefficient trace and a determinism check -- four arms, three opens, ~30 s.
# It also prints both continuations, which is what separates a steered model
# from a destroyed one; no number in it can.
#
# TURBOSPARK_STEERING_SCALE defaults to 0.3 and NOT 1.0, because ablating at
# full strength over every layer collapses the turn and a collapsed turn
# scores an enormous divergence. Set it to 1.0 to watch the collapse
# detector fire.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
TURBOSPARK_STEERING_VECTOR=/tmp/steer/d.gguf \
  cargo test -p turbospark-bench --test steering_probe --release -- --ignored --nocapture
```

```sh
# The alpha sweep: which strength is usable on this direction. Six opens,
# ~2 min. Generates under a steered engine and scores under the unsteered
# one, so it needs no second checkpoint.
#
# READ THE CURVE, NEVER ONE ROW: the number rises both when the edit works
# and when it does damage, and only the shape separates them. The anchor
# (the unsteered model on the frozen reference answer) is what fluent prose
# costs, and the usable band is the last arm below it.
# TURBOSPARK_STEERING_ALPHAS overrides the sweep; it must start at 0.
#
# TURBOSPARK_STEERING_PROMPT sets the text every arm generates from, on BOTH
# this target and `steering_probe`. It defaults to the ocean-adjacent string
# both were written against, so every frozen row on this page reproduces with
# it unset -- and a SECOND direction needs it, because a vector extracted from
# some other concept pair has no reason to move that text. The resolved prompt
# is printed, so a row cannot silently be a different one.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
TURBOSPARK_STEERING_VECTOR=/tmp/steer/d.gguf \
TURBOSPARK_STEERING_PROMPT="Explain why the sky appears blue." \
  cargo test -p turbospark-bench --test steering_sweep --release -- --ignored --nocapture
```

```sh
# The same sweep across the other two axes. TURBOSPARK_STEERING_MODE points
# it at a different edit WITHOUT rewriting the vector, which is what keeps a
# mode comparison single-variable; an unknown spelling is refused rather than
# falling back to the file's. TURBOSPARK_STEERING_BANDS sweeps the layer band
# beside alpha (`all` or START:END, inclusive and 0-based, the spelling
# `--steering-layers` takes); a band covering zero layers is refused, because
# it would steer nothing and read as usable at every strength.
#
# READ THE `distinct` COLUMN BESIDE THE VERDICT. The anchor crossing
# UNDER-CALLS damage in the knee -- measured -- and the run prints a warning
# when the vocabulary collapses at or below its own crossing.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
TURBOSPARK_STEERING_VECTOR=/tmp/steer/d.gguf \
TURBOSPARK_STEERING_MODE=renorm \
TURBOSPARK_STEERING_BANDS=all,0:50 \
TURBOSPARK_STEERING_ALPHAS=0,0.4,0.45,0.5,0.55,0.6 \
  cargo test -p turbospark-bench --test steering_sweep --release -- --ignored --nocapture
```

```sh
# Any control vector on disk, including a published repeng one, reported
# without loading a model.
TURBOSPARK_CONTROL_VECTOR=/tmp/steer/d.gguf \
  cargo test -p turbospark-repack --test control_vector_file -- --ignored --nocapture

# The INTEROP arm, and a different variable: it asserts WHERE a foreign
# vector's directions land rather than that the file parses. It REFUSES a
# file carrying `turbospark.layer_base`, i.e. one this port wrote -- such a
# file is read under whichever convention it declares and so cannot say
# anything about the ecosystem's. Point it at a published vector:
#   hf download jukofyork/creative-writing-control-vectors-v3.0 \
#     "Meta-Llama-3-8B-Instruct/llama-3:8b-optimism_vs_nihilism__optimism.gguf" \
#     --local-dir ~/models/steering-vectors
# Expect `31 covered of 32 spanned`: Llama-3-8B's 32 blocks less the one
# llama.cpp cannot reach. 509 kB, no model load, no GPU.
TURBOSPARK_FOREIGN_CONTROL_VECTOR=~/models/steering-vectors/llama-3:8b-....gguf \
  cargo test -p turbospark-repack --test control_vector_file -- --ignored --nocapture
```

### Running someone else's vector (the 2026-08-24 section)

The vector must be built for the checkpoint it steers. `jukofyork`'s set
covers 70-odd base models and two of them are already catalog rows here,
`Mistral-7B-Instruct-v0.3` and `Mixtral-8x7B-Instruct-v0.1`; picking one of
those is what makes the METHOD the only foreign variable. A vector for a
model this port cannot run is not a substitute -- shapes matching is not the
same as bases matching, and `SteeringSet::validate` checks only the shape.

```sh
# 4.1 GiB, dense `llama`, hidden 4096, 32 layers. ~6 min.
cargo run --release -p turbospark-cli --bin turbospark-model -- pull mistral7b

hf download jukofyork/creative-writing-control-vectors-v3.0 \
  "Mistral-7B-Instruct-v0.3/mistral-0.3:7b-honesty_vs_machiavellianism__machiavellianism.gguf" \
  --local-dir ~/models/steering-vectors

# Expect `steering: ablate at alpha 1 over 31 of 32 layers` on stderr:
# 31 of 32 is llama.cpp's own apply range, with block 0 unsteered.
./target/release/turbospark-check --model mistral7b --messages-file /tmp/p.json \
  --max-new 160 --seed 1 --temperature 0.0001 --top-k 1 \
  --steering ~/models/steering-vectors/honesty_vs_machiavellianism__machiavellianism.gguf \
  --steering-scale 1.0

# The band tables above. ~1 min per band; the alphas and bands are the axes.
TURBOSPARK_PROBE_INSTALL_DIR=~/.turbospark/models/mistral7b.gturbo \
TURBOSPARK_STEERING_VECTOR=~/models/steering-vectors/honesty_vs_machiavellianism__machiavellianism.gguf \
TURBOSPARK_STEERING_ALPHAS=0.0,0.5,1.0,2.0,4.0,8.0 \
TURBOSPARK_STEERING_BANDS=all,1:25,8:23 \
  cargo test -p turbospark-bench --test steering_sweep --release -- --ignored --nocapture

# The capture works on this family too, so a direction can be EXTRACTED from
# it rather than only applied. Writes a 32 x 4096 snapshot at the last prompt
# token; per-layer norms should rise through the stack (0.19 to 30.47 here)
# and a file of zeros means the family is not really feeding it.
TURBOSPARK_RESID_CAPTURE=/tmp/steer-llama/cap.json \
  ./target/release/turbospark-check --model mistral7b \
  --messages-file /tmp/p.json --max-new 1 --temperature 0.0001 --top-k 1
```

### Gemma 4, and its chunked prefill driver (2026-08-24)

Gemma 4 is the family whose steering ALSO has to reach a prompt of more than
one token: `--messages-file` and `--chat` route a longer prompt through
`prefill_chunk_real_gemma4` rather than one token at a time, and that driver
is a different function from the sequential decode path every other example
on this page exercises. The `real_forward_gemma4_steered.rs` test proves this
by a byte-identity check rather than by eye (a steered chunked run must match
a steered sequential run token for token); the command below is the same
property read off a real generation.

```sh
# 128 experts, INT4 shared MLP, so both the per-token routed loop and
# TURBOSPARK_ROUTED_BATCH's batched-routed tail are reachable on this install.
./target/release/turbospark-check --model gemma4 \
  --messages-file /tmp/p.json --max-new 200 --seed 1 --temperature 0.0001 --top-k 1 \
  --steering /tmp/steer/d.gguf --steering-mode ablate --steering-scale 1.0

# The same prompt through the batched-routed prefill seam, which only a
# multi-token prompt (not a one-token capture probe) reaches:
TURBOSPARK_PREFILL_CHUNK=128 TURBOSPARK_ROUTED_BATCH=1 \
  ./target/release/turbospark-check --model gemma4 \
  --messages-file /tmp/p.json --max-new 200 --seed 1 --temperature 0.0001 --top-k 1 \
  --steering /tmp/steer/d.gguf --steering-mode ablate --steering-scale 1.0

# Both must produce IDENTICAL output to each other and to the sequential
# per-token routed path (TURBOSPARK_ROUTED_BATCH unset) at the same alpha --
# grouping tokens into a command buffer, batched or not, must not change
# which model answered.
```

The server takes the same flags, resolved once at startup. Unlike
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
  weight-projection half is what ROADMAP item 9 scoped. Its steering-vector
  half is what this page builds. **AGPL-3.0, so read-only for an MIT
  workspace**. What was reviewed, taken and declined is recorded above rather
  than left to be re-derived.
