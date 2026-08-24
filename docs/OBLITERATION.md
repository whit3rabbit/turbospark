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
| 3 | the A/B probe, coefficient trace, KL against unsteered | **LANDED**, four arms green |
| 4 | the alpha sweep, the `renorm` mode, the layer-band axis | **LANDED**; both of the sweep's predictions REFUTED, which is the result |

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
past). Arms 1, 2 and 4 reproduce to the digit, so nothing regressed -- the
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
variation rather than a discrepancy. Both are fine; they are different
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
a tiny within-set spread, and `sep` divides by that. Worth measuring before
it is believed; llama.cpp never applies a direction at layer 0 anyway.

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
apart. That is the derived ceiling validated rather than merely plausible.

At 0.6 the model emits template markup (`assistant\n<think>\n\n</think>`
repeating) at 13 distinct tokens; at 0.8 it says `" contains"` and stops.

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
the collapse. And at alpha 1.0 `renorm` scores WORSE than `ablate` (8927
against 280); both are word salad there, and ordering broken outputs by
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
  effect; the ratio form is this port's, because the mechanism was already
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
built; the rest are under Declined.

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
  question `separation` already answers. Not obviously better; no reason to
  hold two.

### Noted, and it bounds what this design can do

They use up to 8 directions and a "concept cone" that distinguishes single
from multiple refusal mechanisms. A `SteeringSet` here is ONE direction per
layer. If a behaviour is mediated by several directions, a rank-1 edit
cannot remove it however the alpha is tuned, and no amount of measurement on
this page would reveal that -- every arm would read as a working but weak
steer. That is a limit of the shape, not of the tuning.

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
- **No throughput number, and it is blocked on MACHINE CONDITIONS rather
  than on code.** The kernel adds one dispatch per steered layer per token --
  64 on this model against a decode step's ~810 -- and the bytes are
  negligible against a projection's weight read. That is an arithmetic
  expectation and NOT a measurement; nothing has timed it. `renorm` adds a
  second reduction inside that same dispatch, which is one fma per element
  and no extra memory traffic, so the expectation does not change for it.
  Three sessions have now declined to take it, each for the same two reasons
  and both worth stating so the next one does not read it as neglect: the
  expected effect is a few percent, Gotcha 43 says a loaded machine cannot
  see it, and Gotcha 22 says an effect that size cannot be measured on
  battery AT ALL. Everything else on this page is deterministic -- greedy
  generation and teacher-forced NLL -- which is exactly why the rest could be
  measured on a machine that could not support this one. The layer band is
  the lever if it turns out to matter.
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
## Next, in order

Ranked by value per cost.

1. ~~Coherence as a measured quantity~~ and ~~an alpha sweep~~ -- **LANDED**
   as `steering_sweep.rs`.
2. ~~Norm-preserving projection~~ -- **LANDED as `renorm`, and its prediction
   REFUTED.** See above; it buys graceful degradation through the knee rather
   than a usable full ablation.
3. ~~A layer-band sweep~~ -- **LANDED as the sweep's second axis, and that
   prediction refuted too.** Excluding layers 51-63 leaves the greedy path
   byte-identical at every usable alpha.
4. **A throughput number**, on a quiet machine, with and without the edit at
   a fixed layer band. **Now the only Phase 3 deliverable outstanding**, and
   it has been an arithmetic expectation for three sessions. It is blocked on
   MACHINE CONDITIONS rather than on code, which is worth stating plainly so
   the next session does not mistake it for work: it needs a quiet machine
   (Gotcha 43) and AC power, because the expected effect -- 64 extra
   dispatches against a decode step's ~810 -- is a few percent, and Gotcha 22
   says an effect that size cannot be measured on battery at all. Every other
   number on this page is deterministic and so was measurable without either.
5. **A second direction and a second prompt**, promoted from last. Every
   number here is one 6+6 corpus on one prompt, and TWO predictions have now
   been refuted on it -- which is either a fact about steering on this engine
   or a fact about this direction, and nothing here can tell those apart. It
   is the cheapest way to find out which, and it is now the main threat to
   the page's conclusions.
6. **The batched path**, which is the largest structural gap: until it
   carries the edit, steering and speculation stay mutually exclusive.
7. **llama.cpp interop**, one `#[ignore]`d test against a published `repeng`
   vector. Cheap, and it is the only open item that could invalidate files
   already written by this port.

## Reproducing

```sh
# 1. Capture, one run per prompt. `--max-new 1` is enough; the hook keys on
#    the prefill/decode transition, so a larger budget captures the same row.
MFERENCE_RESID_CAPTURE=/tmp/steer/pos/p1.json \
  ./target/release/turbospark-check --model ~/models/qwen38-27b.gturbo \
  --messages-file /tmp/prompt.json --max-new 1 --temperature 0.0001 --top-k 1
```

```sh
# 2. Extract. Reads a DIRECTORY of captures per set and prints THREE columns
#    that rank layers differently on purpose: `sep` (effect size) is the one
#    to read when picking layers, `share` (||d||/||x||) is what predicts the
#    collapse, and `norm` is raw and comparable across layers only by
#    accident. It ends with a suggested ablate ceiling; `--alpha-budget`
#    moves the 10% that ceiling is derived from.
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
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
TURBOSPARK_STEERING_VECTOR=/tmp/steer/d.gguf \
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
  half is what this page builds. **AGPL-3.0, so read-only for an MIT
  workspace**; what was reviewed, taken and declined is recorded above rather
  than left to be re-derived.
