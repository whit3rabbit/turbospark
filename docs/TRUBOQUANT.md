# TurboQuant KV-Cache Quantization

**Status (2026-09-06): built and shipped as `--kv-bits off|2|3|3.5|4`,
default OFF.** This page used to be titled "Assessed, Not Adopted" and
conclude that KV quantization did not pay on this engine. That verdict is
reversed -- see "The reversal" below for what changed and why the original
reasoning (kept intact further down) was answering a different question
than the one that mattered. The short version: "does this help today" was
the wrong test for an opt-in flag that costs every other caller nothing.

The rest of this page is in two parts. **The reversal and what was built**
describes the shipped feature: the codec, the kernels, the per-layer
policy, the flag surface, and what is still owed. **The original
assessment (2026-08-15)** is kept below it unedited except for this
banner, because it is the record of why the decision needed reversing
rather than defaulting to on, and its negative findings (the QJL residual,
the sub-4-bit quality cliff, the head-dimension sensitivity) are still
true and still worth citing on their own terms.

## The reversal

The 2026-08-15 assessment read `sharpner/turboquant-mlx`
(https://github.com/sharpner/turboquant-mlx), a proof of concept of
Google's 2025 TurboQuant paper, against this port's own memory and
throughput profile and concluded: KV is not the dominant memory term for
any family near a ceiling, attention is a single-digit share of decode at
this port's 4-8K contexts, and the one thing that makes turboquant-mlx
fast (MLX's fused `quantized_matmul`) is exactly the kernel this port
would have had to write itself with no equivalent to borrow. All true, and
all beside the point once the feature is opt-in: those are arguments for
defaulting `--kv-bits` OFF, which it is, not for refusing to build it. A
caller who wants a smaller KV cache at a stated quality cost -- a longer
context on a memory-constrained machine, or a deliberate memory/quality
trade this repo has no standing to make on someone else's behalf -- had no
way to ask for one before this landed.

**What was built goes further than the assessment's own recommendation.**
Item 2 under "Ideas worth taking" said: if this is ever built, 4-bit
affine is the only point worth building, skip fractional rates, skip
Lloyd-Max, skip 2-bit. What actually shipped is mlx-vlm's real
`_TurboQuantMSECodec` (`mlx_vlm/turboquant.py`) ported whole --
`crates/compute::kv_quant` is a portable CPU reference for it, checked
bit-identical against mlx-vlm's own printed sign vectors and packed words
(`crates/compute/tests/kv_quant.rs`) -- which is the paper-correct
Lloyd-Max path (their "V3"), not the affine one ("V2") the assessment
called safe. Per K or V row: divide by its own norm (stored separately),
apply a Randomized Hadamard Transform with a FIXED +/-1 sign vector (one
per K or V for the whole model, mlx-vlm's own `KEY_SEED`/`VALUE_SEED`
defaults of 0/1 -- not baked into the Q/K projections at repack time the
way the assessment's "rotation stage" predicted, so no `crates/repack` or
install-format change was needed at all), then quantize each rotated
coordinate against a 1-D Lloyd-Max codebook fit to the sphere-marginal
density a coordinate of a random rotated unit vector in `dim` dimensions
follows. Widths are `off`, `2`, `3`, `3.5` (K3/V4, mlx-vlm's own split for
its one fractional rate), and `4`.

**And the hard part the assessment named -- a real fused dequant inside
the attention kernel -- exists now.** `crates/gpu/src/shaders/attention_tq.metal`
reads packed rows directly in the QK^T and AV passes;
`crates/gpu/src/shaders/kv_quantize_tq.metal` is the write-side quantizer.
Both are parity-tested against the CPU reference
(`crates/gpu/tests/attention_tq_parity.rs`,
`crates/gpu/tests/kv_quantize_parity.rs`).

**Layer eligibility is mlx-vlm's own policy, not this port's invention.**
`crates/model-io::kv_quant::layer_is_quantized` mirrors
`should_quantize_kv_layer` exactly: every layer quantizes when the stack
is two layers deep or fewer, otherwise every layer but the last does
(mlx-vlm's own measurement that the last full-attention layer is
quantization-sensitive on Gemma-class models). Sliding-window and linear
(gated-DeltaNet) layers are never candidates regardless of position --
callers gate on the mask value first. A head_dim outside `32..=512` or not
a power of two REFUSES `--kv-bits` at open by name (`rht_supported`)
rather than silently falling back to a slower path nothing here
implements; mlx-vlm's own dense-rotation-matrix fallback for other widths
was deliberately not ported.

**Wired into every family and the whole stack, off by default
everywhere.** `families/{gemma4,qwen,qwen4,llama,gptoss,museglimmer}/attn.rs`
all dispatch the quantized path when `--kv-bits` is on (`qwen` covers both
the dense and MoE halves of that architecture). The flag threads through
`crates/invocation` (`KvBits`, the 5-place rule), `crates/cli`
(`turbospark-check --kv-bits`), `crates/server` (same flag, resolved once
at process open like `--speculative` and `--steering`), `crates/ffi`
(`kvBits` wire option, `SessionInfo.kvBits` reporting what was resolved),
and the Swift binding (`OpenOptions.KvBits`). `crates/bench` reaches it
ONLY through a separate, more-parameterized opener
(`open_model_runner_for_protocol_speculative_kv_quant`) that the plain
protocol opener calls with `KvQuant::Off` explicitly -- the same
structural guard `DraftPolicies` and `SteeringPolicy` already use so a new
axis cannot reach a frozen memory-oracle or quality-gate row by accident
(AGENTS.md Gotcha 35, `crates/bench/CLAUDE.md` Gotchas 4-5). Every family that existed when this
landed gained a `real_forward_<family>_kv_quant.rs` fixture-level suite.
Spark landed later and initially lacked that coverage. The P0 follow-up
below adds its seventh suite and real-install reading through the same
shared `kv_write` seam; 9 of its 36 layers are full attention.

**RUN 2026-09-07** on the real `~/models/qwen38-27b.gturbo` install (M4
Max, AC), `TURBOSPARK_KV_QUANT_INSTALL_DIR=~/models/qwen38-27b.gturbo
cargo test -p turbospark-bench --test kv_quant_probe --release --
--ignored --nocapture`:

| width | peak MiB | delta         | ref ppl | ppl delta |
|-------|----------|---------------|---------|-----------|
| off   | 631.5    | (baseline)    | 4.9432  | (baseline)|
| 2     | 523.1    | -108.4 (-17.2%) | 4.9452 | +0.0020 |
| 3     | 511.3    | -120.1 (-19.0%) | 4.9854 | +0.0422 |
| 3.5   | 448.2    | -183.3 (-29.0%) | 4.9675 | +0.0243 |
| 4     | 624.8    | -6.7 (-1.1%)    | 4.9617 | +0.0185 |

All five widths finite, perplexity delta small and monotone-ish across
widths. Read the MiB delta per the probe's own note: this is measured at
the protocol's own context window on a family where KV dominates the
counted footprint (Gotcha 40), so it is not a footprint ceiling and does
not transfer to a family where KV is a smaller share. `--kv-bits 4`'s
-1.1% is real but small at this window; it would read larger at a longer
context, where KV is a bigger fraction of `phys_footprint`.

**RUN 2026-09-07**, same session: `AGENTS.md`'s greedy-then-sampled pair,
`--kv-bits 4`, both installs. All four runs reached `stop=MaxTokens` with
no error, and the text stayed coherent through 400 new tokens on both
models (gemma4 42.5/42.3 tok/s greedy/sampled, qwen38-27b 18.2/18.4).
`--kv-bits off` byte-identical to `HEAD~1` (a two-binary diff) is still
owed and not run this session. Structurally, every family's fixture-level
suite already carries a `kv_quant_off_matches_the_original_open_entry_point`
case that asserts `KvQuant::Off` reproduces `RealForwardRunner::open`'s
logits exactly on synthetic weights -- a weaker guarantee than a real-binary
diff (untrained weights cannot see a real numerics regression), but it does
mean the off path is provably the SAME CODE, not merely unchanged output on
one run.

**RUN 2026-09-09, on battery (correctness-only, no throughput/power
claims made from these numbers): extended `kv_quant_probe` to the three
remaining wired families that had real installs on disk.**

`gpt-oss-20b` (`~/.turbospark/models/gptoss-20b.gturbo`, family `GptOss`):

| width | peak MiB | delta | ref ppl | ppl delta |
|-------|----------|-------|---------|-----------|
| off   | 5428.6   | (baseline) | 161002.5320 | (baseline) |
| 2     | 5253.3   | -175.3 (-3.2%) | 626.8034 | -160375.7286 |
| 3     | 5392.6   | -35.9 (-0.7%)  | 172868.2789 | +11865.7469 |
| 3.5   | 5393.0   | -35.6 (-0.7%)  | 97664.5041 | -63338.0279 |
| 4     | 5399.4   | -29.2 (-0.5%)  | 1007.1123 | -159995.4197 |

All five widths finite and deterministic (the test's own pass/fail
criteria). **The perplexity column is not a quality reading on this
install and should not be quoted as one**: this is the exact artifact
`reference_ppl`'s own doc comment and `crates/bench/CLAUDE.md` Gotcha 13
describe -- the probe's assistant prefix is empty, which is correct for
Gemma/ChatML/Mistral/Qwen and "understates badly" for Harmony, whose
generation prompt ends mid-frame before `<|channel|>`. A five-figure,
non-monotone perplexity that swings by orders of magnitude between
adjacent widths (2-bit reading BETTER than off) is that framing artifact
dominating the signal, not TurboQuant damage; the module doc's own advice
("read a huge absolute number as that artifact rather than as TurboQuant
damage") applies verbatim. Getting a real quality read on this family
would need `reference_ppl` to splice in `<|channel|>final<|message|>`
first, which this probe does not do. The footprint delta is small and
that direction is real: `gpt-oss`'s counted peak is expert-slot-cache
dominated (Gotcha 1/36), not KV, so quantizing KV alone should move
little of it.

`qwen3moe` (`~/.turbospark/models/qwen3moe.gturbo`, the `llama` flow's
Qwen3-MoE half, ChatML dialect, no known framing artifact):

| width | peak MiB | delta | ref ppl | ppl delta |
|-------|----------|-------|---------|-----------|
| off   | 2755.3   | (baseline) | 14.5988 | (baseline) |
| 2     | 2427.0   | -328.3 (-11.9%) | 46.4347 | +31.8360 |
| 3     | 2480.1   | -275.2 (-10.0%) | 14.2285 | -0.3702 |
| 3.5   | 2508.6   | -246.6 (-9.0%)  | 15.4556 | +0.8569 |
| 4     | 2519.8   | -235.4 (-8.5%)  | 14.9719 | +0.3732 |

Clean and legible: all five widths finite and deterministic, footprint
down 8.5-11.9% (bigger than gpt-oss's, still a minority of the counted
peak on an MoE family), and the perplexity delta is small at 3/3.5/4-bit
(-0.37 to +0.86 nats) with a visible quality cliff at 2-bit (+31.8 nats)
-- exactly the shape the original 2026-08-15 assessment predicted for
sub-4-bit widths on a D=128 family (the "Ideas worth taking" item 2, now
overridden to ship anyway as opt-in). This is the third family with a
clean, quotable real-install reading, after gemma4 and qwen38-27b (dense).

`qwen4_exp` (`~/.turbospark/models/qwen4-reap288.gturbo`, 68 GiB, 288
experts top-10, ~10 minutes to run):

| width | peak MiB | delta | ref ppl | ppl delta |
|-------|----------|-------|---------|-----------|
| off   | 2541.6   | (baseline) | 8.7224 | (baseline) |
| 2     | 2472.3   | -69.2 (-2.7%)  | 9.4305 | +0.7081 |
| 3     | 2675.3   | +133.8 (+5.3%) | 8.7502 | +0.0278 |
| 3.5   | 2677.4   | +135.8 (+5.3%) | 8.6615 | -0.0609 |
| 4     | 2679.2   | +137.6 (+5.4%) | 8.7539 | +0.0315 |

All five widths finite and deterministic. Perplexity is the tightest
reading of any family measured so far (-0.06 to +0.71 nats), which is
plausible: this family's QSA/GDN layer mix leaves few full-attention
layers eligible for quantization at all (`model_io::kv_quant::layer_is_
quantized`), so there is little KV in play to damage.

**One genuine anomaly, and TWO REFUTED EXPLANATIONS OF IT** (amended
2026-09-09): at 3, 3.5 and 4-bit widths the counted footprint INCREASES
over `off`, the only family measured here where quantizing KV costs MiB
rather than saving them, and only 2-bit shows the expected decrease.

The first version of this paragraph offered codec overhead (per-row norms
stored separately, the Hadamard sign vectors, group scale planes)
outweighing the bit savings on a family with little quantizable KV. **The
source refutes that by arithmetic and it should not be repeated.** Every
extra allocation is KiB-scale and fixed: `KvQuantTables` is one sign
vector plus a codebook and its midpoints per side
(`crates/gpu/src/kv_quant_tables.rs`), and `DecodeScratch::kv_stage` and
`tq_attn` are tens to hundreds of KiB
(`crates/runtime/src/real_forward_types.rs`). The quantized cache
REPLACES the FP16 one rather than shadowing it -- `crates/gpu/src/
kv_cache.rs` sizes each layer's buffer from `kv_layer_strides`, and
`tq_row_bytes` is strictly smaller than FP16 at every supported width. A
few hundred KiB cannot produce +137 MiB.

The tempting replacement is one-time MSL pipeline compilation, which
`crates/bench/CLAUDE.md` Gotcha 1 measures at +139.2 MiB, within a few MiB
of this cluster. **That does not survive the SHAPE of the data.** `bits`
rides as a runtime uniform rather than a function constant
(`crates/gpu/src/kv_quantize.rs` passes an empty `FunctionConstantValues`,
and `attention_tq.rs` passes `k_bits`/`v_bits` as buffer arguments), so
all four widths share ONE pipeline. The 2-bit row opens first among the
quantized rows and would pay any compilation, yet 2-bit is the only row
that goes DOWN.

So the mechanism is unexplained, and the honest reason is that **the
probe cannot separate it as written**: `off` runs exactly once, always
first, and is never repeated, so there is no spread bound at all on a
+/-135 MiB delta. `crates/bench/CLAUDE.md` Gotcha 23's "measure the
reference arm's own spread first" is not applied here. The discriminator
is one line -- append a second `off` row to `rows` and reorder so `2` is
not the first quantized open -- and until it runs, this number is neither
a regression nor noise. It is tracked as `ROADMAP.md` Priority 0 item 6.

`museGlimmer` has no install on disk (`CLAUDE.local.md`'s "gone from
disk" list) and was not attempted.

**Coverage after this session: 5 of 7 wired families have a real-install
`kv_quant_probe` reading** (gemma4, qwen38-27b dense, gpt-oss, qwen3moe,
qwen4_exp), all finite and deterministic at every width, with 4 of the 5
also giving a legible perplexity signal (gpt-oss's is masked by the
Harmony framing artifact above). The remaining "still owed" items: the
literal two-binary `--kv-bits off` diff against a pre-feature build (the
structural fixture-level proof above is a substitute, not that proof
itself), a museGlimmer reading once that install exists again, and
`AGENTS.md`'s greedy-then-sampled CLI-level coherence smoke specifically
with `--kv-bits` on for `gpt-oss`, `qwen3moe` and `qwen4_exp` (this
session ran the memory/perplexity probe on all three but not that smoke).
Added 2026-09-09: `spark2_5` has neither a fixture suite nor a reading,
and unlike museGlimmer its install is already on disk
(`~/.turbospark/models/spark25.gturbo`, 2.4G), so nothing blocks it.

## P0 verification follow-up (2026-09-09)

This follow-up supersedes earlier "still owed" notes. Older single-reference
footprint deltas above remain historical observations; their causal wording
is not supported by a measured reference spread and should not be reused as
a quantified savings claim.

The probe now opens `off, off, off, 3, 3.5, 4, 2, off`. Every delta stays
anchored to the FIRST off row; subsequent references no longer replace the
baseline. The footer reports all four FP16 peaks' minimum, maximum, and
spread. A delta inside that spread is unresolved. A larger delta warrants
investigation, not automatic attribution to quantization. Row order and
process history remain potential confounds; this is an observed spread,
not a statistical confidence interval.

Spark now has `real_forward_spark_kv_quant.rs`: five non-ignored Metal
fixture cases covering off equivalence, finite/deterministic width sweeps
past the eight-token sliding window, quantization engagement, chunked
prefill agreement, and named batched-GEMV refusal. Eight fixture layers
leave only layer 3 quantized. The fixture supplies Spark's own fused QKV,
per-class RoPE, and headwise gate. All five pass on the host GPU.

Mutation checks: changing the off-comparison runner to K2/V2, the second
width-sweep runner to off, the engagement runner to off, and the chunked
comparison runner to off each fail their respective assertion. Disabling
the runtime's batched-GEMV refusal fails the refusal case. Every mutation
was uniquely matched, read back, and restored; the restored suite passes.
The first four are mismatched-configuration negative controls, not claims
of exhaustive kernel mutation coverage.

Workspace build, formatting and Clippy passed. The workspace test run
failed in the unrelated mapped-residency family-list assertion: an existing
working-tree `Qwen3Dense` addition makes `ModelFamily::ALL` length 11 while
that test lists 10. Remaining test targets were run with `--no-fail-fast`;
there were no additional failures. The concurrent family change was not
modified. Raw commands and failures are retained in the evidence.

### Literal pre-feature compatibility

Built `673341e^` and `673341e` from separate source archives and separate
Cargo target directories. On the existing Gemma install, both greedy
(seed 1, T=0.0001, top-k 1) and sampled (seed 20260721, CLI sampling
defaults) runs use 4096 context, 16 slots, 400 new tokens and speculation
off. The old binary receives no KV flag; the feature binary receives
`--kv-bits off`. Only the generated answer bytes are compared, stripping
the explicit diagnostic preamble and keeping the answer's newline.

Both comparisons are byte-identical: greedy 1895 bytes, SHA-256
`f0484a2e3090922d560e37d64b162cbde8a244405da945239b4d8b972d4992a3`;
sampled 1950 bytes, SHA-256
`6b9fd953ce05d3ef12a1a887eaf75006bd77444f34b39d61550b627146b5d2f4`.
This isolates the feature commit; it does not assert every later commit
is byte-identical to that historical revision.

### gpt-oss greedy completion remains open

With K4/V4, greedy generation on the bare wetlands ask still exhausts the
family's 3072-token budget in repetitive reasoning without a visible
answer. The matched FP16 greedy control reaches EndOfTurn at 2170 tokens
and emits an answer. Sampled K4/V4 reaches EndOfTurn at 1521 tokens with
coherent prose. All use 8192 context, 16 slots, speculation off and the
same seeds/sampling settings as the other smoke runs. This is an open
quality finding, not a proven kernel mechanism and not a passing greedy
smoke. Earlier 400-token runs exhausted the budget in both FP16 and K4/V4;
those short controls could not distinguish the arms.

The separate, correctly framed FP16 quality gate reproduced perplexity
12.0801 and both digests. Its memory oracle passed all three cases at
EndOfTurn, peak 5415.9 MiB against 5700, and replay growth 0.00 MiB. Neither
baseline gate establishes the quantized greedy arm's quality. Do not
substitute the CLI process's zero exit status for a coherence verdict.

### Qwen3-MoE smoke completed

Reinstalled the catalog's Qwen3-30B-A3B GGUF Q4_K_M into a task-owned
path. K4/V4, 4096 context, 16 slots and speculation off: the initial
400-token runs were coherent but truncated. With the family's 1024-token
budget, greedy (seed 1, T=0.0001, top-k 1) reaches EndOfTurn at 540 tokens;
sampled (seed 20260721, CLI defaults) at 620. Both answers remain coherent.
The quality gate reproduces perplexity 14.5988 and both frozen digests.
The memory oracle passes all three cases at EndOfTurn, peak 2740.2 MiB
against 2900, replay growth 0.22 MiB. The task-owned install was removed
only after its metadata, commands, outputs and results were saved.

### Spark real-install probe

Apple M4 Max, existing `spark25.gturbo`, frozen 8192 context, 16 slots,
48 greedy decode steps per probe row. No concurrent inference run was
started by this session. Raw run took 135 seconds; all rows finite and all
quantized replays deterministic.

| width | peak MiB | delta from first off MiB | unframed ref ppl |
| --- | ---: | ---: | ---: |
| off | 573.5 | baseline | 12.0190 |
| off | 582.1 | +8.6 | 12.0190 |
| off | 592.5 | +19.1 | 12.0190 |
| 3 | 395.2 | -178.3 | 12.2784 |
| 3.5 | 419.8 | -153.7 | 12.1145 |
| 4 | 440.1 | -133.4 | 12.0312 |
| 2 | 420.4 | -153.1 | 12.6491 |
| off | 627.6 | +54.1 | 12.0190 |

FP16 reference spread is **54.1 MiB**. Every quantized row is below every
FP16 reference in this run, but the non-monotone width ordering is not a
ranking of codec memory efficiency. The probe leaves Spark's think frame
open; its perplexity is not the answer-slot reading from
`spark_quality_gate`. That separate gate reproduced **12.6162**, both
frozen digests, and the constrained-cache digest unchanged.

Greedy and sampled CLI smoke with `--kv-bits 4`, 8192 context, 16 slots,
400-token budget, seeds 1/20260721 and speculation off produced coherent
wetlands explanations, both reaching EndOfTurn (377/400 new tokens).
The memory oracle also passed: session peak 572.9 MiB against 700 MiB,
all three protocol cases at EndOfTurn, replay growth 0.00 MiB. Full commands
and output are retained in the [verification evidence](verification/p0-2026-09-09.json).

### museGlimmer real-install probe

Restored `mlx-community/Muse-Glimmer-30B-4bit` at revision
`3e7677d7a40d348a3daba263a2b1c0aa41910710` into a task-owned temporary
install (14.6 GiB). Apple M4 Max on AC, frozen 8192 context, 16 requested
slots, 48 greedy steps per row, no concurrent inference. The revised
probe passed in 343.60 seconds, all rows finite and quantized replays
deterministic.

| width | peak MiB | delta from first off MiB | unframed ref ppl |
| --- | ---: | ---: | ---: |
| off | 544.4 | baseline | 6.5606 |
| off | 465.8 | -78.6 | 6.5606 |
| off | 514.1 | -30.3 | 6.5606 |
| 3 | 452.8 | -91.6 | 6.6954 |
| 3.5 | 428.0 | -116.4 | 6.5963 |
| 4 | 399.8 | -144.5 | 6.6255 |
| 2 | 384.2 | -160.2 | 6.6825 |
| off | 489.5 | -54.9 | 6.5606 |

FP16 minimum/maximum/spread: **465.8 / 544.4 / 78.6 MiB**. All quantized
reductions exceed this observed spread, warranting further attribution
work rather than establishing exact savings or a width ranking. The
unframed perplexity here differs from the answer-slot quality gate: this
probe does not append museGlimmer's `to=user` assistant prefix.

The `--kv-bits 4` greedy/sampled CLI pair used 8192 context, 2048 new-token
budget, 16 slots, speculation off and seeds 1/20260721. Both produced
coherent wetlands answers and reached EndOfTurn at 995/857 new tokens.
The frozen quality gate reproduced perplexity 6.2810 and the greedy and
constrained-cache digests, but FAILED its sampled golden: actual
`4125a2a664f8df3e6af1ed7c5a497d8c433c952540a1be010bb878e5baadc1c9`,
expected `b8349cd1a303346812d3262e32020158eef6c5c15583dfa25dc7599e23fb508a`.
This gate runs with KV quantization OFF. Its failure is retained separately
from the successful KV4 smoke; the golden is not changed.

A second current-source run and an isolated `673341e` build both reproduced
the same actual sampled digest, perplexity and greedy digest. That excludes
this task's verification edits and later working-tree changes as the source
of this mismatch, but does not establish its underlying cause. The sampled
golden remains an open finding. The memory oracle passed all three cases
at EndOfTurn, peak 533.0 MiB against 650 MiB, replay growth 0.00 MiB.
After saving the raw logs and checkpoint metadata, only this task-created
museGlimmer install was removed.

### Qwen4 REAP-288 spread-controlled probe

Restored `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` at revision
`668f31bcc56bf9400e64c9463445eee47597c2d9` into a task-owned 68.1 GiB
install. Apple M4 Max on AC, frozen 2048 context, 16 slots, 48 greedy steps
per row. The revised probe passed in 464.89 seconds: every row finite,
every quantized replay deterministic.

| width | peak MiB | delta from first off MiB | ref ppl |
| --- | ---: | ---: | ---: |
| off | 2527.4 | baseline | 8.7224 |
| off | 2503.8 | -23.6 | 8.7224 |
| off | 2572.1 | +44.6 | 8.7224 |
| 3 | 2537.3 | +9.9 | 8.7502 |
| 3.5 | 2538.7 | +11.3 | 8.6615 |
| 4 | 2540.3 | +12.9 | 8.7539 |
| 2 | 2535.3 | +7.9 | 9.4305 |
| off | 2575.5 | +48.0 | 8.7224 |

FP16 minimum/maximum/spread: **2503.8 / 2575.5 / 71.6 MiB**, independently
rounded from raw bytes. Every quantized delta is within that spread and
every quantized peak is inside the reference range. These differences are
**unresolved**. The earlier approximately +135 MiB pattern at 3/3.5/4-bit
does not reproduce under this row order, but that does not identify its
cause or establish a quantization memory win. Both earlier refuted causal
explanations remain refuted; no production policy or frozen baseline changes.

The KV4 CLI pair used 4096 context, 1024 new-token budget, 16 slots,
speculation off and seeds 1/20260721. Both reached EndOfTurn (499/542 new
tokens). Greedy remained readable, but the sampled answer explicitly denied
that coastal wetlands reduce flood damage. A matched `--kv-bits off`
sampled control completed at 676 tokens and kept the correct overall premise,
while also making unsupported numerical claims. Retain this as a sampled
answer-quality concern from one matched pair, not a clean smoke pass or a
proven general quantization regression.

The separate frozen FP16 quality gate PASSED: perplexity 8.7224, greedy
digest `9f9ed49203dda1aad1b379eb503a3dd8b565d53ccc38ff417fb35e43ed9e0795`,
sampled `4cf6da5560936e306df09fa09bca7bd19950429e910cee2b66c93c8ed0335772`.
The family has no legal constrained-cache arm below 16 slots. Its memory
oracle also passed at the frozen 2048 window: both supported protocol cases
at EndOfTurn, peak 2514.4 MiB against 3000 MiB, replay growth +0.23 MiB.
Neither baseline gate uses KV quantization and neither dismisses the
sampled KV4 answer concern.
After the phase measurements documented in `QWEN4_EXP.md` completed, raw
logs and checkpoint metadata were saved and only the task-created Qwen4
install was removed. All pre-existing user installs were retained.

## The original assessment (2026-08-15)

Kept for the record: this is what made KV quantization look like a bad
trade before it was reframed as an opt-in flag rather than a default. Its
factual claims about turboquant-mlx's own measurements, and about this
port's memory and throughput profile as they stood in 2026-08-15, are
unedited below.

The question this page originally answered: should this engine adopt
KV-cache quantization in the shape of `sharpner/turboquant-mlx`, a proof
of concept of Google's 2025 TurboQuant paper on Apple Silicon, and if not,
are any of its ideas worth taking anyway?

The answer, assessed 2026-08-15 by reading their README against this port's
own recorded measurements (no code was run and nothing here is a new
measurement): implementable, but it does not pay on this engine today.
KV is not the dominant memory term for any family that is near a memory
ceiling, attention is a single-digit share of decode at this port's 4-8K
contexts, and the one thing that makes turboquant-mlx fast (MLX's fused
`quantized_matmul`) is exactly the kernel this port would have to write
itself. What is worth keeping is their negative results and one favorable
datapoint about Gemma's head dimensions, all recorded below with explicit
revisit triggers.

This is a desk assessment, not a measured negative like
`docs/EXPERT_ROUTING.md` or `docs/SPECULATIVE_DECODING.md`: their numbers
are theirs (different models, different engine) and this port's numbers
come from its own frozen rows. The two are never blended into one
comparison here.

## What turboquant-mlx is

A proof of concept of the TurboQuant paper (Google, 2025): quantize the
attention KV cache to 2-4 bits so long contexts fit in memory and the
attention pass reads fewer bytes. Weights are untouched. This is entirely
orthogonal to the weight quantization this port already does everywhere.

Two implementation paths:

- **V2 (affine, hardware-accelerated):** K/V quantized through
  `mx.quantize` and consumed by `mx.quantized_matmul`, MLX's fused
  dequant-inside-the-matmul Metal kernel. Variants add a random QR
  rotation before quantizing (distributes outlier channels so the values
  are closer to iid Gaussian) and a 1-bit QJL residual (a
  Johnson-Lindenstrauss sign-bit correction to attention scores).
- **V3 (Lloyd-Max codebooks, paper-correct):** optimal centroids at 2-4
  bits, plus outlier-channel splitting for fractional rates (2.5/3.5
  bit). No fused kernel: dequant is software in the SDPA hot path.

Tested on Llama 3.2 3B, Llama 3.1 8B, Mistral 7B, and Gemma 3 4B on an
M4 Max (64 GB). Note their baselines are 4-bit WEIGHT-quantized models,
so KV error compounds on top of weight error. That is the same regime
this port's installs are in, which makes their quality numbers more
relevant here than the paper's full-precision ones.

## Their measured results

All numbers below are theirs, from their README, on their models. None
of them transfer to this port's installs as values. The SHAPE of the
results is what transfers.

Perplexity (lower is better, deltas vs their fp16-KV baseline):

| strategy | Llama 3.2 3B | Llama 3.1 8B | Mistral 7B | Gemma 3 4B |
|---|---|---|---|---|
| fp16 baseline | 12.94 | 9.47 | 6.79 | 12.18 |
| V2 4-bit rotated | -0.8% | +1.4% | +1.4% | +2.9% |
| V2 3-bit rot+QJL | +5.3% | +7.8% | +5.1% | **-1.1%** |
| V3 3.5-bit mixed | +0.3% | +6.7% | +4.0% | +2.1% |
| V3 3-bit Lloyd-Max | +5.1% | +8.6% | +7.0% | +6.2% |
| V3 2.5-bit mixed | +27.0% | +35.2% | +10.8% | +7.0% |
| V3 2-bit Lloyd-Max | +64.3% | +65.5% | +19.3% | +20.2% |

Throughput (Llama 3.2 3B, tok/s) and KV compression at T=8192:

| strategy | T=512 | T=8192 | KV at 8192 | ratio |
|---|---|---|---|---|
| fp16 | 208 | 148 | 969 MB | 1x |
| V2 4-bit LEAN | 188 | **156** | 266 MB | 3.6x |
| V2 3-bit rot+QJL | 101 | 45 | -- | -- |
| V3 3.5-bit mixed | 82 | 24 | 236 MB | 4.1x |
| V3 3-bit Lloyd-Max | 98 | 27 | 207 MB | 4.7x |

Four findings worth extracting:

1. **4-bit affine KV is near-lossless and near-free** (when a fused
   kernel exists). V2 4-bit runs -0.8% to +2.9% perplexity at ~105% of
   fp16 speed at 8K (the compressed cache reads fewer bytes, so it wins
   at long context).
2. **Below 4 bits there is a quality cliff.** 3-bit costs 5-9% on
   D=128 models, and 2-bit costs 19-65% everywhere. Consistent with this
   port's own weight-side experience that sub-4-bit needs either
   codebooks (Phase S) or QAT (the ternary family), neither of which
   their V3 software path could afford at speed.
3. **The paper's QJL residual is a measured negative in practice.**
   Their own explanation: the correction is linear in attention scores,
   and softmax amplifies score errors exponentially, so (b-1)-bit MSE
   plus a 1-bit residual loses to plain b-bit. A residual-correction
   scheme for KV should not be built here without a number that
   contradicts theirs.
4. **Head dimension decides quantization tolerance.** D=256 (their
   Gemma 3) quantizes far better than D=128 (their Llamas): rot+QJL at
   3 bits even beat fp16 on Gemma (-1.1%, a regularizer effect). More
   coordinates per head means outliers average out.

And one about cost: **V3, the path with no fused kernel, runs at ~16%
of fp16.** That is the price of software dequant in the SDPA hot path,
and it is the floor this port would start from without kernel work.

## What implementing it here would take (2026-08-15 estimate; see "The reversal" above for what actually landed)

This port's KV cache was FP16 end to end at the time of this estimate.
`crates/gpu/src/kv_cache.rs`
allocates one K and one V Metal buffer per layer at construction (linear
for full-attention layers, a `sliding_window + 128` ring for SWA layers
under `fp16_ring_enabled`), the host memcpys one token's row per layer
per step (`write_k`/`write_v`), and `attention_decode.rs`'s split-KV
kernel reads the buffers through `KvView` with ring addressing
specialized via `FC_ATTN_RING_CAP`.

A V2-shaped 4-bit KV would need:

- **A write-side quantizer.** Per-token, per-head-group affine
  quantization of the K row and V row before (or instead of) the memcpy.
  Cheap either way: one row per layer per token.
- **A read-side FUSED dequant inside `attention.metal`**, in both
  places attention touches KV: the QK^T score loop of
  `attention_decode_partial` and the AV accumulation. This is the hard
  part. turboquant-mlx never wrote this kernel: `mx.quantized_matmul`
  is MLX's, and their V3 numbers (16% of fp16) show what happens without
  it. This port has no equivalent to borrow. The existing dequant GEMVs
  (INT4/INT8/Q8_0/...) are weight-matrix kernels with the wrong
  addressing for a ring-buffered, GQA-shared, split-KV cache.
- **New specialization axes done right.** Block format and group size
  would have to join `attention_constants_key` beside scale, ring
  capacity, chunk count, and sinks (crates/gpu CLAUDE.md Gotcha 7), or
  quantized dispatches silently reuse the FP16 pipeline, the exact trap
  AGENTS.md Gotcha 18 records for the ring capacity itself.
- **A rotation stage if quality needs it**: a fixed random orthogonal
  matrix folded into the Q/K projections at repack time (QuaRot
  lineage), which touches `crates/repack` and every family's install
  format.
- **Re-frozen quality gates for every family.** Quantized KV moves
  numerics on every generated token of every install. Per the
  verification policy, that is all three real-model gates per family
  PLUS the quality gate, with every digest re-frozen and the perplexity
  movement explained. Eight families now.

Effort class: a full phase, comparable to a family bring-up (the
attention kernel alone is harder than a GEMV port: it has the ring, GQA,
split-KV combine, and sinks axes to preserve). That cost is not the
reason to decline, but it sets the bar the benefit has to clear.

## Whether it pays here, as a DEFAULT (2026-08-15 analysis; still the reason `--kv-bits` defaults OFF)

It does not, today, on either axis -- which is an argument for the flag
defaulting off, not for refusing to offer it. See "The reversal" above.

**Memory.** Where KV actually sits in this port's measured footprints
(all numbers from the recorded oracle rows and their accounting in
AGENTS.md / CLAUDE.local.md):

| family | KV | measured peak | KV share | dominant term |
|---|---|---|---|---|
| mistral7b dense @ 8192 | 1,024 MiB | 1,201 MiB | ~85% | KV |
| qwen3_5 dense 27B @ 4096 | 256 MiB | 660 MiB | ~39% | KV + GDN state |
| gemma4 @ 4096 | ~320 MiB | 2,100-2,200 MiB | ~15% | expert slot cache |
| qwen3moe @ 4096 | small | ~2,750 MiB | <10% | slot cache (2,094 MiB) |
| gpt-oss @ 8192 | small-ish | ~5,420 MiB | <10% | slot cache (4,854 MiB) |

The families near their ceilings are MoE, and their dominant term is the
expert slot cache, which KV quantization cannot touch. The one family
where KV dominates (dense Mistral at 8,192) already runs in 1.2 GiB on a
36 GB machine, so cutting it to ~450 MiB enables nothing that is blocked
today. And this engine already carries three KV mitigations that
turboquant-mlx's uniform cache does not: the SWA ring (AGENTS.md Gotcha
18: 922 -> 320 MiB on Gemma), gpt-oss's alternating 128-token window on
even layers, and Qwen's linear-attention layers carrying no per-token KV
at all (30 of 40 layers on qwen36, 36 of 48 on the dense 27B).

**Throughput.** Post-split-KV, `attention_decode_partial` is 13-19% of
cb1's GPU busy time (CLAUDE.local.md, 2026-08-07 dispatch ranking),
which is single-digit percent of a whole decode token. Reading 4x fewer
KV bytes cannot buy much from a term that small. Their V2 speed win
appears at 8K+ context on a model whose attention share is large. This
port's protocol runs at 4,096-8,192 by design, and at those windows
attention is already flat and cheap here.

**Quality risk.** Every point of KV quantization is pure downside on
the quality axis until context grows: the gates would absorb a
perplexity perturbation on all eight families to save memory that
nothing is short of.

The general form, consistent with this repo's own findings: KV
quantization is a LONG-CONTEXT technology. TurboQuant's own headline
(969 MB of KV at a mere 8K on a 3B model) describes a model with
D=128 x 28-ish layers of full attention and no ring. This engine's
families either window, ring, or linearize most of their layers, which
is the same memory being saved by architecture instead of by
quantization.

## Ideas worth taking

Knowledge, mostly. In order of value:

1. **Do not build QJL-style residual correction for KV.** Their measured
   negative, with a mechanism (softmax amplifies score error). Cite this
   page if it comes up.
2. **If KV quantization is ever built, 4-bit affine is the only point
   worth building.** The 3-bit and below cliff is steep on D=128 and
   their V3 codebook path shows the kernel cost of fancier schemes.
   Skip fractional rates, skip Lloyd-Max, skip 2-bit. **OVERRIDDEN
   2026-09-06**: what shipped is the full Lloyd-Max V3 path at all five
   widths including 2-bit, because a real fused kernel closed the cost gap
   this item was reacting to, and every width below `off` is opt-in rather
   than a default anyone pays for unasked. The quality-cliff finding
   itself is not disputed -- a caller choosing 2-bit is choosing that cliff
   with eyes open, which is a different thing from this port choosing it
   for them.
3. **Gemma is the pilot family.** Its head dims (256 SWA / 512 full) sit
   in the regime their data says is most quantization-tolerant: their
   D=256 model was the one where quantized KV matched or beat fp16.
   The D=128 families (llama, qwen3moe, gpt-oss) are where the quality
   risk concentrates.
4. **Random-rotation preprocessing is a real quality lever** (their
   rotated variants beat plain at every width). It is not specific to
   KV (it is the QuaRot family of tricks) and could matter to future
   weight quantization work here, but nothing current asks for it: the
   incumbent MLX INT4 installs are already the best-measured
   quantizations in `docs/BENCHMARKS.md`.
5. **Norm-baking** (fold L2 norms into the quant scales so dequant needs
   no separate elementwise pass) is a tidy micro-optimization that only
   exists once a quantized-KV path exists. Note only.

Already present here, no action: pre-allocated per-layer buffers (their
step=256 growth trick solves a problem this port's
allocate-once-at-construction design does not have), ring buffers,
`MADV_DONTNEED` on reset, fused Metal attention.

## What would have changed the 2026-08-15 answer (now moot; kept for the reasoning)

This section asked what would justify building the feature at all. That
question is answered -- it is built, opt-in, default off. What is left of
it is a note on when `--kv-bits` might be worth DEFAULTING to something
other than off, which nothing here proposes today:

- **Long-context support landing.** If the protocol context ever moves
  to 32K+, KV becomes the growing term for every family (it is the only
  per-token term on dense families, per Gotcha 40's accounting), and a
  case could be made for resolving `--kv-bits auto` to 4-bit on the
  families closest to a ceiling. Nothing here proposes that default change
  today; it would need its own quality-gate sign-off per family.
- **A memory ceiling actually binding.** Today no install is refused for
  KV. If a family arrives whose KV at its own protocol window does not
  fit beside its slot cache, `--kv-bits` is already the tool to reach for.
- **An attention-bound profile.** If a future dispatch ranking shows
  `attention_decode_partial` dominating cb1 (it grows with context, and
  today it is third), the shipped kernel is already the throughput lever;
  nothing further needs building.

## Real-model verification policy

Real-install results and remaining failures are recorded in the dated
sections above. Before quoting a new memory or quality number for any
`--kv-bits` width: run `crates/bench/tests/kv_quant_probe.rs`
(`TURBOSPARK_KV_QUANT_INSTALL_DIR=<install> cargo test -p turbospark-bench
--test kv_quant_probe --release -- --ignored --nocapture`) and the
standard greedy-then-sampled real-model smoke from `AGENTS.md` with
`--kv-bits` set, per family. Interpret footprint against the repeated-off
spread and retain framing limitations and failed gates in the report.
