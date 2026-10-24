# Kapitsa kernel comparison and adoption

A read-only comparison of `nace-ai/kapitsa` (C++/Metal engines for the Qwen
3.5/3.6/3.8 and Nemotron families) against TurboSpark's Metal kernels, and the
record of what was adopted from it. The full ranked comparison that motivated
this work lived in the review that opened it; this page holds what outlived
that session: the pinned source, the decision table, and per-item design notes
with their measured results.

Scope: kernel and dispatch-structure ideas only. Kapitsa's headline advantage
over this engine is its DFlash2-style block drafter and batched multi-request
verify (`Engine::run_batch`), which is a serving-shape difference, not a
kernel one, and is not covered here.

## Pinned sources and license

- `nace-ai/kapitsa`, Apache-2.0, with MLX-derived (MIT) kernel layouts.
  Reviewed from its `src/kernels/*.metal` and `src/metal/gpu.mm`.
- Ideas are reimplemented here; no code was pasted. Copied code would need a
  NOTICE entry, and nothing here is a copy.
- Our kernels that the comparison ranked against: `crates/gpu/src/shaders/`
  (the kapitsa-side file names `verify.metal`/`verify8.metal` in the original
  review are ITS files; ours are `gdn.metal`, `dequant_int4*.metal`,
  `moe*.metal`, dispatched from `crates/gpu/src/*.rs`).

## Where we were already ahead (not adopted, do not re-propose)

- Decode GQA attention: ours is barrierless with unroll 4; kapitsa stages
  16-position tiles with 2 barriers.
- GDN decode grid `(Hv, Dv/4)` splits more finely than kapitsa's
  one-threadgroup-per-head.
- Shape baking via function constants; kapitsa has none.
- GPU top-k sampling and greedy LM head (`sample_topk64_*`,
  `lm_head_greedy_int4_rows_*`).
- Kapitsa's half-precision inner products (`qmvh`) can NaN under fast-math per
  its own comment.
- Kapitsa's multi-row attention sharing chunk boundaries is not bit-exact;
  AGENTS.md forbids that for verify.

## Decision table

| # | Idea | Status | Where |
|---|------|--------|-------|
| 1 | Lazy rank-1 GDN verify tape | Adopted, default on (`TURBOSPARK_GDN_LAZY_TAPE=0` opts out) | Item 1 below |
| 2 | Concurrent compute encoder with explicit barriers | Plumbing landed and tested; not wired, no runtime flag | Item 2 below |
| 3 | Row-batched verify elementwise ops | Was already in flight in this tree (`TURBOSPARK_QWEN_BATCHED_ROW_OPS`) | not this page |
| 4 | Fused MoE decode half (segment trick) | Adopted bit-exact form | Item 4 below |
| 5 | Fused dense FFN tail | Adopted bit-exact form | Item 5 below |
| 6 | Shift-free nibble masking in the int4 GEMV | Bench prototype | Item 6 below |
| 7 | Integer-code 8-row verify GEMM | Bench prototype | Item 7 below |
| 8 | Expert-sorted grouped GEMM MoE prefill | Bench prototype | Item 8 below |
| 9 | Fused GDN prep (conv+SiLU+qk-norm+g/beta) | Not adopted: <= 1% decode/verify est., breaks the `(32,4)` delta layout contract | declined |
| 10 | Host-side FCV/set-bytes dedup in MoE decode | Was already in flight in this tree | not this page |
| 11 | GPU-side routing + single command buffer per token | Not adopted: `fast::exp` vs f32 `exp` can flip f16 routing weights and break frozen digests; streamed mode cannot use it | declined |
| 12 | Prompt-lookup (n-gram) draft source from mlx-serve | Standalone drafter declined on a replay (flat or losing on prose, ~1.5x ceiling on copy-heavy, about MTP's 1.44x). Hybrid with MTP unmeasured | Section "Prompt lookup" below |
| 13 | Adaptive speculation depth in the server | Candidate: the controller exists (`speculative_adaptive.rs`) and only the CLI uses it; constants are unmeasured | not started |

## What makes the headline numbers fast, and what applies here

A later research report compared kapitsa and mlx-serve on Qwen3.8-27B. This
section keeps the part that survived checking it against this tree. The
report's sources are the two projects' READMEs and release notes, which were
not re-read for this page.

Vendor-reported decode speed, Qwen3.8-27B 4-bit, a 512-token coding answer,
each engine in its own "fastest setup" (kapitsa's README table):

| Engine | M5 Max 128 GB | M4 Pro 48 GB |
|--------|--------------:|-------------:|
| kapitsa | 102 tok/s | 63 tok/s |
| mlx-serve | 130 tok/s | 54 tok/s |

These M5 Max and M4 Pro rows are UNVERIFIED here and not targets. Neither
chip is the M4 Max 36 GB this repo measures on, drafter type, depth, sampling
and prompt state are not stated, and kapitsa's own formal M5 Max document
reports a lower median for the same model (72.9 tok/s sampled, 83.7 greedy,
run with background GPU contention). Do not quote them next to our rows.
Kapitsa WAS reproduced on this machine on 2026-10-22, and its claim for the
27B holds up here: see "Measured: kapitsa on this machine" below. The gap to
us is real and about 2x on the same prompts, not a table artifact.

The mechanisms, in order of how much of the number they explain:

1. **Speculation is the multiplier.** One verify pass of the target commits
   several tokens, so the cost per token is roughly `(draft + verify) /
   accepted-per-verify`. Both engines run an MTP head and a block drafter.
   Here that is item 1's territory and `docs/MTP.md`: block 2 is the only
   block that pays, 1.44x measured.
2. **INT4 bandwidth and in-register dequant.** Batch-one decode streams the
   weights once per target pass, so 4-bit is a 3-4x cut in bytes against BF16.
   Our decode GEMV already dequantizes in registers, so there is no
   materialized FP16 buffer to remove. Profile GB/s per kernel before
   assuming a gap.
3. **Specialized Metal per model family.** Kapitsa's own claim. Ours is the
   shape-baked function-constant kernels and the fused FFN (items 4 and 5).
4. **Neural Engine prefill** (kapitsa for 27B, ours opt-in in
   `docs/SPEEDUP_PLAYBOOK.md`). TTFT only, no effect on decode tok/s.
5. **Prefix reuse and concurrency.** Latency and aggregate throughput, not
   single-stream decode. Ours: `kv_prefix.rs` and `session_pool.rs`, which
   refuse a rewind on GDN models and with a drafter installed.
6. **Prompt lookup (mlx-serve, default on).** A model-free draft source. It
   has no shipped counterpart here, but it was measured once: see below.

What the report got wrong about this tree: its P1 list (native MTP, a fused
INT4 decode kernel, a persistent prefix cache) is already built, and its
"+20-40% from speculation" target is met by the 1.44x above. Its framing of a
17-19 tok/s TurboSpark baseline is the plain-decode row, not the MTP row.

### Measured: kapitsa on this machine, 2026-10-22

Apple M4 Max (reported by kapitsa as 32-core GPU), 36 GB, macOS 26.6.2, AC
power, host load 1.2-2.4 at each arm start (swap was 8.7-10.9 GB used and
rising, a standing condition of this host, not cleared). Kapitsa 0.2.6
(`76a4970`), Qwen3.8-27B 4-bit from its own pack
(`lmstudio-community/Qwen3.8-27B-MLX-4bit`, MTP head
`Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`, `incoai/Qwen3.8-27B-DFlash2`).
TurboSpark: clean HEAD worktree `2c02a5be`, release `turbospark-server`,
`qwen38-27b-mtp.gturbo` (`mlx-community/Qwen3.8-27B-4bit` trunk). The trunk
shards are 90-190 bytes apart in size and hash differently; tensor equality was
not checked, so read this as the same model and quantization scheme.

**Kapitsa's own harness** (`kapitsa bench --model 27b`, 512-token coding
answers, median of 5): greedy 90.7 tok/s (90.5-91.2), T=0.6 82.7 tok/s
(79.8-89.0), reading a 4K prompt 417 tok/s, a 16K prompt 403 tok/s, peak
memory 19.6 GB, no contention flagged. Decoding line in the result: "KFlash
speculative decoding (DFlash2 lattice, tree verify)".

**One external harness over each engine's HTTP API** (`scripts/bench_api.py`, driven by `scripts/bench_api_drive.sh`, same
three prompts, temperature 0, 512 max tokens, tokens counted with one
tokenizer, decode tok/s excludes the first chunk). Median of 3 runs per arm,
two rounds with the arm order reversed, mean of the two rounds shown (rounds
agreed within 3%):

| Arm | short-explanation | medium-review | copy-heavy |
|-----|------------------:|--------------:|-----------:|
| TurboSpark, speculation off | 22.3 | 22.3 | 22.4 |
| TurboSpark, MTP block 2 (default) | 34.9 | 37.2 | 47.8 |
| Kapitsa, MTP head only (`--dflash none`) | 41.5 | 46.2 | 64.1 |
| Kapitsa, KFlash (DFlash2 tree, default) | 52.6 | 68.4 | 101.3 |
| KFlash over our MTP | 1.51x | 1.84x | 2.12x |
| Kapitsa MTP over our MTP | 1.19x | 1.24x | 1.34x |

Both of our speculative arms' text is identical to plain decode on
medium-review and copy-heavy (same hash), and kapitsa's two modes agree with
each other on all three, so the speedups are lossless. On short-explanation
our MTP and plain text differ (487 and 512 tokens), the known
speculative-start difference in `docs/BENCHMARKS.md`.

Caveats that bound the claim:

- Texts differ between engines (different trunk files, chat template and
  possibly thinking behavior). The copy-heavy answer was 501 tokens on
  kapitsa and 187 on ours, so its acceptance is not measured on equal text.
  The 2.12x on that column is the weakest number here.
- Time to first token is not comparable: kapitsa reads 0.06 s on every run
  because the repeated identical prompt hits its prefix cache.
- Not measured: kapitsa with speculation off, memory under this harness,
  sampled runs through the API, and long contexts beyond kapitsa's own 4K/16K
  reads.

**What the gap is made of.** Our own plain decode is 22.3 tok/s, so the
engines start from the same place per step. Kapitsa's design doc
(`docs/kflash.md` in its repo, an M4 Pro measurement) gives the reason: on the
dense 27B a verify pass of 1, 2, 4 and 8 rows takes 74.4, 70.6, 71.6 and 74.3
ms against 67.3 ms for a plain one-row step. That is `c(8)` of about 1.1.
Ours is `c(9) = 4.53` (`docs/MTP.md`). With a flat verify cost, a deeper
block is nearly free, a block drafter proposing 7 tokens pays, and a tree of
candidates fits in rows that were already paid for. With ours, block 2 is the
only block that wins, which is why every block-size result on this page
looked like a dead end. The kernel that makes the verify flat is item 7 below,
which this page called "a project, not a bench" before the gap was measured.
That kernel exists in the tree as of this measurement (uncommitted at the time
of writing, in `docs/INT4_SMALL_M_MMA.md`): 8 rows at 1.25-1.29x one row on
the qwen38 shapes, against 3.1-3.6x for the shipped GEMM, behind
`TURBOSPARK_INT4_MMA8=1`. So `c(8)` is no longer the open question. What is
left, ranked by the arithmetic below:

| # | Lever | State | Why it matters |
|---|-------|-------|----------------|
| 1 | Flat verify (MMA8) | Built, opt-in, lossless, block 4 beats the default on code (50.1 vs 43.3 tok/s) and loses 9-12% on prose | Prerequisite for everything below |
| 2 | Pick the block per request or per round | DECLINED for DFlash2 under the flag: best per-prompt depth beats fixed 7 by 0-1.7% (model below). `AdaptiveDepth` is tuned for the flag-off cost curve and would sit at its floor of 2 | Under a flat verify an extra row costs about 1.3 ms and pays once its survival chance exceeds about 5% |
| 3 | DFlash2 drafter at depth 7 over the flat verify | MEASURED 2026-10-22 below: 31-37 tok/s, 2.1x its flag-off self, about 55-60% of kapitsa | Confirms the flat verify is what unlocks long blocks |
| 4 | Tree verify (KFlash) | A replay probe exists in the tree (`kflash_tree_replay_probe.rs`, another session); no kernel | Kapitsa's own doc credits it +6% code, +16% prose over its chain |
| 5 | Close the one-row gap (1.27x GEMV) | Open in the MMA8 doc | MEASURED below: it is the verify floor, about 13 ms of an 85 ms round |
| 6 | MTP head quality | RESOLVED: equal. Kapitsa 2.39 committed per cycle on short-explanation, ours 2.38, same depth 4 | The 1.4x gap at equal acceptance is round cost, 57 ms against 86 ms |
| 7 | DFlash2 drafter pass | Profiled and trimmed ~2 ms (per-row loops batched, selector scan fused), uncommitted in `dflash_draft/` | Only ~2 ms more is in sight; the selector was 1.2 ms, not the suspect |
| 8 | DFlash2 drafter attention at long context | DONE, uncommitted: one line in `dflash.rs`, draft phase 21.2 to 11.4 ms at 2,940 tokens and 14.5 to 11.0 ms at 426, +3.5% and +11% tok/s | The barrier decode kernel was used by the drafter only; the trunk already used the GQA-shared one |
| 9 | Small projections in the 8-row verify | MEASURED: tile beats the scalar GEMM 5.6x (a/b) and 2.7x (K/V) per dispatch; real verify 66.3 to 63.8 ms/token in the profile. A threshold change for the tile's owner, not made here | About 3 ms of an 85 ms round; decode must move with it, about 0.7 ms a token derived, not measured |

#### Measured: DFlash2 and the flat verify, 2026-10-22

Setup: the DFlash2 install built from `incoai/Qwen3.8-27B-DFlash2` and the
`mlx-community` trunk (`~/models/qwen38-27b-dflash2.gturbo`, 15 GB, by
`crates/repack/tests/dflash2_checkpoint_network.rs`), `turbospark-bench
--shaping greedy`, one process per run, two rounds with rotated arm order, load
under 2.5 at each start, M4 Max 36 GB on AC. Flag-off arms: clean HEAD
`2c02a5be`. Flag-on arms (`TURBOSPARK_INT4_MMA8=1`): HEAD plus a COPY of the
other session's uncommitted MMA8 changes, taken 2026-10-22 into a scratch
worktree; that code was not yet committed, so these rows describe that
snapshot. Decode tok/s, mean of the two rounds (rounds within 3%):

| Case | Arm | Flag off | Flag on | Ratio | ms/round off to on | Committed/round |
|------|-----|---------:|--------:|------:|-------------------:|----------------:|
| short | plain | 22.0 | 17.4 | 0.79x | | |
| short | MTP 2 | 32.5 | 30.5 | 0.94x | 67 to 72 | 2.18 |
| short | MTP 4 | 18.3 | 29.3 | 1.60x | 130 to 86 | 2.50 |
| short | DFlash2 2 | 31.8 | 29.6 | 0.93x | 67 to 72 | 2.13 |
| short | DFlash2 4 | 19.4 | 32.0 | 1.65x | 131 to 79 | 2.53 |
| short | DFlash2 7 | 14.6 | 31.2 | 2.14x | 177 to 83 | 2.60 |
| medium | plain | 22.1 | 17.5 | 0.79x | | |
| medium | MTP 2 | 33.9 | 31.6 | 0.93x | 68 to 73 | 2.30 |
| medium | MTP 4 | 21.4 | 31.7 | 1.48x | 129 to 87 | 2.75 |
| medium | DFlash2 2 | 34.4 | 32.7 | 0.95x | 69 to 73 | 2.38 |
| medium | DFlash2 4 | 22.6 | 36.6 | 1.62x | 131 to 81 | 2.96 |
| medium | DFlash2 7 | 17.6 | 36.8 | 2.09x | 182 to 87 | 3.20 |

Against kapitsa on the same cases (API harness above): DFlash2 7 with the
flag is 31.2 against 52.6 (59%) on short and 36.8 against 68.4 (54%) on
medium. The best flag-on arm is DFlash2 7 on both. The best flag-off arm is
block 2, which the flag makes 5-7% slower. Losslessness was NOT checked in
these runs: `turbospark-bench` does not hash output, and the flag changes
decode numerics (token counts differ between flag states, 543 against 488 on
short). The MMA8 doc's `mtp_generation_gate` runs are the lossless evidence
for MTP, and none has been run for DFlash2 under the flag.

Where a round goes (`TURBOSPARK_SPEC_STATS=1` now also prints `ms/round` per
phase; wall time, so GPU waits are inside the phase that reads back), flag on,
medium-review:

| Arm | draft | verify | accept | rollback | commit | rewind | bonus |
|-----|------:|-------:|-------:|---------:|-------:|-------:|------:|
| DFlash2 2 (3 rows) | 11.1 | 62.4 | 1.0 | 0.06 | 0.08 | 0.00 | 0.49 |
| DFlash2 7 (8 rows) | 15.2 | 66.5 | 1.7 | 0.14 | 0.01 | 0.00 | 0.49 |
| MTP 4 (5 rows) | 19.2 | 70.2 | 1.4 | 0.12 | 0.03 | 0.00 | 0.49 |

What it says:

- **Rows are free; the floor is not.** Verify goes from 3 rows to 8 for 4 ms,
  about 1.3 ms a row. What remains is a floor of about 58 ms, which is the
  one-row MMA tile (1.27x our 44.8 ms GEMV step). Kapitsa's verify is about
  1.1x its step. Closing that floor to the GEMV's efficiency is worth about
  10-13 ms of the 85 ms round, the largest single item left.
- **Rollback and commit are nothing.** 93% of block-7 rounds roll back, and the
  rollback plus commit cost 0.15 ms. The lazy GDN tape did its job, and no
  further work on rollback is justified.
- **The drafter pass is smaller than it looked.** DFlash2 costs 11 ms at 2 rows
  and 15 ms at 7. This section first guessed 6-10 ms was recoverable, with the
  host selector as the suspect. Profiling below shows the selector is 1.2 ms
  and the recoverable part is about 2 ms. MTP's draft is 19 ms for four head
  steps and was not examined.
- **Accept and bonus are 2 ms together.** Not worth touching.
- **Head quality is not the gap.** Kapitsa's MTP-only arm logged 214 cycles
  for 512 tokens on short-explanation (`NI_LOG=2`, "spec: 0.42
  accepted/drafted"), which is 2.39 committed per cycle against our 2.38 at
  the same depth 4. At equal acceptance its 41.9 tok/s against our 29.3 is a
  57 ms cycle against 86 ms. Medium-review is not comparable: the texts differ.
- **Projection from these rows, not a measurement.** If the verify floor drops
  to 48 ms and the drafter pass to 8 ms, a block-7 round is about 60 ms: 53
  tok/s on medium and 43 on short. A tree (kapitsa credits +8-17% tokens per
  pass) would add to that. Kapitsa's 68 and 53 would still need the rest.

The flat verify also has a cost the table shows: plain decode and block 2 get
5-7% and 21% slower under the flag. It only pays if the engine always
speculates at depth 4 or more, which is why the per-request block choice
(lever 2) and the one-row gap (lever 5) are linked. Do not default the flag on
until those are decided. The in-flight MMA8 wiring is uncommitted in the
shared tree (`real_forward_dispatch.rs`, the qwen `attn.rs`, `dense.rs`,
`moe.rs`, `gpu/src/lib.rs`), and the drafter work below touches none of
those.

#### Measured: the DFlash2 drafter pass, 2026-10-22

Where one draft call goes at block 7, flag on, medium-review, host load 3-5 so
read the proportions and not the absolute milliseconds (draft phase 13-17 ms
across runs). Sub-phase timers around the call, then attribution by skipping
parts of the forward behind a scratch-only env var (the playbook's method;
quality of the skipped output is irrelevant to timing):

| Part | ms per call | Note |
|------|------------:|------|
| Context write (separate pass) | 1.9-2.4 | `fc` fusion plus K/V of the accepted rows |
| Block forward, whole | 9.7-12.4 | 8 rows, one command buffer |
| of which lm_head (248,320 x 5,120 INT4) | 2.2 | about 0.7 GB, already at its bandwidth floor |
| of which per-row norm, RoPE, residual, SiLU loops | 1.1 | about 100 dispatches per layer, 65 of them per-row |
| of which attention (8 per-row dispatches a layer) | about 1.1 | not changed here |
| of which layer GEMMs and convs | about 4.6 | floor about 3 ms at the tile's 80% bandwidth |
| of which pass overhead (commit and wait) | 0.26 | nothing to recover |
| Host selector (top-16 and codebook walk) | 1.1-1.3 | 0.34 at block 2 |

Changes made, both in `crates/runtime/src/families/qwen/dflash_draft/`, both
bit-identical by construction:

- The per-row norm, residual add and SiLU loops became single dispatches
  (`rmsnorm_bf16w_perhead` with `rows` heads of width `hidden`, which
  `crates/gpu/tests/rms_norm_rows_as_heads.rs` pins bitwise against the
  whole-vector norm; the elementwise ops over `rows * width`; q/k head norms
  over `rows * heads`, with k split at the ring's wrap exactly as the
  projections already are). RoPE stays per row because its position differs.
- The selector's non-finite refusal and top-16 became one scan over each
  row (`top_k_finite`), not two. Three unit tests pin it against the old
  form, including ties and the first non-finite id.

Gates: `the_draft_logits_have_a_frozen_digest` and the other 14 tests in
`real_forward_qwen35_dflash.rs` pass flag off (15 of 15). With the MMA8 flag on
and every matrix served only the frozen digest fails, which is that flag's
documented behavior. Mutation: dropping the last element of the final residual
add fails the digest test, and the restored code passes. On the real install,
greedy block 7 on medium-review gives 264 rounds and the same per-position
acceptance before and after (0.79 0.68 0.64 0.63 0.63 0.72 0.69), so the
proposals did not change.

A/B, old and new binaries built from the same tree, four interleaved pairs
with the order alternated, load 1.4-2.5 at each start except one run:

| Case | Draft phase old | new | tok/s old | new |
|------|----------------:|----:|----------:|----:|
| medium-review | 16.5 ms | 14.2 ms | 35.5 | 37.7 |
| short-explanation | 14.0 ms | 12.1 ms | 29.3 | 30.8 |

The draft phase fell in all four medium pairs and in three of four short pairs
(the fourth had an outlier run at 87 ms verify). About 2 ms of an 85 ms round,
roughly 2.5%. The tok/s medians moved 4-6%, more than 2 ms explains, so treat
the host noise as part of that figure and quote the draft-phase delta.

#### Measured: does adaptive depth have headroom? 2026-10-23

A cost model, not a controller. Round time under the flag is a fixed part
(draft pass, accept, commit) plus about 1.3 ms per verify row beyond three,
and tokens per round are `1 + sum of cumulative acceptance`. DFlash2 always
runs its full 8-row forward, so a shorter round saves only verify rows. With
the measured conditional per-position acceptance at block 7 (flag on):

| Case | Per-position acceptance | Committed at 7 | Best depth | Gain over fixed 7 |
|------|-------------------------|---------------:|-----------:|------------------:|
| short-explanation | 0.66 0.65 0.64 0.47 0.42 0.67 0.67 | 2.61 | 5 | 1.7% |
| medium-review | 0.79 0.68 0.64 0.63 0.63 0.72 0.69 | 3.19 | 7 | 0.0% |
| long-synthesis | 0.75 0.72 0.67 0.61 0.64 0.64 0.56 | 3.16 | 7 | 0.0% |

The model predicts the measured tok/s at depths 2, 4 and 7 within 2% on both
cases where all three were run (short 29.0, 31.2, 30.7 against 29.6, 32.0,
31.2; medium 32.3, 36.1, 37.5 against 32.7, 36.6, 36.8). A row is worth
including while its cumulative survival times the cost of a token exceeds the
row's cost, which at about 26-40 ms a token and 1.3 ms a row is a survival of
3-5%. Code has higher acceptance than any of these, so it only pushes the
answer toward 7. Verdict: no controller. A fixed block 7 with the flag is
within 2% of the best per-request choice on every case measured, and the
existing `AdaptiveDepth` (floor 2, 4 to 8 clean rounds to climb) would do
worse than fixed 7 here. What the flag changes is the whole premise: flag off,
block 2 wins and depth matters; flag on, block 7 wins and depth does not.

New finding from the same run, CORRECTED below: the DFlash2 draft phase grew
with context. This paragraph first reported 29.6 ms at a 2,940-token prompt
from the bench's first generation pass on a loaded host; the measured second
pass reads 21.2 ms, against 14.5 ms at 426 tokens. Cause and fix are in the
next subsection.

#### Measured: GQA-shared attention in the drafter, 2026-10-23

The drafter's `AttentionScratch` was built with the default barrier decode
kernel, while the trunk's scratch for this family already uses the GQA-shared
one. The drafter runs `block + 1` one-row attention dispatches per layer over
a 2,048-token ring, and the barrier kernel's cost grows with that span. The fix
is `.with_decode_kernel(GqaShared)` on the drafter's scratch in
`families/qwen/dflash.rs` (the dispatcher already falls back to barrier for a
shape it does not serve; the drafter's, head_dim 128 with 4 query heads per KV
head, is served). A/B with `TURBOSPARK_ATTN_DECODE_KERNEL=gqa` against unset
(an earlier arm that forced `barrier` also slowed the trunk's verify and is not
the baseline), flag on, DFlash2 block 7, HEAD plus the drafter edits above:

| Prompt | Draft phase before | after | tok/s before | after |
|--------|-------------------:|------:|-------------:|------:|
| 426 tokens (medium-review), 3 pairs | 14.2 / 15.5 / 14.5 ms | 11.0 / 11.4 / 11.1 ms | 38.9 / 36.1 / 37.9 | 39.2 / 39.1 / 39.2 |
| 2,940 tokens (long-synthesis), 1 pair | 21.2 ms | 11.4 ms | 33.8 | 37.6 |

Acceptance was identical in every pair (medium: 264 rounds, 2.20 accepted,
same per-position curve; long: 202 rounds, 2.16 accepted, same curve), and the
token counts matched (845 and 639). Verify was unchanged within noise (67-70
ms), so the saving is all in the drafter. With the change compiled in, with no
env override, medium reads draft 11.0 ms and 38.8 tok/s.

What this does and does not prove. The kernel is not bit-identical to the
barrier kernel, so proposals can differ in principle; on both real prompts
the acceptance statistics matched exactly, which is strong but not a proof of
identical proposals. The committed text is the target's regardless, since
verification is unchanged. The synthetic DFlash2 suite (15 of 15, including the
frozen draft-logits digest) does NOT exercise this change, because its shapes
fall back to the barrier kernel. One long pair is one pair; the long-context
gain should be re-measured with more rounds before it is quoted.

What is left in the drafter, in order: the layer GEMMs (4.6 ms against a 3 ms
floor, mostly the one-row tile gap again, which lever 5 owns), a rows-batched
attention kernel in place of 8 one-row dispatches per layer (the GQA-shared
kernel is now in; the multi-row variant is untried and shares chunk boundaries
across rows), and merging the context write into the forward's command buffer
(one commit and wait fewer). Together perhaps another 1.5 ms. The drafter is
not where kapitsa's lead is: the verify floor is.

#### Measured: where the 8-row verify goes, and the small projections, 2026-10-23

Dispatch profile of one batched verify (DFlash2 block 7, 8 rows, 8-row tile
flag on, short-explanation, 209 passes). The profiler gives each dispatch its
own encoder and waits on it, so absolute times are inflated and only the
ranking is meant:

| Kernel | Dispatches | ms | Share |
|--------|-----------:|---:|------:|
| `dequant_int4_mma8_gemm` (the large projections) | 369 | 47.5 | 71.6% |
| `dequant_int4_gemm_simd` (the small ones, scalar) | 128 | 6.1 | 9.2% |
| `gdn_delta_step_prefill` | 48 | 4.0 | 6.0% |
| `dequant_int4_mma8_table` (activation prep) | 369 | 2.0 | 3.0% |
| `attention_decode_gqa_partial` plus combine | 256 | 2.3 | 3.5% |
| `rmsnorm_bf16w_perhead` | 161 | 1.7 | 2.5% |
| `copy_strided_rows_fp16` (aux capture) | 197 | 0.9 | 1.4% |
| the rest | | 1.8 | 2.8% |
| total | 1,936 | 66.3 | |

The scalar GEMM's 128 dispatches are the attention K/V projections (1024 x
5120, 32 of them) and the GDN `in_proj_a`/`in_proj_b` (48 x 5120, 96). The
tile is routed to projections with at least 4096 output rows only
(`int4_mma8_takes`), because at ONE row it loses to the GEMV by ratio. Their
weights total about 95 MB, under 0.3 ms of bandwidth, so the 6.1 ms is launch
and occupancy latency. Per-dispatch GPU time on the real shapes
(`crates/gpu/tests/small_n_gemm_bench.rs`, a pool of 64 distinct weight copies
per shape, median of 5 rounds, in the tree that carries the tile):

| Shape | GEMV, 1 row | 8 GEMVs | Shipped GEMM, 8 rows | Tile, 8 rows |
|-------|------------:|--------:|---------------------:|-------------:|
| GDN a/b 48 x 5120 | 3.8 us | 29.9 us | 46.2 us | 8.2 us |
| attn K/V 1024 x 5120 | 11.9 us | 78.4 us | 48.7 us | 17.9 us |
| o_proj 5120 x 6144 (reference) | 57.5 us | 369.9 us | 177.7 us | 70.5 us |

At eight rows the tile is 5.6x and 2.7x faster than the shipped GEMM on the
two small shapes. Over a verify's 96 and 32 dispatches that is 6.0 ms down to
1.4 ms, a predicted saving of 4.6 ms.

On the real model, `TURBOSPARK_INT4_MMA8_MIN_ROWS=1` (an existing knob, read
once) serves every projection by the tile in decode and verify alike. Its
dispatch profile on the same prompt reads the batched verify at **63.8 ms
against 66.3**, with the scalar GEMM gone from the list. That is 2.5 ms, less
than the microbench predicts, because the tile's activation-prep dispatches
rise (369 to 492, 2.0 to 3.0 ms) and its own time on the large shapes does
too. One unprofiled quiet-host run of the knob read verify 63.3 ms against
66-68, in line with that. Honest limits:

- The default-threshold arm of this profile pair timed out in a host stall, so
  its baseline is the earlier default profile of the same prompt, taken at a
  different time. The accepted counts differ (205 and 209 rounds) because the
  knob changes decode numerics and so the token stream.
- The interleaved wall-clock pairs were not usable: swap was 7-8 of 8 GB, one
  run took 22 minutes, and another session's probe was running. Treat the
  3 ms as GPU-side, about 3.5% of an 85 ms round.
- Plain decode under the knob was NOT measured (one of four runs finished, at
  15.3 tok/s against about 17.4 on a quiet host, which is host noise). The
  derived cost is the one-row tile against the GEMV on these shapes, about
  4.4 us x 96 plus about 9 us x 32, which is **about 0.7 ms a token**. That
  figure is derived, not measured.

The knob must move decode with verify: the speculative stream is only
bit-identical to sequential decode while both run the same arithmetic on every
matrix. So this is a change to the default threshold, to be made by whoever
promotes the tile, with the usual gates (`mtp_generation_gate` lossless at
several blocks, the quality gate re-frozen, a decode A/B on a quiet host). It
lives in `real_forward_dispatch.rs`, which carries that session's uncommitted
wiring, so it is left to them and not edited here. Not promoted as a default
by this section.

Other items the same profile ranks, in rough order of size: the GDN delta
kernel (4.0 ms over 48 dispatches), the tile's activation prep (2-3 ms, and
the same input feeds q/k/v and gate/up so one prep per input would save part
of it), verify attention (2.3 ms over 8 one-row dispatches a layer, which the
rows-batched kernel in the playbook's Next work item 1 targets), the per-head
norms (1.7 ms) and the aux capture copies (0.9 ms).

#### Measured: the three remaining ideas, 2026-10-23

**Hybrid prompt lookup with a drafter: declined, with a measurement.** The
copy-heavy prompt (rename a variable in a 25-line function), served through
`turbospark-server` with DFlash2 at block 7 and the 8-row tile on, ran at
**87.3 tok/s** with output byte-identical to the flag-off runs of the same
prompt (hash `b99aac05c37b`, 187 tokens, one request, so a direction and not a
rate). Its stats: 24 rounds, **6.83 accepted and 7.83 committed per round**,
acceptance 1.00 at all seven positions, one rollback. The round ceiling at
block 7 is 8. The lookup replay (`pld_replay_probe.rs`) reached at most 5.3
committed per round on the same prompt at its longest draft. A hybrid can only
add where the drafter falls short, and here it commits 98% of the ceiling, so
it has nothing to add; the limit is the block, not the draft source. (The same
server run on short-explanation read 28.6 tok/s at 2.65 committed per round,
where lookup also finds nothing.) A longer block is a different question:
kapitsa's own experiment with 16 and 32 row verifies on this model found 6.0
tokens per cycle at 32 rows costing 3.3x the cycle time of 8 rows, so it does
not pay either.

**Verify attention, rows-batched kernel: not pursued; the prize is small and
the constraint is expensive.** Verify attention is 2.3 ms of the 66 ms pass
(128 `attention_decode_gqa_partial` dispatches plus 128 combines) on a short
prompt, and it grows only 3-4 ms from a 426-token to a 2,940-token prompt
(verify 66-68 to 70.3 ms), so even halving it saves 1-2 ms. The rows kernel
shares chunk boundaries across rows and is not bit-identical to the one-row
kernel that decode runs, so using it in verify would require decode to use it
too (a batch of one, 0.89-1.07x the one-row cost) and a re-freeze of every
digest that family pins. That is a numerics change for 1-2% of a round, in
files (`families/qwen/attn.rs` and the dispatch layer) that carry another
session's uncommitted wiring. The playbook's Next work item 1 already records
the route and a bit-exact variant that measured 1.06-1.22x for about 2% of a
round; nothing here changes that verdict.

**Sharing the tile's activation prep: small, and in the tile owner's files.**
The profile counts 369 `dequant_int4_mma8_table` dispatches (2.0 ms,
3% of the pass). Two shared inputs account for 112 of them: gate and up read the
same normed input on all 64 layers (64 preps saved), and the GDN qkv and z
projections read the same input on 48 layers (48 saved). That is 30% of the
dispatches, about 0.6 ms in the inflating profiler and nearer 0.3-0.5 ms in a
real pass, around half a percent of a round. The larger form, writing the
fragment-ordered table directly from the producing norm or SiLU so no prep
dispatch is needed, would remove all 2.0 ms but changes those producers. Both
live in `dequant_int4_mma8.rs` and `real_forward_dispatch.rs`, so they are left
to that work. These two sizings come from the profile's dispatch counts and
times, not from experiments.

What is left is the verify floor: the 8-row tile's own efficiency (the large
projections are 71.6% of the pass) and the three small rows above.

### Prompt lookup

Prior measurement, so this is not a fresh idea: `accept_length_probe.rs`
(`docs/SPECULATIVE_DECODING.md`, "Accept length") ran a 3-gram lookup drafter
on the Qwen 3.6 35B-A3B MoE install, greedy, 300 tokens per arm. It fired on
10-11% of rounds and accepted 2.46 / 2.76 / 2.76 tokens (bonus included) at
blocks 4 / 8 / 16, saturating because the matched continuation ends before
the block does. That refuted lookup as a general free drafter. What it did not
cover: the dense 27B, the lazy-tape verify that now makes a partial round
cheaper, and a copy-heavy prompt, which is the only place lookup is expected
to pay. The new work is scoped to that gap, and "no match means no verify"
(step 3) is what keeps the 10% firing rate from costing the other 90%.

Plan, nothing measured yet. Prompt-lookup decoding proposes the tokens that
followed the most recent earlier occurrence of the current n-gram suffix in
the prompt plus the generated text. It costs no weights and no drafter pass,
and it pays on copy-heavy output (code edits, quoting, repeated tool
transcripts) and not on novel prose.

Order of work, each step gating the next:

1. `crates/runtime/src/prompt_lookup.rs`: a pure matcher with unit tests.
2. `crates/bench/tests/pld_replay_probe.rs`: replay the greedy streams of the
   protocol cases and count, per position, whether a match exists and how many
   proposed tokens the stream would have accepted. Weigh that against the
   `c(M)` break-even in `docs/SPECULATIVE_DECODING.md`. No GPU, no kernels.
3. Only if the replay beats break-even on a copy-heavy case: a greedy-only
   `Lookup` drafter. A round with no match must skip the verify and run one
   sequential step, so novel prose pays nothing. Sampled runs refuse until a
   point-mass proposal distribution exists (accept when `r <= p(x)`, residual
   is `p` with `x` zeroed). `draft_qs` is indexed per proposal in the sampled
   accept loop, so a block-style drafter must not reach it silently.

#### Measured: replay on the dense 27B, 2026-10-21

`crates/bench/tests/pld_replay_probe.rs`, `qwen38-27b.gturbo`, greedy, 512
tokens per case, host busy (load ~9-10, swap nearly full; irrelevant to
acceptance, which is read off the reference stream, but it makes this a
ceiling and not a timing). Cost per round is the interpolated `c(M)` from the
`docs/MTP.md` break-even column with no rollback term, so the speedups below
are OPTIMISTIC ceilings. A wrong match is charged for its verify.

| Case | n-gram | draft | fired | accepted/fired | ceiling |
|------|-------:|------:|------:|---------------:|--------:|
| short-explanation | 2 | 2 | 8% | 0.32 | 0.970x |
| short-explanation | 4 | 2 | 0% | 0.50 | 0.999x |
| medium-review | 2 | 2 | 17% | 0.43 | 0.959x |
| medium-review | 4 | 2 | 2% | 1.25 | 1.013x |
| long-synthesis | 2 | 4 | 23% | 0.44 | 0.771x |
| long-synthesis | 4 | 2 | 4% | 1.06 | 1.012x |
| copy-heavy (rename in a 25-line function) | 2 | 2 | 96% | 1.59 | 1.504x |
| copy-heavy | 3 | 2 | 90% | 1.73 | 1.562x |
| copy-heavy | 4 | 2 | 85% | 1.80 | 1.575x |
| copy-heavy | 4 | 8 | 74% | 4.34 | 1.449x |

Full grid (n-gram 2/3/4, draft 2/4/8) is in the probe's output. What it says:

- On the three protocol prose cases lookup is flat at best (n-gram 4, draft 2:
  0.999-1.013x) and loses everywhere a short n-gram fires (0.62-0.97x). Short
  n-grams match often and are wrong, and a wrong match pays a full verify.
  "No match, no verify" does not rescue prose: the loss comes from matches
  that fail, not from missing ones.
- On a copy-heavy case it reaches 1.33-1.58x, best at a short draft (2) and a
  long n-gram (4). Longer drafts lose ground because the copied span ends.
- That is about what MTP block 2 already gives on EVERYTHING (1.44x measured,
  `docs/MTP.md`). A standalone lookup drafter therefore does not beat the
  shipped default on either kind of text, and the install it needs a verify
  scratch from is the MTP install anyway.

Verdict: do not build a standalone `Lookup` drafter. The one open question is
a HYBRID: when a long n-gram (4+) matches, propose the copied span (draft 4-8
accepted 3-4 tokens per round on the copy case) instead of the MTP head's two
tokens, and otherwise fall back to MTP. That needs an MTP replay on the same
copy-heavy stream to price, because its ceiling over MTP's 1.44x is the only
number that matters. Not measured.

Gates: greedy output identical to the sequential loop at several draft
lengths (`mtp_generation_gate` shape), the memory oracle unchanged, and an
interleaved paired A/B on AC power. A negative result is recorded here.

## Item 1: the lazy rank-1 GDN verify tape

Adopted, default on, `TURBOSPARK_GDN_LAZY_TAPE=0` restores the eager path.

### What changed

The speculative verify used to advance every linear layer's gated-DeltaNet
state in place, which forced the round to pay for being speculative: a
snapshot blit before every verify, and on a partially-accepted round a
BLOCKING restore (`GdnStateManager::restore` waits) plus a kernel replay of
the accepted prefix over the verify tape. The lazy tape makes the verify a
recording instead:

- `gdn_delta_step_prefill` gained function constants 96/97
  (RECORD/WRITE_STATE, both in the pipeline cache key). The lazy build
  (true/false) writes each row's `delta` -- the beta-scaled rank-1
  coefficient, already simd-reduced -- to a tape and leaves the live state at
  the block start. y is bit-identical in both builds: nothing after the
  register load reads the state back.
- A strided copy (`encode_copy_rows_fp16_strided`, the dflash row-copy kernel
  with its source stride exposed) captures the post-qk-norm `k` slice per
  row. The conv-tail update is simply skipped on the lazy path.
- New `gdn_state_fold` kernel: per row, `s *= g` then `s = fma(k, delta, s)`
  -- the same element-wise sequence the eager kernel runs in registers, with
  `g` recomputed from taped `a` by the same expression and `k`/`delta` read
  back as the exact recorded values. No cross-lane reduction anywhere in the
  state update, so it is bit-identical to the discarded register state, which
  the gpu-side parity tests pin (`crates/gpu/tests/gdn_lazy_fold.rs`).
- The round end commits: `SpeculativeProducer::commit_pending(keep_rows)`
  runs `gdn_conv_tail_update` over the kept taped rows (the same kernel the
  replay used) plus the fold, per linear layer -- two small dispatches. The
  speculative loop calls it EVERY round (full acceptance included, which the
  eager path served for free); a partial round's `rollback_retaining` folds
  the same way, replacing restore-plus-replay.
- The speculative loop takes a ROUND checkpoint
  (`RealForwardRunner::checkpoint_round`, what the `SpeculativeProducer`
  `checkpoint` calls) that skips the snapshot while the live state is clean
  (`gdn_state_clean`: a fold, restore or reset just ran, no eager advance
  since). Every speculative round after the first qualifies, so the ~150 MiB
  blit is gone from them. The generic `checkpoint()` ALWAYS snapshots, so
  callers holding several points (`HiddenTrunk` prefix snapshots, the bench
  probes) keep the full contract. The skip stays general because both ways
  the state can leave the point are covered: an eager advance
  (`note_eager_advance`, called before every `produce`, prefill pass and
  hidden-row step) first takes the snapshot the checkpoint skipped
  (`deferred_gdn_snapshot`), and a fold or other restore that moves the state
  past the point taints it, after which a rollback to it panics and a
  retaining rollback refuses. Only one round checkpoint is tracked; a newer
  one replaces it.
- Trunk passes (`produce`, chunked prefill) refuse by name if a lazy tape is
  still uncommitted. The eager path could silently discard a tape; the lazy
  path must not, because its state is still waiting for the fold.

### Why not the alternatives

- Fold-by-replay (re-running the conv and delta kernels over the kept rows,
  without the restore) is bit-exact and needs no new kernel. But it re-runs
  the whole delta pass on every full-acceptance round to save a blit.
  Full acceptance is 2-91% of rounds depending on workload and block
  (the complement of the 9-98% rollback rates in `docs/DFLASH2.md`). Net
  negative.
- Fold-inside-next-verify (kapitsa's shape) hides the fold latency but makes
  every state reader order-dependent. The eager fold is two dispatches and
  keeps the invariant trivial: nothing outside the round-end commit ever
  sees pending state.
- Skipping the snapshot unconditionally was tried and is WRONG: a plain
  `rollback()` after an eager `produce` still needs the snapshot.
- Skipping it in the generic `checkpoint()` whenever the state is clean, and
  tainting the point on an eager advance, was the first design and is ALSO
  wrong: a clean state is the normal state of a fresh or just-restored trunk,
  so `HiddenTrunk::snapshot` -> `step` -> `restore` panicked. The skip now
  lives only in the round checkpoint, and an eager advance takes the deferred
  snapshot instead of tainting.

### Evidence

- `crates/gpu/tests/gdn_lazy_fold.rs` pins three properties on Metal: lazy y
  is bit-identical to eager; the state is untouched between record and
  fold; and fold(N rows) matches the eager write-back after the same N rows
  at full and partial counts.
- The dflash and MTP suites (`retaining_rollback_matches_rollback_plus_reverify`,
  `retaining_rollback_matches_at_a_shrunk_round`,
  `a_batched_pass_is_bit_identical_to_the_same_tokens_run_sequentially`,
  `speculative_rounds_commit_the_sequential_stream`) pass byte-identically
  under BOTH flag values on the synthetic GDN installs.
- The round checkpoint's contract is pinned in
  `real_forward_qwen35_dflash.rs`
  (`a_round_checkpoint_stays_a_general_restore_point`: eager steps then
  rollback replay bit-identically;
  `a_round_checkpoint_is_reusable_before_commit_and_validates_safely`;
  `pending_tape_on_error_can_be_discarded_cleanly`; and
  `rolling_back_past_a_committed_round_refuses_a_snapshotless_checkpoint`)
  and in `hidden_trunk.rs`
  (`a_snapshot_taken_on_a_clean_trunk_restores_after_steps`). The first was
  mutation-checked: with the deferred snapshot disabled it fails.
- `mtp_generation_gate` (speculative greedy output identical to the
  sequential loop, `qwen38-27b-mtp`, release, lazy tape and fused FFN at
  their defaults): "lossless at every block" at blocks 2 and 4.
- Real-model paired A/B, MEASURED 2026-10-20 (below). Not a frozen row.

#### Measured: lazy tape and fused FFN, `qwen38-27b-mtp`, 2026-10-20

Apple M4 Max 36 GB, macOS 26.5.2, AC power, release build.
`turbospark-bench --model ~/models/qwen38-27b-mtp.gturbo --speculative auto
--shaping greedy` (MTP head, block 2), 4,096 context. One process per run
because both flags are read once. Arm order rotated each round, a warmup
round discarded. Median decode tok/s with the min-max spread; ratios are
against the pre-Kapitsa arm (lazy off, fused off).

Run 1 (all six arms, 5 measured rounds, 1-minute host load 0.34-2.73). The
binary was built from the working tree before the Kapitsa commit settled
(`90d74bd6` to `e81cba38`); gpu, runtime and bench sources had no uncommitted
changes at the later check, but byte-for-byte identity with HEAD was not
verified. Treat it as corroboration. The sequential-decode rows exist only
here.

| Arm (lazy, fused) | short-explanation (488 tok) | medium-review (845 tok) |
|-------------------|-----------------------------|-------------------------|
| off, off | 31.47 [31.26-31.70] | 33.72 [33.57-34.43] |
| on, off | 32.49 [32.33-32.76], 1.032x | 33.89 [33.75-35.02], 1.005x |
| off, on | 31.63 [31.35-31.93], 1.005x | 33.22 [32.62-33.89], 0.985x |
| on, on (default) | 32.13 [31.65-32.75], 1.021x | 33.58 [33.00-34.26], 0.996x |

Run 2 (confirmatory, built from clean HEAD `e81cba38`, no uncommitted
changes in gpu, runtime or bench; 3 measured rounds, load 0.51-2.39). This
is the run of record for the speculative arms.

| Arm (lazy, fused) | short-explanation | medium-review |
|-------------------|-------------------|---------------|
| off, off | 31.82 [31.49-31.90] | 33.89 [33.81-34.00] |
| on, off | 32.68 [31.89-32.75], 1.027x | 34.70 [34.53-34.82], 1.024x |
| on, on (default) | 32.61 [32.09-32.83], 1.025x | 34.66 [34.62-34.75], 1.023x |

Sequential decode (speculation off, run 1 only; fused FFN):

| fused FFN | short-explanation | medium-review |
|-----------|-------------------|---------------|
| off | 21.22 [21.06-21.79] | 21.41 [21.16-21.79] |
| on | 21.69 [21.47-21.71], 1.022x | 21.70 [21.65-21.78], 1.014x |

Every speculative arm produced the same token count per case (488, 845) and
peak footprint varied by under 10 MiB (860-868 MiB). The speculation-off arms
emit 505 and 846 tokens: the two paths decode from slightly different states
(`docs/BENCHMARKS.md`), so text is never compared across that line, and
`mtp_generation_gate` is the lossless evidence.

What it says:

- The lazy tape is worth about 2-3% in speculative decode. Run 2 shows it on
  both cases (1.027x, 1.024x) and run 1 agrees on the short case and is
  noise on the medium one. The motivation (no snapshot blit, no blocking
  restore, no replay) holds, but at block 2 on these prompts rollbacks are
  rare enough that it buys a few percent and no more.
- The fused FFN does nothing in speculative mode: lazy-on with and without
  it differ by 0.1-0.2% in run 2, and run 1 straddles 1.0.
- The fused FFN is worth 1.4-2.2% on plain sequential decode. Small, but
  consistent in sign and a little above the spread.
- Not measured: rollback counts and ms/round (the bench prints neither),
  blocks above 2, the long-synthesis case, a rollback-heavy workload (where
  the lazy tape should matter most), and the MoE shared-expert fusion (this
  is a dense install).

### Round checkpoint and error boundaries

- A round checkpoint (`checkpoint_round`) skips the GDN snapshot when the
  live state is clean and stays reusable for multiple rollbacks before a
  commit occurs. A commit past it folds the state forward and taints it;
  callers can check `is_valid_rollback_point` or use `try_rollback` to
  refuse gracefully rather than panic.
- An error between `verify` and `commit_pending` in `run_speculative` (a
  `select` failure, for example) cleans up through `discard_pending()`,
  rewinding uncommitted KV rows and clearing the lazy tape so subsequent
  trunk passes proceed without poisoning the runner.

## Items 4 and 5: bit-exact fused FFN dispatches

ADOPTED behind `TURBOSPARK_QWEN_FUSED_FFN` (default on, `=0` opts out).

Design: `shaders/dequant_int4_fused.metal` holds two kernels.
`dequant_int4_gate_up_act_gemv_simd` computes the gate AND up rows of one
output index in a single simdgroup over the same x, reusing
`dequant_int4_gemv_simd_body`'s exact per-output accumulation order, and
rounds each accumulator to half BEFORE the activation reads it.
`dequant_int4_down_epi_gemv_simd` is the plain GEMV with a store epilogue:
residual add for the dense tail, or the shared expert's scalar gate in
`sigmoid_scalar_mul_fp16`'s division form. Wiring: the dense FFN drops from
5 dispatches to 2 (`families/qwen/dense.rs`), the MoE shared expert from 6
to 3 (`families/qwen/moe.rs`, with the logit GEMV reordered before down).
Only INT4 weights take the fused branch; hadamard-folded layers and other
dtypes keep the per-tensor chain. Both kernels derive every row stride from
the BAKED `(M, N)` constants (the first cut of the down kernel read the
runtime `N` and silently lost the shape-baking gain; bit-exactness hid it).

Measured so far: gpu parity bit-identical at every fixture shape for silu
and gelu, all three epilogues, and the plain store; the synthetic dflash
and MTP suites (including `the_draft_logits_have_a_frozen_digest` and the
batched-vs-sequential byte-identity test) pass under all four
flag combinations with item 1. Real-model paired A/B: MEASURED 2026-10-20 on
the dense `qwen38-27b-mtp` install (tables under item 1's evidence): +1.4% to
+2.2% on sequential decode, no measurable change in speculative mode. The MoE
shared-expert fusion has no real-model A/B yet. The dispatch-count deltas are
static facts: -3 per dense layer, -3 per MoE layer's shared expert.

One find worth its own paragraph: Metal's default fast-math ELIDES a
f32->f16->f32 convert pair it can see through, so `float(half(acc))` in the
epilogue silently kept the unrounded accumulator. The error is invisible
except at an exact f16 tie, where it flips the rounding by 1 ulp -- which is
exactly where the parity test caught it (`dequant_int4_fused_parity.rs`,
down-residual arm). The kernels now force the round-trip through `volatile`
half locals, and both fusions are bit-identical to the unfused chains at
every fixture shape for silu and gelu, all three epilogues, and the plain
store. Any future in-kernel consumer of a rounded half needs the same
treatment.

## Item 2: opt-in concurrent compute encoder

Plumbing landed, production wiring deferred. `MetalContext::
begin_pass_concurrent_labeled` opens a `MTLDispatchType::Concurrent`
encoder; `PassEncoder::memory_barrier_with_buffers` (existing) separates
dependency groups. `tests/concurrent_pass_semantics.rs` proves the
contract on Metal: a barriered concurrent chain is bit-identical to the
serial pass, and eight independent dispatches in one concurrent pass stay
bit-identical to serial.

Nothing flips this at runtime, so there is no flag yet. The wiring that
would use it (the decode path's q/k/v projection triple is the natural
first site; the audit list is the in-place qk-norm/rope on `batched.q`,
the `h1`/`h2`/`x` chains, and the batched scratch liveness notes in
`batched_layers.rs`) waits on the same A/B that item 1's bench waits on:
our own prior bench measured only 1.07-1.40x for same-weight GEMV
concurrency, so the win has to be measured before the serial-order audit
is worth paying for.

## Items 6, 7, 8: bench prototypes

Each follows the `dequant_int4_mma.metal` protocol: a prototype kept out of
the production library, an `#[ignore]`d release bench whose ratio is the
deliverable, and a bit-identity gate where the idea claims exactness.
Ratios below are indicative only until an AC-power paired run; absolutes
move with clocks.

- Item 6 (shift-free nibble masking): MEASURED DEAD END, 2026-10-19.
  `tests/gemv_mask_dequant_bench.rs`. The exactness claim held -- the
  masked rewrite is bit-identical (the 16x mask factor cancels against x
  prescaled by 1/16, both exact powers of two). The speed claim did not:
  masked/scalar ran 1.17-1.60x SLOWER across the qwen38 shapes over 5
  interleaved in-process rounds (mean 1.44), monotonically worse on the
  smaller shapes where the shifts matter most. The doubled x loads
  (original plus prescaled copy) cost more than four shifts per lane per
  block save. Nothing to revert: the kernel never left the bench. Recorded
  in ROADMAP's Do Not Revisit list.
- Item 7 (integer-code 8-row verify GEMM): NOT BUILT, and the in-tree
  evidence reframes it. `dequant_int4_mma.metal`'s header records that its
  OWN tile lost at every width because packed INT4 cannot be
  simdgroup-loaded, so every element pays a dequant into threadgroup memory
  that is independent of B -- and the header explicitly scopes the verdict:
  the shape loses, not matrix hardware, and the missing piece is weights
  "already in a loadable format". Kapitsa's subnormal-code trick IS that
  missing piece: codes repacked so the raw bytes ARE half values
  simdgroup_load can consume. That requires a new weight layout at 2 or
  more bytes per code (4x the bytes of packed INT4), which trades dequant
  work for bandwidth -- plausibly a win at M=8 where the batch GEMM's c(M)
  of 0.41-0.46 says the kernel is not bandwidth-bound, and unmeasured
  either way. Building it means a new repack format feeding a new kernel;
  that is a project, not a bench, and it starts from
  `scripts/mlx_qmm_reference.py`'s shape (stage x, widen the threadgroup)
  before it borrows kapitsa's encoding.
- Item 8 (expert-sorted grouped GEMM MoE prefill): MECHANISM POSITIVE,
  not promoted. `tests/moe_grouped_gemm_bench.rs` measures one M-row GEMM
  per expert against one GEMV per route instance, same packed weights,
  same routes served: gemm/gemv = 0.44-0.80 over five interleaved rounds
  (best 0.44 at the wide 17408x5120 shape, 0.58 at g=4 on the routed
  shape). Reading each expert's weights once instead of once per route is
  worth 1.25-2.3x at the kernel level. NOT promoted: the bench does not
  model the route permutation, the row gather, the combine order, or the
  reduction-order drift bound (1e-2 nats mean KL, top-1 >= 50%, per
  `crates/runtime/AGENTS.md` and `docs/BATCHED_PREFILL.md`). An
  integration starts from `moe_prefill_batch.metal` and needs the paired
  A/B this page's other rows wait on.

The item 6 and item 8 ratios are INDICATIVE: one process, interleaved arms,
on battery power with a busy host. They support adopt/skip decisions, not
frozen rows; an AC-power rerun precedes any promotion. Item 7 was not built,
so it has no ratio.
