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
kernel one, and is not covered here. Two speculation ideas are the exception,
prompt lookup and KFlash tree verify, because their results decide whether
the verify kernel items (2 and 7) are worth building.

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
| 7 | Integer-code 8-row verify GEMM | Kernel: 8 rows cost 1.25-1.29x a 1-row GEMV (shipped GEMM 3.1-3.6x). Wired opt-in (`TURBOSPARK_INT4_MMA8=1`), lossless. End to end +16% on code (block 4), 9-12% behind the default on prose | Item 7 below, `docs/INT4_SMALL_M_MMA.md` |
| 8 | Expert-sorted grouped GEMM MoE prefill | Bench prototype | Item 8 below |
| 9 | Fused GDN prep (conv+SiLU+qk-norm+g/beta) | Not adopted: <= 1% decode/verify est., breaks the `(32,4)` delta layout contract | declined |
| 10 | Host-side FCV/set-bytes dedup in MoE decode | Was already in flight in this tree | not this page |
| 11 | GPU-side routing + single command buffer per token | Not adopted: `fast::exp` vs f32 `exp` can flip f16 routing weights and break frozen digests; streamed mode cannot use it | declined |
| 12 | Prompt-lookup (n-gram) draft source from mlx-serve | Standalone drafter declined on a replay (flat or losing on prose, ~1.5x ceiling on copy-heavy, about MTP's 1.44x). Hybrid with MTP unmeasured | Section "Prompt lookup" below |
| 13 | Adaptive speculation depth in the server | Server routing (`--speculative adaptive[:N]`) and a tunable controller. MTP replay: flag off, no setting beats fixed block 2 (live check agrees). Flag on, a tuned 2..8 controller is 1.10x fixed 2 (modeled, not yet run live) | Lever 2 below |
| 14 | KFlash tree verify (best-first candidate tree in the verify rows) | Not adopted for MTP: a replay gives 0.995-1.009x committed/round at block 2, the only block that pays. DFlash2 replay at depth 7, the block it serves over the flat verify: 1.07-1.12x committed/round. The tree's overhead is estimated at about 1% of a round, so the projection is about 1.06-1.11x tok/s. Nothing is built. Rerun with `tslab tree-replay` | Section "KFlash: tree verify" below |

The measured rows on this page are also recorded as findings in the lab's
family profile, `crates/lab/families/qwen35.json`, each with the heading it
came from and its conditions (`tslab family --family qwen35 --findings`).
Where a ledger id exists the finding carries that instead of the heading.
The profile is the place an agent reads before re-proposing an item above.

## What makes the headline numbers fast, and what applies here

A later research report compared kapitsa and mlx-serve on Qwen3.8-27B. This
section keeps the part that survived checking it against this tree. The
report's sources are the two projects' READMEs and release notes, which were
not re-read for this page. mlx-serve's source, kernels, and measured gotchas
were read on 24 October 2026 for [MLX_SERVE_KERNELS.md](MLX_SERVE_KERNELS.md),
which supersedes this section's mlx-serve claims where they differ.

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
| 3 | DFlash2 drafter at depth 7 over the flat verify | MEASURED 2026-10-22 below: 31-37 tok/s, 2.1x its flag-off self, about 55-60% of kapitsa. LOSSLESS 2026-10-25: `dflash_generation_gate` matches sequential decode at blocks 2 and 7, chain and tree, under `TURBOSPARK_INT4_MMA8=1` | Confirms the flat verify is what unlocks long blocks |
| 4 | Tree verify (KFlash) | ADOPTED 2026-10-26, default on, `TURBOSPARK_DFLASH_TREE=0` keeps the chain, lossless. Default-config A/B (`dflash-tree-default-d1`): WIN +9.2% on medium-review, WIN +4.3% on short-explanation, peak footprint +0.8 MB. Quiet-host A/B 2026-10-26 (`kapitsa-quiet-c1`, 3 clean pairs each): WIN +8.6% decode tok/s on medium-review, WIN +3.1% on short-explanation. The 5-7% verify overhead read on 2026-10-25 was host contention: quiet, the tree round costs about 1.1 ms (1.5%) more than the chain's | "KFlash: tree verify", "Built and measured" |
| 5 | Close the one-row gap (1.27x GEMV) | DECLINED 2026-10-25: the one untried design, a two-pass split-K (no barrier tail, KS up to 32), is bit-exact at the shipped KS but 1.6-28% slower on four of five shapes; only `down` gains 1.5-5.1% (`docs/INT4_SMALL_M_MMA.md`, open question 10) | MEASURED below: it is the verify floor, about 13 ms of an 85 ms round. The floor is weight streaming, not the reduction tail |
| 6 | MTP head quality | RESOLVED: equal. Kapitsa 2.39 committed per cycle on short-explanation, ours 2.38, same depth 4 | The 1.4x gap at equal acceptance is round cost, 57 ms against 86 ms |
| 7 | DFlash2 drafter pass | Profiled and trimmed ~2 ms (per-row loops batched, selector scan fused), committed in `9abbdcf0` (`dflash_draft/`) | Only ~2 ms more is in sight; the selector was 1.2 ms, not the suspect |
| 8 | DFlash2 drafter attention at long context | DONE, committed (one line in `dflash.rs`, the drafter's `with_decode_kernel(GqaShared)`), draft phase 21.2 to 11.4 ms at 2,940 tokens and 14.5 to 11.0 ms at 426, +3.5% and +11% tok/s | The barrier decode kernel was used by the drafter only; the trunk already used the GQA-shared one |
| 9 | Small projections in the 8-row verify | NOT CONFIRMED. Per dispatch the tile is 2.7-5.6x faster than the scalar GEMM and the profile read the verify at 66.3 to 63.8 ms/token, but the clean two-round wall-clock sweep read verify 75.3 and 72.0 ms with the threshold at 1 against 69.8 and 70.5 at its default | Leave the threshold where the tile's owner put it; the disagreement is unexplained |
| 10 | DFlash2 install prefill | DONE 2026-10-25: the chunked driver captures the drafter's aux states (`TURBOSPARK_DFLASH_CHUNKED_PREFILL`, default on). Paired verdict 2026-10-26 (`kapitsa-quiet-a2`): WIN on 3 clean pairs, 158.8-158.9 tok/s against 23.4-23.5 for the walk on 426 tokens (+577%) | Server TTFT re-measured 2026-10-26: 0.39 s at 62 tokens, 2.7-2.8 s at 426 (prefill table) |
| 11 | Which tile mode ships | A DECISION FOR THE TILE'S OWNER: `=verify` speculates as fast as `=1` with no plain-decode or prefill penalty, but its speculative greedy stream parts from sequential decode (561 against 505 tokens) | `=1` is lossless and costs plain decode 21-23%; neither is a free default |

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
until those are decided. The MMA8 wiring landed in `937a9c9f` (PR #176:
`real_forward_dispatch.rs`, the qwen `attn.rs`, `dense.rs`, `moe.rs`,
`gpu/src/lib.rs`), and the drafter work below touches none of those.

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

**Correction, same day.** The profile and microbench below are real, but the
conclusion they were used for (a roughly 3 ms saving from lowering the tile's
output-row threshold) is NOT supported by the clean wall-clock sweep taken
afterwards ("Measured: the full sweep on current HEAD", below): with every
projection on the tile, DFlash2 7 verify read 75.3 and 72.0 ms against 69.8
and 70.5 at the default threshold. Read the "predicted saving of 4.6 ms",
"about 3 ms" and "about 0.7 ms a token" statements in this subsection as
predictions the real model did not confirm. The per-dispatch tables stand as
what they are.


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

#### Measured: the full sweep on current HEAD, 2026-10-23

Everything below is one script, 16 arms, short-explanation and medium-review,
two rounds with the arm order reversed in the second, `turbospark-bench
--shaping greedy`, one process per run, load under 4.0 at every start (the
script waits for under 2.5 and then proceeds), 64 of 64 runs completed. The
host was quiet (swap 3.6 GB, steady). Two binaries: **HEAD** is `2666539a`
(it carries the drafter changes above and the GDN conv-tap fix `7b37fd36`),
and **tile** is HEAD plus a COPY of the 8-row kernel owner's uncommitted
working tree taken that morning (patch `96adbc76e8b4`, kernel sources
`954ef847140c` and `705eeee0901d`). The tile has three modes by environment:
`TURBOSPARK_INT4_MMA8=1` (**shared**: decode and verify both run the tile, so
speculative output stays bit-identical to plain decode), `=verify` (only
batched calls run the tile and one-row decode keeps the GEMV, so verify and
decode differ in the last bits), and `TURBOSPARK_INT4_MMA8_MIN_ROWS=1` (serve
every projection, the lever 9 candidate). MTP arms use `qwen38-27b-mtp`,
DFlash2 arms `qwen38-27b-dflash2`, plain decode runs on the MTP install.

short-explanation (62-token prompt):

| Arm | tok/s, round 1 / round 2 | committed per round | draft ms | verify ms | tokens | prefill s |
|-----|-------------------------:|--------------------:|---------:|----------:|-------:|----------:|
| HEAD, plain decode | 22.1 / 21.8 | - | - | - | 505 | 2.7 |
| HEAD, MTP 2 (the shipped default) | 33.1 / 32.9 | 2.15 | 8.5 | 55.2 | 561 | 0.4 |
| HEAD, MTP 4 | 18.2 / 19.1 | 2.40 | 15.5 | 111.4 | 561 | 0.4 |
| HEAD, DFlash2 4 | 20.0 / 20.3 | 2.57 | 14.6 | 111.4 | 505 | 2.7 |
| HEAD, DFlash2 7 | 15.2 / 16.1 | 2.71 | 19.8 | 151.8 | 505 | 2.7 |
| tile shared (=1), plain decode | 16.5 / 17.2 | - | - | - | 533 | 3.5 |
| tile shared, MTP 4 | 27.7 / 28.3 | 2.38 | 18.0 | 65.4 | 488 | 0.4 |
| tile shared, DFlash2 4 | 32.1 / 32.7 | 2.53 | 10.2 | 66.4 | 533 | 3.5 |
| tile shared, DFlash2 7 | 32.4 / 33.4 | 2.72 | 11.0 | 69.8 | 533 | 3.5 |
| tile verify-only, plain decode | 21.7 / 21.8 | - | - | - | 505 | 2.7 |
| tile verify-only, MTP 2 | 30.4 / 30.3 | 2.16 | 8.6 | 61.5 | 488 | 0.5 |
| tile verify-only, MTP 4 | 28.4 / 29.2 | 2.39 | 15.5 | 65.8 | 488 | 0.4 |
| tile verify-only, DFlash2 4 | 31.1 / 33.2 | 2.49 | 10.1 | 65.7 | 561 | 2.7 |
| tile verify-only, DFlash2 7 | 32.4 / 30.4 | 2.60 | 11.0 | 69.6 | 561 | 2.9 |
| tile shared + MIN_ROWS=1, plain decode | 17.3 / 15.8 | - | - | - | 488 | 3.7 |
| tile shared + MIN_ROWS=1, DFlash2 7 | 29.1 / 29.5 | 2.61 | 11.5 | 75.3 | 488 | 4.1 |

medium-review (426-token prompt):

| Arm | tok/s, round 1 / round 2 | committed per round | draft ms | verify ms | tokens | prefill s |
|-----|-------------------------:|--------------------:|---------:|----------:|-------:|----------:|
| HEAD, plain decode | 21.8 / 21.7 | - | - | - | 846 | 18.6 |
| HEAD, MTP 2 (the shipped default) | 34.2 / 34.8 | 2.30 | 8.5 | 56.8 | 846 | 3.0 |
| HEAD, MTP 4 | 21.0 / 21.6 | 2.75 | 15.7 | 111.2 | 846 | 2.9 |
| HEAD, DFlash2 4 | 22.7 / 23.3 | 2.97 | 15.1 | 112.2 | 846 | 18.8 |
| HEAD, DFlash2 7 | 17.7 / 18.4 | 3.20 | 20.6 | 154.5 | 846 | 18.9 |
| tile shared (=1), plain decode | 16.9 / 17.5 | - | - | - | 846 | 23.6 |
| tile shared, MTP 4 | 31.4 / 32.0 | 2.74 | 18.3 | 66.4 | 810 | 2.9 |
| tile shared, DFlash2 4 | 36.8 / 37.8 | 2.97 | 10.6 | 67.1 | 846 | 24.4 |
| tile shared, DFlash2 7 | 37.5 / 38.6 | 3.20 | 11.5 | 70.5 | 846 | 24.4 |
| tile verify-only, plain decode | 21.7 / 21.8 | - | - | - | 846 | 18.6 |
| tile verify-only, MTP 2 | 32.1 / 32.0 | 2.30 | 8.6 | 61.7 | 810 | 3.0 |
| tile verify-only, MTP 4 | 32.2 / 32.5 | 2.74 | 15.8 | 67.1 | 810 | 3.0 |
| tile verify-only, DFlash2 4 | 31.2 / 37.8 | 2.96 | 14.4 | 69.3 | 845 | 21.2 |
| tile verify-only, DFlash2 7 | 36.2 / 39.7 | 3.20 | 11.6 | 70.3 | 845 | 18.9 |
| tile shared + MIN_ROWS=1, plain decode | 14.7 / 15.4 | - | - | - | 808 | 24.9 |
| tile shared + MIN_ROWS=1, DFlash2 7 | 35.1 / 39.0 | 3.18 | 11.5 | 72.0 | 808 | 26.6 |

What it says:

- **The shipped default is strong on prose.** HEAD with MTP block 2 reads 33.0
  and 34.5 tok/s against 22.0 and 21.8 plain. Without the tile every longer
  block loses (MTP 4: 18.6 and 21.3; DFlash2 7: 15.7 and 18.1), which is
  the c(M) problem again, now measured on current HEAD.
- **The tile turns block 7 from the worst arm into the best.** DFlash2 7
  reads 32.9 and 38.1 under the shared mode and 31.4 and 37.9 verify-only.
  Against the shipped default that is 0.95 to 1.0x on short-explanation and
  +10% on medium-review. The copy-heavy prompt (server, separate run) reads
  87.3 against 47.8-48.0, 1.8x (separate runs, and the 48 is from the earlier
  HEAD, so a direction). So the tile plus DFlash2 pays on predictable
  text and roughly breaks even on unpredictable prose, where only 2.6-2.7
  tokens commit per round.
- **The mode matters for plain decode and prefill, not for speculation.**
  Shared plain decode is 16.8 and 17.2 tok/s against 22.0 and 21.8 (-21% to
  -23%), and its sequential prefill is 27-31% slower (18.6 to 23.6-24.4 s on
  medium).
  Verify-only leaves both unchanged (21.7 and 21.7 tok/s, 18.6 s) and
  speculates as fast: DFlash2 7 within 1.5 tok/s of shared on both cases.
- **Verify-only is not lossless against plain decode.** DFlash2 emitted 561
  tokens against plain decode's 505 on short-explanation (and 845 against
  846 on medium-review), so the speculative greedy stream parts from the
  sequential one at a near-tie. The shared mode matches its own plain decode
  (533 and 533, 846 and 846). The size of the drift is the tile owner's
  `int4_mma8_drift_probe.rs` question; this sweep only shows it exists and
  that it buys nothing in speed here. Note the MTP install's own speculative
  path already differs from plain decode on HEAD (561 tokens against 505 on
  short-explanation), a known quirk of that install (`docs/BENCHMARKS.md`), so
  token counts across the MTP arms are not an exactness test.
- **Lever 9 is not confirmed.** With every projection on the tile
  (`MIN_ROWS=1`), DFlash2 7 verify reads 75.3 and 72.0 ms against 69.8 and
  70.5 for the shared mode at its default threshold, plain decode 16.5 and
  15.1 against 16.8 and 17.2, and sequential prefill is slower again. The
  microbench (tile 2.7-5.6x faster per dispatch on those shapes) and the
  dispatch profile (66.3 to 63.8 ms) pointed the other way, and one earlier
  unprofiled run read 63.3. The two-round A/B is the clean one. Whatever the
  cause (the extra activation-prep dispatches and the shared prep scratch are
  the suspects, unexamined), a per-dispatch time is not a round time, and the
  threshold should stay where the tile's owner put it.
- **Where that leaves the gap to kapitsa.** Our best arm on this machine is
  DFlash2 7 under the tile, 32.9 and 38.1 tok/s, against kapitsa's 52.6 and
  68.4: 63% and 56%. At equal acceptance (MTP head: 2.39 against 2.38) the
  difference is round cost, about 83 ms against about 52 ms (kapitsa's figure
  derived by assuming equal acceptance), and the
  remaining pieces are the verify floor (70 ms for 8 rows against a
  1.1x-of-a-step 50) and the drafter (11 ms).

#### Measured: the kernel-modes tooling, projections and live checks, 2026-10-25

Apple M4 Max 36 GB, AC power, `tslab` built from the `lab/kernel-modes`
branch (`docs/KERNEL_MODES.md`), release build, one measuring session. Host
load was 3-9 from other sessions' builds, and the runs inside the lossless gate
were flagged dirty for foreign CPU (a bit-equality check, so timing does not
matter there). Ledger ids are in the lab dir (`~/.turbospark/lab/<experiment>`).

**DFlash2 depth 7 under `TURBOSPARK_INT4_MMA8=1`, lossless (lever 3's owed
gate).** `tslab gate` on `qwen38-27b-dflash2`, `bit_exact`, plain against
`dflash:7`, both with `TURBOSPARK_BATCHED_GEMV=0`: PASS, mean KL 0, top-1 1.00,
tokens equal (experiment `km-lossless`, gate `1792959758519-40525-11`). The
gate judges six repo-document prompts of 32 decode tokens. A second check
compared decode digests on the protocol prompts and the copy-heavy prompt,
256 decode tokens, two rounds (`km-lossless-digests`): short-explanation and
medium-review are identical across plain, `dflash:2` and `dflash:7`. On
copy-heavy the two speculative arms agree with each other and plain differs,
but plain decoded 256 tokens and the speculative arms stopped at 188, so those
digests cover different lengths. Rerun at 128 decode tokens, under the stop
(`km-copy-digest-128`, two rounds): plain, `dflash:2` and `dflash:7` all give
digest `2b70123f77`, so the three prompts are lossless under the flag. The
256-token difference was plain decode running past the end-of-turn token.

**The tuned adaptive controller, live, under the tile (lever 2).** MTP install,
`mtp:8a` with `TURBOSPARK_ADAPTIVE_DEPTH=3,1,1,2` and `TURBOSPARK_INT4_MMA8=1`,
256 decode tokens, 3 clean pairs each:

| Workload | Against fixed 2 with the tile | Against today's default |
|----------|------------------------------:|------------------------:|
| medium-review (prose) | +3.5%, NOISE (`km-adapt-prose`) | -5.6%, LOSS (`km-adapt-vs-default-prose`) |
| code (merge sorted slices) | +16.1% WIN (`km-adapt-code`) | +6.4% WIN (`km-adapt-vs-default-code`) |
| copy-heavy (rename) | +26.6% WIN (`km-adapt-copy`) | +17.1% WIN (`km-adapt-vs-default-copy`) |

The offline replay predicted code +16%, quote-back +7% and prose -6% against the
default. Prose agrees. Code is lower live (+6.4%) and copy-heavy higher
(+17.1%), and the replay's prompts were different texts, so these are not like
for like. The controller is not a default: prose loses.

**Projection against model A/B (`tslab project`, `tslab calibrate`).**

| Mode and row count | Predicted | Measured | Read |
|--------------------|----------:|---------:|------|
| `gdn_dk128`, 8 rows (`km-proj-dk`) | +3.4% | +4.1% short, +5.0% medium (`gdn-dk128-dflash7`) | within the floor |
| `gdn_dk128`, one row (`km-proj-dk1`) | +0.9%, under the floor | -0.4%, NOISE (`gdn-dk128-decode`) | within the floor |
| `int4_mma8`, one row, plain decode (`km-proj`) | -16.9% (0.831x) | -17.4%, interval [-19.0%, -17.1%], 0 better / 5 worse of 6 pairs (`km-mma8-plain6`); a first 3-pair run read -26.9% with an interval spanning zero (`km-mma8-plain`) | within the floor on the larger sample; the doc's 0.81x agrees |
| lever 9, small projections at 8 rows (`km-l9-proj2`) | +5.8% (4.1 ms of 75 ms) | +3.3% (`km-l9-ab`, 2 clean pairs, inconclusive) | 2.5 points under, one inconclusive A/B |
| lever 9 again, from the primed bench (`km-floor-proj`, `1792977511459-47500-0`) | +4.8% (3.4 ms of 75 ms) | +3.3% (`km-l9-ab`, 2 clean pairs, inconclusive); then +2.3% WIN, interval [+1.9%, +3.8%], 3 better of 3 clean pairs of 6 (`km-l9-ab3`, DFlash2 depth 7, medium-review, 256 decode tokens, 5 of 12 runs dirty for foreign CPU) | 1.5 points under the first (`km-floor-cal`, `1792977565457-47778-0`, within the floor); 2.5 points over the conclusive one (`1792979021129-90371-0`, MISS). The 1024-row curve came from a DIRTY run, and the projection covers 6% of the round |

Coverage was 2-7% for the GDN and small-projection cases and 85% for the tile
on decode, so the GDN rows are priced from a small slice of the round and still
landed, because the effect lives entirely in that kernel.

A bench bias this found: `int4_matmul` timed a tiny shape in a pass of at most
8 dispatches, so the command buffer's fixed cost dominated. The 48-row gate
projection read 157 us for the scalar GEMM at 8 rows (46 us in the 64-copy
bench) and the first lever 9 projection was +21%, a 17.8-point miss
(`km-l9-proj`). The bench now repeats the copy sweep until a pass holds at
least 64 dispatches, and the same shape reads about 42 us scalar and 8 us for
the tile. That is the fix to the bench, not a lever 9 result. One-row
`gemv_per_row` on those shapes still read about 3x its per-dispatch share in an
8-GEMV pass (13.5 us against 3.9 us). A separate output buffer per weight copy
did not move it, and neither did a single cache-warm weight copy, so output
ordering and cache warmth were ruled out before the cause was found.

**The one-row floor, explained (2026-10-25).** `crates/lab/tests/int4_one_row_floor_probe.rs`
(M4 Max, AC power, host load 4-7 from other sessions' builds and model tests;
probe output, not ledgered). With the device warm, the one-row 48x5120 GEMV
reads 3.0-3.3 us per dispatch whether the pass holds 8 or 2048 dispatches, with
rotating x and y offsets, after a scalar or mma8 pass, after a 1-5 ms host gap,
and with or without a 1024-dispatch primer: nothing in the pass's structure
moves it, so the overlap reading above was wrong. The bench's own `run_shape`
called three times in one process read the one row at 13.6, 3.8 and 3.0 us.
From a cold device (3 s idle before each sample) a 64-dispatch pass reads 18.5
us per dispatch, a 16384-dispatch pass stays slow for its whole 280 ms (16.9 us
per dispatch), untimed primers of 1024 and 4096 dispatches committed right
before the pass leave it at 16-18 us, and a primer of 16384 (about 50 ms)
brings it to 3.2 us. The device's speed is set by the state it is in when a
pass starts and does not rise within the pass; only tens of milliseconds of
work queued ahead of it raise that state. M = 1 was the bench's first
measurement, with host work between passes, so the one-row unit and every
c(M) read against it was the device's state and not the kernel. A GPU effect,
but one a decode step never sees: its command buffer is long and busy, so the
warm number is the one that prices a round.

The fix is a primer (`kernel::support::Primer`, 16384 one-row GEMVs on its own
small matrix) committed right before every timed pass of `int4_matmul`; the
record carries `subject.primer.gpu_ms_median` (51.8 ms here). A "settle until
consecutive windows agree" warm-up was tried first and rejected: a cold device
plateaus at 7-14 us and the loop accepts it. Re-timed with the primer
(`km-floor-curves`, `1792977170082-27350-0`, clean): the 48x5120 one-row GEMV
reads 3.25 us and `gemv_per_row` c(8) is 0.97; scalar at 8 rows 33.2 us and the
tile 6.8 us (42 and 8.2 us before). The 1024x5120 curve
(`1792977177535-27848-0`, DIRTY for foreign CPU) reads the one row at 10.2 us
against 8.3 us in `km-retro2`, scalar at 8 rows 44.5 us (40.9), tile 15.8 us
(13.6). The lever 9 projection from these curves is +4.8%; the A/B re-run
with 6 pairs on decode tok/s reads +2.3%, a WIN, so the primed bench still
overstates the lever by 2.5 points (table above). The two misses the
projection cannot see are the ones `docs/KERNEL_MODES.md` lists: the tile's
activation-prep dispatches and shared scratch, which the kernel curve does not
time, and a base of which the benched kernels are 6%. The primer fixed the
one-row unit, not the coverage. The other kernel benches do not prime yet, so
a tiny-shape number from `int8_matmul`, `gdn_delta` or `attn_decode` carries
the same suspicion until they do.

**Lever 9 under a trace.** The first two `tslab trace` windows landed in the
sequential prefill and said nothing. Rerun on a short prompt with 768 decode
tokens so the 8 s window sits in the 8-row verify (`km-l9-trace2-d7`,
`km-l9-trace2-min1`, DFlash2 depth 7, tile on): GPU busy is 92.1% at the
default threshold and 92.5% with `TURBOSPARK_INT4_MMA8_MIN_ROWS=1`, so neither
is launch-bound at window level. The scalar batched GEMM is 3.8% of kernel
samples at the default and absent with the knob, and the tile's GEMM rises from
95.6% to 99.4% of samples. Compute encoders in the window went 433 to 455.
Shares come from samples (about a tenth of GPU time), not times. They fit a
gain of at most a few percent from moving the small projections to the tile,
which agrees with the +3.3% measured and sits under the +5.8% projected. They
do not show an extra cost of the extra prep dispatches.

**Mode candidates, priced or not (2026-10-25).** Only what the lab can price
today is priced. The rest names what is missing, and none is built.

| Candidate | Projection | What is missing |
|-----------|-----------|-----------------|
| Drafter-gated tile (mlx-serve E1: tile only when a drafter is bound) | Removes the plain-decode loss the tile projects (0.831x, `km-proj`) on a headless install, keeps the code gain | No bench arm separates "drafter bound" from the flag; the routing lives in `real_forward_dispatch.rs`, the tile owner's file |
| Concurrent q/k/v encoder (item 2) | +0.4% on a decode token, UNDER THE FLOOR (`km-proj-qkv`, new `qkv_concurrent` bench: the triple takes 0.896 of its serial time, 0.19 ms across 16 layers of 45.5 ms, output bits identical) | Nothing to build as a standalone mode; it would have to cover far more dispatches than the one triple per layer to clear 2% |
| Expert-sorted grouped GEMM MoE prefill (item 8) | +4.0% to +4.1% of a Flint prefill token against one GEMV per route, at 4 and 16 rows per expert (`km-proj-moe`; kernels cover 6% of a 40 ms token at 24.9 tok/s) | The base is dominated by something the bench does not time (likely expert I/O), the engine's batched MoE prefill paths may already capture part of it, and the drift bound in `BATCHED_PREFILL.md` still applies. Measure the routed share of a prefill token before building |
| Hybrid prompt lookup plus MTP | Priced by replay (`tslab hybrid-replay`, "Measured: the hybrid replay" under "Prompt lookup"). As specified (4-gram, draft up to 8, lookup when longer than the head's depth): -4.3% committed/round against the tuned 2..8 controller on copy-heavy, -1.4% prose, -3.9% code. With an 8-gram gate: +7.1% on copy-heavy, 0 lookup rounds on prose and code | One greedy stream per case, 512 tokens; round costs from the modeled flag-on table (the measured ones were dirty). Clears the 5% bar only in the 8-gram form, on one stream. Proposed, not built |

#### Re-measured on the shipped defaults, 2026-10-25

HEAD `762e8f65`, tile off (the default), `qwen38-27b-mtp`, MTP block 2 against
speculation off, two rounds with the arm order reversed, a quiet host (load
1.4-2.4 at all 12 run starts). Decode tok/s, mean of the two rounds:

| Case | Off | MTP 2 | Ratio | Committed/round |
|------|----:|------:|------:|----------------:|
| short-explanation | 21.7 | 33.1 | 1.53x | 2.15 |
| medium-review | 21.4 | 35.5 | 1.66x | 2.30 |
| long-synthesis (2,940-token prompt) | 21.4 | 34.6 | 1.61x | 2.30 |

The first two cases match the quiet-host HEAD sweep above (MTP 2 at 33.0 and
34.5, plain at 22.0 and 21.8) to within 3%, so the defaults did not move
between 2026-10-23 and now. The long-synthesis row is new, and its speculative
prefill reads 19.3 s (152 tok/s). The full table, the caveats and the
comparison with the 2026-10-11 rows are in [BENCHMARKS.md](BENCHMARKS.md)
under "The `qwen38-27b-mtp` install, measured 2026-10-11".

#### What the testing so far did not cover, and what it found late

Everything on this page was measured on one family (the dense `qwen35`,
Qwen3.8-27B 4-bit) on one machine (M4 Max, 36 GB). Beyond that:

**Prefill, which this page did not report until the end.** The decode work
above says nothing about time to first token, and the logs show a larger gap
there than in decode:

| Install and driver | Prompt | Prefill | Tokens per second |
|--------------------|-------:|--------:|------------------:|
| HEAD plain decode, bench default (sequential driver) | 426 | 18.6 s | 23 |
| HEAD MTP install, speculative path (chunked driver) | 426 | 3.0 s | 142 |
| HEAD DFlash2 install, bench (sequential) | 426 | 18.9 s | 23 |
| DFlash2 install, speculative path, chunked driver with aux capture (2026-10-25, 5 clean runs, `kapitsa-levers-a6`) | 426 | 2.7-3.3 s | 130-158 |
| DFlash2 install, same A/B on a quiet host (2026-10-26, `kapitsa-quiet-a2`, 3 clean pairs), token walk | 426 | 18.1-18.2 s | 23.4-23.5 |
| DFlash2 install, same A/B, chunked driver | 426 | 2.68 s | 158.8-158.9 |
| DFlash2 install through `turbospark-server`, before lever 10 (token walk), warm | 62 | 3.83 s | 16 |
| DFlash2 install through `turbospark-server`, before lever 10, first request (includes model page-in) | 204 | 11.95 s | 17 |
| DFlash2 install through `turbospark-server`, chunked driver, warm (2026-10-26, 4 runs) | 62 | 0.39 s | 159 |
| same, copy-heavy prompt | 224 | 1.36 s | 165 |
| same, medium-review prompt | 426 | 2.67-2.80 s | 152-160 |
| tile shared mode, sequential driver | 426 | 23.6-24.4 s | 18 |
| MTP install, long prompt (playbook, chunked, Steel) | 2,727 | 21-25 s | 110-130 |
| kapitsa, its own bench | 4,133 | 9.9 s | 417 |
| kapitsa, its own bench | 16,287 | 40.4 s | 403 |

Kapitsa also reports 0.17 s and 0.21 s for a follow-up turn (a prefix cache and
an SSD cache), and its time to first token read 0.06 s on every repeated
prompt in our HTTP runs. The reading: even our best install, the MTP one on
the chunked Steel path, prefills at roughly 3x slower than kapitsa, and **the
DFlash2 install prefills token by token at about 17-23 tokens per second, 18x
to 24x slower**, because the chunked driver does not yet capture the drafter's
per-layer auxiliary states (an open item in `docs/SPEEDUP_PLAYBOOK.md`, "MTP
and other drafters"). A DFlash2 decode win on an install whose first token
takes ten seconds for a 200-token prompt is a latency regression for any agent
workload. The bench's plain-decode prefill is its sequential harness default
(`prefill=sequential` in its header), not a measurement of the server's plain
path, which was not timed here. This is lever 10 in the table above.

**Lever 10 since (2026-10-25).** The chunked driver now encodes the aux
captures (`docs/DFLASH2.md`, "Chunked prefill with the drafter open"), so the
DFlash2 speculative path prefills on the same driver as the MTP install. On
the 426-token prompt the five clean chunked runs in tslab experiment
`kapitsa-levers-a6` read 130.3-157.5 tok/s (M4 Max, AC, warm page cache).
All five token-walk runs in that A/B were DIRTY (other sessions' compiles and
tests landed in their 20-second prefill window) and read 15.9-22.5 tok/s, so
the A/B's verdict is INCONCLUSIVE on zero clean pairs and owed a quiet host;
the gap is far outside run-to-run spread either way. The server TTFT rows
above were not re-measured.

**Lever 10, paired verdict (2026-10-26).** Same command on a quieter host
(`tslab ab --prompt protocol:medium-review --decode 16 --baseline
walk~dflash:7:dflash_chunked_prefill=0 --candidate chunked~dflash:7 --metric
prefill_tok_s --max-pairs 12`), experiment `kapitsa-quiet-a2`, verdict
`1793051988452-69657-20`: **WIN on 3 clean pairs, median +576.7%, 95% CI
[+575.7%, +577.6%]**. Walk 23.4-23.5 tok/s (18.1-18.2 s), chunked
158.8-158.9 tok/s (2.68 s), 426 prompt tokens. Conditions: M4 Max, AC power,
warm page cache (cache class `no_reads`), swap 14.3 GB used and steady (a
standing condition of this host), worktree HEAD `f9751f12`, `tslab-worker`
SHA-256 `632e0f9e`. The null floor for this workload is 0.66%
(`kapitsa-quiet-a1`, record `1793051786408-61542-16`). Two of the eight runs
were dirty (`mediaanalysisd` at 280-300% CPU) and are excluded. An earlier
attempt in `kapitsa-quiet-a1` stopped at 2 clean pairs (browser and
WindowServer load) and is INCONCLUSIVE.

**Server TTFT after lever 10 (2026-10-26).** Release `turbospark-server` from
the same tree, `TURBOSPARK_INT4_MMA8=1 --speculative 7
--speculative-drafter dflash`, driven by `scripts/bench_api.py` (temperature
0, 32 max tokens, 4 runs per prompt, streamed, time to the first non-empty
delta). Prompt sizes are the server's own `usage.prompt_tokens`. With
`--prefix-reuse off` the 62-token short-explanation prompt reads 0.39 s on
every run, copy-heavy (224 tokens) 1.36 s and medium-review (426 tokens)
2.67-2.80 s. That is 10x faster than the 3.83 s warm row from before lever
10. The 204-token row has no current twin (the old prompt is not recorded,
and the nearest case is copy-heavy at 224), and no cold page-in reading was
taken because the page cache cannot be purged without `sudo`. These are not
tslab runs, so they carry no ledger id and no host guard, but load was
1.3-1.6 at both ends of each arm. With `--prefix-reuse on`, the shipped
default, repeating the identical prompt is no faster (0.39 s, 1.36 s,
2.90-2.96 s): a drafter install reuses only a pure continuation
(`docs/DFLASH2.md`, section 7, "what was adopted", item 2), and a repeated prompt is a rewind
past the previous reply. Kapitsa's 0.06 s on a repeated prompt is a prefix
cache this path does not have.

**Memory.** The bench's peak footprint reads 658 MiB for the plain install,
868 for MTP and 896 for DFlash2, against 19.6 GB in kapitsa's own bench. The
two counts differ (a mapped-weight footprint against allocated buffers
including four conversation slots), so they are not a like-for-like ratio, but
they describe different designs and the difference is real on a 36 GB machine.

**Not measured at all:**

- Sampled (temperature above zero) speculative throughput. Kapitsa reads 82.7
  tok/s sampled against 90.7 greedy (-9%). The engine's loop does exact
  rejection sampling, but the server's help text still says speculation
  applies to temperature-0 requests only; one of the two is out of date.
- Decode at 16K tokens and beyond. Only a 2,940-token prompt was run here.
- Concurrent requests. Kapitsa runs up to four conversations and reports
  +15-23% aggregate for three sharing a verify.
- Output quality against kapitsa. The tile owner's quality gate covers the
  tile against our own baseline (perplexity 4.9429 under the shared mode).
- Kapitsa with speculation off, energy and power for either engine.
- Any family other than the dense `qwen35`. The tile's flag is global and only
  this family was gated; the verify cost curve for a routed-MoE family is
  different (extra rows bring in extra experts), and the GQA-shared attention
  kernel serves head_dim 128 or 256 with at most 8 query heads per KV head.
  The checklist for those is `docs/NEW_MODEL.md`, "Phase 6b".

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
number that matters. Measured below.

Gates: greedy output identical to the sequential loop at several draft
lengths (`mtp_generation_gate` shape), the memory oracle unchanged, and an
interleaved paired A/B on AC power. A negative result is recorded here.

#### Measured: the hybrid replay, 2026-10-25

`tslab hybrid-replay` (`crates/lab/src/hybrid_replay.rs`) walks one greedy
stream per case, captured once by `tslab chain-capture` on `qwen38-27b-mtp`
with the token ids kept (experiment `km-hybrid`, captures
`1792977261266-28363-0` medium-review, `-1` `code.messages.json`, `-2`
`copy-heavy.messages.json`; 512 tokens each, depth-8 chains; Apple M4 Max, AC
power, host busy with another session's model tests, which does not touch
acceptance). Per round it takes the lookup proposal (the continuation of the
last n tokens of prompt plus history, up to 8 tokens) when it is longer than
the depth the head would propose, with exact acceptance against the stream,
and never feeds the controller from a lookup round (mlx-serve's two E6 rules).
Head rounds are `depth_replay::simulate` with the runtime's own
`AdaptiveDepth`. Costs: the modeled flag-on table (MTP 2 and 4 measured,
other depths on their line) for the first run, then the round costs measured
for this on a quiet host (`km-hybrid-costs`, MTP blocks 2, 4 and 8 with the
tile, copy-heavy, 256 decode tokens, two clean runs each: 69.5-70.2, 81.6-82.3
and 154.6-157.1 ms a round; the first profile of the same arms, `km-hybrid`,
was all DIRTY and is not used). A lookup round is priced at the same width's
MTP round, head draft steps included, so the tok/s column is conservative for
the hybrid; the committed-per-round column needs no cost table and is what
the bar reads.

Committed per round, head alone against the hybrid, and the hybrid's change
(`km-hybrid` `hybrid_replay` records `1792977511638-47511-0` for the 4-gram,
`1792977580265-47921-0` for the 6-gram, `1792977580237-47913-0` for the 8-gram):

| Policy | Case | Head alone | 4-gram, draft 8 | 6-gram, draft 6 | 8-gram, draft 8 |
|--------|------|-----------:|----------------:|----------------:|----------------:|
| fixed 2 | medium-review | 2.35 | 2.31 (-1.8%) | 2.34 (-0.5%) | 2.35 (0, no lookup rounds) |
| fixed 2 | code | 2.73 | 2.78 (+1.6%) | 2.73 (0) | 2.73 (0) |
| fixed 2 | copy-heavy | 2.97 | 5.27 (+77%) | 4.48 (+51%) | 5.38 (+81%) |
| shipped adaptive 2..4 | medium-review | 2.35 | 2.31 (-1.8%) | | |
| shipped adaptive 2..4 | code | 3.10 | 2.90 (-6.3%) | | |
| shipped adaptive 2..4 | copy-heavy | 3.84 | 5.32 (+38%) | | |
| tuned 2..8 (3,1,1,2) | medium-review | 2.44 | 2.41 (-1.4%) | 2.43 (-0.5%) | 2.44 (0) |
| tuned 2..8 (3,1,1,2) | code | 4.15 | 3.99 (-3.9%) | 4.15 (0) | 4.15 (0) |
| tuned 2..8 (3,1,1,2) | copy-heavy | 5.68 | 5.44 (-4.3%) | 5.87 (+3.4%) | 6.08 (+7.1%) |

Projected tok/s, tuned controller, copy-heavy: on the modeled table 59.7 head
alone, 54.7 with the 4-gram hybrid (-8.3%), 61.7 with the 8-gram (+3.4%); on
the measured table (`1792978451801-28468-0` for the 4-gram, `1792978451817-28472-0` for the 8-gram) 49.8 head alone,
44.6 with the 4-gram (-10.4%), 50.7 with the 8-gram (+1.7%). Against fixed 2
the 4-gram hybrid reads +32% (modeled) or +4% (measured) tok/s on copy-heavy
and -3 to -8% on prose and code, the shape the standalone replay predicted.

What it says:

- **The hybrid as specified loses to the tuned controller on copy-heavy.**
  A 4-gram matches on 83% of copy-heavy rounds, each lookup draft of 8 is
  longer than the controller's depth 2, so lookup takes nearly every round
  and, by the rule that a lookup round never feeds the controller, the
  controller never climbs: the hybrid becomes the standalone lookup drafter
  (5.44 per round, the 5.3 the earlier replay found) and gives up the head's
  5.68 at depth 8. The two E6 rules are right on their own and wrong
  together with a "longer than the head" gate.
- **An 8-gram gate clears the bar on this stream.** Lookup then fires on 68%
  of copy-heavy rounds, the head rounds between them still feed the
  controller, which climbs, and the hybrid adds where a verbatim span is long
  and the controller is still low: 6.08 against 5.68 per round (+7.1%), and
  +3.4% projected tok/s with the conservative price. On medium-review and
  code an 8-gram never fires, so prose and code are exactly flat, not merely
  losing little.
- **This is one stream per case,** 512 tokens of one greedy decode, and the
  gain is 7 points over a 5-point bar. A second copy-heavy prompt (a RAG-style
  quote-back, a tool transcript) belongs in the capture before anyone builds
  on it. The lookup draft is capped at 8 by the cost table; longer copied
  spans (the standalone replay accepted 4.3 per fired round at draft 8) are an
  untested axis that needs a measured round cost at width 9 and up.
- **The measured round cost at depth 8 is well above the modeled table,
  and it is the tile boundary, measured.** The hybrid's measured table
  (`km-hybrid-costs`, run ids `1792977781487-51633-9` and
  `1792977794665-51633-15` at block 8, `1792978407691-21493-4` and
  `1792978420956-21493-10` at block 2, `1792978414866-21493-7` and
  `1792978428142-21493-13` at block 4, KV bucket 1024, load under 4) read
  70, 82, and 156 ms a round against the flag-on table's 72.5, 86.5, and
  114.5, with the verify phase at 59, 62, and 117 ms: flat from 3 to 5 rows
  and doubled at 9. Width 9 crosses the 8-row MMA tile: `encode_gemm_any`
  serves up to 16 rows, and under the flag `encode_int4_mma8_rows`
  dispatches them in tiles of `INT4_MMA8_MAX_ROWS`, 8, so a width-9 verify
  runs an 8-row tile and a 1-row tile per projection, each reading the
  weights once. A `tslab profile` at blocks 7 and 8, then 2 and 4, then
  5 and 6, all in one experiment, confirms it. Conditions: experiment
  `km-tile-7v8`, `qwen38-27b-mtp`, `copy-heavy.messages.json`, 256 decode
  tokens, `TURBOSPARK_INT4_MMA8=1` and `TURBOSPARK_SPEC_STATS=1`, KV bucket
  1024. Host: Apple M4 Max, AC power, swap 14.7 GiB used of 16, load1
  between 2.8 and 5.1 across the clean runs. 19 of 30 measured runs were
  clean. The rest were DIRTY for foreign CPU from the IDE helpers and Apple
  background daemons, ledgered and not used. Medians of the clean
  `round_cost` records, with the live decode tok/s range of the same runs:

  | Block | Width | Clean | ms a round | verify | draft | committed/round | decode tok/s |
  |------:|------:|------:|-----------:|-------:|------:|----------------:|-------------:|
  | 2 | 3 | 3 | 67.3 | 56.4 | 9.3 | 2.98 | 44.2-44.3 |
  | 4 | 5 | 3 | 81.8 | 61.8 | 17.5 | 4.67 | 57.2-57.5 |
  | 5 | 6 | 3 | 86.1 | 62.0 | 21.1 | 5.70 | 65.6-66.9 |
  | 6 | 7 | 3 | 91.8 | 63.0 | 25.3 | 6.48 | 70.4-70.9 |
  | 7 | 8 | 3 | 101.0 | 65.1 | 30.6 | 6.96 | 68.4-69.0 |
  | 8 | 9 | 4 | 159.9 | 116.8 | 35.4 | 7.83 | 48.4-49.7 |

  `round_cost` record ids:

  - block 2: `1793012101181-81202-6`, `1793012114513-81202-12`,
    `1793012127616-81202-18`
  - block 4: `1793012107311-81202-9`, `1793012120654-81202-15`,
    `1793012133740-81202-21`
  - block 5: `1793014266550-918-6`, `1793014277535-918-12`,
    `1793014288496-918-18`
  - block 6: `1793014271917-918-9`, `1793014282885-918-15`,
    `1793014293861-918-21`
  - block 7: `1793011648833-57728-6`, `1793011821114-65691-6`,
    `1793011833513-65691-12`
  - block 8: `1793011655490-57728-9`, `1793011667740-57728-15`,
    `1793011679978-57728-21`, `1793011840256-65691-15`

  Block 7 verifies in the block-4 time: 65.1 ms against 61.8, 3.3 ms for
  three more rows. The verify is flat from 3 to 8 rows at about 1.7 ms a
  row, then steps 51.7 ms at 9 rows for the second tile pass, a third of
  the block-8 round. A block-8 round costs 58% more than a block-7 round
  and commits 0.87 more tokens. Live, fixed block 7 decodes copy-heavy at
  68.4-69.0 tok/s against 48.4-49.7 at block 8. Under the tile the tuned
  controller's cap belongs at 7, not 8.

  Fixed block 6 is the fastest fixed depth live, 70.4-70.9 tok/s, because
  the draft phase grows about 4.7 ms a step while the verify stays flat.
  That is a different question from the cap, and the controller answers it
  by climbing only when acceptance earns it.

  Replayed with `tslab depth-replay --experiment km-hybrid --costs
  ledger:km-tile-7v8@TURBOSPARK_INT4_MMA8=1` on the three captures above,
  512 tokens, with every depth from 2 to 8 except 3 measured (`km-hybrid`
  record `1793014663810-51455-0`, after a first pass with depths 5 and 6
  still modeled, `1793012166406-88279-1`, beside the modeled flag-on table,
  `1793012166392-88279-0`). Cap 7 was added to the replay grid in
  `crates/lab/src/depth_replay.rs` for this.

  On that table the tuned `3,1,1,2` controller at cap 8 is 1.130x fixed 2
  by geomean (prose 36.2, code 48.2, copy-heavy 51.9 tok/s, with 37 of 90
  copy-heavy rounds at depth 8). The same thresholds at cap 7 are 1.223x
  (36.2, 50.8, and 62.4 tok/s), and the best setting in the grid is cap 7
  with `2,1,1,3` at 1.231x, worst case 1.040x. Prose is identical across
  caps because the controller never climbs past 4 there. Replayed fixed 6
  also beats fixed 7 (1.144x against 1.092x), as it does live.

  Run live, `tslab ab` on the same install: `mtp:8a` as the baseline
  against `mtp:7a`, both with `TURBOSPARK_ADAPTIVE_DEPTH=3,1,1,2`, the tile
  and spec stats, 256 decode tokens, metric decode tok/s. Each workload was
  null-calibrated first (floors 0.14% and 0.37%, so the 2% default
  decides). 3 clean pairs each, M4 Max, AC power, load1 2.2 to 3.7, swap
  14.7 GiB of 16.

  Copy-heavy (`km-cap7-copy-heavy`, verdict
  `1793014453668-17296-16`): WIN, +24.2%, interval [+24.1%, +24.6%], 3
  better of 3, 56.1-56.3 tok/s at cap 8 against 69.7-70.0 at cap 7. Both
  arms decode the same 188 tokens in the same 30 rounds with the same 8
  rollbacks and 6.27 committed a round. The cap-8 arm spends 12 of those
  rounds at depth 8, 7 at depth 7, and 3 at 6. The cap-7 arm spends 17 at
  depth 7 and 5 at 6. The spans between rejections are short enough that a
  9-token round never saved a round, so every depth-8 round paid the second
  tile (verify 81.8 ms a round against 61.9) for nothing.

  Code (`km-cap7-code`, verdict `1793014615484-40479-16`): NOISE, +1.45%,
  interval [+1.3%, +1.7%], 3 better of 3, under the floor, with the cap-8
  arm at depth 8 in 1 of 59 rounds. Prose was not run, since the controller
  never passes depth 4 there. The lever 2 live table above was measured at
  cap 8, so its copy-heavy row understates the tuned controller by about a
  quarter. Committed per round in the hybrid table above is unaffected, and
  its measured-table tok/s columns already carry the block-8 cost.

  Acted on: the runtime clamps an adaptive cap to the tile's rows less one while the
  tile is on (`AdaptiveDepth::tile_cap`, read at the speculative loop's one
  construction site), and `tslab hybrid-replay` defaults to `--cap 7`. A
  fixed block is not clamped, and the shipped `DEFAULT_ADAPTIVE_CAP` of 4 is
  untouched because it is a flag-off number. Checked end to end on the
  same install (`km-cap7-clamp`, runs `1793045453067-83736-5` with the tile
  and `1793045460487-83736-7` without, on battery, so a histogram and not a
  timing): `mtp:8a` with the tile on now spends 17 rounds at depth 7 and
  none at 8, and with the tile off still spends 12 at depth 8, both at 6.27
  committed a round.

Proposal, not built, contingent on a second copy-heavy capture agreeing: a
`Lookup` source inside the MTP speculative loop, gated by an 8-gram suffix over
prompt plus committed tokens, drafting up to the controller's cap, taken only
when its draft is longer than the depth the controller asks for, verified with
the greedy exact-match accept the loop already has for the head (no
approximate acceptance on a copied draft), and its rounds excluded from the
controller's `observe`. Off by default behind `TURBOSPARK_LOOKUP_NGRAM=N`
(unset means off), on the MTP install only. Gates before any default: greedy
output identical to the sequential loop at several n-grams
(`mtp_generation_gate` shape), the memory oracle unchanged, and `tslab ab` on
`qwen38-27b-mtp` with `copy-heavy` and `medium-review`, arms `mtp:7a` with
`TURBOSPARK_ADAPTIVE_DEPTH=3,1,1,2` with and without the n-gram, 6 pairs.

## The 2026-10-22 kapitsa and mlx-serve report, checked against this tree

A research report on how both engines reach triple-digit decode on
Qwen3.8-27B was read against the code here. It cites kapitsa's README
(commit `bfe46a34`), its `docs/benchmarks-m5-max.md`, and mlx-serve's
`docs/performance.md` (commit `e6a35b14`) and release notes. None of it was
re-measured for this section.

What it gets right, and what this page already agrees with:

- Speculation is the multiplier, and the cost of a verify pass decides how
  deep it can go. The "Measured: kapitsa on this machine" section above
  makes the same point with numbers from this host.
- 102 tok/s is a headline "fastest setup" row. Kapitsa's own formal M5 Max
  run reports 72.9 tok/s sampled and 83.7 greedy with GPU contention
  flagged, and the same table puts mlx-serve at 130 on the M5 Max and below
  kapitsa (54 against 63) on the M4 Pro. No ordering of the two engines
  holds across chips.
- Decode tok/s is generated tokens over the time from the first to the last
  token, with time to first token reported separately, and never a count of
  stream chunks. `scripts/bench_api.py` and tslab both measure it that way.

What it gets wrong about this tree, because it read the README only:

- **The TurboSpark baseline.** It quotes 16.8-19.0 tok/s for Qwen3.8-27B.
  That is the plain-decode README row. The shipped default on
  `qwen38-27b-mtp` is MTP block 2, measured above at 34.9-47.8 tok/s.
- **"Add MTP, PLD and prefix caching."** MTP ships (block 2, lossless,
  exact sampled rejection). Prompt lookup was replayed and declined as a
  standalone drafter (section above). Prefix reuse exists (`kv_prefix.rs`,
  `session_pool.rs`) with the GDN limit described below.
- **"A fused INT4 decode path."** Decode already dequantizes in registers
  and the dense FFN tail is fused (items 4 and 5). There is no materialized
  FP16 weight buffer to remove. The open verify-side kernel is the MMA8 tile.
- **Adaptive depth "CLI only".** This page said so too, and it is wrong:
  the FFI path the app and its embedded server use calls
  `run_raw_completion_speculative_adaptive` as well
  (`crates/ffi/src/server_model.rs`, `crates/ffi/src/generate/mod.rs`). The
  standalone HTTP server crate (`crates/server`) does not on `main`. Lever 2
  below adds it.

## How each remaining lever would be implemented here

Ordered by expected single-stream value on Qwen3.8-27B, then by aggregate
value. Each entry says why it is faster, where it goes, how it is gated,
and how tslab measures it (`crates/lab/program.md`). Estimates are marked.

1. **Close the one-row tile gap in the MMA8 verify.** Why: the verify floor
   is about 58 ms of an 85 ms block-7 round, and one row through the tile
   costs 1.24-1.27x the GEMV. At GEMV efficiency a round is about 72 ms
   (estimate from the round table above). Where:
   `shaders/dequant_int4_mma8.metal` and its dispatch in
   `real_forward_dispatch.rs`. The levers are the ones
   `docs/INT4_SMALL_M_MMA.md` lists: operand build cost per group, group
   sums, and occupancy at one row. Gate: bit equality between a one-row
   decode and the same row inside a verify (decode and verify share the
   tile), `mtp_generation_gate`, and the drift probe. Measure:
   `tslab kernel --bench int4_matmul --rows 1,2,4,8,9` for the curve, then
   `tslab ab` on `qwen38-27b-mtp` with arms `plain`, `mtp2~mtp:2` and
   `mtp4~mtp:4` under the flag.
2. **Choose the block per request, in the server too.** Why: under the
   flag the best block is 4 on code and 2 on prose, so any fixed block loses
   somewhere. `AdaptiveDepth` already moves depth per round in the FFI
   path. Done:
   `turbospark-server --speculative adaptive[:N]` runs the adaptive loop, and
   `TURBOSPARK_ADAPTIVE_DEPTH=LOW,MID,HIGH,DROP` overrides the controller's
   climb and drop thresholds (default `8,6,4,2`, the old constants). Output
   is unchanged by any setting: adaptive changes only how many rows a round
   verifies, and the live arms below decoded identical tokens.

   Tuned offline, not by sweeping live runs. `tslab chain-capture` records,
   once per prompt, how many of the MTP head's next 8 chained proposals the
   target accepts at every position. `tslab depth-replay` then walks the
   speculative loop's round rules over that record for each controller
   setting (fixed 1-8, and a grid of floor 2, caps 4 and 8, and climb and
   drop thresholds) and prices each round from the MTP rows of the round
   table in "DFlash2 and the flat verify" above: 67.5 ms at depth 2 and 129.5
   at 4 flag off, 72.5 and 86.5 flag on, other depths linear (modeled).
   Committed tokens per round in the replay match live fixed-block runs
   within 2%.

   Results on `qwen38-27b-mtp`, four prompts (short, medium, a code edit, a
   quote-back), 512 tokens each, speedup over fixed block 2 under the same
   flag:

   | Setting | Flag off, geomean (worst) | Flag on, geomean (worst) |
   |---------|--------------------------:|-------------------------:|
   | shipped, 2..4, `8,6,4,2` | 0.96x (0.90x) | 1.04x |
   | 2..4, `1,1,1,3` | below 1.0 | 1.07x (1.007x) |
   | 2..8, `3,1,1,2` | below 1.0 | 1.10x (1.007x), above depth 4 modeled |

   Flag off, every adaptive setting loses: a depth-4 round costs 1.9x a
   depth-2 round, and no prompt accepts enough to pay for it. A live tslab
   check (ledger `qwen38-depth-confirm`, M4 Max, AC, 3 clean rounds, code and
   quote-back prompts) agreed on order: fixed 2 37.4 tok/s, shipped 35.3
   (0.945x, replay 0.92x), `3,1,1,2` at cap 4 33.2 (0.888x, replay 0.87x).
   Absolute tok/s ran 12% under the replay on that host. Flag on with the
   tuned 2..8 controller, against today's default (flag off, fixed 2), the
   replay gives code +16%, quote-back +7%, prose -6%: the one-row tile cost
   in lever 1 decides whether the flag-on setting is a net win. Not run
   live: it needs a worker built with the MMA8 tile.
   Measured since, under the tile: a block-8 verify crosses the 8-row MMA
   tile and costs a second pass, so the cap belongs at 7, not 8. Live, cap 7
   is +24.2% over cap 8 on copy-heavy (WIN) and +1.45% on code (NOISE), same
   thresholds ("Measured: the hybrid replay, 2026-10-25" under "Prompt
   lookup"). The runtime now holds an adaptive cap to the tile's rows less
   one while the tile is on (`AdaptiveDepth::tile_cap`), so `adaptive:8`
   under `TURBOSPARK_INT4_MMA8=1` runs at 7. `DEFAULT_ADAPTIVE_CAP` stays 4:
   with the tile off, nothing above depth 4 has paid.
3. **DFlash2 at depth 7 over the flat verify.** Why: a block drafter
   proposes 7 tokens for one pass, and the flat verify makes those rows
   nearly free (31-37 tok/s measured above, 2.1x its flag-off self). Where:
   nothing new beyond lever 1. What is owed is the lossless gate under the
   flag, which has not run for DFlash2. Measure: arms `d2~dflash:2` and
   `d7~dflash:7` on `qwen38-27b-dflash2.gturbo`, with the bit-exact gate on
   generated tokens against block 2.
4. **Prefix reuse across turns on GDN models.** Why: an agent turn that
   appends 500 tokens to 30K of history should prefill 500. Today
   `try_reuse_prefix` refuses to move a GDN model's cursor backwards, so
   reuse fires only when the new prompt extends exactly what was fed. A
   template that re-renders the previous assistant turn (thinking stripped,
   different whitespace) misses and prefills everything. Where: keep a GDN
   snapshot at each turn boundary in the existing snapshot pool
   (`gdn_state.rs`), and let `try_reuse_prefix` restore the newest snapshot
   at or below the common prefix, then prefill the difference. Gate: the
   reuse rate on a real multi-turn chat, and logits after reuse bit-equal to
   a full prefill of the same ids. Measure: TTFT on a cached follow-up
   (this is not a decode tok/s lever).

   Built on branch `feat/gdn-turn-snapshot` (not on `main`), opt-in as
   `TURBOSPARK_GDN_TURN_SNAPSHOT=1`. One snapshot, not a pool: the prefill
   splits 8 tokens before each prompt's end and keeps a full rollback point
   there, and the next turn restores it when its prompt agrees with the fed
   ids past it, rewinding the KV and the MTP drafter too. On Qwen3.8's
   template a client that drops the reasoning diverges 1 token before the
   previous prompt's end, so the 8-token margin always lands inside the
   match. Dense qwen only. DFlash2, `qwen4`, MoE, images and session swaps
   keep the old refusal.

   Gate (`crates/bench/tests/turn_snapshot_gate.rs`, `qwen38-27b-mtp`, MTP
   block 2, three turns with the reasoning dropped): greedy tokens identical
   to a full prefill on every turn, reusing 30 and 66 tokens. Switch off,
   the same gate reuses 0.

   Measured 2026-10-24, M4 Max 36 GB, AC, warm page cache, `qwen38-27b-mtp`
   MTP block 2, a 6,000-token document (`docs/MTP.md`) and four follow-ups,
   32 tokens a turn, replies carried back without reasoning
   (`crates/lab/chats/long-doc.chat.json`):

   | Run | Follow-up TTFT, off | On | Turn 1, on/off |
   |-----|--------------------:|---:|---------------:|
   | `turn_snapshot_ttft_probe`, 2 clean interleaved rounds | 43.6-44.1 s | 0.62-0.75 s | 0.98-1.00x |
   | `tslab ab` (ledger `gdn-turn-snapshot-ttft`, 3 clean pairs) | | -98.5% median, CI [-98.50%, -98.48%] | |

   Each follow-up reused all but the last 30-35 prompt tokens (6,013 of
   6,045 on turn 2). Generated tokens were identical across arms in every
   pair. The cost is memory: peak `phys_footprint` +147 MiB (one GDN
   mirror), which FAILS tslab's 64 MiB memory gate and is not yet counted
   by `context_policy`, so the switch stays off by default until both are
   settled. Without reuse the off arm prefills 6,000 tokens at about 139
   tok/s, which is what makes a long follow-up take 44 s. That is the
   trunk's own rate, not the MTP priming: `tslab profile` (ledger
   `qwen38-prefill-gap`, `corpus:6000`, two runs of 3-4 interleaved rounds,
   few clean because of Spotlight and a system install) read plain chunked
   prefill at 138.2 tok/s (2 clean, 132.7-143.6), the speculative loop's
   batched-priming prefill at 0.96-0.99x of plain in the same round, and
   per-row priming (`TURBOSPARK_MTP_BATCHED_PRIME=0`) at 0.83-0.92x. Faster
   long-prompt TTFT is the dense prefill problem (Steel GEMMs, GDN,
   attention), or the opt-in ANE+Steel split, not a speculation one.
5. **Hybrid lookup plus MTP.** Why: on copy-heavy output a long n-gram
   match proposes 4-8 tokens that are accepted, where the head proposes
   two. Where: a draft source that tries the n-gram first and falls back to
   the head. Price it with the replay before building (section above).
6. **Shared verification across concurrent requests.** Why: mlx-serve
   batches different requests' draft rows into one verify pass, which raises
   aggregate tok/s (its release notes report 185 tok/s over 32 streams on an
   M4 Max). With the flat verify, rows from several requests cost little more
   than one. Where: `crates/server` would need a scheduler that groups
   requests by phase (draft, verify, commit) over one runner with per-request
   KV and GDN state, which `session_pool.rs` partly provides. Aggregate only:
   it can slow a single stream. Not started, and the largest of these.
7. **Interruptible prefill in the server.** Why: a long prompt from one
   client freezes another client's decode. Chunked prefill already exists,
   so the change is to yield between chunks to ready decode steps. This
   improves p95 inter-token latency, not single-stream throughput.

Not for this gap: HTTP or tokenizer tuning (a 27B verify is tens of
milliseconds), and CUDA-side techniques. KV quantization exists
(`--kv-bits`, `docs/TRUBOQUANT.md`) and is a memory lever. Its decode speed
on the 27B at long context has not been measured.

## KFlash: tree verify

Source: kapitsa's `docs/kflash.md` and `docs/kflash-map.md`, main as of
2026-10-21 (their PR 187). KFlash builds a best-first tree from the drafter's
per-position candidates and verifies it in the 8 rows the chain already uses.
Each row attends to the cache plus its own ancestors, recurrent layers start
from the parent row's state, and RoPE uses the row's depth. Acceptance walks
into the child equal to the target's own token, so output stays exact. They
report +6% greedy on the 27B (M4 Pro), mostly from short code and prose
answers, and -5% to +3% on long agent turns.

Why it does not transfer as-is:

- Their extra rows are nearly free. A dense 8-row verify costs 1.1-1.25x a
  decode on their 8x8 MMA tile, so the tree fills rows already paid for.
  Ours is a batched GEMV at c(M) 0.41-0.56 per row at M=8
  (`docs/SPECULATIVE_DECODING.md`). That makes 8 rows cost 3.3-4.5x a
  decode, and only block 2 pays (the sweep above). Here a tree has to win at
  the same row count.
- Four pieces are missing, and each is a new correctness surface:
  - ancestor-masked verify attention (`encode_attention_decode` takes a
    span, not a mask)
  - GDN rows that start from a parent state (`gdn_delta_step_prefill` runs
    the rows in one register loop)
  - per-row RoPE positions (`encode_rope_neox_subdim_rows` is start plus row
    index)
  - compaction of the accepted path's KV rows (only `rewind_by` exists)

  Folding only the accepted path through the lazy tape is the easy part,
  because `gdn_state_fold` is elementwise.
- Their exactness lesson matches ours. The tree-zone attention had to sum a
  row's keys in an order fixed by position alone, or the MoE router flipped
  on last-bit differences between tree schedules. A tree verify here inherits
  the split-KV bit-exactness rule.

#### Measured: tree vs chain at equal rows, MTP replay, 2026-10-21

`crates/bench/tests/kflash_tree_replay_probe.rs`, with the pure sim in
`crates/bench/src/tree_replay.rs`. `qwen38-27b-mtp.gturbo`, greedy, 512
tokens per case. At every position the head drafts a fresh depth-8 chain
and keeps its top-4 per depth. The replay then walks the speculative loop
twice, once for the chain and once for KFlash's best-first tree, with the
same number of proposals.

The token count is exact, because an accepted path is always the reference
prefix. Tree overhead is not modelled, so the tree columns are ceilings. An
MTP head only drafts under a token it was fed, so siblings are leaves and a
tree gains at most one token per round. Built in a HEAD worktree with only
these files. The drafted stream was asserted identical to plain greedy.

| Case | Block | Chain committed/round | Tree | Oracle | Tree/chain |
|------|------:|----------------------:|-----:|-------:|-----------:|
| short-explanation | 2 | 2.133 | 2.133 | 2.522 | 1.000x |
| short-explanation | 4 | 2.393 | 2.522 | 2.943 | 1.054x |
| short-explanation | 8 | 2.450 | 2.783 | 3.012 | 1.136x |
| medium-review | 2 | 2.349 | 2.370 | 2.586 | 1.009x |
| medium-review | 4 | 2.860 | 2.943 | 3.261 | 1.029x |
| medium-review | 8 | 3.048 | 3.346 | 3.531 | 1.098x |
| code (merge sorted lists) | 2 | 2.723 | 2.709 | 2.844 | 0.995x |
| code | 4 | 3.879 | 3.879 | 4.267 | 1.000x |
| code | 8 | 5.020 | 5.172 | 5.626 | 1.030x |

Oracle is the best tree the lattice allows, not a buildable one. Sanity
check: the replay chain at block 2 commits 2.13 and 2.35 per round, and the
live sweep commits 2.16 and 2.30. Every replay head row is primed from the
real trunk hidden, and that edge over the live drafter is small.

What it says:

- At block 2 the tree gains nothing: 0.995-1.009x. The head's top-1 is
  confident enough that best-first keeps the chain.
- The tree helps at long blocks (1.03-1.14x at block 8), but those blocks
  lose to plain decode (0.54-0.74x projected on prose). Across blocks, the
  best tree arm beats the best chain arm by at most 1%.
- Headroom exists. The oracle is 1.04-1.18x the chain at block 2, and the
  target is the head's depth-0 runner-up on 10-12% of prose positions.
  KFlash's per-depth calibration is the lever for that. Its ceiling is the
  oracle row, before tree overhead and the four kernel pieces.

Verdict: not adopted for MTP. Recorded so it is not re-proposed on kapitsa's
numbers. DFlash2 is measured below.

#### Measured: DFlash2 lattice replay, and the check as a lab command, 2026-10-24

The check is now `tslab tree-replay --model M --drafter mtp|dflash`
(`crates/lab/src/tree_replay.rs`, `crates/lab/program.md` new-family item
12). It writes a `tree_replay` record to the lab ledger. The recorders live in
`crates/bench/src/tree_replay_record.rs` and are shared with the probe above.
For DFlash2, `RealForwardRunner::dflash_lattice` reads back the selector's
whole lattice after each draft pass (`crates/runtime/src/draft_lattice.rs`):
the top-16 per depth and every pair score the selector maximizes. The tree
takes q(c | p) as a softmax over p's 16 pair scores, and any node can have
children, so the one-token cap does not apply. Each round asserts that the
lattice's greedy path equals the selector's proposals, and each prompt
asserts that the drafted stream equals plain greedy. Both held.

Measured on the `lab/ext-round-phases` branch (since folded into `main` with the tslab PR), HEAD plus these files, default verify (flag
off), 512 greedy tokens, depth 7 for DFlash2 and depth 8 top-4 for MTP. The
host was eligible for both runs. The token count does not depend on the
verify, so these rows also hold under `TURBOSPARK_INT4_MMA8`.

| Drafter | Prompt | Block | Chain | Tree | Oracle | Tree/chain |
|---------|--------|------:|------:|-----:|-------:|-----------:|
| DFlash2 | short-explanation | 2 | 2.116 | 2.142 | 2.829 | 1.013x |
| DFlash2 | short-explanation | 4 | 2.535 | 2.626 | 3.969 | 1.036x |
| DFlash2 | short-explanation | 7 | 2.639 | 2.960 | 4.697 | 1.121x |
| DFlash2 | medium-review | 2 | 2.370 | 2.370 | 2.876 | 1.000x |
| DFlash2 | medium-review | 4 | 2.893 | 2.943 | 4.414 | 1.017x |
| DFlash2 | medium-review | 7 | 3.030 | 3.346 | 5.447 | 1.105x |
| DFlash2 | copy-heavy | 2 | 2.317 | 2.317 | 2.844 | 1.000x |
| DFlash2 | copy-heavy | 4 | 2.994 | 3.012 | 4.231 | 1.006x |
| DFlash2 | copy-heavy | 7 | 3.580 | 3.821 | 5.626 | 1.067x |
| MTP | copy-heavy | 2 / 4 / 8 | 2.960 / 4.267 / 5.953 | same as chain | 2.960 / 4.303 / 6.024 | 1.000x |

The MTP rows for the two protocol prompts reproduce the probe's table above
to the third decimal, so the command and the probe agree.

What it says:

- At block 2, the flag-off serving block (`DFLASH_SERVING_BLOCK`), the tree
  gains 0-1.3%. Same verdict as MTP there.
- At depth 7, the block DFlash2 serves over the flat verify (lever 3 above),
  the tree gains 6.7-12.1%. That clears the 5% bar on all three prompts,
  copy-heavy included, where the MTP tree gains nothing at any block.
- The oracle is 1.57-1.80x the chain at depth 7. Best-first over softmaxed
  pair scores finds a small part of that, so a better node score (KFlash's
  per-depth calibration) is a separate lever with more headroom than the
  tree shape.

Verdict: open for DFlash2 under the flat verify, not built. The count
clears the bar. The next step is costing the tree: the four kernel pieces
above (ancestor-masked verify attention, parent-state GDN rows, per-row RoPE
positions, KV compaction) plus their overhead on 8 rows. Kapitsa measured
1-6% on dense models. That cost is not measured here. A 1.07-1.12x count gain
minus a 1-6% overhead could be anywhere from a small win to break-even.

#### Costing the tree: DFlash2 at depth 7, 2026-10-24

The question: what does each of the four kernel pieces add to a depth-7
round over the flat verify, and does the 1.07-1.12x count gain survive it?
The round itself is measured above ("DFlash2 and the flat verify"): about 83-87
ms, of which the verify is 66.5 ms and the draft is 15.2 ms.

The GDN piece and KV compaction are timed in
`crates/gpu/tests/tree_verify_cost_bench.rs` (qwen38 shapes, 8 rows, 256 reps
in one command buffer, best of 5 interleaved rounds, two runs that agree
within 0.3%). Host load was 5-6 from system daemons, which a GPU-bound
microbench tolerates; the two runs show it. The bench carries a bench-only
tree build of the GDN verify kernel, `tv_gdn_tree`, which starts node `t`
from the register state its parent left. A thread holds only `Dk / 32 = 4`
floats of state, so all 8 nodes' states fit in registers with no spill and
no recompute. Two parity tests pin it to the shipped kernel. With chain
parents its y is bit-identical. On a branching tree, each node's y is
bit-identical to the shipped kernel run over that node's root path. A
mutation that ignores the parent array fails the second test.

| GDN arm, 48 layers | ms per round | vs shipped | vs chain4 |
|--------------------|-------------:|-----------:|----------:|
| Shipped lazy kernel (chain) | 3.82 | 1.000x | 2.78x |
| chain4: same math, `Dk = 128` unrolled, chain | 1.37 | 0.360x | 1.000x |
| Tree build, chain parents | 1.60 | 0.419x | 1.166x |
| Tree build, branching tree | 1.60 | 0.419x | 1.166x |

The fair cost of the tree is against chain4, because whatever specialization
the tree build gets, the chain can get too: +0.23 ms a round. The tree's
shape does not change the cost.

| Piece | Cost per depth-7 round | How it was sized |
|-------|-----------------------:|------------------|
| GDN rows from the parent's state | +0.23 ms | Measured above, against chain4 |
| GDN conv taps along the path, tape fold over path rows | about 0.05 ms | Arithmetic: a gather of under 10 MB over 48 layers |
| KV compaction | 0.34 ms or less | Measured: 224 single-row copies (7 nodes, K and V, 16 layers). One gather dispatch per buffer would be cheaper |
| Per-row RoPE positions | about 0 | A position array in place of `start + index` |
| Ancestor-masked verify attention | under 0.2 ms, not measured | The verify's attention is 8 rows at 0.41-0.46 ms each (`tslab kernel --bench attn_decode`, ctx 512-1024), about 3.5 ms. The mask touches at most 8 of the 500-1,100 keys a row reads. 0.2 ms assumes a masked tail costs as much as 5% of the whole attention |
| Host tree build | about 0 | It replaces the selector's greedy walk with the same 7 expansions of 16 pair scores. `dflash_lattice` scores all 16 x 16 pairs per depth, which a live build must not do: score children only when a node is expanded |
| Drafter capture gather for the accepted path | about 0.02 ms | Arithmetic: 7 rows of 5 x 5,120 halves |

Total: about 0.6-0.85 ms, roughly 1% of an 85 ms round. That is at the low
end of kapitsa's 1-6%, which fits: their tree also pays for a tree-zone
attention kernel, and here the per-thread GDN state is small enough to
branch in registers.

Projection, not a measurement: tok/s ratio = committed/round ratio /
(1 + overhead). With overhead at 1%, the replay's 1.067-1.121x becomes
about 1.06-1.11x. On the measured flag-on arm that would take
DFlash2 7 from 31.2 to about 34.6 tok/s on short and from 36.8 to about
40.3 on medium. Three things can move it:

- The replay drafter is primed from the trunk's real hidden state, a small
  edge over the live loop (`tree_replay_record.rs`). The chain/tree ratio
  shares that bias. The absolute numbers do not.
- The attention mask is the only piece sized by assumption. If it cost 25%
  of verify attention (0.9 ms), the overhead would be about 1.8% and the
  projection 1.05-1.10x. The verdict holds either way.
- Losslessness under the flag has not been checked for DFlash2 (see the
  flat-verify section). A tree verify inherits that open item. (Closed
  2026-10-25: `crates/bench/tests/dflash_generation_gate.rs`, below.)

Verdict: tree verify pays for DFlash2 at depth 7 over the flat verify, by
this costing. The order to build it, cheapest proof first: the GDN tree
kernel (prototype and parity tests exist in the bench), the attention mask
with a row-against-decode parity test, KV and capture gathers, then the
host tree build in place of the selector walk, and finally a live A/B
against DFlash2 7 under `TURBOSPARK_INT4_MMA8` with a losslessness check.

Side finding, separate from the tree: chain4 is the shipped GDN verify
kernel with `Dk = 128` unrolled. It is bit-identical (parity test) and 2.78x
faster at 8 rows, which would save about 2.4 ms a round (about 2.9% at depth
7), with or without a tree. The shipped kernel loops a runtime
`nPerLane` over `float s[8]`. Since wired for all four GDN delta kernels and
measured in the model: verify fell 69.3 to 66.8 ms a round, DFlash2 7 under
the flag gained 4-5% and prefill 5% (see "`Dk = 128` baked into the GDN delta
kernels" under Item 1).

Reopen condition: the verdict assumes today's verify cost. The item 7 bench
prototype (`docs/INT4_SMALL_M_MMA.md`) measured 8 rows at 1.25-1.29x a
1-row GEMV. If that lands in the verify, blocks 4 and 8 stop losing and the
1.03-1.14x tree gains at those blocks become reachable. Re-run this replay
and the block sweep then.

### Built and measured, 2026-10-25

`TURBOSPARK_DFLASH_TREE=1`, opt-in when built, greedy fixed-depth DFlash2
rounds only. Default on since 2026-10-26 ("Promotion to the default"
below), `TURBOSPARK_DFLASH_TREE=0` keeps the chain.
A round verifies a best-first tree of `block` nodes in the chain's rows
(anchor plus 7 at depth 7), so the MMA8 tile shape is unchanged.

The pieces, each with a bit-parity test and a mutation it catches:

- `gather_rows_fp16`: in-place compaction of the accepted path to rows
  `0..keep`, one threadgroup walking rows in order with a barrier between
  them (a flat dispatch races on paths like `[0, 2, 3]`).
- `FC_GDN_TREE` (99) on `gdn_delta_step_prefill` and `gdn_conv_mix_prefill`:
  a row starts from its parent's register state, and its conv taps read its
  root path, then the tail. Chain parents are the shipped kernels' bits.
- `rope_neox_subdim_rows_positions`: each row at its depth.
- `FC_ATTN_TREE` (77) on the GQA-shared decode partial: keys past the base
  map to the row's ancestor slots, at the chain row's own `seq_len`, so the
  chunking and loads in flight are the chain's. The identity path is the
  chain row's bits. The barrier override declines and keeps the chain.
- Host: the best-first builder scores a node's 16 children only when it
  expands the node, and it and the acceptance walk are pinned to
  `tslab tree-replay`'s (`crates/bench/tests/dflash_tree_pin.rs`). The
  accepted path is gathered (each attention layer's K and V, the verify
  tape, the lazy tape's k and delta, the drafter's capture, `scratch.x`)
  before the unchanged round-end.

Refused, chain kept: eager tape, the barrier attention, a GDN key dim not
baked at 128, adaptive depth, budget-shortened rounds.

Synthetic gates (`crates/runtime/tests/real_forward_qwen35_dflash_tree.rs`,
on a wide-head fixture because the kernels need head_dim 128): a chain-shaped
tree round is the chain round byte for byte; every branching slot is the
chain verify over its root path; branching and teacher-forced multi-round
walks commit the sequential stream. Dropping the compaction, the K/V gather
or the delta gather each fails them.

Real install (`qwen38-27b-dflash2`, M4 Max, AC):

- Lossless: `dflash_generation_gate` matches sequential decode token for
  token at blocks 2 and 7 on a code and a prose prompt, tree on and off,
  under `TURBOSPARK_INT4_MMA8=1` and `TURBOSPARK_BATCHED_GEMV=0`.
- Acceptance (`tslab ab`, experiment `kapitsa-levers-c10`, 128 tokens,
  identical on every repeat since greedy):

| Prompt | Chain committed/round | Tree committed/round | Ratio | Replay |
|---|---:|---:|---:|---:|
| medium-review | 2.98 | 3.28 | 1.10x | 1.07-1.12x |
| short-explanation | 3.05 | 3.20 | 1.05x | |

- Cost: 9 of 10 runs were DIRTY (foreign CPU), so the A/B verdict is
  INCONCLUSIVE and no tok/s is quoted. The phase table of the least-loaded
  runs reads the verify at 64-67 ms/round for the chain and 68-72 ms for the
  tree (+5-7%), plus about 0.6 ms of compaction in the rollback phase, where
  the costing above assumed about 1%. On those readings the net is even on
  short-explanation and about +4% per token on medium-review.

Verdict (2026-10-25): built, lossless, and the acceptance the replay
predicted is real, but the verify overhead seemed to eat most of it. The
quiet-host re-measure below overturns the overhead reading.

#### Quiet-host re-measure, 2026-10-26

The A/B (`tslab ab`, arms `chain~dflash:7:TURBOSPARK_INT4_MMA8=1,spec_stats=1`
and `tree~dflash:7:TURBOSPARK_INT4_MMA8=1,TURBOSPARK_DFLASH_TREE=1,spec_stats=1`,
`--metric decode_tok_s --decode 128`), experiment `kapitsa-quiet-c1`. Every
run was clean. M4 Max, AC power, warm page cache, swap 14.3 GB used and
steady, worktree HEAD `f9751f12`.

| Prompt | Null floor | Chain tok/s | Tree tok/s | Verdict |
|---|---:|---:|---:|---|
| medium-review (426 tokens) | 0.02% (`1793052090530-86510-22`) | 40.19-40.31 | 42.97-43.86 | **WIN +8.57%**, CI [+6.90%, +8.85%], 3/3 pairs (`1793052167891-94971-22`) |
| short-explanation (62 tokens) | 0.18% (`1793052226998-3692-22`) | 41.42-41.53 | 42.61-42.94 | **WIN +3.07%**, CI [+2.85%, +3.69%], 3/3 pairs (`1793052286041-11187-22`) |

Phase table of the same runs, ms per round (`spec_stats`, ranges over the
three measured runs per arm):

| Phase | Chain, medium-review | Tree, medium-review | Chain, short | Tree, short |
|---|---:|---:|---:|---:|
| draft | 9.53-9.63 | 10.01-10.42 | 9.36-9.44 | 9.81-10.02 |
| verify | 62.23-62.33 | 62.54-63.62 | 61.93-62.08 | 62.30-62.58 |
| accept | 1.40-1.43 | 1.56-1.58 | 1.44-1.46 | 1.55-1.61 |
| rollback | 0.15 | 0.66-0.71 | 0.15 | 0.82-0.86 |
| bonus | 0.45-0.46 | 0.01 | 0.45-0.46 | 0.00 |
| committed/round | 2.98 | 3.28 | 3.05 | 3.20 |

**The 4-5 ms of extra verify does not exist on a quiet host.** The tree
verify is 0.3-0.6 ms (0.5-1%) dearer than the chain's, with one tree run at
+1.3 ms, and the whole round is about 1.1 ms (1.5%) dearer: verify +0.4,
draft +0.5, compaction in rollback +0.5-0.7, accept +0.15, less the chain's
0.45 ms bonus phase, which the tree does not run. The 64-67 against 68-72 ms
read on 2026-10-25 came from contended runs. With a 1.5% dearer round and
5-10% more committed per round, the per-token gain matches the verdicts.

**Where the extra time goes** (dispatch profile, `tslab profile` with
`TURBOSPARK_DISPATCH_PROFILE=1`, experiment `kapitsa-quiet-p1`, runs
`1793052356928-18808-10` chain and `1793052365432-18808-13` tree,
medium-review, 128 tokens, normalized by the 45 and 41 rounds each run
took including the warmup's two). The profiler waits on every dispatch, so
the absolute times are inflated and only the differences rank:

| Kernel, per verify round | Chain | Tree | Difference |
|---|---:|---:|---:|
| whole batched verify | 63.39 ms | 63.52 ms | +0.13 |
| `gdn_delta_step_prefill` (tree register-state variant, `FC_GDN_TREE`) | 1.45 | 1.69 | +0.24 (+16%) |
| `gdn_conv_mix_prefill` (tree conv taps) | 0.21 | 0.24 | +0.03 |
| `attention_decode_gqa_partial` (`FC_ATTN_TREE`) | 2.03 | 2.06 | +0.03 |
| `rope_neox_subdim_rows_positions` against `rope_neox_subdim_rows` | 0.09 | 0.08 | -0.01 |
| INT4 GEMMs and table prep | 55.6 | 55.5 | -0.1 (noise) |
| `dflash tree compact` pass (`gather_rows_fp16`, its own command buffer) | 0 | 0.45 | +0.45 |

Ranked: the compaction pass, then the tree GDN delta kernel. The per-row
tree attention and the position table cost nothing measurable. The draft
phase's +0.5 ms is not GPU work: the drafter's own dispatches read the same
in both arms (7.51 and 7.50 ms per round) and the drafter context write
grows by 0.14 ms because the tree commits more rows per round. The rest is
presumably the host's best-first builder, which was not timed separately.

If the tree is promoted, the two places to trim are folding the compaction
into the round-end command buffer instead of its own `commit_and_wait`, and
the GDN tree kernel's parent-state load. Together they are under 1 ms of a
75 ms round.

#### Promotion to the default, 2026-10-26

The user decided to promote the tree. The runs above force
`TURBOSPARK_INT4_MMA8=1` in both arms, and that is not the shipping
default, so the flip was gated again in the default configuration, after
the code change, with the opt-out as the baseline: arms
`chain~dflash:7:TURBOSPARK_DFLASH_TREE=0,spec_stats=1` and
`tree~dflash:7:spec_stats=1`, `--metric decode_tok_s --decode 128`,
experiment `dflash-tree-default-d1`. M4 Max, AC power, warm page cache
(`no_reads`), swap 14.3 GB used and steady, branch
`claude/dflash-tree-default` on `76e03aab` plus the flip.

| Prompt | Null floor | Chain tok/s | Tree tok/s | Verdict | Peak footprint |
|---|---:|---:|---:|---|---:|
| medium-review (426 tokens) | 2.04%, 2 clean pairs (`1793059042395-5641-40`) | 18.01-18.24 | 19.68-20.08 | **WIN +9.22%**, CI [+9.10%, +10.48%], 3/3 pairs (`1793059215378-52766-28`) | +0.85 MB, passed |
| short-explanation (62 tokens) | 0.68% (`1793059341778-68001-22`) | 18.74-18.75 | 19.54-19.56 | **WIN +4.32%**, CI [+4.29%, +4.38%], 3/3 pairs (`1793059477494-79368-34`) | +0.75 MB, passed |

Committed per round matches the MMA8 runs (2.98 to 3.28, 3.05 to 3.20).
The verify reads about 142 ms per round in both arms against 62 ms under
MMA8, so the round costs more and the tree's fixed overhead weighs less,
which is why the gains are slightly larger here. Every tree run reported the
`dflash_tree` engagement line and no chain run did.

Losslessness, after the flip: `dflash_generation_gate`
(`TURBOSPARK_INT4_MMA8=1 TURBOSPARK_BATCHED_GEMV=0`) matches sequential
decode token for token at blocks 2 and 7 on both prompts, once with the
switch unset (tree) and once with `TURBOSPARK_DFLASH_TREE=0` (chain). The
synthetic suite (`real_forward_qwen35_dflash_tree.rs`) now runs the default
with the switch cleared, and `real_forward_qwen35_dflash_tree_off.rs` holds
the opt-out. Reverting the default to off fails five of the six default
tests.

Not covered: sampled runs and adaptive depth never take the tree, so they
are unchanged. No other DFlash2 install, prompt length past 426 tokens, or
chip was measured.

### Compaction off the critical path, and the tile-off A/B (2026-10-26)

The tree compaction (`dflash_tree_compact`) ran in its own command buffer
with `commit_and_wait`. It is now committed without a wait: every reader of
what it writes is later GPU work on the same queue (the fold, the next
context write, the next verify). Its one host reader, the rollback's copy of
residual row `keep - 1`, is served by a one-row host move, so `scratch.x` is
no longer gathered on the GPU (no DFlash2 reader depends on that row; only
the MTP head reads row 0, and dropping the move fails no test). The eager
tape's `b` is no longer gathered either: the tree runs only on the lazy
tape, whose fold never reads it (48 fewer dispatches a round).

Lossless on the real install with the tree on, tile on and tile off
(`dflash_generation_gate`, `TURBOSPARK_BATCHED_GEMV=0`). `tslab ab` on
`qwen38-27b-dflash2`, medium-review and short-explanation, 128 tokens, M4
Max on AC:

| Arms | Experiment | Verdict |
|---|---|---|
| chain vs tree, `TURBOSPARK_INT4_MMA8=1` | `tree-compact-async` | WIN +4.80%, 95% CI [+4.59%, +6.46%], 3 clean pairs |
| chain vs tree, tile off (knob unset) | `tree-tile-off-2` | WIN +8.67%, 95% CI [+6.21%, +9.07%], 3 clean pairs |

Both pool the two prompts, so they do not compare directly with the
per-prompt +8.6% and +3.1% above. The rollback phase, where the compaction
sat, now reads 0.17-0.32 ms a round for the tree against 0.16-0.25 for the
chain, about +0.05 ms, where it read +0.5-0.7 ms before.

Tile off, a tree round verifies the same 8 rows as the chain (140-148 ms
against 140-150 ms of verify a round) and commits 5-10% more, so the tree
does not regress a DFlash2 user without the tile. Depth 7 itself is the
cost there: the lossless gate's informal clocks read 0.81x sequential on
its prose prompt at block 7 with the tile off, for chain and tree alike.
The tree has since become the default (PR #207); which depth DFlash2
serves without the tile is the open decision.

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

Run 1 (all six arms: the four below plus the two sequential rows further
down; 5 measured rounds, 1-minute host load 0.34-2.73). The
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
  noise on the medium one. The block sweep below shows rollbacks are NOT
  rare at block 2 (49-60% of rounds), and that the gain does not grow with
  the rollback rate: it stays at a few percent whatever the block.
- The fused FFN does nothing in speculative mode: lazy-on with and without
  it differ by 0.1-0.2% in run 2, and run 1 straddles 1.0.
- The fused FFN is worth 1.4-2.2% on plain sequential decode. Small, but
  consistent in sign and a little above the spread.
- Not measured: ms/round, the long-synthesis case, and the MoE shared-expert
  fusion (this is a dense install). Rollback counts and blocks 4 and 8 are
  in the sweep below.

#### Measured: MTP block sweep, lazy tape off and on, 2026-10-20 and 21

Same install, flags and machine as above, AC power, `TURBOSPARK_SPEC_STATS=1`
for the round and rollback counts (the speculative loop already prints
them; the bench just forwards stderr). `--speculative 2|4|8` against a
speculation-off reference, fused FFN at its default, 3 measured rounds after
a discarded warmup, arm order rotated. Binary built from the gpu, runtime and
bench sources of `e81cba38` (the docs-only commit above it changes none of
them). The host was NOT quiet for part of this sweep: 1-minute load reached 57
in the warmup round and 17-46 across round 1 of the short case, from other
sessions. Rows below drop every run whose end-of-run load exceeded 8 (n shows
what is left); the rounds and rollback columns are deterministic per arm and
unaffected. Unfiltered medians agree on every ratio to within 0.015x.

| Case | Block | Rounds | Rollbacks | Accepted/round | Eager tok/s | Lazy tok/s | Lazy/eager | Lazy vs off |
|------|------:|-------:|----------:|---------------:|------------:|-----------:|-----------:|------------:|
| short-explanation | 2 | 225 | 134 (60%) | 1.16 | 31.85 | 32.75 | 1.028x | 1.50x |
| short-explanation | 4 | 205 | 187 (91%) | 1.38 | 18.22 (n=1) | 18.71 | 1.027x | 0.86x |
| short-explanation | 8 | 196 | 196 (100%) | 1.48 | 12.03 | 12.20 | 1.014x | 0.56x |
| medium-review | 2 | 367 | 178 (49%) | 1.30 | 34.25 | 34.86 | 1.018x | 1.61x |
| medium-review | 4 | 307 | 253 (82%) | 1.75 | 21.02 | 21.55 | 1.025x | 0.99x |
| medium-review | 8 | 292 | 288 (99%) | 1.89 | 13.93 | 14.07 | 1.011x | 0.65x |

Speculation off: 21.82 tok/s (short), 21.71 tok/s (medium). Every arm of a
case generated the same token count (488 and 845; off 505 and 846), so the
streams agree across blocks and tape modes.

What the sweep says:

- The original prediction was wrong. The lazy tape was
  expected to matter most where rollbacks dominate, and it does not: its gain
  is 1.8-2.8% at block 2 (49-60% rollbacks), 2.5-2.7% at block 4 (82-91%) and
  1.1-1.4% at block 8 (~100%). Eliminating the blocking restore and the
  replay is real but small next to whatever else a round costs. What that
  other cost is was not measured here.
- Block 2 is the only block that pays. Block 4 is 0.86x (short) to 0.99x
  (medium) of plain decode, and block 8 is 0.56-0.65x. Acceptance per
  position falls off quickly (0.76, 0.53, 0.48, 0.47 on the short case), so a
  longer block mostly adds rows that get thrown away. The lazy tape makes the
  loss slightly smaller; it does not turn it into a win.
- This is the same conclusion `docs/MTP.md` reached for the eager path
  (batching loses at blocks 8 and 15), now with the restore cost mostly
  removed, which says the restore was not the dominant term.

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

### `Dk = 128` baked into the GDN delta kernels, 2026-10-24

ADOPTED, default on, `TURBOSPARK_GDN_DK128=0` opts out (the baked build prints
`[engaged] gdn_dk128`). Found while costing
a tree verify for DFlash2 ("Costing the tree" above,
`tree_verify_cost_bench.rs`): a bench-only copy of the lazy verify kernel
with the per-lane state count hard-coded to 4 was 2.78x faster and
bit-identical. The shipped kernels loop a runtime `nPerLane = Dk / 32` over
`float s[8]`.

What changed: function constant 98 (`FC_GDN_DK`) makes `Dk` a compile-time
constant in `gdn_delta_step_decode`, `gdn_delta_step_prefill` (eager and the
lazy RECORD build) and `gdn_state_fold`. The host bakes it only when
`key_head_dim == 128`, every shipped qwen GDN checkpoint. Other key dims keep
the runtime loop. The arithmetic and its order do not change, so the baked
build is bit-identical. `src/gdn_tests.rs` pins that on Metal for all four
kernels (y, state, delta tape), checks the `Dk` 64 and 256 fallback, and
guards that the two builds are separate cache entries. Dropping `Dk` from the
key made the bit test pass on one pipeline twice, and only that guard failed.

Kernel bench (`gdn::tests::baked_dk128_speed_at_qwen38_shapes`, qwen38
shapes, arms interleaved in one process, three runs): decode 3.7-3.8x, eager
prefill 1.9-3.5x, lazy prefill 1.9-3.8x and fold 2.7-6.9x at 8 and 16 rows.
The host was busy (other sessions' builds, GPU utilization 94-99%), so these
are ratios, not quotable times.

Model A/B (`tslab run-experiment` plus `confirm`, bit-exact gate passed on
every spec, M4 Max 36 GB, AC, load under 1, three clean pairs each, workers
built from branch `lab/gdn-dk128`: e0f5de70 plus this change):

| Spec, install | Workload | Runtime Dk | Baked | Median | Ledger id |
|---------------|----------|-----------:|------:|-------:|-----------|
| DFlash2 7, MMA8 on, `qwen38-27b-dflash2` | short, 256 decode | 33.2 tok/s | 34.6 | +4.07% WIN | 1792872140163-38512-21 |
| same, confirm | medium, 256 decode | 37.6 | 39.8 | +4.95% WIN | 1792873152767-30621-17 |
| Plain decode, `qwen38-27b` | short, 128 decode | 20.7 | 20.7 | -0.36% NOISE | 1792872315739-96209-21 |
| Prefill, `qwen38-27b` | long-synthesis | 147.5 tok/s | 155.6 | +5.33% WIN | 1792872593319-9671-21 |
| same, confirm | corpus 2000 | 156.7 | 165.0 | +5.36% WIN | 1792873269182-50140-17 |

Where the DFlash2 round goes (`TURBOSPARK_SPEC_STATS=1`, short): verify 69.3
to 66.8 ms, matching the 2.4 ms the tree bench predicted, and draft 10.9 to
10.1 ms. Committed tokens per round are unchanged (2.71), as a bit-exact
change requires. Plain decode does run `gdn_delta_step_decode`, so its NOISE
means that kernel's in-model share is under the 2% floor, despite the 3.8x
kernel ratio.

The first DFlash2 run-experiment (ledger 1792868671395-16692-5) ran beside
another session's 27B probe, its quality runs timed out, and tslab recorded a
`discard`. It measured nothing. Ignore it.

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
`batched_layers.rs`) stays deferred. Item 1's paired A/B is in, and
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
- Item 7 (integer-code 8-row verify GEMM): NOT BUILT, now the leading
  candidate (see "Measured: kapitsa on this machine"; the text below predates
  that measurement and its "project, not a bench" verdict was about cost, not
  value). The in-tree
  evidence reframes it. `dequant_int4_mma.metal`'s header records that its
  OWN tile lost at every width because packed INT4 cannot be
  simdgroup-loaded, so every element pays a dequant into threadgroup memory
  that is independent of B -- and the header explicitly scopes the verdict:
  the shape loses, not matrix hardware, and the missing piece is weights
  "already in a loadable format".

  CORRECTED 2026-10-21 from kapitsa's `src/kernels/verify8.metal` (main):
  the earlier reading here (a repack at 2 or more bytes per code) was
  wrong. `qmm8*` reads the MLX affine layout directly, with no repacked
  weight image, and that layout is ours (low nibble first, groups of 64,
  BF16 scale and bias). Three things replace the threadgroup dequant:
  - MMA operands are built in registers through `thread_elements()`, so
    nothing goes through `simdgroup_load` or threadgroup memory.
  - A code becomes an exact half in one or two integer ops: OR'd with
    `0x6400` it is `1024 + q`, or bare it is the subnormal `q * 2^-24`
    (their default dense GEMM). The fp32 epilogue per group applies
    `s * dot + b * sum(x)`, with the group sums of the 8 rows precomputed.
  - The permutation goes into the ACTIVATIONS. A prep pass writes the 8
    rows' inputs in fragment order, so one lane's A operand for a whole
    group is 16 contiguous codes (one 8-byte load).

  So the weights stay 4-bit and in place, and the shape is a port, not a
  new format. Not exact against our GEMV: the MMA chain sums in another
  order and the activations are half. Their own timings are not in the
  repo (`tools/bench_qmm8.cpp` prints them, none are recorded). It is also
  KFlash's precondition (section above): only a verify whose extra rows
  are nearly free lets a tree pay at blocks above 2.

  MEASURED 2026-10-21, bench prototype `tests/int4_mma8_reg_bench.rs`, our
  own reimplementation: 8 rows cost 1.25-1.29x a 1-row GEMV at the qwen38
  shapes, against 3.1-3.6x for the shipped GEMM. One row through the same
  tile costs the same (1.24-1.27x), so rows are free and the 25% is kernel
  efficiency. Subnormal codes match the shipped GEMM's error against f64,
  with 99.2-100% of outputs bit-equal, and a row's bits do not depend on
  the rows beside it. Technique, results, exactness route and the
  per-format checklist: `docs/INT4_SMALL_M_MMA.md`.

  WIRED 2026-10-21 behind `TURBOSPARK_INT4_MMA8=1` (default off) for decode
  and verify both, so speculation stays lossless (`mtp_generation_gate`
  passes on the 27B). End to end on the two prose protocol cases, block 4
  under the flag runs 1.52-1.54x its flag-off self, but the best flag-on arm
  is still 6-16% behind flag-off block 2: plain decode pays 19-23% for the
  tile. After routing only matrices of 4096+ rows to the tile, a code
  prompt runs 16% faster than the default at block 4, while prose stays
  9-12% behind. Picking the block per request is the open step. Tables in
  that page.
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
