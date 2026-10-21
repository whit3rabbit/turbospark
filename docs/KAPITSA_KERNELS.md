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
- Real-model paired A/B: PENDING. Blocked on a quiet AC-powered host and a
  `qwen38-27b-mtp` install; neither was available when this landed. The row
  stays a placeholder until then, per the no-fabricated-numbers rule.
  The fold pass is committed asynchronously (`pass.commit()`) so CPU drafting
  overlaps with GPU state updates without blocking per round. The first round
  after any eager advance (a prefill, a fresh turn) still pays the blit.

| Arm | tok/s | ms/round | rollbacks | provenance |
|-----|-------|----------|-----------|------------|
| eager (flag 0) | pending | pending | pending | pending |
| lazy (default) | pending | pending | pending | pending |

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
flag combinations with item 1. Real-model paired A/B: PENDING (AC power and
a quiet host were not available). The dispatch-count deltas are static
facts: -3 per dense layer, -3 per MoE layer's shared expert.

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
