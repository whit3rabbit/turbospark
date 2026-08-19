# DFlash2 speculative decoding on Metal, for the dense `qwen3_5` family

Port Inco AI's DFlash2 drafter (`incoai/Qwen3.8-27B-DFlash2`) into this engine as a
second drafter beside the shipped MTP head, wired into the existing speculative loop
behind a CLI choice, with byte-identity and clock gates against the MTP path.
Everything lands in the repo's measured-decision style: facts page first, fixture
before download, gates before conclusions.

## What DFlash2 is (established during research)

- A ~2B BF16 draft model for exactly the checkpoint this engine runs as dense
  `qwen3_5`: 5 sliding-window attention layers (hidden 5120, head_dim 128, 32q/8kv
  heads, window 2048, RoPE theta 1e7, eps 1e-6), `block_size 8`.
- Two DFlash2 additions over v1: a 2-tap grouped dynamic depthwise conv
  (`conv_kernel_size 2`, `conv_group_size 16`) wrapped around every attention and
  MLP sublayer, and a top-16 candidate selector (rank-256 codebooks, bilinear edge
  scoring, greedy path walk at temperature 0).
- Eagle3-style conditioning: the drafter consumes target hidden states captured at
  trunk layers `[5, 19, 33, 47, 61]`, fused by a `fc` projection; its per-layer
  context K/V are computed directly from those states via the drafter's own k/v
  rows (vLLM `precompute_and_store_context_kv`). The drafter never runs over the
  context; each round is ONE forward over `1 + block` query rows (anchor row +
  mask-token rows, `mask_token_id 248070`).
- Shares the target's embedding and LM head (the ~2B = backbone + fc + selector +
  convs; this also matches the MTP head's sharing precedent).
- Metal feasibility is proven in the wild: llama.cpp's `draft-dflash` reports
  1.77-1.85x on an M5 Pro (acceptance 4.9-5.1), and oMLX ships it. No CUDA-only
  piece exists.
- The verify half already exists here unchanged: `produce_batched`
  (`crates/runtime/src/families/qwen/batched.rs`), `checkpoint`/`rollback`, and
  `run_raw_completion_speculative` (`crates/runtime/src/speculative.rs`).

Scope boundaries: greedy-only (the loop's existing sampling gate covers both
drafters; lossless sampled rejection sampling stays future work); dense INT4
`qwen3_5` only (the same refusals `MtpState::speculation_blocker` applies); the
in-flight uncommitted Gemma4 batched-prefill work is untouched.

## Phase 0: Facts (no behavior change)

1. Read the real checkpoint header and file list via the HF API (KBs, the probe
   methodology): exact tensor names/shapes/dtypes, whether `mask_embedding` ships
   as a sidecar, confirm no embed/lm_head tensors. Pin the norm convention off the
   weight means (plain vs centered; Gotcha 50's lesson, do not assume).
2. Read (never copy; Apache-2.0 references, this repo is MIT - the MTPLX
   precedent) vLLM main's `qwen3_dflash.py` + the DFlash speculator, the PR-52816
   `qwen3_dflash2.py`/`dflash2/speculator.py`, and llama.cpp's `draft-dflash`.
   Cross-check the two implementations against each other (`gguf_fused_gate_network`
   methodology). Pin exactly: (a) the anchor row's position and input (target
   states vs embedding) and the mask-row count; (b) the `fc`/`hidden_norm`/
   fused-KV recipe and which rows get context-KV rewritten after acceptance;
   (c) conv placement (`prepare`/`finish` around each sublayer) and the exact
   tap formula; (d) the selector's unary/edge/walk formulas at temperature 0.
3. Write `docs/DFLASH2.md` (facts page, nothing projected) and add the pointer
   edits to `docs/MTP_SPECULATIVE.md` and AGENTS.md at the END of the effort,
   with measured numbers.

## Phase 1: Repack the drafter into an install

1. `crates/repack`: extend the qwen38 walk with an optional extra
   `(header, source)` pair streaming from `incoai/Qwen3.8-27B-DFlash2`, written
   under a `dflash.*` namespace (the source names are unprefixed `model.layers.*`,
   unlike `mtp.*`). Backbone + `fc` rank-2 through the existing INT4 group-64
   quantizer; rank-1 norms through `narrow_raw_to_bf16`; selector codebooks
   `[vocab, 256]` resident BF16 for v1 (host gathers); conv tensors BF16.
2. Fixture first (`crates/repack/CLAUDE.md` Gotcha 8), then the network test
   `crates/repack/tests/dflash2_checkpoint_network.rs` producing
   `~/models/qwen38-27b-dflash2.gturbo`. Guard BOTH writers (the MTP lesson:
   `both_writers_carry_the_mtp_head` is the pattern; the streamed writer dropped
   the head once). Presence is read off the resident index, no manifest field.

## Phase 2: GPU kernels

1. `crates/gpu/src/dflash_conv.rs` + `shaders/dflash_conv.metal`: the grouped
   dynamic depthwise conv (prepare = tap-0 + coefficient projection, finish =
   tap-1 with stored coefficients), following the module pattern: `include_str!`,
   function constants with a constants key carrying every baked value (crate
   Gotcha 1), parity test `crates/gpu/tests/dflash_conv_parity.rs` with
   mutation cases and a real-shape case.
2. Verify reuse rather than new code for: batched INT4 GEMMs
   (`dequant_int4_gemm_simd`, M+1 rows), the sliding-window decode attention with
   ring addressing (`FC_ATTN_RING_CAP`) at head_dim 128 / theta 1e7 (RoPE is
   parameterized; new constant keys only), and the codebook gathers (host reads
   for v1).

## Phase 3: Runtime drafter (`crates/runtime/src/families/qwen/dflash.rs`)

1. `DflashState` following `MtpState`'s doctrine: its own 5-layer SWA
   `KvCacheManager` (rings at `min(max_context, 2048 + 128)`), capture buffers,
   block scratch, selector tables; allocates NOTHING unless the drafter is asked
   for at open (frozen oracle rows must not move).
2. Target-state capture: taps at layers `target_layer_ids` inside
   `produce_real_qwen` and `produce_batched` (a 5x rowsx5120 strided copy per
   pass; zero dispatches when off). Captured states also drive `prime_drafter`
   (context-KV rows for the prompt, one fused-KV GEMM per layer) and the
   post-accept rewrite of committed rows.
3. The one-pass draft forward: anchor + mask rows through the 5 layers with
   batched GEMMs, per-row attention into the drafter's ring KV (within-block
   causal, serial dispatches in one command buffer), conv prepare/finish around
   each sublayer, final norm, trunk `lm_head`; logits readback; host top-16,
   edge scoring, greedy walk (the whole selector is microseconds on host for v1).
   One `gpu::autorelease_pool` per pass.
4. Correctness instrument: `MFERENCE_DFLASH_DUMP` + a `scripts/dflash_bisect.py`
   (the `mtp_bisect.py` pattern) teacher-forcing the drafter against the target's
   real next tokens, so a convention bug reads as a number, not fluent garbage.

## Phase 4: Loop and CLI wiring

1. Extend `SpeculativeProducer` (`producer.rs`) with a block-draft mode:
   `draft_block(&mut self, anchor, base, block, proposals: &mut Vec<TokenId>)`
   plus a kind probe; the MTP implementor keeps the step loop, DFlash2 overrides.
   `run_raw_completion_speculative` branches only on the draft phase; verify,
   accept/commit, rollback, bonus, and rewind stay shared.
2. Drafter policy at open: extend the speculation plumbing
   (`open_with_slot_policy_and_speculation`, `MtpDraftPolicy` becomes a drafter
   enum or gains a sibling) with `MFERENCE_DFLASH_DRAFT` for harnesses; every
   MEASURING caller keeps drafter Off (Gotcha 35 doctrine).
3. CLI: `--speculative-drafter mtp|dflash` (default mtp), hard-failing when the
   install cannot serve the choice. Five places in `crates/invocation` (Gotcha 14)
   plus `tests/usage_and_status.rs`'s count assertion.

## Phase 5: Gates and the verdict

1. `crates/bench/tests/dflash2_generation_gate.rs`: DFlash2 speculative stream
   byte-identical to the non-speculative greedy reference (the `mtp_generation_gate`
   pattern).
2. `crates/bench/tests/dflash2_accept_length_probe.rs`: per-position acceptance
   curve at blocks 2/4/8 against the INT4 target, sequential vs batched verify,
   first-proposal acceptance floor (the broken-drafter guard), and an interleaved
   clock vs the MTP arm (warmup discarded, AC, ratio-not-absolute per Gotchas
   20/22; record power source and thermal pressure).
3. Rollback probe still green (`rollback_probe.rs` on the dflash install), and a
   determinism check (same generation twice on one warm runner - Gotcha 27's
   lesson: the drafter adds hidden state; ask what its ordering depends on).
4. Record the measured verdict in `docs/DFLASH2.md` (including the rollback-term
   reading at block 8 and whether it beats the shipped 1.44x), and update
   `docs/MTP_SPECULATIVE.md` and AGENTS.md pointers.

## Verification per phase

`cargo fmt --check`, `cargo clippy --workspace --tests`, `cargo test --workspace`
after every phase; the phase's own network/gate tests with their env vars; both
real-model smokes (greedy + sampled at CLI defaults) on the dflash install at the
end. ASCII, no em dashes, files under ~400 lines.

## Known risks, named up front

- The exact anchor-row input and post-accept KV-rewrite recipe is the one
  architectural unknown left; Phase 0 resolves it from two independent
  implementations before any code.
- Acceptance against this INT4 target may sit below the published 5.34 (drafter
  trained against the BF16 checkpoint); losslessness is unaffected either way,
  and the probe measures it rather than assuming it.
- The composite math says the payoff over the shipped 1.44x may be small at
  block 2 and gated by the recurrent rollback term at block 8; the gates in
  Phase 5 are what turns that into a measured answer, and a negative result
  still lands as a documented page.