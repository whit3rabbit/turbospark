---
title: Add a New Model Family
description: "Bring a new checkpoint architecture from config probe to decode gates, following the qwen4_exp (Qwen3.8-Flash-Next) bring-up as the pattern to copy."
diataxisType: "howto"
---

<!-- authored: howto lane, grounded in the qwen4_exp bring-up. Sources: crates/model-io/src/arch_baselines/{mod,qwen}.rs, crates/repack/src/gguf_names/mod.rs, crates/repack/src/synthetic_qwen/qwen4.rs, crates/runtime/src/families/qwen4/{mod,produce}.rs, crates/runtime/tests/real_forward_qwen4.rs, crates/repack/tests/qwen4_config.rs -->

## Goal

Get a new model architecture running end to end: config parsed, install
written, decode flow wired, and the real-install gates (memory oracle and
quality gate) passing. This guide walks the path the `qwen4_exp`
(Qwen3.8-Flash-Next) bring-up took, naming its files as the pattern to copy
at each step.

A "family" here is an `ArchConfig` baseline plus a decode flow, selected by
`ArchConfig.family` (a `ModelFamily` variant). Most new checkpoints are
NOT new families: `Qwen/Qwen3.8-27B` needed zero changes to
`crates/model-io`, `crates/gpu` or `crates/runtime` because its
`text_config` matched a shipped baseline. Step 1 exists to find that out
before you download anything.

## Prerequisites

- The verify-a-change workflow: the four-command gate
  (`cargo build --workspace`, `cargo test --workspace`, `cargo fmt --check`,
  `cargo clippy --workspace --tests`), plus per-family real-model gates when
  the decode path moves. See the verify-a-change guide.
- A published checkpoint to probe. Its `config.json` is a few KB over HTTP;
  the full stream for the exemplar family was 68 GiB.
- macOS with a Metal device for the runtime steps (`crates/runtime`'s real
  flows and `crates/gpu` are macOS-only).

## Steps

### 1. Diff the checkpoint's config against existing baselines first

Fetch the checkpoint's `config.json` (a few KB) and parse it with the
existing config parsers before downloading any weights. The question to
answer: does this parse to a baseline that already exists?

The pattern is `crates/repack/tests/qwen4_config.rs`. It is the Phase 0
gate for the family and costs milliseconds against a 68 GiB stream:

- Its fixture is the production `text_config` verbatim, trimmed to the keys
  the parser reads plus the keys it must ignore. A shape-only fixture
  cannot catch a key mapped to the wrong field, because every dimension
  would be some other made-up number either way.
- `every_published_checkpoint_parses_to_one_baseline` pins that BOTH
  published checkpoints (`pipenetwork/Qwen3.8-Flash-Next-MLX-4bit` at 512
  experts, `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` at 288) reduce to
  one baseline, with `num_experts` the ONLY field that moves.
- If a future point release moves a shape key, this test reddens in
  milliseconds rather than a multi-GB stream failing at some tensor offset.

Decision point: if the parse matches an existing baseline field for field,
you are adding a checkpoint to an existing family, not a new family. Stop
here and verify a real install runs. Only continue when a field genuinely
has no home.

### 2. Add the baseline and the registry entry

Two edits in `crates/model-io/src/arch_baselines/`:

1. Add a `ModelFamily` variant and write the baseline function. The
   exemplar is `qwen4_exp_125b_a6b()` in
   `crates/model-io/src/arch_baselines/qwen.rs`. Read every number off the
   checkpoint's own `config.json`, never from a sibling baseline: the
   doc comment on `qwen_gdn_dense_27b()` records that every behavioural
   field of that baseline is Qwen 3.6's while every shape field differs,
   and that all of it was read off the checkpoint rather than inferred.
2. Wire the variant into `known_architecture(family)` and
   `all_known_architectures()` in
   `crates/model-io/src/arch_baselines/mod.rs`. `known_architecture` is
   exhaustive; `arch_validation` compares an install's manifest against it
   field by field, so an invented baseline validates installs against
   fiction.

Ground rules the exemplar baseline records in its own doc comments:

- Name the baseline for the ARCHITECTURE, not a checkpoint
  (`qwen4_exp_125b_a6b` serves two published checkpoints).
- Floats must be binary fractions to survive serde_json's ~1-ULP parser
  (the exemplar's `attention_scale: 0.0625` carries a comment saying so).
- A value standing in for an absent config key must be the format's own
  default, recorded as such: `ple.seed: 1234` is the reference's default,
  not in the file, and the doc comment says so.
- Add honest refusals for anything you have not ported. In
  `crates/repack/src/gguf_names/mod.rs`, `gguf_architecture()` returns
  `None` for `Qwen4Exp` even though real GGUFs exist, because that table
  feeds `family_for_architecture`, whose contract is the architectures
  that RUN: `Some` would let a caller mistake recognition for support.

Extend the repack config parser the same way
(`parse_qwen4_exp_config`): missing extension keys are refused by name
rather than defaulted (`a_missing_extension_key_is_refused_rather_than_defaulted`
in `qwen4_config.rs`), consistency is checked
(`an_inconsistent_ple_or_indexer_block_is_refused`: an indivisible
`ple_embed_dim`, a ragged `indexer_budget`, a zero layer id), and foreign
configs are refused quoting the file's own `model_type`
(`the_two_qwen_configs_refuse_each_other`).

### 3. Map names and intake, and build a synthetic fixture

Names and intake live in `crates/repack`:

- GGUF tensor names map to canonical HF-style names in
  `crates/repack/src/gguf_names/mod.rs` (`map_gguf_name`,
  `gguf_architecture`, `family_for_architecture`). Every row is read off a
  real file, cross-checked against a real install's resident index. An
  unmapped name is an ERROR (`GgufNameError::Unmapped`), never a skip: an
  unrecognized routed marker silently makes every expert a resident tensor
  and the install runs at many times its intended footprint. Do not borrow
  a neighbour's table for the shared two thirds of names: the exemplar
  deliberately leaves `Qwen4Exp` unmapped because borrowing Qwen 3.6's
  table would map the trunk and silently drop the n-gram shards, the
  hyper-connections and the indexer, which is a partial install that opens.
- GGUF metadata maps to `ArchConfig` through the `gguf_config` module
  (`arch_from_gguf`), starting from `known_architecture(family)` and
  overriding only what the metadata determines.
- Safetensors intake extends the family walk; the exemplar's distinctive
  component (the n-gram PLE table) got its own writer arm in
  `gemma4_checkpoint/ngram.rs`.

Build a synthetic fixture through the REAL walk before any download. The
pattern is `crates/repack/src/synthetic_qwen/qwen4.rs`:

- `build_synthetic_qwen4_exp_install` (non-streamed writer) and
  `build_synthetic_qwen4_exp_install_streamed` exist side by side on
  purpose: every real checkpoint takes the STREAMED writer, so the fixture
  must exercise both or it proves nothing about the path the download
  takes. The decode-flow fixture came later as
  `build_synthetic_qwen4_exp_decode_install` (plus a
  `_with_indexer_budget` variant), exported from
  `crates/repack/src/synthetic_qwen/mod.rs`.
- Fixture constants are chosen to DISCRIMINATE, not to be tidy: 3 shards
  (not 1, where a shard-count bug and a row-count bug read the same),
  2 rows per shard (not a power of two), filler bytes keyed by shard index
  so a misplaced shard is visible.
- Synthetic weights are untrained, so the fixture proves writer WIRING
  (placement, both writers, manifest contents), never decode math. That is
  step 5's job.

### 4. Wire the decode flow under families/&lt;family&gt;/

The exemplar is `crates/runtime/src/families/qwen4/`, split as:

| File | Owns |
|---|---|
| `mod.rs` | The layer-loop contract: a module doc with the decoder layer as pseudocode, transcribed from the reference, plus shared constants (`TRUNK_PREFIX`, `RMS_EPS`) and `layer_tensor`. |
| `produce.rs` | The per-token forward (`produce_real_qwen4`): embed, per-layer encode calls, mid-layer commit where a host readback forces it, final head. Returns logits, never probabilities. |
| `state.rs` | `RealQwen4State`: everything allocated once at open (the wide residual, GDN recurrent state, PLE buffers and mmap'd n-gram table). |
| `attn.rs`, `hc.rs`, `moe.rs`, `ple.rs` | One encoder module per sublayer. |

What to copy from it:

- Start the module doc with the layer as pseudocode and cite where each
  formula came from. `qwen4/mod.rs` also records which formulas were
  cross-checked against the reference's actual source and which were not:
  re-verify a formula against source before extending code built from it.
- Dispatch keys on `ArchConfig.family`, never on tensor names. The runner
  holds `real_qwen4: Option<RealQwen4State>`
  (`crates/runtime/src/real_forward.rs`), built at open when the family
  matches; `produce_inner` in `crates/runtime/src/families/synthetic/mod.rs`
  dispatches on that state field.
- REUSE shared helpers where the math coincides. Three from the exemplar:
  the MoE router reuses the existing `router_topk_gemma4` unchanged
  (softmax-before-topk is algebraically identical to top-k over the raw
  logits when there is no router bias); QSA attention is
  `families/qwen/attn.rs::encode_full_attention_block`'s shape with an
  indexer in front; the head goes through the shared `encode_gemv_any`.
- FORK rather than parameterize when sharing would thread an `Option`
  through a verified flow that four other families depend on. The exemplar
  is deliberately "a seventh flow, not a variant of `families/qwen/`",
  and its wide residual is a family-local buffer (`wide_x`) rather than
  `DecodeScratch::x`, precisely so no other family's frozen memory-oracle
  row moves for a buffer it never touches.
- Commit command buffers only where a host readback forces it (the router
  top-k), and never drop a `PassEncoder` without committing it: the
  exemplar's `produce.rs` carries a comment that a dropped encoder
  silently discards every dispatch encoded onto it.

### 5. Climb the test ladder

Each rung costs more than the one before it; do not skip ahead.

1. Config test (milliseconds, offline):
   `cargo test -p turbospark-repack --test qwen4_config`
   Whole-struct equality against the baseline, refusal cases, both
   published checkpoints.
2. Repack walk test (fixture, milliseconds): builds the synthetic install
   through the real walk and both writers, asserting placement and
   manifest contents. For GGUF name tables, the network test that reads
   only real-file headers (`tests/gguf_checkpoint_network.rs`, `#[ignore]`d)
   is the only place a name-mapping hole can surface, because a synthetic
   fixture only ever contains names its author already knew.
3. Decode-flow test on the synthetic install:
   `cargo test -p turbospark-runtime --test real_forward_qwen4`
   The exemplar's shape is three layers of assertion:
   - Perturbation cases, one per sublayer: patch a tensor's bytes, require
     the logits to move. Each case decodes at least 8 positions, because a
     single position hides every q/k transform.
   - A frozen digest over the logits. Perturbation cases alone rebuild
     their own baseline inside the same binary, so they stay green under a
     mutation that moves the math for every arm equally. The digest is the
     one assertion comparing against something computed before the
     mutation.
   - A stated bound on what that proves: untrained weights make the digest
     a change detector, never a correctness claim.
4. Real-install gates (needs the streamed install, `--release`,
   `--ignored`):

   ```sh
   TURBOSPARK_QWEN4EXP_INSTALL_DIR=~/.turbospark/models/qwen4-reap288.gturbo \
     cargo test -p turbospark-bench --test qwen4exp_memory_oracle --release -- --ignored --nocapture

   TURBOSPARK_QWEN4EXP_INSTALL_DIR=~/.turbospark/models/qwen4-reap288.gturbo \
     cargo test -p turbospark-bench --test qwen4exp_quality_gate --release -- --ignored --nocapture
   ```

   One real model per process, which is why these are separate targets.
   The exemplar's quality gate pins its own window (2,048, this family's
   indexer budget) rather than the shared 4,096, and its header records
   the install variable to set.

## Verify

Before handing off, against the real install:

1. Greedy generation stays coherent (catches broken math):
   `--temperature 0.0001 --top-k 1`, 400 new tokens.
2. SAMPLED generation stays coherent at the CLI defaults (T=0.2, top-k 64,
   top-p 0.95): greedy is `argmax` and is invariant under monotone
   transforms, so it stays byte-identical to correct through bugs that
   destroy sampling entirely.
3. The memory oracle ceiling holds at the pinned slot count and window.

All three, plus the four workspace commands, are the verify-a-change
guide's contract; a change touching the decode path is not done without
them.

## Pitfalls

- **The family is chosen by `ArchConfig.family`, never by tensor names.**
  Two families can share `language_model.model.embed_tokens.weight`; a
  naming probe cannot tell them apart, and picking a neighbour's flow
  yields fluent WRONG output rather than an error.
- **A shared architecture's second half is not covered by the first
  half's tests.** Before adding a checkpoint of a shared architecture's
  other half (dense against MoE), GREP for the family name across the
  crates you touch and prefer `matches!(family, A | B)` over `== A`
  wherever the two share the code below it. A condition naming only one
  half reads as correct and is a latent bug for exactly as long as no
  checkpoint of the other half exists: the dense half of the Qwen GDN
  architecture shipped with three such conditions (a missing
  `linear_attention` assignment, a `None` from `v_head_axis`, and a
  refusing mask arm) that the MoE half's tests could never see because
  its own baseline happened to hold the right answers.
- **One-character convention differences decode fluently and wrongly.**
  The exemplar's GDN output gate is `sigmoid` where every earlier family
  declares silu, read off the config key `output_gate_type`; its PLE
  `layer_ids` are ONE-BASED in the file and zero-based in a layer loop.
  Both are pinned as discriminating-pair tests in `qwen4_config.rs`.
- **Never downgrade an unmapped name or an unported kernel to a skip.**
  Skips produce installs that open: a partial install (dropped n-gram
  shards), an all-resident expert table, a model wrong only at long
  context. Refuse by name.
- **Perturbation tests are self-relative.** A file of them catches almost
  nothing; close it with a frozen digest, and re-freeze only with a stated
  reason. When a digest stops reproducing, bisect to AND INCLUDING the
  commit that wrote it before assuming a regression.
