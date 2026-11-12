# DFlash2 Draft Pass

## Overview
The DFlash2 block-diffusion drafter for the Qwen family (`docs/DFLASH2.md`). A round writes target-derived context KV from the trunk's capture, runs one non-causal block forward of `block + 1` rows through the drafter layers and two head GEMMs, then selects proposals on the host from codebook scores. Methods are `impl RealForwardRunner`, operating on `real_dflash` state defined in the sibling `dflash.rs`.

## Interface
- `RealForwardRunner::dflash_draft_block` [mod.rs:dflash_draft_block]: context write, block forward, greedy selector; fills `proposals`
- `RealForwardRunner::dflash_draft_tree` [mod.rs:dflash_draft_tree]: same GPU half, best-first tree selector
- `RealForwardRunner::dflash_prime_from_capture` [mod.rs:dflash_prime_from_capture]: writes one captured row into the drafter cache
- `RealForwardRunner::dflash_rewind_to` [mod.rs:dflash_rewind_to]: cursor move back, bounded by ring slack
- `RealForwardRunner::dflash_probe_logits` [mod.rs:dflash_probe_logits]: reads one row of drafter logits
- `RealForwardRunner::dflash_probe_buffers` [mod.rs:dflash_probe_buffers]: NaN/min/max scan of persistent draft buffers
- `RealForwardRunner::dflash_handoff_state` [mod.rs:dflash_handoff_state]: `(cursor, capture_base, capture_rows)`
- `DFLASH_THETA`, `RESIDUAL_RESCALE` [mod.rs:DFLASH_THETA]: RoPE theta 1e7 and embedding rescale (reciprocal of `DFLASH_RESIDUAL_SCALE`)
- `dflash_context_write` [context_kv.rs:dflash_context_write]: fc, hidden norm, per-layer k/v projection into ring slots, k_norm, RoPE
- `dflash_forward_block` [forward.rs:dflash_forward_block]: embeddings, layers, final norm, trunk head GEMM and hidden projection GEMM
- `encode_dflash_layer` [layer.rs:encode_dflash_layer]: attention conv, non-causal attention, MLP conv for one layer
- `dflash_select` / `dflash_select_tree` / `dflash_lattice` [select.rs:dflash_select]: host selectors over top-16 candidates and codebook bilinear scores
- `read_f16_rows` [select.rs:read_f16_rows]: f16 buffer readback to f32

## Invariants
- Attention inside the block is non-causal: every row uses `seq_len = base + rows` [layer.rs:encode_dflash_layer]
- Per-row k norms and RoPE run before any attention encode, so no row reads a raw key [layer.rs:encode_dflash_layer]
- Projections into the ring are split at the wrap via `ring_spans`, and RoPE uses per-position slots [context_kv.rs:dflash_context_write]
- Context write never moves the cursor backward and rejects capture ranges beyond capacity [context_kv.rs:dflash_context_write]
- A drafter cursor past the round base or a base behind the capture base is refused [mod.rs:dflash_run_block]
- Rewind to a position at or ahead of the cursor is a no-op; rewinds beyond `max_safe_rewind` are refused [mod.rs:dflash_rewind_to]
- A non-finite logit row is refused rather than walked (NaN would silently yield token 0) [select.rs:top_k_finite]
- Ties in top-K keep the lower token id; the tree path's forced-greedy chain matches `dflash_select` [select.rs:top_k_finite]
- Codebook reads bound-check token id, rank and offsets with checked arithmetic [select.rs:read_bf16_row]
- Anchor and mask token ids must be inside vocab [mod.rs:dflash_run_block]

## Side Effects
- Metal command buffers, each committed and waited (`dflash ctx kv`, `dflash draft`) [context_kv.rs:dflash_context_write]
- Writes drafter KV ring slots, advances the cursor, zeroes `capture_rows` as the last step of a write so a later draft does not rewrite consumed rows [context_kv.rs:dflash_context_write]
- Autorelease pool around each round [mod.rs:dflash_run_block]
- Host readback of 248k-wide f16 logit rows per step [select.rs:dflash_select]
- `real_dflash` exists only if the policy builds it: unset `TURBOSPARK_DFLASH_DRAFT` is Auto (built on dense installs carrying `dflash.fc.weight`), `N` forces block size N, `0` turns it off; Auto block size is `dflash_serving_block` (`DFLASH_SERVING_BLOCK`, or `DFLASH_SERVING_BLOCK_TILE` when `TURBOSPARK_INT4_MMA8` is on) [dflash_state.rs:DflashDraftPolicy::from_env] [dflash_state.rs:dflash_serving_block]

## Verification
- Run: `cargo test -p turbospark-runtime`
