# Qwen Real-Checkpoint Decode Flow

## Overview
Shared real-checkpoint decode, prefill and speculative-verify flow for the Qwen hybrid families (qwen36 MoE and dense qwen3_5): gated DeltaNet (GDN) linear layers (mask 2) interleaved with gated full attention (mask 1). It encodes Metal passes for sequential decode, chunked prefill, M-row batched verify (MTP head and DFlash2 drafter), and the routed-MoE path. Opt-in accelerators (ANE+Steel prefill split, router-lookahead prefetch, Hadamard-folded weights) hang off the same runner. State is built once at open and validated by name.

## Interface
- `RealQwenState` [state.rs:RealQwenState]: GDN state, per-token scratch, router buffers, optional pilot/prefetch/batched-prefill/hadamard; `build` validates the arch and probes tensors
- `RealForwardRunner::produce_real_qwen` [produce.rs:produce_real_qwen]: sequential single-token pass (logits, hidden-out, or no-head prefill)
- `encode_linear_block` / `encode_full_attention_block` [attn.rs:encode_linear_block]: per-layer GDN and gated-attention decode encoders; `QkNormConvention` selects Plain vs Centered q/k norm
- `encode_qwen_layer_dense` [dense.rs:encode_qwen_layer_dense]: dense FFN half with fused INT4 2-dispatch path and per-tensor fallback
- `encode_qwen_layer_moe` [moe.rs:encode_qwen_layer_moe]: one token's routed half, shared expert and residual add; `qwen_router_offsets` locates the INT8 router
- `encode_qwen_layer_moe_batched` / `encode_qwen_layer_routed_prefill` [moe_batch.rs:encode_qwen_layer_moe_batched]: M-row verify and chunked-prefill routed halves
- `prefill_chunk_real_qwen_dense`, `prefill_chunk_real_qwen_dense_priming_mtp`, and `prefill_chunk_real_qwen_dense_priming_dflash` [prefill.rs:prefill_chunk_real_qwen_dense]: dense chunked prefill with optional drafter priming (`Priming`)
- `encode_qwen_dense_layer_batched_prefill` [prefill_layers.rs:encode_qwen_dense_layer_batched_prefill]: batched-GEMM and per-token dense prefill layer encoders plus chunk head
- `prefill_chunk_real_qwen_moe` / `qwen_moe_chunked_refusal` [moe_prefill.rs:prefill_chunk_real_qwen_moe]: MoE chunked prefill driver and its install-level refusal; `encode_qwen_moe_cb1_batched` [moe_prefill_batched.rs:encode_qwen_moe_cb1_batched] is the batched arm's cb1
- `RealForwardRunner::produce_batched` [batched.rs:produce_batched]: M-row trunk verify for MTP/DFlash2; layer encoders in [batched_layers.rs:encode_full_attention_block_batched] and [verify_layers.rs:encode_qwen_moe_layer_batched_step]
- `BatchedScratch`, `BatchedVerifyTape`, `Rank1Tape`, `TreeSpec` [batched_scratch.rs:BatchedScratch]: M-row buffers, GDN verify tape, tree-verify parent spec
- `MtpState`, `mtp_draft_step`, `mtp_prime_step` [mtp_state.rs:MtpState]: MTP head state, draft policy (`TURBOSPARK_MTP_DRAFT`) and blocker; step encoders in [mtp.rs:mtp_draft_step]; debug dump in [mtp_dump.rs:dump_mtp_stage]
- `DflashState`, `DflashDraftPolicy`, `DflashShape`, `dflash_speculation_blocker` [dflash.rs:DflashState] / [dflash_state.rs:DflashShape]: DFlash2 drafter state, shape derivation and blocker policy
- `dflash_verify_tree`, `dflash_tree_compact` [dflash_tree.rs:dflash_verify_tree]: tree verify round and in-place accepted-path compaction
- `HadamardPlan` [hadamard.rs:HadamardPlan]: Hadamard-folded weight contract (`build`, `is_folded`, `is_inverse`)
- `Prefetcher` [prefetch.rs:Prefetcher]: experimental router-lookahead expert prefetch worker
- `AnePrefillMode`, `AnePrefill`, `AneSplitState`, `parse_manifest` [ane_split.rs:AnePrefillMode]: opt-in ANE+Steel prefill split (feature `ane-prefill`); stubs in [mod.rs:AnePrefillMode] when off
- `DraftPolicies`, `fused_ffn` [mod.rs:DraftPolicies]: bundled draft policies, shared constants, tensor-name helpers
- Subdirectories: [dflash_draft](./dflash_draft/EGM.md)

## Invariants
- Residual adds are RAW (no sandwich norms) and the FFN output rounds to half before meeting the stream [dense.rs:encode_qwen_layer_dense] [moe.rs:encode_qwen_layer_moe]
- Routed expert slots dispatch in router ranking order in mapped and slot-cache arms; the gated shared output seeds phase 2, it is not added afterwards [moe.rs:encode_qwen_layer_moe]
- Trunk q/k norms are Plain, the MTP head's are Centered, with the same tensor names [attn.rs:QkNormConvention] [mtp.rs:mtp_step]
- MTP step position must equal the head's KV cursor, and the step runs after trunk logits are read because it clobbers shared scratch [mtp.rs:mtp_step]
- Batched verify: start_position == kv.position(), token ids < vocab, vision refused, and a block that would wrap a full-attention KV ring is refused [batched.rs:produce_batched]
- GDN tape replay and commit use the recording pass's batch as slot stride, not scratch capacity [batched_layers.rs:replay_linear_state_batched]
- Tree verify requires parent[0] == -1 and parents preceding children; compaction path must be strictly increasing starting at slot 0 and runs before rollback and the next draft [batched_scratch.rs:TreeSpec::new] [dflash_tree.rs:dflash_tree_compact]
- Drafters follow an Off/Auto/Fixed policy: unset `TURBOSPARK_MTP_DRAFT` and `TURBOSPARK_DFLASH_DRAFT` mean Auto, which builds the MTP head (depth `AUTO_DEPTH`) or DFlash drafter only when the install carries one; `0` allocates nothing; MoE installs are refused for DFlash2 by name [mtp_state.rs:MtpDraftPolicy::from_env] [mtp_state.rs:MtpState::build] [dflash.rs:DflashState::build] [dflash.rs:dflash_moe_unsupported]
- Hadamard transforms activations, never weights; folded names outside the routed suffix list are refused at open, not at token 1 [hadamard.rs:HadamardPlan] [hadamard.rs:routing_error]
- Router weight must be INT8 (dtype 5) with num_experts*hidden bytes [moe.rs:qwen_router_offsets] [verify_layers.rs:encode_router_gemv_batched]
- Prefill arms: forced batched arm refuses by name, auto falls back silently, TURBOSPARK_BATCHED_GEMV=0 keeps the per-token reference [prefill.rs:prefill_chunk_real_qwen_dense]
- Prefetch only chooses which bytes load early, never which experts run; an unparseable env value panics [prefetch.rs:run_job] [prefetch.rs:width_from_env]
- Any ANE manifest mismatch is refused and a Steel redo follows any failure [ane_split.rs:parse_manifest]

## Side Effects
- Encodes Metal dispatches, and commits and waits command buffers (decode cb1/routed cb/final cb, batched verify, prefill micro-batches) [produce.rs:produce_real_qwen_inner] [batched.rs:produce_batched]
- Mutates KV cursor, GDN conv-tail and recurrent state, drafter KV and capture buffers, phase counters [attn.rs:encode_linear_block] [prefill.rs:finish_qwen_dense_micro_batch]
- Allocates Metal buffers at open or lazily (scratch, batched prefill, tape, signs, MTP/DFlash state); ANE split adds IOSurface planes and a wider scratch.x [state.rs:build] [batched_scratch.rs:BatchedScratch::new] [ane_split.rs:ane_prefill_steel_rows]
- Env reads: TURBOSPARK_QWEN_FUSED_FFN, _QWEN_BATCHED_ROW_OPS, _DFLASH_TREE, _DFLASH_DRAFT, _MTP_DRAFT, _MTP_DUMP, _QWEN_PREFETCH, _QWEN_MOE_CHUNKED, _ANE_PREFILL(_DIR/_DFLASH/_STATS/_FAULT), _GDN_LAZY_TAPE, mostly once per process via OnceLock [mod.rs:fused_ffn] [batched_layers.rs:row_batched_ops] [ane_split.rs:AnePrefill]; when `TURBOSPARK_ANE_PREFILL_DIR` is unset or has no `manifest.json`, the ANE programs directory is searched under `$HOME` (`.turbospark/models/ane-prefill/qwen38`, `.turbospark/ane-prefill/qwen38`, `models/ane-prefill/qwen38`) [ane_split.rs:ensure]
- Further env switches: `TURBOSPARK_EXPERT_RESIDENCY=mapped` and `TURBOSPARK_PILOT_PROBE` are refused with batched routed verify [moe_batch.rs:encode_qwen_layer_moe_batched]; `TURBOSPARK_QWEN_SHARED_READ_OVERLAP` is refused by the MoE chunked driver [moe_prefill.rs:prefill_chunk_real_qwen_moe]; `TURBOSPARK_MAPPED_DEMAND_PREP` (off/advice/touch, invalid is an error) [state.rs:RealQwenState::build]; `TURBOSPARK_ROUTED_BATCH` and `_OVERLAP` [moe_batch.rs:encode_qwen_layer_moe_batched]; `TURBOSPARK_DFLASH_CHUNKED_PREFILL` (on unless 0) [dflash.rs]; `TURBOSPARK_INT4_MMA8` switches the GEMV arithmetic path [dense.rs]; `TURBOSPARK_PREFILL_STEEL=0`, `TURBOSPARK_MTP_BATCHED_PRIME=0` and `TURBOSPARK_PREFILL_ROWS_ATTN` pick prefill arms [prefill.rs]; `TURBOSPARK_STANDIN` skips FFN or attention blocks in the decode loop [produce.rs]; `TURBOSPARK_ROUTER_LOGITS_LAYER` captures router logits [moe.rs]; `TURBOSPARK_ATTN_DECODE_KERNEL` picks the decode attention kernel [batched_layers.rs]
- Blocking host readback of router logits and logits; expert pread and slot-cache mutation [moe.rs:encode_qwen_layer_moe] [moe_batch.rs:encode_qwen_layer_moe_batched]
- Spawns the 'qwen-prefetch' thread and prints stats to stderr on drop [prefetch.rs:spawn]
- Reads Core ML packages and manifest.json from the ANE programs dir; writes .f16 and meta.json dumps when TURBOSPARK_MTP_DUMP is set [ane_split.rs:AneSplitState] [mtp_dump.rs:dump_mtp_stage]

## Verification
- Run: `cargo test -p turbospark-runtime`
