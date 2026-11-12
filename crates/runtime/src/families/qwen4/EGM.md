# Qwen4 Family Flow (qwen4_exp, Qwen3.8-Flash-Next)

## Overview
The real-checkpoint decode and chunked-prefill flow for the `qwen4_exp` family, driven by `RealForwardRunner`. It is a separate flow from `families/qwen/`: the residual stream is `hc_count * hidden` wide (10240), every layer uses a hyper-connection read/inject pair instead of a plain residual add, layer 1 adds a PLE sublayer fed by a host-mmapped n-gram table, and attention layers are either gated DeltaNet (mask 2) or QSA (mask 1, indexer plus dense/indexed attention). `state.rs` owns all buffers and opt-in capture hooks, `produce.rs` is the sequential decode path, `prefill.rs` the micro-batched prefill path with optional batched seams.

## Interface
- `RealQwen4State` [state.rs:RealQwen4State]: all per-install buffers, GDN/QSA state, PLE table, capture hooks; built by `build`, cleared by `reset`
- `RealQwen4State::qsa_can_select` [state.rs:qsa_can_select]: true when QSA selection can run at the opened context
- `TRUNK_PREFIX`, `RMS_EPS`, `layer_tensor` [mod.rs:layer_tensor]: tensor naming and norm epsilon
- `rms_norm_grouped`, `rms_norm_perhead` [mod.rs:rms_norm_grouped]: norm encoders that pick the centered (HF/MLX) or direct (GGUF) weight convention
- `encode_hyper_connection` [hc.rs:encode_hyper_connection]: attn_hc/mlp_hc/final mixer (hc_norm, low-rank mix, mean over streams, inject gate)
- `encode_linear_block`, `encode_full_attention_block` [attn.rs:encode_linear_block]: GDN and QSA sublayers
- `encode_ple_layer` [ple.rs:encode_ple_layer]: PLE sublayer with 16 host row lookups
- `encode_moe_router`, `encode_moe_layer` [moe.rs:encode_moe_router]: router GEMV (cb1) and the host-readback MoE step; shared expert added after FP16 routed sum
- `produce_real_qwen4` [produce.rs:produce_real_qwen4]: sequential single-token decode
- `refuse_qwen4_vision_past_qsa_budget` [produce.rs:refuse_qwen4_vision_past_qsa_budget]: image-prompt refusal where QSA could select
- `prefill_chunk_real_qwen4` [prefill.rs:prefill_chunk_real_qwen4]: prefill in `MAX_PREFILL_BATCH` micro-batches
- `encode_qwen4_layer_routed_prefill`, `RoutedBatchScratch` [moe_batch.rs:encode_qwen4_layer_routed_prefill]: opt-in batched routed MoE (`TURBOSPARK_ROUTED_BATCH`)
- `BatchedGemvScratch`, `encode_layer_cb1_batched` [batched.rs:encode_layer_cb1_batched]: opt-in M-row projection GEMMs (`TURBOSPARK_BATCHED_GEMV`)
- `PrefillExperiment`, `configure_moe_prefill_bank`, `supports_moe_prefill_bank` [prefill_bank.rs:PrefillExperiment]: experimental wide/bank prefill with memory admission
- `cb1_trace` [cb1_trace.rs:PassTrace]: opt-in per-layer cb1 timing TSV (decode only)
- Subdirectories: None

## Invariants
- Hyper-connection injection uses the un-normalized `raw` stream, never the normed one; the mix reads the normed one [hc.rs:encode_hyper_connection]
- Only two `/ hc_count` divisions: before silu on the down-projection and on the inject weight input; none on the up-projection [hc.rs:encode_hyper_connection]
- Inject gate row is per token (`inject_row_offset`) inside a prefill micro-batch so a later token cannot overwrite an earlier gate before it is read [hc.rs:encode_hyper_connection]
- Decode requires `position == kv.position()`; non-sequential positions are refused [produce.rs:produce_real_qwen4_inner]
- `build` refuses unsupported configs (bad indexer or hyper-connection params, softcap, tied lm_head, layer masks other than 1/2, no linear layers, inactive PLE, a PLE layer that is not linear/GDN) rather than running them [state.rs:RealQwen4State::build] [state.rs:resolve_ple_layer]
- The wide residual is family-local (`wide_x`), not the shared `DecodeScratch::x` [state.rs:RealQwen4State]
- The shared expert is added after the routed sum is rounded to FP16; phase 2 seeds from zero rows [moe.rs:encode_moe_layer, moe_batch.rs:encode_qwen4_layer_routed_prefill]
- MoE host readback requires commit and wait of the router pass before `encode_moe_layer` [moe.rs:encode_moe_router]
- Image prompts are refused when QSA selection can run (mRoPE vs scalar-index rotation of pooled keys differs) [produce.rs:refuse_qwen4_vision_past_qsa_budget]
- Experimental prefill admits one at a time, requires streamed text prefill without the routed-batch/batched-GEMV seams, and checks headroom plus memory pressure before allocating [prefill_bank.rs:PrefillExperiment::allocate]
- Speculative decoding is unsupported: no rollback for recurrent state [mod.rs]

## Side Effects
- Env reads: `TURBOSPARK_QWEN4_PREFILL` (off/wide/bank, invalid value is an error) [prefill_bank.rs:PrefillExperiment::from_env], `TURBOSPARK_QSA_FORCE_DENSE=1` (also `RealForwardRunner::set_qsa_force_dense`; keeps the dense kernel to measure KL against the indexed path) [state.rs:RealQwen4State::build]
- Env-gated diagnostics: `TURBOSPARK_QWEN4_GDN_NORM_CAPTURE`, `TURBOSPARK_QWEN4_TAP_FILE`, `TURBOSPARK_QWEN4_LAYER_CAPTURE`, `TURBOSPARK_QWEN4_ATTENTION_CAPTURE` and their `_POSITION`/`_LAYERS`/`_LAYER` companions [state.rs:GdnNormCapture]
- `TURBOSPARK_QWEN4_TAP_FILE` enables QTAP capture and `TURBOSPARK_QWEN4_TAP_LAYERS` selects tap layers (default `5,15,25,35,45`) [state.rs:TapCapture::from_env]; `TURBOSPARK_ROUTED_BATCH_OVERLAP` [moe_batch.rs:encode_qwen4_layer_routed_prefill]; `TURBOSPARK_INT4_MMA8` switches the fused in-proj arithmetic [attn.rs:encode_linear_block]
- Capture writers create files (header plus f16/binary data) and log to stderr [state.rs:GdnNormCapture::flush_if_captured, state.rs:TapCapture::append]
- `TURBOSPARK_CB1_TRACE=<path>` opens a process-global TSV sink behind a `OnceLock<Mutex<File>>` [cb1_trace.rs:sink]
- Metal: allocates shared buffers and encodes command buffers; prefill bank uses an anonymous mmap wrapped as a no-copy Metal buffer [prefill_bank.rs:PrefillExperiment::allocate]
- Global `PREFILL_ADMISSION` mutex serializes experimental prefills [prefill_bank.rs:PREFILL_ADMISSION]
- Host mmap reads of the n-gram table and pread expert streaming [ple.rs:encode_ple_layer, prefill_bank.rs:PrefillBank::load_layer]

## Verification
- Run: `cargo test -p turbospark-runtime`
