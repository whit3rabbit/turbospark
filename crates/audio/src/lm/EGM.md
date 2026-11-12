# Shared LM Backbones (audio::lm)

## Overview
Portable single-batch f32 ports of the mlx-audio `lm` subset that TTS families drive: KV caches (including a rotating sliding-window ring), rope constructions, a shared dense transformer engine behind llama/qwen2/qwen3, plus gpt2, cohere2, bailing_moe (with SwitchGLU), and the falcon_h1 Mamba2 hybrid (with its selective scan). `Weights` is the checkpoint access layer (prefix, quantization scheme, name remap). An optional `ComputeBackend` can offload linears and non-causal attention through `Execution`.

## Interface
- `Weights`, `CheckpointScheme`, `Bias`: checkpoint access, scheme enum, bias policy [mod.rs:Weights] [mod.rs:CheckpointScheme] [mod.rs:Bias]
- `Weights::open`, `open_single`, `open_single_with_scheme`, `with_prefix`, `with_name_map`, `linear`, `linear_absolute`, `embedding`, `tensor`, `tensor_optional`, `tensor_absolute`, `raw_tensor`, `contains`, `contains_absolute`, `shards`, `config` [mod.rs:Weights]
- `Backbone` (alias `LlamaModel`/`Qwen2Model`/`Qwen3Model`), `DenseConfig`, `Input` [engine.rs:Backbone] [engine.rs:DenseConfig] [engine.rs:Input] [llama.rs:LlamaModel] [qwen2.rs:Qwen2Model] [qwen3.rs:Qwen3Model]
- `Backbone::load`, `load_without_embed_table`, `load_without_final_norm`, `load_with_embed_table`, `forward`, `forward_hidden`, `forward_hidden_and_logits`, `set_bidirectional`, `set_backend`, `with_backend`, `embed_table` [engine.rs:Backbone]
- per-flavor `from_json` / `load` [llama.rs:from_json] [qwen2.rs:from_json] [qwen3.rs:from_json] [gpt2.rs:from_json] [cohere2.rs:from_json] [bailing_moe.rs:from_json] [falcon_h1.rs:from_json]
- `Gpt2Model`, `Gpt2Config` [gpt2.rs:Gpt2Model]; `Cohere2Model`, `Cohere2Config` [cohere2.rs:Cohere2Model]; `BailingMoeModel`, `BailingMoeConfig` [bailing_moe.rs:BailingMoeModel]
- `FalconH1Model`, `FalconConfig`, `FalconLayerCache`, `compute_mup_vector`, `FalconConfig::ensure_unity_multipliers`, `forward_stack` [falcon_h1.rs:FalconH1Model] [falcon_h1.rs:compute_mup_vector]
- `KvCache`, `RotatingKvCache`, `PromptCache`, `LayerKv`, `LayerCaches`, `make_prompt_cache` [cache.rs:KvCache] [cache.rs:RotatingKvCache] [cache.rs:PromptCache] [cache.rs:LayerKv] [cache.rs:make_prompt_cache]
- `Rope` (`initialize`, `apply`) [rope.rs:Rope]; `SwitchGlu` [switch_layers.rs:SwitchGlu]
- `ssm` (`ssm_update`, `compute_dt`) is `pub(crate)` only [ssm.rs:ssm_update]; `Execution` is `pub(crate)` [execution.rs:Execution]

## Invariants
- Activations are row-major `[positions, width]`; cache buffers are head-major `[heads, total, head_dim]`; appends must insert per head, not flat-extend [cache.rs:KvCache] [engine.rs:head_major]
- `Backbone` load refuses `sliding_window`, and zero or non-dividing `num_key_value_heads` [engine.rs:load_impl]
- Forward refuses empty input, a cache count differing from the layer count, negative or out-of-vocab token ids, and non-multiple embedding lengths [engine.rs:forward_hidden] [engine.rs:lookup_embeddings] [engine.rs:Input]
- Rope offset is the pre-update cache offset; a rotating cache's returned buffer is temporal order after multi-position updates and physically rotated after single-position updates [engine.rs:forward_hidden] [cache.rs:RotatingKvCache]
- Windowed attention mask: single-position queries see the whole ring, multi-position use `min(w-1, offset) + i`; `bidirectional` skips the causal bound [engine.rs:Attention]
- A quantized tensor without a matching `CheckpointScheme` is refused, and affine versus rowwise scheme mismatches error; weight element counts are checked [mod.rs:load_weight_with_prefix]
- Absolute-name loads bypass the prefix and `name_map` [mod.rs:linear_absolute] [mod.rs:tensor_absolute]
- Rope `initialize` rejects non-positive linear factor, odd proportional rotated count, and mrope sections not equal to 3; unknown type is `Unsupported` [rope.rs:initialize]
- MoE slot order is descending selection score with ties to the lower expert index (reduction order); `expert_bias` affects selection only, `routed_scaling_factor` is applied unconditionally [bailing_moe.rs:MoeGate]
- `SwitchGlu::forward` rejects negative or out-of-range expert indices and returns `[len, slots, input_dims]` [switch_layers.rs:SwitchGlu]
- bailing `norm_head` with tied embeddings is refused [bailing_moe.rs:from_config]
- cohere2 requires `head_dim * heads == hidden_size`, positive window and pattern; layer `i` is global when `i % pattern == pattern - 1` [cohere2.rs:from_config] [cohere2.rs:make_cache]
- falcon_h1 requires 5 `ssm_multipliers`, `d_ssm == n_heads * d_head`, and a conv1d weight of 3 dims; muP folds apply only when layer-0 conv `shape[2] > shape[1]` [falcon_h1.rs:load_impl] [falcon_h1.rs:needs_sanitization]
- `ssm_update` asserts shape consistency and runs the single linear recurrence, chunk invariant [ssm.rs:ssm_update]
- Token ids are signed 32-bit `i32` [engine.rs:Input]

## Side Effects
- Disk reads and shard mapping through `SafetensorsFile::open`, `nn::open_shards`, and `config.json` reads [mod.rs:Weights]
- Optional device offload through `ComputeBackend::load_weight`, `linear`, `attention`; falls back to host on any backend error [execution.rs:Execution]
- `Execution` caches uploaded weights in a `RefCell<HashMap>` keyed by weight pointer address, so a `Linear` weight buffer must not move or be reused [execution.rs:weight]
- Device attention is used only when bidirectional, cacheless, and on device; panics via `expect` if called without a backend [engine.rs:Attention] [execution.rs:attention]
- No env vars, threads, or global statics in this module

## Verification
- Run: `cargo test -p turbospark-audio lm`
