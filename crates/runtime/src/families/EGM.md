# Model Families

## Overview
This module is the namespace for the per-family execution flows of `RealForwardRunner`, covering real checkpoints and the synthetic fallback. It holds no logic of its own: `mod.rs` only declares crate-private child modules, one per family. Decode flow is selected by `ArchConfig.family`, never by tensor names, so each child owns its state, encoders and prefill path.

## Interface
- `deepseek2` [mod.rs:deepseek2]: MLA attention over a compressed KV cache with dense lead layer and fine-grained MoE with fused shared expert. See [deepseek2](./deepseek2/EGM.md)
- `flint` [mod.rs:flint]: 49-layer GQA with sinks, dense lead layers, then sigmoid-routed MoE with ungated shared expert. See [flint](./flint/EGM.md)
- `gemma4` [mod.rs:gemma4]: Gemma 4 dense and routed-MoE decode, chunked prefill and image-span prefill. See [gemma4](./gemma4/EGM.md)
- `gptoss` [mod.rs:gptoss]: gpt-oss MoE flow (MXFP4 experts, biased attention, YaRN RoPE, sinks, alternating sliding window). See [gptoss](./gptoss/EGM.md)
- `lfm3` [mod.rs:lfm3]: LiquidAI LFM3 hybrid of gated short-convolution, full GQA and sliding-window GQA layers. See [lfm3](./lfm3/EGM.md)
- `llama` [mod.rs:llama]: plain GQA decode and chunked prefill serving Llama, Mixtral, Qwen2/3 trunks and MiniMax-M2. See [llama](./llama/EGM.md)
- `museglimmer` [mod.rs:museglimmer]: dense 52-layer sandwich-norm GQA stack with sliding/full pattern and softcapped untied head. See [museglimmer](./museglimmer/EGM.md)
- `qwen` [mod.rs:qwen]: Qwen hybrid GDN plus gated-attention flow for decode, prefill and speculative verify. See [qwen](./qwen/EGM.md)
- `qwen4` [mod.rs:qwen4]: `qwen4_exp` flow with hyper-connection residuals, PLE sublayer and GDN/QSA attention. See [qwen4](./qwen4/EGM.md)
- `spark` [mod.rs:spark]: Spark-X2.5-4B dense GQA flow in one command buffer per token or micro-batch. See [spark](./spark/EGM.md)
- `synthetic` [mod.rs:synthetic]: fallback INT4 GPU decode for short-named checkpoints that match no real family. See [synthetic](./synthetic/EGM.md)
- `unlimited_ocr2` [mod.rs:unlimited_ocr2]: DeepSeek-V2-shaped MoE decoder with DASH block-sparse attention and an image tower. See [unlimited_ocr2](./unlimited_ocr2/EGM.md)
- Subdirectories: [deepseek2](./deepseek2/EGM.md), [flint](./flint/EGM.md), [gemma4](./gemma4/EGM.md), [gptoss](./gptoss/EGM.md), [lfm3](./lfm3/EGM.md), [llama](./llama/EGM.md), [museglimmer](./museglimmer/EGM.md), [qwen](./qwen/EGM.md), [qwen4](./qwen4/EGM.md), [spark](./spark/EGM.md), [synthetic](./synthetic/EGM.md), [unlimited_ocr2](./unlimited_ocr2/EGM.md)

## Invariants
- Every family module is `pub(crate)`, so family flows are reachable only through the crate's runner and not exported [mod.rs:deepseek2]
- Adding a family requires a new `pub(crate) mod` line here plus cross-layer wiring (family enum, persisted string, FFI, catalog), because this file only registers modules [mod.rs:synthetic]
- `synthetic` is the fallback and is not a real family, real families are chosen by `ArchConfig.family` [mod.rs:synthetic]

## Side Effects
- None (declarations only, Metal, filesystem and env effects live in the child modules)

## Verification
- Run: `cargo test -p turbospark-runtime`
