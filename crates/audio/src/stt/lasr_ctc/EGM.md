# LASR CTC (Google MedASR)

## Overview
Self-contained CPU implementation of the LASR CTC conformer ASR family. It computes a float64 log-mel frontend, a dense/conv subsampler, conformer-style encoder blocks with RoPE attention and batch-norm conv modules, a CTC head, and its own greedy CTC decode. It reads MLX-layout conv weights directly and deliberately diverges from the upstream mlx-audio path, which double-transposes them.

## Interface
- `LasrCtc` with `load`, `profile`, `config`, `logits`, `greedy_token_ids`, `transcribe` [mod.rs:LasrCtc]
- `LasrCtcProfile`, `MEDASR_MLX_FP32` pinned profile [mod.rs:MEDASR_MLX_FP32]
- `LasrCtcConfig` and `LasrCtcConfig::from_json` [mod.rs:LasrCtcConfig]
- `LasrVocab` piece table [mod.rs:LasrVocab]
- `ctc_decode(argmax_ids, blank, vocab)` [mod.rs:ctc_decode]

## Invariants
- Pinned checkpoint is `drankush-ai/medasr-mlx-fp32` at revision `3b967580b5176144bc633fac60420d1b122dfba8`, the gated official repo is unverified [mod.rs:MEDASR_MLX_FP32]
- Config must be `model_type == "lasr_ctc"`, SiLU activation, bias-free attention and convolution, MHA only (kv heads equal heads), default RoPE only, `hidden_size` divisible by heads, `pad_token_id < vocab_size`, subsampling stride in 1..=8 [mod.rs:LasrCtcConfig]
- Tensors must be dtype F32 with the exact expected shape or load fails, conv weights are `(out, kernel, in/groups)` [mod.rs:load_tensor_f32] [mod.rs:Conv1dLast]
- The CTC head must be stored as `(vocab, hidden, 1)`, the kernel axis is dropped without transposing [mod.rs:LasrCtc]
- Frontend geometry is fixed: n_fft 512, window 400, hop 160, mel 125 to 7500 Hz, no centering, `ln(max(x, 1e-5))`, audio shorter than 400 samples or non-finite returns `Input` error [mod.rs:LasrFrontend] [mod.rs:N_FFT] [mod.rs:MEL_CLAMP]
- `FftF64::new` asserts a power-of-two size [mod.rs:FftF64]
- Conv `forward` asserts `steps >= kernel`, the depthwise conv uses manual asymmetric same padding left `(k-1)/2` [mod.rs:Conv1dLast] [mod.rs:ConvModule]
- Block residuals use the configured pairs (defaults `[1.5, 0.5]` feed-forward, `[2.0, 1.0]` conv), attention uses a plain residual [mod.rs:EncoderBlock] [mod.rs:residual_weights]
- Layer norm epsilon comes from config `layer_norm_eps` (must be a positive number), not a constant [mod.rs:LasrCtcConfig::from_json]
- Batch norm epsilon is 1e-5, applied with running statistics only (inference) [mod.rs:BatchNormInference]
- Vocab length from `tokenizer.json` must equal `vocab_size`, CTC decode collapses repeats, drops blank and special ids, maps U+2581 to space, strips one leading space [mod.rs:LasrVocab] [mod.rs:ctc_decode]
- Argmax ties resolve to the lowest index [mod.rs:argmax_rows]

## Side Effects
- Reads `config.json`, `model.safetensors`, and `tokenizer.json` from the model directory [mod.rs:LasrCtc]
- Holds all weights as f32 in memory, no GPU, threads, env vars, or global statics [mod.rs:LasrCtc]

## Verification
- Run: `cargo test -p turbospark-audio stt::lasr_ctc`
