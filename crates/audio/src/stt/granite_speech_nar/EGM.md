# Granite Speech 4.1 2B NAR

## Overview
Non-autoregressive speech-to-text for the pinned MLX conversion of Granite Speech 4.1 2B NAR. A single `mod.rs` implements a CPU f32 pipeline: log-mel frontend, Conformer encoder with Shaw relative attention and self-conditioning, posterior-weighted BPE pooling for a CTC hypothesis, a Q-Former projector to LLM space, and a bidirectional Granite editor that rewrites the hypothesis with interleaved blank insertion slots.

## Interface
- `GraniteSpeechNarProfile`, `GRANITE_SPEECH_4_1_2B_NAR` [mod.rs:GraniteSpeechNarProfile] [mod.rs:GRANITE_SPEECH_4_1_2B_NAR]
- `GraniteSpeechNarConfig`, `EncoderConfig`, `ProjectorConfig`, `TextConfig`, and `GraniteSpeechNarConfig::from_json` [mod.rs:GraniteSpeechNarConfig] [mod.rs:from_json]
- `compute_features(samples)` returning a flat row-major buffer of `T/2` rows of 160 values (consecutive 80-mel frame pairs, no separate stacking step) [mod.rs:compute_features]
- `GraniteSpeechNar::load(model_dir)`, `profile()`, `config()`, `transcribe(samples)` [mod.rs:GraniteSpeechNar]

## Invariants
- The checkpoint is pinned to `mlx-community/granite-speech-4.1-2b-nar-mlx` at revision `6acb7892...`, conv weights must be MLX layout, so shape checks reject the PyTorch layout [mod.rs:GRANITE_SPEECH_4_1_2B_NAR] [mod.rs:ConvModule::load].
- Config needs architecture `GraniteSpeechNarForASR` and `scale_projected_embeddings == true`, else `BadConfig` or `Unsupported` [mod.rs:from_json].
- `hidden_dim == num_heads * dim_head`, `block_size % downsample_rate == 0`, `hidden_size % num_heads == 0`, and `num_heads % num_kv_heads == 0` are enforced [mod.rs:from_json].
- `encoder_layer_indices` is non-empty, negative values resolve against `num_layers + 1`, none exceed `num_layers`, and count equals `projector.num_encoder_layers` [mod.rs:from_json].
- The encoder hardcodes a 348-wide character vocabulary for self-conditioning softmax and blank probability (index 0). `encoder.out` output_dim is only checked to be positive, but the slicing assumes 348, so another value would slice the wrong width or panic [mod.rs:ConformerEncoder::forward].
- `self_conditioning_layer` must be reached by a block, else `BadConfig` [mod.rs:ConformerEncoder::forward].
- Frontend: non-empty finite input, 512 FFT, hop 160, 80 mels, frames truncated to `2 * (n / 320)`, floor is peak minus 8 dB [mod.rs:compute_features] [mod.rs:N_MELS] [mod.rs:LOG_FLOOR_DB].
- Pooling weights are `(1 - p_blank)` normalized per window with denominator floor 1e-6 [mod.rs:ConformerEncoder::forward].
- CTC collapse dedups adjacent repeats then drops blanks, insertion slots put blanks at even indices with length `max(2n + 1, min_edit_sequence_length)` [mod.rs:ctc_collapse_decode] [mod.rs:add_insertion_slots].
- Audio embeddings are divided by `embedding_multiplier` before concatenation with token embeddings, and the editor multiplies it back, logits are produced only for rows after the audio prefix, tied to the embedding matrix and divided by `logits_scaling` [mod.rs:GraniteSpeechNar::transcribe] [mod.rs:Editor::forward].
- Editor token ids outside `[0, vocab_size)` return `SpeechError::Input` [mod.rs:Editor::embed].
- Editor attention is bidirectional (no mask) with half-rotation RoPE and GQA grouping [mod.rs:EditorAttention::forward] [mod.rs:rope_half].
- Conformer attention is blockwise over `context_size` with zero-padded tail [mod.rs:ShawAttention::forward].
- Tensor loading checks name existence and exact shape [mod.rs:load_tensor].

## Side Effects
- Reads `config.json`, opens `model.safetensors`, loads the tokenizer through `qwen3_asr::load_tokenizer` [mod.rs:GraniteSpeechNar::load].
- Copies all weights into owned f32 vectors [mod.rs:load_tensor].
- No threads, env vars, globals, or GPU use outside the ignored test (env `TURBOSPARK_GRANITE_SPEECH_NAR_MODEL_DIR`) [mod.rs:tests].

## Verification
- Run: `cargo test -p turbospark-audio stt::granite_speech_nar`
