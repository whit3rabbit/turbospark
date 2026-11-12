# Whistle Speech Recognition

## Overview
Portable CPU implementation of Cactus-Compute Whistle, a Needle 3 conformer encoder plus gated cross-attention decoder for multilingual ASR. It reads a single Needle 3 container with Hadamard-rotated Cactus Quant (CQ) weights, computes a log-mel frontend, a strided-conv stem, an 8-block encoder, and greedy decoding. It is independent of the MLX affine-weight path and refuses any other Needle geometry.

## Interface
- `Whistle` with `load(&Path)`, `tokenize`, `decode_tokens`, `encode`, `transcribe` [mod.rs:Whistle]
- `WhistleTranscription` (`text`, `language`, `tokens`, `truncated`) [mod.rs:WhistleTranscription]
- `WhistleEmbedding` (`values`, `frames`, `width`), row-major [mod.rs:WhistleEmbedding]
- Crate-private building blocks: `Archive`/`CqMatrix`/`hadamard`/`half` [cact.rs:Archive], `Frontend`/`validate_pcm` [frontend.rs:Frontend], `Stem` [stem.rs:Stem], `Encoder` [encoder.rs:Encoder], `Decoder`/`DecodeState` [decoder.rs:Decoder], `Tokenizer` [tokenizer.rs:Tokenizer], `HadamardMlp`/`Mhc`/`Attention` [layers.rs:Mhc], [attention.rs:Attention]

## Invariants
- `load` accepts only the exact header fields, `rope_theta == 100000.0`, zeroed trailing fields, and the 13-value audio manifest at tensor 237; anything else is `BadConfig` [mod.rs:Whistle::load]
- Container tag must be 0x05e12a84, 28 codebook entries, 1..=4096 tensors, total size <= 64 MiB, finite codebooks and theta [cact.rs:Archive::parse]
- Tensor payloads must be 64-byte aligned, non-overlapping, in bounds, and match the length implied by shape; only dtype 1, 2, 3 (CQ2/CQ4, group 128) and 4 are accepted [cact.rs:Archive::parse]
- Tensor shapes are checked against expected shapes before use, and weights must be finite [layers.rs:floats], [layers.rs:quantized], [cact.rs:Archive::floats]
- Hadamard permutation tensors 226 and 227 must be valid permutations of 0..512 [layers.rs:permutation]
- PCM must be 160..=480000 finite samples in [-1,1] at 16 kHz; the caller resamples [frontend.rs:validate_pcm]
- The mel bank must be [257,80], finite and non-negative; the Hann window is symmetric (N-1), not periodic [frontend.rs:Frontend::new]
- Frame count truncates partial hops; gain uses the 99th percentile of 20 ms RMS windows [frontend.rs:Frontend::features], [frontend.rs:level_gain]
- `transcribe` requires `max_tokens` in 1..=318 and language in `LANGUAGES`; exact all-zero PCM returns an empty transcript without running the model [mod.rs:Whistle::transcribe]
- Decoding is greedy over the first 8192 logits (not the full 8199 vocabulary), and non-finite logits error [mod.rs:Whistle::transcribe], [mod.rs:finite]
- Decode order is BOS step, then language token (given or argmax over language markers), then generation until EOS; `truncated` is true if the budget ran out [mod.rs:Whistle::transcribe]
- The tied output projection uses unscaled embedding weights while input embeddings are scaled by sqrt(DIM) [decoder.rs:Decoder::step]
- Engram lookups apply only at decoder layers 3 and 7 [decoder.rs:Decoder::step]
- Tokenizer parse rejects bad vocab size, duplicate or invalid pieces, incomplete byte fallback, bad BOS/EOS/unk types; decode rejects IDs outside the vocabulary [tokenizer.rs:Tokenizer::parse], [tokenizer.rs:Tokenizer::decode]
- CQ projection quantizes activations to int8 per 128-group after Hadamard rotation; `row` returns inverse-rotated dequantized weights, and `project` asserts input length is a multiple of `cols` [cact.rs:CqMatrix::project], [cact.rs:CqMatrix::row]

## Side Effects
- Reads the checkpoint file fully into memory (bounded by 64 MiB + 1 via `take`) [cact.rs:Archive::read]
- No threads, env vars, global statics, caches, or GPU/Metal use; all compute is CPU f32/i32 [mod.rs:Whistle]
- FFT scratch is allocated per `features` call [frontend.rs:Frontend::features]

## Verification
- Run: `cargo test -p turbospark-audio stt::whistle`
