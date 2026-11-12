# Bark
## Overview
Port of the Bark three-stage GPT TTS (`mlx_audio/tts/models/bark`): a semantic GPT, a coarse GPT producing two EnCodec codebooks via sliding windows, and a non-causal fine GPT filling six more, decoded through EnCodec at 24 kHz [mod.rs:Model].

## Interface
- `Model` (open, load, generate, generate_text_semantic, generate_coarse, generate_fine) [mod.rs:Model]
- `ModelConfig` (from_json), `GptConfig`, `FineConfig` [mod.rs:ModelConfig]
- `Gpt`, `FineGpt` stage networks [mod.rs:Gpt] [mod.rs:FineGpt]
- `GenerateParams`, `Generated` [mod.rs:GenerateParams] [mod.rs:Generated]
- Token and rate consts: `REFERENCE_COMMIT`, `TEXT_ENCODING_OFFSET`, `SEMANTIC_PAD_TOKEN`, `TEXT_PAD_TOKEN`, `SEMANTIC_INFER_TOKEN`, `CONTEXT_WINDOW_SIZE`, `SEMANTIC_RATE_HZ`, `SEMANTIC_VOCAB_SIZE`, `CODEBOOK_SIZE`, `N_COARSE_CODEBOOKS`, `N_FINE_CODEBOOKS`, `COARSE_RATE_HZ`, `COARSE_SEMANTIC_PAD_TOKEN`, `COARSE_INFER_TOKEN`, `SAMPLE_RATE` [mod.rs:SAMPLE_RATE]

## Invariants
- `Model::generate` forces coarse params to reference wiring (history 1, window 60, no KV caching), ignoring caller values [mod.rs:Model::generate]
- `sliding_window_len` must be > 0 [mod.rs:Model::generate_coarse]
- `generate_fine` needs at least one coarse codebook [mod.rs:Model::generate_fine]
- Semantic draw index `SEMANTIC_VOCAB_SIZE` stops the loop [mod.rs:Model::generate_text_semantic]
- Sampling is delegated to a caller `draw` closure over temperature-scaled logits [mod.rs:Model::generate]
- Text tokenization and voice-prompt history are caller-owned [mod.rs:module doc]
- Only the text-to-semantic stage uses `semantic_history`; `generate_coarse` accepts both history slices and discards them [mod.rs:Model::generate_coarse]

## Side Effects
- Reads config.json and weights from the checkpoint dir [mod.rs:Model::open]
- No RNG of its own (caller `draw`), no env vars, threads, or logging

## Verification
- Run: `cargo test -p turbospark-audio tts::bark`
