# turbospark-vision-io Blueprint

## System Overview
Portable vision preprocessing: decode, PIL-bicubic smart resize, patchify, and position tables for the vision towers, plus per-family pipelines (CLIP, MiniCpm, DeepSeek-OCR, DeepSeek-VL2, Pixtral, Muse, OneVision, Unlimited OCR2, Gemma 4) and the M-RoPE splice; see src/EGM.md. See [AGENTS.md](./AGENTS.md) and [README.md](./README.md) for the human-maintained guidance.

## Subdirectories & System Boundaries
- src: [./src/EGM.md](./src/EGM.md) - all library code

## Global Invariants
- Library name is `turbospark_vision_io`; depends on `turbospark-core` (imported as `foundation`), `image` (jpeg, png only) and `serde_json` [Cargo.toml:dependencies]

## Global Side Effects
- `decode_image_file` reads a file from disk [src/decode.rs:decode_image_file]; no network, no GPU

## System Verification
- Run: `cargo test -p turbospark-vision-io`
