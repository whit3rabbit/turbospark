# siglip2 (embeddings)

## Overview
google/siglip2-base-patch16-naflex dual tower with variable-resolution (NaFlex) image input, a text tower, and sigmoid image-text logits.

## Interface
- `Siglip2` (`load`, `preprocess`, `embed_pixels`, `embed_image`, `embed_image_file`, `tokenize`, `embed_ids`, `embed_text`, `logit`, `logits`, `probabilities`, `logit_params`), `Siglip2Tokenizer`, `sigmoid` [siglip2/mod.rs:Siglip2]
- `VisionTower`, `VisionTrace`, `resize_grid_aa_bilinear` [siglip2/vision.rs:VisionTower]
- `TextTower`, `TextTrace` [siglip2/text.rs:TextTower]
- `NaflexInput`, `target_size`, `preprocess`, `resized_image`, `resize_bilinear_aa_u8` [siglip2/processing.rs:preprocess]
- `Siglip2Config`, `TextConfig`, `VisionConfig`, `ProcessorConfig`, `TEXT_MAX_LENGTH`, `HIDDEN_ACT`, `FAST_PROCESSOR` [siglip2/config.rs:Siglip2Config]

## Invariants
- Text is padded to `TEXT_MAX_LENGTH` (64) and `tokenize` (so the first `embed_text`, not `load`) fails if the tower holds fewer positions [siglip2/mod.rs:Siglip2::tokenize]
- Only the `gelu_pytorch_tanh` activation is supported [siglip2/config.rs:HIDDEN_ACT]
- Padded patches are excluded via `pixel_attention_mask` [siglip2/processing.rs:NaflexInput]

## Side Effects
- Reads checkpoint, tokenizer and image files from disk [siglip2/mod.rs:Siglip2::load] [siglip2/mod.rs:Siglip2::embed_image_file]
- None otherwise

## Verification
- Run: `cargo test -p turbospark-embeddings`
