# embeddings/src

## Overview
Portable f32 CPU embedding models and the sentence-transformers post-processing they share. Forbids unsafe code. One cargo feature per family, all on by default.

## Interface
- Re-exports: `EmbeddingError`, `Result`, `l2_normalize`, `truncate_and_normalize`, `Family`, `PinnedCheckpoint`, `WeightFormat`, `PINS`, `pool`, `PoolingMode`, `cosine_similarity`, `score_multi_vector`, `score_single_vector`, `similarity_matrix`, `PoolingConfig` [lib.rs:pub use]
- `Weights` (`open`, `open_file`, `load`, `load_any`, `linear`, `embedding`, `is_quantized`), `EmbeddingTable` (`gather`) [weights.rs:Weights]
- `pool`, `pool_one`, `PoolingMode`, `MASK_SUM_FLOOR`, `MAX_POOL_PAD` [pooling.rs:pool]
- `l2_norm`, `l2_normalize_eps`, `L2_EPS` [normalize.rs:l2_normalize]
- `dot`, `similarity_matrix`, `score_single_vector`, `score_multi_vector`, `rank_descending` [similarity.rs:score_multi_vector]
- `PoolingConfig`, `DenseConfig`, `StModule`, `read_modules`, `PromptConfig` [st_config.rs:read_modules]
- `TextTokenizer` [tokenize.rs:TextTokenizer]
- `Family`, `PINS`, `pin`, `pins_for` [pins.rs:PINS]
- `EmbeddingError` (`BadConfig`, `Tensor`, `Input`, `Unsupported`, `Tokenizer`, `Io`, `Kernel`, `ModelIo`), `bad_config`, `tensor_error`, `input_error` [error.rs:EmbeddingError]
- Family modules behind features:
  - [bert](./bert/EGM.md)
  - [clap](./clap/EGM.md)
  - [embeddinggemma](./embeddinggemma/EGM.md)
  - [gte](./gte/EGM.md)
  - [nomic_bert](./nomic_bert/EGM.md)
  - [siglip2](./siglip2/EGM.md)

## Invariants
- `#![forbid(unsafe_code)]` crate-wide [lib.rs:forbid]
- Loads are shape-checked so a config disagreeing with the checkpoint fails at load [weights.rs:Weights::load]
- Only MLX `affine` quantization is accepted: an explicit non-affine `mode` errors, a `quantization` block with no `mode` is treated as affine [weights.rs:quant_scheme]
- All-padding rows divide by a floored mask sum and max-pool to `-1e9` [pooling.rs:MASK_SUM_FLOOR] [pooling.rs:MAX_POOL_PAD]
- L2 norm is accumulated in f64 with epsilon `1e-12` [normalize.rs:l2_norm]
- Every supported checkpoint has an immutable pinned revision [pins.rs:PINS]

## Side Effects
- Reads safetensors, JSON and tokenizer files from a caller-supplied directory [weights.rs:Weights::open] [st_config.rs:read_modules] [tokenize.rs:TextTokenizer::from_file]
- None otherwise (no network, no global state)

## Verification
- Run: `cargo test -p turbospark-embeddings`
