# turbospark-embeddings Blueprint

## System Overview
Portable embedding models (EmbeddingGemma, Snowflake arctic-embed BERT/XLM-R/GTE/Nomic, SigLIP2 NaFlex, CLAP) plus sentence-transformers pooling, normalization, similarity and late-interaction scoring. Plain f32 CPU, builds off macOS. Depends on `turbospark-audio`, `turbospark-model-io` and `turbospark-vision-io` [Cargo.toml:dependencies].

## Subdirectories & System Boundaries
- src: [./src/EGM.md](./src/EGM.md) - library code, shared modules and one module per family (bert, clap, embeddinggemma, gte, nomic_bert, siglip2)
- tools: [./tools/EGM.md](./tools/EGM.md) - Python fixture generators

## Global Invariants
- One cargo feature per family, all default [Cargo.toml:features]
- Crate library name is `turbospark_embeddings` [Cargo.toml:lib]
- `testdata/` and `tools/` are excluded from the published crate [Cargo.toml:exclude]

## Global Side Effects
- None at crate level; file reads happen only when callers load a checkpoint directory

## System Verification
- Run: `cargo test -p turbospark-embeddings`
