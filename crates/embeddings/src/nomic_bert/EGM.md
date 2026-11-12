# nomic_bert (embeddings)

## Overview
The Nomic BERT encoder behind arctic-embed-m-long: non-interleaved (NeoX) rotary attention with dynamic NTK base scaling for long sequences.

## Interface
- `NomicBert` (`load`, `with_truncate_dim`, `embed`, `embed_with_prompt`, `embed_batch`, `embed_ids`, `head`, `tokenize`) [nomic_bert/mod.rs:NomicBert]
- `Encoder` (`load`, `forward`, `forward_traced`, `inv_freq_for`), `ntk_base`, `inv_freq`, `rope_tables` [nomic_bert/encoder.rs:Encoder]
- `NomicBertConfig`, `MLP_MULTIPLE_OF` [nomic_bert/config.rs:NomicBertConfig]

## Invariants
- The RoPE base is rescaled per sequence length by `ntk_base`, so tables depend on `n` [nomic_bert/encoder.rs:ntk_base]
- MLP hidden size is rounded to `MLP_MULTIPLE_OF` (256); padded vocab is rounded to the checkpoint's `pad_vocab_size_multiple` (default 1) [nomic_bert/config.rs:padded_vocab_size]

## Side Effects
- Reads checkpoint and tokenizer files at `load` [nomic_bert/mod.rs:NomicBert]
- None otherwise

## Verification
- Run: `cargo test -p turbospark-embeddings`
