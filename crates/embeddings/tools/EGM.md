# embeddings/tools

## Overview
Python scripts that regenerate the parity fixtures under `testdata/<family>/` from the pinned Hugging Face reference implementations (transformers, sentence-transformers). Not shipped in the crate (`exclude` in Cargo.toml). Tests never download; regeneration is a separate manual step.

## Interface
- `gen_bert_fixtures.py` [tools/gen_bert_fixtures.py:--tiny, --real TAG MODEL_DIR, --mlx TAG MLX_DIR, --model-dir]
- `gen_gte_fixtures.py` [tools/gen_gte_fixtures.py:argparse]
- `gen_nomic_bert_fixtures.py` [tools/gen_nomic_bert_fixtures.py:argparse]
- `gen_embeddinggemma_fixtures.py` [tools/gen_embeddinggemma_fixtures.py:--tiny, --real, --mlx, --qat, --model-dir]
- `gen_siglip2_fixtures.py` [tools/gen_siglip2_fixtures.py:argparse]
- `gen_clap_fixtures.py` [tools/gen_clap_fixtures.py:argparse]

## Invariants
- Reference is Hugging Face transformers 4.57.1 in f32 on CPU. The bert generator uses no trust_remote_code [tools/gen_bert_fixtures.py:module docstring]; the gte and nomic_bert generators load their pinned remote code with trust_remote_code=True and HF_HUB_OFFLINE forced [tools/gen_nomic_bert_fixtures.py:module docstring]
- The SigLIP2 generator must run with torchvision so the Fast image processor is used [tools/gen_siglip2_fixtures.py:module docstring]

## Side Effects
- Writes fixture files (npy, json, safetensors, manifest) under `testdata/<family>/`
- Reads real checkpoint directories passed on the command line for `--real` runs

## Verification
- Run with the pinned venv python (the scripts import torch at module level, so a bare python3 fails): `emb-venv/bin/python -I crates/embeddings/tools/gen_bert_fixtures.py --help` (regeneration only); consumers are checked with `cargo test -p turbospark-embeddings`
