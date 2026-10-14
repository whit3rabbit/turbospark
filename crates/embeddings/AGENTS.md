# turbospark-embeddings

Portable embedding models plus the sentence-transformers post-processing
they share: pooling, L2 and Matryoshka normalization, cosine and ColVision
(single-vector and MaxSim) scoring.

## Read first

- [Model inventory and pins](MODELS.md)
- Family docs: [EmbeddingGemma](src/embeddinggemma/README.md),
  [SigLIP2 NaFlex](src/siglip2/README.md), [CLAP](src/clap/README.md),
  [BERT / XLM-RoBERTa](src/bert/README.md), [GTE](src/gte/README.md),
  [Nomic BERT](src/nomic_bert/README.md)
- [Audio crate rules](../audio/AGENTS.md): this crate reuses its kernels,
  loaders, and porting conventions
- [Testing rules](../../docs/TESTING.md)

## Scope and references

- The family list follows the mlxrs `embeddings` module
  ([findit-studio/mlxrs](https://github.com/findit-studio/mlxrs/tree/main/mlxrs/src/embeddings)).
  mlxrs is a design guide only. The numeric reference is Hugging Face
  `transformers` 4.57.1 and `sentence-transformers` 5.1.2 run on the pinned
  checkpoint, because mlxrs's own CLAP gate compares against int8 ONNX.
- Every checkpoint is pinned to a full commit in `src/pins.rs`, and every
  MLX conversion names the upstream checkpoint it came from. Add the pin
  before adding a loader path for a new checkpoint.
- Never load a pickle (`pytorch_model.bin`). CLAP is pinned to the HF
  safetensors-conversion revision for this reason.

## Rules

- Plain f32 on the CPU, no Metal, no macOS APIs, no unsafe. Metal offload
  belongs in the runtime layer later. Verify with
  `cargo check --target x86_64-unknown-linux-gnu -p turbospark-embeddings`.
- Kernels come from `turbospark_audio::ops` and `nn`. Fix a kernel there,
  not in a private copy. A family-specific variant is allowed only when the
  shared kernel's numeric contract differs from the reference, and the
  module says why.
- Shape-check every load (`weights::Weights::load`). A config that
  disagrees with the checkpoint must fail at load.
- MLX affine checkpoints dequantize at load. Quantized embedding tables
  stay packed and dequantize per gathered row. Refuse any quantization
  mode other than affine.
- Refuse config values a family has not implemented (activation, pooling
  mode, fusion, softcapping) instead of ignoring them.
- One cargo feature per family, all default. Shared modules must build with
  `--no-default-features`.

## Fixtures and tests

- `tools/gen_<family>_fixtures.py` writes `testdata/<family>/` from the
  pinned reference, with a `manifest.json` recording package versions,
  revisions, and file digests. Regeneration is separate from tests, and
  tests never download.
- Default suite: tiny random-init models built by the reference, plus
  frontend (tokenizer, image, mel) parity. No install needed.
- `#[ignore]`d suite: the pinned real checkpoints, resolved from
  `TURBOSPARK_<FAMILY>_DIR` or `~/models/embeddings/<name>`, skipping when
  absent. Report max abs diff and min cosine. Do not widen a tolerance
  without a new measurement.

## Verification

```sh
cargo test -p turbospark-embeddings
cargo test -p turbospark-embeddings --no-default-features
cargo test -p turbospark-embeddings --release -- --ignored --nocapture
cargo check --target x86_64-unknown-linux-gnu -p turbospark-embeddings
cargo clippy -p turbospark-embeddings --tests
```

From a Mac the Linux check fails in `onig_sys` (C code under
`tokenizers`, same as the audio crate) unless a Linux C compiler is set.
`zig cc -target x86_64-linux-gnu` works when wrapped to drop the
`--target=` flag cc-rs adds, passed as `CC_x86_64_unknown_linux_gnu` with
`zig ar` as `AR_x86_64_unknown_linux_gnu`. CI's native Linux leg needs
none of this.

Format your own files with `rustfmt --edition 2021 <file>`. Package-wide
`cargo fmt` rewrites other sessions' work.
