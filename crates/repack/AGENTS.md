# turbospark-repack

Checkpoint intake and `.gturbo` writing.

## Read first

- [Detailed module guide](../../.claude/docs/modules/repack.md)
- [GTurbo format](../../docs/GTURBO.md)
- [Catalog contract](../../docs/MODELS.md)
- [Diagnostics](../../.claude/docs/diagnostics.md)

## Rules

- Keep `#![forbid(unsafe_code)]`.
- Probe config, architecture, tensor names, block types, tokenizer sidecars,
  and published file lists before streaming weights.
- Stream checkpoints layer or tensor at a time. Do not require the full source
  artifact on disk.
- Keep GGUF and safetensors mappings explicit. A header parse or synthetic
  fixture proves intake structure, not runnable model parity.
- Preserve tensor layout and quantization metadata. Independent decoder or
  real-install checks are required before publishing a new path.
- Update both writers when a multimodal or vision component is present. A
  headless install can compile and still be invalid for the advertised model.
- MLX mxfp4 and GGUF MXFP4 hold the same values in a different nibble order.
  MLX packs adjacent elements per byte; GGUF pairs element `j` with `j + 16`
  and folds the exponent into a 17-byte block. `mlx_mxfp4.rs` re-blocks.
  Reading one as the other gives a correctly scaled permutation that
  correlates well and is wrong. Exponent byte 255 is NaN in MLX and finite
  in ggml, so it is refused rather than copied.

## Checks

```sh
cargo test -p turbospark-repack
cargo test -p turbospark-repack --test gguf_checkpoint_network --release -- --ignored --nocapture
```

Run only the network target relevant to the changed intake path.
