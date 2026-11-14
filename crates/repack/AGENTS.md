# turbospark-repack

Checkpoint intake and `.gturbo` writing.

## Read first

- [Detailed module guide](../../.claude/docs/modules/repack.md)
- [GTurbo format](../../docs/GTURBO.md)
- [Catalog contract](../../docs/MODELS.md)
- [Flint intake](../../docs/FLINT.md)
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
- Flint intake canonicalizes at the door. `canonicalize_flint_header` renames
  the MLX namespace and relabels U8-packed affine weights as U32, refusing a
  U8 tensor that is not an affine weight. The raw BF16 router is force
  quantized to INT8, the INT8 embedding is declared under the INT4 default,
  and the `softmax_offset` sinks are resident. `parse_flint_config` refuses
  what the decode flow does not implement instead of installing it.
- Every flint install carries a fixed 2^6 SwiGLU up/down rebalance
  (`flint_rebalance.rs`): every `up_proj`'s scales and biases are divided by
  2^6 and the matching down side multiplied, on the dense lead, the shared
  expert, and every routed expert. It is part of install correctness, not an
  optimization. `up` and `down` are linear so the composed result is
  unchanged, the gate is untouched because silu is not linear, and without
  the shift `silu(gate) * up` overflows FP16 and NaNs the stream on ordinary
  prompts.
- The rebalance shifts BF16 exponent fields, so it is exact and invertible
  bit for bit; a value whose shifted exponent would leave range is left
  unchanged and counted, never wrapped. `flint_rebalance.json` records the
  install's net shift (absent means zero). The writer records it at install
  time; `rebalance_install_in_place`, driven by the `flint_rebalance`
  example, migrates an older install in place, and the example refuses a
  second application or an undo of an unshifted install.
- Quantize-on-install (`quantize_on_install.rs`) rewrites the SOURCE, not a
  writer: a BF16 checkpoint is served to the walk as the checkpoint
  `mlx_lm.convert` would publish, so both writers, MTP and vision need no
  change. Quantize with `compute::quant_mlx`, never `quantize_int4_affine`:
  MLX snaps the scale to the larger-magnitude edge and the two differ in
  bytes. Byte parity against the published mlx-community `-bf16`/`-4bit`
  pair is the gate (`tests/mlx_quantize_parity_network.rs`), not
  correlation, which the wrong quantizer also passes.
- Quantize-on-install admits `qwen2` and `qwen3_vl` on inference only: their
  norms are assumed plain `x * w` in both HF and MLX (no shift, from the
  runtime's `encode_rms_norm_bf16w`), and nobody has diffed HF bytes against a
  published conversion. Treat it as unproven until a `*_parity_network` test
  exists for each. A Qwen2 text-only checkpoint spells the trunk `model.*` in
  both layouts, so `canonicalize_for_family` must not run the qwen35 prefix
  map on it (it would rename a bare `lm_head.` and the later
  `canonicalize_qwen2_header` refuses the file as mixed).
- 2-bit quantize-on-install writes group 128 with F16 scales and biases (the
  walk requires F16 for 1- and 2-bit). `mx.quantize` stores companions in the
  input dtype, so byte parity with `mlx_lm.convert` holds only against a
  float16 conversion. A trunk matrix whose columns are not whole groups is
  refused, not served raw.
- An HF transformers export of `qwen3_5` stores trunk RMSNorms zero-centred.
  mlx-vlm adds 1 to `input_layernorm`, `post_attention_layernorm`, `q_norm`,
  `k_norm` and `model.norm`, and the runtime expects that. `linear_attn.norm`
  and `mtp.*` are NOT shifted. Missing the shift, or applying it twice, gives
  fluent wrong output, so the layout is decided by name prefix and refused
  when the layernorm mean disagrees with it.
- Flint expert blob files are numbered by position (layer minus the dense
  lead), not by checkpoint layer, because the lead's dense layers write no
  expert file. `manifest.json` and `packed_experts/layout.json` index them
  by position.

## Checks

```sh
cargo test -p turbospark-repack
cargo test -p turbospark-repack --test gguf_checkpoint_network --release -- --ignored --nocapture
cargo test -p turbospark-repack --test mlx_quantize_parity_network --release -- --ignored --nocapture
```

Run only the network target relevant to the changed intake path. Build the
flint synthetic install test with `--release`; the debug profile is far
slower on it.
