# QwenImage vendored package provenance

This package is an original TurboSpark port of the Qwen-Image-2.1
text-to-image pipeline to Swift + MLX. It is vendored next to the Z-Image
package and follows the same bring-up pattern, but it is not a fork of an
existing Swift repository: no public Swift or Python MLX implementation of
Qwen-Image-2.1 existed when this was written.

The behavioral reference is the HuggingFace diffusers implementation at the
commit range current when this port was made (diffusers 0.37.0.dev0):

- `src/diffusers/models/transformers/transformer_qwenimage21.py`
  (single-stream block-causal transformer, 3-axis rope, shared modulation,
  prefix KV cache)
- `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`
  (raw prompt template strings, last-decoder-hidden-state extraction before
  the final norm, latent packing, true-CFG-off default sampling,
  64-channel per-channel latent denormalization)
- `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`
  (RMS-norm residual decoder with DupUp3D channel-duplication shortcuts)
- `src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`
  (dynamic exponential time shift and stretch-to-terminal sigma schedule)

The tokenizer wrapper follows the same swift-transformers `AutoTokenizer`
approach as the vendored Z-Image package.

Weight layout target: `mlx-community/Qwen-Image-2.1-MLX-4bit` (revision
`4db4e8c0c0e7a1debf0320415bec8388e888494c`), an MLX affine 4-bit group-64
conversion whose component configs embed their quantization metadata.

First-cut scope, kept deliberately narrow and documented in
`docs/QWEN_IMAGE_21_MLX.md`:

- text-to-image only (no reference images, masks, or RGBA compositing)
- batch size 1 (no left padding path, no cross-batch mask)
- no true CFG (the model samples without guidance; `trueCfgScale` is fixed
  at 1.0)
- local snapshot directories only (no Hub download fallback)

The `qwen-research` license of the upstream weights governs their use; this
package contains no weights.
