# Flint Phase 0

Facts for the `flint` family (Recursive Flint), read off the published
`RecursiveSI/flint-int4` artifact at revision
`3b93f9053b530488405ae9fca51d4900d6ce21f0` on 2026-10-21, and cross-checked
against the open llama.cpp port (ggml-org/llama.cpp#30928, unmerged at the
time of writing). The checkpoint ships no modeling code, so neither source is
a vendor reference. Treat the llama.cpp port as an independent second reading,
not as proof. Read `docs/NEW_MODEL.md` for the bring-up order.

## Architecture

- 49 layers, hidden 2048, vocab 150144, context 262144.
- Layers 0 and 1 are dense (`mlp_only_layers`), FFN width 6144. Layers 2 to 48
  are MoE: 256 routed experts of width 512, top-6, plus one shared expert of
  width 1024 with NO gate tensor.
- Attention: 32 query heads over 4 KV heads, `head_dim` 128, `q_proj` 4096
  rows (no packed output gate), per-head `q_norm` and `k_norm` (plain weight,
  width 128), one learned sink per query head (`self_attn.softmax_offset`, no
  `.weight` suffix), scale `128^-0.5`.
- 14 full-attention layers (0, 1, 5, 9, ..., 45, 48) and 35 sliding layers
  (window 513). `no_rope_layer_indices` equals the full-attention set exactly,
  so NoPE is tied to layer mask 1. The two dense layers are also NoPE.
- Router: `scoring_func: sigmoid`, `norm_topk_prob: true`, no bias. Weight
  `sigmoid(logit_i) / sum(sigmoid over the selected 6)`. This is neither the
  llama flow's softmax over the selected nor deepseek2's softmax over all.
- Quantization: MLX affine group 64. Projections and experts INT4, embedding
  INT8, `lm_head` INT4 and untied. Packed weights are `U8`, not the usual
  `U32`. The router `mlp.gate.weight` is an unquantized BF16 `[256, 2048]`.
- No MTP or nextn tensor in any shard.

## Decisions

- **U8 packing.** `canonicalize_flint_header` re-labels a `U8` weight with a
  `.scales` companion as `U32` (last dimension divided by 4, byte offsets
  untouched). The bit layout is identical: byte `k` holds INT4 elements `2k`
  (low nibble) and `2k+1`, and a little-endian `u32` over four bytes yields the
  same order MLX's `U32` packing writes.
- **Embedding width.** `config.json -> quantization` declares only the INT4
  default. The INT8 embedding is in the producer's `mlx_export.embedding_bits`,
  which `parse_flint_quantization` reads.
- **Router.** Force-quantized to INT8 at repack (the Qwen 3.6 raw-router
  path), so the existing INT8 router GEMV serves it. The manifest router slot
  states 8 bits, as `qwen4_exp` does. **The ranking this produces against the
  BF16 original is unmeasured** (see Open items).
- **Dense lead in the HF writer.** Expert blobs exist only for layers
  `lead..num_layers` and are numbered by POSITION (`layer - lead`), matching
  the GGUF walk and the runtime's `L - lead` mapping.
- **Sliding window.** The flow attends `sliding_window` keys including the
  current one, which is llama.cpp's `LLAMA_SWA_TYPE_STANDARD` for
  `n_swa = 513`.
- **Attention scale, norm convention.** `128^-0.5`, plain RMS (no `1 + w`),
  eps 1e-6, q and k normed before RoPE. From the llama.cpp port.

## Not wired (recorded, not half-built)

- Chunked and batched prefill. Prefill runs through the sequential `produce`
  path.
- Mapped expert residency (the 19.8 GiB expert table is unlikely to fit the
  GPU working set on a 36 GB machine in any case).
- Steering, speculation. There is no MTP head, and measured speculation on
  streamed MoE is about 1.08x at best (`docs/SPECULATIVE_DECODING.md`).
- GGUF intake. Q4_1 is not an executable block type here.
- Tool-call parsing for the GLM-style `<tool_call>` format.
- Catalog row, memory oracle and quality gate. None can exist before a real
  install has been streamed and run.

## Open items

1. **F16 range.** The llama.cpp port needed adaptive MoE-input scaling
   (threshold 1/1024) so Q8 activation block sums stay inside F16, at a 10 to
   30 percent speed cost. This engine keeps FP16 activations. Per-layer maxima
   of the residual stream, the post-norm MoE input and the expert outputs must
   be measured on real prompts before any quality claim.
2. **Router quantization.** Measure top-6 agreement between the INT8 router
   and the BF16 original on real hidden states.
3. **Real-model gates.** Greedy 400-token coherence, sampled coherence ending
   in `EndOfTurn`, the 8-slot against 16-slot digest, memory oracle and quality
   gate have not been run. Nothing in the synthetic suite can show that RoPE is
   correctly absent on the full-attention layers, or compare against a
   reference implementation.
4. **Chat template.** It uses `replace("\x00", "")` and
   `tojson(ensure_ascii=False)`. Check the minijinja renderer accepts both.
   Stop tokens are `<|end_of_text|>` (150001), `<|user|>` (150003),
   `<|observation|>` (150005) and `<|developer|>` (150018).
5. **Dialect.** Confirm `detect_dialect` does not resolve this checkpoint's
   GLM-style special tokens to an existing dialect whose resolver then requires
   a token it lacks.
