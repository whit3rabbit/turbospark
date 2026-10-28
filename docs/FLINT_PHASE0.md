# Flint Phase 0

Facts for the `flint` family (Recursive Flint). The default build is now
`mlx-community/flint-mixed-3-4bit` (its own section below Architecture); the
rest was read off the published `RecursiveSI/flint-int4` artifact at
revision `3b93f9053b530488405ae9fca51d4900d6ce21f0` on 2026-10-21, and
cross-checked
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

## The default build: `mlx-community/flint-mixed-3-4bit`

Read off revision `b1c305855ae7c3464813815be4dbd544b22c36aa` on 2026-10-25:
four shards plus an index, 17,837,530,688 tensor bytes. Same architecture
and config as `flint-int4`, with these differences:

- Packed weights are MLX's usual `U32`, not `U8`.
- `config.json -> quantization` carries per-tensor entries in the SOURCE
  namespace: `model.embed_tokens` at 8 bits and all 141
  `model.layers.{2..48}.mlp.switch_mlp.{gate,up,down}_proj` at 3 bits, group
  64, under a 4-bit default. `mlx_export` says the same
  (`routed_expert_bits: 3`, `embedding_bits: 8`).
- MLX 3-bit packing: a little-endian bitstream, element `j` of a row at bits
  `[3j, 3j+3)`; 8 codes in 3 bytes, 24 bytes per group. Confirmed against
  `mx.quantize` (MLX 0.32.4), and pinned by `mlx_pack_matches_mlx` in
  `crates/gpu/tests/moe_decode_int3.rs`.
- The router `mlp.gate.weight` is still BF16 `[256, 2048]`; intake still
  force-quantizes it to INT8.
- The sidecars (tokenizer, template, generation config) are byte-identical
  to `flint-int4`'s.

Expert blobs are 1,376,256 bytes (1,769,472 at INT4); the install is
17,850,301,582 bytes. Status and measurements: `FLINT.md`.

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
  states 8 bits, as `qwen4_exp` does. The ranking this produces against the
  BF16 original is measured below: a near-identity, not an identity.
- **Dense lead in the HF writer.** Expert blobs exist only for layers
  `lead..num_layers` and are numbered by POSITION (`layer - lead`), matching
  the GGUF walk and the runtime's `L - lead` mapping.
- **Sliding window.** The flow attends `sliding_window` keys including the
  current one, which is llama.cpp's `LLAMA_SWA_TYPE_STANDARD` for
  `n_swa = 513`.
- **Attention scale, norm convention.** `128^-0.5`, plain RMS (no `1 + w`),
  eps 1e-6, q and k normed before RoPE. From the llama.cpp port.

## FP16 range: measured, and fixed by a weight transform

**The open item turned out to be the first real failure.** On the published
`flint-int4`, the prompt "Write a detailed explanation of how a bill becomes a
law in the United States." produced `score vector must not contain a
non-finite entry` at the first token. `TURBOSPARK_FLINT_DEBUG=1` (a per-layer
range report, off by default) located it:

- the residual stream is large but finite (6,072 at layer 46, 6,244 at layer
  47, on a token near the start of the prompt);
- the routed experts' phase-1 product `silu(gate) * up` reaches 16,912, 33,536,
  37,024 and 58,496 on other layers and tokens, and overflowed to `inf` on two
  elements at position 2, layer 46;
- one `inf` in phase 1 makes phase 2's weighted sum NaN in every element, and
  the next RMS norm turns the stream to NaN. The shared expert's output stayed
  finite.

With `up` divided by 2^4 the same prompt ran, and the stored product peaked at
6,056, so the true peak is about 97,000: past FP16's 65,504, so no FP16 engine
can hold it unscaled. This matches why the llama.cpp port added adaptive
input scaling.

**Fix: `crates/repack/src/flint_rebalance.rs`.** Every `up_proj` has its scales
and biases divided by 2^6 and the matching `down_proj`'s multiplied by 2^6
(dense lead, shared expert and every routed expert; the gate is untouched
because `silu` is not linear). `up` and `down` are both linear, so the result
is unchanged, and a BF16 exponent shift is exact and bit-for-bit invertible.
The walk applies it to every new Flint install and records the net shift in
`flint_rebalance.json`; an install built before it exists migrates in place
with `cargo run --release -p turbospark-repack --example flint_rebalance --
<dir>` (refuses a second application, `--undo` reverses it). An install
WITHOUT the shift overflows on prompts like the one above.

Why 2^6: the measured peak needs at least 2^1, and 64 leaves about 43x
headroom for tokens not yet seen. The cost is subnormal precision for stored
products under about 4e-3, negligible beside the large terms that dominate a
down-projection's sum. The shift is a fixed constant, not adaptive; a token
with a product above about 4 million would still overflow.

## Real-model results (2026-10-22, M4 Max 36 GB, not a benchmark)

All on `RecursiveSI/flint-int4@3b93f9053b530488405ae9fca51d4900d6ce21f0`
migrated in place, streamed experts:

- greedy, chat template applied, "capital of France": coherent, a `<think>`
  block then "The capital of France is Paris.", `stop=Eos`;
- greedy, 1,100 new tokens on a long prompt, crossing the 513-token window
  with the sliding layers' ring (641) wrapped: coherent throughout, stopped at
  the token budget while the final answer was beginning;
- the same greedy prompt at 8, 16 and 96 expert-cache slots, and a repeat
  process: **byte-identical output** (slot order is router rank);
- peak physical footprint 0.98 GiB at 8 slots, 1.61 GiB at 16, 7.93 GiB at 96,
  for a 21 GiB model; decode about 19 to 22 tok/s at 8 to 16 slots. A machine
  that was not confirmed quiet, no AC check, one prompt: indicative only.

## Not wired (recorded, not half-built)

- Chunked and batched prefill. Prefill runs through the sequential `produce`
  path.
- Mapped expert residency (the 19.8 GiB expert table is unlikely to fit the
  GPU working set on a 36 GB machine in any case).
- Steering, speculation. There is no MTP head, and measured speculation on
  streamed MoE is about 1.08x at best (`docs/SPECULATIVE_DECODING.md`).
- GGUF intake. Q4_1 is not an executable block type here.
- Tool-call parsing for the GLM-style `<tool_call>` format.
- Memory oracle and quality gate. The catalog rows exist (`flint` for the
  mixed build, `flint-int4`).

## Open items

1. **F16 range.** Fixed for the prompts tried by the rebalance above, with a
   fixed 2^6. Broader prompt coverage (code, other languages, very long
   contexts) has not been measured; run with `TURBOSPARK_FLINT_DEBUG=1` and
   watch for reports near 65,504. The residual stream itself reached 6,244.
2. **Router quantization: measured, effect on quality not.** Over 14,711
   layer-token records from five prompts (English, code, Japanese, a system
   turn, a multi-turn chat), with the router input and INT8 logits dumped by
   `TURBOSPARK_FLINT_ROUTER_DUMP=<file>` and compared with a float32 matmul of
   the original BF16 `mlp.gate.weight` (fetched from the checkpoint) over the
   same FP16 input:
   - the top-6 SET is identical in 97.27 percent of records, and the same SET
     in the same ORDER in 92.58 percent;
   - in the 402 records whose set differs, the 6th and 7th experts are a
     near-tie: the logit gap has a median of 0.004 and a maximum of 0.024, so
     the swapped-in expert carries almost the same weight;
   - where the sets match, the renormalised routing weights differ by at most
     0.004; the largest raw logit error is 1.95, which only matters at a
     selection boundary;
   - the worst layers are 47, 48, 43, 27 and 40 (3.8 to 7.0 percent set
     mismatch).
   About 1.3 of a token's 47 MoE layers run a near-tied neighbour. The
   reference is float32 over FP16-rounded inputs, not the model's own BF16
   arithmetic, and no perplexity or task comparison exists, so whether this
   moves quality is unknown. Removing it means a BF16 router GEMV and a
   manifest router slot that is not 8-bit, which `validate_quant` refuses
   today.
3. **Real-model gates.** Greedy 400-token coherence, sampled coherence ending
   in `EndOfTurn`, the 8-slot against 16-slot digest, memory oracle and quality
   gate have not been run. Nothing in the synthetic suite can show that RoPE is
   correctly absent on the full-attention layers, or compare against a
   reference implementation.
4. **Chat template.** It rendered and the model answered in the expected
   shape, so minijinja accepts `replace("\x00", "")` and
   `tojson(ensure_ascii=False)`. Not yet exercised: system and developer
   messages, assistant reasoning, tool calls and tool results.
5. **Dialect.** Resolved: `ChatDialect::Flint`, probed before GLM (the table
   carries GLM's `<|assistant|>` and `<|observation|>` and was refused for a
   missing `<|endoftext|>` before it existed). Stops are `<|end_of_text|>`
   (150001), `<|user|>` (150003), `<|observation|>` (150005) and
   `<|developer|>` (150018). `<think>` and the tool markup are plain text.
   The think pair now splits into reasoning by id (see `FLINT.md`); no tool
   parser is wired.
6. **Sharding without an index.** The repo ships `model-00000..00004` and no
   `model.safetensors.index.json`; `shard_names` now discovers shards from the
   file listing when there is no usable index.
