# Speedup playbook: what made Qwen3.8 faster, and where else it applies

October 2026. Started from a review of open-jet's CUDA engine for the same
model (`docs/MLX_KERNELS.md`, "open-jet megakernel review"). Six changes
landed for `QwenGdnDense` (Qwen3.8-27B and the other `qwen35` dense
installs). This page records what each one bought, why it worked, how it was
qualified, and how to carry it to other kernels, families, and drafters. The
two open follow-ups are at the end with enough detail to start from.

All numbers are from an M4 Max with other sessions on the host, interleaved
fresh-process pairs. They are not frozen `docs/BENCHMARKS.md` rows.

## What changed and what it bought

| change | where | effect on Qwen3.8 |
| --- | --- | --- |
| GQA-shared decode attention partial | `crates/gpu/src/shaders/attention.metal` (`attention_decode_gqa_partial`) | decode 1.24x at 5.8K context, 1.76x at 18K; kernel 3-13x from 512 to 32K positions |
| Row-only MTP draft step skips the lm_head | `crates/runtime/src/speculative.rs` | one of three 0.7 GB head reads per block-2 round removed |
| GDN rollback snapshot as a GPU blit | `crates/gpu/src/gdn_state.rs` | 3.7 ms of host copy per round to 0.06 ms; restore 3.8 to 1.95 ms |
| Batched prefill arm by default | `crates/runtime/src/families/qwen/prefill.rs` | 2.4-3.0x prefill over the per-token arm |
| Steel INT4 GEMM in batched prefill | `crates/gpu/src/dequant_int4_steel.rs` | another 2.2x: 2,727 tokens in 21 s against 139 s at the old default |
| Chunked prefill with the MTP head open | `crates/runtime/src/speculative.rs`, `families/qwen/prefill.rs` | MTP-install prefill 7.4x (2,727 tokens: 24.6 s against 181.7 s) |

Together, block-2 MTP decode moved from 29.65 to 32.58 tok/s on a short
prompt, and a 2.7K-token prompt plus ~100 generated tokens went from ~144 s
to ~26 s. The memory oracle peak did not move (661 MiB). Perplexity went
4.9432 to 4.9444 from the attention change and nothing else moved quality.

## How they were found

1. **Profile the real model before touching a kernel.** `TURBOSPARK_PHASES=1
   TURBOSPARK_DISPATCH_PROFILE=1 turbospark-check ...` gives per-kernel
   shares. The prefill profile showed the default arm running the one-row
   GEMV 39 times per token, which is the whole story of slow prefill on this
   family. Shares transfer across runs, absolute times under the profiler do
   not.
2. **Compare the design, not the code, of a faster engine.** open-jet's CUDA
   techniques (dp4a, cp.async, a persistent megakernel) do not port to Metal.
   Its DESIGN choices did: load each K/V tile once per GQA group, never
   snapshot recurrent state you can replay, draft with less than the full
   head. Read for "what does it never do that we do".
3. **Time the kernel alone at the real shapes, then the model.** Each probe
   (`crates/gpu/tests/qwen38_attention_decode_probe.rs`,
   `qwen38_int4_prefill_gemm_probe.rs`) uses distinct weight or K/V buffers
   per sample so nothing is cache-resident from the previous dispatch, and
   interleaves arms. Ask what the bytes and FLOPs say the kernel should do:
   the old attention kernel read 8-25 GB/s on unique K/V, the scalar INT4
   GEMM sat at 3.5 TFLOPS at every width. Both numbers said "not bandwidth,
   not compute, something structural".
4. **Keep the old path one env var away.** Every change has a switch
   (`TURBOSPARK_ATTN_DECODE_KERNEL=barrier`, `TURBOSPARK_BATCHED_GEMV=0`,
   `TURBOSPARK_PREFILL_STEEL=0`, `TURBOSPARK_MTP_CHUNKED_PREFILL=0`). It makes
   the real-model A/B a one-line diff, it is the kill switch, and it keeps the
   old frozen rows reproducible.

## How they were qualified

- **Kernel parity against an FP64 reference**, plus against the kernel being
  replaced, over shapes that exercise masking and ragged edges, then
  mutation-checked: each new test was broken on purpose (wrong head mapping,
  dropped rescale, swapped nibbles, missing bias, under-dispatched grid,
  and missing row copy) and had to fail.
- **Byte identity where the arithmetic allows it.** Three changes are
  bit-exact: the MTP head skip, the GPU snapshot, and chunked priming on the
  per-token and scalar batched arms. Their tests say `==`.
- **A distribution bound where it does not.** Steel rounds weights to FP16
  and reorders the reduction. On the real install it is held to a KL bound
  against the per-token arm (`crates/bench/tests/qwen38_prefill_arm_probe.rs`,
  2.0e-5 nats mean). On fixtures it stays within 1% of the largest logit.
- **Quality gates on both arms.** Run the gate with the old path first. It
  must reproduce the frozen row exactly, which proves the install and host.
  Then run the new path and diff the generated text at the first divergence.
  A reduction-order change moves digests at near-ties. Re-freeze only when
  perplexity holds, the divergence is a swapped phrase, and two fresh
  processes agree. The attention change re-froze `qwen38`, `ternary27b`, and
  `bonsai2`, because a family-wide default reaches every install of the
  family.
- **Speculative losslessness** (`mtp_generation_gate`) after anything that
  touches verify, drafting, rollback, or prefill on an MTP install.

## Measurement traps that cost time here

- **A cold page cache looks like a slow kernel.** A 38-token prompt read 15 s
  on the batched arm against 2 s per token. It was the 15 GB model being
  re-read after another session's job pushed it out of memory, not pipeline
  compilation, and a length threshold built on that reading had to be taken
  back out. Time short prompts warm and check `sysctl vm.swapusage`.
- **Another session's model run halves everything.** Two 27B processes on 36
  GB swap, and a synthetic test that takes 0.1 s took 383 s. Check `ps` for
  other `turbospark-check`, gate or oracle processes before timing.
- **A test that runs two cases at once on one GPU reports nothing about
  speed.** `mtp_generation_gate` prints speedups, but its two tests share the
  device.
- **The gates hash text and print none of it.** To see a divergence, add a
  temporary env-gated dump to `quality_common::generation_digest` and revert
  it with a byte-compare afterwards.

## Applying it elsewhere

### GQA-shared decode attention, other families

Any family whose decode reaches `gpu::encode_attention_decode` with a group
size above 1 can opt in where its `AttentionScratch` is built:
`.with_decode_kernel(gpu::DecodeAttentionKernel::GqaShared)`
(`crates/runtime/src/real_forward_types.rs`). `attention_decode_gqa_supported`
says which shapes the kernel serves today: head_dim 128 or 256 and up to 8 Q
heads per KV head. Ring caches and sinks already work (the combine pass is
shared). To enable a family:

1. Confirm its head shapes pass `attention_decode_gqa_supported`. A 512-wide
   head (some Gemma 4 layers) needs `kGqaMaxPerLane` raised to 16, which costs
   registers, so re-run the probe.
2. Run its quality gate with `TURBOSPARK_ATTN_DECODE_KERNEL=barrier` (must
   reproduce the frozen row) and without, and re-freeze by the protocol above.
3. Re-run its memory oracle. The scratch is the same size, so the peak should
   not move.

The TurboQuant path (`attention_tq.metal`) and the indexed path
(`attention_indexed.metal`) have the same per-Q-head, barrier-per-position
shape and would take the same rewrite.

### Batched, Steel prefill, other families

A chunked-prefill driver whose batched arm calls `encode_gemm_any` can switch
to `encode_gemm_rows` (`crates/runtime/src/real_forward_dispatch.rs`). Its
scratch then carries a flag saying Steel is acceptable, and its micro-batch
can widen toward `MAX_PREFILL_ROWS`. Gemma 4's and dense Llama's batched arms
are the next candidates.

Steel today serves INT4 affine at group 64 with BF16 companions only. Another
width needs another loader adaptation in `scripts/vendor_mlx_steel_qmm.py`,
and MLX's `QuantizedBlockLoader` already takes the bit width as a template
argument. Keep decode and speculative verify on the scalar kernels, because
they must stay bit-exact against sequential decode. Hold a new family to a KL
probe like `qwen38_prefill_arm_probe` before making it the default.

### MTP and other drafters

- **Skip work whose result is discarded.** The row-only draft step was paying
  a full-vocab GEMV to throw it away. Look for the same in DFlash2's block
  draft and any drafter that runs a step only for its KV row.
- **Never copy state on the host per round.** The GDN snapshot pool fits any
  recurrent state. qwen4's QSA and n-gram context are the other cases.
  The step past the blit is open-jet's lazy commit. Commit the recurrent
  state only through the surely-accepted token, and replay accepted drafts
  from their saved `(k, v, g, beta)` inputs at the start of the next verify.
  It needs a commit-row output from the multi-row delta kernel.
- **Prime from the prefill you already ran.** Chunked priming works for any
  drafter whose input is the trunk's per-position hidden state. DFlash2 needs
  its per-layer aux captures encoded in the chunked driver first.
- **Draft with less than the full head.** open-jet drafts against 32K
  frequency-ranked rows plus the sequence's own tokens. That would cut the
  two remaining head reads per round to about 15% of their cost. Taking the
  first N token ids does not work for this tokenizer, since 98K ids are
  needed to cover 99.9% of a code-and-docs corpus (`docs/MTP_SPECULATIVE.md`).

## Next work

### 1. Prime the MTP head per micro-batch, not per row

Today `prime_mtp_rows` (`families/qwen/prefill.rs`) calls
`mtp_prime_step_row` once per prompt position, and each call encodes and
waits on its own command buffer of about twenty single-row dispatches. On a
2,727-token prompt that is 2,727 round trips, and MTP-install prefill reads
24.6 s against 21.0 s for the same prompt without the head, so priming costs
about 3.6 s (~1.3 ms per token).

The MTP block is one full-attention layer plus the dense FFN at trunk
shapes, which is exactly what `batched_layers.rs` already runs for M rows
(`encode_full_attention_block_batched`, `encode_dense_ffn_batched`). The plan:

1. Let those encoders take a tensor prefix (`TRUNK_PREFIX` today, `MTP_PREFIX`
   for the head) and a KV target (the head has its own `MtpState::kv`).
2. Give `MtpState` an M-row `concat` buffer (`MAX_PREFILL_ROWS * 2 * hidden`).
   Build the input stage for M rows: the trunk final norm of `scratch.x` row
   t into `pre_fc_norm_hidden`, the embedding of token t+1 into
   `pre_fc_norm_embedding`, then one `encode_gemm_rows` for `fc`.
3. Run the block over the rows, writing M adjacent MTP KV slots, and advance
   the MTP cursor by the number of rows primed. No head and no logits.
4. Keep the last-row rule: a row with no known next token is not primed and
   its trunk hidden state is copied to row 0.

Gates: `real_forward_qwen35_mtp_chunked.rs` must stay `==` on the per-token
and scalar batched arms (extend it to the batched head), `mtp_generation_gate`
lossless, and `mtp_accept_length_probe` not lower than today, since drafts
primed through Steel would round differently. Target: priming under 0.5 s
for 2.7K tokens.

### 2. Share K/V reads across the rows of a verify or prefill micro-batch

`batched_layers.rs` (the attention loop in `encode_full_attention_block_batched`)
calls `gpu::encode_attention_decode` once per row. Each call reads the whole
K/V range again, so a block-2 verify reads it three times and a 64-row Steel
prefill micro-batch reads it 64 times. At 16K context one row of attention is
about 4.4 ms per token over the 16 attention layers, so a 3-row verify pays
~13 ms where ~4.4 ms would do, and long-prompt prefill becomes
attention-bound as the prompt grows.

Two routes, in order:

1. **Measure what already exists.** `gpu::encode_attention_decode_batch` was
   built for dense Llama prefill. It stages the union of four rows' K/V once
   per tile, and it accepts the linear FP16 K/V this family uses.
   Time it against the per-row GQA loop at Qwen3.8's shape and at M = 3, 16,
   and 64 with `crates/gpu/tests/attention_chunk_bench.rs`, after
   parametrizing its head shape. It does not share across Q heads, so it may
   lose at long context.
2. **Extend the GQA kernel to R query rows.** One threadgroup per (KV head,
   chunk) already serves every Q head of the group. Stage a tile of K/V
   positions in threadgroup memory. Let simdgroups own (row, head) pairs,
   each applying its row's causal end, and merge per (row, head) as today.
   Register pressure rules out holding R rows of Q and O per lane, which is
   why the staging is needed.

**The bit-exactness trap.** Speculative verify must reproduce sequential
decode exactly, or `mtp_generation_gate` stops being lossless. The one-row
kernel's chunk boundaries depend on each row's own range
(`chunk_len = ceil(range / NC)`). Two rows of one verify therefore split
their common positions differently. A kernel that shares K/V tiles across
rows cannot reproduce both one-row results.

Make the chunk geometry position-aligned first, with chunk boundaries that do
not depend on `seq_len`. That changes decode numerics once and needs the
re-freeze protocol above. After it, a multi-row kernel can share K/V and
still be exact against decode. Prefill is held to a KL bound rather than to
identity, so it can take the multi-row kernel before verify does.

Gates:

- A parity test. The multi-row kernel equals the per-row kernel bit for bit
  for verify, and stays within the prefill bound for prefill.
- `mtp_generation_gate` and the quality gate.
- Long-context timing at 4K, 16K, and 32K.
