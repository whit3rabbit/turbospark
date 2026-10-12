# Speedup playbook: what made Qwen3.8 faster, and where else it applies

October 2026. Started from a review of open-jet's CUDA engine for the same
model (`docs/MLX_KERNELS.md`, "open-jet megakernel review"). Eight changes
landed for `QwenGdnDense` (Qwen3.8-27B and the other `qwen35` dense
installs), the last two from this page's own follow-up list. This page
records what each one bought, why it worked, how it was qualified, and how
to carry it to other kernels, families, and drafters. The open follow-ups
are at the end with enough detail to start from.

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
| MTP head primed per micro-batch | `families/qwen/mtp.rs` (`mtp_prime_rows_batched`) | 2,727-token MTP prefill 1.0-3.3 s faster in interleaved pairs (23.7-25.5 s against 24.6-27.8 s per row); the head now adds ~0.6 s over headless |
| Multi-row GQA prefill attention | `crates/gpu/src/shaders/attention.metal` (`attention_decode_gqa_rows_partial`) | prefill 1.12-1.13x at 18K tokens (156 s against 176 s), 1.02-1.06x at 2.7-5.8K; kernel 1.8-4.2x over the per-row loop |

Together, block-2 MTP decode moved from 29.65 to 32.58 tok/s on a short
prompt, and a 2.7K-token prompt plus ~100 generated tokens went from ~144 s
to ~26 s. The memory oracle peak did not move (661 MiB). Perplexity went
4.9432 to 4.9444 from the attention change and nothing else moved quality.
After the last two, the head adds about 0.6 s to a 2,727-token prompt, and
an 18K-token prompt prefills in 156 s, against 859 s on the per-token arm
the GQA kernel was first measured on.

**Which install to use.** `qwen38-27b-mtp` is the recommended Qwen3.8
install, and the catalog ranks it above `qwen38-27b` wherever both fit
(`the_mtp_install_outranks_the_headless_qwen38_where_both_fit`). The CLI and
server speculate with its head by default (`--speculative auto`, block 2).
On the greedy protocol it decodes at 21.9-28.7 tok/s against 14.7-17.9
without speculation, for ~212 MiB more peak (872 against 660 MiB at 4K).
Sampled runs speculate too, through exact rejection sampling, but their
speedup is unmeasured, so do not quote the greedy band for them.
Measurement and caveats: `docs/BENCHMARKS.md`, the Qwen3.8 section.

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
   `TURBOSPARK_PREFILL_STEEL=0`, `TURBOSPARK_MTP_CHUNKED_PREFILL=0`,
   `TURBOSPARK_MTP_BATCHED_PRIME=0`, `TURBOSPARK_PREFILL_ROWS_ATTN=0`). It
   makes the real-model A/B a one-line diff, it is the kill switch, and it
   keeps the old frozen rows reproducible.
5. **Count dispatches, not just bytes.** The multi-row attention kernel was
   built to read each K/V tile once per group of rows. The sweep said
   otherwise: more rows per threadgroup lost, and one row per threadgroup
   still beat the per-row loop 1.4-3.8x. The loop paid two serial dispatches
   per row, each 32 threadgroups, too few to fill the GPU. What won was one
   dispatch for the micro-batch with one Q head per simdgroup, which cut
   registers and raised occupancy. Sweep the layout (heads per simdgroup,
   rows per threadgroup, chunks) before reading a speedup as the mechanism
   you designed for.

## How they were qualified

- **Kernel parity against an FP64 reference**, plus against the kernel being
  replaced, over shapes that exercise masking and ragged edges, then
  mutation-checked: each new test was broken on purpose (wrong head mapping,
  dropped rescale, swapped nibbles, missing bias, under-dispatched grid,
  and missing row copy) and had to fail.
- **Byte identity where the arithmetic allows it.** Four changes are
  bit-exact: the MTP head skip, the GPU snapshot, chunked priming on the
  per-token and scalar batched arms, and per-micro-batch priming on the
  scalar arm (the INT4 fixture cannot see the batched GEMM's real-install
  drift). Their tests say `==`.
- **A distribution bound where it does not.** Steel rounds weights to FP16
  and reorders the reduction. On the real install it is held to a KL bound
  against the per-token arm (`crates/bench/tests/qwen38_prefill_arm_probe.rs`,
  2.0e-5 nats mean). On fixtures it stays within 1% of the largest logit.
  The multi-row prefill attention shares chunk boundaries across the batch,
  so it is held to the same probe, which now runs Steel with and without it:
  1.89e-5 against 2.03e-5 nats mean, top-1 6/6 both. Its fixture-level
  parity is in `crates/gpu/tests/attention_decode_gqa_rows_parity.rs`
  (FP64 reference and per-row loop, sentinel rows past the batch).
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
- **A baseline from earlier in the day is not a baseline.** The priming plan
  expected 3.6 s back from a 21.0 s headless number taken hours before. On
  the same host and binary, interleaved, the headless prefill read 23.0-23.9
  s: per-micro-batch priming costs ~0.6 s over it, and per-row priming about
  3 s (1.0-3.3 s more than per micro-batch, paired). Run the reference arm
  in the same interleaved script.
- **Check that the probe reaches the code you changed.**
  `mtp_accept_length_probe` primes the head with `mtp_prime_step`, one token
  at a time, so it cannot see chunked priming. On the real path,
  `TURBOSPARK_SPEC_STATS=1` on the CLI reports accepted tokens per round:
  1.40 and 1.45 at 1,094 and 2,727 tokens, identical between priming modes.

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

A batched prefill arm that loops `encode_attention_decode` per row over
linear FP16 K/V can call `gpu::encode_attention_decode_gqa_rows` once per
micro-batch instead, with a `GqaRowsScratch` (3.1 MiB at Qwen3.8's shape
and 64 rows) on its prefill scratch. It serves the shapes the one-row GQA
kernel serves. It is not bit-exact, so hold the family to a KL probe and
keep decode and verify on the loop. Leave a one-row micro-batch on the loop
too (0.89-1.07x there). Do not start from dense Llama's
`attention_decode_batch_partial` at a GQA shape: it measured 4-12x slower
than the per-row GQA loop at Qwen3.8's.

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
- **Prime from the prefill you already ran, in batches.** Chunked priming
  works for any drafter whose input is the trunk's per-position hidden
  state. Priming one row per command buffer cost about 3 s on 2.7K tokens;
  running the drafter's own block through the trunk's M-row encoders under
  its prefix costs ~0.6 s. DFlash2 needs its per-layer aux captures encoded
  in the chunked driver first.
- **Draft with less than the full head.** open-jet drafts against 32K
  frequency-ranked rows plus the sequence's own tokens. That would cut the
  two remaining head reads per round to about 15% of their cost. Taking the
  first N token ids does not work for this tokenizer, since 98K ids are
  needed to cover 99.9% of a code-and-docs corpus (`docs/MTP_SPECULATIVE.md`).

## Next work

### 1. Give speculative verify the multi-row attention

A block-2 verify runs three rows through `encode_full_attention_block_batched`
and still loops the one-row kernel, because it must match sequential decode
bit for bit. At 3 rows the shared-chunk kernel's production launch measured
1.42-2.38x the loop from 512 to 32K positions (6.7 against 9.8 ms per round
over the 16 layers at 16K).

**A bit-exact batcher is not the way in. This was measured.** A variant that
keeps each row's own chunk length and the one-row kernel's arithmetic,
generated mechanically from `attention_decode_gqa_partial` with only a row
index added and the one-row combine reused over (row, head) pairs, matched
the loop bit for bit, including across the 16K chunk-count step and a
wrapped ring. It ran only 1.06-1.22x at 3 rows and 1.0x at 32K: the dispatch
count alone is worth little, and the layout is what pays. About 2% of a
round, below what the host can confirm, so it was not kept.

So the route is the one the bit-exactness trap already named. Make decode
and verify share one arithmetic. Pick chunk boundaries that do not depend on
`seq_len` or on which rows share a dispatch, and run the rows kernel's
layout for decode as well (one row is a batch of one). That changes decode
numerics once, so re-freeze `qwen38`, `ternary27b` and `bonsai2` by the
protocol above. Price the decode side first: at one row the rows kernel
measured 0.89-1.07x the one-row kernel, a small loss for a headless install
that never verifies.

Gates: bit equality between a one-row decode step and the same row inside a
verify; `mtp_generation_gate` lossless at blocks 2 and 4; the three quality
gates; decode timing at 4K, 16K and 32K with and without speculation.

### 2. Matrix-unit attention for long prompts

After the rows kernel, prefill attention runs at about 1 TFLOPS (1.74 ms
per token over the 16 layers at 16K positions, 64 rows) on scalar FMAs and
`simd_sum`. From the probe's per-token costs, that is roughly a tenth of an
18K-token prefill and a growing share past it. A flash-attention tile on
simdgroup matrices (64 rows x 6 heads is 384 queries per KV head) is the
next step there. Hold it to the same parity test and KL probe, and time it
at 16K and 32K. Below ~8K tokens the INT4 GEMMs dominate prefill and this
buys little.
