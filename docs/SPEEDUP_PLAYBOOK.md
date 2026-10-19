# Speedup playbook: what made Qwen3.8 faster, and where else it applies

October 2026. Started from a review of open-jet's CUDA engine for the same
model (`docs/MLX_KERNELS.md`, "open-jet megakernel review"). Nine changes
landed for `QwenGdnDense` (Qwen3.8-27B and the other `qwen35` dense
installs), two from this page's own follow-up list and the ninth from a later
read of kapitsa, another Apple-silicon engine for the same models. This page
records what each one bought, why it worked, how it was qualified, and how
to carry it to other kernels, families, and drafters. The open follow-ups
are at the end with enough detail to start from.

All numbers are from an M4 Max with other sessions on the host, interleaved
fresh-process pairs. They are not frozen `docs/BENCHMARKS.md` rows.

The audio ports (Whisper, Music 3, Moonshine, Qwen3-ASR) were sped up
separately and found a different bottleneck class: dispatch and sync count,
memory layout, and GEMV thread use, not attention or prefill GEMMs. What
carries over is in "What the audio ports add" below. Their own stories are in
[WHISPER.md](WHISPER.md) and the Music 3 README
(`crates/audio/src/music/minimax_music3/README.md`).

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
| Per-row elementwise and norm ops batched across the micro-batch | `families/qwen/batched_layers.rs`, `prefill_layers.rs`, `verify_layers.rs`, `mtp.rs`; `rope_neox_subdim_rows` in `rope.metal` | prefill 7.5% on 2,564 tokens (20.38 s against 18.85 s, 6 of 6 pairs, identical text); verify and MTP priming not measured, see below |

Together, block-2 MTP decode moved from 29.65 to 32.58 tok/s on a short
prompt, and a 2.7K-token prompt plus ~100 generated tokens went from ~144 s
to ~26 s. The memory oracle peak did not move (661 MiB). Perplexity went
4.9432 to 4.9444 from the attention change and nothing else moved quality.
After the last two, the head adds about 0.6 s to a 2,727-token prompt, and
an 18K-token prompt prefills in 156 s, against 859 s on the per-token arm
the GQA kernel was first measured on.

**The last row, in detail.** The batched Qwen layers still encoded their
small per-row ops (q/gate split, q/k norm, RoPE and output gate in attention,
the FFN activation and residual add, the post-attention residual add and
norm) once per row, in every layer. Each is per head or per element over
contiguous rows, so M rows are the same arithmetic as M times the heads in
one dispatch. RoPE needed one new kernel, `rope_neox_subdim_rows`, which
takes row z's position as `start + z` from the grid. The norm reuses
`rmsnorm_bf16w_perhead` with M "heads" of width `hidden`, which shares the
whole-vector kernel's helper, loop and launch width.

Measured on the 27B (`qwen38-27b.gturbo`, 2,564-token prompt, greedy,
speculation off, six interleaved fresh-process pairs after a discarded
warmup, AC power):

| arm | what is batched | per-row | batched | pairs won |
| --- | --- | --- | --- | --- |
| attention ops only | split, q/k norm, RoPE, gate | 19.57 s | 19.17 s (2.0%) | 5 of 6 |
| plus FFN and prefill residual/norm | the above, `silu_mul`, residual adds, post-attention norm | 20.38 s | 18.85 s (7.5%) | 6 of 6 |

The generated text hashed identically on all 24 runs. The two rows ran on
different builds hours apart, so do not chain their baselines. Host load was
about 4 in the second run and 4.5 to 8 in the first. The switch for the
per-row form is `TURBOSPARK_QWEN_BATCHED_ROW_OPS=0`.

The verify path (`verify_layers.rs`) and the MTP head's batched priming
(`mtp.rs`) take the same change, but there was no MTP or DFlash install on
the host, so they were checked on synthetic installs and bitwise kernel tests
only. There is no real-model verify run and no speculation A/B. The expected
verify saving is about 6 dispatches per layer per round, roughly 1% of a
round, and that figure is an estimate.

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
   `TURBOSPARK_MTP_BATCHED_PRIME=0`, `TURBOSPARK_PREFILL_ROWS_ATTN=0`,
   `TURBOSPARK_QWEN_BATCHED_ROW_OPS=0`). It makes the real-model A/B a
   one-line diff, it is the kill switch, and it keeps the old frozen rows
   reproducible. Read it once through a `OnceLock`, never per layer or token.
   One binary with the switch also removes the build as a variable, and it
   avoids a second multi-GB worktree build when the disk is nearly full.
5. **Count dispatches, not just bytes.** The multi-row attention kernel was
   built to read each K/V tile once per group of rows. The sweep said
   otherwise: more rows per threadgroup lost, and one row per threadgroup
   still beat the per-row loop 1.4-3.8x. The loop paid two serial dispatches
   per row, each 32 threadgroups, too few to fill the GPU. What won was one
   dispatch for the micro-batch with one Q head per simdgroup, which cut
   registers and raised occupancy. Sweep the layout (heads per simdgroup,
   rows per threadgroup, chunks) before reading a speedup as the mechanism
   you designed for.
6. **Grep the batched encoders for a loop over rows.** A `for m in 0..batch`
   that encodes one tiny kernel per row pays the dispatch floor M times in
   every layer. It hid in plain sight because each dispatch is cheap and the
   profile ranks kernels, not launches. The kapitsa comparison pointed at it
   (its `attn_prep` normalizes, rotates and writes K/V for all rows in one
   dispatch). The test for whether a loop can collapse is that the op is per
   head or per element over rows that sit contiguously, with the same stride
   in and out. A norm that writes at twice its input stride, such as the MTP
   priming step that fills one half of the concat buffer, does not qualify.

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
- **Prove the primitive bitwise, then the use.** The row-batched ops are
  trivially exact when elementwise, but the two that borrow another kernel
  are not obvious, so each has a test with `to_bits()` equality.
  `rope_parity.rs` pins `rope_neox_subdim_rows` against one dispatch per row
  at positions past 1,000. `rms_norm_rows_as_heads.rs` pins the per-head norm
  at width 5,120 against the whole-vector norm per row, plain and centered,
  on rows of different magnitude so a reduction that spanned rows would show.
  Both mutations were tried (drop the row term from the position, drop the
  last row from a residual add) and fail the suite, which proves the tests
  reach the code. Use inputs that differ per row, or a kernel that ignores
  the row index passes.
- **Hash the generated text, not the footer.** The first comparison script
  hashed the `[stop=...]` line with its timings and reported 12 different
  outputs for 12 identical runs. Strip the footer before hashing.

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
- **A probe may refuse the install you have.** `batched_forward_probe`
  compares a batched verify pass against a sequential step on a real
  install, but `produce_batched` needs the M-row scratch, which exists only
  when a drafter is asked for at open. A headless `qwen38-27b.gturbo` fails
  with "no multi-token-prediction head", so the real-install verify check
  needs an MTP install. Say so in the commit instead of implying it ran.
- **A dispatch-count estimate overshot by about three.** The first estimate
  used Whisper's 5.4 us floor and predicted 6% from the attention ops. The
  measurement gave 2.0%. Dividing the seconds saved by the dispatches removed
  gives about 1.6 us each for back-to-back tiny kernels in one encoder here
  (0.40 s over roughly 246,000 for the attention ops, 1.53 s over roughly
  900,000 for all of them). This is derived from two timing samples, not a
  floor measurement, but use it for the next estimate and re-check it.

## What the audio ports add

Whisper-tiny.en went from 1.65 s for the first Metal port to 160 ms warm
for a 20 s clip (178-183 ms for whisper.cpp on the same clip), and Music 3's
1 s, 2-step smoke from 118 s to 12 s. None of it came from a new attention
kernel. Sources: [WHISPER.md](WHISPER.md) and commits `277ef473`,
`09d25ea0`, `785e9f95`, `8f83f998`.

Whisper's own history reads as three changes in the order they were found:
a single command pass per token (the logits projection used to be a second
pass), warp-per-row GEMV with packed QKV at m = 1, and a transposed
cross-attention V cache plus a key-range split in the same kernel (that step
2.9 to 1.6 ms). WHISPER.md does not separate the first two, so do not quote
a per-change number for them.

whisper.cpp was an oracle here, not a source. Nothing was ported from it. It
served for text equality on the same clip (the acceptance bar for greedy
decoding) and for its phase split, which calibrated what "fast" meant (about
0.26 ms per warm decode token). Do the same with any engine we compare
against: take its timings as the target and read its design for what it
never does, as with open-jet above, but do not read its code for kernels.

### Techniques that transfer

- **Check which regime you are in with bytes over time.** Whisper's decode
  step reads about 88 MB per token, of which the 80 MB logits projection is
  one kernel at roughly 0.26 ms (about 300 GB/s, most of what this device
  can stream). That leaves ~8 MB of work spread over ~2 ms, so the step is
  dispatch-bound, and skipping the logits GEMV moved the pass only 2.5 to
  2.2 ms. These are arithmetic on the numbers in WHISPER.md, not new
  measurements. A 27B INT4 decode is the opposite regime (bandwidth), so the
  rule is to compute GB/s per kernel first, then pick the lever. For small
  models, dtype was not the lever: f32 weights matched whisper.cpp's fp16.
- **Measure the per-dispatch floor.** Time N repetitions of one tiny kernel
  in a single pass. Whisper's floor is ~5.4 us GPU / ~7 us wall at 100-200
  dispatches per pass. Music 3's synchronous path pays 0.2-0.3 ms per
  dispatch (about 830 per AR frame), a 30-50x higher floor, because each
  dispatch commits and waits. Two workloads on different code, so the ratio
  is indicative only, but it says where the cost is: move work into one pass
  with direct-bound buffer chains before tuning any single kernel. The floor
  is workload-specific, so calibrate before estimating. Qwen3.8 prefill
  measured about 1.6 us per dispatch removed (see the traps list), against
  Whisper's 5.4 us.
- **Attribute by skipping.** Gate one kernel behind an env flag and
  re-measure the pass. It sizes a kernel without the per-dispatch profiler's
  inflation, which only ranks. Pair it with wall versus GPU-busy per pass:
  wall near GPU-busy means the GPU timeline is the cost, wall far above it
  means sync or CPU overhead.
- **Contiguous inner loops beat clever indexing.** Whisper's cross-attention
  was 77% of its decode step because V was stored `[key, dim]` and every read
  strode `d_model`. One transpose to `[dim, key]` at cache build fixed it.
  This only works for caches written once (cross-attention over encoder
  output, or a static prefix). A self-attention KV cache is appended per
  token, so a transpose there costs a strided write per step. Check the
  write pattern before copying the idea.
- **Count lanes doing work, not only dispatches.** The same kernel wasted
  three quarters of its lanes in the value phase (head_dim 64 against a
  256-thread group). Music 3's affine GEMV used 8 of 128 threads per output
  and decoded one 4-bit element at a time. Packing 16 outputs per threadgroup
  and decoding eight codes per 32-bit load cut a path that was 93% of the run
  (microbench 8-19x per call, byte-identical output). A fast probe is
  "active lanes per output" at the real shape.
- **Store scales at the width they need.** Music 3 keeps affine scales and
  offsets as 16 bits when they are exactly representable in bf16 and widens by
  a shift in the kernel (-621 MB resident on the 4-bit profile, any non-bf16
  scale stays f32). The exactness check is what makes it bit-preserving.
- **No env reads in hot loops.** Commit `8f83f998` removed Kokoro and
  Moonshine debug dumps that read env vars per iteration. Resolve a knob once
  at construction. A quick look found the families/qwen reads at state
  construction rather than per token, but nobody has checked every decode and
  prefill loop for this.

### Qualification patterns worth reusing

- **A hash corpus for bit-preserving kernel changes.** `music3_linear_bench`
  (`crates/gpu/examples/`) prints an FNV-1a hash of output bits over 847
  linear and 20 convolution configurations (bit widths, groups, row counts,
  dtypes, ragged tiles, split-K). Diff before and after. Any difference means
  the arithmetic changed. This is cheaper and broader than a handful of
  parity fixtures, and it fits the Steel and GEMV kernels here.
- **Keep the old code verbatim in the test.** `8f83f998` compared every
  changed audio computation against the previous implementation with
  `to_bits()`, which let it share layers across about 30 private copies and
  speed up FFT, mel, resample, conv, sdpa and linear without moving an output
  bit.
- **Whole-pipeline hashes.** Music 3 kept the SHA-256 of the WAV from before
  the work and required every optimization to reproduce it. Keep the
  pre-change artifact hash next to the claim.

### Tried and not kept (audio)

Recorded so they are not re-derived:

- Pooling per-dispatch scratch buffers, and spin-waiting on command-buffer
  completion instead of a condvar. The 64x64 linear floor moved 0.29 to 0.21
  ms, but end-to-end timing did not separate from noise on a busy host.
- A 64x32 tile for tall inputs, and remapping the weight-tile write to avoid
  bank conflicts: no measurable change.
- Skipping the zero-padded taps of a transposed convolution. It can flip the
  sign of a zero accumulator, which is observable, so it is not
  bit-preserving. The same trap applies to any "skip masked or padded rows"
  idea on a path that must match another path bit for bit, speculative
  verify included.
- Restricting Music 3's LM head to its 16,385 audio rows: exact, but the head
  is about 5% of the run, so it was left as a candidate.
- Zero-padding Whisper's 400-point frames to 512 for a faster FFT: it moves
  the bin frequencies the mel filterbank is defined on. A speedup that
  changes what is computed is not a speedup.
- Barriers between dispatches in one pass. A fix that bundled "add barriers"
  with "bind buffers to distinct argument slots" credited the barrier. Remove
  one change at a time before attributing a fix. Binding every buffer to
  slot 0 also caused a 33-minute GPU stall, so require the slot index at each
  call site.

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

### Row-batching small per-row ops, other families

Anything shaped like the Qwen batched layers can take the same change. The
remaining sites found by searching `crates/runtime` for `for m in 0..batch`
and `for r in 0..rows` are listed here so nobody has to re-search.

- **Same file, already done:** attention, dense FFN, prefill residual and
  norm, verify, final head norm, MTP priming (the one skipped loop is the
  strided concat norm).
- **Not done:** the per-row attention-decode loop when `rows_attention` is
  off (it is a real kernel per row, not an elementwise op), the router GEMV
  per row in `verify_layers.rs`, `moe_batch.rs`, `moe_prefill.rs` (a residual
  add at line 493), the sequential arm in `prefill_layers.rs`, and seven
  loops in `dflash_draft/layer.rs` plus `context_kv.rs` and `forward.rs`.
  The drafter loops only move acceptance work, since verification stays
  lossless, so they carry less risk. An earlier read of the rows attention
  kernel suggested it also needs its causal mask turned off, because all
  eight rows share one span. That was not re-checked.
- **Other families unchecked.** The search used the Qwen spelling of the
  loop. Gemma 4 and dense Llama batched arms were not searched, and they are
  the next Steel candidates anyway.

Rules that held: the op must be per head or per element, rows must be
contiguous with the same stride in and out, the new kernel must compute the
position (or any per-row scalar) from the grid with the same float
expression, and a dispatch that crosses a speculative-verify boundary needs a
bitwise test against the per-row form, not a tolerance.

### MTP and other drafters

- **Skip work whose result is discarded.** The row-only draft step was paying
  a full-vocab GEMV to throw it away. Look for the same in DFlash2's block
  draft and any drafter that runs a step only for its KV row.
- **Never copy state on the host per round.** The GDN snapshot pool fits any
  recurrent state. qwen4's QSA and n-gram context are the other cases.
  The step past the blit is open-jet's lazy commit. Commit the recurrent
  state only through the surely-accepted token, and replay accepted drafts
  from their saved `(k, v, g, beta)` inputs at the start of the next verify.
  It needs a commit-row output from the multi-row delta kernel. kapitsa does
  exactly this (`gdn_prep_verify` and `gdn_step_verify_impl` in its
  `verify.metal`, folding `n_pend` accepted rows at the next verify and
  never applying rejected ones), so two engines now agree on it. Our cost to
  remove is the per-round snapshot blit, the 1.95 ms restore wait, the
  replay through `rollback_retaining`, and about 3 tape-copy dispatches per
  GDN layer. Estimated at 3 to 6% on partial-accept rounds, unmeasured. The
  risk is that every other reader of the state (sequential decode,
  checkpoint, prefix-cache save) must flush the pending rows first.
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

### 3. Matrix-unit GEMMs for the audio encoders

Qwen3-ASR's Metal decoder prefills and decodes in about 0.28 s, but the clip
takes a 5.4 s median end to end because the audio tower still runs on the CPU
(`docs/SPEECH_TO_TEXT.md`). Whisper's encoder is 46 of 160 ms warm on an f32
tiled GEMM near 250 GFLOPS, which WHISPER.md says a simdgroup-tiled kernel
would beat several times over. Music 3's 8x8 MMA kernels already got 3x on
DiT-sized linears and 3-4x on convolutions by staging a 32-wide K chunk once
per threadgroup and keeping 2x2 accumulators per simdgroup (`277ef473`,
`785e9f95`). That layout is the starting point, and the Steel loader work above
is the other. Gates: the hash corpus for any kernel that must stay
bit-identical, or the existing encoder parity test (6.4e-4 max on Whisper) for
one that need not. Do the Qwen3-ASR tower first, since it is the larger share
of a real request.

Moonshine is a different case. Metal is already 17.91 against 175.99 ms on one
clip, and what blocks default-on is the 315.9 MiB peak against 119.7 MiB on
the CPU path. That is a memory problem, so apply the memory oracle protocol,
not this page's speed levers.

### 4. Audit the other decode families for passes and lanes

This is a hypothesis list, not a finding. Whisper's biggest early wins were
fewer command passes per token, GEMV at m = 1 with packed projections, and
lane use in the attention value phase. For Gemma 4, dense Llama and qwen4,
count commits per decoded token (`TURBOSPARK_PHASES=1` already splits `cb1`,
routed and final), check for a second pass that only does the final norm and
logits, and compute active lanes per output for their INT4 and INT8 GEMVs at
the real shapes. The GQA kernel serves head_dim 128 and 256 only, so a family
with head_dim 64 never got the lane fix. Measure before changing anything.
Large GEMVs on a 27B model are bandwidth-bound by construction, so expect a
gain only where a profile shows dispatch or sync time.

### 5. What else the kapitsa comparison left open

A read-only comparison of kapitsa's Metal kernels (Apache-2.0, `nace-ai/kapitsa`)
against ours. Nothing below was run or timed, and its host code was not in
the tree that was read. Its headline speed is mostly speculative decoding
(DFlash2), not kernels. In rough order of expected value:

- **Lazy GDN tape.** See the drafters section above.
- **Fewer dispatches in the MoE decode half.** The shared expert runs as six
  separate dispatches in `qwen/moe.rs` and about 11 dispatches per layer in
  all, against about 4 or 5 in kapitsa's fused `moe_gate_up` and `moe_down2`.
  Our shared output is seeded first and rounded to f16 between steps, and
  routed order is reduction order, so fuse by partitioning threadgroups across
  the existing kernel bodies, not by copying its fp32 arithmetic.
- **Fused FFN tail.** Gate, up and `silu_mul` in one dispatch, residual add
  in the down epilogue. Do not copy its folded RMSNorm, which reassociates
  sums.
- **A concurrent compute encoder with explicit barriers.** Ours is serial
  (`context/device.rs`). The same-weight GEMV bench measured only
  1.07-1.40x, so expect little, and a missed barrier gives plausible wrong
  text.
- **Decode matvec shape.** Four rows per simdgroup, sixteen values per lane,
  shift-free nibble masking with a prescaled `x`. Try the masking alone
  first, since it is bit-exact. Our R=4 batch kernel already lost at M=1.
- **Integer-code 8-row verify GEMM** on `simdgroup_half8x8` with no
  threadgroup memory. The largest claimed gain and the riskiest: it is not
  bit-exact against the scalar kernels, and it relies on subnormal half
  codes surviving the matrix unit, which is unverified on our GPU. Prototype
  in a bench first.

Skip: its half-precision inner products (its own comment records NaNs under
fast math), its multi-row attention with shared chunk boundaries for verify
(not bit-exact), and its head-major KV layout.

One measured counterpoint on host-side overhead: building the
function-constant values before a pipeline-cache lookup (`moe_decode.rs`
does it per call) costs 0.56 us, about 45 us per MoE decode token. That is
under 1% of a step and sits behind the GPU, so it is not worth restructuring.

### 6. Device-resident layers for Music 3's AR stage

The documented floor is about 830 synchronous dispatches per AR frame at
0.2-0.3 ms each, roughly 0.30 s per frame against the 0.04 s real time needs.
The Whisper recipe is the model: one pass per step, directly bound buffer
chains, no per-dispatch wait. It needs residual adds, dtype rounding and SiLU
on the device with the host's arithmetic, then a depth-decoder KV cache, and
every step must keep the byte-identical WAV check. Detail and the open gates
are in the Music 3 README.
