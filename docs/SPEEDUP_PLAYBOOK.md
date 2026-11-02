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
  900,000 for all of them). That was derived from two timing samples. The
  dispatch probe has since measured it: **1.87 us per chained dispatch** in
  Qwen3.8's dense decode trunk (per-round fits 1.71-1.96 us, ledger
  `qwen38-dispatch-probe-3`, `docs/MLX_SERVE_KERNELS.md` L1). Use that for
  the next estimate, and remember it prices a dispatch on the dependency
  chain: a fusion that removes dispatches the GPU already overlaps saves
  less.

- **A forced-kernel variable changes more than the thing you meant.**
  `TURBOSPARK_ATTN_DECODE_KERNEL=barrier` forces every attention scratch,
  including the trunk's verify, so an arm built with it read a verify of 75 ms
  against 67 and was not the baseline. The baseline is the variable unset.
- **The bench generates twice and prints its stats twice.** The first pass
  warms up and the second is the one the result row reports. On a loaded host
  the two can differ a lot (a draft phase of 29 against 21 ms), so use the
  second block. The dispatch profile accumulates across both passes: use the
  first block, or divide.
- **A wall-clock A/B on a swapping host is not an A/B.** With 7-8 GB of swap
  in use one run took 22 minutes for 845 tokens. When the host is in that
  state use GPU-time sources (the dispatch profile, a per-dispatch microbench)
  and say that is what the number is.
- **A shared `CARGO_TARGET_DIR` across worktrees can skip a rebuild.**
  Workspace crates hash without their absolute path, and a tree whose sources
  are older than the first tree's outputs counts as fresh: the second build
  said `Finished` in 0.08 s and produced a byte-identical binary. Touch the
  changed files and check `strings` for a symbol only that tree has.
- **Skipping a step to attribute its cost can abort the run.** Skipping an
  attention dispatch left stale NaNs that the next stage refused. Skip only
  steps whose output the rest tolerates, and check the run completed before
  reading a time.
- **A model-free draft source is cheap to size offline.** The prompt-lookup
  replay needed one plain generation per case and no verify pass, because
  greedy acceptance is equality against that stream. Replay before wiring.
- **A per-dispatch time is not a round time.** A microbench put the 8-row tile
  2.7-5.6x ahead of the scalar GEMM on two small shapes, and the dispatch
  profile agreed (a 2.5 ms saving), but a clean two-round wall-clock A/B read
  the verify 1.5-5.5 ms slower. The profile gives each dispatch its own encoder
  and waits on it, and a standing microbench pool is not the layer-to-layer
  cache and dependency state of a real pass. Rank with the profile and
  microbench, but accept a change on the wall-clock A/B, taken on a quiet host.

## Opt-in: the ANE+Steel prefill split

Not a default. Steel alone stays the dense qwen prefill arm. The split is a
cargo feature (`ane-prefill`, off by default) plus a runtime switch, for
long prompts on Apple Silicon.

**What it does.** Each dense projection class is split by output rows. A
Core ML program runs the first rows on the Neural Engine while Steel
computes the rest straight into the same output. The classes are MLP
gate/up, MLP down, attention q with its gate, GDN qkv with z, and
o_proj/out_proj. K/V and GDN a/b stay on the GPU. The code is
`crates/runtime/src/families/qwen/ane_split.rs` and
`crates/gpu/src/ane_split.rs`.

**What it bought.** Measured on Qwen3.8-27B on an M4 Max with a 2,940-token
prompt at a 2048-row micro-batch. Prefill ran 1.347x faster than Steel alone
at the same width (mean of six interleaved runs, 1.370x median). Modeled energy was
0.646x (IOReport, not wall power). Greedy output was unchanged in all 47
runs, and `qwen38_prefill_arm_probe` read 1.512e-5 nats mean KL against
Steel's 1.547e-5 at the same width.

**How to run it.**

```sh
uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \
    python -I crates/coreml/tools/gen_ane_prefill_programs.py ~/models/ane-prefill/qwen38
cargo build --release -p turbospark-cli --features ane-prefill
TURBOSPARK_ANE_PREFILL=1 TURBOSPARK_ANE_PREFILL_DIR=~/models/ane-prefill/qwen38 \
    target/release/turbospark-check --model <qwen38 install> --prefill-chunk 2048 ...
```

The generator writes the Core ML programs and a `manifest.json` with the
shares. A library caller uses `RealForwardRunner::set_ane_prefill` instead
of the env vars. The split needs micro-batches of at least 512 rows, so a
prefill chunk of 2048. It steps aside while an MTP head or DFlash2 drafter
is open, and for image micro-batches.

The programs load on the first split micro-batch, outside the timed prefill
above. That took about 6 s the first time a new build ran them and about
0.5 s after that.

Two diagnostics. `TURBOSPARK_ANE_PREFILL=break` keeps every buffer break
with every row on Steel, which prices the breaks alone.
`TURBOSPARK_ANE_PREFILL_STATS=1` prints per-class prediction and Steel
times.

**Failure handling.** Missing or mismatched programs turn the split off with
one warning, and prefill runs on Steel. Each split micro-batch takes a GDN
snapshot before it commits. A failed prediction or a non-finite ANE row
restores it and redoes the micro-batch on Steel.

**What transferred.**

- **Balance each share against the buffer it overlaps.** The prediction runs
  beside Steel's rows of the same class plus that class's Steel-only work.
  For GDN that includes a/b on the scalar kernel, about 19 ms per layer at
  2048 rows. Phase 0's per-matmul 47% left the ANE idle 16 ms per qkv+z
  point. About 70% balanced it. Fit per shape from in-engine counters: a
  prediction cost about 1.4 ms plus 2.0 us per ANE row at K = 5120 and
  2048 tokens. Each share leaves the prediction at about 0.96 of its
  buffer, so the GPU never waits.
- **Breaking the command buffer costs nothing on the GPU.** All 256 split
  points ran 0.998x Steel's speed with bit-identical logits. That is 513
  buffers, every one encoded before any commit. The queue needs a raised buffer
  cap, because a default queue blocks the 65th uncommitted buffer forever.
- **The host handoff is the exposed cost, and it grows under load.** On a
  quiet host the handoffs cost about 0.3 ms per point. Under host load in
  Phase 1 they reached 0.9 s per micro-batch. Two of six all-classes runs
  passed the dirty rule but took 0.6 to 1.0 s longer. Their GPU ran at lower
  power, and the cause was not isolated. Keep the ANE finishing first. A late
  prediction stalls the GPU and pays a resume cost too.
- **Measure in the engine, not per matmul.** Phase 0's per-matmul ratios
  predicted 1.32x for this prompt and the engine measured 1.35x. The total
  agreed, but the parts did not. Overlapped a/b beat the model on the
  2048-row micro-batch (1.39x against 1.32x). The padded 892-row tail lost
  to it (1.29x against 1.32x).

**Why it is not a default.** One prompt, one chip, modeled energy. The shares
are per machine, and the 1024-token program's shares were tuned on the
evaluation prompt. Width 2048 costs about 674 MiB of scratch, plus about 816
MiB of IOSurface planes (computed). The FFI builds it by
default and the app's setting defaults to `.auto` (`swift/AGENTS.md`), so
the split runs in the app on M4+ whenever the programs are installed.

The first measurement is a `tslab sweep ane_split` from 2026-10-26 (finding
`qwen35.sweep-ane-split-2026-10-26`). It put the split's peak 1,770 MiB over
Steel's, and `break` (the width without the planes) 774 MiB over. The same
sweep read the split at +46.1% prefill on long-synthesis, and +45.1% with a
DFlash2 drafter open (`docs/DFLASH2.md`).

A default would owe the gates in `docs/BATCHED_PREFILL.md`, the memory
oracle, and wall power. It would also owe a second chip (a base M4 first)
and programs emitted from Rust instead of from Python.

The measurements are in [MOE_PRIOR_ART.md](MOE_PRIOR_ART.md):
[Phase 0](MOE_PRIOR_ART.md#phase-0-in-this-engine-core-ml-matmuls-against-steel)
(per-matmul ratios),
[the width sweep](MOE_PRIOR_ART.md#micro-batch-width-and-the-prefill-split-end-to-end),
[Phase 1](MOE_PRIOR_ART.md#phase-1-in-this-engine-the-gateup-split-end-to-end)
(gate/up alone, 1.153x), and
[Phase 1 extended](MOE_PRIOR_ART.md#phase-1-extended-every-dense-projection-class-end-to-end)
(every class).

### Not pursued: ANE routed experts and asynchronous handoffs (2026-10-21)

Two follow-ups to the split were tested and stopped. Their code and
raw evidence are kept as the tags `archive/codex-ane-moe`,
`archive/codex-ane-async` and `archive/codex-ane-events`. Nothing from
them reached the runtime except the router batch capture below.

- **ANE for routed experts (`matmul_id` with experts used 17+ times).** A
  real Qwen 3.6 35B-A3B capture (`TURBOSPARK_ROUTER_BATCHES=1`, 676,480
  prefill selections) put 28.75% of selections in experts used 17+ times
  within an executable 64-row batch, with a largest group of 63. Core ML
  planned standalone expert programs (2048x512, F16 weights) on the CPU,
  not the Neural Engine, at every group size up to 64. Only 128 rows moved
  gate/up to the ANE, and only 256 moved down. The economics agree
  (modeled): a 17-63 row expert is 0.1-0.4 GFLOP against a measured host
  handoff of about 0.35 ms, the ANE needs F16 weights (3.5x the INT4 bytes,
  expanded on the GPU), and this model's prefill runs about 19 ms per token
  at 16 slots against under 1 ms of routed math. MoE prefill here is bound
  by expert I/O and the per-token routed loop, not compute, so ANE compute
  attacks the wrong term.
- **Asynchronous Core ML prediction.** A completion-handler
  `predict_shared_async` matched synchronous output bits and did not
  predict faster (six fresh-process pairs on the 1024-token gate/up program:
  -0.17 ms, then +0.28 ms). The ceiling is low anyway: 256 handoffs at about
  0.35 ms is about 90 ms of a 15.7 s prefill on a quiet host.
- **Private ANE/Metal shared events.** A standalone probe showed
  device-enforced waits in both directions, and removing either wait failed
  its stall test. It uses the private ANE submission path, not `MLModel`,
  so adopting it means replacing Core ML for the split with a private
  framework to recover under 1%. It is research only.

Reopen the expert idea only once the GPU routed path makes MoE prefill
compute bound, and then as one multi-expert Core ML call rather than one
call per expert. Reopen async or events only if a controlled-contention
run shows handoff tails dominating. The end-to-end reading is the `~hog:N`
A/B below. The microbench that prices the poll alternative is
`tslab kernel --bench ane_handoff --contend N` (a build with
`--features ane-prefill`): it times one split point's `wake + resume` for
each of the install's programs, with `waitUntilCompleted` against a host
poll of a GPU-signaled event, quiet and under N spinner threads. Its spinners load
the CPU only. A tail that comes from GPU or memory contention needs a
different control.

**How to re-test (tslab, 2026-10-27).** Each reopen condition is a ledgered
reading now, so another chip, model or OS re-asks the question without a
campaign:

- *Is MoE prefill compute bound?* `tslab profile --model <moe install>`
  prints `moe bound: ...` per phase and writes `moe_bound.class` into
  `profile_summary` (`io_bound`, `loop_bound`, `compute_bound`, `dense`
  from the routed GPU busy, expert I/O and host-loop shares,
  `crates/lab/src/phases.rs`). The expert idea reopens at `compute_bound`
  on prefill, not before.
- *How hot are the hot experts, and does a batch fit the slots?*
  `tslab expert-census --model <moe install>` runs one router capture
  (`TURBOSPARK_ROUTER_HIST` plus `TURBOSPARK_ROUTER_BATCHES=1`) and records
  `expert_census`: the share of selections in experts used 8, 16, 17 or
  32+ times within an executed batch, the largest in-batch group, the
  expert union per full batch against 16, 32, 64 and 96 slots, and the
  routed sub-batch sizes the union caps allow. The same record is the
  streaming cache's sizing input (`docs/EXPERT_RESIDENCY.md`).
- *Would one multi-expert Core ML call run on the ANE, and at what cost?*
  `tslab kernel --bench ane_matmul --model <install> --programs DIR` (a
  worker built with `--features ane-prefill`; `DIR` from
  `crates/coreml/tools/gen_ane_sweep_programs.py`, whose exact line the
  bench prints) times Steel, the ANE program with its conversions and
  handoff, and the row split at several shares, per install shape and
  token count, with modeled joules beside each point and Core ML's
  placement per program. Packed-expert cases (`E` experts of `g` rows as
  one batched program) run against `E` Steel dispatches. The `ane gate`
  notes fit `main-npu`'s two thresholds (minimum MACs and MACs per byte)
  on this engine's numbers, for speed, energy, or their product.
- *Do handoff tails dominate under load?* An `ane_prefill=split` arm with
  `~hog:N` (N busy threads the orchestrator runs beside the worker, half
  the performance cores is the intended N) against the same arm quiet, on
  an `ane-prefill` worker with `TURBOSPARK_ANE_PREFILL_STATS=1`:
  `tslab ab --model <qwen38> --chunk 2048 --baseline split:ane_prefill=split,ane_prefill_stats=1 --candidate hog:ane_prefill=split,ane_prefill_stats=1~hog:8`.
  The run records parse the `[ane-prefill]` lines into `ane_stats.summary`:
  `handoff_share` (`wake_ms + resume_ms` over `wall_ms`) and its per-point
  p50 and p95. Device-side events are worth their private framework only
  if the hogged share passes 5% while the quiet share stays under 1%; the
  quiet reading here was about 0.3 ms a point, 90 ms of 15.7 s. Hogged runs
  are never quotable.

  **Read on 2026-10-27 (M4 Max, AC, `qwen38-27b`, chunk 2048, 512 points):
  the tails do not dominate.** `handoff_share` was 0.55% quiet and 0.43%,
  0.46%, 0.42%, 0.86% under `~hog:2/4/8/12`, never near 5%. The host wake
  (`wake_ms`, 56 ms) outweighs the GPU resume (`resume_ms`, 5 ms), but there
  is no tail for a spin or QoS change to flatten. Prefill fell from 248.9 to
  226.0 tok/s at `hog:12`. That loss tracks `gpu_busy_ms` (+1.2 s, the Steel
  rows) while `ane_ms` stayed flat, and the handoff delta is about 0.05 s.
  Async handoffs and device events stay closed. Limits: two micro-batches
  per run, integer-thread hogs only, and the median includes the warmup
  round. The summary's p50 equals p95 because it aggregates per
  micro-batch. The earlier 0.9 s slowdown did not reproduce. Ledger:
  `~/.turbospark/lab/ane-handoff-hog`.

  **Queue depth, baked weights and the token axis (same day, same host,
  `tslab kernel --bench ane_matmul`, 17408x5120 and 5120x17408, 2048
  tokens, ms per matmul; Steel is 34.2 to 34.3).**
  - *A second prediction in flight does not help.* Raw predictions on one
    model cost 38.6 to 39.3 ms and on two models driven by two threads 38.6
    to 39.4 ms (`ane_pred_x1`, `ane_pred_x2`). The ANE serializes them, so
    its fixed cost does not overlap.
  - *INT4 constants inside the program are the largest ANE gain so far.*
    `ane_c4` (`gen_ane_sweep_programs.py --const-int4`, uint4 block 64,
    random weights) runs 29.2 to 30.8 ms against 41.1 to 42.9 for the F16
    plane form, beats Steel above about 540 to 840 tokens, and uses about
    0.38 to 0.41 J against 2.1 to 2.4 J for Steel. Core ML placed it on the
    ANE (cost share 1.0). Not yet shown: quality with real weights (the
    engine's affine layout has an fp16 bias, coremltools writes a uint4
    zero point), the memory and compile cost of one program per layer and
    class, and whether the ANE keeps the weights packed at run time.
  - *Splitting tokens works about as well as splitting rows.* `tsplit`
    (ANE takes the first half of the tokens through every row, Steel the
    rest) reads 23.7 to 25.7 ms. `split@0.5` (rows) reads 22.1 to 24.4 and
    peaks near share 0.45 (20.2 to 21.7). Neither prices a fused block's
    lower handoff count, since the bench runs one matrix at a time.
  - *Baked INT4 with a row split is the best reading so far.*
    `split_c4@0.5` (the ANE bakes rows [0, 8704) or [0, 2560) as uint4
    constants, Steel takes the rest) reads 17.9 to 18.3 ms at 2048 tokens
    against Steel's 34.2 to 34.4, about 1.9x. At 1024 tokens it reads 9.2
    to 9.4 against 17.2 to 17.4, and at 512 4.9 to 5.2 against 8.8 to 8.9.
    Share 0.5 is the best of 0.35 to 0.7 at every count: 0.45 reads 19.6 to
    20.1 and 0.6 reads 18.0 to 19.5 at 2048. It also uses about 1.3 to 1.4 J
    against Steel's 2.2 to 2.4. Same limits as `ane_c4`: random weights, no
    quality check, one program per layer and class. This is one matrix, so
    the end-to-end gain will be smaller (the F16 split read 1.35x on a
    whole prefill, with other work unchanged). Run-to-run spread is a few
    percent, with one 34 ms `ane_c4` outlier against 29 elsewhere.
  - *Real weights confirm it.* `crates/coreml/tools/bake_ane_prefill_programs.py`
    bakes rows [0, MA) of named install tensors into programs, refusing a
    tensor whose fp16 scale and offset do not reproduce the BF16 values.
    `tslab kernel --bench ane_matmul --programs DIR --case real:0.mlp.gate_proj`
    then runs Steel, the F16-plane split and the baked split on that tensor's
    own bytes with unit-variance activations. On eight tensors (layers 0, 3
    and 10: gate, down, qkv, out, q and o projections, share 0.5, 2048
    tokens) the baked split reads 7.1 to 19.6 ms against Steel's 12.6 to
    35.4, a 1.78x to 1.93x gain, and is 7% to 22% faster than the F16-plane
    split (1.46x to 1.77x). Its ANE rows differ from Steel's output by a
    relative rms of 6.0e-4 to 8.1e-4 (the F16-plane split: 4.3e-4 to 6.9e-4),
    Steel's own rows are bit-identical, nothing is non-finite, and Core ML
    placed every baked program fully on the ANE. That includes layer 10's
    `in_proj_qkv`, whose tiny scales are subnormal in fp16. Limits: single
    matrices, not a prefill; random activations, not hidden states; the host
    carried other sessions' work (load average 3 to 6).
  - *The runtime takes baked programs (manifest format 2).* One program per
    class, layer and token count, written by `bake_ane_prefill_programs.py
    --layers 0,3` into a `manifest.json` that `TURBOSPARK_ANE_PREFILL_DIR`
    points at. The format picks the form, there is no extra switch. A baked
    point has no weight plane and no dequant pass, and a layer or class with
    no program runs on Steel. With layers 0 and 3 baked (16 programs, the
    same ANE rows as the F16 programs) the split ran 8 points per
    micro-batch instead of 256, and the 24 greedy tokens matched Steel's
    (digest `cc79da79...`). Loading 16 programs took 117.7 s cold (ANE
    compile, about 7.4 s each) and 0.82 s warm (about 0.05 s each). The
    default F16 path is unchanged: 256 points, the same digest, prefill
    15.5 to 15.7 s against Steel's 21.7 s. Only two layers are baked, so
    this proves the path and not the gain.
  - *Baked programs do not scale to the whole model: the ANE stops
    keeping them resident.* Baking gateup and down for all 64 layers at
    width 2048 (128 programs) and running the split gave 51 to 64 s of
    prefill against Steel's 17 s, with the same tokens. The stats show why:
    the Steel rows take the same time as in the F16 run, but the ANE takes
    190 ms per gateup point and 80 ms per down point, against 34 and 18 ms
    for the F16 form and 26 and 15 ms for the same baked programs when only
    16 to 48 are loaded. Sweeping the number of loaded programs: 16, 32 and
    48 are fast, 64 (32 layers of gateup and down) is slow at 140 and 62 ms
    per point, and 64 down programs alone sit in between at 32 ms. The
    cliff does not follow package bytes: 64 uint4 programs (2.1 GiB) are as
    slow as 64 uint8 programs (4.3 GiB). It looks like a limit on how many
    models, or how many weights once compiled, the ANE keeps loaded, with
    each miss costing a reload that grows with program size. The F16 form
    escapes it because two programs serve every layer. Consequence: a
    per-layer baked manifest can cover roughly the first 20 to 24 layers
    of gateup and down, and the other layers fall back to Steel or to a
    shared F16 program, so the whole-model gain measured on single matrices
    (1.74x to 1.87x against 1.30x to 1.71x) does not carry over.
  - *The cliff is wired memory, not a program count.* A standalone probe
    (`turbospark-coreml`, `predict_shared`, no GPU or model) loaded N baked
    gateup programs and timed predictions. 64 programs loaded at once ran at
    a median 26 ms with no slow predictions, the same as 16, and idle gaps
    of 20 or 100 ms and 20 GB of ordinary memory ballast changed nothing.
    Each loaded program holds about 299 MiB of system wired memory (8
    programs +2.4 GB, 64 programs +19.1 GB, linear), about six times its
    47 MB uint4 package: the ANE expands the weights to fp16 (168 MB for
    gateup at 8192 rows x 2) and adds about 90 MB of input and output
    buffers. The runtime process also maps a 15 GB model for the GPU, and on
    a 36 GB machine 64 programs (about 14 GB mixed, 19 GB gateup alone) do not
    fit beside it. Running the probe while a Steel prefill held the GPU and
    the model busy gave 5% slow predictions with stalls up to 9 s. That fits
    thrashing under a wired budget, and it explains why uint8 and uint4
    packages behaved alike. Not proven in isolation: the probe has no
    wired GPU ballast, so the budget itself is inferred.
  - *LUT weights halve the footprint.* The same eight gateup layers built
    with `constexpr_lut_to_dense` (4-bit indices, a 16-entry fp16 table per
    64 weights, exact for the affine grid) hold 160 MiB wired each against
    302 MiB for the shift-scale form, run at the same speed (25.3 ms against
    25.5 ms) and are placed 100% on the ANE. The package is 80 MiB, so the
    ANE appears to keep LUT weights compressed. At that size gateup and
    down for all 64 layers still need about 17 GB wired, more than fits
    beside the model on a 36 GB machine, so it would cover perhaps half the
    layers here and the whole model only on a larger Mac.
  - *The 16-layer LUT hybrid is slower than the F16 split (2026-10-28).*
    Manifest format 2 now mixes baked `const_int4` layers with shared `f16`
    programs (a layer's own program wins), and `bake_ane_prefill_programs.py
    --lut --shared-f16` writes one. Gateup and down for the even layers 0 to
    30 were baked as LUT programs at 2048 tokens (32 programs, 17 s; worst
    table error 0.007 level), the other layers and classes stayed on the F16
    programs. Quality holds: `qwen38_prefill_arm_probe` reads mean KL 2.38e-5
    nats for the split against 2.47e-5 for `break` and 1.73e-5 for Steel
    (bound 1e-3), top-1 6/6, but only the 2000- and 3000-token prompts reach
    a baked layer. Speed does not: on AC, `tslab profile` chunk 2048, three
    rounds each, interleaved hybrid, F16, F16, hybrid, Steel read 172.8 to
    173.2 tok/s in all four runs, the F16 split 238 to 246 tok/s (1.4x), and
    the hybrid 177 to 197 tok/s (1.13x, about 0.81x of the F16 split). The
    ANE arms were flagged DIRTY for foreign CPU over 150% (the Steel arms
    were not), so treat them as steady readings, not frozen rows. The gap
    has two parts, found from per-point timelines of one micro-batch
    (a temporary dump of each point's segment, Steel rows, prediction and
    signal times, since removed). First, the baked form buys nothing: per
    point, the F16 and baked predictions take the same time (gateup 34.3 ms
    against 31.8 ms for 8192 rows, down 18.3 against 16.5), the point is
    bound by the Steel rows (36 ms and 19 ms), and the GPU dequant pass the
    baked form skips costs only 0.7 ms per gateup point. Wall time for a
    2048-row micro-batch is 7.5 s for the F16 split and 7.6 s for the
    hybrid, and 3.75 s against 3.80 s for the 892-row one. The ANE does not
    run faster for having its weights inside the program; the bench's 1.9x
    was against Steel, which the F16 form also gets. Second, a process that
    loads baked programs stalls before its first Steel work: the first
    segment starts 0.45 to 0.64 s after commit with 32 baked programs, 0.2
    to 0.4 s with 16, and 2 to 4 s in a cold process, against 4 ms with the
    F16 programs. That is the rest of the 195 against 240 tok/s, because
    `tslab profile` starts a fresh process every round. The cause was not
    isolated; it grows with the number of baked programs, and the machine
    had 30 GB of 31 GB swap in use. Verdict: baking does not pay on this
    machine in either form, so the F16 split stays the default.
  - *Two mistakes in the baked work, both fixed.* `np.array(q,
    dtype=np_uint4_dtype)` drops the sub-byte tag, so the first baked
    programs held uint8 constants (1.06 bytes per weight, not 0.56). The
    tool now uses `astype`, and the eight-tensor A/B was repeated: same
    error (relative rms 6.0e-4 to 8.1e-4), 1.74x to 1.87x over Steel. And
    a baked manifest lists each class once per layer, which made the split
    count 8,192 points and the runtime refuse with "16385 command buffers";
    `classes_at` now yields each class once. Converting with
    `skip_model_load=True` takes a bake from 8 to 16 s per program to 1 s
    and stops it compiling every program into the Core ML and ANE caches.
  - *Disk.* First use compiles each program into the Core ML `compiled/`
    directory (equal to the package size) and into an ANE cache this
    session cannot read. Observed: 16 uint8 programs used 1.7 GiB, 128 uint8
    programs about 15 GiB, 128 uint4 programs about 13 GiB with transient
    spikes near 8 GiB that vanish when the compile is killed. Cold load is
    about 10 s per program.
  - *The bench's row split was mismeasured until today.* It committed the
    pass that waits on the host ahead of Steel's rows on the same queue,
    so Steel could not start until the prediction finished and the halves
    added (`split@0.5` read 40.1 ms, no better than the ANE alone). With
    Steel committed first it reads 22.3 ms. An earlier `ane_matmul`
    `split@` curve in a ledger likely reads too slow by about the Steel
    half. The runtime split commits Steel's rows first and is not affected.
  Ledgers: `ane-depth`, `ane-c4`, `ane-tsplit2`, `ane-rowsplit-asis`,
  `ane-rowsplit-fixed` under `~/.turbospark/lab`.

**How to re-test (tslab, 2026-10-27).** Each reopen condition is a ledgered
reading now, so another chip, model or OS re-asks the question without a
campaign:

- *Is MoE prefill compute bound?* `tslab profile --model <moe install>`
  prints `moe bound: ...` per phase and writes `moe_bound.class` into
  `profile_summary` (`io_bound`, `loop_bound`, `compute_bound`, `dense`
  from the routed GPU busy, expert I/O and host-loop shares,
  `crates/lab/src/phases.rs`). The expert idea reopens at `compute_bound`
  on prefill, not before.
- *How hot are the hot experts, and does a batch fit the slots?*
  `tslab expert-census --model <moe install>` runs one router capture
  (`TURBOSPARK_ROUTER_HIST` plus `TURBOSPARK_ROUTER_BATCHES=1`) and records
  `expert_census`: the share of selections in experts used 8, 16, 17 or
  32+ times within an executed batch, the largest in-batch group, the
  expert union per full batch against 16, 32, 64 and 96 slots, and the
  routed sub-batch sizes the union caps allow. The same record is the
  streaming cache's sizing input (`docs/EXPERT_RESIDENCY.md`).
- *Would one multi-expert Core ML call run on the ANE, and at what cost?*
  `tslab kernel --bench ane_matmul --model <install> --programs DIR` (a
  worker built with `--features ane-prefill`; `DIR` from
  `crates/coreml/tools/gen_ane_sweep_programs.py`, whose exact line the
  bench prints) times Steel, the ANE program with its conversions and
  handoff, and the row split at several shares, per install shape and
  token count, with modeled joules beside each point and Core ML's
  placement per program. Packed-expert cases (`E` experts of `g` rows as
  one batched program) run against `E` Steel dispatches. The `ane gate`
  notes fit `main-npu`'s two thresholds (minimum MACs and MACs per byte)
  on this engine's numbers, for speed, energy, or their product.
- *Do handoff tails dominate under load?* An `ane_prefill=split` arm with
  `~hog:N` (N busy threads the orchestrator runs beside the worker, half
  the performance cores is the intended N) against the same arm quiet, on
  an `ane-prefill` worker with `TURBOSPARK_ANE_PREFILL_STATS=1`:
  `tslab ab --model <qwen38> --chunk 2048 --baseline split:ane_prefill=split,ane_prefill_stats=1 --candidate hog:ane_prefill=split,ane_prefill_stats=1~hog:8`.
  The run records parse the `[ane-prefill]` lines into `ane_stats.summary`:
  `handoff_share` (`wake_ms + resume_ms` over `wall_ms`) and its per-point
  p50 and p95. Device-side events are worth their private framework only
  if the hogged share passes 5% while the quiet share stays under 1%; the
  quiet reading here was about 0.3 ms a point, 90 ms of 15.7 s. Hogged runs
  are never quotable.

  **Read on 2026-10-27 (M4 Max, AC, `qwen38-27b`, chunk 2048, 512 points):
  the tails do not dominate.** `handoff_share` was 0.55% quiet and 0.43%,
  0.46%, 0.42%, 0.86% under `~hog:2/4/8/12`, never near 5%. The host wake
  (`wake_ms`, 56 ms) outweighs the GPU resume (`resume_ms`, 5 ms), but there
  is no tail for a spin or QoS change to flatten. Prefill fell from 248.9 to
  226.0 tok/s at `hog:12`. That loss tracks `gpu_busy_ms` (+1.2 s, the Steel
  rows) while `ane_ms` stayed flat, and the handoff delta is about 0.05 s.
  Async handoffs and device events stay closed. Limits: two micro-batches
  per run, integer-thread hogs only, and the median includes the warmup
  round. The summary's p50 equals p95 because it aggregates per
  micro-batch. The earlier 0.9 s slowdown did not reproduce. Ledger:
  `~/.turbospark/lab/ane-handoff-hog`.

  **Queue depth, baked weights and the token axis (same day, same host,
  `tslab kernel --bench ane_matmul`, 17408x5120 and 5120x17408, 2048
  tokens, ms per matmul; Steel is 34.2 to 34.3).**
  - *A second prediction in flight does not help.* Raw predictions on one
    model cost 38.6 to 39.3 ms and on two models driven by two threads 38.6
    to 39.4 ms (`ane_pred_x1`, `ane_pred_x2`). The ANE serializes them, so
    its fixed cost does not overlap.
  - *INT4 constants inside the program are the largest ANE gain so far.*
    `ane_c4` (`gen_ane_sweep_programs.py --const-int4`, uint4 block 64,
    random weights) runs 29.2 to 30.8 ms against 41.1 to 42.9 for the F16
    plane form, beats Steel above about 540 to 840 tokens, and uses about
    0.38 to 0.41 J against 2.1 to 2.4 J for Steel. Core ML placed it on the
    ANE (cost share 1.0). Not yet shown: quality with real weights (the
    engine's affine layout has an fp16 bias, coremltools writes a uint4
    zero point), the memory and compile cost of one program per layer and
    class, and whether the ANE keeps the weights packed at run time.
  - *Splitting tokens works about as well as splitting rows.* `tsplit`
    (ANE takes the first half of the tokens through every row, Steel the
    rest) reads 23.7 to 25.7 ms. `split@0.5` (rows) reads 22.1 to 24.4 and
    peaks near share 0.45 (20.2 to 21.7). Neither prices a fused block's
    lower handoff count, since the bench runs one matrix at a time.
  - *Baked INT4 with a row split is the best reading so far.*
    `split_c4@0.5` (the ANE bakes rows [0, 8704) or [0, 2560) as uint4
    constants, Steel takes the rest) reads 17.9 to 18.3 ms at 2048 tokens
    against Steel's 34.2 to 34.4, about 1.9x. At 1024 tokens it reads 9.2
    to 9.4 against 17.2 to 17.4, and at 512 4.9 to 5.2 against 8.8 to 8.9.
    Share 0.5 is the best of 0.35 to 0.7 at every count: 0.45 reads 19.6 to
    20.1 and 0.6 reads 18.0 to 19.5 at 2048. It also uses about 1.3 to 1.4 J
    against Steel's 2.2 to 2.4. Same limits as `ane_c4`: random weights, no
    quality check, one program per layer and class. This is one matrix, so
    the end-to-end gain will be smaller (the F16 split read 1.35x on a
    whole prefill, with other work unchanged). Run-to-run spread is a few
    percent, with one 34 ms `ane_c4` outlier against 29 elsewhere.
  - *Real weights confirm it.* `crates/coreml/tools/bake_ane_prefill_programs.py`
    bakes rows [0, MA) of named install tensors into programs, refusing a
    tensor whose fp16 scale and offset do not reproduce the BF16 values.
    `tslab kernel --bench ane_matmul --programs DIR --case real:0.mlp.gate_proj`
    then runs Steel, the F16-plane split and the baked split on that tensor's
    own bytes with unit-variance activations. On eight tensors (layers 0, 3
    and 10: gate, down, qkv, out, q and o projections, share 0.5, 2048
    tokens) the baked split reads 7.1 to 19.6 ms against Steel's 12.6 to
    35.4, a 1.78x to 1.93x gain, and is 7% to 22% faster than the F16-plane
    split (1.46x to 1.77x). Its ANE rows differ from Steel's output by a
    relative rms of 6.0e-4 to 8.1e-4 (the F16-plane split: 4.3e-4 to 6.9e-4),
    Steel's own rows are bit-identical, nothing is non-finite, and Core ML
    placed every baked program fully on the ANE. That includes layer 10's
    `in_proj_qkv`, whose tiny scales are subnormal in fp16. Limits: single
    matrices, not a prefill; random activations, not hidden states; the host
    carried other sessions' work (load average 3 to 6).
  - *The runtime takes baked programs (manifest format 2).* One program per
    class, layer and token count, written by `bake_ane_prefill_programs.py
    --layers 0,3` into a `manifest.json` that `TURBOSPARK_ANE_PREFILL_DIR`
    points at. The format picks the form, there is no extra switch. A baked
    point has no weight plane and no dequant pass, and a layer or class with
    no program runs on Steel. With layers 0 and 3 baked (16 programs, the
    same ANE rows as the F16 programs) the split ran 8 points per
    micro-batch instead of 256, and the 24 greedy tokens matched Steel's
    (digest `cc79da79...`). Loading 16 programs took 117.7 s cold (ANE
    compile, about 7.4 s each) and 0.82 s warm (about 0.05 s each). The
    default F16 path is unchanged: 256 points, the same digest, prefill
    15.5 to 15.7 s against Steel's 21.7 s. Only two layers are baked, so
    this proves the path and not the gain.
  - *Baked programs do not scale to the whole model: the ANE stops
    keeping them resident.* Baking gateup and down for all 64 layers at
    width 2048 (128 programs) and running the split gave 51 to 64 s of
    prefill against Steel's 17 s, with the same tokens. The stats show why:
    the Steel rows take the same time as in the F16 run, but the ANE takes
    190 ms per gateup point and 80 ms per down point, against 34 and 18 ms
    for the F16 form and 26 and 15 ms for the same baked programs when only
    16 to 48 are loaded. Sweeping the number of loaded programs: 16, 32 and
    48 are fast, 64 (32 layers of gateup and down) is slow at 140 and 62 ms
    per point, and 64 down programs alone sit in between at 32 ms. The
    cliff does not follow package bytes: 64 uint4 programs (2.1 GiB) are as
    slow as 64 uint8 programs (4.3 GiB). It looks like a limit on how many
    models, or how many weights once compiled, the ANE keeps loaded, with
    each miss costing a reload that grows with program size. The F16 form
    escapes it because two programs serve every layer. Consequence: a
    per-layer baked manifest can cover roughly the first 20 to 24 layers
    of gateup and down, and the other layers fall back to Steel or to a
    shared F16 program, so the whole-model gain measured on single matrices
    (1.74x to 1.87x against 1.30x to 1.71x) does not carry over.
  - *The cliff is wired memory, not a program count.* A standalone probe
    (`turbospark-coreml`, `predict_shared`, no GPU or model) loaded N baked
    gateup programs and timed predictions. 64 programs loaded at once ran at
    a median 26 ms with no slow predictions, the same as 16, and idle gaps
    of 20 or 100 ms and 20 GB of ordinary memory ballast changed nothing.
    Each loaded program holds about 299 MiB of system wired memory (8
    programs +2.4 GB, 64 programs +19.1 GB, linear), about six times its
    47 MB uint4 package: the ANE expands the weights to fp16 (168 MB for
    gateup at 8192 rows x 2) and adds about 90 MB of input and output
    buffers. The runtime process also maps a 15 GB model for the GPU, and on
    a 36 GB machine 64 programs (about 14 GB mixed, 19 GB gateup alone) do not
    fit beside it. Running the probe while a Steel prefill held the GPU and
    the model busy gave 5% slow predictions with stalls up to 9 s. That fits
    thrashing under a wired budget, and it explains why uint8 and uint4
    packages behaved alike. Not proven in isolation: the probe has no
    wired GPU ballast, so the budget itself is inferred.
  - *LUT weights halve the footprint.* The same eight gateup layers built
    with `constexpr_lut_to_dense` (4-bit indices, a 16-entry fp16 table per
    64 weights, exact for the affine grid) hold 160 MiB wired each against
    302 MiB for the shift-scale form, run at the same speed (25.3 ms against
    25.5 ms) and are placed 100% on the ANE. The package is 80 MiB, so the
    ANE appears to keep LUT weights compressed. At that size gateup and
    down for all 64 layers still need about 17 GB wired, more than fits
    beside the model on a 36 GB machine, so it would cover perhaps half the
    layers here and the whole model only on a larger Mac.
  - *The 16-layer LUT hybrid is slower than the F16 split (2026-10-28).*
    Manifest format 2 now mixes baked `const_int4` layers with shared `f16`
    programs (a layer's own program wins), and `bake_ane_prefill_programs.py
    --lut --shared-f16` writes one. Gateup and down for the even layers 0 to
    30 were baked as LUT programs at 2048 tokens (32 programs, 17 s; worst
    table error 0.007 level), the other layers and classes stayed on the F16
    programs. Quality holds: `qwen38_prefill_arm_probe` reads mean KL 2.38e-5
    nats for the split against 2.47e-5 for `break` and 1.73e-5 for Steel
    (bound 1e-3), top-1 6/6, but only the 2000- and 3000-token prompts reach
    a baked layer. Speed does not: on AC, `tslab profile` chunk 2048, three
    rounds each, interleaved hybrid, F16, F16, hybrid, Steel read 172.8 to
    173.2 tok/s in all four runs, the F16 split 238 to 246 tok/s (1.4x), and
    the hybrid 177 to 197 tok/s (1.13x, about 0.81x of the F16 split). The
    ANE arms were flagged DIRTY for foreign CPU over 150% (the Steel arms
    were not), so treat them as steady readings, not frozen rows. The gap
    has two parts, found from per-point timelines of one micro-batch
    (a temporary dump of each point's segment, Steel rows, prediction and
    signal times, since removed). First, the baked form buys nothing: per
    point, the F16 and baked predictions take the same time (gateup 34.3 ms
    against 31.8 ms for 8192 rows, down 18.3 against 16.5), the point is
    bound by the Steel rows (36 ms and 19 ms), and the GPU dequant pass the
    baked form skips costs only 0.7 ms per gateup point. Wall time for a
    2048-row micro-batch is 7.5 s for the F16 split and 7.6 s for the
    hybrid, and 3.75 s against 3.80 s for the 892-row one. The ANE does not
    run faster for having its weights inside the program; the bench's 1.9x
    was against Steel, which the F16 form also gets. Second, a process that
    loads baked programs stalls before its first Steel work: the first
    segment starts 0.45 to 0.64 s after commit with 32 baked programs, 0.2
    to 0.4 s with 16, and 2 to 4 s in a cold process, against 4 ms with the
    F16 programs. That is the rest of the 195 against 240 tok/s, because
    `tslab profile` starts a fresh process every round. The cause was not
    isolated; it grows with the number of baked programs, and the machine
    had 30 GB of 31 GB swap in use. Verdict: baking does not pay on this
    machine in either form, so the F16 split stays the default.
  - *Two mistakes in the baked work, both fixed.* `np.array(q,
    dtype=np_uint4_dtype)` drops the sub-byte tag, so the first baked
    programs held uint8 constants (1.06 bytes per weight, not 0.56). The
    tool now uses `astype`, and the eight-tensor A/B was repeated: same
    error (relative rms 6.0e-4 to 8.1e-4), 1.74x to 1.87x over Steel. And
    a baked manifest lists each class once per layer, which made the split
    count 8,192 points and the runtime refuse with "16385 command buffers";
    `classes_at` now yields each class once. Converting with
    `skip_model_load=True` takes a bake from 8 to 16 s per program to 1 s
    and stops it compiling every program into the Core ML and ANE caches.
  - *Disk.* First use compiles each program into the Core ML `compiled/`
    directory (equal to the package size) and into an ANE cache this
    session cannot read. Observed: 16 uint8 programs used 1.7 GiB, 128 uint8
    programs about 15 GiB, 128 uint4 programs about 13 GiB with transient
    spikes near 8 GiB that vanish when the compile is killed. Cold load is
    about 10 s per program.
  - *The bench's row split was mismeasured until today.* It committed the
    pass that waits on the host ahead of Steel's rows on the same queue,
    so Steel could not start until the prediction finished and the halves
    added (`split@0.5` read 40.1 ms, no better than the ANE alone). With
    Steel committed first it reads 22.3 ms. An earlier `ane_matmul`
    `split@` curve in a ledger likely reads too slow by about the Steel
    half. The runtime split commits Steel's rows first and is not affected.
  Ledgers: `ane-depth`, `ane-c4`, `ane-tsplit2`, `ane-rowsplit-asis`,
  `ane-rowsplit-fixed` under `~/.turbospark/lab`.

**What the capture says about the GPU path instead.**
`scripts/moe_route_replay.py --union-cap N` replays
`next_routed_sub_batch`'s greedy shrink over that capture. On full 64-row
layer batches (modeled, not timed):

| union cap | sub-batches per layer batch (median) | tokens per sub-batch (mean) |
| ---: | ---: | ---: |
| 16 | 26 | 2.5 |
| 32 (`MAX_PREFILL_EXPERT_BINDINGS`) | 9 | 6.9 |
| 64 | 3 | 20.1 |

At 16 slots this is why the qwen MoE chunked driver refuses
`TURBOSPARK_ROUTED_BATCH`. At 32 slots the sub-batches sit near the
`c(8) = 0.66` regime the batched routed pair measured in
`docs/BATCHED_PREFILL.md`, so a Qwen routed batch needs at least 32 slots
or a binding scheme past 32.

### The vision tower on the Neural Engine: not pursued (2026-11-01)

The question was whether a Qwen vision tower block runs faster on the ANE
than on the GPU's FP16 vision kernels. Recommending an ANE tower needs both:

1. Per block, the ANE's qkv, proj, fc1 and fc2, with the activation scale,
   the output convert and the host handoff paid, beat
   `gpu::encode_vision_matmul` at the tower's patch-row counts.
2. The tower is compute bound, not bound by streaming its blocks from
   `packed_vision/` (two slots, each block's host `pread` runs before its
   compute, not beside it).

Instruments: `tslab kernel --bench vision_encode` (per-stage tower time on
three synthetic pages, `stream_share` = block read over block read plus
block GPU) and the `vision_*` cases of `--bench ane_matmul`, with programs
from `crates/coreml/tools/bake_ane_vision_programs.py` (weights as F16
constants, since a tower's weights are FP16 already).

Measured on Apple M4 Max, macOS 26.6, AC, the combined `qwen3vl-4b-vision`
install (`docs/MODELS.md`'s pinned pull). Ledgers `vision-lane` and
`vision-ane`, findings in `crates/lab/families/qwen3_vl.json`.

Tower per page, two runs (both DIRTY from another session's CPU load,
agreeing within 5%), median of 5 reps:

| page | patches | total ms | block GPU ms | block read ms | stream share | peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512x640 | 1280 | 2546 / 2450 | 2280 / 2135 | 38 / 60 | 2-3% | 632 / 620 |
| 1024x1280 | 5120 | 20123 / 20424 | 18820 / 18944 | 52 / 92 | under 1% | 634 / 648 |
| 1536x1536 | 9216 | 64097 / 65807 | 62214 / 63727 | 75 / 183 | under 1% | 663 / 712 |

Half 2 holds: the tower is compute bound, and a prefetch of the block
reads would buy at most 3%. Block GPU time grows faster than the patch
count (27x for 7.2x the patches), so full-page attention is most of a large
page.

Per projection, clean host, every program planned on the ANE:

| projection | rows | gpu ms | ane ms (with handoff) | ane predict alone | ane/gpu |
| --- | ---: | ---: | ---: | ---: | ---: |
| qkv 3072x1024 | 1280 | 16.06 | 1.24 | 0.72 | 0.08 |
| proj 1024x1024 | 1280 | 5.59 | 0.88 | 0.39 | 0.16 |
| fc1 4096x1024 | 1280 | 21.25 | 1.41 | 0.89 | 0.07 |
| fc2 1024x4096 | 1280 | 13.23 | 1.62 | 1.15 | 0.12 |
| qkv | 4096 | 58.42 | 2.61 | 1.83 | 0.04 |
| fc2 | 4096 | 42.38 | 4.15 | 3.16 | 0.10 |

Half 1 holds by 6x to 25x. That ratio is mostly the GPU kernel, not the
ANE. `vision_matmul_fp16` gives each output element its own threadgroup and
reduces one dot product, which reaches about 0.5 TFLOP/s here (qkv at 1280
rows is 8 GFLOP in 16 ms). The ANE does the same projection at about 6.5
TFLOP/s with the handoff paid. A matrix-unit FP16 GEMM on the GPU
(`simdgroup_matrix`, as the INT4 MMA, Kokoro and Music 3 kernels use) should
close most of that gap with no handoff, no second weight copy in a Core ML
program, and no parity risk from a second engine. It did (below).

Against the old kernel (modeled from the two tables, not run): the four
projections are about 59% of block GPU time at 1280 patches, about 30% at
5120 and about 16% at 9216. Moving all four to the ANE would save about
48%, 25% and 15% of the page. Attention stays on the GPU either way and
is the larger target for big pages.

The gate as first run passed only because its GPU arm was a naive kernel.
So a matrix-unit FP16 GEMM came first (`vision_matmul_fp16_mma`, commit
`f1427d81` on its own branch, not yet merged here, kill switch
`TURBOSPARK_VISION_MMA=0`, tslab knob `vision_mma`). It runs the tower's
projections at 7 to 9.5 TFLOP/s against 0.5 to 0.8 for the old kernel.

Whole tower, interleaved GEMM, old, GEMM, old, median of 5 warm reps
(all four runs DIRTY from a long-running user Python job, repeats within
4.5%). Experiments `vision-gemm-mma` and `vision-gemm-per`:

| page | patches | GEMM ms | old ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| 512x640 | 1280 | 534 / 510 | 1766 / 1767 | 3.4x |
| 1024x1280 | 5120 | 9322 / 9304 | 14372 / 14461 | 1.55x |
| 1536x1536 | 9216 | 40990 / 40348 | 49420 / 49248 | 1.22x |

The `vision_*` gate rerun with the GEMM as the `gpu` arm (experiment
`vision-ane-gemm`, clean host) reads ane/gpu:

| projection | 1280 | 2048 | 4096 |
| --- | ---: | ---: | ---: |
| qkv | 0.81 | 0.96 | 1.00 |
| proj | 0.60 | not loaded (compute plan failed) | 1.10 (planned on CPU) |
| fc1 | 0.66 | 0.87 | 0.91 |
| fc2 | 1.09 | 1.20 (planned on CPU) | 1.09 |

Verdict: no ANE tower. Against a 9 TFLOP/s GEMM the ANE wins only at 1280
rows, where both arms sit near a 1 ms per-call floor, and ties or loses
above it. Under the GEMM the projections are about 19% of block time on a
512x640 page and under 2% on a 1536x1536 page (arithmetic from the GEMM's
rate, not a profile), so an ANE tower would save at most about 5% of a small
page. It would also add a second weight residency and Core ML's plan
failures. The tower's remaining cost is `vision_attention_bidir_fp16`, one
threadgroup per query and head, which grows with patches squared. The next
lever is a matrix-unit, flash-style bidirectional attention on the GPU.

Keep the negative: the ANE gate passed against the naive per-output kernel
(0.04 to 0.16) and does not pass against a matrix-unit GEMM.

That lever is measured. `vision_attention_bidir_fp16_mma` (branch commit
4eadb5ba on top of the GEMM, kill switch `TURBOSPARK_VISION_ATTN_MMA=0`,
knob `vision_attn_mma`) puts 32 query rows of one head per threadgroup,
stages K and V 16 keys at a time, and runs QK^T and PV on simdgroup_matrix
with FP32 accumulators and the online softmax in registers. Both arms keep
the GEMM on. Experiments `vision-attn-mma` and `vision-attn-scalar`, two runs
per arm interleaved, all four DIRTY (the same Python job, plus foreign CPU
over 150% on the second pair), totals in ms:

| page | patches | flash | scalar | speedup |
| --- | ---: | ---: | ---: | ---: |
| 512x640 | 1280 | 172 / 196 | 636 / 550 | 3.2x |
| 1024x1280 | 5120 | 963 / 1057 | 11371 / 11611 | 11.4x |
| 1536x1536 | 9216 | 2499 / 2738 | 47458 / 47081 | 18.1x |

The host was noisier than in the GEMM runs (the scalar arm here is about 15%
slower than the same configuration there), so quote the ratios. Peak
phys_footprint rises from 635 to 748 MiB to 835 to 889 MiB under the flash
kernel. That rise has not been attributed yet.

Quality:
- `vision_parity.rs` holds the kernel to the scalar one and the CPU reference
  at 1 to 1280 patches and head_dim 4 to 128, within the existing FP16 bound.
- On Qwen3-VL-4B, the mean per-row embedding cosine is 0.99992 or better on
  every page. That matches the noise the accepted GEMM change shows when
  measured the same way.
- The worst rows reach 0.977 on deepstack 1. They are the same rows under
  either change, and the attention change moves them about 3x further. An FP32
  P variant was worse on those rows, so the cause is not narrowing P to FP16.
- A 160-token greedy transcription is byte-identical across arms.
- One synthetic injected digest was re-frozen. The scalar arm still
  reproduces the old digest, and the argmax and the top-5 set are unchanged.

Under the flash kernel the projections and attention are the same order of
cost on a 1536x1536 page (about 1.1 s of the 2.5 s block time is projections
by arithmetic). On a 512x640 page, block reads are 18 to 23% of the block
loop.

The same pair on the Qwen3.8-27B tower (`qwen38-27b-vision`: 27 blocks,
hidden 1152, head_dim 72, a 4304-wide MLP and no deepstack). Experiments
`vision-q38-new` (two runs) and `vision-q38-old` (one run, both kill
switches set), all DIRTY from foreign CPU, totals in ms:

| page | patches | new kernels | old kernels | speedup |
| --- | ---: | ---: | ---: | ---: |
| 512x640 | 1280 | 259 / 294 | 3487 | 12x to 13x |
| 1024x1280 | 5120 | 1612 / 2142 | 35857 | 17x to 22x |
| 1536x1536 | 9216 | 4148 / 5614 | 93682 | 17x to 23x |

The two new-kernel runs differ by up to 35%, so read the ratios. The 785 MB
block stream takes about 50 ms per page at any size. That is 18 to 20% of a
1280-patch page under the new kernels and 1 to 3% at 5120 patches and up.
This tower is compute bound except on small pages, where a deeper prefetch
than the two read slots would be the next lever. Peak footprint is about
190 MiB higher under the new kernels (1026 against 836 MiB at 9216 patches),
the same rise as on Qwen3-VL-4B.

Also measured: placement is not what keeps EmbeddingGemma 2's vision tower
on the GPU. `tslab ane-plan --units ane --function NAME` plans 100% of the
op cost on the ANE for each of `vision_70`, `vision_140` and `vision_280`
(2389 of 4589 ops placed, the rest unplaced constants). Without
`--function`, a multifunction package plans its default function only and
the others read 0% everywhere, which looks like a refusal and is not one.

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
  is workload-specific, so calibrate before estimating: `tslab profile
  --dispatch-probe 0,4,8,16` measures it for a dense qwen or llama install.
  Qwen3.8's decode trunk reads 1.87 us per chained dispatch (see the traps
  list), against Whisper's 5.4 us. Its stand-ins (`--standin`) put the same
  token at 61% FFN, 24% gated-DeltaNet and 7% attention, which is where a
  decode lever has room.
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
  (`crates/gpu/examples/`) prints an FNV-1a hash of output bits over every
  configuration in its goldens (`crates/gpu/tests/golden/music3_*.txt`: 1,021
  linear, 192 DiT and 30 convolution lines today, covering bit widths, groups,
  row counts, dtypes, ragged tiles and split-K). Diff before and after. Any difference means
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

The dense llama-flow families (dense Llama, Qwen2, Qwen3, Qwen3-VL, Spark 2.5)
took the GQA-shared kernel as their default on the 2026-10-29 lab flip: the
family match in `DecodeScratch`
(`crates/runtime/src/real_forward_types.rs`), with
`TURBOSPARK_ATTN_DECODE_KERNEL=barrier` as the kill arm. What follows is the
procedure for a family that is not on that list.

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
  its prefix costs ~0.6 s. Done for DFlash2 on 2026-10-25: the chunked
  driver encodes its per-layer aux captures and context-writes each
  micro-batch (`TURBOSPARK_DFLASH_CHUNKED_PREFILL`), 130-158 tok/s against
  16-23 for the walk on a 426-token prompt (`docs/DFLASH2.md`).
- **Draft with less than the full head.** open-jet drafts against 32K
  frequency-ranked rows plus the sequence's own tokens. That would cut the
  two remaining head reads per round to about 15% of their cost. Taking the
  first N token ids does not work for this tokenizer, since 98K ids are
  needed to cover 99.9% of a code-and-docs corpus (`docs/MTP_SPECULATIVE.md`).

### Checklist for a new draft source

Written before the prompt-lookup experiment (`docs/KAPITSA_KERNELS.md`, "Prompt
lookup") so the next drafter is held to it from the start.

1. **Replay before wiring.** Take the greedy stream of the protocol cases and
   count, offline, what the draft source would have proposed and the target
   would have accepted. It needs no GPU and decides go or no-go. The MTP and
   DFlash2 work found per-position acceptance, not single-step acceptance, is
   the number that predicts speedup (`docs/MTP.md`).
2. **Price it against `c(M)`.** A round of `k` proposals pays one `k+1`-row
   verify. Break-even accepted-per-round comes from
   `docs/SPECULATIVE_DECODING.md`, not from intuition. Block 2 pays, 4 and 8
   do not, so a long draft needs a very high match quality.
3. **A round with nothing to propose must cost nothing.** A variable-length
   source (lookup, n-gram) skips the verify and runs one sequential step. A
   fixed-block source cannot, which is why DFlash2 reads 0.90x on prose.
4. **Say how it behaves sampled.** A model-free source has a point-mass
   proposal: accept when `r <= p(x)`, correct from `p` with `x` zeroed. The
   accept loop indexes `draft_qs` per proposal, so refuse sampled runs
   explicitly until that exists.
5. **Stats before speed.** `TURBOSPARK_SPEC_STATS` prints rounds and accept
   counts but no milliseconds. Add draft ms and verify ms per round, or a
   regression cannot be attributed to acceptance versus kernel time.
6. **Count tokens, not stream chunks.** A speculative round commits several
   tokens in one event. Decode tok/s is generated tokens over the time from
   the first to the last token, with TTFT reported separately.

### Checklist for a new or changed kernel

An alternative path that ships beside the default one follows
[KERNEL_MODES.md](KERNEL_MODES.md): one switch, an untouched default, a lane
row, a parity test, and a `tslab project` price before the model run.

1. Compute GB/s per kernel at the real shape first. A 27B INT4 GEMV is
   bandwidth-bound by construction, so only a profile showing dispatch or sync
   time licenses work there.
2. Count active lanes per output and dispatches per layer. Collapse a loop over
   rows into one dispatch only when the op is per head or per element.
3. State the exactness class up front: bit-exact (test with `to_bits()`), or
   KL-bounded (name the probe and the bound). Decode and verify must stay in
   the first class.
4. Force any f32 to f16 round trip that a later stage reads through a
   `volatile` half; fast-math elides it (`docs/KAPITSA_KERNELS.md`, items 4 and 5).
5. Put the kill switch behind an env var read once through a `OnceLock`, keep
   the old path one flag away, and mutation-check the new test.
6. Time it in the engine on AC power, interleaved, with the reference arm in
   the same script. A kernel win under 2% needs three or more paired rounds to
   mean anything on this host.
7. If the kernel can serve more than one row, plot its cost against rows,
   not just one point. What made speculation pay elsewhere is a verify
   whose 8 rows cost about 1.1x one row (`docs/KAPITSA_KERNELS.md`, "What
   the gap is made of"). A kernel that is fast at one row and linear in
   rows caps the drafter at block 2. Report the 1-row cost against the GEMV
   too, because decode pays it on every token if decode shares the kernel.
8. Say which pass the kernel serves (prefill, decode, verify, drafter) and
   measure that pass. A prefill kernel can win on a 3K prompt and do nothing
   for decode tok/s. A verify kernel shows in ms per round
   (`TURBOSPARK_SPEC_STATS=1`), not in plain decode.
9. Use the shared harness rather than a new script (`crates/lab`, `tslab`):
   - `tslab profile` for where the pass's time goes, bucket by bucket.
   - `tslab kernel` for the cost curve at the install's own shapes.
   - `tslab ab --null` once per workload, then `tslab ab`, for a verdict
     against this host's measured noise.
   - Speculative arms (`NAME~mtp:2`, `~dflash:7`, `~mtp:4a` for adaptive
     depth) to measure a verify or drafter change where it pays.

   - `tslab capabilities` first, and `tslab workflow new-family` for an
     install the lab has not measured.
   - `tslab trace` for launch-bound against kernel-bound, and
     `profile --dispatch-probe 0,4,8` for the price of a chained dispatch
     before estimating a fusion.
   - `kernel --hash` on both builds and `hash-diff` for a change that must
     not move output bits.

   tslab gives each arm a fresh process, so `OnceLock` switches are safe to
   A/B, and it records dirty runs rather than dropping them. Give a new
   switch an engagement line (`foundation::engaged!`) so an arm that did not
   take it is refused instead of reading as noise.
10. Measure on a quiet machine, and that includes your own session. A
    `cargo build` or test run, or an editor's background `cargo check` on
    your worktree, is enough to mark runs dirty (it did, on 2026-10-22).
    Finish compiling before you start timing, and do not edit Rust files
    in a watched worktree while a run is in flight.

## Next work

### 0. Prompt-lookup draft source

Replayed offline on the 27B (2026-10-21, `pld_replay_probe.rs`): flat or
losing on the three prose cases, 1.33-1.58x ceiling on a copy-heavy case,
which MTP block 2 already matches at 1.44x on all text. A standalone drafter is
declined. The open item is a hybrid that takes a long n-gram match (4+) over
the MTP head's two tokens, which needs an MTP replay on the copy-heavy stream
first. An older probe (`accept_length_probe.rs`, MoE install) agreed on prose.
Results and
gates are in `docs/KAPITSA_KERNELS.md`, "Prompt lookup". Start with the
offline replay probe, because novel prose (the protocol's three cases) will
likely have few matches, and the question is the copy-heavy case: a code edit
or a quote-back prompt.

### 0b. Small-M matrix-unit GEMM for verify and other 2-8 row paths

Bench prototype, 2026-10-21 (`docs/INT4_SMALL_M_MMA.md`): an 8-row INT4
GEMM on `simdgroup_matrix` with register operands, reading the resident
layout in place, costs 1.25-1.29x a 1-row GEMV at the qwen38 shapes. The
shipped batched GEMM costs 3.1-3.6x. That verify cost is what limits MTP to
block 2 and keeps tree verify from paying. The page holds the per-format
and per-family checklist.

The route that keeps speculation lossless is the
one item 1 names: run decode through the same tile. A row's bits do not
depend on the rows beside it, so this works, and the price is the
1.24-1.27x one-row cost until the kernel is tuned.

Wired opt-in as `TURBOSPARK_INT4_MMA8=1`, shape-routed (4096+ rows), and
gated lossless on the 27B. On a code prompt block 4 under the flag beats the
default by 16%. On prose the best arm is still flag-off block 2 (9-12%
ahead), because plain decode pays 19% for the tile. Choosing the block per
request (the adaptive depth controller) is what would take the code gain
without the prose loss.

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

### 7. What the mlx-serve comparison adds

[MLX_SERVE_KERNELS.md](MLX_SERVE_KERNELS.md) reads `ddalcu/mlx-serve` the
way this page read open-jet and kapitsa. Its M4 Max Qwen3.8-27B cell moved
from 70 to 106 tok/s when a DFlash2 draft tree of 8 rows started verifying
through row-exact matrix-unit kernels whose rows carry the one-row step's
bits at any width, gated on a drafter being bound. That is item 1 above and
`docs/INT4_SMALL_M_MMA.md` item 7, shipped by another engine, with the
gating and the per-shape one-row probe as the parts this tree lacks. The
page's engine levers (E1 to E6) and evaluator additions (L1 to L7: a
dispatch probe, a whole-forward verify-width ladder, engagement lines per
arm, stand-in ablation, a Metal System Trace command, rows per simdgroup as
a bench axis, a live round-cost table) are listed there with their gates. Its dead ends are recorded
so they are not re-derived here.
