# Flint: status and what is left

Recursive Flint runs here: 49 layers, 256 routed experts top-6 streamed
through the slot cache, an ungated shared expert, a sigmoid router, 14 NoPE
full-attention layers and 35 sliding ones (window 513) with learned sinks.
This page is the handoff.
Facts, decisions, and measurements live in
[FLINT_PHASE0.md](FLINT_PHASE0.md). Read that first. Nothing here is a
benchmark: every number below is from one machine (M4 Max, 36 GB) that was not
always quiet, and none is a frozen row.

## Which build

The default Flint target, and the one to test and build against, is the
community build `mlx-community/flint-mixed-3-4bit` at revision
`b1c305855ae7c3464813815be4dbd544b22c36aa` (catalog alias `flint`, 17.8 GB,
fits 24 to 32 GB Macs). It is Recursive's trained INT4 with only the routed
experts requantized to MLX 3-bit (group 64, BF16 scales and biases); the
embedding stays INT8, everything else keeps the trained INT4 codes, and the
router is BF16 at source. Recursive's own `RecursiveSI/flint-int4@3b93f905`
(alias `flint-int4`, 22.6 GB) still installs and runs on the same flow.

What the 3-bit experts needed, all in place:

- a routed INT3 decode pair, `moe_phase1_gate_up_act_int3` and
  `moe_phase2_down_reduce_int3` in `crates/gpu/src/shaders/moe.metal`. MLX
  packs 3-bit codes as a little-endian bitstream (element `j` at bits
  `[3j, 3j+3)`, 24 bytes per group); the kernels read bytes and repeat the
  INT4 pair's accumulation term by term;
- `RoutedBlobLayout::AffineInt3`, resolved per phase from the blob's
  weight-to-scale byte ratio (`4 * bits` at group 64 BF16). Any other ratio is
  refused at open; before this a 2-bit affine blob was read as INT4;
- intake: `parse_flint_quantization` renames MLX per-tensor quant keys onto
  the canonical namespace and admits 3-bit on `switch_mlp` tensors only, the
  expert writer cross-checks packed words against the architecture, the probe
  parses Flint through the same reader, and `validate_quant` admits 3 on
  `routedExpert` only. The 2^6 rebalance applies unchanged (it moves scales
  and biases, not codes).

Tests: `crates/gpu/tests/moe_decode_int3.rs` (CPU dequant reference at
Flint's shape and on the tail path, 7 shader mutations killed) and
`three_bit_routed_experts_decode_bit_identically_to_the_same_codes_at_four_bits`
in `crates/runtime/tests/real_forward_flint.rs` (a synthetic install whose
3-bit experts must decode bit for bit like the same codes at 4 bits; intake,
width detection and dispatch are on its path).

## The mixed build against mlx-vlm (2026-10-25, M4 Max 36 GB)

Same bytes on both sides: this port's install of the pinned revision against
mlx-vlm 0.7.10, which loads Flint through `mlx_vlm/models/flint` (FP32 router
scores, FP32 routed sum, FP32 `lm_head`). Teacher-forced, ids not text
(`scripts/kld_flint_mlx_vlm.py`, procedure in
`.claude/docs/cross-engine-kl.md`). The floor is mlx-vlm against itself,
one token at a time against one batched pass.

| Sequence | Rows | Mean KL (nats) | p99 | Top-1 | mlx-vlm floor mean / top-1 |
|---|---|---|---|---|---|
| quality corpus | 529 | 0.0070 | 0.067 | 97.0% | 0.0066 / 96.8% |
| mlx-vlm's greedy path, "17 times 23" | 69 | 0.0055 | 0.105 | 95.7% | 0.0067 / 98.6% |
| mlx-vlm's greedy path, coding prompt | 704 | 0.0011 | 0.023 | 98.7% | 0.0020 / 99.3% |

So the port sits at mlx-vlm's own shape floor, which on an MoE is near-tie
routing and reduction order. Known differences that remain on purpose: the
INT8 router (mlx-vlm keeps BF16), and FP16 activations with the rebalance.
Assistant-slot perplexity on the corpus: 12.82 here, 12.89 mlx-vlm cached,
12.69 batched.

Greedy through the chat template (`turbospark-check --messages-file
--temperature 0`, which is what `turbospark run` drives):

- "What is 17 times 23? Answer briefly.": a short `<think>` block, then
  `391`, `stop=Eos`. Text identical to mlx-vlm's greedy output (57 tokens
  each), and identical here at 8, 16 and 96 expert-cache slots.
- A coding prompt (merge overlapping intervals, docstring, empty list, one
  example): reasoning, then a correct function and the example's output. The
  two engines' greedy paths part about 90 tokens into the reasoning on a
  near-synonym ("likely" against "presumably"); the final code is the same.

Against Recursive's INT4 on the same corpus (both through this port): mean KL
0.213 nats, top-1 82.8%, assistant-slot perplexity 12.82 against 11.69. The
publisher reports 0.343 and 86.2% on its own corpus. Both arms run through
this port, so the gap is the requantization.

Indicative only (one prompt, a download running alongside, no AC check):
peak physical footprint 0.90 GiB at 8 slots and 1.40 GiB at 16, against
0.98 and 1.61 for the INT4 install.

## GQA-shared decode attention (2026-10-25)

Flint's attention scratch takes `DecodeAttentionKernel::GqaShared`
(`crates/runtime/src/real_forward_types.rs`): one threadgroup per (KV head,
chunk) serves all 8 Q heads of a group from one K/V read. The tslab
`attn_decode` bench at this shape, GPU time per token: the 14 full layers
1.49 ms barrier against 0.40 GQA at 512 positions and 91.3 against 3.6 at 32K,
and the 35 sliding layers about 2.6 against 0.67 at every context. It is not
bit-exact, so it was held to the mlx-vlm KL on the mixed build, both arms in
fresh processes (`TURBOSPARK_ATTN_DECODE_KERNEL=barrier|gqa`):

| Arm | Mean KL vs mlx-vlm (nats) | p99 | Top-1 | Perplexity |
|---|---|---|---|---|
| barrier (reproduces the table above) | 0.0070 | 0.067 | 97.0% | 12.818 |
| GQA | 0.0072 | 0.086 | 96.6% | 12.825 |
| mlx-vlm against itself | 0.0066 | 0.056 | 96.8% | |

Barrier against GQA directly: mean KL 0.0022 nats (a third of mlx-vlm's own
floor), top-1 97.9%. Greedy through the template parts on near-synonyms
("multiplication" against "arithmetic") and lands on the same answers: `391`,
a correct `merge_intervals`, and the right fact from 1,650 tokens back across
the wrapped 513-position ring. Peak footprint 6.33 GiB on both arms (the
scratch is the same size). `crates/gpu/tests/attention_decode_gqa_parity.rs`
pins this shape with sinks, the window and a wrapped 641-row ring.

**Still open: the paired performance A/B.** Every `tslab ab` attempt
(`flint-gqa-decode`, `flint-gqa-decode-2`) was INCONCLUSIVE: other sessions'
builds and real-model tests kept the host busy. Clean single-arm readings
only, so indicative and not a verdict: the GQA default read 40.3 to 41.0
decode tok/s over four clean null-calibration runs (`flint-gqa-decode-2`),
against 30.3 for barrier over two clean runs in `flint-mixed-3-4bit-profile`,
both on `protocol:long-synthesis` with 64 decode tokens. Rerun
`tslab ab --baseline barrier:TURBOSPARK_ATTN_DECODE_KERNEL=barrier
--candidate gqa:TURBOSPARK_ATTN_DECODE_KERNEL=gqa --metric decode_tok_s` on a
quiet host before quoting a speedup.

## What works (verified on the real checkpoint)

The rest of this section was measured on `flint-int4`; the flow is shared.

- `turbospark-model probe` says RUNNABLE, `pull` installs, and the catalog row
  `flint` (status `runs`) matches the published byte total.
- Greedy and sampled generation are coherent: English, code, Spanish,
  Japanese, math, a system turn, and a multi-turn chat. A 1,100-token greedy run
  stays coherent past the 513-token window with the ring wrapped. A fact 1,650
  tokens back is retrieved correctly (Dmitri Nolan, a chapel on Cedar Street).
- Output is byte-identical at 8, 16, and 96 expert-cache slots and across fresh
  processes.
- Peak physical footprint was 0.98 GiB at 8 slots and 1.61 GiB at 16 for a
  21 GiB install. Decode ran at about 19 to 28 tok/s (indicative).
- Every install carries a fixed 2^6 up/down rebalance that keeps the SwiGLU
  product inside FP16 (`crates/repack/src/flint_rebalance.rs`). Without it a
  plain English prompt overflows to NaN. An install made before it existed
  migrates in place with
  `cargo run --release -p turbospark-repack --example flint_rebalance -- <dir>`.
  `flint_rebalance.json` records the net shift and a second run is refused.
- `ChatDialect::Flint` (probed before GLM, whose markers the table shares) and
  shard discovery for a repo that ships no `model.safetensors.index.json`.
- A layer-batched chunked prefill driver
  (`crates/runtime/src/families/flint/prefill.rs`) behind
  `TURBOSPARK_FLINT_CHUNKED_PREFILL=1`, default OFF. It is byte-identical to the
  sequential path (a new test file, mutation-checked, and the lab's bit-exact
  quality gate on the real model). Its speedup is NOT established: see item 1.

## What is left, in order

1. **A clean timing for the chunked prefill.** The lab's run returned
   Inconclusive with 0 clean pairs because every round was dirty (macOS
   `mediaanalysisd`, `ANECompilerService` and `corespotlightd` above 90% CPU,
   compiles, other sessions' probes). The three least dirty rounds read about
   42 to 44 tok/s sequential against 55 to 56 batched. That is not quotable.
   Re-run `flint-chunked-prefill` on a quiet machine (spec and notes are in
   `~/.turbospark/lab/flint-chunked-prefill/`). The lab itself is `crates/lab` (`tslab`). Making the switch the default is a
   human decision after a clean WIN.
2. **M-row INT4 GEMMs for the projections and shared expert**
   (`TURBOSPARK_BATCHED_GEMV`, Steel). The sequential path's GPU is busy about
   23 ms per token, far above a bandwidth bound, so it looks dispatch-bound. The
   precedent (Qwen MoE) is 1.4 to 1.6x on top of the Step 1 driver. It is NOT
   bit-exact on a MoE (class `moe_kl`: it can swap near-tie experts), it needs
   a new KL probe, and Flint's FP16 margins make any rounding or reduction-order
   change a real risk. The driver currently refuses this switch by name.
3. **Expert-major (grouped) MoE prefill.** About a quarter of sequential
   prefill is expert `pread` at an 82.6% cache hit rate, and a token re-reads
   its experts. Processing a whole chunk expert by expert reads each once
   (`docs/KAPITSA_KERNELS.md` item 8 measured the GEMM side at 0.44 to 0.80 of
   the GEMV cost). It is the one open MoE-prefill idea, and a project: new
   route-permutation and combine kernels, a drift bound, and FP16 overflow in
   phase 1 to re-check.
4. **The remaining gates.** No memory oracle, no quality gate, and no frozen
   benchmark row exist, so the catalog status cannot be `verified`. The bench
   protocol row for flint is provisional (8,192 context, 4,096 new tokens,
   because it reasons). Add `flint_memory_oracle.rs` and a quality gate next to
   the existing ones, one model per process.
5. **Quality unknowns.** (The INT8 router and FP16 range items below apply to
   both builds; the reference comparison now exists for the mixed build, see
   above.)
   - The INT8 router agrees with the BF16 original on the top-6 set 97.27% of
     the time (every difference a near-tie), measured with
     `TURBOSPARK_FLINT_ROUTER_DUMP`. Whether that moves quality is unmeasured.
     Removing it needs a BF16 router GEMV and a non-8-bit router slot, which
     `validate_quant` refuses today.
   - The 2^6 rebalance was exercised on about a dozen prompts. A token whose
     product exceeds about 4 million would still overflow. Run new workloads with
     `TURBOSPARK_FLINT_DEBUG=1` and watch for reports near 65,504.
   - The mixed build agrees with mlx-vlm to mlx-vlm's own floor (above), and
     mlx-vlm skips RoPE on the NoPE layers, so a RoPE mistake there would land
     far above it. Still no unit test pins it. The INT4 build has no
     reference comparison of its own.
6. **Features not wired.**
   - Reasoning splits by token id: `<think>` (150006) and `</think>` (150007)
     are added but not special, and the decoder keys on ids, as GLM's does.
     The template has no thinking-off switch and does not prefill the open,
     so the split runs at every reasoning effort.
   - Tool calls are GLM-style `<tool_call>name<arg_key>..` markup with no
     parser (dialect `Prompted`).
   - Steering and speculation: the checkpoint ships no MTP head, and measured
     speculation on streamed MoE is about 1.08x at best.
   - Mapped expert residency: the 19.8 GiB expert table is unlikely to fit the
     GPU working set on a 36 GB machine.
   - GGUF intake: Q4_1 is not an executable block type here.
7. **Docs and surfaces never written.** Zero Flint mentions in `DEVIATIONS.md`,
   `AGENTS.md` (a gotcha for the sigmoid router, the dense-lead blob numbering
   and the FP16 range), `docs/MODELS.md` and the README model table. The Swift
   app's `knownMoEFamilies` set and `ModelFamilyVisuals` also lack `flint`, so it
   shows as a generic family. Prefill itself is documented only here and in
   `FLINT_PHASE0.md`.

## Things that will bite the next person

- `RecursiveSI/flint-int4` ships `U8`-packed weights, re-labelled `U32` at
  repack. Its router is raw BF16, force-quantized to INT8. Its embedding is INT8
  under an INT4 default declared only in `mlx_export`.
- The HF writer numbers expert files by POSITION (`layer - 2`), not by
  checkpoint layer, because layers 0 and 1 are dense.
- `turbospark-check` and `turbospark-model` are separate binaries: rebuild the
  one you run, or a stale one reports `unknown arch.family "flint"`. The
  `turbospark` dispatcher execs a sibling `turbospark-check`, so building only
  `--bin turbospark` runs whatever older one is on `PATH`.
- Test through the template: `turbospark run <install> --messages-file m.json`
  (a JSON array of `{"role", "content"}`) or `--chat`. A bare prompt encodes
  verbatim with no `<|user|>`/`<|assistant|>`, and Flint then continues the
  user's text like a base model. Greedy through the template, "What is 17
  times 23? Answer briefly." gives a short thought on stderr and `391` on
  stdout, the same answer mlx-vlm reports.
- A synthetic install is built in the checkpoint's source form
  (`build_synthetic_flint_install`) so tests exercise the rename, the U8 relabel
  and the router force-quantize. Build tests with `--release`: the debug profile
  is about 1,000 times slower on these tests.
- Disk: the 21 GiB install is the least of it. Release builds are about 5 GB
  per worktree, and the main checkout's shared `target/` was 196 GB when this
  was written.

## Switches and diagnostics

| Variable | What it does |
|---|---|
| `TURBOSPARK_FLINT_CHUNKED_PREFILL=1` | layer-batched prefill driver (default off) |
| `TURBOSPARK_FLINT_DEBUG=1` | per-MoE-layer report of non-finite or large residual, router, activation and shared values |
| `TURBOSPARK_FLINT_ROUTER_DUMP=<file>` | appends the post-norm MoE input and the INT8 router logits per layer per token |
| `TURBOSPARK_PHASES=1` | the phase-bucket table (cb1 wait, expert io, encode) |
| `TURBOSPARK_RESID_CAPTURE=<file>` | residual stream per layer at the last prompt token (written only on success). Refused for Flint today: the capture is gated on `steering::family_dispatches_steering`, which says `false` for Flint although `families/flint/mod.rs` encodes both hooks |

## Where the work is

- `main` has everything above, including the lab. The experiment branch is gone
  (its driver commit is on `main`), so re-create the tree for item 1:

  ```sh
  git worktree add .claude/worktrees/lab-flint -b lab/flint main
  cd .claude/worktrees/lab-flint
  cargo build --release -p turbospark-lab
  LAB=$HOME/.turbospark/lab; REV=$(git rev-parse --short HEAD)
  mkdir -p $LAB/bin/$REV && cp target/release/tslab target/release/tslab-worker $LAB/bin/$REV/
  $LAB/bin/$REV/tslab run-experiment --base $REV \
      --spec $HOME/.turbospark/lab/flint-chunked-prefill/spec.copy.json
  ```

  `--base` must be the merge commit, or the integrity check lists the whole lab
  as touched evaluator files. The spec's `messages_file` points at a scratch
  directory that may be gone: regenerate it as one user message holding 45 lines
  `Fact i: the village of Brookmere number i has N houses, a <mill|bakery|school|chapel|market> on <Oak|Elm|Pine|Cedar|Birch> Street, and its mayor in year 1800+i was named <name> <surname>.`
  followed by a question about village 3 in 1803 (about 1,670 tokens), and edit
  the path. Check `ps` for other sessions' model processes and system indexing
  daemons first: they are what made the first run dirty.
- Installs on the bring-up machine: the INT4 build at
  `~/.turbospark/models/text/flint.gturbo` (already migrated), the mixed build
  at `~/models/flint-mixed-3-4bit.gturbo` (pulled through the `flint` row with
  `--out`). From a local `hf download`, the same walk the pull runs, writing a
  byte-identical install (checked against that pull), is
  `cargo run --release -p turbospark-repack --example flint_local_intake --
  <checkpoint dir> <new install dir> [model id]`.
