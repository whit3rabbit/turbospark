# Flint: status and what is left

Recursive Flint (`RecursiveSI/flint-int4`, MLX affine INT4 group 64) runs
here: 49 layers, 256 routed experts top-6 streamed through the slot cache,
an ungated shared expert, a sigmoid router, 14 NoPE full-attention layers and
35 sliding ones (window 513) with learned sinks. This page is the handoff.
Facts, decisions, and measurements live in
[FLINT_PHASE0.md](FLINT_PHASE0.md). Read that first. Nothing here is a
benchmark: every number below is from one machine (M4 Max, 36 GB) that was not
always quiet, and none is a frozen row.

## What works (verified on the real checkpoint)

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
  (`crates/runtime/src/families/flint/prefill.rs`), default ON since the
  lab's clean paired win; `TURBOSPARK_FLINT_CHUNKED_PREFILL=0` restores the
  sequential walk. It is byte-identical to the sequential path (a test file
  per arm, mutation-checked, and the lab's bit-exact quality gate on the real
  model). The clean verdict: +38.2% prefill tok/s, 95% CI [+33.6%, +41.8%],
  bit-exact, on the `flint` install (`crates/lab/families/flint.json`,
  finding `flint.chunked-prefill-clean-2026-10-28`).

## What is left, in order

1. **M-row INT4 GEMMs for the projections and shared expert**
   (`TURBOSPARK_BATCHED_GEMV`, Steel). The sequential path's GPU is busy about
   23 ms per token, far above a bandwidth bound, so it looks dispatch-bound. The
   precedent (Qwen MoE) is 1.4 to 1.6x on top of the Step 1 driver. It is NOT
   bit-exact on a MoE (class `moe_kl`: it can swap near-tie experts), it needs
   a new KL probe, and Flint's FP16 margins make any rounding or reduction-order
   change a real risk. The driver currently refuses this switch by name.
2. **Expert-major (grouped) MoE prefill.** About a quarter of sequential
   prefill is expert `pread` at an 82.6% cache hit rate, and a token re-reads
   its experts. Processing a whole chunk expert by expert reads each once
   (`docs/KAPITSA_KERNELS.md` item 8 measured the GEMM side at 0.44 to 0.80 of
   the GEMV cost). It is the one open MoE-prefill idea, and a project: new
   route-permutation and combine kernels, a drift bound, and FP16 overflow in
   phase 1 to re-check.
3. **The remaining gates.** No memory oracle, no quality gate, and no frozen
   benchmark row exist, so the catalog status cannot be `verified`. The bench
   protocol row for flint is provisional (8,192 context, 4,096 new tokens,
   because it reasons). Add `flint_memory_oracle.rs` and a quality gate next to
   the existing ones, one model per process.
4. **Quality unknowns.**
   - The INT8 router agrees with the BF16 original on the top-6 set 97.27% of
     the time (every difference a near-tie), measured with
     `TURBOSPARK_FLINT_ROUTER_DUMP`. Whether that moves quality is unmeasured.
     Removing it needs a BF16 router GEMV and a non-8-bit router slot, which
     `validate_quant` refuses today.
   - The 2^6 rebalance was exercised on about a dozen prompts. A token whose
     product exceeds about 4 million would still overflow. Run new workloads with
     `TURBOSPARK_FLINT_DEBUG=1` and watch for reports near 65,504.
   - Nothing compares against a reference implementation (SGLang or llama.cpp
     PR 30928), and no test shows RoPE is correctly absent on the
     full-attention layers.
5. **Features not wired.**
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
6. **Docs and surfaces never written.** Zero Flint mentions in `DEVIATIONS.md`,
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
| `TURBOSPARK_FLINT_CHUNKED_PREFILL=0` | restore the sequential token walk (the layer-batched prefill driver is the default) |
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
- Install: `~/.turbospark/models/text/flint.gturbo` (already migrated).
