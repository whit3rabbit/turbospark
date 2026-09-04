# turbospark-bench

Throughput benchmark harness (`turbospark-bench`), mach memory sampler (`memory.rs`), frozen community benchmark protocol (`protocol.rs`), real model benchmark runner (`real_model.rs`), and memory oracle integration test (`tests/memory_oracle.rs`).

## Directory & File Structure

```
crates/bench/
+-- Cargo.toml              # Crate manifest
+-- src/
|   +-- lib.rs              # Library entry point (turbospark_bench)
|   +-- main.rs             # Binary entry point (turbospark-bench)
|   +-- model_mode.rs       # Real install benchmark mode runner
|   +-- scripted.rs         # Scripted and synthetic real benchmark mode runners
|   +-- memory.rs           # Mach memory sampler for physical footprint tracking
|   +-- protocol.rs         # Frozen community benchmark protocol definitions
|   +-- real_model.rs       # Real model benchmark runner driving RealForwardRunner
|   +-- real_model_open.rs  # Model runner open helpers and residency resolution
|   \-- real_model_params.rs# Protocol parameters resolution per model family
+-- tests/
|   +-- accept_length_probe.rs # Speculative accept length probe
|   +-- batched_forward_probe.rs # Batched forward numerical equivalence probe
|   +-- dflash2_accept_length_probe.rs # The BLOCK drafter's sweep, plus the losslessness floor
|   +-- dflash2_bisect_probe.rs # Localizes a broken drafter; one runner, one variable
|   +-- gguf_nondeterminism_probe.rs # Repeated warm greedy runs must produce ONE output
|   +-- gptoss_memory_oracle.rs # Memory oracle for gpt-oss-20b, at 8192 context AND a 3072 budget
|   +-- gptoss_quality_gate.rs  # Quality gate for gpt-oss-20b; pins the template's date
|   +-- iq3_quality_gate.rs     # Quality gate for IQ3 quantized models
|   +-- logit_dump.rs       # Full-vocab logit dump for the cross-engine KLD (scripts/kld{,_llamacpp,_mlx_affine}.py)
|   +-- mapped_expert_probe.rs # What phys_footprint charges for an mmap Metal wrapped and the GPU read
|   +-- memory_oracle.rs    # Memory oracle asserting peak footprint ceiling & steady state (Gemma 4)
|   +-- mference_bench.rs   # Benchmark harness integration smoke test
|   +-- mistral_memory_oracle.rs # Memory oracle for the DENSE llama family, at 8192 context
|   +-- mtp_accept_length_probe.rs # MTP head as drafter: accepted vs the block table's break-even
|   +-- mtp_generation_gate.rs  # MTP speculative generation gate
|   +-- mtp_head_probe.rs    # What the head predicts, when the probe above reads zero
|   +-- museglimmer_memory_oracle.rs # Memory oracle for Muse Glimmer family
|   +-- museglimmer_quality_gate.rs  # Quality gate for Muse Glimmer family
|   +-- oracle_common/      # Shared memory oracle assertion helpers (mod.rs)
|   +-- ornith35b_memory_oracle.rs # Memory oracle for Ornith 35B MoE family
|   +-- ornith35b_quality_gate.rs  # Quality gate for Ornith 35B MoE family
|   +-- ornith9b_memory_oracle.rs  # Memory oracle for Ornith 9B dense family
|   +-- ornith9b_quality_gate.rs   # Quality gate for Ornith 9B dense family
|   +-- quality_common/     # Shared quality gate evaluation helpers (mod.rs)
|   +-- quality_gate.rs     # Quality gate integration test (Gemma 4)
|   +-- quality_sensitivity.rs # Proof the gate sees quantization damage (Gemma 4)
|   +-- qwen36_memory_oracle.rs # Memory oracle for Qwen 3.6 family
|   +-- qwen36_quality_gate.rs  # Quality gate for Qwen 3.6 family
|   +-- qwen38_memory_oracle.rs # Memory oracle for Qwen3.8-27B (`qwen35`), the family's FIRST
|   +-- qwen38_quality_gate.rs  # Quality gate for Qwen3.8-27B; no assistant prefix, and see its header for why
|   +-- qwen3moe_memory_oracle.rs # Memory oracle for Qwen3-30B-A3B (`qwen3moe`)
|   +-- qwen3moe_quality_gate.rs  # Quality gate for Qwen3-30B-A3B
|   +-- rollback_probe.rs   # RollbackPoint state restoration probe
|   +-- steering_probe.rs   # Live steering: inert at alpha 0, and a real edit above the floor
|   +-- steering_sweep.rs   # Which alpha is usable: steered output scored under the UNSTEERED model
|   +-- ternary_memory_oracle.rs # Its oracle; the peak is Qwen3.8's on HALF the weights (Gotcha 40)
|   +-- ternary_quality_gate.rs # Quality gate for Ternary-Bonsai-27B (2-bit), the same family's third checkpoint
|   +-- vision_logit_dump.rs # Vision model cross-engine logit dump
|   \-- vision_memory_oracle.rs # Memory oracle for vision models
\-- prompts/
    +-- quality-v1/         # Quality gate reference prompt fixtures
    |   \-- assistant-reference.txt
    \-- real-generation-v1/ # Standardized benchmark protocol prompt fixtures
        +-- short-explanation.txt
        +-- medium-review.txt
        \-- long-synthesis.txt
```

## Key Modules

- `main.rs`: Handles CLI benchmark modes (scripted producer overhead vs real model protocol).
- `memory.rs`: Mach kernel task info sampler for tracking peak physical memory footprint (`phys_footprint`).
- `protocol.rs`: Frozen benchmark protocol case definitions and step evaluators.
- `real_model.rs`: Runs protocol cases against real `.gturbo` installs using `RealForwardRunner`. Also owns `protocol_parameters`, the per-family context window and generation budget (Gotcha 16), which `open_model_runner_for_protocol` resolves from the install's manifest and hands back beside the runner.
- `tests/memory_oracle.rs`: Ignored test gated on `TURBOSPARK_GEMMA4_INSTALL_DIR` asserting per-chip memory ceilings and zero leak growth for Gemma 4.
- `tests/qwen36_memory_oracle.rs`: Ignored test gated on `TURBOSPARK_QWEN36_INSTALL_DIR` asserting per-chip memory ceilings and steady state for Qwen 3.6.
- `tests/quality_gate.rs` & `tests/qwen36_quality_gate.rs`: Quality gate evaluation verifying generated output metrics against reference fixtures.

## Development & Test Commands

```sh
# Run fast unit and integration tests for turbospark-bench
cargo test -p turbospark-bench

# Run scripted producer throughput benchmark
cargo run -p turbospark-bench --bin turbospark-bench -- <tokenizer-dir>

# Run real install benchmark (macOS, release mode required). The context
# window and the generation budget come from the install's own family and
# are printed in the header (Gotcha 16): 4,096/1,024 for gemma4, qwen36 and
# qwen3moe, 8,192/1,024 for the dense `llama` half, 8,192/3,072 for gpt-oss.
cargo run --release -p turbospark-bench --bin turbospark-bench -- --model ~/models/gemma4.gturbo

# One protocol case per process (the protocol's fresh-process leg, and what
# a cross-engine comparison needs since Swift's CLI launches once per case)
cargo run --release -p turbospark-bench --bin turbospark-bench -- \
  --model ~/models/gemma4.gturbo --case short-explanation

# Vary the routed-expert cache size (allowed 8/16/24/32; THIS BINARY
# defaults to 16, the same set and default MferenceCLI takes, while
# turbospark-check and turbospark-server default to `auto` -- see Gotcha 5).
# 32 buys ~15% decode and ~1.5 GB
# of footprint, so it leaves the ~2 GB working-set claim behind; every
# published number is measured at the default.
cargo run --release -p turbospark-bench --bin turbospark-bench -- \
  --model ~/models/gemma4.gturbo --case short-explanation --expert-cache-slots 32

# Head-to-head against ../Mference's MferenceCLI, same install, interleaved
# arms, one fresh process per run. Results: docs/BENCHMARKS.md
scripts/parity.sh

# Bucket-level decode phase diff against the Swift engine (both print a
# split under TURBOSPARK_PHASES=1, but Swift's is decode-only and this
# port's divides by all forward passes -- hence the short-prompt rule
# baked into the script). Not a benchmark; an attribution aid.
scripts/phasediff.sh [pairs] [slots]

# Run memory oracle test for Gemma 4 (macOS, takes ~10 mins, requires model env var)
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test memory_oracle --release -- --ignored --nocapture

# Run memory oracle test for Qwen 3.6 (macOS, separate process target)
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_memory_oracle --release -- --ignored --nocapture

# The DENSE family's oracle (ROADMAP M4), at 8,192 context rather than the
# shared 4,096. See crate gotcha 11 before comparing its ceiling to any other.
TURBOSPARK_MISTRAL_INSTALL_DIR=~/models/mistral7b-dense.gturbo \
  cargo test -p turbospark-bench --test mistral_memory_oracle --release -- --ignored --nocapture

# Run memory oracle for Qwen3-30B-A3B. Its ceiling is 2,900 MiB rather than
# the other families' 1,700-2,300, and that is arithmetic: 48 layers times a
# ~2.9 MiB expert is 2,094 MiB of slot capacity at 16 slots.
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
  cargo test -p turbospark-bench --test qwen3moe_memory_oracle --release -- --ignored --nocapture

# Quality gate: reference-answer perplexity + frozen output digests
# (ROADMAP Phase Q). One target per family, ~1 min each.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_gate --release -- --ignored --nocapture
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_quality_gate --release -- --ignored --nocapture
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
  cargo test -p turbospark-bench --test qwen3moe_quality_gate --release -- --ignored --nocapture

# Cross-engine KLD against mlx-lm (Phase Q's last item). Step 1 dumps this
# port's full-vocab logits plus the token ids; step 2 replays those IDS
# through mlx-lm in a uv ephemeral env. Needs the 14.6 GB reference
# checkpoint. TURBOSPARK_LOGIT_DUMP_COLD=1 skips the warmup walk and
# reproduces quality_gate's frozen perplexity exactly.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/turbospark \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with mlx-lm --with numpy scripts/kld.py /tmp/kld/turbospark gemma4

# The same for the 1-BIT family (ROADMAP's 1-bit entry, step 5). Its driver
# is separate because upstream mlx REFUSES bits=1 at the API level on every
# device, so the reference is github.com/PrismML-Eng/mlx@prism built from
# source into a venv (~3 min) rather than a `uv run --with mlx-lm` env.
TURBOSPARK_QWEN35_INSTALL_DIR=~/models/bonsai27b.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/bonsai-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
/tmp/prism-venv/bin/python scripts/kld_mlx_affine.py /tmp/kld/bonsai-warm bonsai-1bit

# The same for the TERNARY checkpoint of that architecture (ROADMAP's ternary
# entry). One driver, and the checkpoint NAME is required rather than
# defaulted: the two published checkpoints have the same module count and the
# same shapes, so pairing a dump with the wrong reference passes every check
# and reads as a kernel bug. Upstream mlx supports bits=2, so this arm needs
# no fork.
TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/ternary-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with 'mlx-lm==0.31.2' --with numpy \
  scripts/kld_mlx_affine.py /tmp/kld/ternary-warm ternary-2bit

# The DENSE `qwen35` half (Ornith-1.5-9B) against llama.cpp on the identical
# Q8_0 bytes, and the family's FIRST external check -- its four frozen gate
# rows are all self-referential, and `ornith_tensor_probe.rs` beside them is
# a STATIC per-tensor check that cannot see how the runtime USES what it
# validates. Needs no driver change at all: `kld_llamacpp.py` reads the
# softcap off the dump's own install (Gotcha 38), and this family's is 0.0,
# so both maxima are reported rather than asserted. Reads 0.000138 mean nats
# at 99.65% top-1, FIFTEEN TIMES BELOW its 0.00202 backend floor. See Gotcha
# 8 before reading its 58x SHAPE-floor ratio as a defect.
TURBOSPARK_ORNITH9B_INSTALL_DIR=~/models/ornith9b.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/ornith9b-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/Ornith-1.5-9B-Q8_0.gguf /tmp/kld/ornith9b-warm

# The same family's MoE half against MLX, which is the OTHER intake format --
# this install is streamed from an affine INT4 conversion where the 9B's is
# streamed from a GGUF. Upstream mlx-lm carries `qwen3_5_moe`, so no fork.
# Reads 0.02739 at 91.10% top-1 against a 0.02780 shape floor, i.e. below it.
# The pair is the cleanest demonstration in the repo of what a shape floor is
# MADE of: same family, same day, dense 0.0000024 against MoE 0.02780.
TURBOSPARK_ORNITH35B_INSTALL_DIR=~/models/ornith35b.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/ornith35b-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with mlx-lm --with numpy \
  scripts/kld_mlx_affine.py /tmp/kld/ornith35b-warm ornith-35b-4bit

# The MTP head as a drafter (docs/MTP.md for the facts, MTP_SPECULATIVE.md
# for the decision). It runs BOTH verify strategies per block and reports
# WALL CLOCK against a non-speculative reference, which is the number step 4
# exists to produce; the `projected` column beside it is step 3's composed
# break-even and the two disagree on purpose. Measured: block 2 batched pays
# 1.44x against a 1.66x projection, and batching LOSES to a sequential verify
# at blocks 8 and 15 because a rejected batched round restores the whole
# gated-DeltaNet snapshot and replays -- a term no composite here models.
# The two arms' round and accept counts come out identical at every block,
# which is the equivalence check the losslessness gate cannot make on its
# own. The probe ASSERTS a
# functional drafter and FAILS rather than printing an accept length nobody
# can read -- its first run read 0 of 7,168 (a centered-norm bug, since
# fixed) and would otherwise have published a quotable "loses" table.
# mtp_head_probe is the paired instrument: it reports the rank of the true
# token in the head's own distribution, which separates broken from weak.
TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
  cargo test -p turbospark-bench --test mtp_accept_length_probe --release -- --ignored --nocapture
TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
  cargo test -p turbospark-bench --test mtp_head_probe --release -- --ignored --nocapture

# The BLOCK drafter's equivalent (docs/DFLASH2.md). ~12 min: THREE workloads
# (code, math, prose), each swept across all four blocks. It sweeps WALL CLOCK
# beside the deterministic columns, and only the latter are quotable -- the
# accept lengths and rollback counts reproduce to the last digit across runs
# while the speedup column moves with desktop load (Gotcha 3).
#
# THE THREE WORKLOADS DO DIFFERENT JOBS and span per-position acceptance 0.53
# to 0.98. `prose` is the one that DECIDES: lowest acceptance, the only one on
# which the serving block does not pay, and where a large block is
# catastrophic rather than merely worse -- so DFLASH_SERVING_BLOCK's
# justification rests on its rows, and it is swept here so they are
# reproducible from this repo. `math` was added as the expected HARD case and
# measured as the easiest; that is recorded rather than tidied away, because
# the refuted prediction is what sharpened the mechanism (block choice turns
# on compounding and rollback cost, not on acceptance).
#
# One confound to know before quoting the seconds: `BLOCKS` is swept 7, 8, 4,
# 2, so block 2 always runs LAST and a monotone load trend aliases with the
# block ordering. Interleaving the blocks would remove it.
#
# TWO ASSERTIONS RATHER THAN THE BYTE-IDENTITY ONE IT USED TO MAKE. Every
# block size must generate IDENTICAL text to every other, which is strictly
# stronger and has no length below which it stops looking; and each arm must
# match the SEQUENTIAL stream for a floor of 64 tokens, printing where it
# actually diverged. Byte-identity to sequential is measured FALSE past a few
# hundred tokens -- a batched row differs from a one-row pass in the last
# bits, which is this port's SHAPE FLOOR and not a defect (1e-5 nats with the
# argmax agreeing, against 7.4e-6 on MLX for the same architecture; an earlier
# note here blamed the batched INT4 kernel, which is bit-exact against the
# GEMV). So a floor is what is true -- 64 against an observed 154, because the
# divergence point is data dependent.
#
# It samples through `selection::select` under the same greedy ShapingConfig
# `real_model.rs` builds, not a local argmax, so a sampler regression is
# visible to it. Routing it there moved no deterministic column.
TURBOSPARK_DFLASH2_INSTALL_DIR=~/models/qwen38-27b-dflash2.gturbo \
  cargo test -p turbospark-bench --test dflash2_accept_length_probe --release -- --ignored --nocapture
TURBOSPARK_DFLASH2_INSTALL_DIR=~/models/qwen38-27b-dflash2.gturbo \
  cargo test -p turbospark-bench --test dflash2_bisect_probe --release -- --ignored --nocapture

# Live directional steering (`docs/OBLITERATION.md`, ROADMAP item 9 phase 3).
# Three opens of ONE install, ~30 s: the edit is rank-1 and reversible, so
# comparing a steered engine against its original needs no second model.
#
# FOUR ARMS, and only the first is an assertion about the KERNEL: at alpha 0
# the dispatch runs at every covered layer and the output must be
# BIT-IDENTICAL to steering off, which covers a wrong reduction, a wrong
# buffer offset and a wrong row stride in one comparison. Then the divergence
# in nats against the dense shape floor, the per-layer coefficient trace, and
# a determinism check.
#
# **ARM 2 IS THE ONE PLACE IN THIS CRATE WHERE A LARGE KL IS THE GOOD
# OUTCOME**, which inverts every other divergence measurement here -- and it
# is why the scale DEFAULTS TO 0.3 rather than 1.0. See Gotcha 18.
#
# **DO NOT REGENERATE THE VECTORS THESE ROWS WERE FROZEN ON.** The control
# vector numbering was corrected on 2026-08-24 (`direction.N` is llama.cpp's
# block N, not N-1), and `extract_direction.py` can no longer write a
# direction for block 0. The files below predate that, declare
# `turbospark.layer_base = 0`, and are still read under the old convention on
# purpose, so every frozen row reproduces against THEM -- both still read 64
# covered of 64 spanned. A re-extraction is a different experiment rather
# than a reproduction, and block 0 is not a small thing to drop: its TRUE
# removed fraction is 72.1%, the highest in the model. See
# `docs/OBLITERATION.md`.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
TURBOSPARK_STEERING_VECTOR=/tmp/steer/ocean.gguf \
  cargo test -p turbospark-bench --test steering_probe --release -- --ignored --nocapture

# Which steering alpha is usable on a given direction. Six opens, ~2 min.
# Generates greedily under a STEERED engine at each alpha, then teacher-forces
# those ids through the UNSTEERED one -- no second checkpoint, because the
# edit is rank-1 and reversible.
#
# **THE NUMBER IS NOT A COHERENCE SCORE** and one row of it says nothing: it
# rises both when the edit WORKS (a steered model should say things the
# unsteered one would not) and when it does DAMAGE. Only the shape separates
# them, which is why this is a sweep and not a knob on `steering_probe`.
# The ANCHOR is the unsteered model on the frozen reference answer -- what
# fluent prose costs it -- and on `qwen38-27b` it reproduces
# `qwen38_quality_gate`'s 4.9432 exactly, which is a free check that this
# target and that gate walk the same ids. See Gotcha 19.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
TURBOSPARK_STEERING_VECTOR=/tmp/steer/ocean.gguf \
  cargo test -p turbospark-bench --test steering_sweep --release -- --ignored --nocapture

# Its other two axes. `TURBOSPARK_STEERING_MODE` points it at a different edit
# WITHOUT rewriting the vector, which is what keeps a mode comparison
# single-variable (one artifact, swept twice); an unknown spelling is REFUSED
# rather than falling back to the file's declaration.
# `TURBOSPARK_STEERING_BANDS` sweeps the layer band beside alpha -- `all` or
# START:END, inclusive and 0-based, the spelling `--steering-layers` takes,
# applied through the same `SteeringSet::restrict_to_range` the CLI calls. A
# band covering ZERO layers is refused: it steers nothing, so every alpha
# under it would read as usable.
#
# **READ THE `distinct` COLUMN BESIDE THE VERDICT** -- see Gotcha 20.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
TURBOSPARK_STEERING_VECTOR=/tmp/steer/ocean.gguf \
TURBOSPARK_STEERING_MODE=renorm \
TURBOSPARK_STEERING_BANDS=all,0:50 \
TURBOSPARK_STEERING_ALPHAS=0,0.4,0.45,0.5,0.55,0.6 \
  cargo test -p turbospark-bench --test steering_sweep --release -- --ignored --nocapture

# `TURBOSPARK_STEERING_PROMPT` is what makes a SECOND direction measurable, on
# this target and on `steering_probe` alike. Both were written against one
# ocean-adjacent string, and a vector extracted from some other concept pair
# has no reason to move it -- so without this the sweep measures the prompt as
# much as the direction. It defaults to that same string, so every frozen row
# in `docs/OBLITERATION.md` reproduces with it unset (checked), and the
# resolved prompt is PRINTED so a published row cannot silently be a different
# one. Crossing two directions with two prompts is what established that the
# usable band belongs to the direction -- see Gotcha 22.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
TURBOSPARK_STEERING_VECTOR=/tmp/steer2/register.gguf \
TURBOSPARK_STEERING_PROMPT="Explain why the sky appears blue." \
  cargo test -p turbospark-bench --test steering_sweep --release -- --ignored --nocapture

# Sensitivity proof for the gate above: APFS-clone the install, shift one
# quantization level in a strided subset of the routed experts, re-measure.
# ~30 s. Response curve and detection floor in docs/BENCHMARKS.md.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_sensitivity --release -- --ignored --nocapture
```

## Crate Gotchas

1. **Footprint Accounting**: `phys_footprint` is KV cache + expert slot capacity + process baseline. The slot term DOMINATES at `slots x layers x expert_stride`, so depth counts as much as expert size: Qwen3-30B-A3B peaks at 2,751 MiB against Gemma 4's ~2,100 because it is 48 layers deep at a ~2.9 MiB expert (2,094 MiB of slot capacity) rather than 30 at ~3.2 MiB. Do not assume a new family lands in the 1.6-2.2 GiB band; compute the product.

   **`phys_footprint` DOES NOT COUNT A FILE-BACKED MAPPING, EVEN ONCE METAL HAS
   WRAPPED IT AND THE GPU HAS READ IT.** This gotcha used to open "includes
   resident weight mapping (`mmap` pinned by Metal `newBufferWithBytesNoCopy`)".
   That was never measured, it is wrong, and it contradicted Gotcha 11 in this
   same file (a dense install's 4.07 GiB of weights reading a 684 MiB footprint)
   plus AGENTS.md Gotcha 40. Two entries disagreed with it and it survived anyway.
   MEASURED 2026-08-23 on the real Gemma 4 install
   (`tests/mapped_expert_probe.rs`), mapping the whole 12.3 GB expert table as 30
   per-layer files:

   | stage | phys_footprint | delta |
   |---|---|---|
   | baseline | 31.9 MiB | |
   | after `mmap` of 12.3 GB | 31.9 MiB | +0.0 |
   | after 30 `newBufferWithBytesNoCopy` wraps | 34.8 MiB | **+2.9** |
   | after ONE GPU expert read | 174.1 MiB | +139.2 |
   | after one expert on each of 30 layers | 174.2 MiB | **+0.1** |

   The wrap costs the buffer OBJECTS and nothing else. The +139.2 is one-time MSL
   pipeline compilation, not pages: the sweep after it faulted in ~96 MiB of fresh
   file-backed pages across 30 separate mappings, through the GPU, and moved the
   counter by 0.1 MiB. Clean file-backed pages are excluded whoever reads them.

   **AND THE SLOT TERM IS CONDITIONAL ON THE RESIDENCY MODE SINCE 2026-08-23.**
   Under `TURBOSPARK_EXPERT_RESIDENCY=mapped` there is no slot cache at all -- the
   routed experts are read in place out of one `mmap` per layer -- so the dominant
   term goes to zero and what is left is KV plus baseline. Every frozen row in this
   crate is a STREAMED row, and the seam is off by default precisely so they stand;
   a mapped row is a NEW row rather than a re-freeze of an old one. See
   `docs/EXPERT_RESIDENCY.md` and AGENTS.md Gotcha 36, whose slot-cache
   multiplication is conditional on the same mode.

   **What survives from the old text is the warning that motivated it**: do not
   "explain" a footprint number by assuming you know which term dominates. Measure
   it. `tests/memory_oracle.rs` asserts the session peak against the published
   ceiling, and separately that a replayed warm case stops growing (Gotcha 17).
2. **Cold GPU Benchmark Artifacts**: The first run after a build executes on a cold GPU at low DVFS clock states (up to 53% slower). Always discard at least one warmup run.
3. **Power Source & Cross-Session Ratios**: Thermal throttling and battery state (`pmset -g ps`) alter absolute tok/s. Always measure ratios back-to-back in the same session.

   **The axis that moves is THERMAL HEADROOM, not energy.** One binary ran the whole protocol on both power sources (`docs/POWER_BASELINE.md`): watts and joules-per-token differ by a few percent with no consistent sign, which is ordinary cross-session drift. Pressure is what diverges. On AC 50 of 50 arms held Nominal; on battery `long-synthesis` left Nominal on every run of both installs and two further runs were lost the same way, so the battery capture has holes the AC one does not. Keep recording the source, and expect the difference to arrive as throttling rather than as a watts correction. The practical rule is stronger than "prefer AC": an effect smaller than a few percent CANNOT be measured on battery at all. The read-pool QoS seam read as a clear loss there (one pair at +8.8% energy) and as a null result on AC, and the AC reading is the correct one.

   Cross-session absolute numbers here have repeatedly failed to reproduce while ratios measured back to back within one session have held. Prefer the ratio; `crates/gpu/tests/attention_chunk_bench.rs` reports speedups rather than absolute microseconds for this reason.
4. **`--model` mode does NOT auto-detect Low Power Mode, and that asymmetry against the CLI and server is deliberate.** `rate_control_for` here is handed `PowerProfile::Performance` unless `--power-profile` says otherwise, where `crates/cli` and `crates/server` call `resolve_profile` and let LPM select `efficiency`. A measurement tool has to be explicit: on an LPM-enabled machine the implicit default would cap the `performance` arm of a `scripts/power.sh` A/B at reading speed and manufacture exactly the efficiency gain the A/B exists to measure, with nothing in the output saying so. The two oracles and the quality gates pass `RateControl::default()` for the same reason -- the oracle asserts a decode tok/s FLOOR, which any cap would fail.

   **THE HEADER ECHOES THE RESOLVED PAIR** (`power_profile=`, `max_tok_s=`, `thermal_stepping=`), and that is the arm's only tell inside the artifact it labels. An arm of a `scripts/power.sh` A/B is named entirely OUTSIDE this process, so without those fields a `--power-profile efficiency` run and an uncapped one differ only in the tok/s column -- and "the number moved, so the flag must have worked" is not evidence, it is the assumption that let `COOLING=max` be passed to a script that ignored it and reported success. Both values are printed because neither implies the other: an explicit `--max-tokens-per-sec` overrides the profile's cap without changing whether the ladder runs, so `performance` at 15 tok/s and `efficiency` at 15 tok/s are different runs agreeing on every other column. Note every Phase P2 row published before 2026-08-18 was captured without this line.
5. **Slot count is pinned, not defaulted-into**: `protocol::PROTOCOL_EXPERT_CACHE_SLOTS` (16) is what `docs/BENCHMARKS.md`, the memory oracle's per-chip rows, and Swift's own default all sit at. `--expert-cache-slots` exists so a Swift comparison can match a non-default setting, not so the protocol can drift; the oracle passes the constant explicitly for that reason. Output IS identical across slot counts on both families as of 2026-08-08 (routed slots dispatch in router rank; AGENTS.md Gotcha 27), so a slot-count change is a throughput axis only.

   **THAT SENTENCE STOPPED BEING A CONVENTION AND BECAME LOAD-BEARING ON 2026-08-16**, when `--expert-cache-slots` gained an `auto` value and the two user-facing binaries defaulted to it. `turbospark-check` and `turbospark-server` now size the cache against the machine at open; this crate does not, and every gate reached through it inherits the pin. What guarantees the pin cannot leak is a signature: `RealForwardRunner::open_with_options` still takes a plain `usize` and the policy-taking `open_with_slot_policy` is a separate function, so an `Auto` cannot reach a frozen footprint row by defaulting into one. Do not "simplify" those two into one entry point. The practical consequence for a reader comparing numbers: a tok/s footer from `turbospark-check` is NOT comparable to a row here unless it was run with `--expert-cache-slots 16`.
6. **The phase report does not account for a decode run**: `TURBOSPARK_PHASES=1` covers the inside of `produce` only; the sampler and detokenizer are outside it (AGENTS.md Gotcha 23). Subtract the phase total from the footer's `decode=` seconds before trusting a phase table as a full attribution.
7. **Perplexity is only meaningful on ASSISTANT-position tokens**: both supported checkpoints are instruction-tuned, and instruction tuning masks the loss on the prompt, so the model was never trained to predict user-turn text or the markup closing it. Measured on the real Gemma 4 install, teacher-forcing the PROMPT gave a mean NLL of 15.3 nats against a uniform bound of 12.5 (worse than guessing) while assistant-side tokens in the same sequence scored 0.000; a replay of the model's own greedy output agreed 39/40, so the measurement was sound and the corpus was not. `quality_common` therefore teacher-forces a fixed reference answer into the assistant slot and scores only that. Sibling note: digests used to be irreproducible from a cold expert cache (slots were ordered misses-first, permuting the phase-2 reduce order); slots now dispatch in router rank, so cold and warm agree. The warmup before each measured generation is kept for throughput, not for the digest.
8. **A cross-engine KL number is meaningless without a floor measured beside it, and there are TWO floors, not one.** `scripts/kld.py` runs mlx-lm TWICE on purpose: once token-by-token through a cache (the shape this port runs in, and the headline comparison) and once as a single batched pass. The divergence between those two holds the weights, the kernels, and the engine fixed and varies only the reduce shape, and at 4 bits it still costs 0.0352 mean nats and 4% of the argmaxes -- MORE than this port's 0.0264 against mlx-lm. Drop that second run and the headline has no scale to be read against. Two sibling traps. Do NOT carry `quality_common`'s assistant-slot-only rule over to the KL: that rule exists because SFT masks prompt loss, while a distribution comparison is valid at every position, so all 550 are used. And do NOT budget for an f16 storage floor: mlx-lm returns bfloat16, whose 8 mantissa bits are coarser than f16's 10 at these softcapped magnitudes, so the round trip through this port's dump width is lossless (measured 3.5e-22 nats) and mlx is the lower-precision side. **The SECOND floor is the BACKEND**, and it was missed on the first pass of the llama.cpp comparison (`scripts/kld_llamacpp.py`, AGENTS.md Gotcha 34): the same dump against llama.cpp on CPU reads 0.05838 mean nats and looks like a defect, against llama.cpp on Metal 0.00845, because ggml's own two backends disagree by 0.05510 on this model -- 41x its shape floor. Same bytes is not enough; a reference engine with more than one arithmetic backend has an axis that is invisible unless it is measured, since both arms are "the same engine on the same file". **AND A FLOOR CAN BE UNAFFORDABLE RATHER THAN FORGOTTEN, which has to be said in the report** (ROADMAP's 1-bit entry): `scripts/kld_mlx_affine.py` measures the shape floor and NOT the backend floor, because that family is a DENSE 27B whose CPU arm would read all 24.8B backbone weights per token where `qwen3moe` and `gpt-oss` read 3B. **THAT LAST CLAUSE INVITES A WRONG GENERALISATION AND IT WAS MEASURED FALSE 2026-08-20**: Ornith-1.5-35B-A3B is an MoE at 3B active, so by the active-parameter reasoning its CPU arm should have been cheap like `qwen3moe`'s ~40 s -- mlx runs it at **15.4 s per position**, about 2.5 h for the corpus. `qwen3moe`'s number is llama.cpp's CPU backend, not mlx's, and the two are not interchangeable. Affordability of a backend floor is a property of the REFERENCE ENGINE's CPU path; probe it (five positions and a multiply) rather than inferring it from the model. Its report carries the string rather than the number, so a reader cannot mistake one floor for two. The same run is also where BOTH numbers came out tiny (0.0000157 against a 0.0000074 floor, 100.0% top-1) and that is the MoE term's absence rather than a better port: batched-vs-cached expert routing and reduce order is what makes the other families' floors big, so the RATIO transfers between families and the absolutes do not. **CORROBORATED ON A SECOND DENSE FAMILY 2026-08-20, AND IT ADDS A RULE: on a dense model, read the BACKEND floor and not the shape floor.** The `qwen35` 9B against llama.cpp reads a shape floor of 0.0000024 against the MoE families' ~0.00135 -- a different reference engine, a different quantization and three orders of magnitude on the same axis, which is what makes it the model SHAPE rather than one artifact. **AND THE CONTROLLED VERSION OF THAT COMPARISON EXISTS SINCE 2026-08-20**: the same family's MoE half (Ornith-1.5-35B-A3B, MLX reference) reads a shape floor of **0.02780** against its dense sibling's 0.0000024, measured the same day on the same corpus -- four orders of magnitude with the architecture lineage, the tokenizer and the protocol held fixed, so the only thing varying is whether there are experts to route. The consequence is that its headline is 58x that floor, the worst ratio on record here, while sitting 15x BELOW its backend floor and 61x below Gemma's headline in absolute terms. Nothing is wrong: the denominator collapsed, because a dense model's batched and cached passes are nearly the same computation. A shape floor is a scale only while there is something for it to be made of, and the reflex to quote the shape-floor ratio is exactly what would have turned a clean result into a false alarm here.
9. **Byte-identity under a constrained expert cache holds on BOTH families, and the gate asserts it rather than freezing a second digest**: the fourth arm reopens the install at 8 slots, repeats the greedy digest, and requires it to equal the 16-slot one. That assertion is the regression test for AGENTS.md Gotcha 27 -- it fails if a layer's routed slots are ever again dispatched in an order that depends on cache state rather than on the route. It did fail on Gemma before 2026-08-08, which is why the row used to carry a `constrained_digest` field; that field is gone.
10. **A logits buffer's width is the MODEL's, never the tokenizer's.** Every harness here that drives a `RealForwardRunner` calls `RealForwardRunner::vocab_size()`. `MfTokenizer::vocab_size` is a per-DIALECT constant standing in for a padded head width, which was correct only while one model used each dialect -- ChatML's row is Qwen 3.6's 248,320, and Qwen3-30B-A3B is also ChatML at 151,936. See AGENTS.md Gotcha 37. The scripted paths still use the tokenizer's, correctly: they have no model to ask.

11. **A ceiling is a ceiling AT ONE CONTEXT WINDOW, and one family's row is measured at a different one.** `run_oracle` takes the protocol's 4,096; `run_oracle_at_context` takes a window, and `mistral_memory_oracle.rs` passes 8,192. Two things follow. First, why it must: the protocol freezes the PROSE and its token count is the checkpoint's tokenizer's answer, so `long-synthesis` is 3,444 tokens under Mistral's 32k vocab against Qwen3-30B-A3B's 2,842, and `3444 + PROTOCOL_MAX_NEW` does not fit 4,096 -- the case does not run and the `endOfTurn` gate cannot be met on two of three cases. Raising `PROTOCOL_MAX_CONTEXT` is not an option: KV is sized `max_context * kv_stride` at open, so it would move every already-frozen peak in every other family. Second, why the window has to be READ with the number: on a dense install KV is nearly all of the counted footprint, because the resident weights are not counted at all (AGENTS.md Gotcha 40) and there is no expert slot cache, so Mistral 7B reads 1,201 MiB at 8,192 and 684 MiB at 4,096 -- the SAME install and the same model. Gotcha 1's `slots x layers x expert_stride` is the MoE families' dominant term and is exactly zero here. The window is printed on every run beside the ceiling for that reason; a row comparison that omits it is meaningless in a way the two numbers alone do not reveal. The BINARY resolves the same window per family now (Gotcha 16) and prints it in its header, so `turbospark-bench --model` and this row measure one workload.

12. **A GENERATION BUDGET is the second per-family protocol parameter, and it is a property of the MODEL rather than of its tokenizer** (ROADMAP M5). `run_oracle_with_budget` takes it; `gptoss_memory_oracle.rs` moves it to 3,072 and `museglimmer_memory_oracle.rs` to 2,048, both from the shared 1,024. Harmony puts gpt-oss's reasoning in an `analysis` channel BEFORE the answer, so the three cases need 818 / 2,153 / 1,108 tokens to reach `<|return|>` and two of three stop on `maxTokens` at the shared budget -- which the validity gate refuses, correctly (a truncated run is not comparable) and for a reason that is not a defect. Raising the SHARED constant is not an option in either direction: a larger budget lets every other family generate further and moves published rows, exactly as a wider window would (Gotcha 11). TWO THINGS TO CARRY. The budget was first derived from a GREEDY probe (1,780 for `medium-review`) and was too small: under the protocol's own T=0.2/top-k 64/top-p 0.95 the same case needs 2,153, because sampling lengthens the reasoning channel. And a reasoning model's answer length is a DISTRIBUTION, so the margin is margin and not a bound -- if this gate ever trips, raise the budget rather than reading it as a regression. The BINARY resolves this budget per family now (Gotcha 16); before that it ran the shared 1,024 whatever the install, so only `short-explanation` was a valid protocol case for gpt-oss and its power capture is n=1 on one case (`docs/POWER_BASELINE.md`). **AND THE MARGIN IS THINNER THAN THIS NOTE IMPLIES**: the first three-case binary run (2026-08-12) needed 926 / 2,597 / 1,303 tokens against the oracle's 818 / 2,153 / 1,108, i.e. `medium-review` came within 15% of the 3,072 budget. Nothing was wrong with either reading -- the bench does not pin the chat date, so gpt-oss's Harmony preamble differs from the gate's, and the answer length is a distribution. Read the 3,072 as one sample's margin plus 40%, and raise it rather than diagnosing a regression if a run ever truncates.

13. **The perplexity arm needs a per-family ASSISTANT PREFIX where the assistant slot is structured, and without it the number looks exactly like a broken model** (ROADMAP M5). `run_quality_gate_with_assistant_prefix` takes markup that is scored as PROMPT, never as a target. Empty for four of five families, because Gemma, ChatML, Mistral and Qwen all open an assistant turn and then say words. Harmony does not: its generation prompt ends at `<|start|>assistant` and the next token must be `<|channel|>`. Splicing the reference answer's prose straight in reads perplexity **148,421.76** against 6-38 for every other family, and against 255,409 for the genuinely broken Qwen of `5279c88`; with `<|channel|>final<|message|>` in front it reads 12.0801. THE THING THAT TELLS A FRAMING ARTIFACT FROM A BROKEN MODEL is that the generations were coherent throughout -- a model that cannot predict its own output does not write fluent prose, so a huge perplexity beside fluent text is a corpus-position bug every time. This is Gotcha 7's rule arriving from a new direction: not the wrong tokens, but the right tokens in a position the family's framing does not put them in.

14. **A gate whose prompt reads a CLOCK cannot be frozen without pinning it** (ROADMAP M5). Harmony's template writes `Current date: ` into its system preamble via transformers' `strftime_now`, so `gptoss_quality_gate.rs` sets `tokenizer::CHAT_DATE_ENV` before any render. Without it both digests and the perplexity expire at midnight and read as a numerics regression the next morning -- worse than having no digest. The pin lives in the GATE and not in the renderer on purpose: the renderer's job is to send what transformers, vLLM and llama.cpp send, which is the real date; only the measurement needs determinism. Any future template that calls a nondeterministic global needs the same treatment, and the tell is a digest that fails on a machine where nothing changed.

15. **Set a tok/s floor from at least two readings, and take the slower one.** The Mistral row's peak reproduced to 1.6 MiB across two back-to-back runs while its slowest case moved 12.5% (18.673 then 16.338 tok/s). A floor derived from the first reading at the usual ~0.75 margin lands at 14.0, which is 14% under the second reading and would flake on machine state rather than on a regression. Memory and throughput do NOT have the same reproducibility here, and a dense install makes the gap wider: there is no expert-slot warming to spread the peak, so the memory number is unusually tight and it is tempting to trust the throughput number beside it just as much.

    **THE VARIANCE IS PER CASE, AND THE CASE THAT SETS THE FLOOR IS USUALLY THE NOISIEST ONE.** One floor is asserted against every case, so the binding constraint is the slowest case's WORST reading -- and on Gemma 4, four readings on one day put `short-explanation` at a 3.8% spread, `medium-review` at 2.6%, and `long-synthesis` at **14.0%** (33.039 to 37.559). That is structural: `long-synthesis` prefills 3,015 tokens for ~67 s and decodes only 599, so a short decode window sits downstream of a long hot prefill. Two readings of the WHOLE protocol is therefore the minimum and not obviously enough; sample the governing case specifically before tightening a floor, and treat a single favourable reading of it as the least trustworthy number in the run. This is also why the Gemma floor stayed at 25.0 when a handoff proposed raising it against a 38.2 reading of that case -- see the row's comment block.

16. **`turbospark-bench --model` RESOLVES both per-family parameters from the install's own manifest, and the resolver's match is exhaustive on purpose.** `real_model::protocol_parameters` is the one table; `open_model_runner_for_protocol` returns it beside the runner, so the window used at OPEN and the window used at RUN cannot diverge (they are the same value, and a divergence is silent -- opening at 8,192 and running at 4,096 refuses the long case with nothing pointing at the mismatch). The header prints `family=`, `context=`, `max_new=` and `expert_cache_slots=` for the reason Gotchas 11 and 12 give. THREE THINGS TO CARRY. The match has NO wildcard arm, which is the actual guard: a `_ =>` would let a seventh family silently inherit Gemma's window and budget, the shape of AGENTS.md Gotchas 24/37/39. The two moved rows' numbers stay LOCAL to `mistral_memory_oracle.rs` and `gptoss_memory_oracle.rs` (they carry the row's provenance) with a `const` block asserting they equal the resolver's -- a BUILD failure, not a runtime one in an `#[ignore]`d target that almost never runs; both mutations were checked and both redden. And **a dense `llama` bench number taken before 2026-08-12 is at a different window**: the binary used to run that install at 4,096, so `long-synthesis` did not fit and the peak read ~684 MiB, where the resolved 8,192 gives 1,196.5 MiB and all three cases complete. That is the binary catching up to its own oracle row (1,201-1,203 MiB), not a regression. The four unmoved families are byte-for-byte unaffected: Gemma re-ran at 61 prompt / 505 new tokens, endOfTurn, peak 2,186 MiB. VERIFIED ON GPT-OSS the same day: all three cases endOfTurn (926 / 2,597 / 1,303 new tokens) where two of three used to stop on `maxTokens`, session peak 5,420.2 MiB against the oracle's own 5,417-5,421 -- which is the cross-check that says the binary and the oracle now run one workload.

17. **EVERY SELF-MEASURED `ChipBaseline` IS CHECKED AGAINST `models.json`, AND
   THAT CHECK IS NOT `#[ignore]`d.** A baseline row is an ASSERTION -- a
   ceiling and a floor with a per-row margin and a paragraph justifying it --
   while the catalog's `measured` block is the OBSERVATION that margin was
   chosen from, and `turbospark-model recommend` quotes the latter. Two files
   describing one run, with nothing structural keeping them in step, is the
   count-that-rots shape. `oracle_common::assert_agrees_with_catalog(alias,
   BASELINES, context, slots)` is the tie, called from a three-line `#[test]`
   in each of the eight oracle targets: it needs no install and no GPU, so a
   contradiction reddens on the edit rather than the next time somebody has a
   13 GB install on disk.

   Two things it deliberately does not do. It skips rows whose `source` is
   not this port's own measurement -- `memory_oracle.rs` carries Swift-sourced
   rows for chips nothing here has ever run on, and demanding a catalog row
   for those would mean inventing numbers. And it asserts the ceiling is
   within 1.5x its own evidence as well as above it, because a ceiling that
   has drifted far above the peak it was calibrated from has stopped being a
   guard.

   **When you re-freeze a row, move BOTH sides.** The catalog convention is
   worst-observed: slowest reading of the slowest case, fastest of the
   fastest, highest peak, plus the context and slot count they were taken at.

18. **A DIVERGENCE NUMBER CANNOT TELL A STEERED MODEL FROM A DESTROYED ONE,
   and `steering_probe.rs` is the one place here where a LARGE KL is the
   success signal.** Every other divergence measurement in this crate -- the
   cross-engine KLs, `batched_forward_probe` -- wants a small number read
   against a floor (Gotcha 8). Steering inverts that: the floor still says
   "this is the edit rather than noise", but the edit is supposed to move the
   distribution a lot. That inversion has a trap under it that fired on the
   probe's first run.

   Ablating a captured direction at `alpha = 1` over all 64 layers COLLAPSES
   the turn on `qwen3_5` -- the model emits end-of-turn immediately and
   generates nothing (`docs/OBLITERATION.md` records the mechanism: the
   direction's norm at the late layers is a quarter of the residual stream's,
   and that stream is the output head's input). A collapsed turn reads
   **21.818 nats, 2,948,321x the shape floor**, which is indistinguishable in
   that number from a spectacularly effective steer. The probe defaulted to
   exactly that operating point, passed, and reported the documented failure
   mode as a success.

   Two consequences, and the first generalises past steering.
   **A probe whose DEFAULT parameters land on a known-degenerate point is
   worse than no probe**, because it produces a quotable number with a green
   tick beside it; the default is 0.3 now, reproducing the doc's coherent row.
   And **the check that caught it was printing the generated TEXT**, not any
   statistic -- so the probe prints both continuations side by side and runs
   an objective collapse detector (first token is end-of-turn, or the whole
   continuation is one or two distinct tokens) that REPORTS rather than
   asserts, since deliberately measuring the collapse point is legitimate.
   Fluency is a judgement no test can make; "the turn ended immediately" is
   not. This is AGENTS.md Gotchas 30/57/59's shape on a new axis: an
   instrument reading a plausible value on degenerate input.

   One arm's mutation check is worth knowing rather than repeating. Deleting
   the rollback between the determinism arm's two runs reddens the test
   through the ENGINE's own KV-position guard and never reaches the equality,
   so an invariant is doing the work there. The only leak that arm can
   observe is one preserving the cursor while changing what the kernels read
   -- which is the shape a scratch buffer carrying state across calls has,
   and the reason to keep it.

19. **A VERDICT RULE CAN BE PRINCIPLED AND STILL WRONG, AND THE STEEPEST STEP
   IS THE ONE THAT WAS.** `steering_sweep.rs` reports which steering alpha is
   usable. Its first rule was the largest multiplicative jump between
   neighbouring arms, chosen deliberately to avoid a fabricated threshold
   (Gotcha 8's discipline, applied to a curve). It gives the wrong answer:
   on the real `qwen38-27b` it picked alpha 0.6 -> 0.8 at 60.7x and concluded
   everything at or below 0.6 was usable, while 0.6 was already emitting
   template markup at 13 distinct tokens against 34-36 for the fluent arms.

   **The largest jump lands INSIDE the wreckage**, because once output is
   degenerate the number keeps climbing and the interesting transition is
   already behind it. The criterion is the ANCHOR CROSSING instead: the last
   arm whose perplexity stays under what the unsteered model pays for
   ordinary human prose. That is not a fabricated constant either -- the
   anchor is measured, and the argument is structural, since a model's own
   GREEDY output is the argmax path and should be far MORE predictable to
   that model than human writing. The fluent arms sit at 0.3x the anchor.

   **THE COLUMN STOPS BEING MONOTONE PAST THE CROSSING** (0.8 scores 313.99
   against 1.0's 280.31), so any rule that assumes a monotone curve is
   reading noise once it is past the point it should have stopped at.

   Two further things this target establishes about its own instrument.
   Its anchor reproduces `qwen38_quality_gate`'s frozen 4.9432 to the last
   digit -- an independent code path over the same corpus, which is what says
   the two walk the same ids -- and mutating `teacher_forced_nll`'s scoring
   window moves it to 12.7892, which is what says the match discriminates
   rather than being a coincidence. That second run is also Gotcha 7 in
   miniature: scoring prompt tokens on an instruction-tuned checkpoint
   inflates perplexity, because the model was never trained to predict them.

   **And the number is NOT a coherence score**, however it is used. It rises
   when the edit works and when the edit does damage, with nothing in a
   single value separating those. Only the shape across a sweep does, which
   is why a one-alpha version would have been misleading rather than partial.

20. **THE ANCHOR CROSSING UNDER-CALLS DAMAGE, AND `distinct` IS THE SECOND
   SIGNAL THAT CATCHES IT.** Gotcha 19 records the steepest step being
   replaced by the anchor crossing, and the crossing is better and still not
   sufficient. Measured 2026-08-23 on the real `qwen38-27b` at a FINE alpha
   grid: `ablate` at 0.55 reads perplexity 2.1309, which is 0.4x the anchor
   and comfortably "usable" by the stated criterion, while its actual output
   is one sentence repeated with template markup between the copies -- 18
   distinct tokens against the unsteered arm's 34. The number is not lying; a
   short degenerate repetition genuinely IS predictable to the unsteered
   model. It is answering a different question from the one being asked.

   **THE COARSE GRID COULD NOT SEE THIS**, which is the part to carry. On
   `[0, 0.2, 0.4, 0.6, 0.8, 1.0]` both `ablate` and `renorm` report a usable
   band of 0.4 and look identical. On `[0, 0.4, 0.45, 0.5, 0.55, 0.6]` they
   report the same crossing (0.55) and are NOT identical: `ablate`'s
   vocabulary collapses 34 -> 18 AT that crossing while `renorm`'s falls
   36 -> 26 ABOVE it. A sweep's resolution is part of its verdict.

   The target REPORTS the disagreement and deliberately does not resolve it.
   There is no measured basis for a "distinct must stay above N" line, so
   inventing one would be exactly the fabricated threshold Gotcha 38 warns
   against; it prints the largest consecutive drop and warns when that drop
   lands at or before the crossing. Being a reporting feature, no test can
   redden on it -- its discrimination check is that on one real grid it fires
   for `ablate` and stays quiet for `renorm`, which is a pair of known-
   different cases rather than a single confirming one.

21. **A LAYER BAND IS APPLIED BY RESTRICTING THE SET, NOT BY A POLICY FIELD,
   and a band that covers nothing is the degenerate input to fear.**
   `SteeringPolicy` has no layer range; `SteeringSet::restrict_to_range` is
   the mechanism and it is what `crates/cli` and `crates/server` both call, so
   a band measured here is one a caller can actually ask for. A restriction
   that lands outside the vector's covered layers leaves ZERO layers steered,
   and an engine that steers nothing scores exactly like the unsteered one --
   so every alpha reads as usable and the run reports a flawless result for no
   work done. `steering_sweep.rs` refuses it by name and prints each band's
   covered count. Same species as AGENTS.md Gotchas 30, 57 and 59: an
   instrument returning a plausible value on degenerate input, where nobody
   investigates a good-looking answer.

22. **TWO INSTRUMENTS AGREEING ONCE IS ONE DATAPOINT, NOT A VALIDATION -- and
   the agreement is most convincing exactly where there is only one case to
   check it on.** `scripts/extract_direction.py` derives an alpha ceiling from
   activations with no generation at all; `steering_sweep.rs` measures a
   usable band by generating and scoring. On the ocean/mountain direction they
   read 0.36 and 0.4, one grid step apart, and `docs/OBLITERATION.md` recorded
   that as "the derived ceiling validated rather than merely plausible". The
   argument was good -- the two share no code and no inputs beyond the corpus.
   It was still a sample of one.

   On a second direction off the SAME checkpoint the derived number reads 0.09
   against a measured 0.8. **The failure is not a mis-scaling but a SIGN**: the
   second direction carries 3.8x the stream share and tolerates 2x MORE alpha,
   where the formula says tolerable alpha falls as share rises. No budget
   constant fixes that, which is what turns "recalibrate it" into "it does not
   predict this".

   **THE UNDERLYING DEFECT IS A QUANTITY SUBSTITUTION, and it is the shape to
   look for.** The column divides by `||d||` -- what the direction IS -- where
   `ablate` subtracts `|c_hat| = |d.x|/||d||`, what the STREAM carries of it.
   Through the deep layers those sit a factor of exactly 2.0 apart on both
   corpora, which is structural (a difference of means puts a positive row near
   `+||d||/2` along itself), and a constant factor is precisely why the column
   ranks layers usefully and calibrates badly. At layer 0 they are 180x apart
   and the conclusion INVERTS: the page's flagged "separates strongly and
   ablation is nearly free" layer removes 72.1% of its row, the most expensive
   in the model. A near-proportional wrong quantity is worse than an obviously
   wrong one, because it survives every eyeball check that is not a second case.

   Two habits this pays for. **Cross the variables before attributing a
   difference** -- the second direction arrived with a second prompt, and only
   a 2x2 (two directions x two prompts, four sweeps, ~5 min) showed the band is
   a property of the direction and the prompt moves it not at all. And **state
   which claims replicated**: both of the sweep's earlier refutations held on
   the new direction, so those are facts about this engine, while the ceiling
   was a fact about one corpus. A page that reports "we tested a second case"
   without splitting its claims that way has learned nothing transferable.

23. **THE LOAD AVERAGE IS THE WRONG INSTRUMENT FOR "CAN I MEASURE THIS?", AND
   FOUR SESSIONS DECLINED A MEASUREMENT ON IT.** `docs/OBLITERATION.md` owed a
   steering throughput number for five sessions. Each time it was deferred
   because the machine was busy, citing Gotcha 43 -- correctly in spirit and
   using a statistic that cannot answer the question. A load average says how
   much CPU is being consumed; it says nothing about whether the contention
   reaches the workload being timed.

   Measured 2026-08-24: with `spotlightknowledged` pegging a full core
   (indexing the several hundred test binaries the session had just built),
   three IDENTICAL decode arms read 22.227 / 22.254 / 22.233 tok/s -- a spread
   of **0.12%** against an expected effect of a few percent. Spotlight is
   CPU-bound and decode on this engine is GPU-bound, which is exactly the
   insulation Gotcha 43 already describes; nobody had tested for it. The A/B
   then resolved -1.72% cleanly.

   **The statistic that answers the question is the REFERENCE ARM'S OWN
   SPREAD**, and it costs three runs. It needs no norm, no previous clean
   capture and no judgement about which processes matter, and it measures the
   only thing that counts: whether this machine, right now, reproduces the
   quantity about to be compared. `docs/DFLASH2.md` already used it as a
   cleanliness TELL after the fact (three no-drafter arms at 0.4%); the change
   is to run it FIRST, as the gate on whether to measure at all.

   Two limits worth keeping. It is not a substitute for Gotcha 43 on ENERGY --
   watts are system-wide by construction, so a CPU-bound neighbour lands in
   `cpu_W` whatever the workload does, and the contamination floor stays the
   check there. And a tight spread licenses a RATIO between interleaved arms,
   not an absolute row: the `off` arm in that same capture drifted 22.376 ->
   22.232 across three rounds while the steered arms stayed flat, so the
   paired deltas were the honest reading and interleaving is what made the
   drift harmless.

24. **BEFORE RE-FREEZING A DIGEST THAT STOPPED MATCHING, PROVE IT ISN'T A
   REGRESSION.** `museglimmer_quality_gate.rs`'s 2026-08-15 golden stopped
   reproducing; `git bisect` from that commit to HEAD landed on the SAME
   new digest at every step, including the freezing commit itself rebuilt
   fresh -- which is what proves it, since a commit cannot fail to
   reproduce its own recorded output unless something outside the source
   tree changed. Checklist before re-freezing: (1) bisect to and including
   the freezing commit, not just to a suspect commit; (2) rule out the
   install's own files (check mtimes); (3) rule out the toolchain (macOS
   build, Xcode version); (4) manually inspect the new output for
   coherence, not just a healthy perplexity number; (5) confirm the new
   value is stable across several fresh runs, not a one-off. Isolate
   whether YOUR uncommitted diff is the cause first with
   `git stash push -- <the one file>`, rebuild, re-run, `git stash pop` --
   cheaper than a full bisect and rules out the easy case immediately.

25. **`steering_probe.rs`'S SINGLE-POSITION DIVERGENCE CHECK CAN ALSO FAIL IN
   THE OPPOSITE DIRECTION FROM GOTCHA 18, BY READING A POSITION THE CHAT
   TEMPLATE HAS ALREADY DECIDED.** Gotcha 18 is a KL too LARGE to trust
   (collapse reads as a huge success). `gpt-oss` (2026-08-25) found the
   other failure mode: a KL too SMALL to trust, on a checkpoint where the
   edit is demonstrably real. The probe measures divergence at exactly one
   position -- the token right after the rendered prompt -- and on a
   Harmony checkpoint that position is `<|channel|>`: the assistant turn
   always opens `<|start|>assistant<|channel|>`, so that token is close to
   fixed by the TEMPLATE regardless of the prompt, the direction, or
   whether steering is even dispatched. Checked directly against the
   install's `tokenizer.json`, not inferred: the winning id (200005) is
   `<|channel|>` on every prompt/alpha combination tried, steered or not.
   So the single-position number was reading the format's own
   near-determinism, not the direction's absence -- and the coefficient
   trace (`|c|` in the thousands at several layers) plus a full CLI
   generation past that token both confirm the edit is real and visible in
   the actual output.
   **The general form, one layer past Gotcha 18's: a single fixed
   measurement position is only informative if that position is free to
   vary.** Before trusting (or distrusting) a one-position divergence
   number on a new family, decode what token sits at that position and ask
   whether the CHAT TEMPLATE, not the model, put it there. `museGlimmer`'s
   real-install write-up has the same shortfall recorded a session earlier
   as an open question ("this family's very first generated token sits at
   an unusually confident point") -- gpt-oss is the case where that guess
   became a checked, exact mechanism rather than staying a hypothesis. The
   probe has no per-template branch to detect this automatically; the
   mitigation used here was to fall back to reading generated TEXT past
   the forced token, the same move Gotcha 18 already made for the
   opposite failure.

   **THE MoE FLOOR MISMATCH THIS GOTCHA ORIGINALLY LEFT OPEN IS NOW FIXED,
   AND IT WAS NOT THE CAUSE.** `steering_probe.rs` used
   `DENSE_SHAPE_FLOOR_NATS` (a `qwen3_5`, i.e. dense, number) as the
   threshold for every install, which per AGENTS.md Gotcha 8's
   ratio-not-absolute rule checked `gpt-oss` against a bound ~182x too
   tight. `shape_floor_for` now reads `ArchConfig.num_experts` (not
   `family` -- several families cover both a dense and an MoE checkpoint
   under one tag, Gotcha 61) and picks `MOE_SHAPE_FLOOR_NATS = 0.00135`
   (`qwen3moe`, llama.cpp batched vs cached -- `docs/BENCHMARKS.md`, and
   the same number `batched_forward_probe.rs`'s own doc table already
   cited) for any install with `num_experts > 0`. Sourced the same way as
   the dense constant: an EXTERNAL reference engine's batched-vs-cached
   number for the shape, not this port's own -- `produce_batched`
   structurally cannot supply one, since it refuses a MoE install by name
   (`batched_forward_probe.rs`), so there is no batched-verify path here
   to measure this port's OWN MoE floor against.

   Re-run against `gpt-oss` with the fix in place: the printed floor
   correctly resolves to `MoE shape floor 1.35e-3 nats`, and the steered
   KL still reads ~1.6e-9 nats -- `0x` even that wider floor, not merely
   under the dense one. **So the floor was a real bug (a threshold off by
   over two orders of magnitude) and fixing it changes nothing about
   `gpt-oss`'s verdict**, because that reading's cause is the position
   pinned by the chat template above, an entirely different failure mode
   from the one a floor threshold can detect. The two fixes are
   independent: get the floor right so a future MoE family with a
   genuinely weak direction is judged correctly, and separately fix (or
   route around) a position the template has already decided before
   trusting any single-position divergence number on a new dialect.

   **THE SECOND FIX IS ALSO IN NOW, AND IT DID NOT NEED A PER-DIALECT
   BRANCH.** Arm 2 no longer measures one fixed position at all. It
   teacher-forces the UNSTEERED engine's own greedy continuation (already
   generated for arm 4's determinism check) through the STEERED engine,
   position by position, computes KL at every one of those positions plus
   the original prompt-final one, and asserts on the WINDOW's MAXIMUM
   rather than on entry 0 alone. A template-pinned position still
   contributes a near-zero entry, exactly as before -- the fix is not
   detecting that a position is pinned, it is refusing to trust any single
   entry, so the position where real content diverges (the doc's manual
   gpt-oss investigation found this "within the first two sentences" past
   the forced token) is what the max finds instead. This is the general
   form the paragraph above asked for: no knowledge of Harmony, or of any
   other dialect's framing, anywhere in the new code. Both the prompt-final
   KL and the window's max (with its position and the decoded argmax token
   on both sides) are still printed, so a template-pinned entry stays
   visible for diagnosis even though it no longer decides the verdict by
   itself. Backward-safe by construction, since the window strictly
   contains the old single position: a checkpoint that already passed
   (any dense family, where position 0 already clears the floor) cannot
   newly fail.

   **CONFIRMED ON BOTH REAL INSTALLS THE SAME DAY, with a fresh
   self-extracted direction on each (no published vector exists for
   either).** `museGlimmer`: prompt-final KL `8.3235e-10` nats (`0x` the
   dense floor, the exact false negative), window max `8.4657e-2` nats
   (**11440x** the floor) at window position 21, argmax `" provided"` on
   both sides. `gpt-oss`: prompt-final argmax decodes to `<|channel|>` on
   BOTH engines (confirming the mechanism directly, not by inference), KL
   `1.6498e-12` nats (`0x` the MoE floor), window max `4.4824e-2` nats
   (**33x** the floor) at window position 23, with the generated text
   confirming real divergence (`"...Likely they want a"` vs `"...This is
   ambiguous. We need context. The"`). Arm 2 PASSES on both. A second bug
   surfaced verifying this -- two of arm 2's OWN diagnostic `println!`
   arguments were transposed on the first pass (the `Nx the floor` ratio
   and the decoded window position printed in each other's slots), caught
   by the printed numbers not matching their own labels and fixed before
   these values were taken; the assertion itself, which compares the raw
   `f64`s directly, was never affected.

26. **THE 8-SLOT RATIO ARM IS THE FLAKIEST ASSERTION IN THIS CRATE, AND ITS
   FLOOR SITS INSIDE THE RANGE ONE MACHINE PRODUCES IN ONE SESSION.**
   Gotcha 15 says derive a tok/s floor from two readings and take the slower;
   Gotcha 23 says gate on the reference arm's own spread. This is the case
   where both rules were needed at once and the arm still went red on a
   correct tree.

   Measured 2026-08-29, `quality_gate.rs` on the real Gemma 4 install, three
   runs inside one hour on one machine: the constrained-vs-16-slot ratio read
   **0.27x, 0.68x and 0.85x** against a 0.50x floor. Two of the three passed.
   All three produced the IDENTICAL perplexity (37.4176) and the identical
   greedy, sampled and constrained digests -- so nothing about the model
   moved, and the arm was measuring the machine.

   **The contaminating load was Gatekeeper, and the trigger is in the workflow
   rather than in the environment.** `syspolicyd` verifies each freshly built
   test binary on first execution (`CLAUDE.local.md`), so running
   `cargo test --workspace` and then this gate puts hundreds of new binaries
   into verification exactly while the 8-slot arm is timing decode. The 0.27x
   run followed a full workspace test; the 0.85x run did not.

   **TWO THINGS TO DO WHEN THIS ARM FAILS.** Read the DIGESTS first: they are
   printed above the panic, and if they match the frozen row then no numerics
   moved and the failure is throughput alone. Then re-run on a settled machine
   before believing it -- and if a diff is in flight, `git stash push` the
   change and run the gate at HEAD, which is what separated "mine" from
   "the machine" here in one run.

   **8 slots is also structurally the worst case on this family**, which is
   why the ratio is wide rather than merely noisy: Gemma 4 is top_k 8, so
   `slots >= 2 * top_k` fails at 8, the pipelined routed branch does not
   engage, and the arm measures the fallback (AGENTS.md Gotcha 64). The floor
   is guarding a real property; it is the VARIANCE that makes 0.50x a
   coin-flip on a busy machine.
