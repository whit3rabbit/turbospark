# turbospark-bench

Throughput benchmark harness (`turbospark-bench`), mach memory sampler (`memory.rs`), frozen community benchmark protocol (`protocol.rs`), real model benchmark runner (`real_model.rs`), and memory oracle integration test (`tests/memory_oracle.rs`).

## Directory & File Structure

```
crates/bench/
+-- Cargo.toml              # Crate manifest
+-- src/
|   +-- lib.rs              # Library entry point (turbospark_bench)
|   +-- main.rs             # Binary entry point (turbospark-bench)
|   +-- memory.rs           # Mach memory sampler for physical footprint tracking
|   +-- protocol.rs         # Frozen community benchmark protocol definitions
|   \-- real_model.rs       # Real model benchmark runner driving RealForwardRunner
+-- tests/
|   +-- logit_dump.rs       # Full-vocab logit dump for the cross-engine KLD (scripts/kld{,_llamacpp,_mlx_affine}.py)
|   +-- memory_oracle.rs    # Memory oracle asserting peak footprint ceiling & steady state (Gemma 4)
|   +-- mference_bench.rs   # Benchmark harness integration smoke test
|   +-- oracle_common/      # Shared memory oracle assertion helpers (mod.rs)
|   +-- quality_common/     # Shared quality gate evaluation helpers (mod.rs)
|   +-- gguf_nondeterminism_probe.rs # Repeated warm greedy runs must produce ONE output
|   +-- quality_gate.rs     # Quality gate integration test (Gemma 4)
|   +-- quality_sensitivity.rs # Proof the gate sees quantization damage (Gemma 4)
|   +-- qwen36_memory_oracle.rs # Memory oracle for Qwen 3.6 family
|   +-- qwen36_quality_gate.rs  # Quality gate for Qwen 3.6 family
|   +-- mistral_memory_oracle.rs # Memory oracle for the DENSE llama family, at 8192 context
|   +-- qwen3moe_memory_oracle.rs # Memory oracle for Qwen3-30B-A3B (`qwen3moe`)
|   +-- qwen3moe_quality_gate.rs  # Quality gate for Qwen3-30B-A3B
|   +-- gptoss_memory_oracle.rs # Memory oracle for gpt-oss-20b, at 8192 context AND a 3072 budget
|   +-- gptoss_quality_gate.rs  # Quality gate for gpt-oss-20b; pins the template's date
|   +-- qwen38_memory_oracle.rs # Memory oracle for Qwen3.8-27B (`qwen35`), the family's FIRST
|   +-- qwen38_quality_gate.rs  # Quality gate for Qwen3.8-27B; no assistant prefix, and see its header for why
|   +-- ternary_quality_gate.rs # Quality gate for Ternary-Bonsai-27B (2-bit), the same family's third checkpoint
|   \-- ternary_memory_oracle.rs # Its oracle; the peak is Qwen3.8's on HALF the weights (Gotcha 40)
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
# split under MFERENCE_PHASES=1, but Swift's is decode-only and this
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
uv run --python 3.12 --with mlx-lm --with numpy scripts/kld.py /tmp/kld/turbospark

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

# Sensitivity proof for the gate above: APFS-clone the install, shift one
# quantization level in a strided subset of the routed experts, re-measure.
# ~30 s. Response curve and detection floor in docs/BENCHMARKS.md.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_sensitivity --release -- --ignored --nocapture
```

## Crate Gotchas

1. **Footprint Accounting**: `phys_footprint` includes resident weight mapping (`mmap` pinned by Metal `newBufferWithBytesNoCopy`) + KV cache + expert slot capacity + process baseline. The slot term is `slots x layers x expert_stride` and it DOMINATES, so depth counts as much as expert size: Qwen3-30B-A3B peaks at 2,751 MiB against Gemma 4's ~2,100 because it is 48 layers deep at a ~2.9 MiB expert (2,094 MiB of slot capacity) rather than 30 at ~3.2 MiB. Do not assume a new family lands in the 1.6-2.2 GiB band; compute the product.
2. **Cold GPU Benchmark Artifacts**: The first run after a build executes on a cold GPU at low DVFS clock states (up to 53% slower). Always discard at least one warmup run.
3. **Power Source & Cross-Session Ratios**: Thermal throttling and battery state (`pmset -g ps`) alter absolute tok/s. Always measure ratios back-to-back in the same session.
4. **`--model` mode does NOT auto-detect Low Power Mode, and that asymmetry against the CLI and server is deliberate.** `rate_control_for` here is handed `PowerProfile::Performance` unless `--power-profile` says otherwise, where `crates/cli` and `crates/server` call `resolve_profile` and let LPM select `efficiency`. A measurement tool has to be explicit: on an LPM-enabled machine the implicit default would cap the `performance` arm of a `scripts/power.sh` A/B at reading speed and manufacture exactly the efficiency gain the A/B exists to measure, with nothing in the output saying so. The two oracles and the quality gates pass `RateControl::default()` for the same reason -- the oracle asserts a decode tok/s FLOOR, which any cap would fail.
5. **Slot count is pinned, not defaulted-into**: `protocol::PROTOCOL_EXPERT_CACHE_SLOTS` (16) is what `docs/BENCHMARKS.md`, the memory oracle's per-chip rows, and Swift's own default all sit at. `--expert-cache-slots` exists so a Swift comparison can match a non-default setting, not so the protocol can drift; the oracle passes the constant explicitly for that reason. Output IS identical across slot counts on both families as of 2026-08-08 (routed slots dispatch in router rank; AGENTS.md Gotcha 27), so a slot-count change is a throughput axis only.

   **THAT SENTENCE STOPPED BEING A CONVENTION AND BECAME LOAD-BEARING ON 2026-08-16**, when `--expert-cache-slots` gained an `auto` value and the two user-facing binaries defaulted to it. `turbospark-check` and `turbospark-server` now size the cache against the machine at open; this crate does not, and every gate reached through it inherits the pin. What guarantees the pin cannot leak is a signature: `RealForwardRunner::open_with_options` still takes a plain `usize` and the policy-taking `open_with_slot_policy` is a separate function, so an `Auto` cannot reach a frozen footprint row by defaulting into one. Do not "simplify" those two into one entry point. The practical consequence for a reader comparing numbers: a tok/s footer from `turbospark-check` is NOT comparable to a row here unless it was run with `--expert-cache-slots 16`.
6. **The phase report does not account for a decode run**: `MFERENCE_PHASES=1` covers the inside of `produce` only; the sampler and detokenizer are outside it (AGENTS.md Gotcha 23). Subtract the phase total from the footer's `decode=` seconds before trusting a phase table as a full attribution.
7. **Perplexity is only meaningful on ASSISTANT-position tokens**: both supported checkpoints are instruction-tuned, and instruction tuning masks the loss on the prompt, so the model was never trained to predict user-turn text or the markup closing it. Measured on the real Gemma 4 install, teacher-forcing the PROMPT gave a mean NLL of 15.3 nats against a uniform bound of 12.5 (worse than guessing) while assistant-side tokens in the same sequence scored 0.000; a replay of the model's own greedy output agreed 39/40, so the measurement was sound and the corpus was not. `quality_common` therefore teacher-forces a fixed reference answer into the assistant slot and scores only that. Sibling note: digests used to be irreproducible from a cold expert cache (slots were ordered misses-first, permuting the phase-2 reduce order); slots now dispatch in router rank, so cold and warm agree. The warmup before each measured generation is kept for throughput, not for the digest.
8. **A cross-engine KL number is meaningless without a floor measured beside it, and there are TWO floors, not one.** `scripts/kld.py` runs mlx-lm TWICE on purpose: once token-by-token through a cache (the shape this port runs in, and the headline comparison) and once as a single batched pass. The divergence between those two holds the weights, the kernels, and the engine fixed and varies only the reduce shape, and at 4 bits it still costs 0.0352 mean nats and 4% of the argmaxes -- MORE than this port's 0.0264 against mlx-lm. Drop that second run and the headline has no scale to be read against. Two sibling traps. Do NOT carry `quality_common`'s assistant-slot-only rule over to the KL: that rule exists because SFT masks prompt loss, while a distribution comparison is valid at every position, so all 550 are used. And do NOT budget for an f16 storage floor: mlx-lm returns bfloat16, whose 8 mantissa bits are coarser than f16's 10 at these softcapped magnitudes, so the round trip through this port's dump width is lossless (measured 3.5e-22 nats) and mlx is the lower-precision side. **The SECOND floor is the BACKEND**, and it was missed on the first pass of the llama.cpp comparison (`scripts/kld_llamacpp.py`, AGENTS.md Gotcha 34): the same dump against llama.cpp on CPU reads 0.05838 mean nats and looks like a defect, against llama.cpp on Metal 0.00845, because ggml's own two backends disagree by 0.05510 on this model -- 41x its shape floor. Same bytes is not enough; a reference engine with more than one arithmetic backend has an axis that is invisible unless it is measured, since both arms are "the same engine on the same file". **AND A FLOOR CAN BE UNAFFORDABLE RATHER THAN FORGOTTEN, which has to be said in the report** (ROADMAP's 1-bit entry): `scripts/kld_mlx_affine.py` measures the shape floor and NOT the backend floor, because that family is a DENSE 27B whose CPU arm would read all 24.8B backbone weights per token where `qwen3moe` and `gpt-oss` read 3B. Its report carries the string rather than the number, so a reader cannot mistake one floor for two. The same run is also where BOTH numbers came out tiny (0.0000157 against a 0.0000074 floor, 100.0% top-1) and that is the MoE term's absence rather than a better port: batched-vs-cached expert routing and reduce order is what makes the other families' floors big, so the RATIO transfers between families and the absolutes do not.
9. **Byte-identity under a constrained expert cache holds on BOTH families, and the gate asserts it rather than freezing a second digest**: the fourth arm reopens the install at 8 slots, repeats the greedy digest, and requires it to equal the 16-slot one. That assertion is the regression test for AGENTS.md Gotcha 27 -- it fails if a layer's routed slots are ever again dispatched in an order that depends on cache state rather than on the route. It did fail on Gemma before 2026-08-08, which is why the row used to carry a `constrained_digest` field; that field is gone.
10. **A logits buffer's width is the MODEL's, never the tokenizer's.** Every harness here that drives a `RealForwardRunner` calls `RealForwardRunner::vocab_size()`. `MfTokenizer::vocab_size` is a per-DIALECT constant standing in for a padded head width, which was correct only while one model used each dialect -- ChatML's row is Qwen 3.6's 248,320, and Qwen3-30B-A3B is also ChatML at 151,936. See AGENTS.md Gotcha 37. The scripted paths still use the tokenizer's, correctly: they have no model to ask.

11. **A ceiling is a ceiling AT ONE CONTEXT WINDOW, and one family's row is measured at a different one.** `run_oracle` takes the protocol's 4,096; `run_oracle_at_context` takes a window, and `mistral_memory_oracle.rs` passes 8,192. Two things follow. First, why it must: the protocol freezes the PROSE and its token count is the checkpoint's tokenizer's answer, so `long-synthesis` is 3,444 tokens under Mistral's 32k vocab against Qwen3-30B-A3B's 2,842, and `3444 + PROTOCOL_MAX_NEW` does not fit 4,096 -- the case does not run and the `endOfTurn` gate cannot be met on two of three cases. Raising `PROTOCOL_MAX_CONTEXT` is not an option: KV is sized `max_context * kv_stride` at open, so it would move every already-frozen peak in every other family. Second, why the window has to be READ with the number: on a dense install KV is nearly all of the counted footprint, because the resident weights are not counted at all (AGENTS.md Gotcha 40) and there is no expert slot cache, so Mistral 7B reads 1,201 MiB at 8,192 and 684 MiB at 4,096 -- the SAME install and the same model. Gotcha 1's `slots x layers x expert_stride` is the MoE families' dominant term and is exactly zero here. The window is printed on every run beside the ceiling for that reason; a row comparison that omits it is meaningless in a way the two numbers alone do not reveal. The BINARY resolves the same window per family now (Gotcha 16) and prints it in its header, so `turbospark-bench --model` and this row measure one workload.

12. **A GENERATION BUDGET is the second per-family protocol parameter, and it is a property of the MODEL rather than of its tokenizer** (ROADMAP M5). `run_oracle_with_budget` takes it; only `gptoss_memory_oracle.rs` moves it, to 3,072 from the shared 1,024. Harmony puts gpt-oss's reasoning in an `analysis` channel BEFORE the answer, so the three cases need 818 / 2,153 / 1,108 tokens to reach `<|return|>` and two of three stop on `maxTokens` at the shared budget -- which the validity gate refuses, correctly (a truncated run is not comparable) and for a reason that is not a defect. Raising the SHARED constant is not an option in either direction: a larger budget lets every other family generate further and moves published rows, exactly as a wider window would (Gotcha 11). TWO THINGS TO CARRY. The budget was first derived from a GREEDY probe (1,780 for `medium-review`) and was too small: under the protocol's own T=0.2/top-k 64/top-p 0.95 the same case needs 2,153, because sampling lengthens the reasoning channel. And a reasoning model's answer length is a DISTRIBUTION, so the margin is margin and not a bound -- if this gate ever trips, raise the budget rather than reading it as a regression. The BINARY resolves this budget per family now (Gotcha 16); before that it ran the shared 1,024 whatever the install, so only `short-explanation` was a valid protocol case for gpt-oss and its power capture is n=1 on one case (`docs/POWER_BASELINE.md`). **AND THE MARGIN IS THINNER THAN THIS NOTE IMPLIES**: the first three-case binary run (2026-08-12) needed 926 / 2,597 / 1,303 tokens against the oracle's 818 / 2,153 / 1,108, i.e. `medium-review` came within 15% of the 3,072 budget. Nothing was wrong with either reading -- the bench does not pin the chat date, so gpt-oss's Harmony preamble differs from the gate's, and the answer length is a distribution. Read the 3,072 as one sample's margin plus 40%, and raise it rather than diagnosing a regression if a run ever truncates.

13. **The perplexity arm needs a per-family ASSISTANT PREFIX where the assistant slot is structured, and without it the number looks exactly like a broken model** (ROADMAP M5). `run_quality_gate_with_assistant_prefix` takes markup that is scored as PROMPT, never as a target. Empty for four of five families, because Gemma, ChatML, Mistral and Qwen all open an assistant turn and then say words. Harmony does not: its generation prompt ends at `<|start|>assistant` and the next token must be `<|channel|>`. Splicing the reference answer's prose straight in reads perplexity **148,421.76** against 6-38 for every other family, and against 255,409 for the genuinely broken Qwen of `5279c88`; with `<|channel|>final<|message|>` in front it reads 12.0801. THE THING THAT TELLS A FRAMING ARTIFACT FROM A BROKEN MODEL is that the generations were coherent throughout -- a model that cannot predict its own output does not write fluent prose, so a huge perplexity beside fluent text is a corpus-position bug every time. This is Gotcha 7's rule arriving from a new direction: not the wrong tokens, but the right tokens in a position the family's framing does not put them in.

14. **A gate whose prompt reads a CLOCK cannot be frozen without pinning it** (ROADMAP M5). Harmony's template writes `Current date: ` into its system preamble via transformers' `strftime_now`, so `gptoss_quality_gate.rs` sets `tokenizer::CHAT_DATE_ENV` before any render. Without it both digests and the perplexity expire at midnight and read as a numerics regression the next morning -- worse than having no digest. The pin lives in the GATE and not in the renderer on purpose: the renderer's job is to send what transformers, vLLM and llama.cpp send, which is the real date; only the measurement needs determinism. Any future template that calls a nondeterministic global needs the same treatment, and the tell is a digest that fails on a machine where nothing changed.

15. **Set a tok/s floor from at least two readings, and take the slower one.** The Mistral row's peak reproduced to 1.6 MiB across two back-to-back runs while its slowest case moved 12.5% (18.673 then 16.338 tok/s). A floor derived from the first reading at the usual ~0.75 margin lands at 14.0, which is 14% under the second reading and would flake on machine state rather than on a regression. Memory and throughput do NOT have the same reproducibility here, and a dense install makes the gap wider: there is no expert-slot warming to spread the peak, so the memory number is unusually tight and it is tempting to trust the throughput number beside it just as much.

16. **`turbospark-bench --model` RESOLVES both per-family parameters from the install's own manifest, and the resolver's match is exhaustive on purpose.** `real_model::protocol_parameters` is the one table; `open_model_runner_for_protocol` returns it beside the runner, so the window used at OPEN and the window used at RUN cannot diverge (they are the same value, and a divergence is silent -- opening at 8,192 and running at 4,096 refuses the long case with nothing pointing at the mismatch). The header prints `family=`, `context=`, `max_new=` and `expert_cache_slots=` for the reason Gotchas 11 and 12 give. THREE THINGS TO CARRY. The match has NO wildcard arm, which is the actual guard: a `_ =>` would let a seventh family silently inherit Gemma's window and budget, the shape of AGENTS.md Gotchas 24/37/39. The two moved rows' numbers stay LOCAL to `mistral_memory_oracle.rs` and `gptoss_memory_oracle.rs` (they carry the row's provenance) with a `const` block asserting they equal the resolver's -- a BUILD failure, not a runtime one in an `#[ignore]`d target that almost never runs; both mutations were checked and both redden. And **a dense `llama` bench number taken before 2026-08-12 is at a different window**: the binary used to run that install at 4,096, so `long-synthesis` did not fit and the peak read ~684 MiB, where the resolved 8,192 gives 1,196.5 MiB and all three cases complete. That is the binary catching up to its own oracle row (1,201-1,203 MiB), not a regression. The four unmoved families are byte-for-byte unaffected: Gemma re-ran at 61 prompt / 505 new tokens, endOfTurn, peak 2,186 MiB. VERIFIED ON GPT-OSS the same day: all three cases endOfTurn (926 / 2,597 / 1,303 new tokens) where two of three used to stop on `maxTokens`, session peak 5,420.2 MiB against the oracle's own 5,417-5,421 -- which is the cross-check that says the binary and the oracle now run one workload.
