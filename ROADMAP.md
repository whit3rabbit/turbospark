# Roadmap

The forward-looking roadmap and prioritized task tracker for this engine. All core port phases (Q, P1, G, S, P2, M1-M5) are complete and green. This document functions as an active TODO list for forward engineering, measurements, and architectural bring-ups.

All completed work, historical milestones, and landed features have been removed to focus strictly on remaining tasks.

---

## Current Status

- **Swift conversation layout (2026-09-19)**: Active chats now share a bounded reading column with the composer, compact plans and tool activity, and expose tasks at the upper right. Model Settings folds when a conversation begins and reopens on request. Layout contracts and visual-review commands live in [swift/docs/SWIFT_CONVERSATION_LAYOUT.md](swift/docs/SWIFT_CONVERSATION_LAYOUT.md).
- **Test Suite**: Counts and gating conventions live in [docs/TESTING.md](docs/TESTING.md). A fresh `cargo test --workspace` passed on 2026-09-17 with no failures; this includes the two `turbospark-bench --test vision_sidecar_opener` cases, which now pass against the current 27-block synthetic tower. `cargo fmt --check`, full workspace Clippy, the Swift package suite (78 tests, 17 expected real-model skips), the focused app image suite (9/9), the debug app build, and the signed release app bundle are green. The broader app XCTest suite still has 34 failures in concurrent non-image work, including system-prompt/date injection, tool catalog parity, font propagation, folder import, and agent advertisement; these are not IG4 image failures.
- **Architectures**: 15 declared `ModelFamily` variants. The fifteenth, `qwen3_vl`, landed 2026-09-18: the Qwen3-VL trunk runs the shared Llama flow, the pinned `mlx-community/Qwen3-VL-4B-Instruct-4bit` install passes real greedy and sampled CLI smokes with frozen quality (17.3463) and memory (793 MiB at 4096) gates, and the catalog row `qwen3vl-4b` is `verified`. Text and image intake: the deepstack vision half landed 2026-09-19 (the Vision bullet below; `DEVIATIONS.md`'s `qwen3_vl` section). Dense `qwen3` GGUF execution is verified on 0.6B Q8_0. Dense Qwen2/Qwen2.5 runs three real artifacts end to end: the MLX/HF 4-bit install and both single-file GGUF conversions (the pinned official Q3_K_M on the Q3_K resident kernels, and a single-file Q4_K_M), each with frozen quality and memory rows. MiniMax-M2 GGUF execution is implemented, but repetitive low-temperature smokes block release and catalog promotion. `deepseekV4Flash` remains scaffolded. The full support matrix lives in [docs/MODEL_FAMILY.md](docs/MODEL_FAMILY.md).
- **Vision**: Dense Qwen GDN (`qwen35`, upstream `qwen3_5`) has real-gated still-image support through the CLI, server, FFI, and Swift-facing APIs, including the verified standalone vision sidecar. Qwen GDN MoE (`qwen35moe`) is now real-gated too (2026-09-19): the first MoE vision artifact (`mlx-community/Qwen3.6-35B-A3B-4bit`, whose bytes carry the same 333-tensor tower) streams combined, reads a test page through the CLI and the server, and passes the four-stage mlx-vlm tower parity. `Qwen3Vl`'s deepstack vision landed the same day (2026-09-19): the depth-24 tower plus its three deepstack mergers ingest combined and sidecar, the tower emits three distinct row sets on the real bytes, the per-layer adds are perturbation-proven wired, the chunked and sequential prefill sites are byte-identical on an image prompt, and all seven tower stages agree with mlx-vlm at cosine 0.99999+. The other 12 registered families are text-only for images ([docs/QWEN3VL_PHASE0.md](docs/QWEN3VL_PHASE0.md), [docs/VISION.md](docs/VISION.md)).

---

## Prioritized Task Backlog

### Priority 0: Immediate / Startable Now

Low-friction, high-impact fixes, unblocked measurement runs, or low-hanging symmetry tasks that can be executed immediately.

#### 1. Prefill Energy Capture
- **Objective**: Run `ARMS=seq,chunked scripts/power.sh` on AC quiet machine (~12 min, sudo) to capture baseline chunked prefill power and J/tok.
- **Why Open**: Tooling was unblocked (`turbospark-bench --prefill-chunk off|auto|N` and `scripts/power.sh seq|chunked` wired), but clean baseline row run is still owed.
- **Preflight repair (2026-09-10)**: `scripts/power.sh` now refuses competing `turbospark-check`, `turbospark-server`, `turbospark-bench`, and `TurboSparkApp` processes alongside the legacy names. `python3 scripts/test_power_preflight.py` checks refusal before capture setup and sudo, plus the idle path. Removing either new name group fails its matching cases.
- **Capture attempted (2026-09-10)**: The user completed two sequential/chunked pairs per case on AC with automatic cooling. Medium/long arms reached Moderate or Heavy pressure; long-prefill J/token spread was 76.7% sequential and 65.5% chunked. The capture is recorded in `docs/POWER_BASELINE.md`, with raw rows in `docs/verification/prefill-energy-2026-09-10.tsv`, but is not an accepted baseline. The forced-cooling retry below is a separate operating point from the automatic-cooling baseline still owed.
- **Forced-cooling retry (2026-09-10)**: Three pairs per case with `COOLING=max` kept every phase Nominal, but long sequential prefill still had 62.1% energy spread (CPU power 21.30 W in pair 1 versus 8.43/8.85 W later). Short and medium paired energy differences change sign. Evidence and interpretation are in `docs/POWER_BASELINE.md` and `docs/verification/prefill-energy-max-2026-09-10.tsv`. Next: correlate per-process CPU activity with capture windows before another full run; do not discard the outlier or freeze an energy saving without explaining it.
- **Process attribution (2026-09-10, 21:52 UTC capture)**: The new process timeline caught `mediaanalysisd` consuming 69.55 CPU-seconds during about 46.9 seconds of pair 1 chunked prefill, alongside updater activity. All phases were Nominal, but chunked energy spread was still 29.2% (sequential 3.3%). Later pairs used 21.8%/16.8% less chunked prefill energy; keep them as observations, not a frozen saving. Evidence is in `docs/POWER_BASELINE.md` and `docs/verification/prefill-energy-cpu-trace-2026-09-10-process-summary.json`. Next: wait for the observed background work to subside and retain per-window process checks. The morning anomalies remain unattributed.
- **Capture attempt (2026-09-17)**: The current preflight passed on AC with no competing inference process, but `scripts/power.sh` stopped at the interactive `sudo powermetrics` password prompt before any benchmark arm ran. No energy row was recorded; rerun requires local sudo access.
- **Files to Touch / Run**:
  - `scripts/power.sh` (execute benchmark)
  - `docs/POWER_BASELINE.md` (record resulting rows)

#### 2. TurboQuant Real-Model Verification Findings
- **Verification completed (2026-09-09)**: Real probe readings now exist for gemma4, qwen38-27b dense, gpt-oss, qwen3moe, qwen4_exp, Spark and museGlimmer. Spark's five-case KV suite passes with individually targeted mutation checks. Literal Gemma output bytes match between `673341e^` and `673341e` in greedy and sampled modes with KV quantization off. All requested CLI smokes and applicable frozen family gates have run; task-created installs were removed after evidence capture.
- **Why Open**: gpt-oss KV4 greedy exhausts 3072 tokens in reasoning while its matched FP16 control completes (KV4 sampled completes). museGlimmer's FP16 sampled golden mismatches identically in two current runs and the isolated feature-era build, despite matching perplexity and greedy output. The remaining Qwen4 finding re-CONFIRMED 2026-09-18 on a freshly re-streamed install at identical token counts (542/676/499): the KV4 sampled answer denies the prompt's wetlands premise while the matched FP16 control keeps it, with unsupported claims in the latter too. These remain findings, not clean quality passes or proven general kernel regressions.
- **Interpretation**: gpt-oss's unframed probe perplexity remains unusable as a quality signal because of Harmony framing. Spark and museGlimmer also require their answer-slot prefixes for comparison with their quality gates. No quantization policy or golden has changed.
- **Evidence**: `docs/TRUBOQUANT.md`, `docs/verification/p0-2026-09-09.json`, `docs/verification/qwen4-2026-09-18.json` (Qwen4 KV4 re-run).

#### 3. `kv_quant_probe` Footprint Attribution (spread control complete)
- **Verification completed (2026-09-09)**: Probe order is `off, off, off, 3, 3.5, 4, 2, off`; deltas keep the first baseline and the footer reports repeated-off minimum, maximum and spread. Revised real-install probes passed on Spark, museGlimmer and Qwen4.
- **Why Open**: Qwen4's quantized deltas re-read +81.7 to +86.9 MiB on the 2026-09-18 re-run, but inside a repeated-off range that itself doubled to 149.0 MiB (off rows trending 2516.7 -> 2597.3 -> 2665.7 across the probe's opens). The candidate cause the earlier entry said was missing is now recorded: resident-page warming of the streamed 68 GiB install, whose mapping phys_footprint counts. Deltas remain within the reference spread; unresolved by the probe's own rule. An arm order interleaving quantized widths between the off rows would separate "quantized costs footprint" from "the run warmed up" and has not been run. Per-width perplexities reproduce 2026-09-09's values to the fourth decimal on the fresh artifact. Spark and museGlimmer reductions exceed their observed spreads; this warrants attribution work rather than an automatic causal conclusion or precise width ranking.
- **Evidence**: `crates/bench/tests/kv_quant_probe.rs`, `docs/TRUBOQUANT.md`, `docs/verification/qwen4-2026-09-18.json`.

---

### Priority 1: Near-Term Core Engine & Infrastructure

All Priority 1 items have landed and closed (vision memory sidecar parity and oracle arms, multimodal speculative safety boundary, mapped residency eviction benchmark and policy unification, exact rejection sampling for MTP speculation, and server request queue with arrival-order fairness).

---

### Priority 2: Architecture Bring-ups & Kernel Scaling

Adding missing high-demand model families, specialized Metal kernels, and architectural extensions.

#### 1. `gpt-oss-120b` (MXFP4 GGUF) Evaluation under Mapped Residency
- **Objective**: 36 layers, 128 experts top-4, 12.6 MiB stride, 59 GiB total. Evaluate under mapped expert residency to test SSD throughput limits on large models.
- **Why Open**: Model fits unified memory on 64GB+ Macs; benchmarks evaluate page cache bounds vs SSD read bandwidth.
- **Files to Touch**:
  - `crates/model-io/src/manifest.rs`
  - `crates/runtime/src/families/gptoss/`
  - `crates/catalog/src/`

#### 2. Step 4 Batched Attention Kernel
- **Objective**: Widen `attention_decode_partial` to hold M query rows per KV chunk with per-row online-softmax state generalized to M rows. Scope specifically on an attention-dominant family (e.g. dense Llama/Gemma).
- **Why Open**: Low value on GDN-heavy Qwen (only 2.6% of prefill), but valuable for pure attention transformers.
- **Files to Touch**:
  - `crates/gpu/src/shaders/attention.metal`
  - `crates/gpu/src/attention.rs`
  - `crates/runtime/src/families/llama/` or `crates/runtime/src/families/gemma4/`
  - `docs/BATCHED_PREFILL.md`

#### 3. SigLIP-Class Vision Tower Bring-up (`gemma4_unified`)
- **Objective**: Implement SigLIP-class vision tower encoder and intake for `gemma4_unified`. Gemma 4 text trunk already runs here; intake currently drops ~815 vision tensors.
- **Why Open**: Nearest-term VLM candidate to expand multimodal support beyond Qwen3-VL.
- **Files to Touch / Create**:
  - `crates/vision-io/src/siglip.rs` [NEW]
  - `crates/runtime/src/vision/`
  - `crates/repack/src/gemma4_checkpoint/classify.rs`
  - `docs/VISION.md`

#### 4. Missing Tool-Calling Markups
- **Objective**: Implement structured tool-call decoders for formats found in modern checkpoints: GLM XML `<arg_key>/<arg_value>`, MiniMax namespaced `<minimax:tool_call>`, Mistral `[TOOL_CALLS]`, and Kimi K2 section markers `<|tool_calls_section_begin|>`.
- **Landed**: Mistral `[TOOL_CALLS]` (2026-09-15), GLM XML dialect, and Kimi K2 dialect (2026-09-19) landed with `ChatDialect` resolvers, native decoders, and unit tests.
- **Still Open**:
  - **MiniMax**: The published M2 `tokenizer_config.json` shows `<minimax:tool_call>` wrapper tokens are non-special added tokens (ids 200052/200053), requiring a text-marker arm rather than an id-bracket arm. Dialect probe strings (`]~!b[`/`]~b]`/`[e~[`) need re-derivation against published M2 tables.
  - **GLM and Kimi K2 Real Generation Smokes**: Both dialect fixtures are reduced from real tables, but end-to-end real-checkpoint generation smoke tests remain owed. GLM-4.7-Flash is gated at deepseek2 intake by four recorded descopes (`q_lora_rank > 0`, split `attn_k_b`/`attn_v_b`, noaux_tc routing, and MXFP4 experts). Kimi K2 is blocked by missing `tokenizer.json` and model scale.
- **Files to Touch**:
  - `crates/tokenizer/src/structured_decoder/`
  - `crates/tokenizer/src/chat_template.rs`
  - `docs/TOOL_CALLING.md`

#### 5. `spark2_5` Follow-ups (Bring-up Landed)
- **Status**: The family landed in `2a1f1fc` (2026-09-09) with GGUF support, Metal kernels, memory oracle (575 MiB peak), and frozen quality gate (perplexity 12.6162). Bring-up details are in `docs/SPARK_PHASE0.md`.
- **What is Still Open**: The four deliberate descopes detailed in `DEVIATIONS.md`:
  - HF safetensors / MLX intake (currently GGUF-only).
  - Tool-call DSL parser (markup currently flows as ordinary content).
  - 1.7B sibling baseline scheme (requires per-checkpoint baseline scheme).
  - `scripts/kld_llamacpp.py` `CHECKPOINTS` row.
- **Files to Touch / Create**:
  - `crates/repack/src/arch_registry.rs`, `crates/catalog/src/`
  - `crates/tokenizer/src/structured_decoder/`
  - `scripts/kld_llamacpp.py`

#### 6. Native Z-Image-Turbo Optimization (IG5)
- **Status**: IG0 through IG4 are closed on the pinned 1024-by-1024 install. Text encoding, transformer denoise, VAE decode, PNG publication, cancellation, Swift C ABI, and the app's top-level Images destination with Create/Gallery views are complete and verified.
- **What is Open (IG5)**: Tune measured bottlenecks without weakening quality or memory gates; approximation work needs a separate proposal.
  - VAE tiling under heavy memory constraints.
  - Prefetch and allocator reuse.
  - Measured performance optimizations with fresh quality and memory evidence.
- **Reference Docs**: [docs/IMAGE_GENERATION.md](docs/IMAGE_GENERATION.md), [docs/ZIMAGE_TURBO.md](docs/ZIMAGE_TURBO.md).

#### 7. MiniMax-M2 Release Gates (GGUF Execution Implemented)
- **Status**: Split-GGUF streaming, FP32 router/bias preservation, whole-projection Q/K normalization, partial RoPE, sigmoid expert selection, and checkpoint framing landed on 2026-09-10. Real memory oracle passes at 8192 context; quality process reproduces perplexity and digests (`docs/MINIMAX_M2_PHASE0.md`).
- **Release Blocker**: Greedy and low-temperature sampled coastal-wetlands smokes repeat reasoning and exhaust 400 tokens. Temperature 1 completes coherently at 1240 tokens, but does not waive those failures.
- **Remaining**: Resolve repetition, pass release smokes, review/freeze quality and quiet-machine performance baselines, add exact artifact to catalog, and promote support.
- **Deferred**: Safetensors/FP8 intake, mapped expert residency, native tool-call parsing, MTP, vision, and later MiniMax variants.

---

### Priority 3: Long-Term Extensions & Platform Expansion

System architecture extensions, platform ports, and developer tooling.

#### 1. Remote Plugin Marketplace Registry Indexing (`TurboSparkApp`)
- **Status**: `PluginMarketplaceManager.swift` covers all four `MarketplaceSource` cases (HTTPS via URLSession, `.github`/`.git` via `MarketplaceGit`, local directories). Install, versioned caching, and v2 ledger are tested.
- **Still Open**: Shipped/builtin registry of community marketplaces (registry indexing), auto-update, and dependency closure (`swift/docs/SWIFT_PLUGINS.md`).

#### 2. Multi-Direction Steering & Automated Alpha Calibration
- **Status**: Multi-direction steering policy landed 2026-09-15 (`runtime::SteeringPolicy`, sequential dispatches, CLI/server `--steering` knob lists).
- **Still Open**: Automated alpha calibration command (`steering_sweep.rs` multi-vector mode), `activation_capture.rs` contrastive fixture pipeline, second real direction for an install, and Swift preset surface.

#### 3. DeepSeek-V4-Flash Metal Kernels & Feasibility (Scaffolded)
- **Objective**: Port CSA/HCA attention, unrolled mHC Sinkhorn, and sub-3bit GEMV Metal kernels (`dsv4.metal`), assessing 106.9 GB peak RSS memory feasibility.
- **Why Open**: Architecture is scaffolded; full implementation requires large unified memory (128 GB Mac).
- **Files to Touch / Create**:
  - `crates/gpu/src/shaders/dsv4.metal` [NEW]
  - `crates/gpu/src/`
  - `crates/runtime/src/families/dsv4/` [NEW]

#### 4. Linux Backend & Vulkan Compute (Portable Architecture)
- **Status**: `io_uring` + `O_DIRECT` read source landed in `crates/streaming/src/linux_uring.rs` (auto = pread until proven on Linux), `posix_fadvise(WILLNEED)` wired, and `crates/model-io/src/cgroup.rs` cgroup-v2 memory limit probe added.
- **Still Open**: Linux hardware session to run the uring path and flip default; runtime cfg arm consuming the cgroup probe; Vulkan compute backend (`ComputeStrategy` dispatch trait design and shader pipeline).

#### 5. Server Multi-Runner Concurrency Real-Model Evaluation
- **Status**: Multi-runner pool landed as `--pool-size N` on `turbospark-server` with `PoolRegistry`. Repeatable `--model` landed for multiple generation installs.
- **Still Open**: Concurrency real-model evaluation on a machine with sufficient free memory for two distinct runners or pool members.

---

### Priority 4: Measurements, Baselines & Quality Sweeps

Verification sweeps, cross-engine KL proofs, and power captures.

#### 1. Cross-Engine KL Verification (`qwen36`)
- **Status**: `qwen38` KL row is frozen (forward KL mean 0.000788 nats, below MLX floor; `docs/verification/kld_mlx_affine-qwen38.json`).
- **Still Open**: `qwen36` row is pinned in `scripts/kld_mlx_affine.py` but blocked on downloading `qwen36.gturbo` (~19 GB) and reference `mlx-community/Qwen3.6-35B-A3B-4bit` (~18 GB).

#### 2. Missing Quality & Memory Oracle Baseline Rows (`museglimmer`, `minimax`)
- **Status**: Mistral (9.3971), Bonsai-2 (5.8134), Bonsai (8.3554), TinyLlama (17.6084), and Qwen3-MoE (14.5988) quality and memory rows are frozen and asserted.
- **Still Open**: `museglimmer` and `minimax` gate files exist but installs are not currently on disk.

#### 3. Power Profile Sweep Across Remaining Catalog Rows
- **Status**: Ternary, Qwen38, Bonsai-2, and Qwen3-VL rows are captured under unconstrained or `COOLING=max` conditions in `docs/POWER_BASELINE.md`. Clean same-architecture 2-bit vs 4-bit pairing is recorded (1.60x J/token at 67% throughput).
- **Still Open**:
  - `ornith9b` re-pull (8.9 GB) and `ornith35b` re-pull (18 GB) to capture missing power rows.
  - Rate-cap sweep (`ARMS=default,30,20,15,10` under `COOLING=max` on gemma4).

#### 4. `qwen4_exp` QSA Post-change Real-Checkpoint Gate
- **Objective**: Re-run the above-budget sparse-vs-force-dense production comparison after GPU Top-K; capture QSA quality and memory gates before claiming a speedup or changing release status.
- **Why Open**: Post-change real-checkpoint throughput, quality, and memory have not been measured. `qwen4-reap288.gturbo` (68G) must be re-pulled.

---

## Artifact State & Disk Usage

Disk space is a key constraint for downloading large checkpoints and running cross-engine KL reference dumps.

Re-derived from `ls ~/models` and `ls ~/.turbospark/models` on **2026-09-20** (~20 GiB available). Do not start another large pull without reclaiming space or an explicit storage decision.

| Path | Size | Status / Associated Targets |
|---|---|---|
| `~/models/qwen38-27b.gturbo` | 14G | Backs `qwen38_{quality_gate,memory_oracle}`, KL row, steering probes, power rows |
| `~/models/ternary27b.gturbo` | 7.1G | Backs `ternary_{quality_gate,memory_oracle}`, power rows (`docs/POWER_BASELINE.md`) |
| `~/.turbospark/models/text/bonsai2.gturbo` | 7.6G | Backs `bonsai2_{quality_gate,memory_oracle}` (perplexity 5.8134, peak 660 MiB), power row |
| `~/.turbospark/models/text/qwen36.gturbo` | 19G | MoE vision combined install; tower-parity and CLI gated |
| `~/.turbospark/models/image/z-image-turbo-mlx-8bit.gturbo` | 6.2G | Validated 8-bit native image generation install |

**Missing from disk** (re-pull before dependent item can run):
- `museglimmer-30b.gturbo` (15G) -- museGlimmer steering probe + gates
- `ornith9b.gturbo` (8.9G) -- power row
- `ornith35b.gturbo` (18G) -- power row
- `qwen4-reap288.gturbo` (68G) -- P4.4 post-change QSA real-checkpoint throughput, quality, and memory gates
- `minimax-m2-q4km.gturbo` (129G) -- release gates (repetition blocker stands)

Storage preflights should retain at least 20 GiB headroom.

---

## Guiding Principles

1. **End-to-end gates decide everything**: End-to-end throughput, quality, and memory determine if an optimization ships, not isolated microbenchmarks.
2. **Bytes-per-token is the primary metric**: Decode is expert-I/O bound. Latency and energy optimizations reduce to reading fewer bytes, hiding reads better, or avoiding copying bytes that are already addressable.
3. **Lossless repack, always**: Installers shuffle quantized bytes without re-quantizing. Only upcast F32 norms/routers in GGUFs are transcoded.
4. **Enumerate architectures; do not abstract prematurely**: Shared streaming engine + explicit per-family modules + per-model manifests.
5. **Wasted power first, throttled power second**: Eliminate spin-waits and redundant compute before adding user-facing throttle knobs.

---

## Do Not Revisit (Measured Dead Ends)

1. ~~**Quantized KV Cache**~~: Reversed 2026-09-06. Built as opt-in `--kv-bits off|2|3|3.5|4` using Lloyd-Max codebooks and random Hadamard rotations (`docs/TRUBOQUANT.md`).
2. **Cold mmap as a Replacement for Streaming**: `pread` is strictly superior for cold uncached experts (74.8s vs 2.5s prefill). `mmap` is only used for warm residency (`docs/EXPERT_RESIDENCY.md`).
3. **RDADVISE as Default**: No stable production benefit.
4. **Expert Prefetch / Speculation**: Two predictors measured negative. Copied expert IDs hit 7%. PILOT router lookahead hits 70.6% recall but scales total bytes read above demand path (1.03x-1.85x) at high cache hit rates (`docs/EXPERT_ROUTING.md`).
5. **Expert Pread Tuning**: `MISS_READ_CHUNK_BYTES` (840 KiB) and `POOL_THREADS` (8) are at measured local optima.
6. **Deferred End-of-Token Wait**: Max theoretical gain 0.25 ms/token (1.4%); bottlenecked by layer dependencies.
7. **GPU-Side Router Top-K**: Host top-k overhead is 0.13 ms/token; GPU top-k only relocates synchronization.
8. **Buying Back Hit-CB Overlap in Kernel**: Determinism fix costs <2% throughput; modifying vendored kernel not justified.
9. **Monolithic Mega-Fusions (`fused.metal`)**: Host CPU encode is ~1 ms/token post cache-key fix; no headroom.
10. **Offset-Sorted Reads / Fine-Grained Read Dispatch**: Slower or nondeterministic.
11. **Domain-Restricted Expert Sets (pruned / pinned)**: 95% of routed mass touches ~67 of 128 experts across domains; static pruning damages quality (`docs/EXPERT_ROUTING.md`).
12. **Sub-4-bit Experts as an Efficiency Win**: IQ3_XXS / IQ4_NL is 35% slower and roughly doubles joules/token (GPU codebook dequant bound); valid only as a memory/disk tradeoff (`docs/POWER_BASELINE.md`).
13. **Staging `x` in `dequant_int4_gemm_mma`**: 3.3x to 5.9x slower at every width; penalty grows with B.
14. **The dequant loader and `kMmaTile` as levers on the matrix kernel**: `kMmaTile` refuted by arithmetic (`N/B` scaling); loader refuted by measurement (`FC_MMA_SKIP_DEQUANT` still 3x slower than MLX).
15. **Chunked WY-representation gated DeltaNet prefill, and GDN threadgroup staging**: Flash-linear-attention chunking takes 2x FLOPs of blocked sequential. Threadgroup staging targets only 4.66% of prefill traffic.
16. **Four-SIMD-group re-tile of `dequant_int4_gemm_mma` (PF-02 Step 7)**: 128-thread re-tile measured 2.10x slower than exact GEMV baseline. Staging `x` still loses inside wide shape, and 4 SIMD groups does not move the matrix path bottleneck (`docs/BATCHED_PREFILL.md`).

---

## Cross-Cutting Rules & Definition of Done

### Definition of Done per Phase
Every new feature or model bring-up requires:
1. Quality harness verified (perplexity delta within floor or explicitly accepted).
2. Frozen-protocol benchmarks recorded (`turbospark-bench`).
3. Power profile captured with `scripts/power.sh` on AC.
4. Memory oracle green with peak footprint asserted.
5. Win demonstrated under interleaved-pairs testing on quiet machine.

### Descoped Components
- Upstream Swift UI: Out of scope (superseded by native `TurboSparkApp` desktop application and `.app`/DMG release packaging).
- `prefill.metal` GPU tile pipeline: Descoping retained; chunked prefill driver reuses standard kernels.
- `logit.metal` `sample` kernel: Host sampling via `crates/selection` is standard.
