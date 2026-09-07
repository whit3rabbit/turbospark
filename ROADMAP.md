# Roadmap

The forward-looking roadmap and prioritized task tracker for this engine, last reconciled against the tree on 2026-09-07. All core port phases (Q, P1, G, S, P2, M1-M5) are complete and green. This document functions as an active TODO list for forward engineering, measurements, and architectural bring-ups.

All completed work, historical milestones, and landed features have been removed to focus strictly on remaining tasks.

---

## Current Status

- **Test Suite**: 2,090 tests declared workspace-wide, of which 1,953 run under the standing gate and 137 are `#[ignore]`d (the checkpoint downloads, memory oracles, quality gates, sensitivity proof, cross-engine dumps, and offline benchmarks).
- **Architectures**: 8 `ModelFamily` variants running across 20 curated catalog rows (`gemma4`, `qwenGdnMoe`, `llama`, `qwen3moe`, `gptOss`, `museGlimmer`, `qwenGdnDense`, `qwen4Exp`). A ninth variant, `deepseekV4Flash`, is declared and scaffolded only.

---

## Prioritized Task Backlog

### Priority 0: Immediate / Startable Now

Low-friction, high-impact fixes, unblocked measurement runs, or low-hanging symmetry tasks that can be executed immediately.

#### 1. Turn Decoder `finish()` Symmetry on CLI and FFI
- **Objective**: Server calls `runtime::TurnSplitter::finish` to flush trailing text and stop-token tool calls, but CLI and FFI loops do not call it yet (`DEVIATIONS.md`). Symmetrically wire `finish()` across all generation consumers so withheld text (`held_text` from DeepSeek) and pending tool calls flush properly.
- **Why Open**: Was previously inert because CLI and FFI ran with empty tool allowlists, but creating symmetry prevents subtle output dropping when tool execution expands.
- **Files to Touch**:
  - `crates/cli/src/generate/mod.rs` (invoke `split.finish()` after decode loop)
  - `crates/ffi/src/generate/mod.rs` (invoke `split.finish()` after decode loop)
  - `crates/runtime/src/turn_stream.rs` (verify finish event propagation)
  - `docs/STREAMING.md`, `DEVIATIONS.md`

#### 2. Cancellable Model Installation (`ts_install_cancel`)
- **Objective**: `ts_install` currently blocks its thread for the whole streaming walk; GUI Cancel only detaches observation while the download continues in the background (`DEVIATIONS.md`). Thread an atomic cancellation flag through `catalog::install` and expose `ts_install_cancel` in the C ABI and Swift package.
- **Why Open**: Required for genuine user cancellation in `TurboSparkApp` without background resource leaks or file write collisions.
- **Files to Touch**:
  - `crates/catalog/src/install.rs` (thread cancellation atomic through download and extract loops)
  - `crates/ffi/include/turbospark.h` (declare `ts_install_cancel`)
  - `crates/ffi/src/c_surface.rs` / `crates/ffi/src/install.rs` (expose C FFI cancel endpoint)
  - `swift/TurboSpark/Sources/TurboSpark/ModelCatalog.swift` (wrap cancellation in Swift library)
  - `swift/TurboSparkApp/Sources/TurboSparkApp/ModelInstallView.swift` / `CatalogSheet.swift` (wire UI Cancel button)

#### 3. Prefill Energy Capture
- **Objective**: Run `ARMS=seq,chunked scripts/power.sh` on AC quiet machine (~12 min, sudo) to capture baseline chunked prefill power and J/tok.
- **Why Open**: Tooling was unblocked (`turbospark-bench --prefill-chunk off|auto|N` and `scripts/power.sh seq|chunked` wired), but clean baseline row run is still owed.
- **Files to Touch / Run**:
  - `scripts/power.sh` (execute benchmark)
  - `docs/POWER_BASELINE.md` (record resulting rows)

#### 4. Step 6 Batched GEMV Throughput A/B Re-run
- **Objective**: Re-run throughput A/B on a quiet machine via `turbospark-bench --prefill-chunk auto` with `TURBOSPARK_BATCHED_GEMV=1` exported, polling `pgrep -x rustc` throughout to guard against mid-run build contamination.
- **Why Open**: Direction confirmed twice over (never slower on Gemma 4 install), but exact magnitude was invalidated by concurrent build processes.
- **Files to Touch / Run**:
  - `crates/bench/` (`turbospark-bench`)
  - `docs/BATCHED_PREFILL.md` (freeze verified throughput speedup)

#### 5. `qwen4_exp` Chunked Prefill Throughput & Pread Verification
- **Objective**: Difference two `TURBOSPARK_PHASES=1` runs' `expert io (pread)` buckets via `turbospark-check` (short prompt vs long prompt at same `--max-new`) to test if prefill is pread-bound.
- **Why Open**: Seventh `ChunkedPrefillRunner` landed and is bit-identical, but throughput numbers remain unmeasured.
- **Files to Touch / Run**:
  - `crates/cli/src/bin/check.rs`
  - `crates/runtime/src/families/qwen4/prefill.rs`
  - `docs/QWEN4_EXP.md`

#### 6. Real-Model Smoke for TurboQuant `--kv-bits`
- **Objective**: Run `turbospark-check --kv-bits 2|3|3.5|4` on real installs and record sanity output and perplexity checks.
- **Why Open**: Pipeline and Metal kernels landed across all 7 families, but real-model execution has not been validated against real hardware from the main worktree.
- **Files to Touch / Run**:
  - `crates/bench/tests/kv_quant_probe.rs`
  - `docs/TRUBOQUANT.md`

---

### Priority 1: Near-Term Core Engine & Infrastructure

High-leverage engine improvements, memory policy unifications, and front-end wirings.

#### 1. Vision Memory Sidecar Parity & Memory Oracle Arms
- **Objective**: Wire sidecar-aware test arms into `crates/runtime/tests/vision_tower_parity.rs` and `crates/bench/tests/vision_memory_oracle.rs` to measure mlx-vlm cosine parity and multi-page memory ceilings through a sidecar-attached trunk.
- **Why Open**: Sidecar format and runtime attachment landed, but the two real-model vision gates do not yet have sidecar-aware arms.
- **Files to Touch**:
  - `crates/runtime/tests/vision_tower_parity.rs` (add sidecar attach test arm)
  - `crates/bench/tests/vision_memory_oracle.rs` (add sidecar memory assertion)
  - `docs/VISION.md`

#### 2. `--vision-sidecar auto` Front-End Resolution
- **Objective**: Wire catalog-based automatic sidecar resolution by family and hidden size (via `catalog::resolve_vision_sidecar`) to the CLI and server front ends.
- **Why Open**: Backend resolution exists in `crates/catalog/src/resolve.rs`, but CLI and server currently only accept explicit `--vision-sidecar <PATH>`.
- **Files to Touch**:
  - `crates/cli/src/args.rs`
  - `crates/server/src/args.rs`
  - `crates/catalog/src/resolve.rs`

#### 3. Server Multimodal Chunked Prefill
- **Objective**: Allow server image completion endpoints to use chunked prefill. CLI and FFI already chunk image prompts, but the server image path does not chunk.
- **Why Open**: Image prompts generate >1,000 merged vision tokens; chunking prefill on the server is critical to prevent request stalls.
- **Files to Touch**:
  - `crates/server/src/completions.rs`
  - `crates/server/src/handler/`

#### 4. MTP / DFlash2 Verify Pass for Multimodal Prompts
- **Objective**: Make speculative verify passes vision-aware (handling image token injection) or enforce an explicit open-time refusal when both an MTP head and vision sidecar are active.
- **Why Open**: Verify pass currently assumes text tokens only; no install currently combines both, but combination is unhandled.
- **Files to Touch**:
  - `crates/runtime/src/speculative.rs`
  - `crates/runtime/src/families/qwen/mtp.rs`
  - `crates/runtime/src/families/qwen/dflash.rs`
  - `docs/VISION.md`

#### 5. Mapped Residency Eviction Benchmark & Policy Unification
- **Objective**: Measure paging overhead and fault costs when OS reclaims clean mapped pages under memory pressure. Unify slot cache policy with mapped expert residency: pick residency mode first (`mapped` vs `streamed`), then slot count only if `streamed` is active. Expose `--expert-residency auto|streamed|mapped`.
- **Why Open**: Mapped residency is landed and measured, but requires automated selection based on system memory headroom.
- **Files to Touch / Create**:
  - `crates/bench/tests/mapped_residency_eviction.rs` [NEW]
  - `crates/invocation/src/options.rs`
  - `crates/cli/src/args.rs`
  - `crates/server/src/args.rs`
  - `crates/runtime/src/runner.rs`
  - `crates/model-io/src/expert_cache_policy.rs`
  - `docs/EXPERT_RESIDENCY.md`

#### 6. Exact Rejection Sampling for Speculation (T > 0)
- **Objective**: Implement Leviathan/Chen algorithm on shaped distributions for non-greedy sampling during speculative verification.
- **Why Open**: Current speculative verification only supports greedy decoding (T = 0); non-greedy sampling requires distribution-preserving rejection sampling.
- **Files to Touch**:
  - `crates/selection/src/` (rejection sampling algorithm)
  - `crates/runtime/src/speculative.rs`
  - `crates/runtime/src/speculation_policy.rs`
  - `docs/SPECULATIVE_DECODING.md`

#### 7. Server Request Queue & Fairness (Option 1)
- **Objective**: Implement request FIFO queue with streaming-aware fairness and cancellation handling in `turbospark-server`.
- **Why Open**: Currently single-runner concurrency relies on mutex serialization and session-pool KV reuse; request queueing provides fairness under high client concurrency.
- **Files to Touch / Create**:
  - `crates/server/src/queue.rs` [NEW]
  - `crates/server/src/server.rs`
  - `crates/server/src/chat.rs`

---

### Priority 2: Architecture Bring-ups & Kernel Scaling

Adding missing high-demand model families, specialized Metal kernels, and architectural extensions.

#### 1. `qwen4_exp` GPU Top-K for Block Selection
- **Objective**: Above `index_budget`, QSA block scoring commits and waits on the host for `compute::select_blocks` once per QSA layer per token (12 host commits per token). Profile whether this is a bottleneck, and implement a GPU top-k kernel to eliminate host-device synchronization.
- **Why Open**: Host-side top-k was wired for bring-up; GPU kernel removes synchronization points during decode.
- **Files to Touch / Create**:
  - `crates/gpu/src/shaders/qsa_topk.metal` [NEW]
  - `crates/gpu/src/` (pipeline dispatch)
  - `crates/runtime/src/families/qwen4/attn.rs`
  - `docs/QWEN4_EXP.md`

#### 2. Dense `qwen3` / `qwen2.5` Architecture Bring-up (Usage-Weighted Priority)
- **Objective**: Bring up dense Qwen family (0.6B to 32B, Coder, QwQ, R1 distills) which forms the primary backbone of user-downloaded GGUF repositories.
- **Why Open**: Our qwen family currently covers GDN and MoE lines, but standard dense Qwen is missing from the architecture registry.
- **Files to Touch / Create**:
  - `crates/model-io/src/arch_config/family.rs`
  - `crates/model-io/src/manifest.rs`
  - `crates/repack/src/`
  - `crates/runtime/src/families/qwen_dense/` [NEW]
  - `docs/NEW_MODEL.md`, `docs/MODEL_FAMILY.md`

#### 3. `deepseek2` Architecture Support (High-Leverage Multi-Model Unlock)
- **Objective**: Implement Multi-head Latent Attention (MLA) bring-up to unlock Kimi K2.5, Kimi K2.6, GLM-4.7-Flash, and Mistral-Large-3-675B under one `deepseek2` architecture string.
- **Why Open**: Recognition-only in registry today; single highest-leverage architectural unlock across the catalog.
- **Files to Touch / Create**:
  - `crates/gpu/src/shaders/mla.metal` [NEW]
  - `crates/gpu/src/mla.rs` [NEW]
  - `crates/runtime/src/families/deepseek2/` [NEW]
  - `crates/model-io/src/arch_config/family.rs`

#### 4. `gpt-oss-120b` (MXFP4 GGUF) Evaluation under Mapped Residency
- **Objective**: 36 layers, 128 experts top-4, 12.6 MiB stride, 59 GiB total. Evaluate under mapped expert residency to test SSD throughput limits on large models.
- **Why Open**: Model fits unified memory on 64GB+ Macs; benchmarks evaluate page cache bounds vs SSD read bandwidth.
- **Files to Touch**:
  - `crates/model-io/src/manifest.rs`
  - `crates/runtime/src/families/gptoss/`
  - `crates/catalog/src/`

#### 5. MoE Drafter Ingestion & Conversion
- **Objective**: Build ingest/repack for MoE MTP heads (e.g. Ornith 35B with 256 per-expert tensors) or adapt DFlash2 block drafters for MoE architectures.
- **Why Open**: Speculative drafters currently only operate on dense models (`qwen38-27b`).
- **Files to Touch**:
  - `crates/repack/src/`
  - `crates/runtime/src/families/qwen/mtp.rs`
  - `crates/runtime/src/families/llama/`
  - `docs/SPECULATIVE_DECODING.md`

#### 6. Step 4 Batched Attention Kernel
- **Objective**: Widen `attention_decode_partial` to hold M query rows per KV chunk with per-row online-softmax state generalized to M rows. Scope specifically on an attention-dominant family (e.g. dense Llama/Gemma).
- **Why Open**: Low value on GDN-heavy Qwen (only 2.6% of prefill), but valuable for pure attention transformers.
- **Files to Touch**:
  - `crates/gpu/src/shaders/attention.metal`
  - `crates/gpu/src/attention.rs`
  - `crates/runtime/src/families/llama/` or `crates/runtime/src/families/gemma4/`
  - `docs/BATCHED_PREFILL.md`

#### 7. SigLIP-Class Vision Tower Bring-up (`gemma4_unified`)
- **Objective**: Implement SigLIP-class vision tower encoder and intake for `gemma4_unified`. Gemma 4 text trunk already runs here; intake currently drops ~815 vision tensors.
- **Why Open**: Nearest-term VLM candidate to expand multimodal support beyond Qwen3-VL.
- **Files to Touch / Create**:
  - `crates/vision-io/src/siglip.rs` [NEW]
  - `crates/runtime/src/vision/`
  - `crates/repack/src/gemma4_checkpoint/classify.rs`
  - `docs/VISION.md`

#### 8. Missing Tool-Calling Markups
- **Objective**: Implement structured tool-call decoders for formats found in modern checkpoints: GLM XML `<arg_key>/<arg_value>`, MiniMax namespaced `<minimax:tool_call>`, Mistral `[TOOL_CALLS]`, and Kimi K2 section markers `<|tool_calls_section_begin|>`.
- **Why Open**: Parser currently covers ChatML, DeepSeek, Harmony, and Gemma channels only.
- **Files to Touch**:
  - `crates/tokenizer/src/structured_decoder/`
  - `crates/tokenizer/src/chat_template.rs`
  - `docs/TOOL_CALLING.md`

#### 9. Vision Ingestion from HF Hub in `turbospark-model pull`
- **Objective**: Fix production intake so `turbospark-model pull` parses vision config instead of hardcoding `vision: VisionConfig::NONE` in `parse_qwen_gdn_dense_config`.
- **Why Open**: Currently vision installs on disk were streamed by test harnesses, not through `turbospark-model pull`.
- **Files to Touch**:
  - `crates/repack/src/qwen36_config.rs`
  - `crates/catalog/src/install.rs`

---

### Priority 3: Long-Term Extensions & Platform Expansion

System architecture extensions, platform ports, and developer tooling.

#### 1. Remote Plugin Marketplace & Registry Indexing (`TurboSparkApp`)
- **Objective**: Wire network discovery and remote repository manifest fetching in `PluginMarketplaceManager` beyond local directories.
- **Why Open**: Local plugin management and manifest enable cascades are complete; remote registry indexing allows community plugin discovery.
- **Files to Touch**:
  - `swift/TurboSparkApp/Sources/TurboSparkApp/Plugins/PluginMarketplaceManager.swift`
  - `swift/TurboSparkApp/Sources/TurboSparkApp/Plugins/PluginManifest.swift`
  - `swift/docs/SWIFT_PLUGINS.md`

#### 2. Multi-Direction Steering & Automated Alpha Calibration
- **Objective**: Support simultaneous application of multiple steering vectors with per-vector scales and layer masks; implement automated alpha calibration to detect semantic steering collapse thresholds.
- **Why Open**: Single-direction steering is shipped; multi-direction and automated tuning improve developer ergonomics.
- **Files to Touch / Create**:
  - `crates/runtime/src/steering.rs`
  - `crates/runtime/tests/activation_capture.rs` [NEW]
  - `crates/invocation/src/options.rs`
  - `docs/OBLITERATION.md`

#### 3. Batched Sub-Byte GEMMs for Bonsai / Ternary Speculation
- **Objective**: Implement batched INT1 and INT2 GEMM kernels to unblock speculative verification for Bonsai-27B and Ternary-Bonsai.
- **Why Open**: Decode GEMVs exist; batched GEMMs are required for speculative verification.
- **Files to Touch / Create**:
  - `crates/gpu/src/shaders/gemv_int1.metal`
  - `crates/gpu/src/shaders/gemv_int2.metal`
  - `crates/gpu/src/gemv_int1.rs`
  - `crates/gpu/src/gemv_int2.rs`

#### 4. DeepSeek-V4-Flash Metal Kernels & Feasibility (Scaffolded)
- **Objective**: Port CSA/HCA attention, unrolled mHC Sinkhorn, and sub-3bit GEMV Metal kernels (`dsv4.metal`), assessing 106.9 GB peak RSS memory feasibility.
- **Why Open**: Architecture is scaffolded; full implementation requires large unified memory (128 GB Mac).
- **Files to Touch / Create**:
  - `crates/gpu/src/shaders/dsv4.metal` [NEW]
  - `crates/gpu/src/`
  - `crates/runtime/src/families/dsv4/` [NEW]

#### 5. Linux Backend (Portable Architecture)
- **Objective**: Implement `io_uring` + `O_DIRECT` streaming I/O layer paired with portable CPU/Vulkan compute backend and cgroup memory limit support on Linux.
- **Why Open**: Current runtime and streaming layers are Metal and macOS unified memory optimized.
- **Files to Touch / Create**:
  - `crates/streaming/src/linux_uring.rs` [NEW]
  - `crates/compute/src/vulkan/` [NEW]

#### 6. Server Multi-Runner Pool (Option 2)
- **Objective**: Support N active `RealForwardRunner` instances for concurrent request serving where VRAM/RAM permits.
- **Why Open**: Multiplexed session state (Option 3) is complete; full multi-runner pool allows parallel batch compute on high-memory hardware.
- **Files to Touch**:
  - `crates/server/src/session_pool.rs`
  - `crates/server/src/server.rs`

---

### Priority 4: Measurements, Baselines & Quality Sweeps

Verification sweeps, cross-engine KL proofs, and power captures.

#### 1. Cross-Engine KL Verification (`qwen38`, `qwen36`)
- **Objective**: Add `qwen38` and `qwen36` to `scripts/kld_mlx_affine.py` test suite against upstream MLX reference outputs.
- **Why Open**: Verification script exists, but table entries for these models need to be frozen.
- **Files to Touch / Run**:
  - `scripts/kld_mlx_affine.py`
  - `docs/BENCHMARKS.md`

#### 2. Missing Quality & Memory Oracle Baseline Rows
- **Objective**: Freeze quality gate and memory oracle rows for `bonsai27b`, dense `llama` (Mistral 7B / TinyLlama), and `qwen38` external follow-ups.
- **Why Open**: Requires downloading reference checkpoints and running frozen protocol sweeps.
- **Files to Touch / Run**:
  - `crates/bench/tests/`
  - `docs/BENCHMARKS.md`

#### 3. Power Profile Sweep Across Remaining Catalog Rows
- **Objective**: Capture baseline power and J/tok for `qwen3_5` 27B variants, `ornith9b`, and `ornith35b`.
- **Why Open**: Requires re-pulling Ornith checkpoints to disk before executing `scripts/power.sh`.
- **Files to Touch / Run**:
  - `scripts/power.sh`
  - `docs/POWER_BASELINE.md`

---

## Artifact State & Disk Usage

Disk space is a key constraint for downloading large checkpoints and running cross-engine KL reference dumps.

Re-derived from `ls ~/models` and `ls ~/.turbospark/models`:

| Path in `~/models/` | Size | Status / Associated Targets |
|---|---|---|
| `gemma4.gturbo` | 13G | PINNED: smoke, memory oracle, sensitivity proof, mapped residency |
| `ternary27b.gturbo` | 7.1G | `ternary_{quality_gate,memory_oracle}` |
| `qwen38-27b.gturbo` | 14G | `qwen38_{quality_gate,memory_oracle}`, steering baseline |
| `qwen38-27b-mtp.gturbo` | 14G | MTP speculative validation. Second copy sits at `~/.turbospark/models/qwen38-27b-mtp.gturbo` |
| `qwen38-27b-dflash2.gturbo` | 15G | DFlash2 block drafter validation |
| `qwen38-gguf.gturbo` | 15G | Qwen 3.8 27B via GGUF intake |
| `steering-vectors/` | ~3M | `ocean` and `register` legacy vectors for regression tests |
| `gguf-ref/`, `qwen38-mtp-ref/`, `skill-state-probe/` | -- | Reference and probe sidecars |
| `qwen38-27b-vision.gturbo` | 15G | Combined vision trunk + tower install |
| `vision-probe-qwen38/` | 4.8G | `mlx-community/Qwen3.8-27B-4bit` tower (revision `3e6447f0`) |
| `vision-probe/` | 879M | `prism-ml/Bonsai-27B-mlx-1bit` tower |

Store models in `~/.turbospark/models/`:

| Path in `~/.turbospark/models/` | Size | Status / Associated Targets |
|---|---|---|
| `qwen4-reap288.gturbo` | 68G | Qwen3.8-Flash-Next REAP-288 (`top_k_experts=10`). Backs `qwen4exp_{quality_gate,memory_oracle}` |
| `qwen3moe.gturbo` | 17G | `llama` flow's `Qwen3Moe` half. `qwen3moe_{quality_gate,memory_oracle}` |
| `gptoss-20b.gturbo` | 11G | `gptoss_{quality_gate,memory_oracle}`, steering validation, mapped residency |
| `qwen38-27b-mtp.gturbo` | 14G | Duplicate of `~/models/qwen38-27b-mtp.gturbo` |

**Missing from disk** (require re-pull before dependent benchmark/oracle tasks can run):
- `museglimmer-30b.gturbo` (15G)
- `ornith9b.gturbo` (8.9G)
- `ornith35b.gturbo` (18G)
- `ornith35b-gguf.gturbo` (34G)
- `mistral7b.gturbo`
- `llama3-8b-instruct.gturbo`

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
