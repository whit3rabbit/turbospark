# Roadmap

The forward-looking roadmap and task tracker for this engine, last reconciled against the tree on 2026-08-30. All core port phases (Q, P1, G, S, P2, M1-M5) are complete and green. This document functions as an active TODO list for forward engineering, measurements, and architectural bring-ups.

---

## Current Status (2026-08-30)

- **Test Suite**: **1,758 tests declared workspace-wide, of which 1,627 run under the standing gate and 131 are `#[ignore]`d** (the checkpoint downloads, the memory oracles, the quality gates, the sensitivity proof, the cross-engine dumps, and offline benchmarks). Re-counted 2026-08-30 with `cargo test --workspace -- --list` and `--list --ignored` after landing Mapped Expert Residency, on macOS, so the macOS-only crates are included. The prior count on this same line (1,770 / 1,640 / 130) was from earlier the same day; this repo's own house rule is not to trust a prose test count without re-deriving it, so this is the current one rather than an increment on the old one. Strict formatting, clippy and cross-target checks clean.
- **Architectures**: 7 `ModelFamily` variants running across 17 curated catalog rows (`gemma4`, `qwenGdnMoe`, `llama`, `qwen3moe`, `gptOss`, `museGlimmer`, `qwenGdnDense`).
- **Recent Landings**:
  - **Vision (`qwen3_5`), M-V0 through M-V9**: an image reaches a generated token from the COMMAND LINE and from BOTH server endpoints. The tower agrees with mlx-vlm at its own FP16 floor, this port builds the spliced prompt byte-identically to the reference processor, and `--image` / `--image-batch` transcribe a real page. M-V9 landed 2026-08-29 (multi-page memory oracle, the last two NaN-safe parity guards, FP16 overflow capture). See `docs/VISION.md`'s "What is not built" section for the full record; nothing is open there now.
  - **Batch INT4 GEMM Row Blocking (`dequant_int4_batch.rs` & `dequant_int4_mma.metal`)**: Row-blocked dispatch wired for M-row batch GEMMs with optimal tile dispatch (`R=1, 2, 4`), register limit queries, and crossover points documented in `docs/BATCHED_PREFILL.md` and `docs/BENCHMARKS.md`.
  - **Reasoning Effort & Thinking Token Protocol**: Multi-dialect support for `--reasoning` / `reasoning_effort` across CLI and server (`off`, `low`, `medium`, `high`, `xhigh`), ChatML/Gemma thinking extraction, and Swift UI integration.
  - **Expert Disk I/O & Bypass Telemetry**: Disk I/O tracking and cache-bypass telemetry in `crates/streaming` (`MFERENCE_PILOT_PROBE` validation and pread streamer metrics).
  - **Prefix KV Reuse (cached-prompt continuation)**: a turn continues from the previous turn's KV wherever the two prompts agree, instead of resetting and re-prefilling the whole transcript. `runtime::kv_prefix` plus one defaulted `LogitProducer` seam. Measured on the real 26B: prefill **1.777s -> 0.153s (11.6x)** on a transcript-shaped prompt, generated tokens byte-identical to the re-prefilled reference; in the real `--chat` REPL, 13/33 then 29/49 tokens continued.
  - **Native macOS App (`TurboSparkApp`) & Release Automation**: Full Swift desktop app with multi-chat persistence, project/agent system with tool execution and permissions engine, reasoning controls, model management, document attachments, and official `.app`/DMG packaging automation (`scripts/make-app-bundle.sh`, `scripts/make-dmg.sh`, Homebrew cask).
  - **Swift Agent & Tool Execution Subsystem (2026-08-31)**: AGENTS.md parser and custom agent definition loader (`AppAgentDefinition`, `AgentParser`, `AgentManager`), autonomous subagent runner (`SubagentRunner`), custom tool execution engine (`CustomToolExecutor`), and built-in executors (`WebFetchExecutor` with SSRF protection, `TodoWriteExecutor` for project task management). Fully integrated with settings UI (`AgentsSettingsPaneView`) and verified across 352 passing unit and integration tests.
  - **Directional Weight Steering**: Runtime abliteration, ActAdd, clamping, and renorm shipped across 7 families (`docs/OBLITERATION.md`).
  - **Prefill Batching (PF-02)**: Steps 1-3 and 6 landed (1.54x prefill speedup on Gemma 4, `docs/BATCHED_PREFILL.md`). `--prefill-chunk` is wired as the default now (family-gated, falls back to sequential with no error on an unsupported install), and the chunked driver now runs every MoE-capable family: the dense half of `llama` and `muse_glimmer` (no router, no mid-layer commit), plus the MoE half of `llama` and `gpt-oss` (Step 1's per-layer commit and per-token routed loop, no new kernel needed). **Step 5's MXFP4 arm landed 2026-08-27 at 1.31x on `gpt-oss`** -- chosen over the GGUF arm by measurement, which inverted the order the item was written in. **The DENSE half of the qwen linear-attention flow landed 2026-08-29** (`families/qwen/prefill.rs`, `qwenGdnDense`, `qwen38-27b.gturbo`): the same Step 1 shape as dense `llama`/`muse_glimmer`, no new kernel and no new buffer at all, with the GDN recurrent state's correctness following from in-order per-token calls rather than any new machinery. Only the MoE half (`qwenGdnMoe`) remains unsupported. **Step 6 was then wired to that same family the same day at 1.86-2.13x prefill**, so the "no new buffer at all" clause above describes its DEFAULT arm alone; the seam's arm reuses the verify pass's M-row encoders and allocates a `BatchedScratch` lazily.
  - **Prefill Kernel-Quality Reference & Three Settled Dead Ends (2026-08-29)**: reviewed `jundot/omlx`'s `qwen35_prefill` custom kernels against this port. Its qmm is MLX's own `qmm_t_impl` re-tiled, so the 210.3 bar is STOCK MLX plus a tile sweep worth less than the noise band. `scripts/mlx_qmm_reference.py` replaces this document's extrapolated 2.2x/2.3x split with a same-session measurement: the width term saturates at **M=32** (not M=36 or M=64) and the gap decomposes into **1.50x kernel at equal width and 2.00x width**, 3.00x total, on gate/up. Three candidate levers were then closed by measurement rather than argument -- see "Do Not Revisit" 13-15.
  - **DFlash2 Block Drafter**: Complete block-diffusion speculative drafter for dense Qwen 3.8 (`docs/DFLASH2.md`).
  - **Ornith-1.5 Checkpoints**: 9B dense and 35B MoE in both GGUF and MLX INT4 formats.

---

## Active Tasks & Measurement Owed

In estimated cost/effort order:

| Task / Item | Cost / Dependencies | Details / Why Open |
|---|---|---|
| **PF-02 Step 6 Throughput A/B** | ~20 min, quiet machine, no code | Compare `MFERENCE_PREFILL_CHUNK=128 MFERENCE_ROUTED_BATCH=1` with vs without `MFERENCE_BATCHED_GEMV=1`. No longer gates the default-on wiring, which landed independent of this measurement (family support, not throughput, is what the default checks). |
| **Prefill Energy Capture** | ~12 min, sudo, quiet machine | Re-measure prefill Joules on `long-synthesis` prompt post-PF-02 chunking/batching. Batch with other `power.sh` runs. |
| **PF-02 Step 4 (Batched Attention Kernel)** | New Metal kernel, real risk. **LOW VALUE on `qwenGdnDense`; scope it on a family where attention is a real share.** | Widen `attention_decode_partial` to hold M query rows per KV chunk. Real new-kernel work: a new function-constant axis, per-row online-softmax state generalized to M rows, and register-pressure risk per `dequant_int4_gemm_simd`'s spill history. **On the dense qwen flow it targets 2.6% of prefill in the 16 of 64 layers that have attention at all** (the other 48 are gated DeltaNet), against a GEMM holding 85.4% -- so on THIS family it is the wrong term. oMLX ships an `fa256` prefill attention and it is 35 lines of `instantiate_kernel` over MLX's steel kernel, i.e. nothing to port. |
| **PF-02: Qwen Dense GEMV-to-GEMM Widening** | DONE 2026-08-29 | Landed as a WIRING pass, not the new-kernel work this row predicted: `families/qwen/batched_layers.rs` already owned all three M-row encoders for the MTP/DFlash2 verify, at the row convention the chunk driver writes, and `MAX_PREFILL_BATCH` IS `gpu::MAX_BATCH_ROWS`. Measured **1.86x to 2.13x** on prefill (21.40 -> 40.79 tok/s, interleaved pairs, warmup discarded, contended machine so a lower bound). Against oMLX's 210.3 tok/s that is now roughly a FIFTH rather than a tenth -- still not closed. What remains unbatched is ATTENTION, i.e. Step 4 above, which IS real new-kernel work. The batched arm is not byte-identical on a real install and that is a measured shape floor (`e8deb6c`), so its gate is the quality gate; the DEFAULT arm stays byte-identical. `qwenGdnMoe` (Ornith 35B) is still unattempted. |
| **Mapped Residency on MoE Flows** | DONE 2026-08-30 | Wired into `qwen`, `llama` (both `Llama` and `Qwen3Moe`), and `gptoss`; each family's named refusal was replaced rather than bypassed. See section 9's TODO for the verification detail and the one open gap (`qwen`'s own family has no real install left on this machine to verify against). |
| **Mapped Residency Eviction Benchmark** | Investigation / memory pressure test | Measure degradation/fault costs when OS reclaims clean mapped pages under memory pressure. Gates `auto` default and CLI flag. |
| **Phase-2 `top_k` Specialization** | Metal kernel tweak + test | Specialize `moe_phase2_down_reduce_k8` by `top_k` (saves 50% down-GEMV on `gpt-oss` top-4). Widen `constants_key`. |
| **MoE Speculative Refusal Strings** | Doc/string fix across 5 files | Update stale error messages claiming MoE has no batched kernel (the kernel shipped in `moe_batch.rs`). |
| **`qwen38` Cross-Engine KL** | Table entry in script + ref download | Add `qwen38` to `scripts/kld_mlx_affine.py` test suite against upstream MLX. |
| **`qwen38` External Follow-Up Benchmark** | Read-only, link-only | TerminalBytes (2026-08) benchmarked Qwen3.8 27B on a Mac Studio M3 Ultra / 256 GB: Ollama Q4_K_M at **14.0 tok/s** (vs qwen3.6 27B at 28.6 tok/s on the same machine); 1-bit Unsloth quant at **27 tok/s** in 6.7 GB but unusable for tool calling. Cross-checks the dense-row direction here (Qwen3.8 slower-per-token than qwen3.6, 1-bit breaks agentic) and adds hardware tiers this repo does not measure (M3 Ultra, Strix Halo, RTX 3090/5090) plus AC power draw (~64W GPU, 291W system mid-generation) and the 262k long-context claim. No action required; reference: <https://terminalbytes.com/run-qwen-3-8-27b-locally/>. |
| **`qwen38` Community Benchmark Target (oMLX)** | See the GEMV-to-GEMM row above for PP; freeze the MTP projection for the other half of TG | oMLX community numbers for the same Qwen3.8-27B 4-bit install define the "beat this" bar. **M3 Max 40c / 64 GB, MTPLX-Optimized-Speed 4-bit @ 4k ctx: PP 210.3 tok/s, TG 17.1 tok/s** (vanilla 4-bit on M2 Pro 32 GB reads TG 11.2 tok/s). This port's own M4 Max 40c / 36 GB row in `docs/BENCHMARKS.md` reads TG **18.6-21.1 tok/s** -- already at or above the MTPLX build's TG, on less RAM. **PP is measured and does NOT clear the bar, but the bar is now understood.** 21.31 tok/s at Step 1, **40.79** after Step 6's GEMV-to-GEMM widening, **42.1** after `FC_GEMM_R` row blocking, against 210.3. **The 210.3 is not an oMLX achievement to reverse-engineer**: its custom `qwen35_qmm.metal` calls MLX's own `qmm_t_impl` at swept tiles, and mlx-lm STOCK on this machine reads 195.4-201.7, i.e. within 4-7%. So the target is stock MLX and the tile sweep is inside the noise. `scripts/mlx_qmm_reference.py` decomposes the remaining gap on the same machine and the same yardstick: **1.50x kernel at M=16 and 2.00x width from M=16 to M=32**, where MLX's `c(M)` goes 0.289 -> 0.145 and is then FLAT to M=512 (its `ms` column is identical at M=16 and M=32, i.e. a `BM=32` tile paying for empty rows). The two terms multiply and neither pays alone: the width is unreachable at `MAX_BATCH_ROWS = 16`, and this port's own `c` is flat across M=2..16, so widening amortizes bytes that were never the cost. Kernel first, then re-price the width. Base TG already clears its half of the bar; the other open piece is freezing the MTP-layered projection: the standing native MTP drafter (1.44x at block 2, `docs/MTP.md`) projects speculative TG to roughly 27-30 tok/s on top of the base rate, worth measuring rather than leaving implicit. Mapped Expert Residency (section 9 below) does NOT apply here -- this is a dense install with no expert cache, so it is not the memory lever for this comparison. Reference: <https://share.google/KzY3rnCKaUxdYwrpz>. |
| **`qwen36` Cross-Engine KL** | Reference download + `pull qwen36` (~23 min) | Cross-engine KL for Qwen 3.6 MoE against MLX reference. |
| **`bonsai27b` Quality & Oracle Rows** | `pull bonsai27b` (~10 min) + run | Freeze quality gate and memory oracle rows for the 1-bit Bonsai model. |
| **Dense `llama` Quality-Gate Rows** | Reference answer suited to 7B + pull | Add frozen quality-gate rows for dense Mistral-7B / TinyLlama installs. |
| **`power.sh` Sweep for Remaining Models** | ~12 min each, sudo, quiet machine | Capture baseline power and J/tok for `qwen3_5` 27B variants, `ornith9b`, and `ornith35b`. |

---

## Immediate Next Actions (Startable Now)

1. **PF-02 Completion** (family widening and Step 5's MXFP4 arm are DONE; what is left is measurement plus one kernel):
   - ~~Measure MXFP4 `c(M)` on `moe_prefill_batch_bench.rs`~~ DONE 2026-08-27: 0.76/0.74/0.73/0.74 at M=2/4/8/16; the inferred 0.67 was 9% optimistic (`docs/BATCHED_PREFILL.md`).
   - ~~Separate the batched arm's 28% expert-miss drop~~ DONE 2026-08-27, and it REFUTED the reasoned attribution: the protect set is exonerated (misses 9,024 with it, 9,400 without) and the drop is real union dedup -- the one family-scoped exception to "the union saves nothing". Note the seam the task named did not exist on this family and had to be wired first (`docs/BATCHED_PREFILL.md`).
   - Run step 6 throughput A/B on a quiet machine.
   - Implement Step 4 batched attention kernel (widening of `attention_decode_partial`); real new-kernel work, own pass.
   - The GGUF (Q4_K/Q6_K) arm of Step 5 is deliberately NOT queued: measured, its ceiling is the un-batchable `pread` (37.1% of its prefill) rather than the kernel, and it would cost two kernels rather than one.
2. **Phase-2 `top_k` Specialization**:
   - Specialize down-reduce kernel for `top_k < 8`, updating shader constant key cache.
3. **MoE Drafter & Speculation Cleanup**:
   - Fix stale refusal strings in `mtp_state.rs`, `dflash_state.rs`, `speculation_policy.rs`.
   - Investigate ingestible MoE drafters (e.g. Ornith MoE MTP head conversion or lightweight block drafter).
4. **Streamable-MoE Bring-Up**:
   - Test and benchmark `gpt-oss-120b-MXFP4.gguf` under mapped residency (evaluates SSD vs page cache bound decode).

---

## Artifact State & Disk Usage (2026-08-29)

Disk space is a key constraint for downloading large checkpoints and running cross-engine KL reference dumps.

| Path in `~/models/` | Size | Status / Associated Targets |
|---|---|---|
| `gemma4.gturbo` | 13G | PINNED: smoke, memory oracle, sensitivity proof, mapped residency |
| `ternary27b.gturbo` | 7.1G | `ternary_{quality_gate,memory_oracle}` |
| `museglimmer-30b.gturbo` | 15G | `museglimmer_{quality_gate,memory_oracle}`, steering validation |
| `qwen38-27b.gturbo` | 14G | `qwen38_{quality_gate,memory_oracle}`, steering baseline |
| `qwen38-27b-mtp.gturbo` | 14G | MTP speculative validation |
| `qwen38-27b-dflash2.gturbo` | 15G | DFlash2 block drafter validation |
| `ornith9b.gturbo` | 8.9G | `ornith9b_{quality_gate,memory_oracle}`, llama.cpp KL |
| `ornith35b.gturbo` | 18G | `ornith35b_{quality_gate,memory_oracle}`, MLX KL, mapped residency target |
| `ornith35b-gguf.gturbo` | 34G | Q8_0 throughput A/B testbed |
| `gptoss-20b.gturbo` | 14G | **NOT under `~/models/`** -- it lives in `~/.turbospark/models/` (the `turbospark-model pull` store), verified 2026-08-27. `gptoss_{quality_gate,memory_oracle}`, steering validation, and the PF-02 Step 5 MXFP4 arm's throughput and byte-identity runs. |
| `steering-vectors/` | ~3M | `ocean` and `register` legacy vectors for regression tests |
| `qwen38-27b-vision.gturbo` | 15G | The ONLY install carrying a vision tower. `vision_tower_parity`, `vision_logit_dump`, the CLI's `--image` / `--image-batch` runs and the server's `real_backend_reads_an_image_sent_over_both_endpoints`. **Deliberately separate from `qwen38-27b.gturbo`**: that one backs the frozen quality-gate and memory-oracle rows, and ~0.9 GiB of tower would force a re-freeze for a component neither gate exercises. Do not merge them. |
| `vision-probe-qwen38/` | 4.8G | `mlx-community/Qwen3.8-27B-4bit`'s tower alone, pinned to revision `3e6447f0`, which is what `vision_tower_parity` pairs against. 4.8G for 879 MiB of tensors because `vision_tower.*` is not contiguous in this checkpoint and the fetch spans min..max offset (over-fetches, does not miss data). |
| `vision-probe/` | 879M | `prism-ml/Bonsai-27B-mlx-1bit`'s tower, F16. What `docs/VISION_PHASE0.md` items 3 and 4 were measured on. Contiguous, so 879M for 879 MiB. |

Two vision artifacts live OUTSIDE `~/models/` and are the larger half of the
feature's disk cost:

| Path | Size | Status |
|---|---|---|
| `~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit` | 15G | The FULL reference checkpoint, revision `3e6447f0`, needed by `scripts/kld_mlx_vlm.py` because that gate runs the reference's TRUNK as well as its tower. Re-downloadable. |
| `/tmp/vision-kld/` | 4.3G | The cross-engine working dump (`reference.f32` 1.29 GB, `port.f16` 646 MB, pixel and merger sidecars). Regenerable in ~4 minutes, so `/tmp` is correct for it -- unlike the two tower caches above, which a previous handoff lost to a `/tmp` clear. |

*Note*: If disk space is needed for reference dumps, check `target/` first (`cargo clean` often recovers 30+ GB).

---

## Detailed Feature Roadmaps & TODOs

### 1. Prefill Batching (PF-02)

- **Status**: Steps 1-3 (chunked prefill, batched routed pair) and Step 6 (batched resident GEMVs) shipped for Gemma 4. Measured 1.54x speedup on Gemma 4 `long-synthesis`. `--prefill-chunk` is wired as the default (2026-08-26): `RealForwardRunner::supports_chunked_prefill()` is the one predicate both the CLI's default routing and `ChunkedPrefillRunner::prefill_chunk`'s own refusal check, so a caller who never typed the flag cannot see a family it doesn't serve, and `MFERENCE_PREFILL_CHUNK`'s existing hard-fail-on-unsupported-family A/B-seam contract is unchanged. **Every MoE-capable family now serves chunked prefill (2026-08-27)**: `muse_glimmer` landed the same no-mid-layer-commit shape as dense `llama` (no router), and the MoE half of `families/llama/` (Mixtral, Qwen3MoE) plus `gpt-oss` landed Step 1 alone -- a per-layer command buffer for attention-and-router plus a per-token routed loop pipelined via a shared `RoutedSlot` module (`moe_prefill_pipeline.rs`, extracted from Gemma 4's driver) -- with NO new kernel needed, since Step 1 reuses the same layout-agnostic per-token dispatch the sequential decode path already uses. **Step 5's MXFP4 arm landed 2026-08-27 at 1.31x** on the real 20B install (`families/gptoss/moe_batch.rs` over `moe_prefill_batch_gguf.metal`), so TWO families now serve the batched routed pair: Gemma 4 on INT4-affine blobs and `gpt-oss` on MXFP4 ones. Wiring the second one exposed that the unwired MoE `llama` family had been SILENTLY IGNORING `MFERENCE_ROUTED_BATCH` rather than refusing it; that is a named refusal now. **The DENSE half of the qwen linear-attention flow landed 2026-08-29** (`families/qwen/prefill.rs`, `qwenGdnDense`): the same Step 1 shape as dense `llama`/`muse_glimmer`, no new kernel and no new buffer -- the recurrent GDN state's correctness follows from calling the existing per-token kernels in order rather than from any batched machinery, and the driver refuses an image prompt or an open drafter by name rather than growing a second embedding call site or silently starving a drafter's aux capture. Only the MoE half of qwen (`qwenGdnMoe`) remains unsupported. Verified byte-identical against the pre-wiring sequential path on real installs (`~/models/gemma4.gturbo`, `~/.turbospark/models/mistral7b.gturbo`, `~/models/museglimmer-30b.gturbo`, `~/.turbospark/models/gptoss-20b.gturbo`, a freshly-pulled `Qwen/Qwen3-30B-A3B-GGUF`, `~/models/qwen38-27b.gturbo`; greedy and sampled, stdout md5-identical both ways), plus the standing gemma4 chunked parity suite and new per-family parity suites (chunk-span sweep, cache-too-small-to-pipeline case where the family has a slot cache, MoE-still-refused case where applicable).
- **Reference**: `docs/BATCHED_PREFILL.md`.
- **Detailed TODO**:
  - [x] **Kernel-quality reference measured 2026-08-29** (`scripts/mlx_qmm_reference.py`, `docs/BENCHMARKS.md` "The reference curve, measured rather than inferred"). Replaces the extrapolated 2.2x/2.3x pair with 1.50x kernel + 2.00x width, and moves the saturation point to **M=32**. Both engines' M=1 baselines agree to 1.16x, which is what makes `c` comparable at all.
  - [ ] **Step 7 (matrix-path re-tile)**: the only untried lever on `dequant_int4_gemm_mma`. Four SIMD groups (`WM = WN = 2`, 128 threads) with `FC_MMA_STAGE_X` ON, changed TOGETHER -- the three sub-levers are not independent and each was measured to a dead end alone (Do Not Revisit 13, 14). Gate: `c_of_m_matrix_against_exact_at_qwen38_shapes`'s third column below 1.00. It stays PREFILL-ONLY whatever it measures, since the kernel is not bit-exact against the GEMV (AGENTS.md Gotcha 27), and `MAX_BATCH_ROWS = 16` still caps the width term regardless.
  - [ ] **Step 6 Throughput A/B**: Measure `MFERENCE_BATCHED_GEMV=1` on quiet machine.
  - [ ] **Prefill Energy Capture**: Profile J/token with `scripts/power.sh` on chunked prefill.
  - [ ] **Step 4 (Batched Attention Kernel)**: Widen `attention_decode_partial` to process M query rows per KV chunk. Deferred as real new-kernel work (new function-constant axis, per-row online-softmax state generalized to M rows, real register-pressure risk per `dequant_int4_gemm_simd`'s spill history), not attempted alongside the default-on wiring. **Measured LOW VALUE on `qwenGdnDense` (2026-08-29): 2.6% of prefill, in the 16 of 64 layers that have attention at all.** Pick the family before picking this item; see the Active Tasks row.
  - [ ] **Step 5 (GGUF Routed Pair Widening)**: Widen the BATCHED routed kernels (steps 2/3, `MFERENCE_ROUTED_BATCH`) to GGUF and MXFP4 block types if/when the throughput they add becomes a priority; Step 1's per-token routed loop already runs on every layout, so this is a speed lever, not a correctness gap.
    - **SCOPED BY MEASUREMENT 2026-08-27, and the order is the opposite of this item's title** (`docs/BATCHED_PREFILL.md`, "Step 5's two arms, measured before building either"). Do **MXFP4 (`gpt-oss`) first**: its routed pair is 61.4% of prefill GPU device time against Gemma's 38.2%, its un-batchable `pread` bucket is 8.2% against 25-37% (32 experts at top-4 give a 96.7% hit rate), and it is the ONLY family that reaches M=16 -- both 128-expert families cap at M=8 on `union(M) <= slot_count`. It also needs ONE block type for both phases.
    - [x] **MXFP4 arm DONE 2026-08-27**, and it measured **1.31x** on the real 20B install against Gemma 4's 1.19x for the same step -- the share arithmetic held. `crates/gpu/src/shaders/moe_prefill_batch_gguf.metal` plus `crates/runtime/src/families/gptoss/moe_batch.rs`, behind the existing `MFERENCE_ROUTED_BATCH` seam. Bit-exact against M decode-pair calls at the real shape, byte-identical greedy AND sampled on the real install. The pair's own `c(M)` was later measured on the bench's interleaved arms at 0.76/0.74/0.73/0.74 (M=2/4/8/16); the 0.67 first inferred for c(16) from the phase table's device-time rows was 9% optimistic, and the within-a-point agreement with the affine pair holds on same-day same-instrument terms (affine c(8) read 0.73 beside MXFP4's 0.73). The occupancy-not-weight-amortization conclusion stands.
    - [x] **Silu-flag hygiene on the MXFP4 pair DONE 2026-08-28**: `moe_prefill_batch_gguf.rs`'s four public functions no longer take `use_silu`; the specialization is a module-private `const USE_SILU: bool = true` feeding `moe_function_constants`/`constants_key`, so the argument encoder and dispatches cannot disagree and no caller can compile a second copy of the seven-file MSL concatenation mid-prefill. Two things the task's premise got wrong, both settled by grep and by the parity suite: the flag is NOT fully dead for MXFP4 (`moe_activate_mxfp4`'s PLAIN arm falls back to `moe_hidden_activation`, which reads `FC_MOE_ACT_SILU`, and the parity file's plain-activation case reaches it), and the parity suite had been holding bit-identity at silu=false against a production that runs silu=true. Both arms of the parity file now compile the specialization production compiles (the decode-pair oracle's args flipped to true alongside the const), all 7 cases green byte-for-byte. Production output cannot move: the family runs `Mxfp4Activation::GPT_OSS`, where the flag selects a dead branch.
    - **The GGUF (`qwen3moe`) arm is the weak one and may not be worth building.** 37.1% of its prefill is expert `pread`, which batches not at all and is already at the end of its lever (75.7% hit rate at the maximum 32 slots). Its routed device share and reachable M are both Gemma's, and it needs TWO kernels (Q4_K gate/up, Q6_K down).
    - Before quoting any end-to-end number for either arm, measure `c(M)` on the real shape. The only measured `c(M)` anywhere is INT4-affine at Gemma's shape, and this document already recorded being wrong by 2.3x once from borrowing a proxy across kernels.
    - [x] **MXFP4 `c(M)` on the bench harness** DONE 2026-08-27: `moe_prefill_batch_bench.rs` has an `mxfp4` module at the real D=2880 F=2880 top_k=4 shape (unions 6/10/13/17), reading 0.76/0.74/0.73/0.74 at M=2/4/8/16 across four serial runs with spread under 0.01. The inferred 0.67 was 9% optimistic (a real cross-instrument gap); the within-a-point cross-block-type agreement holds when both arms are measured on the same instrument the same day. The two benches in that file are serialized by a static mutex now -- the first `-- --ignored` run put both on the GPU concurrently and read affine c(8) as 0.38 with no error (`docs/BATCHED_PREFILL.md`).
    - [x] **Separate the 28% expert-miss drop** DONE 2026-08-27, and the measurement REFUTED the doc's reasoned attribution. Two corrections to how this task was written. The seam it named did not exist: `MFERENCE_ROUTED_PIPELINE` was read by Gemma 4's sequential decode alone, and the gpt-oss per-token prefill arm passed its `protect` set unconditionally -- so "no new code" was false, and the seam had to be wired first (`families/gptoss/prefill.rs`: off means banks = 1 AND an empty protect set, together, since retire-before-plan is what makes the empty set sound). And the hypothesis it carried was wrong: with the protect set off, misses read 9,400 against the control's 9,024 (stdout md5-identical), recovering NOTHING of the 9,024 -> 6,478 drop. The drop is real union dedup -- the one measured family-scoped exception to "the union saves nothing", consistent with AGENTS.md Gotcha 54's own bound: gpt-oss is the one family whose full 16-token window union (17.2) fits the slot cache (24) while the cache does not hold the expert table (32), so intra-window eviction re-reads exist AND the union can recover them (`docs/BATCHED_PREFILL.md`).
  - [x] **Default-On Configuration**: `--prefill-chunk` wired in the CLI and (with no per-request flag, matching the rate cap and guardrails toggle) automatically in the server, both gated on `supports_chunked_prefill()`.
  - [x] **Family Widening (complete)**: Gemma 4, both halves of `llama` (Mistral, Llama 2/3.x, Mixtral, Qwen3MoE), `muse_glimmer` and `gpt-oss` all land Step 1. Only the qwen linear-attention flow (`qwenGdnMoe` / `qwenGdnDense`) remains, and it was not attempted this pass.
  - [x] **Qwen Dense Chunked Prefill (2026-08-29)**: `families/qwen/prefill.rs` lands `qwenGdnDense` as Step 1, no new kernel, no new buffer -- see the status paragraph above and `docs/BATCHED_PREFILL.md`'s "sixth flow" entry for the design (GDN state ordering, the vision and open-drafter refusals). `qwenGdnMoe` (Ornith 35B) is not attempted this pass, matching how `llama`'s two halves landed as separate steps.

---

### 2. Streamable-MoE Candidate Bring-up

- **Context**: Future large/low-memory models on Mac Unified Memory require fine-grained MoE architecture where expert weights stream or map on demand.
- **Detailed TODO**:
  - [ ] **`gpt-oss-120b` (MXFP4 GGUF)**:
    - 36 layers, 128 experts top-4, 12.6 MiB stride. 59 GiB total.
    - Evaluate under mapped expert residency to test SSD throughput limits.
  - [ ] **`Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF`**:
    - 45.1 GiB, ~1.7 MiB stride (512 experts top-10, 48 layers).
    - Shares GDN + gated attention + fine MoE layer graph with `qwen36`.
  - [ ] **`bartowski/OLMoE-1B-7B-0924-Instruct-GGUF`**:
    - 3.9 GiB, ~3.4 MiB stride (64 experts top-8, 16 layers). Lightweight bring-up testbed.
  - [ ] **`bartowski/baidu_ERNIE-4.5-21B-A3B-Thinking-GGUF`**:
    - 12.6 GiB, ~6.3 MiB stride.
  - [ ] **`LiquidAI/LFM2.5-8B-A1B-GGUF`**:
    - 4.8 GiB, ~5.9 MiB stride. Requires conv-hybrid Metal kernels.

---

### 3. Speculative Decoding & Drafters

- **Dense MTP**: Shipped native MTP drafter for Qwen 3.8 (1.44x decode at block 2).
- **DFlash2 Block Drafter**: Shipped block-diffusion drafter (1.33x code, 1.47x math; `docs/DFLASH2.md`).
- **MoE Speculative Verify**: Kernel half shipped (`moe_batch.rs`, bit-identical to sequential decode).
- **Detailed TODO**:
  - [ ] **Exact Rejection Sampling (T > 0)**: Implement Leviathan/Chen algorithm on shaped distributions for non-greedy sampling.
  - [ ] **Small Qwen Drafter Rows**: Baseline and verify 9B / 4B dense MTP models.
  - [ ] **Batched Sub-byte GEMMs**: Implement batched INT1 and INT2 GEMM kernels to unblock Bonsai-27B and Ternary-Bonsai speculation.
  - [ ] **Speculative Prefill Optimization**: Retain drafter input activations across prefill when `skip_head` is active to eliminate duplicate evaluation.
  - [ ] **MoE Drafter Ingest**:
    - Build ingest for MoE MTP heads (e.g. Ornith 35B with 256 per-expert tensors) or convert block drafter for MoE models.
  - [ ] **Speculation String Cleanup**: Update refusal strings in `mtp_state.rs`, `dflash_state.rs`, `speculation_policy.rs`, and CLI session error reporting.

---

### 4. Server Concurrency & Fairness

- **Current State**: Single mutex-serialized runner per process.
- **Detailed TODO**:
  - [ ] **Prefix KV Reuse on the Server (cheapest item here, ~1 flag)**: the mechanism landed 2026-08-29 and the server does not opt in yet. One runner per process serving requests one at a time is exactly the shape reuse wants, and a chat client resends the whole transcript every turn, so this is the largest single TTFT win available on this surface (11.6x prefill measured through the CLI's equivalent path). Needs `RealForwardRunner::set_prefix_reuse` called at open plus a flag through `crates/invocation`'s five places. **Caveat that makes it a decision rather than a wiring task**: with ONE runner and multiple clients, consecutive requests from DIFFERENT conversations each destroy the other's reusable prefix, so the match rate depends on traffic mix; a `[prefix-reuse]`-style counter should land with it (the CLI's exists because the feature read 0/33 through two rounds of apparently-working implementation). Interacts with Option 3 below, which is the real fix for multi-conversation reuse.
  - [ ] **Request Queue & Fairness (Option 1)**: Implement request FIFO queue with streaming-aware fairness and cancellation handling.
  - [ ] **Multi-Runner Pool (Option 2)**: Support configurable N runners (multiplies KV cache and slot cache memory by N).
  - [ ] **Multiplexed Session State (Option 3)**: Single execution runner multiplexing per-session KV cache and recurrent state buffers.

---

### 5. DeepSeek-V4-Flash Bring-up & Feasibility

- **Status**: Scaffolded (`CompressedAttentionConfig`, `HyperConnectionConfig`, `Dsv4StateManager`).
- **Detailed TODO**:
  - [ ] Port CSA/HCA attention, unrolled mHC Sinkhorn, and sub-3bit GEMV Metal kernels (`dsv4.metal`).
  - [ ] Assess memory constraints: Model requires ~106.9 GB peak RSS (128 GB Unified Memory Mac required; SSD read bounds decode to ~0.05-0.1 tok/s on smaller machines).

---

### 6. Adaptive Expert-Cache Sizing & Mapped Unification

- **Status**: `ExpertCacheSlots::Auto` shipped in `crates/model-io/src/expert_cache_policy.rs`.
- **Detailed TODO**:
  - [ ] **Unify Residency and Cache Policy**: Combine slot cache policy with mapped expert residency -- pick residency mode first (`mapped` vs `streamed`), then slot count only if `streamed` is active. Gated on eviction benchmarks.

---

### 7. Linux Backend

- **Detailed TODO**:
  - [ ] Implement `io_uring` + `O_DIRECT` streaming I/O layer on Linux.
  - [ ] Pair with portable CPU/Vulkan compute backend.
  - [ ] Support cgroup memory and CPU limits.

---

### 8. Directional Weight Steering (Abliteration & ActAdd)

- **Status**: Runtime abliteration, ActAdd (`add`), feature clamping (`clamp`), and norm-preserving projection (`renorm`) shipped across 7 families (`docs/OBLITERATION.md`). CLI and server flags wired.
- **Detailed TODO**:
  - [ ] **Activation Capture Integration Tests**: Add automated integration test verifying end-to-end activation capture and vector export.
  - [ ] **Multi-Direction Steering**: Support applying multiple steering vectors simultaneously with per-vector scales and layer masks.
  - [ ] **Throughput Profiling on Remaining Families**: Benchmark decode overhead of steering across `llama`, `gemma4`, `gpt-oss`, and `museGlimmer`.
  - [ ] **Automated Alpha Calibration**: Improve heuristic/proxy metrics for detecting semantic steering thresholds before collapse.

---

### 9. Mapped Expert Residency

- **Status**: LANDED ON `main` 2026-08-30. The base feature spent a day as a commit nobody could reach: written 2026-08-23 on an unmerged branch `cache-policy`, then rebased onto `expert-residency` in a worktree (`../mrefrust-residency`) that was later removed WITHOUT the branch ever being merged -- the branch ref itself was gone by 2026-08-30, and the tip commit (`702716b`) survived only as a dangling git object one `git gc` away from being pruned (AGENTS.md Gotcha 13's exact failure shape). Recovered with `git branch expert-residency 702716b` and merged into `main`: 8 files conflicted against the router-lookahead PILOT probe and prefix-KV-reuse work that had landed on `main` in the meantime, all independent, non-overlapping additions that `git merge`'s `ort` strategy resolved cleanly except two doc files (`AGENTS.md`'s doc index, `crates/streaming/CLAUDE.md`'s gotcha numbering), fixed by hand. Full workspace suite, fmt, clippy and the cross-target check all green; `mapped_expert_probe`/`mapped_experts`/`mapped_expert_residency` pass against the real install; Gemma 4 quality gate and memory oracle reproduce their frozen rows exactly on the default arm; greedy and sampled smoke are byte-identical between the default and `mapped` arms. Seam: `MFERENCE_EXPERT_RESIDENCY=mapped`, off by default, Gemma 4 only, both arms produce identical tokens.
- **Numbers, RE-MEASURED 2026-08-29 on AC at the `auto`-resolved 32 slots**: peak `phys_footprint` **3,652 -> 559 MiB** (a 3,093 MiB saving against a predicted `32 x 30 x 3.2 MiB` = 3,072 MiB slot cache; both arms reproduce to under 0.6%) and decode **53.8 -> 68.7 tok/s, 1.28x**. The superseded pair was 3,721 -> 606 MiB and 51.9 -> 69.8 tok/s, taken before chunked prefill became the default. **Read the throughput figure with its caveat**: the capture was not on an idle machine, contention costs the `pread` arm more than the mapped one, so 1.28x is a ceiling rather than a floor (`docs/EXPERT_RESIDENCY.md`).
- **Detailed TODO**:
  - [x] **Wire Remaining MoE Families, DONE 2026-08-30**: `qwen`, `llama` (both `Llama` and `Qwen3Moe`), and `gptoss` all carry the same fork Gemma 4's `moe.rs` does, replacing `mapped_residency_refusal`'s named refusal for each rather than adding a branch beside a silent path. `qwen`'s and `gptoss`'s batched-routed drivers (`moe_batch.rs`) each gained the same mapped-vs-batched conflict guard Gemma 4's carries; `llama` needs none, since it has no batched-routed driver to conflict with. Verified: full workspace suite/fmt/clippy green; per-family synthetic tests (`crates/runtime/tests/mapped_expert_residency_{qwen,llama,gptoss}.rs`) pass, each opening under `MFERENCE_EXPERT_RESIDENCY=mapped` and asserting finite non-zero logits plus (where applicable) the batched-conflict refusal firing by name; real-install verification landed for two of the three -- `~/.turbospark/models/qwen3moe.gturbo` (the `llama` flow's `Qwen3Moe` half) and `~/.turbospark/models/gptoss-20b.gturbo` both reproduce byte-identical greedy output between streamed and mapped arms, with `MFERENCE_PHASES=1` confirming 0.0 ms `pread` time and a 100% expert-cache hit rate under mapped mode on both; gptoss's frozen `quality_gate` and `memory_oracle` rows reproduce exactly with the seam unset, confirming the default arm is unmoved. **`qwen`'s own family (`QwenGdnMoe`) has no real install left on this machine** -- Ornith 35B, which this TODO's own text called "on disk; immediate win" when written, is gone by the time the wiring landed (`CLAUDE.local.md`'s artifact inventory has drifted and needs re-checking against `ls ~/models` / `ls ~/.turbospark/models` before the next session trusts it) -- so that family is verified on the synthetic fixture only.
  - [ ] **Memory Pressure Eviction Benchmark**: Measure paging overhead and latency impact when operating under OS memory pressure.
  - [ ] **`auto` Policy Resolution**: Automatically select `mapped` when machine memory accommodates the full model file cache, falling back to `streamed`.
  - [ ] **CLI Flag**: Expose `--expert-residency auto|streamed|mapped` on `turbospark-check` and `turbospark-server`.
  - [ ] **`madvise(MADV_WILLNEED)` Prefetching**: Evaluate background advice on routed offsets to minimize cold-start fault overhead.
  - [ ] **Mapped Memory Oracle Rows**: Record separate frozen memory baseline rows for mapped residency mode across all MoE families.

---

### 10. Phase-2 `top_k` Specialization

- **Status**: Scoped; awaiting implementation.
- **Background**: `moe_phase2_down_reduce_k8` currently executes 8 down-GEMVs regardless of model `top_k`. For `gpt-oss` (`top_k = 4`), 50% of phase-2 compute is wasted on zero-routed slots.
- **Detailed TODO**:
  - [ ] **Specialization Pipeline**: Specialize `moe_phase2_down_reduce` kernel via `FC_MOE_TOP_K` constant.
  - [ ] **Widen Pipeline Cache Key**: Extend `constants_key` in `crates/gpu` to include the `top_k` specialization constant (avoiding pipeline collision).
  - [ ] **Benchmark and Validate**: Re-stream `gptoss-20b` and measure throughput win on real-model gate.

---

### 11. Vision & Multimodal Support

- **Status**: M-V0 through M-V8 have LANDED, against the `qwen3_5` tower rather than
  ViT/SigLIP. An image reaches a generated token from the command line AND from both
  server endpoints: the tower agrees with mlx-vlm at its own FP16 floor, this port
  renders and splices a text+image prompt byte-identically to the reference
  processor, and `turbospark-check --image`, `/v1/chat/completions` and
  `/v1/messages` all transcribe a real page to the same bytes. This entry used to read
  "Implement ViT / SigLIP image encoder pre-pass" as though nothing had started,
  which was wrong in the opposite direction from item 9's.
- **`docs/VISION.md` is the home for this work and carries the milestone list.**
  Do not duplicate the milestones here; a second copy is what rots.
- **Closed 2026-08-29**: M-V9 (multi-page memory oracle, NaN-safe parity
  instruments, FP16 overflow capture). Verified against the real install
  (measured peaks 785.3-871.5 MiB across four pages, ceiling 950 MiB) but
  UNCOMMITTED -- verification was scoped to the five files it touched rather
  than the whole-workspace gate, because a concurrent session's own refactor
  was mid-flight across unrelated crates for most of the session this landed
  in. See `docs/VISION.md`'s "What is not built" section for the full record
  before assuming this has reached `main`.
- **RESOLVED artifact gap**: `~/models/qwen38-27b-vision.gturbo` shipped no
  tokenizer or preprocessor sidecars, having been streamed for the tower alone.
  All five were copied in from `~/models/qwen38-27b.gturbo` on 2026-08-29, each
  verified byte-identical to the pinned reference snapshot
  `models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f0...` first -- the same
  revision the weights were streamed from. A future vision install should be
  `pull`ed with `--sidecar-repo` rather than repaired this way.
- **The lesson M-V7 and M-V8 both acted on**: M-V5 shipped a bug that only an
  end-to-end run could find. `run_raw_completion` calls `producer.reset()` at
  ENTRY and `reset` was clearing the injection map, so every image prompt
  prefilled placeholder embeddings and answered fluently about a picture it had
  not seen -- with the right prompt length and no error anywhere. Every test drove
  `produce` directly and missed it. The server was the SECOND caller of that
  contract and got its own end-to-end arm accordingly
  (`real_backend_reads_an_image_sent_over_both_endpoints`, which asserts the
  transcription carries the page's own line numbers). **M-V9 is the third: an
  oracle that asserts a memory shape cannot see whether the pages were read.**

---

## Guiding Principles

1. **End-to-end gates decide everything**: End-to-end throughput, quality, and memory determine if an optimization ships, not isolated microbenchmarks.
2. **Bytes-per-token is the primary metric**: Decode is expert-I/O bound. Latency and energy optimizations reduce to reading fewer bytes, hiding reads better, or avoiding copying bytes that are already addressable.
3. **Lossless repack, always**: Installers shuffle quantized bytes without re-quantizing. Only upcast F32 norms/routers in GGUFs are transcoded.
4. **Enumerate architectures; do not abstract prematurely**: Shared streaming engine + explicit per-family modules + per-model manifests.
5. **Wasted power first, throttled power second**: Eliminate spin-waits and redundant compute before adding user-facing throttle knobs.

---

## Do Not Revisit (Measured Dead Ends)

1. **Quantized KV Cache**: Quality loss (delta-NLL +0.015).
2. **Cold mmap as a Replacement for Streaming**: `pread` is strictly superior for cold uncached experts (74.8s vs 2.5s prefill). `mmap` is only used for warm residency (`docs/EXPERT_RESIDENCY.md`).
3. **RDADVISE as Default**: No stable production benefit.
4. **Expert Prefetch / Speculation**: TWO different predictors, both measured, both negative. The inherited one copies layer L's selected expert IDs and hits 7% (Jaccard 0.039). The second, measured here 2026-08-29 against colibri's PILOT, RUNS layer L+1's router GEMV on layer L's post-attention residual and genuinely works -- 70.6% recall, reproducing colibri's reported 71.6% on a different architecture, covering 60.4% of misses at 32 slots. It still loses, on a different axis: a prefetcher's COST scales with prediction width while its BENEFIT scales with the miss rate, and at an 84.6% cache hit rate every `PILOT_K` from 1 to 8 reads MORE total expert bytes than the demand path (1.03x to 1.85x). No operating point pays. Reversal condition: a machine where the expert read is genuinely disk-bound rather than a page-cache memcpy. Probe stays wired (`MFERENCE_PILOT_PROBE`, `=self` to validate the instrument); method and full sweep in `docs/EXPERT_ROUTING.md`.
5. **Expert Pread Tuning**: `MISS_READ_CHUNK_BYTES` (840 KiB) and `POOL_THREADS` (8) are at measured local optima.
6. **Deferred End-of-Token Wait**: Max theoretical gain 0.25 ms/token (1.4%); bottlenecked by layer dependencies.
7. **GPU-Side Router Top-K**: Host top-k overhead is 0.13 ms/token; GPU top-k only relocates synchronization.
8. **Buying Back Hit-CB Overlap in Kernel**: Determinism fix costs <2% throughput; modifying vendored kernel not justified.
9. **Monolithic Mega-Fusions (`fused.metal`)**: Host CPU encode is ~1 ms/token post cache-key fix; no headroom.
10. **Offset-Sorted Reads / Fine-Grained Read Dispatch**: Slower or nondeterministic.
11. **Domain-Restricted Expert Sets (pruned / pinned)**: 95% of routed mass touches ~67 of 128 experts across domains; static pruning damages quality (`docs/EXPERT_ROUTING.md`).
12. **Sub-4-bit Experts as an Efficiency Win**: IQ3_XXS / IQ4_NL is 35% slower and roughly doubles joules/token (GPU codebook dequant bound); valid only as a memory/disk tradeoff (`docs/POWER_BASELINE.md`).
13. **Staging `x` in `dequant_int4_gemm_mma`**: measured 2026-08-29 behind `FC_MMA_STAGE_X` (110), bit-identical to the un-staged arm and **3.3x to 5.9x SLOWER at every width, penalty growing with B**. A transposed `simdgroup_load` from device is not the naive strided gather it reads as; hand-staging the same bytes through 32 LANES is. MLX stages `x` and wins because it has 128 threads to do it and `BM` 32-128 to amortize over, so staging is a CONSEQUENCE of the wider threadgroup rather than a separate lever (AGENTS.md Gotcha 65). Reversal condition: only as part of a four-SIMD-group re-tile, never alone.
14. **The dequant loader and `kMmaTile` as levers on the matrix kernel**: two settled by different methods on the same day. `kMmaTile` is refuted by ARITHMETIC -- dequant per output is `N / B` here and `K / BM` in MLX, both keyed on TOKENS, so a bigger weight tile scales dequant and outputs together and changes nothing; this kernel already runs at B=64 at MLX's own BM=64 intensity and is still 3.5x behind. The loader is refuted by MEASUREMENT: `FC_MMA_SKIP_DEQUANT` (111) deletes the unpack and the kernel **still reads 0.46-0.50 past M=16 against MLX's 0.145**, and is still slower with a FREE dequant (0.51 at M=16) than the scalar `dequant_int4_gemm_simd` is with a real one (0.38). What is left is the matrix path itself: one SIMD group per threadgroup, two `simdgroup_barrier`s per 64-element K block, eight `simdgroup_float8x8` accumulators over 32 lanes. That is the ONLY untried lever there.
15. **Chunked WY-representation gated DeltaNet prefill, and GDN threadgroup staging**: two negatives on one kernel. The chunked (flash-linear-attention) reformulation is a negative on oMLX's OWN side -- `gdn.py` ships ~270 lines of it and its docstring says the production path is the blocked-sequential recurrence at "half the FLOPs of the WY-chunked path". And their blocked-sequential kernel's real contribution (threadgroup-staged q/k/v against a `(Hv, Dv/4)` re-read, which this port's `gdn_delta_step_prefill` does have) targets a term measured at **4.66%** of a prefill micro-batch, upper bound (`crates/gpu/tests/gdn_prefill_share_bench.rs`, agreeing with an independent traffic calculation to 0.2 points). Even a perfect 8x traffic reduction there caps out near 4% of prefill.

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
- `MFERENCE_APP` (upstream Swift Mference UI): Out of scope (superseded by this repository's native `TurboSparkApp` desktop application and `.app`/DMG release packaging).
- `prefill.metal` GPU tile pipeline: Descoping retained; chunked prefill driver reuses standard kernels.
- `logit.metal` `sample` kernel: Host sampling via `crates/selection` is standard.

---

## Changelog (Completed Milestones)

- **Core Port**: Gemma 4 26B-A4B and Qwen 3.6 35B-A3B at Swift parity, `.gturbo` streamed format, CLI, server, bench harness, and memory oracle.
- **Phase Q / P1 (2026-08-07)**: Quality harness, perplexity gates, golden digests, cross-engine KL, power baseline (`scripts/power.sh`).
- **Phase G / S (2026-08-08/09)**: GGUF ingestion, repack transcode, Q8_0/Q4_K/Q6_K kernels, sub-4-bit IQ3_XXS/IQ4_NL codebook kernels.
- **Phase P2 / M1 (2026-08-09)**: User power controls (rate limiting, low power), architecture registry and model discovery.
- **Phase M2 / M3 / M4 / M5 (2026-08-10..12)**: Mixtral MoE, Qwen3-30B fine-grained MoE, dense Mistral/TinyLlama, `gpt-oss-20b` MXFP4 MoE.
- **Quantization Widening (2026-08-13..15)**: 1-bit (`Bonsai-27B`) and 2-bit ternary (`Ternary-Bonsai`) affine quantization and Metal GEMVs.
- **Harmony Protocol (2026-08-13..15)**: Thinking token channel splitting (`analysis`), tool call extraction and schema validation.
- **Native MTP Speculation (2026-08-18)**: Native MTP drafter for dense Qwen 3.8 at block 2 (1.44x decode, lossless greedy verification).
- **Batched Routed Prefill (2026-08-18)**: PF-02 steps 2-3 routed pair batching (1.54x prefill speedup on Gemma 4).
- **DFlash2 Block Drafter (2026-08-19..21)**: Block-diffusion drafter for Qwen 3.8, unblocked from FP16 overflow, 1.33x-1.47x speedup on code/math.
- **Ornith-1.5 Checkpoints (2026-08-20)**: 9B dense and 35B MoE in GGUF Q8_0 and MLX INT4 formats, verified via cross-engine KL.
- **Batched Resident GEMVs (2026-08-22)**: PF-02 step 6 resident attention and shared expert GEMVs dispatched as M-row GEMMs.
- **Mapped Expert Residency (written 2026-08-23, landed on `main` 2026-08-30)**: In-place `mmap` expert execution on Gemma 4 (footprint 3,652 -> 559 MiB, decode 53.8 -> 68.7 tok/s, re-measured 2026-08-29). Written on a branch that was never merged and whose worktree was later removed, leaving the tip commit dangling and one `git gc` from lost; recovered by re-creating the branch ref from the dangling object and merged against `main`'s intervening PILOT-probe and prefix-KV-reuse work. See section 9.
- **Mapped Expert Residency, remaining MoE families (2026-08-30)**: `qwen`, `llama` (both `Llama` and `Qwen3Moe`), and `gptoss` all wired the same day the base feature landed, each replacing its named refusal rather than adding a branch beside it. Real-install verification on `qwen3moe.gturbo` and `gptoss-20b.gturbo` reproduces byte-identical greedy output between streamed and mapped arms with a confirmed 100% expert-cache hit rate under mapped mode; `qwen`'s own family has no real install left on this machine to verify against (Ornith 35B is gone from disk since the feature was scoped), so it stands on synthetic-fixture verification alone. See section 9.
- **Directional Weight Steering (2026-08-23..25)**: Runtime abliteration, ActAdd, clamping, and renorm Metal shader across 7 model families with CLI/server flags (`docs/OBLITERATION.md`).
- **PF-02 Default-On and Dense-Llama Widening (2026-08-26)**: `--prefill-chunk` wired as the default in the CLI and automatically in the server, gated on a shared `supports_chunked_prefill()` predicate so an unsupported family falls back to sequential with no error. Chunked prefill also runs the dense half of `llama` now (Mistral, Llama 2/3.x), the second `ChunkedPrefillRunner` implementation, needing no mid-layer host round trip.
- **PF-02 Qwen Dense Widening (2026-08-29)**: chunked prefill now serves `qwenGdnDense` (`qwen38-27b.gturbo`), the sixth `ChunkedPrefillRunner` implementation and the first with no new buffer allocation at all -- the existing per-token GDN and attention kernels reused unmodified inside fewer command buffers. Verified byte-identical against sequential on the real install; unblocks the missing oMLX PP comparison. `qwenGdnMoe` is a follow-up.
- **Reasoning Effort & Thinking Channels (2026-08-29)**: `--reasoning` / `reasoning_effort` wire support across server, CLI, and tokenizer; reasoning channel separation for ChatML and Gemma.
- **Prefix KV Cache Reuse (2026-08-29)**: Cached prompt continuation in `runtime::kv_prefix`, evaluated on real 26B (11.6x prefill speedup on continuation).
- **Batch INT4 GEMM Row Blocking (2026-08-29..30)**: Hardware-optimal row block dispatch (`R=1, 2, 4`) for M-row batched INT4 GEMMs, maximizing GPU compute occupancy and establishing crossover curves (`docs/BATCHED_PREFILL.md`).
- **macOS App & Release Automation (2026-08-29..30)**: `TurboSpark.app` bundle and DMG release pipeline (`scripts/make-app-bundle.sh`, `scripts/make-dmg.sh`, Homebrew cask), permissions engine, and Swift UI localization.
- **Swift Agent Subsystem & Custom Tools (2026-08-31)**: Built-in agent manager, `AGENTS.md` parser, subagent execution runner, custom tool runtime with JSON schema support, and `web_fetch`/`todo_write` executors in `TurboSparkApp`.
