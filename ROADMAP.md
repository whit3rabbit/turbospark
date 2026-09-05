# Roadmap

The forward-looking roadmap and task tracker for this engine, last reconciled against the tree on 2026-09-02. All core port phases (Q, P1, G, S, P2, M1-M5) are complete and green. This document functions as an active TODO list for forward engineering, measurements, and architectural bring-ups.

---

## Current Status (2026-09-02)

- **Test Suite**: **1,758 tests declared workspace-wide, of which 1,627 run under the standing gate and 131 are `#[ignore]`d** (the checkpoint downloads, the memory oracles, the quality gates, the sensitivity proof, the cross-engine dumps, and offline benchmarks). Re-counted 2026-08-30 with `cargo test --workspace -- --list` and `--list --ignored` after landing Mapped Expert Residency, on macOS, so the macOS-only crates are included. The prior count on this same line (1,770 / 1,640 / 130) was from earlier the same day; this repo's own house rule is not to trust a prose test count without re-deriving it, so this is the current one rather than an increment on the old one. Strict formatting, clippy and cross-target checks clean.
- **Architectures**: 7 `ModelFamily` variants running across 17 curated catalog rows (`gemma4`, `qwenGdnMoe`, `llama`, `qwen3moe`, `gptOss`, `museGlimmer`, `qwenGdnDense`).
- **Recent Landings**:
  - **Vision (`qwen3_5`), M-V0 through M-V9**: an image reaches a generated token from the COMMAND LINE and from BOTH server endpoints. The tower agrees with mlx-vlm at its own FP16 floor, this port builds the spliced prompt byte-identically to the reference processor, and `--image` / `--image-batch` transcribe a real page. M-V9 landed 2026-08-29 (multi-page memory oracle, the last two NaN-safe parity guards, FP16 overflow capture). See `docs/VISION.md`'s "What is not built" section for the full record; nothing is open there now.
  - **Batch INT4 GEMM Row Blocking (`dequant_int4_batch.rs` & `dequant_int4_mma.metal`)**: Row-blocked dispatch wired for M-row batch GEMMs with optimal tile dispatch (`R=1, 2, 4`), register limit queries, and crossover points documented in `docs/BATCHED_PREFILL.md` and `docs/BENCHMARKS.md`.
  - **Reasoning Effort & Thinking Token Protocol**: Multi-dialect support for `--reasoning` / `reasoning_effort` across CLI and server (`off`, `low`, `medium`, `high`, `xhigh`), ChatML/Gemma thinking extraction, and Swift UI integration.
  - **Expert Disk I/O & Bypass Telemetry**: Disk I/O tracking and cache-bypass telemetry in `crates/streaming` (`TURBOSPARK_PILOT_PROBE` validation and pread streamer metrics).
  - **Prefix KV Reuse (cached-prompt continuation)**: a turn continues from the previous turn's KV wherever the two prompts agree, instead of resetting and re-prefilling the whole transcript. `runtime::kv_prefix` plus one defaulted `LogitProducer` seam. Measured on the real 26B: prefill **1.777s -> 0.153s (11.6x)** on a transcript-shaped prompt, generated tokens byte-identical to the re-prefilled reference; in the real `--chat` REPL, 13/33 then 29/49 tokens continued. **`crates/ffi` opted in 2026-09-01** (`open.rs`'s `open()`, the one place it deliberately diverges from the CLI's single-shot `open_session`), so `TurboSparkApp`'s multi-chat sessions get the same win the REPL has had since the feature landed -- see section 4 below and `crates/ffi/CLAUDE.md` Gotcha 16. **`turbospark-server` closed the same gap the same day** via a real `--prefix-reuse on|off` flag (default on) rather than the FFI's unconditional opt-in, since a server's one runner serves multiple clients where the FFI's serves one session per model; see `crates/server/CLAUDE.md` Gotcha 31. **A swap-based session pool (`--session-slots`) landed the same day too**, the actual fix for that server's own multi-conversation stomping gap: see section 4's Option 3 entry, and `crates/runtime/CLAUDE.md` Gotcha 32 for the two real bugs a real-install test found in it.
  - **`qwen4_exp` (Qwen3.8-Flash-Next) Intake: Config, Classification, and the N-Gram Store (2026-08-31)**: builds on the Phase 0 fact-finding in `docs/QWEN4_PHASE0.md` (checkpoint config, tensor layout, two independent references cross-checked, 2026-08-27/28). Landed since: the family and config surface (`crates/model-io`'s `arch_config`/`arch_baselines`), Phase 0 config parsing gate, classification of its n-gram table out of the resident index, the table's own on-disk layout and refusals (`model_io::ngram_table`), and a streaming, self-checked writer (`crates/repack`) that never holds the whole 32+ GiB table in memory (peak is one shard in, one shard out). Manifest wiring reaches all three consumers. **This is intake, not a running model**: no `families/qwen4_exp/` decode flow landed with it, so section 2 below still lists it as open. Three findings recorded on the way: HF header probes cheap enough to size a repo before any transfer (`.claude/docs/diagnostics.md`), `ALLOWED_CACHE_SLOTS` (not free RAM) as the real cap on how much of a fine-grained MoE table can be resident (AGENTS.md Gotcha 36), and `gdn_gated_norm`'s hardcoded silu being wrong for this family, which declares sigmoid (`crates/gpu/CLAUDE.md` Gotcha 12).
  - **`qwen4_exp` Decode Flow, Router Dtype Fix, and QSA Sparse Attention (2026-09-02..05)**: `families/qwen4/` now runs a complete per-token forward pass (hyper-connections, GDN's sigmoid gate replacing the earlier silu default, gated MoE, the PLE n-gram chain) wired end to end into `RealForwardRunner` (Phase 3, 2026-09-02). Phase 4 (2026-09-03) widened `ALLOWED_CACHE_SLOTS` from `[8,16,24,32]` to `[8,...,128]`, since this family's 288-expert table costs 126.6 MiB/slot and the old ceiling capped residency at 11%; also added an open-time refusal when `expert_cache_slots < top_k_experts`. **A real install exists**: `~/.turbospark/models/qwen4-reap288.gturbo` (68G, REAP-288, `top_k_experts=10`). It was refused at OPEN until 2026-09-04's fix widened `moe_phase2_down_reduce_k8`'s fixed 8-slot capacity to a runtime width sized to the caller's own `top_k` (`crates/gpu/CLAUDE.md` Gotcha 13) -- every pre-existing top_k=8 family is unmoved, byte-identical dispatch. **The router-dtype blocker this bullet used to name here is fixed, same day it was found**: `612be53` and `27b666f` (2026-09-04) taught the safetensors repack orchestrator (`crates/repack/src/gemma4_checkpoint/orchestrate.rs`) to force-quantize `mlp.gate.weight` and `mlp.shared_expert_gate.weight` to INT8-affine for `Qwen4Exp`, since this REAP-288 checkpoint's from-safetensors publish ships both gating matrices raw BF16 where every other MoE family's upstream MLX conversion happens to pre-pack the router as `U32`. The INT8-only router GEMV refusal itself was correct design, not the defect. **QSA (query-sparse attention) then landed end to end 2026-09-05**: `families/qwen4/attn.rs` runs the indexer and, above the install's own `index_budget`, sparse block-selected attention in place of the dense fallback, verified against 17 new synthetic-fixture tests (mutation-checked) and against the real install -- coherent greedy and sampled smoke on a 2,940-token prompt, `qwen4exp_quality_gate` and `qwen4exp_memory_oracle` both green and frozen (perplexity 8.7224, peak 2521 MiB against a 3000 MiB ceiling), and a KL-based force-dense probe showing bitwise identity below budget and sub-0.1-nat divergence with 100% argmax agreement above it. `docs/QWEN4_EXP.md` is the source of record for the full mechanism and numbers; **Chunked prefill then landed 2026-09-05** as the seventh `ChunkedPrefillRunner` (`families/qwen4/prefill.rs`), byte-identical to sequential on 7 synthetic cases and on the real install, with both batching seams refused by name; see the Active Tasks row for the PLE host-write bug it found on the way. Section 2 below tracks what is still open (this driver's THROUGHPUT number, a GPU top-k for block selection, and the bench-window decision).
  - **One `TurnSplitter` for Turn Splitting (2026-09-05)**: three drifting copies of the same "split a turn into content / reasoning / tool calls" wiring -- `crates/cli`'s `ChannelSplit`, `crates/ffi`'s own port of it, and an inline `needs_decoder` block in `crates/server` -- collapsed into one `runtime::turn_stream` (`TurnSplitter`, `TurnEvent`). `StructuredAssistantDecoder` is now constructed in exactly ONE place in the workspace. The two traps that used to be re-derived per consumer are structural now: the emptiness check lives on the decoder's OUTPUT rather than its input (AGENTS.md Gotcha 44, where an `if text.is_empty()` guard swallows every frame token), and `finish()` is where a stop-token-terminated tool call is emitted (Gotcha 49). The C ABI gained two ADDITIVE event kinds, `TS_EVENT_TOOL` and `TS_EVENT_FINISH`, plus a `toolCalls` result field; Swift gained `.toolCall` and `.stopped`. `docs/STREAMING.md` is the home and carries the measured negatives, chiefly why there is no engine-side iterator or async stream (the loop takes a borrowed non-`Send` producer, so a facade would have to own it and spawn a thread, which is a consumer's policy decision) and why the channel is deliberately unbounded (backpressure in the decode path stalls the one engine every request shares). Read that page before proposing either.
  - **Swift Shell, Hook Contract and Settings Stores (2026-09-05)**: `run_in_background: true` now really backgrounds a command (`BackgroundShellManager`, `bg_N` ids scoped to the launching chat, a 20-shell ceiling, `BashOutput`/`KillShell` tools); shell execution moved out of the tool registry into `Tools/Terminal/` with cwd persistence across calls, ANSI stripping, head+tail compaction, benign-exit mapping (a `grep` exit 1 is an answer, not an error) and a hang-prevention environment. The Claude Code hook contract gained five deliberate divergences documented rather than guessed, chiefly that `continue: false` ends the turn and shows its reason to the USER where a block feeds the reason to the MODEL, plus `PermissionDenied`/`SubagentStart`/`SubagentStop` events and a loud failure for `prompt`/`agent` hooks (which must never fall through to `.command`, i.e. running prompt text as a shell command). The server API key moved from `settings.json` to the login Keychain, and appearance settings from `UserDefaults` to a JSON store with a one-way migration. ~42 new tests. `docs/SWIFT_TOOLS.md` is the home.
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
| **PF-02 Step 6 Throughput A/B** | Attempted 2026-09-05, re-run owed on a confirmed-idle machine | Real Gemma 4 install, `--expert-cache-slots 24` (`TURBOSPARK_ROUTED_BATCH=1` refuses 32+), three interleaved pairs: `BATCHED_GEMV` never slower than the baseline, but the pair-to-pair spread (1.02x to 1.38x, mean 1.19x) is Gotcha 43's own contamination signature (a background MCP process was at 150% CPU during the run) rather than a citable number. Direction confirmed; magnitude is not. See `docs/BATCHED_PREFILL.md`'s Step 6 section. No longer gates the default-on wiring, which landed independent of this measurement (family support, not throughput, is what the default checks). |
| **Prefill Energy Capture** | BLOCKED on a bench gap, not a measurement | Attempted 2026-09-05: `scripts/power.sh` drives `turbospark-bench`, and `crates/bench/src/real_model.rs` calls `run_raw_completion` directly -- it never reaches `run_raw_completion_chunked`, so chunking, `TURBOSPARK_ROUTED_BATCH` and `TURBOSPARK_BATCHED_GEMV` are all invisible to it. A `power.sh` run today would silently re-measure the old sequential path and read as a chunked-prefill row without being one. Needs `crates/bench`'s model mode taught to call the chunked path (mirroring the CLI's own `generate` loop) before this is measurable at all; that is new scope, not a 12-minute capture. **That scope is now being built** as an explicit `--prefill-chunk off|auto|N` on `turbospark-bench` (default OFF, so no frozen row moves) plus a `seq|chunked` arm pair in `scripts/power.sh`. Note only `TURBOSPARK_PREFILL_CHUNK` needs wiring: `TURBOSPARK_ROUTED_BATCH` and `TURBOSPARK_BATCHED_GEMV` are read INSIDE the runtime's chunk drivers, so they become visible the moment the chunked driver is engaged and need a header echo rather than a flag. |
| **PF-02 Step 4 (Batched Attention Kernel)** | New Metal kernel, real risk. **LOW VALUE on `qwenGdnDense`; scope it on a family where attention is a real share.** | Widen `attention_decode_partial` to hold M query rows per KV chunk. Real new-kernel work: a new function-constant axis, per-row online-softmax state generalized to M rows, and register-pressure risk per `dequant_int4_gemm_simd`'s spill history. **On the dense qwen flow it targets 2.6% of prefill in the 16 of 64 layers that have attention at all** (the other 48 are gated DeltaNet), against a GEMM holding 85.4% -- so on THIS family it is the wrong term. oMLX ships an `fa256` prefill attention and it is 35 lines of `instantiate_kernel` over MLX's steel kernel, i.e. nothing to port. |
| **PF-02: Qwen Dense GEMV-to-GEMM Widening** | DONE 2026-08-29 | Landed as a WIRING pass, not the new-kernel work this row predicted: `families/qwen/batched_layers.rs` already owned all three M-row encoders for the MTP/DFlash2 verify, at the row convention the chunk driver writes, and `MAX_PREFILL_BATCH` IS `gpu::MAX_BATCH_ROWS`. Measured **1.86x to 2.13x** on prefill (21.40 -> 40.79 tok/s, interleaved pairs, warmup discarded, contended machine so a lower bound). Against oMLX's 210.3 tok/s that is now roughly a FIFTH rather than a tenth -- still not closed. What remains unbatched is ATTENTION, i.e. Step 4 above, which IS real new-kernel work. The batched arm is not byte-identical on a real install and that is a measured shape floor (`e8deb6c`), so its gate is the quality gate; the DEFAULT arm stays byte-identical. `qwenGdnMoe` (Ornith 35B) is still unattempted. |
| **Mapped Residency on MoE Flows** | DONE 2026-08-30 | Wired into `qwen`, `llama` (both `Llama` and `Qwen3Moe`), and `gptoss`; each family's named refusal was replaced rather than bypassed. See section 9's TODO for the verification detail and the one open gap (`qwen`'s own family has no real install left on this machine to verify against). |
| **Mapped Residency Eviction Benchmark** | Investigation / memory pressure test | Measure degradation/fault costs when OS reclaims clean mapped pages under memory pressure. Gates `auto` default and CLI flag. |
| **Phase-2 `top_k` Specialization** | DONE 2026-08-31 | Landed as `moe_phase2_down_reduce_k8_mxfp4`-only, not the whole `moe_phase2_down_reduce_k8` family named in this row's title -- see section 10 below for why the other kernels are unaffected and what verification it carries. |
| **MoE Speculative Refusal Strings** | DONE (verified 2026-08-31) | `mtp_state.rs` and `speculation_policy.rs` were already corrected in `38553e1` ("a named block on a MoE install named the wrong obstacle"); `dflash_state.rs` already carries the policy-not-capability wording. Re-grepped the whole tree for the old "no batched kernel" MoE phrasing (source and `crates/cli`) and found none left. This row and section 3's matching TODO line were stale prose, not open work. |
| **`qwen4_exp` Chunked Prefill** | DONE 2026-09-05. Throughput still unmeasured. | Landed as `families/qwen4/prefill.rs`, the SEVENTH `ChunkedPrefillRunner` and the same Step 1 shape as the other six. **Both blockers this row used to name were wrong.** QSA needed nothing at all: `encode_full_attention_block` already takes `pass: &mut PassEncoder` and already owns its above-budget mid-layer commit, so calling it once per token in increasing order reproduces the sequential path exactly, and the shared `qsa_positions` buffer stays safe because the driver preserves gemma4's per-layer commit-and-wait ordering (no two QSA layers' writes are ever in flight at once). GDN needed nothing either. **What actually bit was PLE's `ngram_emb`, and it generalizes past this family**: its upload is `gpu::write_buffer_bytes`, a HOST write that executes the instant the encoding function runs rather than a dispatch queued for later, so it does not respect command-buffer commit order AT ALL. A single-row buffer shipped in the first cut and left every token but the last in a micro-batch computing PLE from the WRONG token's n-gram embedding, silently. Caught by `the_chunk_boundary_does_not_move_the_logits` at chunk span 2. Five buffers widened to `MAX_PREFILL_BATCH`, four encoder signatures threaded with row offsets, no new kernel. Verified byte-identical to sequential on 7 synthetic cases (span sweeps, the QSA sparsity boundary, the `banks == 1` fallback, and both seam refusals) and on the real install (greedy and sampled stdout md5-identical, 40-token prompt). **The first real-install run of the driver then found a CRASH at the DEFAULT slot count**, which seven green synthetic cases had missed: `routed_pipeline_banks` degrades to `banks == 1` whenever `expert_cache_slots < 2 * top_k`, and this checkpoint routes top-10 against the bench's pinned 16 slots, so the loop reserved the previous token's 10 slots and left 6 places for a token that can miss on 10 -- `expert cache cannot place requested misses`, AGENTS.md Gotcha 64 arriving on a second family and this time reachable without any exotic flag. Fixed by passing an EMPTY protect set at `banks == 1`, which that gotcha had already argued was correct (the branch calls `retire_routed` before planning, so nothing is in flight). The three other MoE drivers still pass it unconditionally and stay latent, unreachable at their own defaults. **The throughput measurement was then attempted and is NOT CITABLE**: three interleaved pairs on `medium-review` read 1.326x / 1.018x / 0.907x, and the SEQUENTIAL arm doing identical work spans 1.440x on its own, monotonically decreasing, so `crates/bench/CLAUDE.md` Gotcha 23's gate is not met. The ratios do not agree on a sign and dropping the first pair inverts the conclusion. A null is what the arithmetic predicts (chunking batches command buffers, not I/O, and 288 experts at top-10 against 16 slots make this prefill pread-bound -- the same reasoning that scoped out step 5's GGUF arm), but that is a hypothesis: the cheap next step is one `TURBOSPARK_PHASES=1` run read for its `pread` bucket, not another A/B. The 68 GiB install on a 36 GiB machine can never be page-cache warm, which is the likeliest source of the drift and is not fixable with more warmup. |
| **`qwen4_exp` GPU Top-K for Block Selection** | Unbuilt, unmeasured | Above `index_budget`, QSA's block scoring currently commits and waits on the host for `compute::select_blocks` once per QSA layer per token. A GPU top-k would remove that mid-layer commit (12 per token above budget on the real install) but has no evidence yet that it is the bottleneck -- measure before building, per this file's own Guiding Principle 1. |
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
   - ~~Run step 6 throughput A/B on a quiet machine~~ ATTEMPTED 2026-09-05, direction confirmed (never a regression), magnitude contaminated by background CPU load. Re-run owed on a machine confirmed idle by `ps`, not just `pmset -g therm`. See the Active Tasks table row and `docs/BATCHED_PREFILL.md`.
   - Implement Step 4 batched attention kernel (widening of `attention_decode_partial`); real new-kernel work, own pass.
   - The GGUF (Q4_K/Q6_K) arm of Step 5 is deliberately NOT queued: measured, its ceiling is the un-batchable `pread` (37.1% of its prefill) rather than the kernel, and it would cost two kernels rather than one.
2. ~~**Phase-2 `top_k` Specialization**~~ DONE 2026-08-31; see section 10.
3. **MoE Drafter & Speculation Cleanup**:
   - ~~Fix stale refusal strings in `mtp_state.rs`, `dflash_state.rs`, `speculation_policy.rs`~~ DONE, already true before this pass -- see the "MoE Speculative Refusal Strings" row in the Active Tasks table.
   - Investigate ingestible MoE drafters (e.g. Ornith MoE MTP head conversion or lightweight block drafter).
4. **Streamable-MoE Bring-Up**:
   - Test and benchmark `gpt-oss-120b-MXFP4.gguf` under mapped residency (evaluates SSD vs page cache bound decode).

---

## Artifact State & Disk Usage (RE-CHECKED 2026-09-05)

Disk space is a key constraint for downloading large checkpoints and running cross-engine KL reference dumps.

**Re-derived from `ls ~/models` and `ls ~/.turbospark/models` rather than carried
forward.** The 2026-08-29 version of this table named four installs that are gone
and omitted four that are present, which is the same drift `CLAUDE.local.md`
recorded on 2026-08-30. Re-run those two commands before trusting any row here:
this table has now been wrong twice, and nothing goes red when it rots.

| Path in `~/models/` | Size | Status / Associated Targets |
|---|---|---|
| `gemma4.gturbo` | 13G | PINNED: smoke, memory oracle, sensitivity proof, mapped residency |
| `ternary27b.gturbo` | 7.1G | `ternary_{quality_gate,memory_oracle}` |
| `qwen38-27b.gturbo` | 14G | `qwen38_{quality_gate,memory_oracle}`, steering baseline |
| `qwen38-27b-mtp.gturbo` | 14G | MTP speculative validation. **Duplicated**: an independent second copy sits at `~/.turbospark/models/qwen38-27b-mtp.gturbo`, and they are NOT APFS clones (`du -sch` over both reads 29G against a 14G each, i.e. ~2x the sum). Deleting either frees ~14G. The catalog's `installed.json` points at the `.turbospark` one. |
| `qwen38-27b-dflash2.gturbo` | 15G | DFlash2 block drafter validation |
| `qwen38-gguf.gturbo` | 15G | Qwen 3.8 27B via GGUF intake. Present on disk and in `installed.json`, absent from every previous version of this table. |
| `steering-vectors/` | ~3M | `ocean` and `register` legacy vectors for regression tests |
| `gguf-ref/`, `qwen38-mtp-ref/`, `skill-state-probe/` | -- | Reference and probe sidecars (the llama.cpp KL bytes, the MTP reference, `docs/SKILL_STATE.md`'s corpus). Not `.gturbo` installs. |
| `qwen38-27b-vision.gturbo` | 15G | The ONLY install carrying a vision tower. `vision_tower_parity`, `vision_logit_dump`, the CLI's `--image` / `--image-batch` runs and the server's `real_backend_reads_an_image_sent_over_both_endpoints`. **Deliberately separate from `qwen38-27b.gturbo`**: that one backs the frozen quality-gate and memory-oracle rows, and ~0.9 GiB of tower would force a re-freeze for a component neither gate exercises. Do not merge them. |
| `vision-probe-qwen38/` | 4.8G | `mlx-community/Qwen3.8-27B-4bit`'s tower alone, pinned to revision `3e6447f0`, which is what `vision_tower_parity` pairs against. 4.8G for 879 MiB of tensors because `vision_tower.*` is not contiguous in this checkpoint and the fetch spans min..max offset (over-fetches, does not miss data). |
| `vision-probe/` | 879M | `prism-ml/Bonsai-27B-mlx-1bit`'s tower, F16. What `docs/VISION_PHASE0.md` items 3 and 4 were measured on. Contiguous, so 879M for 879 MiB. |

The `turbospark-model pull` store, `~/.turbospark/models/`, which several rows
above and below depend on and which no previous version of this table listed:

| Path in `~/.turbospark/models/` | Size | Status / Associated Targets |
|---|---|---|
| `qwen4-reap288.gturbo` | 68G | **The largest artifact on this machine and the subject of section 2.** Qwen3.8-Flash-Next REAP-288, `top_k_experts=10`. Backs `qwen4exp_{quality_gate,memory_oracle}`, the QSA force-dense probe, and the chunked-prefill throughput measurement that is still owed. |
| `qwen3moe.gturbo` | 17G | The `llama` flow's `Qwen3Moe` half. `qwen3moe_{quality_gate,memory_oracle}`, and one of the two real installs mapped expert residency was verified byte-identical on. |
| `gptoss-20b.gturbo` | 11G | `gptoss_{quality_gate,memory_oracle}`, steering validation, PF-02 Step 5's MXFP4 arm, and the second mapped-residency verification. Reads 11G here against the 14G this table used to claim. |
| `qwen38-27b-mtp.gturbo` | 14G | The duplicate noted above. |

**GONE from disk as of 2026-09-05**, each named as present by the previous
version of this table, and each with rows elsewhere in this file that silently
depend on it: `museglimmer-30b.gturbo` (15G), `ornith9b.gturbo` (8.9G),
`ornith35b.gturbo` (18G), `ornith35b-gguf.gturbo` (34G), plus `mistral7b.gturbo`
and `llama3-8b-instruct.gturbo` from the store. Three consequences to carry
rather than rediscover:

- The **`power.sh` Sweep for Remaining Models** row names `ornith9b` and
  `ornith35b`. Neither is on disk, so that row is a re-pull before it is a
  capture.
- Section 9's one open verification gap (mapped residency on `qwen`'s own
  `QwenGdnMoe` family) still has **no install to close it with**. Ornith 35B was
  already gone when that wiring landed on 2026-08-30 and has not come back.
- `museGlimmer` and both Ornith rows in `docs/BENCHMARKS.md` and
  `docs/POWER_BASELINE.md` are now unreproducible on this machine without a
  re-pull. The numbers stand; the ability to re-derive them does not.

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

- **Status**: Steps 1-3 (chunked prefill, batched routed pair) and Step 6 (batched resident GEMVs) shipped for Gemma 4. Measured 1.54x speedup on Gemma 4 `long-synthesis`. `--prefill-chunk` is wired as the default (2026-08-26): `RealForwardRunner::supports_chunked_prefill()` is the one predicate both the CLI's default routing and `ChunkedPrefillRunner::prefill_chunk`'s own refusal check, so a caller who never typed the flag cannot see a family it doesn't serve, and `TURBOSPARK_PREFILL_CHUNK`'s existing hard-fail-on-unsupported-family A/B-seam contract is unchanged. **Every MoE-capable family now serves chunked prefill (2026-08-27)**: `muse_glimmer` landed the same no-mid-layer-commit shape as dense `llama` (no router), and the MoE half of `families/llama/` (Mixtral, Qwen3MoE) plus `gpt-oss` landed Step 1 alone -- a per-layer command buffer for attention-and-router plus a per-token routed loop pipelined via a shared `RoutedSlot` module (`moe_prefill_pipeline.rs`, extracted from Gemma 4's driver) -- with NO new kernel needed, since Step 1 reuses the same layout-agnostic per-token dispatch the sequential decode path already uses. **Step 5's MXFP4 arm landed 2026-08-27 at 1.31x** on the real 20B install (`families/gptoss/moe_batch.rs` over `moe_prefill_batch_gguf.metal`), so TWO families now serve the batched routed pair: Gemma 4 on INT4-affine blobs and `gpt-oss` on MXFP4 ones. Wiring the second one exposed that the unwired MoE `llama` family had been SILENTLY IGNORING `TURBOSPARK_ROUTED_BATCH` rather than refusing it; that is a named refusal now. **The DENSE half of the qwen linear-attention flow landed 2026-08-29** (`families/qwen/prefill.rs`, `qwenGdnDense`): the same Step 1 shape as dense `llama`/`muse_glimmer`, no new kernel and no new buffer -- the recurrent GDN state's correctness follows from calling the existing per-token kernels in order rather than from any batched machinery, and the driver refuses an image prompt or an open drafter by name rather than growing a second embedding call site or silently starving a drafter's aux capture. Only the MoE half of qwen (`qwenGdnMoe`) remains unsupported. Verified byte-identical against the pre-wiring sequential path on real installs (`~/models/gemma4.gturbo`, `~/.turbospark/models/mistral7b.gturbo`, `~/models/museglimmer-30b.gturbo`, `~/.turbospark/models/gptoss-20b.gturbo`, a freshly-pulled `Qwen/Qwen3-30B-A3B-GGUF`, `~/models/qwen38-27b.gturbo`; greedy and sampled, stdout md5-identical both ways), plus the standing gemma4 chunked parity suite and new per-family parity suites (chunk-span sweep, cache-too-small-to-pipeline case where the family has a slot cache, MoE-still-refused case where applicable).
- **Reference**: `docs/BATCHED_PREFILL.md`.
- **Detailed TODO**:
  - [x] **Kernel-quality reference measured 2026-08-29** (`scripts/mlx_qmm_reference.py`, `docs/BENCHMARKS.md` "The reference curve, measured rather than inferred"). Replaces the extrapolated 2.2x/2.3x pair with 1.50x kernel + 2.00x width, and moves the saturation point to **M=32**. Both engines' M=1 baselines agree to 1.16x, which is what makes `c` comparable at all.
  - [ ] **Step 7 (matrix-path re-tile)**: the only untried lever on `dequant_int4_gemm_mma`. Four SIMD groups (`WM = WN = 2`, 128 threads) with `FC_MMA_STAGE_X` ON, changed TOGETHER -- the three sub-levers are not independent and each was measured to a dead end alone (Do Not Revisit 13, 14). Gate: `c_of_m_matrix_against_exact_at_qwen38_shapes`'s third column below 1.00. It stays PREFILL-ONLY whatever it measures, since the kernel is not bit-exact against the GEMV (AGENTS.md Gotcha 27), and `MAX_BATCH_ROWS = 16` still caps the width term regardless.
  - [ ] **Step 6 Throughput A/B**: Measure `TURBOSPARK_BATCHED_GEMV=1` on quiet machine.
  - [ ] **Prefill Energy Capture**: BLOCKED on the bench gap, see the Active Tasks table row -- `scripts/power.sh` cannot see chunked prefill until `crates/bench`'s model mode calls `run_raw_completion_chunked`. The unblocking flag (`--prefill-chunk`, default OFF) is in progress.
  - [ ] **Step 4 (Batched Attention Kernel)**: Widen `attention_decode_partial` to process M query rows per KV chunk. Deferred as real new-kernel work (new function-constant axis, per-row online-softmax state generalized to M rows, real register-pressure risk per `dequant_int4_gemm_simd`'s spill history), not attempted alongside the default-on wiring. **Measured LOW VALUE on `qwenGdnDense` (2026-08-29): 2.6% of prefill, in the 16 of 64 layers that have attention at all.** Pick the family before picking this item; see the Active Tasks row.
  - [ ] **Step 5 (GGUF Routed Pair Widening)**: Widen the BATCHED routed kernels (steps 2/3, `TURBOSPARK_ROUTED_BATCH`) to GGUF and MXFP4 block types if/when the throughput they add becomes a priority; Step 1's per-token routed loop already runs on every layout, so this is a speed lever, not a correctness gap.
    - **SCOPED BY MEASUREMENT 2026-08-27, and the order is the opposite of this item's title** (`docs/BATCHED_PREFILL.md`, "Step 5's two arms, measured before building either"). Do **MXFP4 (`gpt-oss`) first**: its routed pair is 61.4% of prefill GPU device time against Gemma's 38.2%, its un-batchable `pread` bucket is 8.2% against 25-37% (32 experts at top-4 give a 96.7% hit rate), and it is the ONLY family that reaches M=16 -- both 128-expert families cap at M=8 on `union(M) <= slot_count`. It also needs ONE block type for both phases.
    - [x] **MXFP4 arm DONE 2026-08-27**, and it measured **1.31x** on the real 20B install against Gemma 4's 1.19x for the same step -- the share arithmetic held. `crates/gpu/src/shaders/moe_prefill_batch_gguf.metal` plus `crates/runtime/src/families/gptoss/moe_batch.rs`, behind the existing `TURBOSPARK_ROUTED_BATCH` seam. Bit-exact against M decode-pair calls at the real shape, byte-identical greedy AND sampled on the real install. The pair's own `c(M)` was later measured on the bench's interleaved arms at 0.76/0.74/0.73/0.74 (M=2/4/8/16); the 0.67 first inferred for c(16) from the phase table's device-time rows was 9% optimistic, and the within-a-point agreement with the affine pair holds on same-day same-instrument terms (affine c(8) read 0.73 beside MXFP4's 0.73). The occupancy-not-weight-amortization conclusion stands.
    - [x] **Silu-flag hygiene on the MXFP4 pair DONE 2026-08-28**: `moe_prefill_batch_gguf.rs`'s four public functions no longer take `use_silu`; the specialization is a module-private `const USE_SILU: bool = true` feeding `moe_function_constants`/`constants_key`, so the argument encoder and dispatches cannot disagree and no caller can compile a second copy of the seven-file MSL concatenation mid-prefill. Two things the task's premise got wrong, both settled by grep and by the parity suite: the flag is NOT fully dead for MXFP4 (`moe_activate_mxfp4`'s PLAIN arm falls back to `moe_hidden_activation`, which reads `FC_MOE_ACT_SILU`, and the parity file's plain-activation case reaches it), and the parity suite had been holding bit-identity at silu=false against a production that runs silu=true. Both arms of the parity file now compile the specialization production compiles (the decode-pair oracle's args flipped to true alongside the const), all 7 cases green byte-for-byte. Production output cannot move: the family runs `Mxfp4Activation::GPT_OSS`, where the flag selects a dead branch.
    - **The GGUF (`qwen3moe`) arm is the weak one and may not be worth building.** 37.1% of its prefill is expert `pread`, which batches not at all and is already at the end of its lever (75.7% hit rate at the maximum 32 slots). Its routed device share and reachable M are both Gemma's, and it needs TWO kernels (Q4_K gate/up, Q6_K down).
    - Before quoting any end-to-end number for either arm, measure `c(M)` on the real shape. The only measured `c(M)` anywhere is INT4-affine at Gemma's shape, and this document already recorded being wrong by 2.3x once from borrowing a proxy across kernels.
    - [x] **MXFP4 `c(M)` on the bench harness** DONE 2026-08-27: `moe_prefill_batch_bench.rs` has an `mxfp4` module at the real D=2880 F=2880 top_k=4 shape (unions 6/10/13/17), reading 0.76/0.74/0.73/0.74 at M=2/4/8/16 across four serial runs with spread under 0.01. The inferred 0.67 was 9% optimistic (a real cross-instrument gap); the within-a-point cross-block-type agreement holds when both arms are measured on the same instrument the same day. The two benches in that file are serialized by a static mutex now -- the first `-- --ignored` run put both on the GPU concurrently and read affine c(8) as 0.38 with no error (`docs/BATCHED_PREFILL.md`).
    - [x] **Separate the 28% expert-miss drop** DONE 2026-08-27, and the measurement REFUTED the doc's reasoned attribution. Two corrections to how this task was written. The seam it named did not exist: `TURBOSPARK_ROUTED_PIPELINE` was read by Gemma 4's sequential decode alone, and the gpt-oss per-token prefill arm passed its `protect` set unconditionally -- so "no new code" was false, and the seam had to be wired first (`families/gptoss/prefill.rs`: off means banks = 1 AND an empty protect set, together, since retire-before-plan is what makes the empty set sound). And the hypothesis it carried was wrong: with the protect set off, misses read 9,400 against the control's 9,024 (stdout md5-identical), recovering NOTHING of the 9,024 -> 6,478 drop. The drop is real union dedup -- the one measured family-scoped exception to "the union saves nothing", consistent with AGENTS.md Gotcha 54's own bound: gpt-oss is the one family whose full 16-token window union (17.2) fits the slot cache (24) while the cache does not hold the expert table (32), so intra-window eviction re-reads exist AND the union can recover them (`docs/BATCHED_PREFILL.md`).
  - [x] **Default-On Configuration**: `--prefill-chunk` wired in the CLI and (with no per-request flag, matching the rate cap and guardrails toggle) automatically in the server, both gated on `supports_chunked_prefill()`.
  - [x] **Family Widening (complete)**: Gemma 4, both halves of `llama` (Mistral, Llama 2/3.x, Mixtral, Qwen3MoE), `muse_glimmer` and `gpt-oss` all land Step 1. Only the qwen linear-attention flow (`qwenGdnMoe` / `qwenGdnDense`) remains, and it was not attempted this pass.
  - [x] **Qwen Dense Chunked Prefill (2026-08-29)**: `families/qwen/prefill.rs` lands `qwenGdnDense` as Step 1, no new kernel, no new buffer -- see the status paragraph above and `docs/BATCHED_PREFILL.md`'s "sixth flow" entry for the design (GDN state ordering, the vision and open-drafter refusals). `qwenGdnMoe` (Ornith 35B) is not attempted this pass, matching how `llama`'s two halves landed as separate steps.

---

### 2. Streamable-MoE Candidate Bring-up

- **Context**: Future large/low-memory models on Mac Unified Memory require fine-grained MoE architecture where expert weights stream or map on demand.
- **Detailed TODO**:
  - [ ] **`qwen4_exp` (Qwen3.8-Flash-Next)**: 48 layers, fine-grained MoE (288 or 512 experts), GDN + gated attention, sigmoid-gated norm (not silu); see `docs/QWEN4_PHASE0.md` for the full config/tensor-layout fact-finding. **Intake landed 2026-08-31**: family/config surface, Phase 0 config gate, n-gram table classification and on-disk layout, streaming writer, manifest wiring. **Decode flow landed 2026-09-02** (`families/qwen4/`, Phase 3): a complete per-token forward pass -- hyper-connections, the sigmoid-gated GDN norm this family needed (`gdn_gated_norm_sigmoid`, `crates/gpu/CLAUDE.md` Gotcha 12, no longer open), QSA-as-dense attention, gated MoE, the PLE n-gram chain -- wired into `RealForwardRunner` and verified against a synthetic fixture (12 reachability/refusal cases plus a mutation-checked frozen digest). **`ALLOWED_CACHE_SLOTS` widened to `[8,...,128]` 2026-09-03** (Phase 4), so the 6.25-11%-of-table ceiling this row used to name no longer bounds residency the way it did; an open-time refusal now fires when `expert_cache_slots < top_k_experts` instead. **The real install (`~/.turbospark/models/qwen4-reap288.gturbo`, 68G, REAP-288, top_k=10) now OPENS as of 2026-09-04**, after `moe_phase2_down_reduce_k8`'s fixed 8-slot dispatch was widened to a runtime width sized to the caller's own `top_k` rather than a new fixed ceiling (`crates/gpu/CLAUDE.md` Gotcha 13). **The router-dtype blocker this row used to name was fixed the same day** (`612be53`, `27b666f`): the safetensors repack orchestrator now force-quantizes this checkpoint's router and shared-expert gate to INT8-affine, since its from-safetensors publish shipped both raw BF16 where every other MoE family's MLX conversion pre-packs the router as `U32`. **Decode runs, and QSA landed 2026-09-05**: `families/qwen4/attn.rs` now runs sparse block-selected attention above `index_budget` in place of the dense fallback, verified on the real install with a coherent smoke run, a frozen quality gate and memory oracle, and a KL-based force-dense probe. **Chunked prefill landed 2026-09-05** (`families/qwen4/prefill.rs`, the seventh `ChunkedPrefillRunner`), and it needed NEITHER of the two things this bullet used to predict: the indexer's per-token key write and block pooling stayed per token, and no per-QSA-layer position list was required, because `encode_full_attention_block` already owns its own above-budget mid-layer commit and the driver preserves gemma4's per-layer commit-and-wait ordering. What it did need was a widened PLE `ngram_emb` -- a HOST `write_buffer_bytes` that ignores command-buffer commit order, so a single-row buffer silently fed every token but the last of a micro-batch the wrong n-gram embedding. Open: the THROUGHPUT number for that driver (unmeasured; the driver's whole purpose), a GPU top-k to remove the above-budget mid-layer commit (unbuilt, unmeasured), and the bench-window decision (`QWEN4_EXP_MAX_CONTEXT` stays 2,048 so the frozen rows keep their meaning). See `docs/QWEN4_EXP.md` for the full mechanism and the Recent Landings bullet above for the summary.
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
  - [x] **Speculation String Cleanup**: DONE (verified 2026-08-31, see "MoE Speculative Refusal Strings" in the Active Tasks table above). `mtp_state.rs`/`speculation_policy.rs` were fixed in `38553e1`; `dflash_state.rs` already had the correct wording; `crates/cli` carries no matching string at all.

---

### 4. Server Concurrency & Fairness

- **Current State**: Single mutex-serialized runner per process.
- **Detailed TODO**:
  - [x] **Prefix KV Reuse on the Server**: DONE 2026-09-01. `RealChatModel::open` takes a `prefix_reuse: bool` and calls `RealForwardRunner::set_prefix_reuse` once at open, exercised on the same `run_raw_completion_chunked` path `crates/ffi`'s `open.rs` reaches (Gotcha 16 there). **Unlike the FFI binding, this got a real `--prefix-reuse on|off` flag rather than an unconditional `true`**, because the FFI's one-session-per-model shape has no cross-conversation hazard and this server's one-runner-serves-every-client shape does: two interleaved unrelated conversations each discard the other's reusable prefix, so the match rate is traffic-mix-dependent even though the mechanism itself is provably lossless (a mismatch always falls back to a full reset, never stale KV). Defaults ON, matching this item's own framing as the largest available TTFT win on this surface. **The `crates/invocation`'s-five-places clause in the prior version of this row was wrong and is now removed**: `crates/server` has its own flat parser and no dependency on that crate at all -- the flag follows `--guardrails on|off`'s exact shape in `args.rs` instead (`ModelArgs` field, `USAGE` line, one `match` arm). Observability landed as `ServerEvent::Generated.reusedPrefixTokens` rather than a CLI-style stderr line, since a server has no terminal an operator is watching; `tests/real_backend.rs`'s `real_backend_reuses_kv_across_two_chat_turns` asserts the count is nonzero on a real Gemma 4 install's second turn, not just that both requests returned 200 (the same trivially-passing-test trap `crates/runtime/CLAUDE.md` Gotcha 30 warns against). See `crates/server/CLAUDE.md` Gotcha 31. **Option 3 below is the actual fix for multi-conversation reuse and has now landed too.**
  - [ ] **Request Queue & Fairness (Option 1)**: Implement request FIFO queue with streaming-aware fairness and cancellation handling.
  - [ ] **Multi-Runner Pool (Option 2)**: Support configurable N runners (multiplies KV cache and slot cache memory by N).
  - [x] **Multiplexed Session State (Option 3)**: DONE 2026-09-01. A swap-based bounded pool (`crate::session_pool::SessionPool`, `--session-slots N`, default 1, no pool), not a rewrite of every family's dispatch code and not a duplicate `RealForwardRunner` per session (Option 2's cost): the runner keeps its existing single `kv`/`real_qwen.gdn`/`kv_prefix` fields as the "live" session and holds `session_slots - 1` PARKED `SessionSlot`s, moved onto the live fields via `std::mem::replace` (an O(1) struct swap, never a memcpy -- the exact cost a per-switch KV host-copy was rejected for during design research). `RealForwardRunner::select_session` (called first inside `try_reuse_prefix`) swaps in whichever parked slot best matches the incoming prompt; `reset()` parks the outgoing live session instead of clobbering it, which is the path that actually fixes Gotcha 31's stated stomping problem, since every caller that finds no live match (including the speculative loop) reaches `reset()`. A GDN family's recurrent state swaps as a second LIVE `GdnStateManager` rather than the expensive `GdnSnapshot` host-copy pair speculative rollback uses, costing one extra allocation at open and zero per switch. **Two real bugs were found by the real-install gate (`crates/server/tests/real_backend.rs`'s `real_backend_reuses_kv_across_two_interleaved_conversations`), neither visible to the synthetic fixture written alongside the feature**: a shallow coincidental match (e.g. two unrelated conversations sharing only a chat template's opening tokens) rewinding a live session in place instead of parking it, destroying real content; and an overly strict `back > keep` discriminator that, once fixed for the first bug, also undid `select_session`'s own correct swap decisions, needlessly cascading into evicting a third, unrelated conversation to serve a request that already had its real match in hand. See `crates/runtime/CLAUDE.md` Gotcha 32 (the mechanism and both bugs) and `crates/server/CLAUDE.md` Gotcha 32 (the flag). Verified: full workspace suite/fmt/clippy green; new synthetic tests in `crates/runtime/tests/session_pool.rs` (byte-identity at the default, the destructive-shallow-match regression on both the sequential and chunked-prefill paths, the same regression on a GDN family, eviction reporting); the real-install gate above, plus the two pre-existing prefix-reuse tests, all pass on a real Gemma 4 install; `qwen38_quality_gate`/`qwen38_memory_oracle` reproduce their frozen rows exactly at the default `--session-slots 1`, confirming the mechanism moves no numerics or footprint when unused.

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

- **Status**: LANDED ON `main` 2026-08-30. The base feature spent a day as a commit nobody could reach: written 2026-08-23 on an unmerged branch `cache-policy`, then rebased onto `expert-residency` in a worktree (`../mrefrust-residency`) that was later removed WITHOUT the branch ever being merged -- the branch ref itself was gone by 2026-08-30, and the tip commit (`702716b`) survived only as a dangling git object one `git gc` away from being pruned (AGENTS.md Gotcha 13's exact failure shape). Recovered with `git branch expert-residency 702716b` and merged into `main`: 8 files conflicted against the router-lookahead PILOT probe and prefix-KV-reuse work that had landed on `main` in the meantime, all independent, non-overlapping additions that `git merge`'s `ort` strategy resolved cleanly except two doc files (`AGENTS.md`'s doc index, `crates/streaming/CLAUDE.md`'s gotcha numbering), fixed by hand. Full workspace suite, fmt, clippy and the cross-target check all green; `mapped_expert_probe`/`mapped_experts`/`mapped_expert_residency` pass against the real install; Gemma 4 quality gate and memory oracle reproduce their frozen rows exactly on the default arm; greedy and sampled smoke are byte-identical between the default and `mapped` arms. Seam: `TURBOSPARK_EXPERT_RESIDENCY=mapped`, off by default, Gemma 4 only, both arms produce identical tokens.
- **Numbers, RE-MEASURED 2026-08-29 on AC at the `auto`-resolved 32 slots**: peak `phys_footprint` **3,652 -> 559 MiB** (a 3,093 MiB saving against a predicted `32 x 30 x 3.2 MiB` = 3,072 MiB slot cache; both arms reproduce to under 0.6%) and decode **53.8 -> 68.7 tok/s, 1.28x**. The superseded pair was 3,721 -> 606 MiB and 51.9 -> 69.8 tok/s, taken before chunked prefill became the default. **Read the throughput figure with its caveat**: the capture was not on an idle machine, contention costs the `pread` arm more than the mapped one, so 1.28x is a ceiling rather than a floor (`docs/EXPERT_RESIDENCY.md`).
- **Detailed TODO**:
  - [x] **Wire Remaining MoE Families, DONE 2026-08-30**: `qwen`, `llama` (both `Llama` and `Qwen3Moe`), and `gptoss` all carry the same fork Gemma 4's `moe.rs` does, replacing `mapped_residency_refusal`'s named refusal for each rather than adding a branch beside a silent path. `qwen`'s and `gptoss`'s batched-routed drivers (`moe_batch.rs`) each gained the same mapped-vs-batched conflict guard Gemma 4's carries; `llama` needs none, since it has no batched-routed driver to conflict with. Verified: full workspace suite/fmt/clippy green; per-family synthetic tests (`crates/runtime/tests/mapped_expert_residency_{qwen,llama,gptoss}.rs`) pass, each opening under `TURBOSPARK_EXPERT_RESIDENCY=mapped` and asserting finite non-zero logits plus (where applicable) the batched-conflict refusal firing by name; real-install verification landed for two of the three -- `~/.turbospark/models/qwen3moe.gturbo` (the `llama` flow's `Qwen3Moe` half) and `~/.turbospark/models/gptoss-20b.gturbo` both reproduce byte-identical greedy output between streamed and mapped arms, with `TURBOSPARK_PHASES=1` confirming 0.0 ms `pread` time and a 100% expert-cache hit rate under mapped mode on both; gptoss's frozen `quality_gate` and `memory_oracle` rows reproduce exactly with the seam unset, confirming the default arm is unmoved. **`qwen`'s own family (`QwenGdnMoe`) has no real install left on this machine** -- Ornith 35B, which this TODO's own text called "on disk; immediate win" when written, is gone by the time the wiring landed (`CLAUDE.local.md`'s artifact inventory has drifted and needs re-checking against `ls ~/models` / `ls ~/.turbospark/models` before the next session trusts it) -- so that family is verified on the synthetic fixture only.
  - [ ] **Memory Pressure Eviction Benchmark**: Measure paging overhead and latency impact when operating under OS memory pressure.
  - [ ] **`auto` Policy Resolution**: Automatically select `mapped` when machine memory accommodates the full model file cache, falling back to `streamed`.
  - [ ] **CLI Flag**: Expose `--expert-residency auto|streamed|mapped` on `turbospark-check` and `turbospark-server`.
  - [ ] **`madvise(MADV_WILLNEED)` Prefetching**: Evaluate background advice on routed offsets to minimize cold-start fault overhead.
  - [ ] **Mapped Memory Oracle Rows**: Record separate frozen memory baseline rows for mapped residency mode across all MoE families.

---

### 10. Phase-2 `top_k` Specialization

- **Status**: DONE 2026-08-31. Landed as `moe_phase2_down_reduce_k8_mxfp4`-only
  (`crates/gpu/src/shaders/moe_gguf.metal`, `crates/gpu/src/moe_gguf/mxfp4.rs`),
  the kernel `gpt-oss` actually dispatches, not the whole
  `moe_phase2_down_reduce_k8` family this section's title names -- the
  affine kernel and the other GGUF phase-2 kernels (Q8_0/Q4_K/Q6_K/IQ4_NL)
  serve no family with `top_k < MAX_STREAMED_EXPERTS` today (Gemma 4 and
  `qwen3moe` both route top-8 of 128) and were left untouched.
- **Background**: `moe_phase2_down_reduce_k8_mxfp4` executed 8 down-GEMVs
  regardless of model `top_k`. `gpt-oss` (`top_k = 4` of 32 experts) wasted
  the dequant-and-dot-product on the 4 unused slots every decode step.
- **The mechanism, and the bug it caught on the first attempt**: `FC_MOE_TOP_K`
  is baked via `phase2_function_constants`, and the kernel masks the compute
  (`if (sg_idx < KK) { ... }`) while still reaching the unconditional
  `threadgroup_barrier` for all 256 threads -- masking the compute rather
  than returning early is what avoids Metal's undefined behaviour for
  divergent barrier participation. The first cut baked `top_k` by turning on
  the SHARED `FC_MOE_USE_FC` gate, which `moe_fc_d`/`moe_fc_f` also read --
  so `D`/`F` silently resolved to their baked-but-never-set value of zero and
  the kernel returned before writing anything. Caught immediately by
  `moe_gguf_parity.rs`'s existing tests (`all_eight_mxfp4_slots_participate`,
  `the_mxfp4_decode_pair_matches_the_cpu_reference`,
  `the_oai_activation_matches_its_reference`,
  `the_silu_activation_constant_reaches_the_mxfp4_kernels`), all four reading
  back `0` where the CPU reference expected a real value. Fixed by resolving
  `top_k` with its own `is_function_constant_defined(FC_MOE_TOP_K)` check,
  independent of `FC_MOE_USE_FC`. See `crates/gpu/CLAUDE.md` Gotcha 3.
- **Detailed TODO**:
  - [x] **Specialization Pipeline**: `moe_phase2_down_reduce_k8_mxfp4` masks
    compute for `sg_idx >= FC_MOE_TOP_K`, provably bit-identical to the
    unconditional 8-slot reduce (`0 * value == 0` whether or not the masked
    slots are computed).
  - [x] **Widen Pipeline Cache Key**: `phase2_constants_key` is a
    phase-2-only key (`[use_silu, top_k]`), deliberately NOT a widening of
    the shared `constants_key`/`moe_function_constants` phase 1 and every
    other GGUF pair's phase 1/2 reuse -- that would force every one of them
    to carry a `top_k` byte only this one kernel reads.
  - [x] **Benchmark and Validate**: full `turbospark-gpu` and
    `turbospark-runtime` suites green (`cargo test -p turbospark-gpu` /
    `-p turbospark-runtime`, no filter); workspace build/fmt-check/clippy
    green. Real-model verification on `~/.turbospark/models/gptoss-20b.gturbo`:
    greedy and sampled stdout md5-identical against a pre-change binary built
    in a clean `git worktree` (200 new tokens each), `gptoss_quality_gate`'s
    frozen reference-answer perplexity (12.0801) and all three digests
    reproduced exactly. Throughput, three interleaved pairs, 300 new tokens
    greedy, real install, not a quiet machine: **33.4-34.0 -> 37.5-38.1
    tok/s, a consistent ~1.13x** end-to-end decode win (the kernel itself
    drops close to half its work, but phase 2 is one term among attention,
    phase 1, the router and sampling, so the end-to-end number is smaller
    than a naive 2x and that is expected rather than a shortfall).

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
4. **Expert Prefetch / Speculation**: TWO different predictors, both measured, both negative. The inherited one copies layer L's selected expert IDs and hits 7% (Jaccard 0.039). The second, measured here 2026-08-29 against colibri's PILOT, RUNS layer L+1's router GEMV on layer L's post-attention residual and genuinely works -- 70.6% recall, reproducing colibri's reported 71.6% on a different architecture, covering 60.4% of misses at 32 slots. It still loses, on a different axis: a prefetcher's COST scales with prediction width while its BENEFIT scales with the miss rate, and at an 84.6% cache hit rate every `PILOT_K` from 1 to 8 reads MORE total expert bytes than the demand path (1.03x to 1.85x). No operating point pays. Reversal condition: a machine where the expert read is genuinely disk-bound rather than a page-cache memcpy. Probe stays wired (`TURBOSPARK_PILOT_PROBE`, `=self` to validate the instrument); method and full sweep in `docs/EXPERT_ROUTING.md`.
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
- Upstream Swift UI: Out of scope (superseded by this repository's native `TurboSparkApp` desktop application and `.app`/DMG release packaging).
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
- **Phase-2 `top_k` Specialization (2026-08-31)**: `moe_phase2_down_reduce_k8_mxfp4` skips the down-GEMV for `gpt-oss`'s unused routed slots (top-4 of 32) instead of reducing all 8 unconditionally. Verified bit-identical to a pre-change binary on the real `gpt-oss-20b` install (greedy and sampled, plus `gptoss_quality_gate`'s frozen rows); measured ~1.13x end-to-end decode throughput over three interleaved pairs. See section 10.
- **`crates/ffi` Prefix KV Reuse and Chunked Prefill (2026-09-01)**: `open.rs`'s `open()` now calls `set_prefix_reuse(true)` unconditionally (the one deliberate divergence from the CLI's single-shot `open_session`), and `generate.rs` now routes non-speculative real-model turns through `run_raw_completion_chunked_cancellable` whenever the install's family supports it and the turn carries no image -- both were previously CLI/server-only. `TurboSparkApp`'s multi-chat sessions, which hold one long-lived session per loaded model across a chat's whole lifetime, get both wins with no app-side code change. `GenerateResult` gained a `reusedPrefixTokens` field for observability, matching the CLI's own `[prefix-reuse]` footer line. Verified: `cargo test -p turbospark-ffi` (56 tests), workspace build/fmt/clippy, `make swift-test-real MODEL=~/models/gemma4.gturbo` (48 tests including a new `testASecondTurnReusesThePreviousTurnsKV`, which measured 14/18 reused prompt tokens on a real two-turn conversation), and the full `TurboSparkApp` suite (410 tests). See `crates/ffi/CLAUDE.md` Gotcha 16 and `docs/SWIFT_BINDINGS.md`'s "Generating" section.
- **`qwen4_exp` (Qwen3.8-Flash-Next) Intake (2026-08-31)**: family/config surface, Phase 0 config gate, n-gram table classification, on-disk layout and refusals, and a streaming self-checked writer that never holds the whole table in memory. Manifest wiring reaches all three consumers. Decode flow landed 2026-09-02; see below and section 2.
- **`qwen4_exp` Decode Flow Wired End to End (2026-09-02)**: `families/qwen4/` (hyper-connections, sigmoid-gated GDN norm, QSA-as-dense attention, gated MoE, PLE n-gram chain) wired into `RealForwardRunner`, Phases 2-3. Verified against a synthetic fixture only (12 reachability/refusal cases, a mutation-checked frozen digest); no real checkpoint had been tried yet.
- **`qwen4_exp` Memory Policy and Safetensors Intake (2026-09-03)**: Phase 4 widened `ALLOWED_CACHE_SLOTS` to `[8,...,128]` (this family's 288-expert table costs 126.6 MiB/slot, so the old 32-slot ceiling capped residency at 11%) and added an open-time `expert_cache_slots < top_k_experts` refusal. Wired into the safetensors install path the same day.
- **`qwen4_exp` Real Install Opens (2026-09-04)**: widened `moe_phase2_down_reduce_k8`'s fixed 8-slot dispatch to a runtime width sized to the caller's own `top_k`, unblocking `~/.turbospark/models/qwen4-reap288.gturbo` (REAP-288, top_k=10) at OPEN with no change to any pre-existing top_k=8 family's dispatch. See section 2.
- **`qwen4_exp` Router Dtype Fix (2026-09-04)**: `612be53` and `27b666f` taught the safetensors repack orchestrator (`crates/repack/src/gemma4_checkpoint/orchestrate.rs`) to force-quantize `mlp.gate.weight` and `mlp.shared_expert_gate.weight` to INT8-affine for `Qwen4Exp`. Root cause: this REAP-288 checkpoint is a from-safetensors publish that ships both gating matrices raw BF16, where every other MoE family's upstream MLX conversion happens to pre-pack the router as `U32` -- so `families/qwen4/moe.rs`'s INT8-only router GEMV, correct by design, was refusing a checkpoint the repack path had never learned to transcode. Unblocks decode on the real install. See section 2.
- **`qwen4_exp` QSA Sparse Attention Wired End to End (2026-09-05)**: `families/qwen4/attn.rs` runs the indexer and, above the install's own `index_budget`, block-selected sparse attention in place of the dense fallback. Verified against 17 new synthetic-fixture tests (mutation-checked) and against the real install: coherent greedy and sampled smoke on a 2,940-token prompt, `qwen4exp_quality_gate` and `qwen4exp_memory_oracle` both green and frozen (perplexity 8.7224, peak 2521 MiB against a 3000 MiB ceiling), and a KL-based force-dense probe showing bitwise identity below budget and sub-0.1-nat divergence with 100% argmax agreement above it. Open: chunked prefill, a GPU top-k for block selection, and the bench-window decision. See `docs/QWEN4_EXP.md` and section 2.
- **Server Prefix KV Reuse and Session Pool (2026-09-01)**: `turbospark-server` gained a real `--prefix-reuse on|off` flag (default on), verified against a real Gemma 4 install's second turn. A swap-based bounded session pool (`--session-slots N`, default 1) fixes the cross-conversation KV-stomping hazard the flag exposes when several conversations interleave on one runner; two real bugs (a destructive shallow-match rewind, an overly strict eviction discriminator) surfaced only under the real-install interleaved-conversation gate, not the synthetic fixture. See `crates/runtime/CLAUDE.md` Gotcha 32 and `crates/server/CLAUDE.md` Gotchas 31-32, and section 4.
- **`qwen4_exp` Chunked Prefill, the Seventh Flow (2026-09-05)**: `families/qwen4/prefill.rs`, step 1 again and no new kernel. Both blockers this roadmap had predicted turned out not to exist -- QSA needed nothing (its `encode_full_attention_block` already takes a `&mut PassEncoder` and owns its own above-budget mid-layer commit, and the shared `qsa_positions` buffer is protected by the driver preserving gemma4's per-layer commit-and-wait ordering), and no per-layer position list was required. The real hazard was PLE's `ngram_emb`: a HOST `write_buffer_bytes` that does not respect command-buffer commit order, so a single-row buffer silently fed every token but the last of a micro-batch the wrong n-gram embedding. Caught at chunk span 2 by the boundary test. Both batching seams are now refused by name, the pair every other chunked driver already carried. Verified byte-identical to sequential on 7 synthetic cases and on the real REAP-288 install; **throughput still unmeasured**, which is the one thing the driver exists for.
- **One `TurnSplitter` for Turn Splitting (2026-09-05)**: three drifting copies of the turn-splitting wiring collapsed into `runtime::turn_stream`, with `StructuredAssistantDecoder` now constructed in exactly one place workspace-wide. Two additive C ABI event kinds (`TS_EVENT_TOOL`, `TS_EVENT_FINISH`) and a `toolCalls` result field; `.toolCall` and `.stopped` on the Swift side. `docs/STREAMING.md` is the home and carries the two measured negatives (no engine-side iterator or async stream, no backpressure in the decode path).
- **Swift Shell, Hook Contract and Settings Stores (2026-09-05)**: real background shell execution with per-chat scoped ids, shell execution extracted out of the tool registry with cwd persistence and output shaping, five documented divergences from the Claude Code hook contract, the server API key moved to the login Keychain, and appearance settings moved off `UserDefaults`. ~42 new tests. `docs/SWIFT_TOOLS.md` is the home.
- **Chunked Prefill Becomes Measurable (2026-09-05)**: `turbospark-bench --prefill-chunk off|auto|N` (default OFF) plus a `seq|chunked` arm pair in `scripts/power.sh`. Before this the bench reached only `run_raw_completion` and `run_raw_completion_speculative`, so every throughput and power row ever taken through either tool measured the sequential prefill path regardless of the env seams -- which is why the Prefill Energy Capture row read BLOCKED. Only `TURBOSPARK_PREFILL_CHUNK` needed wiring; the other two seams are read inside the runtime's chunk drivers and needed a header echo. Default-OFF verified by measurement, not argument: pre- and post-change release binaries agree on the stop reason, prompt-token and new-token counts on the real Gemma 4 install, with two pre-change runs agreeing with each other to make the comparison mean something.
- **Worktree Consolidation (2026-09-02)**: merged four development worktrees back into `main` -- the two feature branches above, plus the uncommitted `gpt-oss` phase-2 `top_k` specialization and `crates/ffi` prefix-reuse work that had been sitting unstaged directly on `main`. Two worktrees (`qwen3-8-mtp-support`, `roadmap-next-items`) carried no unique commits past what `origin/main` already had and were removed. Full workspace build/fmt/clippy green post-merge.
