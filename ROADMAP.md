# Roadmap

The forward-looking roadmap and prioritized task tracker for this engine, last reconciled for the P3/P4 pass (steering, sub-byte batched GEMMs, server pool, Linux slice, qwen38 KL, Mistral gate) on 2026-09-15. All core port phases (Q, P1, G, S, P2, M1-M5) are complete and green. This document functions as an active TODO list for forward engineering, measurements, and architectural bring-ups.

All completed work, historical milestones, and landed features have been removed to focus strictly on remaining tasks.

---

## Current Status

- **Swift conversation layout (2026-09-19)**: Active chats now share a bounded reading column with the composer, compact plans and tool activity, and expose tasks at the upper right. Model Settings folds when a conversation begins and reopens on request. Layout contracts and visual-review commands live in [swift/docs/SWIFT_CONVERSATION_LAYOUT.md](swift/docs/SWIFT_CONVERSATION_LAYOUT.md).
- **Test Suite**: Counts and gating conventions live in [docs/TESTING.md](docs/TESTING.md). A fresh `cargo test --workspace` passed on 2026-09-17 with no failures; this includes the two `turbospark-bench --test vision_sidecar_opener` cases, which now pass against the current 27-block synthetic tower. `cargo fmt --check`, full workspace Clippy, the Swift package suite (78 tests, 17 expected real-model skips), the focused app image suite (9/9), the debug app build, and the signed release app bundle are green. The broader app XCTest suite still has 34 failures in concurrent non-image work, including system-prompt/date injection, tool catalog parity, font propagation, folder import, and agent advertisement; these are not IG4 image failures.
- **Architectures**: 15 declared `ModelFamily` variants. The fifteenth,
  `qwen3_vl`, landed 2026-09-18: the Qwen3-VL trunk runs the shared Llama
  flow, the pinned `mlx-community/Qwen3-VL-4B-Instruct-4bit` install passes
  real greedy and sampled CLI smokes with frozen quality (17.3463) and
  memory (793 MiB at 4096) gates, and the catalog row `qwen3vl-4b` is
  `verified`. Text-only intake; the deepstack injection is the open vision
  half (`DEVIATIONS.md`'s `qwen3_vl` section). Dense `qwen3` GGUF execution is verified on 0.6B Q8_0. Dense Qwen2/Qwen2.5 now has a real MLX/HF 4-bit install and greedy plus sampled CLI smokes. MiniMax-M2 GGUF execution is implemented, but repetitive low-temperature smokes block release and catalog promotion. `deepseekV4Flash` remains scaffolded. The full support matrix lives in [docs/MODEL_FAMILY.md](docs/MODEL_FAMILY.md).
- **Vision**: Dense Qwen GDN (`qwen35`, upstream `qwen3_5`) has real-gated still-image support through the CLI, server, FFI, and Swift-facing APIs, including the verified standalone vision sidecar. Qwen GDN MoE (`qwen35moe`) shares the classifier, writer, and sequential runtime path but remains structurally supported and unverified. The other 12 registered families are text-only for images, including `Qwen3Vl` (landed 2026-09-18 text-first: its tower is excluded at repack and the deepstack injection seam is the family's open vision work; [docs/QWEN3VL_PHASE0.md](docs/QWEN3VL_PHASE0.md)).

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
- **Files to Touch / Run**:
  - `scripts/power.sh` (execute benchmark)
  - `docs/POWER_BASELINE.md` (record resulting rows)
- **Process attribution (2026-09-10, 21:52 UTC capture)**: The new process timeline caught `mediaanalysisd` consuming 69.55 CPU-seconds during about 46.9 seconds of pair 1 chunked prefill, alongside updater activity. All phases were Nominal, but chunked energy spread was still 29.2% (sequential 3.3%). Later pairs used 21.8%/16.8% less chunked prefill energy; keep them as observations, not a frozen saving. Evidence is in `docs/POWER_BASELINE.md` and `docs/verification/prefill-energy-cpu-trace-2026-09-10-process-summary.json`. Next: wait for the observed background work to subside and retain per-window process checks. The morning anomalies remain unattributed.
- **Capture attempt (2026-09-17)**: the current preflight passed on AC with no competing inference process, but `scripts/power.sh` stopped at the interactive `sudo powermetrics` password prompt before any benchmark arm ran. No energy row was recorded; rerun requires local sudo access.

#### 3. `qwen4_exp` Chunked Prefill Throughput [closed: measured negative 2026-09-18]
- **Verification completed (2026-09-09)**: Repeated short references and three interleaved sequential/chunked pairs at the frozen 2048 window, with phase totals recovered using the actual divisor. Generated bytes agree across arms. Pread accounts for about half of the long-minus-short forward-time increment.
- **Why Open**: Mean chunked prefill savings of 0.513 s are smaller than the sequential reference's 0.61 s spread. A reproducible throughput gain remains unestablished; no frozen baseline or context-window change is justified.
- **CLOSED (2026-09-18)**: a third attempt -- six interleaved pairs, two discarded warmups, repeated short references, on battery with both arms under the same power state (evidence: `docs/verification/qwen4-2026-09-18.json`) -- read sequential 37.77 s mean (spread 5.29 s) against chunked 37.35 s (spread 3.47 s), signs flipping 3/3. Pread measured 53.3%/53.9% of forward and unchanged between arms; the long-minus-short increment is 51.8% pread. Three attempts now agree: this checkpoint's prefill is pread-bound and command-buffer batching has nothing to win. Do not re-open without a change that touches the expert `pread` term itself.
- **Evidence**: `docs/QWEN4_EXP.md`, `docs/verification/p0-2026-09-09.json`, `docs/verification/qwen4-2026-09-18.json`.

#### 4. TurboQuant Real-Model Verification Findings
- **Verification completed (2026-09-09)**: Real probe readings now exist for gemma4, qwen38-27b dense, gpt-oss, qwen3moe, qwen4_exp, Spark and museGlimmer. Spark's five-case KV suite passes with individually targeted mutation checks. Literal Gemma output bytes match between `673341e^` and `673341e` in greedy and sampled modes with KV quantization off. All requested CLI smokes and applicable frozen family gates have run; task-created installs were removed after evidence capture.
- **Why Open**: gpt-oss KV4 greedy exhausts 3072 tokens in reasoning while its matched FP16 control completes (KV4 sampled completes). museGlimmer's FP16 sampled golden mismatches identically in two current runs and the isolated feature-era build, despite matching perplexity and greedy output. The remaining Qwen4 finding re-CONFIRMED 2026-09-18 on a freshly re-streamed install at identical token counts (542/676/499): the KV4 sampled answer denies the prompt's wetlands premise while the matched FP16 control keeps it, with unsupported claims in the latter too. These remain findings, not clean quality passes or proven general kernel regressions.
- **Interpretation**: gpt-oss's unframed probe perplexity remains unusable as a quality signal because of Harmony framing. Spark and museGlimmer also require their answer-slot prefixes for comparison with their quality gates. No quantization policy or golden has changed.
- **Evidence**: `docs/TRUBOQUANT.md`, `docs/verification/p0-2026-09-09.json`, `docs/verification/qwen4-2026-09-18.json` (Qwen4 KV4 re-run).

#### 6. `kv_quant_probe` Footprint Attribution (spread control complete)
- **Verification completed (2026-09-09)**: Probe order is `off, off, off, 3, 3.5, 4, 2, off`; deltas keep the first baseline and the footer reports repeated-off minimum, maximum and spread. Revised real-install probes passed on Spark, museGlimmer and Qwen4.
- **Why Open**: Qwen4's quantized deltas re-read +81.7 to +86.9 MiB on the 2026-09-18 re-run, but inside a repeated-off range that itself doubled to 149.0 MiB (off rows trending 2516.7 -> 2597.3 -> 2665.7 across the probe's opens). The candidate cause the earlier entry said was missing is now recorded: resident-page warming of the streamed 68 GiB install, whose mapping phys_footprint counts. Deltas remain within the reference spread; unresolved by the probe's own rule. An arm order interleaving quantized widths between the off rows would separate "quantized costs footprint" from "the run warmed up" and has not been run. Per-width perplexities reproduce 2026-09-09's values to the fourth decimal on the fresh artifact. Spark and museGlimmer reductions exceed their observed spreads; this warrants attribution work rather than an automatic causal conclusion or precise width ranking.
- **Evidence**: `crates/bench/tests/kv_quant_probe.rs`, `docs/TRUBOQUANT.md`, `docs/verification/qwen4-2026-09-18.json`.

---

### Priority 1: Near-Term Core Engine & Infrastructure

High-leverage engine improvements, memory policy unifications, and front-end wirings.

#### 1. Vision Memory Sidecar Parity & Memory Oracle Arms [complete]
- **Status**: Sidecar-aware arms landed and ran on real hardware on 2026-09-10. The sidecar tower matches the combined install through the four-stage mlx-vlm parity gate, and the sidecar memory oracle remains within the 950 MiB ceiling with flat repeated-page growth.
- **Evidence**: `crates/runtime/tests/vision_tower_parity.rs`, `crates/bench/tests/vision_memory_oracle.rs`, and [docs/VISION.md](docs/VISION.md#verified-on-real-hardware).

#### 2. MTP / DFlash2 Verify Pass for Multimodal Prompts [complete safety boundary]
- **Status**: The verify pass remains vision-blind, but the unsafe combination is now refused. Explicit speculative blocks hard-fail with a vision reason, `auto` continues text-only, and the batched verify path has a defense-in-depth refusal. A fully vision-aware verify pass remains a conditional future enhancement because no tested artifact combines a vision tower with an MTP or DFlash2 head.
- **Evidence**: `crates/runtime/src/speculative.rs`, `crates/runtime/src/families/qwen/mtp.rs`, `crates/runtime/src/families/qwen/dflash.rs`, and [docs/VISION.md](docs/VISION.md#current-limitations-and-deferred-work).

#### 3. Mapped Residency Eviction Benchmark & Policy Unification [landed 2026-09-17]
- **Objective**: Measure paging overhead and fault costs when OS reclaims clean mapped pages under memory pressure. Unify slot cache policy with mapped expert residency: pick residency mode first (`mapped` vs `streamed`), then slot count only if `streamed` is active. Expose `--expert-residency auto|streamed|mapped`.
- **Status**: The automatic policy is landed. It preserves streamed residency when the minimum cache fits, selects mapped residency when it does not, refuses unsupported families, and carries the resolved mode through CLI, server, C ABI admission sizing, and the actual runner open. The eviction result remains a narrow measurement, not a claim that mapped pages are safe under all pressure regimes.
- **Files to Touch / Create**:
  - `crates/bench/tests/mapped_residency_eviction.rs` [NEW]
  - `crates/invocation/src/options.rs`
  - `crates/cli/src/args.rs`
  - `crates/server/src/args.rs`
  - `crates/runtime/src/runner.rs`
  - `crates/model-io/src/expert_cache_policy.rs`
  - `docs/EXPERT_RESIDENCY.md`

#### 4. Exact Rejection Sampling for Speculation (T > 0) [landed 2026-09-11]
- **Objective**: Implement Leviathan/Chen algorithm on shaped distributions for non-greedy sampling during speculative verification.
- **Landed**: MTP step drafting now samples proposals from the drafter's shaped `q`, accepts against the target's shaped `p`, and draws rejection corrections from normalized `max(p - q, 0)`. The synthetic proof covers perfect, useless, and alternating drafters over 1,500 seeds each, with total-variation and analytic-mass bounds; both algorithm halves were mutation-checked. The focused proof passes in the current tree.
- **Boundary**: DFlash2 remains refused for sampled speculation because its structured selector supplies no proposal distribution `q(x)`. This is an intentional capability boundary, not an unfinished MTP rejection path.
- **Files to Touch**:
  - `crates/selection/src/` (rejection sampling algorithm)
  - `crates/runtime/src/speculative.rs`
  - `crates/runtime/src/speculation_policy.rs`
  - `docs/SPECULATIVE_DECODING.md`

#### 5. Server Request Queue & Fairness (Option 1) [complete]
- **Status**: LANDED in `crates/server/src/queue.rs` (this entry's file list was stale: the other two paths do not exist). Admission is a one-permit tokio semaphore with arrival-order fairness and a cancel check after the grant; the wait is async, so a queued request holds no blocking thread. `RealChatModel` constructs one gate per RUNNER, the acquisition points are a closed non-nested set (`queue.rs`'s header enumerates them), and `tests/generation_queue.rs` pins the ordering. This entry's routing half is what made P3.6's pool a registry-level change rather than a queue one.

---

### Priority 2: Architecture Bring-ups & Kernel Scaling

Adding missing high-demand model families, specialized Metal kernels, and architectural extensions.

#### 1. `qwen4_exp` GPU Top-K for Block Selection [profiled 2026-09-18: build branch FIRES; kernel owed]
- **Objective**: Above `index_budget`, QSA block scoring commits and waits on the host for `compute::select_blocks` once per QSA layer per token (12 host commits per token). Profile whether this is a bottleneck, and implement a GPU top-k kernel to eliminate host-device synchronization.
- **Prior to Beat**: Do Not Revisit 7 is the measured neighbour, not a refutation of this item. That one is the MoE router's expert top-k, where host overhead came in at 0.13 ms/token and a GPU kernel only relocated the synchronization. This is QSA block selection at 12 host commits per token, a different and larger sync cost, so the entry stands. But the profiling step has to clear that bar explicitly rather than assume nothing is known.
- **PROFILED (2026-09-18, decision rule executed as written)**: interleaved `TURBOSPARK_QSA_FORCE_DENSE=1` vs unset through the production binary on a 2,501-token prompt (450 past the budget), `--max-context 4096`, warmup discarded, three pairs on AC (evidence: `docs/verification/qwen4-2026-09-18.json`). Force-dense is FASTER in 3/3 pairs: sparse decode 4.72/4.86/4.80 s vs dense 4.43/4.46/4.39 s per 48 tokens, i.e. the sparse apparatus costs 6.0/8.3/8.5 ms per decode token (~7% of decode) at ~2.5K context. Expert requests are byte-identical between arms (1,223,040 requests, 39.3% hits), so the delta is attention-path. Phase attribution: the visible share is in `cb1` wait, +4.90/+1.29/+2.52 ms per above-budget pass (~0.24-0.4 ms per QSA commit across 12 layers), with the rest in pread wall-overlap. Against the rule's own bar this is unambiguous -- the commit clearly exceeds the router's 0.13 ms/token, and the sparse path is a net LOSS versus force-dense at this operating point -- so the rule's BUILD branch fires.
- **The build, scoped (owes a re-pull)**: remove the WHOLE round trip, not just the sort -- a selection kernel implementing `compute::select_blocks`'s exact semantics (top-`min(topk, complete)` by score with lower-index tie-break, ragged tail always selected, positions ascending; `compute::select_blocks` stays as the CPU oracle), kernel-written position lists into PER-QSA-LAYER buffers (the shared `qsa_positions` is only safe because of the per-layer commit the kernel deletes; see the `attn.rs` safety note), and a count-from-buffer dispatch for `attention_decode_indexed`/`_tq` (the host no longer knows the list length; fixed max chunks early-exit on the device count). The score NaN guard moves into the kernel or a debug readback. Below budget nothing changes; the frozen digests pin that and reproduced on this install the same day. The install was removed after evidence capture per the session goal, so the build starts with a re-pull (see the artifact table).
- **Files to Touch / Create**:
  - `crates/gpu/src/shaders/qsa_topk.metal` [NEW]
  - `crates/gpu/src/` (pipeline dispatch; `attention_indexed` count-from-buffer)
  - `crates/runtime/src/families/qwen4/attn.rs`
  - `docs/QWEN4_EXP.md`

#### 2. Dense `qwen2` / `qwen2.5` Validation and Format Expansion
- **Objective**: Finish real-artifact evidence and broaden format coverage around the landed dense Qwen2/Qwen2.5 path, which forms a large part of user-downloaded Qwen repositories.
- **Landed (2026-09-10)**: Dense `qwen3` GGUF registration and execution use the shared Llama flow. Per-head Q/K normalization is fixed and verified on Qwen3-0.6B Q8_0 with greedy/sample smokes, memory, and a mutation-checked frozen quality gate. See [the regression record](docs/MINIMAX_M2_PHASE0.md#shared-flow-regression-checks).
- **Landed (2026-09-13)**: Dense Qwen2/Qwen2.5 is registered as `ModelFamily::Qwen2Dense`. GGUF `qwen2` and HF `qwen2` intake parse the dense shape, MLX/HF source names are normalized, the shared Llama flow applies Q/K/V biases before RoPE, and synthetic GGUF plus MLX-shaped installs pass manifest, finite-logit, bias-effect, and chunked-prefill gates. Qwen2-MoE, Qwen2-VL, split GGUF, and native unquantized BF16/FP16 safetensors conversion remain out of scope.
- **Real artifact status (2026-09-17)**: The pinned `mlx-community/Qwen2.5-7B-Instruct-4bit` checkpoint streamed into a verified `.gturbo` install and passed greedy and sampled CLI smokes with the official tokenizer sidecars. The new Qwen2 quality gate freezes perplexity `12.4206` with stable greedy and sampled digests; the constrained 8-slot arm reproduces the 16-slot digest at `1.00x` throughput. The memory oracle records a `622 MiB` peak at 8,192 context with `0.03 MiB` replay growth and complete protocol answers. The MLX cross-engine row passes with mean forward KL `0.0002923`, p99 `0.0022752`, and `99.6546%` top-1 agreement. The catalog row is promoted to `verified`; evidence is in [qwen25-kld-2026-09-17.json](docs/verification/qwen25-kld-2026-09-17.json).
- **Why Still Narrow**: The MLX path is now real-gated, but the pinned official single-file Q3_K_M GGUF parses and remains non-executable because this port has no Q3_K resident kernel. A Q4_K or Q8_0 Qwen2 GGUF must still be streamed and run as a separate real artifact. Larger Qwen2 checkpoints and derivatives need their own validation.
- **Files to Touch / Create**:
  - `crates/bench/tests/` (Qwen2 memory and quality gates)
  - `crates/catalog/src/models.json` (additional validated checkpoints)
  - `crates/gpu/src/shaders/` (Q3_K resident kernels, if that scope is chosen)
  - `docs/NEW_MODEL.md`, `docs/MODEL_FAMILY.md`, `docs/TESTING.md`

#### 3. `deepseek2` Architecture Support (High-Leverage Multi-Model Unlock) -- LANDED, numerics closed, RELEASED 2026-09-17
- **Landed**: the full cross-layer bring-up -- `ModelFamily::Deepseek2`
  (mask 5, MLA), baseline + manifest round-trip, GGUF name table, config
  parser, registry promotion (witness: `mradermacher/DeepSeek-V2-Lite-Chat-GGUF`
  Q8_0, sidecars `deepseek-ai/DeepSeek-V2-Lite-Chat`), the absorbed-MLA
  kernels (`mla.metal` + `mla.rs`, parity-tested with mutation checks), the
  `families/deepseek2/` decode flow over the compressed 576-half cache,
  dense-lead blob positions in the walk, the V2 tokenizer dialect, the
  catalog row `dsv2lite-16b`, and the `logit_dump`/`llamacpp_logits`
  cross-engine instruments. The real install runs deterministic generation.
- **Numerics CLOSED (2026-09-17)**: the pos>=1 divergence from llama.cpp on
  the identical Q8_0 bytes was a half-split rope pairing in the MLA kernels
  where ggml pairs consecutive elements. Cache rows now match llama.cpp at
  corr 0.99998+ at every position, logits at 0.984-0.9999 (argmax identical
  on 14 of 15 prompt positions), greedy + sampled generation coherent with
  EndOfTurn reached, catalog row promoted to `runs`
  ([DEEPSEEK2_PHASE0.md](docs/DEEPSEEK2_PHASE0.md)'s closed-numerics section
  carries the record and the two false leads).
- **RELEASED (2026-09-17): the owed rows are frozen.** Quality gate at
  perplexity `14.3688` with stable greedy and sampled digests (two fresh
  processes byte-identical; no assistant prefix; the 8-slot constrained
  digest equals the 16-slot one); memory oracle at `4,103 MiB` peak
  (ceiling `4,300`) with all three protocol cases stopping endOfTurn at
  the family's own `8,192/1,024` window and a `4.0` tok/s decode floor
  (readings `17.533` / `11.285` / `5.722`); catalog `measured` block and
  `gates` filled, with the offline agreement test tying them to the
  oracle's row. The descope list (split `wk_b`/`wv_b` files,
  `q_lora_rank > 0`, batched absorbed prefill, pure-576 KV, and the
  V3/GLM/Kimi witnesses) is detailed once in `DEVIATIONS.md`'s `deepseek2`
  section, its one home. Numbers and interpretation:
  [DEEPSEEK2_PHASE0.md](docs/DEEPSEEK2_PHASE0.md)'s frozen-release-gates
  section.
- **Files touched**: the four below, plus `gguf_names/deepseek2.rs`,
  `gguf_config`, `arch_registry`, `kv_layer_strides` (mask 5), the
  `MlaConfig` manifest block, the dense-lead positional walk, the V2
  tokenizer dialect, and the Swift MoE set.

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
- **Re-cost, measured, 2026-09-15: the economics close negative.** The batched affine routed pair this item was waiting on is built and its kernel-level `c(M)` is measured (c(4) = 0.68, c(8) = 0.66, `crates/gpu/tests/moe_prefill_batch_bench.rs`) -- worse than the 0.5 grant `docs/SPECULATIVE_DECODING.md`'s composite had assumed. Substituting the measured pair, the MoE verify ceiling drops from the recorded 1.14x to **1.08x at block 4** and loses at every larger block; the doc's own verdict already rejects 1.14x. And the motivating artifact (`ornith35b`) is a GGUF install whose routed pair has no batched form at all (BATCHED_PREFILL step 5 is unbuilt), where every block reads sub-break-even (0.99x at best). The head's own forward and the rollback term only subtract. Full table and reopening conditions in [docs/SPECULATIVE_DECODING.md](docs/SPECULATIVE_DECODING.md)'s "What would change the answer". Ingestion work (GGUF nextn block, MoE head tensors, the three `num_experts != 0` policy refusals) is therefore not built; the refusals stay, correctly naming a checkpoint gap rather than a kernel gap.
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
- **Landed (2026-09-15): Mistral `[TOOL_CALLS]`.** The real `mistral7b-dense.gturbo` install's table resolves `[TOOL_CALLS]` as a special token (id 5), so the native arm keys on the id like Gemma's, buffers every token after it, and emits the parsed call from the decoder's `finish` -- a `[TOOL_CALLS]` span has NO closing token (the JSON-array body runs to end of turn, and end of turn is `</s>`, a stop token), which is structurally Harmony's terminator-blind case. `resolve_mistral` resolves the marker OPTIONALLY, so the earliest tables (Mixtral 8x7B-Instruct's three tokens) keep the passthrough. `tool_call_support` flips Mistral to `Native`; `MistralToolCallParser` accepts both the array and the single-object form. Covered by four decoder-level tests, the dialect-support table test (whose harness now calls `finish`, which the previous one never did), and a mutation check on the finish dispatch.
- **Deferred with findings (2026-09-15): MiniMax.** The published M2 `tokenizer_config.json` shows the `<minimax:tool_call>` wrapper tokens are NON-special added tokens (ids 200052/200053, `special: false`), so a native arm would be a text-marker arm like DeepSeek's rather than the id-bracket arm the entry's wording assumed; and this port's MiniMax dialect probe strings (`]~!b[`/`]~b]`/`[e~[`) do not match the published M2 table's bos (`]!p~[`), so the dialect itself may need re-derivation against the M2 table first. The MiniMax-M2 install is no longer on disk (store drift from the artifact table), so no real-model gate is reachable this session; the rescue tier already parses the invoke shape.
- **Deferred with rationale (2026-09-15, updated 2026-09-17): GLM and Kimi K2.** Neither markup has a registered dialect here, and both model families are deepseek2 (MLA), which was recognition-only when this was written -- no loadable checkpoint to smoke a native decoder against, the bar `docs/TOOL_CALLING.md` itself sets. The family side of that precondition is now MET (deepseek2 landed and runs, item 3 above); what remains is deriving each dialect's probe/resolver from a real checkpoint's tokenizer tables and installing a witness for it.
- **Files to Touch**:
  - `crates/tokenizer/src/structured_decoder/`
  - `crates/tokenizer/src/chat_template.rs`
  - `docs/TOOL_CALLING.md`

#### 9. Combined VLM Ingestion from HF Hub in `turbospark-model pull`
- **Objective**: Fix the normal production trunk intake so `turbospark-model pull` preserves a checkpoint's vision config and `vision_tower.*` tensors instead of hardcoding `vision: VisionConfig::NONE` in `parse_qwen_gdn_dense_config`.
- **What Landed**: `turbospark-model pull-vision` is the supported production path for a standalone tower sidecar. It parses and verifies the tower, writes the sidecar, and is covered by real parity and memory evidence.
- **Landed (2026-09-15)**: `stream_mlx` now detects a combined tower from the checkpoint's BYTES. After the shard headers are fetched, a `QwenGdnDense`/`QwenGdnMoe` pull whose headers carry `vision_tower.*` tensors enables `arch.vision` through the existing `parse_vision_config` plus hidden-size cross-check (`catalog::stream::enable_bytes_detected_vision`), and everything downstream -- classification, both writers, `packed_vision/`, the manifest, the runtime's `open_vision_tower` -- already worked. Catalog-row intent (`include_vision`) keeps its precedence and the config parser's text-only default stays: the Ornith declare-but-not-ship rule means the config alone can never decide. Tower bytes with an unparseable or hidden-size-mismatched config refuse the walk BY NAME rather than silently dropping a third of the tensors, which is the behavior this replaces.
- **Evidence**: unit tests over the detection gate in `crates/catalog/src/stream.rs` (bytes-enable, config-declares-without-bytes stays text-only, refusal wordings, intent precedence, family gate mirroring `classify_for_family`), mutation-checked on both the enable and the bytes gate; the on-disk `vision-probe-qwen38` repo's real shard header (333 `vision_tower.` tensors) and `config.json` cross-checked against the existing combined install's manifest fields.
- **Why Open (residual)**: a full end-to-end re-pull of a real vision repo through the new path has not been run (the on-disk probe repo is header-only); the HF-native `model.visual.` spelling remains sidecar-only in the combined walk, per its classify comment.
- **Files to Touch**:
  - `crates/catalog/src/stream.rs`

#### 9a. Qwen GDN MoE Vision Validation
- **Objective**: Validate a real `qwen35moe` vision artifact through the combined and sidecar paths, including CLI, server, FFI, image-output, memory, and parity gates.
- **Why Open**: The classifier, writer, and sequential `families/qwen/` runtime accept the shared tower shape, but no independent MoE real-model gate exists. The dense Qwen sidecar is not interchangeable because pairing uses the exact family identifier, and the MoE chunked-prefill path remains refused.
- **Files to Touch / Create**:
  - `crates/runtime/tests/` (MoE vision parity and real-backend arms)
  - `crates/catalog/src/models.json` (only after a pinned artifact is verified)
  - `docs/VISION.md`, `docs/MODEL_FAMILY.md`

#### 9b. Qwen3-VL Trunk and Deepstack Bring-up [TEXT LANDED 2026-09-18; vision open]

- **Landed (text)**: `ModelFamily::Qwen3Vl` (wire string `qwen3_vl`) is
  registered across model-io, repack, catalog, runtime, bench, FFI and
  Swift. The trunk is the shared Llama flow's third family (per-head q/k
  norms + dense FFN, per `RealLlamaState`); the Phase 0 open items are
  closed (full rotary; deepstack = raw add after trunk layers 0/1/2; naming
  re-verified on the pinned revision, which also exposed a STALE SHARD INDEX
  now defended against in `crates/catalog/src/stream.rs`). Real artifact:
  `mlx-community/Qwen3-VL-4B-Instruct-4bit` @ 2fd8dacb streamed to
  `~/.turbospark/models/text/qwen3vl-4b.gturbo`; greedy + sampled smokes
  coherent with EndOfTurn; quality gate frozen at perplexity 17.3463 with
  two-process digest agreement; memory oracle frozen at 793 MiB peak (4096
  context, +0.02 MiB replay growth; TWO of three protocol cases -- sampled
  medium-review does not terminate on this checkpoint, a documented
  checkpoint property, tinyllama precedent). Catalog row `qwen3vl-4b` is
  `verified` with the rot-guard byte figure. Synthetic gates: 8 parser
  tests, 6 runtime tests (mutation-checked). GGUF intake refused by design
  (`qwen4_exp`'s reasoning). Record: `DEVIATIONS.md`'s `qwen3_vl` section.
- **Why Open (vision half)**: the checkpoint ships the SigLIP-class tower
  this port already runs for `qwen3_5` PLUS three deepstack mergers whose
  outputs raw-add into trunk layers 0/1/2's residuals at image positions.
  That injection seam (mRoPE triples walk + per-layer adds), the tower
  sidecar writer carrying the 18 merger tensors, and the four vision gates
  have not run; text-only intake excludes all of it at repack.
- **Owed (measurements)**: a cross-engine KL row (the `kld_mlx_vlm.py`
  instrument is the reference; no CHECKPOINTS entry yet) and a
  `scripts/power.sh` row, both unscheduled like every sibling's second-row
  measurement. The frozen quality + oracle pair is what `verified` stands
  on.
- **Files to Touch / Create**:
  - `crates/runtime/src/vision/` (deepstack injection)
  - `crates/repack/src/gemma4_checkpoint/vision.rs` (18 resident tensors)
  - `crates/repack/src/qwen36_config.rs` (`parse_vision_config` deepstack fields)
  - `crates/catalog/src/stream.rs` (`stream_vision_sidecar` family arm)
  - `crates/runtime/tests/` (tower + deepstack parity arms)


#### 10. `spark2_5` Follow-ups (the bring-up itself has LANDED)
- **Status**: the family landed in `2a1f1fc` (2026-09-09) and is not open work. `crates/runtime/src/families/spark/`, `crates/model-io/src/arch_baselines/spark.rs`, the GGUF names and config arms, the two new Metal kernels, the dialect and a catalog row all exist, and BOTH real-model gates are frozen from real runs on 2026-09-08: `spark_memory_oracle` at 575 MiB measured (ceiling 700, tok/s floor 32.0) and `spark_quality_gate` at perplexity 12.6162. The install sits at `~/.turbospark/models/spark25.gturbo`. Bring-up facts are in `docs/SPARK_PHASE0.md`.
- **What is Still Open**: the four deliberate descopes, named here and detailed once in `DEVIATIONS.md`'s "`spark2_5` (the ninth family), landed and deliberately not done" section, which stays their only home: HF safetensors / MLX intake (GGUF-only today, so `turbospark-model probe XHToken/Spark-X2.5-4B` still refuses at the registry), the tool-call DSL parser (the markup flows as ordinary content), the 1.7B sibling (which needs a per-checkpoint baseline scheme rather than a config change), and a `scripts/kld_llamacpp.py` `CHECKPOINTS` row, now reachable because upstream llama.cpp PR 27868 runs this architecture.
- **Files to Touch / Create**:
  - `crates/repack/src/arch_registry.rs` (`SUPPORTED_HF` rows), `crates/catalog/src/` (`evaluate_config`, `stream_mlx`)
  - `crates/tokenizer/src/structured_decoder/` (the fifth tool-call DSL)
  - `scripts/kld_llamacpp.py` (one `CHECKPOINTS` row)

#### 11. Native Z-Image-Turbo Image Generation (CLI, then App)

- **Objective**: Add local, quantized text-to-image generation through native Rust/Metal, first in the unified CLI and then through the existing C ABI/Swift package in a canonical top-level `Images` destination with `Create` and `Gallery` views.
- **Status**: IG0 through IG4 are closed on the pinned 1024-by-1024 install. The remaining IG5 work is measured optimization only: VAE tiling, prefetch, allocator reuse, and any approximation proposal require fresh quality, memory, and end-to-end evidence.
- **Design and Gates**: [Native image generation](docs/IMAGE_GENERATION.md) owns the architecture, interfaces, evidence requirements, and release boundaries. [ZIMAGE_TURBO.md](docs/ZIMAGE_TURBO.md) records the reusable image-model bring-up process and benchmark summary. The production gates and app integration are closed; the documented release boundary remains one image per prompt at 1024-by-1024.
- **Sequence**:
  - [x] **IG0**: Close resource evidence with quiet-AC cold/warm measurements, retained memory, swap, physical reads, inclusive activation/scratch accounting, a supported memory envelope, and the final image manifest contract.
    - **Started (2026-09-10)**: [Phase 0 evidence](docs/IMAGE_GENERATION_PHASE0.md) pins all inputs and 1,163 tensors, records tokenizer/scheduler probes, bounded Diffusers/MFLUX block agreement, and Rust-verified group-64 packing. Nine requested steps produce nine forwards in the pinned Diffusers revision. Eight real captures cover the four-prompt BF16/INT4 suite with identical noise and a documented visual review. The component numerical coverage is now closed by IG1. Busy-AC capture timings are not benchmarks; the subsequent quiet-AC resource closure is recorded below.
    - **Measurement attempt (2026-09-13, historical)**: `quiet-03` produced exact lighting-reference outputs for encode and denoise, and recorded process footprint, MPS live/driver bytes, physical reads, retained allocations, and swap. Only warm text encoding qualified. Cold encode and cold denoise were rejected by the interval quietness gate; denoise warm execution refused after the host `ChatGPT` and `synrepo` processes became active, and VAE coverage did not run. The summary is [z-image-ig0-benchmarks-quiet-03.json](docs/verification/z-image-ig0-benchmarks-quiet-03.json). Later quiet runs supplied the missing qualified stage evidence.
    - **Follow-up attempt (2026-09-13, historical)**: `quiet-04` qualified the cold text-encoder arm, including exact output, three resident repetitions, and measured disk reads. Its warm arm was rejected before execution because the active host `ChatGPT` process held about one core, so denoise and VAE did not run. The summary is [z-image-ig0-benchmarks-quiet-04.json](docs/verification/z-image-ig0-benchmarks-quiet-04.json); later quiet runs closed the remaining evidence.
    - **Quiet run (2026-09-13)**: `quiet-05` qualified both encoder cache arms and the cold denoiser arm. The largest qualified process footprints were 8,969,979,152 bytes for text encoding and 17,512,599,008 bytes for denoising; retained MPS driver allocations were 8,322,236,416 and 14,018,134,016 bytes, respectively, and `empty_cache` reduced both to 737,280 bytes. Swap was unchanged during the qualified arms. The denoiser reuse arm ran but was classified as mixed and failed two quiet timed windows, so it remains excluded from the aggregate. VAE measurement did not start because the reference `target/ig0/runs/lighting/decode.json` fixture was missing, not because of a VAE resource failure. See [z-image-ig0-benchmarks-quiet-05.json](docs/verification/z-image-ig0-benchmarks-quiet-05.json).
    - **VAE resource run (2026-09-13)**: `quiet-06` qualified both cold and warm decode arms with exact reference outputs and unchanged swap. The largest qualified process footprint was 11,025,630,840 bytes; the VAE's resident parameters were 335,278,732 bytes, retained MPS driver allocation was 10,060,791,808 bytes, and `empty_cache` reduced it to 2,150,318,080 bytes. The six resident decode samples ranged from 0.9327 to 0.9381 seconds. See [z-image-ig0-benchmarks-quiet-06.json](docs/verification/z-image-ig0-benchmarks-quiet-06.json). This closed the VAE cold/warm observation set; repeated full-pipeline stability is deferred to IG3.
    - **Denoiser phase follow-up (2026-09-13)**: `quiet-07` produced exact nine-forward outputs in both arms. The cold load, first execution, and first two resident phases qualified; its final resident phase was rejected by one background interval. The reuse arm was physically mixed and its load/first windows were rejected, but all three warm resident phases qualified with zero physical reads. See [z-image-ig0-benchmarks-quiet-07.json](docs/verification/z-image-ig0-benchmarks-quiet-07.json). This supplies the resident warm execution evidence without claiming a fully cached whole-process load.
    - **IG0 closure (2026-09-13)**: The measured reference envelope and image-install manifest contract are frozen in [z-image-ig0-resource-contract.json](docs/verification/z-image-ig0-resource-contract.json). The first supported envelope is 1024-by-1024, batch one, nine scheduler steps, nine transformer forwards, guidance zero, and one heavyweight stage resident at a time. Stage ceilings are inclusive fresh-process footprint observations: text encoder 8,969,979,152 bytes, transformer 17,512,599,008 bytes, and VAE decoder 11,025,630,840 bytes. Exact MPS operator scratch is not observable, so driver-retained bytes are not mislabeled as scratch; the contract makes no minimum whole-machine RAM claim. IG2 must validate the packed Metal runtime and populate the manifest's exact packed tensor/file records against this contract. Repeated full-pipeline stability, streamed ownership, and cancellation remain IG3.
  - [x] **IG1**: Validate native conditioning, transformer blocks, scheduler updates, and VAE against the pinned reference.
    - **Progress (2026-09-12)**: New portable crate `crates/image` (`turbospark-image`) delivers native FlowMatchEuler scheduler step parity (exact sequence, timesteps/sigmas, and 9-step Euler integration vs captured latents) and conditioning path (exact Qwen chat template framing and tokenization across all 7 prompt cases; native FP32 text-encoder CPU forward with ~8.7e-3 to ~8.9e-3 rel-L2 tolerance vs captured BF16 MPS reference). The IG1 mutation report now records 16 native assertions with no isolated survivors (`docs/verification/z-image-ig1-mutations.json`).
    - **Progress (2026-09-12, continued)**: `turbospark-image` now includes
      a streamed FP32 checkpoint DiT reference (2 noise-refiner + 2
      context-refiner + 30 main blocks), checked component contracts, and RGB
      PNG conversion. The reference preserves the pinned `[1,16,128,128]` to
      1024-by-1024 VAE geometry. These are implementation milestones, not
      passed end-to-end gates.
    - **Progress (2026-09-12, checkpoint parity)**: The opt-in full-width
      64-token block now passes the frozen FP32 thresholds against Diffusers
      and MFLUX after matching tree-shaped RMSNorm reduction and affine bias
      ordering. The reduction and bias assertions are mutation-checked.
    - **Progress (2026-09-12, nine-step gate)**: All nine captured-input
      scheduler updates pass the unchanged local `2e-2` ceiling. The complete
      rollout measures accumulated relative L2 from `2.02384288e-3` at update
      1 through `1.95015728e-1` at update 9. Rounding the measured maximum up
      to the next `0.001` freezes the cumulative BF16 envelope at `0.196`.
      The curve is accumulated FP32 CPU versus BF16 MPS state sensitivity, not
      a local timestep failure. `latent_08.npy` is byte-identical to
      `final_latents.npy`, so the capture has nine transitions, not ten. The
      real 1024-by-1024 VAE decode gate passes in 4695.11 seconds and produces
      `[3,1024,1024]`. Its tightened latent-geometry assertion fails in
      isolation and is restored; the optional raw-pixel comparison files were
      absent, so the conditional pixel assertions did not run. The independent
      pinned VAE evidence remains below its frozen `6e-5` and `3e-6` limits.
  - [x] **IG2**: Deliver a complete quantized install and staged CLI pipeline with PNG output, metadata, progress, and cancellation.
    - **Closed-gate follow-up**: The bounded trace and complete packed gates are recorded below. App work was deferred until IG3 and is now closed under IG4.
    - **Implementation progress (2026-09-13)**: `turbospark-image` now has a checked image manifest, a packed component format using the shared affine INT4 group-64 layout, an atomic local `image pack` installer with `verified-install.json` binding, explicit text/transformer/VAE stage ownership, progress and cancellation seams, PNG metadata, atomic output publication, a macOS-only `MetalImageBackend`, image-specific MSL operators, and a `turbospark image generate` path that selects native Metal by default on macOS. The CPU backend is explicit reference-only. This is implementation progress, not IG2 closure.
    - **Artifact boundary (2026-09-13, adapter extended 2026-09-18)**: The runtime does not load arbitrary Hugging Face MLX image exports directly. The image packer accepts the pinned Diffusers-style source shape and MLX affine U32 weight planes with F16 or BF16 `.scales` and `.biases` companions, including the non-`.weight` token tensors present in the published Z-Image export. It admits 2, 3, 4, 5, 6, and 8-bit group-64 affine weights through the same `.image.gturbo` install path; protected tensors and image-sensitive operations remain at higher precision. The pinned [`andrevp/Z-Image-Turbo-MLX-2bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-2bit), [`andrevp/Z-Image-Turbo-MLX-4bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-4bit), [`andrevp/Z-Image-Turbo-MLX-8bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-8bit), and [`andrevp/Z-Image-Turbo-MLX`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX) exports now have image aliases, source-shape checks, and selected real-payload pack/decode checks. The 2-bit, 4-bit, and 8-bit full installer gates passed on 2026-09-18; the FP16 full install remains open. Image quality, resource, and Swift parity evidence remains open for every published variant except the already-pinned INT4 profile. Full precision remains an unquantized source path, not an affine-width claim. Do not represent any of these as ordinary text `Mlx` catalog rows.
    - **Memory strategy (2026-09-13)**: The design keeps one heavyweight stage resident at a time, maps the packed payload without expanding every matrix into a separate host copy, dequantizes legacy INT4 and MLX affine linear weights during Metal operations, and releases text and transformer state before the next stage. The native wrappers are still correctness-first and create many operation-level command buffers and temporary buffers, so no native memory claim is frozen yet. The next memory pass is pooled scratch and activation reuse, fewer command-buffer boundaries, safe BF16/FP16 storage for non-INT4 tensors, explicit component release, and cold/warm measurement against the IG0 ceilings.
    - **Progress (2026-09-14)**: The tiled linear Metal launch now matches the shader's 32-column threadgroup stride, and the focused packed linear parity test passes on real Metal. Image operations now defer command-buffer waits until CPU read seams while retaining the producing pass for output lifetime; the caption-refiner result is reused between denoise steps. The noise-refiner path now runs only on image tokens at the next scheduler timestep before the caption is reattached. The original pinned packed nine-step gate ran 4,618.35 seconds and failed at rollout step 1 with relative L2 1.4135604, above the 0.923 envelope. A corrected-policy packed artifact ran 4,711.53 seconds and failed at the same check with 1.4136423. Conditioning completed, and the production-shape grouped-attention test passes, so this is an end-to-end quality blocker rather than a compile or conditioning blocker. The next owner must capture the first divergent denoise intermediate, preserve the frozen envelope, and rerun the full gate only after that boundary is explained.
    - **Progress (2026-09-15)**: Added an opt-in first-step Metal trace that stops after scheduler update one and reports the conditioning, patchification, noise-refiner, final main-transformer, velocity, and scheduler-latent boundaries. The reference capture now records matching bounded arrays with manifest shape checks. The 1.414 rollout blocker is then explained WITHOUT touching the kernel: the frozen fixtures were captured from torch CPU `randn(seed 42)` while the native backend rolls out from its own xorshift and Box-Muller `seeded_noise`, and those two fields are independent draws. Measured directly, native noise reads relative L2 1.4132 against the captured `initial_noise` array at correlation -0.0014, where two uncorrelated unit-scale fields sit at sqrt(2) = 1.4142; both complete gate runs failed at 1.4135604 and 1.4136423, within 0.03 percent of that floor, and the noise-independent conditioning boundary passed at 0.0815. So any implementation, however correct, reads about 1.414 against these fixtures. The packed parity and trace gates now seed the denoiser from the captured `initial_noise` fixture (`denoise_steps_from_noise` and the trace's `initial_noise` parameter), keeping the frozen 0.923 envelope and leaving production generation on native noise. The first matched-noise trace on the pinned install (`z-image-ig2-noise-trace.json`, 1,207.63 seconds) then reads conditioning 0.0815, noise-refiner 0.0644, main-transformer intermediate 1.0856, velocity 0.6029, and first scheduler latent 0.0395 against the 0.923 envelope, so the packed first step is well inside the envelope and the two intermediate readings stand as new evidence without a frozen analogue, because IG0 froze only conditioning and final latents. The first complete matched-noise gate run (4,778.46 seconds) then PASSED every denoise check the old noise mismatch had masked, conditioning through all nine latents and the final latent, and reached the VAE for the first time, where it exposed a real load-time defect: the packed VAE records the four mid-block attention projections as 2-D Diffusers `nn.Linear` weights, `[512, 512]`, while the Metal path demanded `[512, 512, 1, 1]`. The 1x1 conv kernel indexes a weight as `oc * in_channels + ic`, byte-identical to the Linear layout the CPU reference already applies, so the fix is the Metal shape expectation alone; a cross-check of the whole VAE index against the Metal path confirms those four are the only 2-D decoder tensors. A focused opt-in VAE decode gate (`packed_native_vae_decodes_the_frozen_latent`) was added so VAE issues iterate in one decode instead of one denoise plus one decode.
    - **Progress (2026-09-15, main-transformer checkpoint trace)**: A fresh verified pinned source download and packed install reran `packed_native_first_step_trace_localizes_divergent_boundary` in 576.69 seconds. Conditioning remained within the frozen envelope at 0.0815408, patchification was 0.0016627, and noise-refiner output was 0.0509079. Selected main-transformer checkpoints read 0.0688876 at block 0, 0.0762979 at block 15, 0.0878091 at block 16, 0.1795987 at block 20, 0.4174181 at block 24, 0.9539717 at block 28, and 1.0846142 at block 29; velocity was 0.5992641 and scheduler latent was 0.0392788. The first major issue is progressive accumulation in the later main-transformer recurrence, crossing the frozen 0.923 envelope by block 28. This does not yet distinguish packed INT4 drift, BF16-versus-F32 accumulation, or a repeated layout/dispatch error. The next owner must compare those paths inside the recurrence before changing kernels or tolerances.
    - **Progress (2026-09-15, recurrence controls)**: A matched Diffusers reference with the same group-64 INT4 policy reduced native conditioning error to 0.0042043 but left the late recurrence unchanged: native main-transformer error was 1.0849981, with block 0 at 0.0636551, block 16 at 0.0836585, block 24 at 0.4202046, block 28 at 0.9507074, and block 29 at 1.0849981. The ordinary BF16 reference control was effectively the same, ruling out a simple INT4 source-policy or conditioning provenance cause. A fresh packed install narrowed unquantized F32 source tensors to BF16, leaving 238 INT4 projections and 283 BF16 transformer tensors; its block-29 error remained 1.0846142. A diagnostic block-boundary BF16 rounding pass changed it only to 1.0834699 and was removed. The unresolved boundary is intra-block native activation or accumulator precision, or a repeated native dispatch contract. The frozen 0.923 envelope remains unchanged. During the rebuild, a stale-source issue was also found and fixed: `metal_ops` requested `image_rope_orthogonal` while the shader exported `image_rope`.
    - **Progress (2026-09-15, dispatch and math controls)**: The seeded trace was rerun with a CPU completion after every image primitive (`TURBOSPARK_IMAGE_FORCE_SYNC_DISPATCH=1`) and reproduced the baseline exactly, including block 28 at 0.953971744 and block 29 at 1.08461416. A second run disabled Metal fast math (`TURBOSPARK_METAL_PRECISE_MATH=1`); block 28 moved only to 0.953971624 and block 29 to 1.08461368. Same-queue hazard handling and relaxed compiler math are therefore ruled out as material causes. The captured main-block arrays carry `torch.bfloat16` source dtype, while native image tensors and shader outputs remain FP32. The next owner must test intra-block BF16 activation or reduction behavior. The 0.923 envelope remains unchanged.
    - **Progress (2026-09-15, isolated intra-block BF16 experiment)**: `MetalImageBackend::intra_block_trace` now accepts one captured 1024x1024 main-transformer block input and AdaLN embedding, reports labeled modulation, Q/K/V, RoPE, attention, output-projection/residual, and FFN snapshots, and optionally inserts a Metal round-to-nearest-even BF16 storage pass after every operation. The ignored block-28 probe completed on the pinned install in 61.65 seconds. Its isolated comparison against `block_28_output.npy` moved from relative L2 `0.36354998` without inserted rounding to `0.36287159` with rounding. The effect is small, so blanket per-operation BF16 storage is not a sufficient fix and the `0.923` envelope remains unchanged. The next probe should compare these labeled snapshots with a matching reference operation trace before changing a kernel or layout.
    - **Progress (2026-09-15, adjacent-RoPE fix)**: The matching block-28 Diffusers trace from `scripts/z_image_intra_block.py` identified the first native/reference jump at Q/K RoPE. The Metal odd adjacent-RoPE lane used `even*cos + odd*sin`; complex multiplication requires `even*sin + odd*cos`. The one-line shader fix reduced native/reference Q/K RoPE error from `0.8727`/`0.9175` to `0.00403`/`0.00407`, and isolated block-28 output error from `0.36354998` to `0.0848376` (BF16 storage control `0.0849524`). The seeded first-step trace was rerun on the pinned install: block 20 `0.0444867`, block 24 `0.1792571`, block 28 `0.4385121`, block 29 `0.4336081`, with conditioning `0.0815408`, noise-refiner `0.0070217`, velocity `0.2764470`, and scheduler latent `0.0181197`. This is a material numerical improvement below the frozen `0.923` envelope, but it does not close the complete quality, VAE, PNG, cancellation, or quiet-machine resource gates.
    - **Progress (2026-09-15, VAE gate isolation)**: The missing frozen `decoded_pixels.npy` fixture was regenerated from the pinned Diffusers VAE, making the isolated native VAE arm runnable. On the corrected packed install, `packed_native_vae_decodes_the_frozen_latent` exceeded a 180-second safety timeout without a completion or parity result and was terminated with no surviving child process. This reproduces the documented GPU wait/spin blocker and keeps the VAE, full packed quality, PNG, cancellation, and resource gates open. Do not treat the timeout as a numerical parity result.
    - **Progress (2026-09-16, VAE parity closure)**: The apparent wait/spin was long production-shape work, not a wedged kernel. A stage-isolated run completed every decoder boundary in 395.13 seconds, and the non-instrumented pinned gate `packed_native_vae_decodes_the_frozen_latent` passed in 370.37 seconds with max absolute error `4.7907233e-6` and relative L2 `3.1853588e-7`, inside the frozen `6e-5` and `3e-6` limits. The native VAE attention now computes shared 8-query score tiles, GroupNorm reduces each group cooperatively, and convolution reuses loaded weights across eight spatial outputs. The complete matched-noise nine-step gate is being rerun; PNG, cancellation, and quiet-machine resource evidence remain open until that run and their dedicated gates pass.
    - **Progress (2026-09-16, packed quality and intake)**: The complete matched-noise nine-step quality/VAE gate passed on the same pinned packed install, including all nine rollout checks, isolated frozen-latent VAE parity, and finite decode of the packed final latent. The dedicated PNG metadata gate also passed after a full native generation, asserting seed, scheduler steps, transformer-forward count, guidance, model revision, PNG signature, and exact serialized metadata. The live quiet resource oracle completed its cold arm in 6,515.892 seconds: PNG relative L2 was `0.4284977`, peak process footprint was `20,725,728,336` bytes, the VAE was the dominant stage, physical reads were zero, and swap did not increase. Its corrected warm-only rerun passed in 4,572.05 seconds: PNG relative L2 was `0.4284977`, peak process footprint was `19,874,055,248` bytes, physical reads were zero, swap was unchanged, nine forwards completed, and no buffers remained retained at idle. A separate image catalog now pins the canonical Diffusers file set and its live network rot guard passes; `pull-image --repo OWNER/NAME@REV` streamed `32,848,304,827` source bytes and atomically published a `6,667,547,680`-byte verified install whose packed component hashes match the locally exercised artifact. Real CLI cancellation passes on the pinned install, with no PNG published. A sparse view of that real install returns the expected missing VAE payload error before generation. No-device cross-target compilation succeeds with Zig, and the compiled integration test executes on Linux ARM64 in the isolated Lima VM, passing the unsupported-platform refusal before model I/O.
    - **Closure (2026-09-16)**: All five IG2 status items now pass on the pinned real install. The no-device integration test ran on Linux ARM64 and returned 1 passed, 0 failed; the unsupported-platform branch is execution-verified, not only cross-compiled. IG2 is closed with its packed resource peaks recorded as evidence, not as a new minimum-RAM claim. IG3 remains the owner of bounded-memory and lifetime work.
    - **Progress (2026-09-15, continued)**: Two findings postdate the entry above. First, the rerun that exercises the new VAE gate hung: observed at 10:33-10:41 with the test thread parked in `MTLCommandBuffer.waitUntilCompleted` inside `MetalImageBackend::decode_impl`'s first `metal_ops::read`, process CPU frozen at 4:11 total while `IOAccelerator` reports 100 percent device utilization, which is a spinning kernel on the GPU rather than slow work. This is the first production-shape native VAE decode ever executed (the earlier gate failed at the shape check before encoding), so the wedged kernel is a real decode-path blocker and any machine sharing the GPU is stalled until the process is killed. Second, the no-INT4 escalation arm is not runnable on this machine: `z-image-turbo-precise.image.gturbo` (31 GB, zero quantized tensors) aborts at open, because `metal_ops::Component::open` maps the whole payload and wraps it in a single `newBufferWithBytesNoCopy`, which exceeds this 36 GB machine's wired limit; Metal returns null and the metal crate's null assertion aborts before `Component::open`'s `map_err` can produce the intended error string. A graceful over-limit refusal is missing (one pre-check on payload length against the wired limit would turn the abort into the designed error), and the main-transformer 1.086 boundary therefore remains unclassified between INT4 drift and structural error.
    - **Current gate status (2026-09-16)**:
      1. **Native Metal backend**: implemented behind `crates/image/src/runtime.rs`'s `ImageBackend` trait. Shared GPU context, pass, and resident-buffer contracts are reused only where their layouts and lifetimes match; image-specific linear, attention, scheduler, transformer, and VAE operations use dedicated MSL and wrappers.
      2. **Packed-runtime parity**: the opt-in gate compares conditioning, all nine latent updates, final latents, and decoded output against the IG1 fixtures. The complete matched-noise gate now passes all rollout checks, isolated frozen-latent VAE limits, and finite decode of the packed final latent on one pinned install.
      3. **Real install and catalog path**: local `turbospark-model pull-image` records a separate image install and does not create a text `Mlx` row. The image catalog pins the canonical Diffusers source plus the four published MLX aliases, its live rot guard passes, and the real `pull-image --repo OWNER/NAME@REV` exercise streamed the canonical source into a verified atomic install with matching packed payload hashes. The MLX aliases have header and selected-payload compatibility gates; the 2-bit, 4-bit, and 8-bit full installer gates pass, while the FP16 full install remains open.
      4. **Production CLI selection**: native macOS selection, explicit CPU reference mode, validation, overwrite protection, output publication, and unsupported-platform paths are implemented. The real-install missing-component refusal and cancellation paths pass. The no-device integration test executes on Linux ARM64 and passes the unsupported-platform refusal before model I/O.
      5. **Resource and quality closure**: the complete packed quality/VAE gate and PNG metadata gate pass. The quiet resource oracle passes both cold and warm-only arms. Its packed peaks are recorded separately from the smaller IG0 reference ceiling because the packed VAE reaches `20,725,728,336` bytes cold and `19,874,055,248` bytes warm. Dedicated real-install cancellation passes.
    - **IG3 execution and closure**: The packed resource oracle preserved the pinned install and current quality envelopes while measuring component weights, conditioning, latents, activations, scratch, staging, in-flight GPU work, allocator retention, `phys_footprint`, physical reads, and swap as separate evidence. The ownership audit in `crates/image/src/runtime.rs` and `crates/image/src/metal_ops.rs` proved cancellation waits for outstanding GPU and I/O consumers before buffers are freed or reused. The resident and bounded two-slot sequential streaming paths passed exact output agreement, no early slot reuse, live-storage bounds, measured latency, and refusal before execution when the largest required block and workspace cannot fit the requested budget. The repeated-job and repeated-denoise oracle then passed on the pinned install, closing IG3. VAE tiling, prefetch, and allocator reuse remain deferred to IG5. Do not widen the `0.923` envelope or claim a minimum whole-machine RAM figure.
    - **IG2 closure rule**: IG2 stayed unchecked until all five items above passed on one pinned real install. They now pass. The local packer, native backend, CPU reference, metadata, progress, cancellation seams, and synthetic or opt-in tests are implementation groundwork unless the real packed gates support them; this closure is based on the recorded real-install evidence. IG4 app/Swift work is now closed under the bounded native ownership established by IG3.
  - [x] **IG3**: Prove bounded lifetimes and measured memory; add sequential block streaming only where the target budget requires it. Runtime admission, work tracking, the public cancellation idle hook, two-slot accounting, and the repeated-job/denoise resource oracle are implemented and focused-tested. On 2026-09-17 the pinned plan/refusal gate passed, and the current-source resident/streamed gate passed exact PNG agreement with a 0.985694 streamed/resident latency ratio, 214,918,144 slot bytes, and 393 fenced slot reuses. The repeated warm-only hardware oracle then passed with two complete jobs and two matched-noise denoise cycles: complete-job peak and idle footprint growth were 113,557,576 and 83,476,552 bytes, denoise peak growth was 15,663,176 bytes, managed allocations were stable at 7,784 and 7,138 respectively, and page-ins and swap deltas were zero. Exact PNG and final-latent repeat checks passed. IG3 is closed; VAE tiling, prefetch, and allocator reuse remain deferred to IG5. Swift image integration is closed under IG4.
  - [x] **IG4**: Expose the same runtime through Swift; add the top-level Images destination, job serialization, preview/save/regeneration, and profile-scoped artifact persistence.
    - **Closure (2026-09-17)**: Added the separate `TsImageSession` C ABI and Swift `TurboSparkImageSession`, including stage callbacks, cancellation, explicit PNG ownership, and the Swift-facing `modelID` metadata spelling. The app now has a canonical top-level `Images` destination with prompt-first `Create` and `Gallery` tabs, a separate native listing and picker for valid installed image artifacts, a side-load folder override, a process-wide FIFO image-job coordinator, live progress, preview, save, regenerate, profile-scoped PNG artifacts, thumbnails, and a previous/next carousel. Saved image paths render in persisted transcript rows as well as the artifact panel. Interrupted jobs remain transient and are not restored as completed work. Focused Rust ABI tests, the real-install Swift cancellation path, Swift package/app builds, localization, and the signed release app bundle pass. The pinned 1024-by-1024 real-install Swift PNG gate passed in 4,665.912 seconds, the cancellation gate passed in 21.827 seconds, and the focused app image-job suite passes 9/9. The broader P0/P2 measurement and artifact backlog remains open as separate repository work; it does not reopen this image integration milestone.
  - [ ] **IG5**: Tune measured bottlenecks without weakening quality or memory gates; approximation work needs a separate proposal.
- **First Release Boundary**: One image per prompt, native Metal, exact dense execution with validated quantization. Image editing, LoRA, batching, agent tools, HTTP image endpoints, PISA, and approximate timestep reuse are deferred. Existing text consumers must retain their behavior.

#### 12. MiniMax-M2 Release Gates (GGUF Execution Implemented)

- **Landed (2026-09-10)**: Validated split-GGUF streaming, FP32 router/bias preservation, whole-projection Q/K normalization, partial RoPE, sigmoid expert selection, and checkpoint framing. The pinned Q4_K_M artifact is installed; synthetic Metal tests and shared Llama/Qwen3 regressions pass.
- **Verified**: The real memory oracle passes at 8192 context and eight slots; both short-answer EOS checks pass. Two fresh quality processes reproduce perplexity and digests. [The implementation record](docs/MINIMAX_M2_PHASE0.md) owns pins and measured figures.
- **Release Blocker**: Greedy and low-temperature sampled coastal-wetlands smokes repeat reasoning and exhaust 400 tokens. Temperature 1 completes coherently at 1240 tokens, but does not waive those failures.
- **Remaining**: Resolve the repetition, pass the release smokes, review/freeze quality and quiet-machine performance baselines, then add the exact artifact to the catalog and promote support.
- **Deferred**: Safetensors/FP8 intake, mapped expert residency, native tool-call parsing, MTP, vision, and later MiniMax variants.

---

### Priority 3: Long-Term Extensions & Platform Expansion

System architecture extensions, platform ports, and developer tooling.

#### 1. Remote Plugin Marketplace & Registry Indexing (`TurboSparkApp`)
- **Status (corrected 2026-09-15)**: The objective as written LANDED already, and this entry's paths were stale twice over. `PluginMarketplaceManager.swift` lives in `swift/TurboSparkApp/Sources/TurboSparkApp/State/` (there is no `Plugins/` directory and no `PluginManifest.swift`; the manifest types are in `State/PluginManifestParser.swift`), and its `fetchMarketplace(name:source:)` already covers all four `MarketplaceSource` cases: https via URLSession, `.github`/`.git` via `MarketplaceGit` clone-or-pull, and local directories. Install, versioned caching and the v2 ledger are done and tested.
- **Why Still Open**: only the DESCOPED remainder, `swift/docs/SWIFT_PLUGINS.md`'s out-of-scope list: a shipped/builtin registry of community marketplaces (registry INDEXING), auto-update, and dependency closure. Deferred with the rest of the Swift work until the active `State/` session commits.

#### 2. Multi-Direction Steering & Automated Alpha Calibration [multi-direction landed 2026-09-15]
- **Status**: N vectors per policy LANDED. `runtime::SteeringPolicy` carries `Vec<SteeringVector>` (per-vector mode, alpha; band applied at load); the FFI/Swift wire stays single-vector (`primary()`) with the full list on the summary line. Application is K in-order dispatches of the existing kernel at each steered layer -- composition is sequential by construction, and the coeff buffer is per-(layer, vector). CLI/server: repeatable `--steering` with positionally-paired `--steering-mode/--steering-scale/--steering-layers` (`invocation::steering_knob` is the rule; shorter knob lists extend by their last value; more is refused). Gates: 13 runtime unit tests + a gpu composition test (two dispatches on one buffer bit-identical to sequential) + the multi-vector null control on real gptoss (a zero-alpha second vector is bit-identical over a 24-token walk) + the Add-composition arm (200,560/201,088 logits move). Both gemma4 smokes clean. `docs/OBLITERATION.md` has the section; the parse surface is mutation-checked.
- **Still Open**: automated alpha CALIBRATION as a command (`steering_sweep.rs` is the instrument; it needs a multi-vector mode), `activation_capture.rs` (the contrastive-capture fixture pipeline), a second REAL direction for one install (the composition arms ran one real vector plus scaled copies), and the Swift preset surface (deferred with Swift).

#### 3. Batched Sub-Byte GEMMs for Bonsai / Ternary Speculation [landed 2026-09-15]
- **Status**: LANDED, and this entry's paths were stale (the kernels live in `shaders/dequant_1bit.metal` / `dequant_2bit.metal` beside their GEMVs, dispatched by `dequant_{1,2}bit_gemm_batch.rs`). `dequant_int{1,2}_gemm_simd` batch B right-hand sides inside one dispatch with the GEMV's row mapping, byte-walk order and affine factoring unchanged, so B rows are BIT-IDENTICAL to B GEMV calls at every width 1..16 -- the parity tests (`crates/gpu/tests/dequant_{1,2}bit_gemm_parity.rs`) assert that against the real GEMVs, plus a CPU-reference tolerance arm, plus a two-widths-one-context case that reddens a pipeline-cache key missing the baked B. Field order, bit order and x-stride mutations each redden only the parity suites. `encode_gemm_any` dispatches 4/15/16 and refuses the rest by name; `speculation_blocker` (both the MTP and the DFlash2 copy) now probes the set {4, 15, 16}, and the 1-bit-with-head fixture asserts NO blocker and a running `produce_batched`. INT8 is the surviving refused-width fixture (`real_forward_gemma4_chunked.rs`).
- **Honest boundary**: end-to-end speculation on Bonsai/Ternary still needs a DRAFTER artifact (mlx conversions drop `mtp.*`; no DFlash2 state in those installs), so the real-model int2 batched arm runs through `produce_batched` only where a drafter exists. The 1-bit synthetic verify gate covers the kernel end to end on Metal; a real ternary `produce_batched` arm stays gated on a drafter (the probe refuses a headless install at open, by design).

#### 4. DeepSeek-V4-Flash Metal Kernels & Feasibility (Scaffolded)
- **Objective**: Port CSA/HCA attention, unrolled mHC Sinkhorn, and sub-3bit GEMV Metal kernels (`dsv4.metal`), assessing 106.9 GB peak RSS memory feasibility.
- **Why Open**: Architecture is scaffolded; full implementation requires large unified memory (128 GB Mac).
- **Files to Touch / Create**:
  - `crates/gpu/src/shaders/dsv4.metal` [NEW]
  - `crates/gpu/src/`
  - `crates/runtime/src/families/dsv4/` [NEW]

#### 5. Linux Backend (Portable Architecture)
- **Status (slice landed 2026-09-15, compile-gated)**: `crates/streaming/src/linux_uring.rs` [NEW] carries the `io_uring`+`O_DIRECT` read source behind `TURBOSPARK_LINUX_IO=auto|pread|uring` with **auto = pread until a Linux session proves the uring path** (no Linux machine here; the accepted gate is the cross-target `cargo check`, and the portable selector/alignment parts are unit-tested on any machine). `rdadvice.rs` gained the `posix_fadvise(WILLNEED)` Linux arm, and `crates/model-io/src/cgroup.rs` [NEW] is the cgroup-v2 `memory.max`/`memory.high` probe (pure, tested anywhere) that the runtime's Linux `physical_memory()` arm should call.
- **Still Open**: a Linux session to RUN the uring path and flip the default; the runtime-side cfg arm consuming the cgroup probe; and the Vulkan compute backend, which remains a multi-session project -- `crates/compute`'s `ComputeStrategy` is an empty marker struct, so a dispatch trait must be designed first. Budget a Phase-0 fact-finding pass under `docs/NEW_MODEL.md`'s discipline before writing kernels.

#### 6. Server Multi-Runner Pool (Option 2) [landed 2026-09-15]
- **Status**: LANDED as `--pool-size N` on `turbospark-server` plus a `PoolRegistry` (`crates/server/src/registry.rs` -- this entry's old paths never existed; the session pool lives in `crates/runtime`). N independent opens of ONE install sit behind one public id; each member carries its own KV, session pool and one-permit `GenerationQueue`, and `resolve` routes each request to the LEAST-QUEUED member with a round-robin tiebreak, so the fan-out lives at the registry level where the queue and the generation are one self-consistent handle. `/v1/models` reports ONE row. Every member pays the load guard on its own, so a member that does not fit refuses at startup with the subtraction. Tests: identical-identity acceptance (the exact mirror of `StaticRegistry`'s duplicate refusal), pool-of-one refused, context-mismatch refused, one-row reporting, idle round-robin spread by `Arc` identity, and the unknown-name fallback.
- **Multi-install follow-up (2026-09-17)**: `--model` is now repeatable. Distinct generation installs open through the existing `StaticRegistry`, preserve the single-model unknown-name fallback when only one install is attached, and reject the ambiguous combination with `--pool-size`. Parser coverage pins command-line order and the refusal. The observer plus Swift `ServerMetricsStore` already expose in-flight requests, per-model service, decoder-owned token rates, and measured queue wait.
- **Still Open**: a concurrency real-model arm on a machine with enough free memory for two distinct runners or two pool members. Do not call the end-to-end server concurrency gate complete from parser or registry tests alone.

---

### Priority 4: Measurements, Baselines & Quality Sweeps

Verification sweeps, cross-engine KL proofs, and power captures.

#### 1. Cross-Engine KL Verification (`qwen38`, `qwen36`)
- **qwen38 DONE (2026-09-15)**: `logit_dump.rs` gained the `TURBOSPARK_QWEN38_INSTALL_DIR` arm, `kld_mlx_affine.py` gained the `qwen38` CHECKPOINTS row (498 modules, uniform 4/64, reference already in the HF cache), and the run froze `docs/BENCHMARKS.md`'s row: forward KL mean **0.000788 nats at 98.78% top-1, 0.72x MLX's own cached-vs-batched floor** (0.00110) -- the first row here below the reference's floor. Port perplexity 4.9432 reproduced the frozen quality row to the last digit on the freshly re-streamed install. Evidence: `docs/verification/kld_mlx_affine-qwen38.json`.
- **qwen36 still blocked on TWO downloads, nothing else**: the CHECKPOINTS row is fully pinned (NEVER RUN), and `logit_dump.rs` already accepts `TURBOSPARK_QWEN36_INSTALL_DIR`. Missing: `~/models/qwen36.gturbo` (re-stream via `crates/repack/tests/qwen36_checkpoint_network.rs`, ~19 GB) and the HF reference `mlx-community/Qwen3.6-35B-A3B-4bit` (not in the cache, ~18 GB). Do not copy qwen38's numbers across: this checkpoint is MoE, and the MoE floors are 30-300x larger.

#### 2. Missing Quality & Memory Oracle Baseline Rows
- **Mistral quality row DONE (2026-09-15)**: `crates/bench/tests/mistral_quality_gate.rs` [NEW] froze on the first run (perplexity **9.3971**, greedy `522026e6...`, sampled `2a98d760...`, two fresh processes agreeing, constrained-arm digest byte-identical) and was re-run to ASSERT the row. The family's memory row has been frozen since 2026-09-10; this completes dense Mistral's gate pair.
- **qwen38 clause = P4.1's qwen38 row, DONE** (same work, closed once).
- **Bonsai DONE (2026-09-16)**: install re-pulled and `bonsai_{quality_gate,memory_oracle}.rs` [NEW] frozen from first runs and asserted on re-run. Perplexity **8.3554** -- identical to the last digit with the frozen KL row's independent reading (cross-instrument match); peak **660 MiB**, within 4 MiB of ternary's same-architecture same-window row, exactly as the Gotcha 40 prediction said. The oracle's first run read medium/long prefill 12x slow after five back-to-back gates; the cooled re-run read normal (18.5-16.7 tok/s), and BOTH the bonsai and ornith9b first-run throughput anomalies that night were heat-soak -- recorded in the files as a class, with Gotchas 22/28 extended to oracle floors.
- **TinyLlama DONE (2026-09-16)**: install re-pulled and `tinyllama_{quality_gate,memory_oracle}.rs` [NEW] frozen and asserted (perplexity **17.6084**; peak 349 MiB at the llama flow's 8,192 -- within 3 MiB of the KV arithmetic; 185 tok/s short decode). The oracle runs SHORT-EXPLANATION ONLY via the existing `run_oracle_over_cases` seam: both longer cases stop MaxTokens on this checkpoint (a 2023 1.1B model rambles), and the shared validity gate correctly refuses such runs.
- **Qwen3-MoE DONE (2026-09-17)**: the pinned `Qwen/Qwen3-30B-A3B-GGUF` Q4_K_M install was re-streamed to `~/.turbospark/models/qwen3moe.gturbo`. The quality gate reproduced perplexity **14.5988** and both frozen digests; the memory oracle reproduced complete answers at **2,663.8 MiB** peak, **27.907 / 27.176 / 17.245 tok/s**, and **+0.00 MiB** replay growth. The existing same-GGUF llama.cpp KL row remains the cross-engine evidence. The current CLI smoke is coherent. This closes the Qwen3-MoE validation gap; the install is retained locally.
- **Still blocked on downloads, unscheduled**: museglimmer, minimax (their gate files exist; the installs do not). qwen36's clause lives in P4.1's descope note. (`qwen4-reap288` left this list on 2026-09-18: the install was re-streamed, every frozen row re-verified, and the P0 probe/smoke items plus P2.1's profile all ran -- see `docs/verification/qwen4-2026-09-18.json`; the install was then removed per the session's space goal and re-enters the missing list below, now owed specifically by P2.1's kernel build.)

#### 3. Power Profile Sweep Across Remaining Catalog Rows
- **Ternary row CAPTURED (2026-09-15, 23:41 UTC)**: `ternary27b.gturbo` under `power.sh 2`, clean machine (contamination floor 101 mW), but **every measured phase reached Heavy** -- a governed capture with no unconstrained window to contrast. Publishable: short-explanation decode ~31 W, ~2.95 J/token, ~10.4 tok/s, pairs reproducing to 0.4%. Long-synthesis is heat-soak documentation, not a row (p1 prefill 2.2x p2's wall time on identical work; 80.9% spread). Recorded with per-arm rows in `docs/POWER_BASELINE.md`. Owed on this install: a `COOLING=max` rerun for the un-governed point, ideally in one session with a `qwen38-27b` capture -- that pairing is the clean same-architecture 2-bit-vs-4-bit comparison this capture cannot be.
- **Qwen38 row CAPTURED (2026-09-16, 02:04 UTC)**: clean machine (170 mW floor). Long-synthesis decode reproduces to **0.2%** -- ~26.2 W, ~1.507 J/token, ~17.4 tok/s, the tightest multi-pair row on the page and the 4-bit leg of the same-architecture pairing (ternary 2-bit reads ~2.95 J/token at ~10.4 tok/s all-Heavy). Two means REFUSED and recorded per-arm: medium-review p2 ran Heavy and 33% faster than Moderate p1 (label and throughput disagree; unexplained), and short-explanation decodes more expensively than long (arithmetically its cpu_W gap; why the CPU drew 2-4x more is the residual). `docs/POWER_BASELINE.md` has both tables.
- **Owed on the pairing**: the `COOLING=max` ternary rerun (ideally one session with a qwen38 arm). `ornith9b` and `ornith35b` remain MISSING FROM DISK and block their rows.
- **Preflight that both 2026-09-15/16 captures satisfied, and the next one must too** (per `docs/POWER_BASELINE.md` and AGENTS.md Gotchas 22/28/43): AC power (`pmset -g ps`), a quiet machine (the contamination-floor line warns above 2,000 mW; both captures read 101/170 mW), two pairs per case, and `COOLING=max` for a saturating install -- which ternary turned out to BE, so the rerun carries it. One install is one command: `LABEL=ac MODEL=<install> scripts/power.sh 2` (sudo, ~12 min); rows land in `docs/POWER_BASELINE.md` beside the uncooled baseline, per-arm TSV into `docs/verification/`.
- **DESCOPED (user decision, 2026-09-16): the ornith35b-vs-qwen36 same-session J/tok A/B** -- it wanted ~37 GB of re-pulls (ornith35b 18 GB + qwen36 ~19 GB) for one comparison row; the existing separate rows stand, with their cross-session caveat. This also leaves P4.1's qwen36 KL row without a scheduled pull (its checkpoint row stays pinned in `scripts/kld_mlx_affine.py`; re-open by pulling the install + reference).
- **Still Open**: the ornith9b re-pull (8.9 GB -- its gates exist with frozen rows and cannot run until it lands) and the ornith35b re-pull (18 GB, same state, unscheduled), the rate-cap sweep (`ARMS=default,30,20,15,10` under `COOLING=max` on gemma4) still owed from the 2026-09-10 entry, and the `COOLING=max` ternary rerun from above.

---

## Artifact State & Disk Usage

Disk space is a key constraint for downloading large checkpoints and running cross-engine KL reference dumps.

Re-derived from `ls ~/models` and `ls ~/.turbospark/models` on **2026-09-17** (about 8.9 GiB free after the qwen3moe re-stream). Do not start another large pull without reclaiming space or an explicit storage decision: the volume is at 100% and the remaining free space is below the normal 20 GiB preflight headroom.

| Path in `~/models/` | Size | Status / Associated Targets |
|---|---|---|
| `gemma4.gturbo` | 13G | PINNED: smoke, memory oracle, sensitivity proof, mapped residency |
| `qwen38-27b.gturbo` | 14G | RE-STREAMED 2026-09-15 (pinned index SHA verified). Backs `qwen38_{quality_gate,memory_oracle}`, the qwen38 KL row (P4.1, closed), steering probes, power row CAPTURED 2026-09-16 (P4.3) |
| `ternary27b.gturbo` | 7.1G | `ternary_{quality_gate,memory_oracle}`; power row CAPTURED 2026-09-15, governed/all-Heavy (P4.3, `docs/POWER_BASELINE.md`); `COOLING=max` rerun owed |
| `gptoss-20b.gturbo` | 11G | `gptoss_{quality_gate,memory_oracle}`, steering probes (the multi-direction arms ran here), mapped residency |
| `mistral7b-dense.gturbo` | 4.1G | `mistral_memory_oracle`; quality gate frozen 2026-09-15 (P4.2) |
| `qwen3-06b-regression.gturbo` | 604 MiB | Dense Qwen3 Q8_0 regression gates |
| `ornith9b.gturbo` | 8.9G | RE-PULLED 2026-09-16. `ornith9b_{quality_gate,memory_oracle}` re-verified against frozen rows SAME DAY (perplexity 6.0503 + both digests exact; oracle floor held on cooled re-run). KL row was already frozen |
| `bonsai27b.gturbo` | 3.9G | RE-PULLED 2026-09-16. `bonsai_{quality_gate,memory_oracle}` [NEW files, P4.2] frozen and asserted: perplexity 8.3554 (= the frozen KL row's reading, cross-instrument), peak 660 MiB |
| `tinyllama-dense.gturbo` | 862M | RE-PULLED 2026-09-16. `tinyllama_{quality_gate,memory_oracle}` [NEW files, P4.2] frozen and asserted (short-case oracle; see the file for why) |
| `qwen38-27b-mtp.gturbo` | 14G | MTP speculative validation |
| `qwen38-27b-dflash2.gturbo` | 15G | DFlash2 block drafter validation |
| `qwen38-27b-vision.gturbo` | 15G | Combined vision trunk + tower install |
| `steering-vectors/` | ~3M | `ocean-gptoss`, `ocean-museglimmer`, legacy layer-base-0 fixtures for the regression tests and bench probes |
| `gguf-ref/`, `qwen38-mtp-ref/`, `skill-state-probe/`, `vision-probe/`, `vision-probe-qwen38/` | -- | Reference and probe sidecars |

| Path in `~/.turbospark/models/` | Size | Status / Associated Targets |
|---|---|---|
| `spark25.gturbo` | 2.4G | `spark_{quality_gate,memory_oracle}`, both frozen 2026-09-08 |
| `qwen25-7b-4bit.gturbo` | 4.0G | RE-STREAMED 2026-09-17. Qwen2.5 quality, memory, CLI, and MLX cross-engine KL gates pass; catalog row is `verified` |
| `dsv2lite-16b.gturbo` | 16G | RE-STREAMED 2026-09-17 by the concurrent DeepSeek2 bring-up; keep its ownership and gate record with that work |
| `qwen3moe.gturbo` | 17G | RE-STREAMED 2026-09-17. `qwen3moe_{quality_gate,memory_oracle}` reproduced; same-GGUF llama.cpp KL row already frozen |
| `qwen3vl-4b.gturbo` | 2.1G | STREAMED 2026-09-18. `qwen3vl_{quality_gate,memory_oracle}` frozen (17.3463; 793 MiB at 4096, two-case oracle); catalog row `qwen3vl-4b` `verified`. Env key `TURBOSPARK_QWEN3VL_INSTALL_DIR` |

**Missing from disk** (re-pull before the dependent item can run; ornith9b, bonsai27b and tinyllama were re-pulled 2026-09-16 and removed from this list):
- `museglimmer-30b.gturbo` (15G) -- museGlimmer steering probe + gates
- `ornith35b.gturbo` (18G) -- P4.3 power row; unscheduled (see P4.3's descope)
- `ornith35b-gguf.gturbo` (34G) -- P2 cross-engine rows
- `qwen36.gturbo` (~19G) + the `mlx-community/Qwen3.6-35B-A3B-4bit` HF reference (~18G, not cached) -- P4.1's qwen36 clause, unscheduled (see P4.3's descope)
- `qwen4-reap288.gturbo` (68G) -- RE-STREAMED 2026-09-18 (to `~/.turbospark/models/text/`), every gate re-verified against the frozen rows (quality 8.7224 + both digests exact; oracle 2517 MiB; KV4 and kv_quant_probe findings re-confirmed; P0.3 closed negative; P2.1 profiled), then REMOVED after evidence capture per the session's space goal. Owed again specifically by P2.1's kernel build; note the pull-transport observation in `docs/QWEN4_EXP.md` (2026-09-18) before re-streaming
- `minimax-m2-q4km.gturbo` (129G) -- release gates (the repetition blocker stands)

Storage preflights should retain at least 20 GiB headroom. Only disposable compiler caches are reclaimable; no models were deleted in this reconcile.

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
