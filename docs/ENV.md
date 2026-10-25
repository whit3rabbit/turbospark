# Environment variables in TurboSpark

This page lists the environment variables recognized by the CLI tools, HTTP
server, macOS app, Swift bindings, and Rust runtime. All TurboSpark variables
use the `TURBOSPARK_*` prefix. Prefer command-line flags when a flag exists;
use environment variables for test fixtures, diagnostics, and process-wide
defaults.

All TurboSpark-specific environment variables use the `TURBOSPARK_*` prefix.

Keep all docs and configuration ASCII (no emojis, no em dashes).

---

## 1. Core and Model Storage

| Variable | Affected Components | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_HOME` | CLI, server, Swift app, catalog | Base directory for downloaded models, catalog entries, and repack staging. Models are stored in `$TURBOSPARK_HOME/models`. | `~/.turbospark` |
| `TURBOSPARK_LAB_DIR` | `tslab` (`crates/lab`) | Directory for tslab ledgers, run records, frozen specs, trusted evaluator installs (`bin/`) and the machine lock (`.machine.lock`). | `$TURBOSPARK_HOME/lab`, else `~/.turbospark/lab` |
| `HF_TOKEN` | `turbospark-model` | Bearer token for accessing gated Hugging Face repositories during probing or pulling. | unset |
| `HUGGING_FACE_HUB_TOKEN` | `turbospark-model` | Alias for `HF_TOKEN`. Checked if `HF_TOKEN` is unset. | unset |

---

## 2. Server Configuration

| Variable | Affected Components | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_API_KEY` | `turbospark-server` | Optional API key required for incoming HTTP requests (`Authorization: Bearer <key>`). If unset, requests are unauthenticated unless specified by `--api-key`. | unset |

---

## 3. Swift Desktop Application (`TurboSparkApp`)

| Variable | Scope | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_HOME` | App model store | Override directory for installed models inspected or loaded by `TurboSparkApp`. | `~/.turbospark` |
| `TURBOSPARK_STATE_DIR` | App data store | Custom directory override for application state (chat history, projects, settings). Used by tests to sandbox state storage without affecting user data. | `~/Library/Application Support/TurboSparkApp` (app), temporary directory (tests) |
| `TURBOSPARK_HOOK_EVENT` | Subprocess hooks | Exported to hook scripts indicating the lifecycle event being triggered (e.g. `tool_call_start`). | set by app runner |
| `TURBOSPARK_SESSION_ID` | Subprocess hooks | Exported to hook scripts containing the UUID of the active conversation session. | set by app runner |
| `TURBOSPARK_TOOL_NAME` | Subprocess hooks | Exported to hook scripts containing the name of the tool executed (if applicable). | set by app runner |
| `TURBOSPARK_PROJECT_DIR` | Subprocess hooks | Exported to hook scripts containing the working directory of the current project. Also mirrored to `CLAUDE_PROJECT_DIR` for tool compatibility. | set by app runner |
| `TURBOSPARK_OPTION_<KEY>` | Subprocess hooks | Exported to hook scripts containing plugin/hook options with uppercase snake_case keys. Non-sensitive options only. Also mirrored to `CLAUDE_PLUGIN_OPTION_<KEY>`. | set by app runner |

### Swift Integration Tests
The following variables point Swift test targets at local artifacts:

| Variable | Scope | Description |
| --- | --- | --- |
| `TURBOSPARK_TEST_MODEL` | `swift test` (`RealModelTests`) | Absolute path to a `.gturbo` model directory used for real inference verification. |
| `TURBOSPARK_TEST_MODEL_NO_SPECULATION` | `swift test` (`RealModelTests`) | Absolute path to a model without speculative drafting (e.g. MoE or non-MTP) to verify drafter refusal paths. |
| `TURBOSPARK_TEST_IMAGE` | `swift test` (`RealModelTests`) | Absolute path to an image file (PNG/JPEG) used for end-to-end vision inference tests. |
| `TURBOSPARK_SKILL_STATE_SERVER` | `swift test` (`AppSkillStateRealModelTests`) | URL to a running server instance used for bounded-state agent skill tests. |

---

## 4. CLI and Generation Runtime Controls

These variables configure runtime diagnostics, kernel execution seams, and profiling.

| Variable | Affected Binaries / Crates | Purpose | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_PHASES` | `turbospark-check` | Set to `1` to print forward-pass phase timing breakdowns (GPU wait, router sync, expert pread, bind, logits wait) on stderr. | unset |
| `TURBOSPARK_DISPATCH_PROFILE` | `turbospark-check`, `turbospark-server`, `turbospark-bench` | Set to `1` to collect per-dispatch GPU kernel timing and ranking inside command buffers. | unset |
| `TURBOSPARK_DISPATCH_PROBE` | Runtime (qwen and llama dense decode) | Set to N to encode N in-place multiplies by 1.0 on each dense FFN activation, on the chain to the down projection. Bit-exact. `tslab profile --dispatch-probe 0,4,8` fits the price of one chained dispatch from it. Measurement only, never quotable. | unset |
| `TURBOSPARK_STANDIN` | Runtime (qwen and llama decode trunks) | A comma list of `attn`, `mlp`, `gdn`: skip those blocks and their residual adds, leaving the stream unchanged. Output is wrong by design; `tslab profile --standin` prices each block in place from it. Never quotable. | unset |
| `TURBOSPARK_ENGAGEMENT` | Runtime, gpu | Set to `1` to print `[engaged] NAME` to stderr the first time each opt-in lane runs (MMA8 tile, Steel prefill, rows attention, decode attention kernels, fused FFN, lazy GDN tape, probe, stand-ins). `tslab` sets it for every run and refuses an arm whose log contradicts its switches. Prints only. | unset |
| `TURBOSPARK_PREFIX_REUSE` | `turbospark-check --chat` | Set to `quiet` to silence the `[prefix-reuse] N/M` cache reuse line on stderr. | unset |
| `TURBOSPARK_GDN_TURN_SNAPSHOT` | Runtime (dense qwen only, prefix reuse on) | Set to `1` to keep the recurrent state 8 tokens before each prompt's end, so a next turn whose history re-renders the reply (reasoning dropped) rewinds there instead of prefilling the whole conversation. Holds one GDN mirror (about 147 MiB on Qwen3.8-27B) between turns. See `crates/runtime/src/turn_snapshot.rs`. | unset (off) |
| `TURBOSPARK_PREFILL_CHUNK` | `turbospark-check`, `turbospark-server` | Override prompt-processing chunk size (e.g. `128`, `256`, `512`). Takes precedence over `--prefill-chunk`. | unset |
| `TURBOSPARK_CHAT_DATE` | All (chat template rendering) | Override current date/time (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`) in templates calling `strftime_now`. Used to freeze prompts in deterministic benchmarks. | system UTC time |
| `TURBOSPARK_READ_QOS` | Streaming I/O pool | Set to `utility` to drop background routed-expert streaming read QoS on macOS from user-initiated to utility (E-cores). | unset |
| `TURBOSPARK_EXPERT_DISK_IO` | Streaming I/O | Set to `1` to measure physical disk read syscall bytes and amplification vs requested expert bytes. | unset |
| `TURBOSPARK_EXPERT_NOCACHE` | Streaming I/O | Set to `1` to bypass OS page cache on expert chunk reads (`F_NOCACHE`). | unset |
| `TURBOSPARK_EXPERT_CACHE_DECAY` | Streaming slot cache | A positive integer `N` switches the routed slot cache from cumulative LFU to LFU with every count halved each `N` plans (one plan per layer per forward pass). Output is unchanged; only which experts stay resident moves. Read once at open; a non-integer or `0` refuses the open. The CLI startup line names the active policy. See [cache policy](EXPERT_ROUTING.md#cache-policy-cumulative-lfu-forgets-nothing). | unset (cumulative LFU) |
| `TURBOSPARK_SHARED_CB` | Gemma 4 forward pass (`runtime`) | Set to `0` to disable overlapping shared expert command buffer with host expert pread. | `1` (enabled) |
| `TURBOSPARK_QWEN_SHARED_READ_OVERLAP` | Qwen GDN MoE (`runtime`) | Set to `1` to submit the unchanged shared expert on its own command buffer before blocking streamed expert reads. Experimental and captured at open; mapped residency keeps the inline shared branch. See [experiments](MOE_STARTUP.md#expert-scheduling-experiments). | unset (off) |
| `TURBOSPARK_QWEN_MOE_CHUNKED` | Qwen GDN MoE (`runtime`) | Set to `0` to turn off the MoE chunked prefill driver (`families/qwen/moe_prefill.rs`), so `turbospark-check` and `turbospark-server` prefill sequentially. Captured at open. Both arms produce byte-identical output, so it serves as the kill switch and the A/B reference. See [MINICPM.md](MINICPM.md). | unset (on) |
| `TURBOSPARK_MAPPED_DEMAND_PREP` | Qwen GDN MoE (`runtime`) | `off`, `advice`, or `touch`: prepare actual selected mapped expert pages before GPU use, with a 16 MiB page-span budget per layer. Advice failures are counted and tolerated. Experimental and captured at open; streamed residency does not prepare pages. | unset (`off`) |
| `TURBOSPARK_INT4_MMA8` | INT4 GEMV and batched GEMM dispatch (`runtime`) | Experimental, read once. `1` (shared arithmetic): every INT4 projection of at least 4096 output rows, in decode (one row) and speculative verify (up to 16 rows, in tiles of 8), runs the small-M matrix-unit GEMM. Decode and verify stay bit-exact against each other, but decode numerics change, so frozen digests do not hold. `verify` (drift-bound): only batched calls (verify, small prefill batches) take the GEMM, one-row decode keeps the GEMV, so verify differs from decode in the last bits. In both modes a fused FFN or GDN in-proj steps aside only when one of its matrices is served. Only qwen35 dense was gated. `TURBOSPARK_INT4_MMA8_MIN_ROWS` overrides the 4096-row threshold for tests. See [INT4_SMALL_M_MMA.md](INT4_SMALL_M_MMA.md). | unset (off) |
| `TURBOSPARK_ANE_PREFILL` | Dense Qwen prefill (`runtime`, `ane-prefill` cargo feature) | Opt-in ANE+Steel prefill split, captured at open. `1`, `on` or `split` runs each dense projection class split by output rows between a Core ML program on the Neural Engine and Steel inside chunked prefill; `auto` does that only on Apple Silicon M4+ / A18+ and keeps Steel otherwise; `break` keeps every command-buffer break and host handoff but Steel takes every row (a diagnostic for the handoff cost alone); anything else, including unset, is off. Needs the programs from `TURBOSPARK_ANE_PREFILL_DIR`; missing, failing, or shape-mismatched programs turn the split off for the runner with one warning and prefill runs on Steel. A failed prediction or a non-finite ANE row redoes the micro-batch on Steel. Steps aside while an MTP head or DFlash2 drafter is open. See [SPEEDUP_PLAYBOOK.md](SPEEDUP_PLAYBOOK.md). | unset (off) |
| `TURBOSPARK_ANE_PREFILL_DIR` | Dense Qwen prefill (`runtime`) | Directory of ANE prefill programs (`manifest.json` plus Core ML packages) written by `crates/coreml/tools/gen_ane_prefill_programs.py`, read when the split is on. | unset |
| `TURBOSPARK_ANE_PREFILL_STATS` | Dense Qwen prefill (`runtime`) | Set to `1` to print one counter line per split micro-batch on stderr. | unset |
| `TURBOSPARK_ANE_PREFILL_FAULT` | Dense Qwen prefill (`runtime`) | Diagnostic. The prediction at split point `P` of the first split micro-batch that reaches it fails, once per process, so the Steel redo path can be checked end to end. | unset |
| `TURBOSPARK_ROUTED_PIPELINE` | MoE dispatch (`runtime`) | Set to `0` to disable one-layer-pipelined routed command buffer execution. | `1` (enabled) |
| `TURBOSPARK_ROUTED_BATCH` | MoE prefill (`runtime`) | Set to `1` to run each layer's routed experts as one batched dispatch pair per union-bounded sub-batch during chunked prefill (Gemma 4, gpt-oss, and the Qwen MoE batched arm). On Qwen MoE it needs the batched arm and refuses steering and the residual capture by name. See `docs/BATCHED_PREFILL.md`. | `0` (off) |
| `TURBOSPARK_METAL_KERNEL_WARMUP` | Qwen affine-MoE text (`runtime`) | Set to `1` to prepare selected Metal pipelines during open. Experimental; unsupported configurations are refused. Applies to Rust/Metal text, not Swift MLX images. See [scope and results](MOE_STARTUP.md#compile-only-kernel-preparation). | unset (off) |
| `TURBOSPARK_BATCHED_GEMV` | Qwen GDN / MoE prefill (`runtime`) | Controls batched resident weight reads (M-row GEMMs on Steel at 64-row micro-batch). Unset is auto (default on for INT4 installs like Qwen 3.6, Ornith 35B); `=1` forces with by-name refusals; `=0` opts out to the byte-identical per-token arm. BF16 installs (MiniCPM-V 4.7) keep the per-token arm. See [BATCHED_PREFILL.md](BATCHED_PREFILL.md). | unset (auto, on for INT4) |
| `TURBOSPARK_FLINT_CHUNKED_PREFILL` | Flint prefill (`runtime`) | Set to `1` to run Flint's chunked prefill as the layer-batched driver: one command buffer per layer carries all M tokens' attention (still per token: sinks, ring, NoPE), the dense lead's FFN and the INT8 router GEMV, and the routed half then runs per token, pipelined across two banks so the expert pread for token t+1 overlaps the GPU running token t. Read once. Default OFF runs the sequential token-by-token path. The two arms are byte-identical (a runtime test and the lab's bit-exact quality gate pin it); the speedup is not established. Refuses `TURBOSPARK_BATCHED_GEMV` and `TURBOSPARK_ROUTED_BATCH` by name on a Flint prefill chunk. See [FLINT.md](FLINT.md). | unset (off) |
| `TURBOSPARK_FLINT_DEBUG` | Flint forward pass (`runtime`) | Set to `1` to report, at each MoE layer's mid-layer sync, any residual or router value that is non-finite or past 2000 (FP16 overflows at 65504, so 2000 leaves room to see a value approaching it). Exists because FP16 range is Flint's open risk. Adds a readback per layer, so it perturbs timing. | unset |
| `TURBOSPARK_FLINT_ROUTER_DUMP` | Flint forward pass (`runtime`) | File path. Appends one record per MoE layer per token: `position` (u32), `layer` (u32), the post-attention-norm MoE input as little-endian f16, then the INT8 router's logits as little-endian f32. Exists to measure the INT8 router against the checkpoint's BF16 original on real hidden states. Diagnostic only. | unset |
| `TURBOSPARK_EXPERT_RESIDENCY` | MoE runtime (`runtime`) | Set to `mapped` to read routed experts in place from mmap rather than pinned slot cache. Since the `--expert-residency` flag this is the `auto` arm's fallback: honored when the request left the mode at `auto`, ignored under an explicit `streamed`/`mapped` (`runtime::resolve_expert_residency` is the one resolver). | unset |
| `TURBOSPARK_VISION_RESIDENCY` | Vision tower (`runtime`) | Set to `mapped` to read vision tower blocks out of mapped memory rather than pinned slots. | unset |
| `TURBOSPARK_SPEC_STATS` | Speculative decoding (`runtime`) | Set to `1` to log speculative acceptance rate per block position and rollback counts to stderr. | unset |
| `TURBOSPARK_ADAPTIVE_DEPTH` | Speculative decoding (`runtime`) | Tuning for `--speculative adaptive`: `LOW,MID,HIGH,DROP`, the clean rounds needed to climb from depth 2-3, 4 and 5+, and the missed proposals that drop one step. Read once per process. For A/B and replay (`tslab depth-replay`); an unparseable value warns and keeps the default. | `8,6,4,2` |
| `TURBOSPARK_MTP_DRAFT` | MTP drafter (`runtime`) | Draft block depth (positive integer) or policy (`0` to disable, unset for `auto`). | unset (`auto`) |
| `TURBOSPARK_MTP_DUMP` | MTP drafter (`runtime`) | Directory path to dump MTP intermediate hidden states for validation scripts. | unset |
| `TURBOSPARK_DFLASH_DRAFT` | DFlash2 drafter (`runtime`) | DFlash2 draft block depth (positive integer) or policy (`0` to disable, unset for `auto`). | unset (`auto`) |
| `TURBOSPARK_ROUTER_HIST` | MoE runtime (`runtime`) | File path to dump per-layer expert selection frequency histogram (JSON) at exit. | unset |
| `TURBOSPARK_ROUTER_TRACE` | MoE runtime (`runtime`) | Set to collect sequential top-k expert selection trace into histogram output. | unset |
| `TURBOSPARK_ROUTER_BATCHES` | MoE runtime (`runtime`) | Set to `1` to add per-batch rows (reset epoch, phase, layer, token range, ordered routes, streaming-cache counters) to the `TURBOSPARK_ROUTER_HIST` output. Read with `scripts/moe_route_replay.py`. | unset |
| `TURBOSPARK_ROUTER_BATCH_LIMIT` | MoE runtime (`runtime`) | Maximum stored selections for `TURBOSPARK_ROUTER_BATCHES`. Past it the output is marked `truncated`. | `1000000` |
| `TURBOSPARK_PILOT_PROBE` | MoE runtime (`runtime`) | Set to `1` to probe one-layer-ahead router prediction; set to `self` to validate probe self-recall. | unset |
| `TURBOSPARK_FFN_HIST` | Dense FFN (`runtime`) | File path to dump dense FFN neuron activation frequency histogram (JSON) at exit (museGlimmer). | unset |
| `TURBOSPARK_VISION_OVERFLOW` | Vision tower (`runtime`) | File path to dump vision attention overflow and peak activation tensors. | unset |
| `TURBOSPARK_RESID_CAPTURE` | Forward pass (`runtime`) | File path to dump residual stream activations (JSON) at prefill-to-decode transition. | unset |
| `TURBOSPARK_GDN_LAZY_TAPE` | Speculative verify and rollback (`runtime`) | Set to `0` to disable lazy tape optimization during speculative verification and revert to eager checkpoint copy. Default `1` records rank-1 conv and delta updates during verify and folds only accepted updates back into recurrent state at round end. See [SPEEDUP_PLAYBOOK.md](SPEEDUP_PLAYBOOK.md). | `1` (enabled) |
| `TURBOSPARK_GDN_DK128` | GDN delta kernels (`gpu`) | Set to `0` to keep every gated-DeltaNet delta kernel (decode, eager and lazy prefill, state fold) on the runtime key-dim loop instead of the build with `Dk = 128` baked in. Default on; the baked build is bit-identical and only applies when the key head dim is 128. Read once per process, so an A/B needs a fresh process per arm. Prints no engagement line. See [KAPITSA_KERNELS.md](KAPITSA_KERNELS.md) and [KERNEL_MODES.md](KERNEL_MODES.md). | `1` (enabled) |
| `TURBOSPARK_QWEN_FUSED_FFN` | Qwen dense / MoE (`runtime`) | Set to `0` to disable fused INT4 FFN dispatches (gate+up+act in one kernel pass) in Qwen dense tails and shared experts. Default `1` (enabled). | `1` (enabled) |
| `TURBOSPARK_QWEN_BATCHED_ROW_OPS` | Qwen batched prefill / verify (`runtime`) | Set to `0` to restore the per-row form of small ops (RMSNorm, RoPE, gate/act, residual adds). Default `1` encodes them into a single multi-row dispatch. | `1` (enabled) |
| `TURBOSPARK_PREFILL_STEEL` | Dense Qwen prefill (`runtime`) | Set to `0` to opt out of the Steel matrix-multiplication prefill path on INT4 dense models and fall back to the scalar kernel. Default `1` (on for INT4). | `1` (on for INT4) |
| `TURBOSPARK_PREFILL_ROWS_ATTN` | Dense Qwen prefill (`runtime`) | Set to `0` to disable rows-batched prefill attention and restore per-token attention. Default `1` (enabled). | `1` (enabled) |
| `TURBOSPARK_MTP_BATCHED_PRIME` | Qwen MTP priming (`runtime`) | Set to `0` to prime MTP drafter states per row rather than in a batched multi-row dispatch. Default `1` (enabled). | `1` (enabled) |
| `TURBOSPARK_ATTN_DECODE_KERNEL` | Attention decode (`gpu`) | Attention decode kernel selection override (`gqa` for split-KV GQA decode, `barrier` for single-threadgroup barrier decode, or unset for automatic selection based on head dimension and GQA ratio). | unset (auto) |
| `TURBOSPARK_METAL_PIPELINE_CACHE` | Metal pipeline compilation (`gpu`) | Set to `1` to enable persistent Metal binary archive pipeline caching to disk. | unset (off) |
| `TURBOSPARK_METAL_CACHE_DIR` | Metal pipeline compilation (`gpu`) | Directory for persistent Metal pipeline cache files. | unset (`~/.turbospark/cache`) |
| `TURBOSPARK_METAL_PRECOMPILED` | Metal pipeline loading (`gpu`) | Set to `1` to load precompiled metallib binary shaders instead of compiling shader source at runtime. | unset (off) |
| `TURBOSPARK_METAL_PRECISE_MATH` | Metal pipeline compilation (`gpu`) | Set to `1` to compile Metal shaders with precise floating-point math (`-fno-fast-math`). | unset (off) |
| `TURBOSPARK_ROUTER_LOGITS_LAYER` | MoE diagnostics (`runtime`) | Zero-based layer index specifying which layer to record router logits from when `TURBOSPARK_ROUTER_HIST` is set. | unset |
| `TURBOSPARK_ROUTER_LOGITS_PASS` | MoE diagnostics (`runtime`) | Zero-based pass index specifying which forward pass to record router logits from when `TURBOSPARK_ROUTER_HIST` is set. | unset |
| `TURBOSPARK_ROUTER_EVENT` | MoE dispatch (`runtime`) | Set to `0` to disable Metal shared event synchronization between router GEMV and expert dispatch, falling back to command-buffer boundaries. | `1` (enabled) |
| `TURBOSPARK_MOONSHINE_DEVICE` | Moonshine speech path (`runtime`) | Backend device for Moonshine ASR model execution: `cpu` or `metal`. | unset (`cpu`) |
| `TURBOSPARK_WHISPER_DEVICE` | Whisper speech path (`runtime`) | Set to `cpu` to force the CPU reference execution path. On any other value (or unset) the runner builds the Metal engine when a device initializes and falls back to CPU on any engine error. See [WHISPER](WHISPER.md). | unset (Metal when available) |
| `TURBOSPARK_WHISPER_PROFILE` | Whisper speech path (`runtime`) | Set to `1` to print per-window phase timings (mel, encode, cross caches, decode) and per-step pass wall/GPU-busy lines to stderr. With `TURBOSPARK_DISPATCH_PROFILE=1` also prints the per-kernel ranking at the end of the run. Attribution only, never a benchmark row. | unset |
| `TS_STT_TEST_MODEL` | Whisper tests (`runtime`, `ffi`) | Names an installed speech model directory for the real-model, cross-device, and FFI surface tests. Unset skips those tests. | unset |

---

## 5. Model Catalog and Hardware Detection

| Variable | Affected Binaries | Purpose | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_TEST_CHIP` | `turbospark-model recommend` | Override probed Apple Silicon chip model string (e.g. `Apple M3 Max`) for recommendation tests. | probed hardware chip |
| `TURBOSPARK_PROBE_SLOTS` | `turbospark-model probe` | Override expert cache slot count used in probe arithmetic. | calculated / default |

---

## 6. Integration Test Oracles and Benchmark Installations

The integration tests and benchmark oracle suites (`turbospark-bench`, `turbospark-repack`, `turbospark-server`) and the runtime's opt-in audio model tests use dedicated variables pointing to installed `.gturbo` model directories or fixture artifacts:

### Model Installation Directories
- `TURBOSPARK_GEMMA4_INSTALL_DIR`: Local Gemma 4 install directory.
- `TURBOSPARK_GEMMA4_GGUF_INSTALL_DIR`: Local Gemma 4 GGUF repack directory.
- `TURBOSPARK_GEMMA4_IQ_INSTALL_DIR`: Local Gemma 4 IQ-quantized install directory.
- `TURBOSPARK_QWEN2_INSTALL_DIR`: Local Qwen2/Qwen2.5 install directory (repack streaming test).
- `TURBOSPARK_QWEN2_DENSE_INSTALL_DIR`: Local Qwen2/Qwen2.5 install directory (bench oracles and logit dump).
- `TURBOSPARK_QWEN2_GGUF_Q3KM_INSTALL_DIR`: Local Qwen2.5 GGUF Q3_K_M install directory.
- `TURBOSPARK_QWEN2_GGUF_Q4KM_INSTALL_DIR`: Local Qwen2.5 GGUF Q4_K_M install directory.
- `TURBOSPARK_QWEN3_DENSE_INSTALL_DIR`: Local dense Qwen3 (0.6B regression) install directory.
- `TURBOSPARK_QWEN35_INSTALL_DIR`: Local Qwen 3.5 install directory.
- `TURBOSPARK_QWEN36_INSTALL_DIR`: Local Qwen 3.6 install directory.
- `TURBOSPARK_QWEN36_GGUF_INSTALL_DIR`: Local Qwen 3.6 GGUF repack directory.
- `TURBOSPARK_QWEN38_INSTALL_DIR`: Local Qwen 3.8 install directory.
- `TURBOSPARK_QWEN38_MTP_INSTALL_DIR`: Local Qwen 3.8 MTP install directory.
- `TURBOSPARK_QWEN38_DFLASH2_INSTALL_DIR`: Local Qwen 3.8 DFlash2 install directory.
- `TURBOSPARK_QWEN38_VISION_INSTALL_DIR`: Local Qwen 3.8 Vision install directory.
- `TURBOSPARK_QWEN38_TRUNK_INSTALL_DIR`: Local text-trunk install directory paired with a standalone vision tower sidecar.
- `TURBOSPARK_QWEN3MOE_INSTALL_DIR`: Local Qwen MoE install directory.
- `TURBOSPARK_QWEN3VL_INSTALL_DIR`: Local Qwen3-VL text-trunk install directory.
- `TURBOSPARK_QWEN3VL_VISION_INSTALL_DIR`: Local Qwen3-VL combined deepstack vision install directory.
- `TURBOSPARK_QWEN4EXP_INSTALL_DIR`: Local Qwen3.8-Flash-Next REAP-288 install directory.
- `TURBOSPARK_MISTRAL_INSTALL_DIR`: Local Mistral install directory.
- `TURBOSPARK_MIXTRAL_INSTALL_DIR`: Local Mixtral install directory.
- `TURBOSPARK_DENSE_LLAMA_INSTALL_DIR`: Local Dense Llama install directory.
- `TURBOSPARK_GPTOSS_INSTALL_DIR`: Local GPT-OSS install directory.
- `TURBOSPARK_MUSEGLIMMER_INSTALL_DIR`: Local MuseGlimmer install directory.
- `TURBOSPARK_DSV2_INSTALL_DIR`: Local DeepSeek-V2-Lite install directory.
- `TURBOSPARK_MINIMAX_INSTALL_DIR`: Local MiniMax-M2 install directory.
- `TURBOSPARK_SPARK_INSTALL_DIR`: Local Spark-X2.5-4B install directory.
- `TURBOSPARK_BONSAI2_INSTALL_DIR`: Local Bonsai-2 (Hadamard-folded ternary) install directory.
- `TURBOSPARK_TERNARY_INSTALL_DIR`: Local ternary quantized model install directory.
- `TURBOSPARK_MTP_INSTALL_DIR`: General MTP test install directory.
- `TURBOSPARK_DFLASH2_INSTALL_DIR`: General DFlash2 test install directory.
- `TURBOSPARK_ORNITH35B_INSTALL_DIR`: Local Ornith 35B install directory.
- `TURBOSPARK_ORNITH35B_GGUF_INSTALL_DIR`: Local Ornith 35B GGUF install directory.
- `TURBOSPARK_ORNITH9B_INSTALL_DIR`: Local Ornith 9B install directory.
- `TURBOSPARK_KV_QUANT_INSTALL_DIR`: Local install directory for the KV quantization probe.

### Logit Dump, Parity, and Vision Fixtures
- `TURBOSPARK_LOGIT_DUMP_DIR`: Directory containing baseline logit dumps for cross-engine KL divergence verification.
- `TURBOSPARK_LOGIT_DUMP_COLD`: Set to evaluate cold start logit consistency.
- `TURBOSPARK_IMAGE_INSTALL_DIR`: Local image-generation install directory (Z-Image fixtures).
- `TURBOSPARK_ZIMAGE_MLX_INSTALL_DIR`: Local MLX Z-Image-Turbo install directory (parity fixtures).
- `TURBOSPARK_VISION_PAGE`: Path to single image file for vision backend test runs.
- `TURBOSPARK_VISION_DUMP_DIR`: Output directory for dumped vision embeddings.
- `TURBOSPARK_VISION_KLD_DIR`: Directory containing reference vision logits for KL divergence tests.
- `TURBOSPARK_VISION_ORACLE_PAGES_DIR`: Directory of test images for vision oracle suites.
- `TURBOSPARK_VISION_REPLAY_ROWS`: Number of rows to replay during vision tests.
- `TURBOSPARK_VISION_TOKENIZER_DIR`: Path to external vision tokenizer directory if separated from model weights.

### Steering and Directional Vectors
- `TURBOSPARK_PROBE_INSTALL_DIR`: Model directory used for steering probe runs.
- `TURBOSPARK_STEERING_VECTOR`: Path to raw `.f32` steering direction vector.
- `TURBOSPARK_STEERING_ALPHAS`: Comma-separated list of alpha scaling multipliers for steering sweep.
- `TURBOSPARK_STEERING_BANDS`: Layer band specification for steering application (e.g. `10-20`).
- `TURBOSPARK_STEERING_MODE`: Directional steering mode (`add`, `project`, etc.).
- `TURBOSPARK_STEERING_SCALE`: Global scaling multiplier for steering interventions.
- `TURBOSPARK_STEERING_PROMPT`: Text prompt used during steering evaluations.
- `TURBOSPARK_CONTROL_VECTOR`: Path to primary control vector file.
- `TURBOSPARK_FOREIGN_CONTROL_VECTOR`: Path to control vector from a foreign architecture to test refusal.

### Audio and Embedding Fixtures
Variables for the runtime's opt-in (`#[ignore]`d) audio and embedding tests; every test skips when its variable is unset.

- `TURBOSPARK_TEST_OMNIVOICE`: Local OmniVoice install directory (`mlx-community/OmniVoice-8bit`). Gates the OmniVoice encode-parity test and the reference-transcript A/B.
- `TURBOSPARK_TEST_OMNIVOICE_REF`: Directory holding `reference_24k.npy` and `reference_codes.npy` from the tools run against the Python reference tokenizer, for the cross-engine encode parity test.
- `TURBOSPARK_TEST_OMNIVOICE_AB_OUT`: Optional output directory for the reference-transcript A/B's 24 kHz clips.
- `TURBOSPARK_TEST_LIBRISPEECH_WAV`: 16 kHz mono WAVs of LibriSpeech test-clean named `<utterance id>.wav` (default `~/models/librispeech/LibriSpeech/wav16k`), the reference pool for the A/B.
- `TURBOSPARK_PARAKEET_REDUX_MODEL`: Local Parakeet Redux install directory (default `~/models/parakeet-redux`). The CPU transcription oracle for the A/B, and the install the Core ML Parakeet tests compare against.
- `TURBOSPARK_FISH_DRAMA3_INSTALL_DIR`: Local Fish-Drama3 audio model directory.
- `TURBOSPARK_HUMANIZER_INSTALL_DIR`: Local Humanizer model directory.
- `TURBOSPARK_GLMASR_DIR`: Pinned GLM-ASR model directory for speech tests.
- `TURBOSPARK_QWEN3_ASR_DIR`: Pinned Qwen3-ASR model directory.
- `TURBOSPARK_QWEN3_ASR_WAV`: Reference WAV file for Qwen3-ASR tests.
- `TURBOSPARK_EMBEDDINGGEMMA2_COREML_MODEL`: Local Core ML EmbeddingGemma2 model directory.
- `TURBOSPARK_UNLIMITED_OCR_COREML_MODEL`: Local Core ML Unlimited OCR model directory.
- `TURBOSPARK_AUDIO_REAL_INSTALL`: Set to `1` with explicit `TURBOSPARK_HOME` to run live audio download and verification tests.

### Repack and Upstream Comparison
- `TURBOSPARK_LLAMACPP_DIR`: Path to local `llama.cpp` build directory for cross-engine parity tests.
- `TURBOSPARK_LLAMACPP_NGL`: Number of GPU layers offloaded in `llama.cpp` comparisons.
- `TURBOSPARK_LLAMA_ROPE_PATCH`: Set to test Llama RoPE frequency patching.
- `TURBOSPARK_QWEN_PATCH`: Set to test Qwen weight repacking adjustments.
