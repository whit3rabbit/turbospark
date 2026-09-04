# Environment Variables in TurboSpark

This document lists every environment variable recognized by TurboSpark components:
the CLI tools (`turbospark-check`, `turbospark-model`, `turbospark-bench`), the
HTTP server (`turbospark-server`), the macOS desktop application (`TurboSparkApp`),
the Swift bindings (`TurboSpark`), and the underlying Rust runtime crates.

All TurboSpark-specific environment variables use the `TURBOSPARK_*` prefix.

Keep all docs and configuration ASCII (no emojis, no em dashes).

---

## 1. Core and Model Storage

| Variable | Affected Components | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_HOME` | CLI, server, Swift app, catalog | Base directory for downloaded models, catalog entries, and repack staging. Models are stored in `$TURBOSPARK_HOME/models`. | `~/.turbospark` |
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
| `TURBOSPARK_PREFIX_REUSE` | `turbospark-check --chat` | Set to `quiet` to silence the `[prefix-reuse] N/M` cache reuse line on stderr. | unset |
| `TURBOSPARK_PREFILL_CHUNK` | `turbospark-check`, `turbospark-server` | Override prompt-processing chunk size (e.g. `128`, `256`, `512`). Takes precedence over `--prefill-chunk`. | unset |
| `TURBOSPARK_CHAT_DATE` | All (chat template rendering) | Override current date/time (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`) in templates calling `strftime_now`. Used to freeze prompts in deterministic benchmarks. | system UTC time |
| `TURBOSPARK_READ_QOS` | Streaming I/O pool | Set to `utility` to drop background routed-expert streaming read QoS on macOS from user-initiated to utility (E-cores). | unset |
| `TURBOSPARK_EXPERT_DISK_IO` | Streaming I/O | Set to `1` to measure physical disk read syscall bytes and amplification vs requested expert bytes. | unset |
| `TURBOSPARK_EXPERT_NOCACHE` | Streaming I/O | Set to `1` to bypass OS page cache on expert chunk reads (`F_NOCACHE`). | unset |
| `TURBOSPARK_SHARED_CB` | Forward pass (`runtime`) | Set to `0` to disable overlapping shared expert command buffer with host expert pread. | `1` (enabled) |
| `TURBOSPARK_ROUTED_PIPELINE` | MoE dispatch (`runtime`) | Set to `0` to disable one-layer-pipelined routed command buffer execution. | `1` (enabled) |
| `TURBOSPARK_ROUTED_BATCH` | MoE prefill (`runtime`) | Set to `1` to enable experimental routed batch prefill dispatch. | `0` (off) |
| `TURBOSPARK_BATCHED_GEMV` | MoE prefill (`runtime`) | Set to `1` to enable experimental batched resident GEMVs as M-row GEMMs. | `0` (off) |
| `TURBOSPARK_EXPERT_RESIDENCY` | MoE runtime (`runtime`) | Set to `mapped` to read routed experts in place from mmap rather than pinned slot cache. | unset |
| `TURBOSPARK_VISION_RESIDENCY` | Vision tower (`runtime`) | Set to `mapped` to read vision tower blocks out of mapped memory rather than pinned slots. | unset |
| `TURBOSPARK_SPEC_STATS` | Speculative decoding (`runtime`) | Set to `1` to log speculative acceptance rate per block position and rollback counts to stderr. | unset |
| `TURBOSPARK_MTP_DRAFT` | MTP drafter (`runtime`) | Draft block depth (positive integer) or policy (`0` to disable, unset for `auto`). | unset (`auto`) |
| `TURBOSPARK_MTP_DUMP` | MTP drafter (`runtime`) | Directory path to dump MTP intermediate hidden states for validation scripts. | unset |
| `TURBOSPARK_DFLASH_DRAFT` | DFlash2 drafter (`runtime`) | DFlash2 draft block depth (positive integer) or policy (`0` to disable, unset for `auto`). | unset (`auto`) |
| `TURBOSPARK_ROUTER_HIST` | MoE runtime (`runtime`) | File path to dump per-layer expert selection frequency histogram (JSON) at exit. | unset |
| `TURBOSPARK_ROUTER_TRACE` | MoE runtime (`runtime`) | Set to collect sequential top-k expert selection trace into histogram output. | unset |
| `TURBOSPARK_PILOT_PROBE` | MoE runtime (`runtime`) | Set to `1` to probe one-layer-ahead router prediction; set to `self` to validate probe self-recall. | unset |
| `TURBOSPARK_FFN_HIST` | Dense FFN (`runtime`) | File path to dump dense FFN neuron activation frequency histogram (JSON) at exit (museGlimmer). | unset |
| `TURBOSPARK_VISION_OVERFLOW` | Vision tower (`runtime`) | File path to dump vision attention overflow and peak activation tensors. | unset |
| `TURBOSPARK_RESID_CAPTURE` | Forward pass (`runtime`) | File path to dump residual stream activations (JSON) at prefill-to-decode transition. | unset |

---

## 5. Model Catalog and Hardware Detection

| Variable | Affected Binaries | Purpose | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_TEST_CHIP` | `turbospark-model recommend` | Override probed Apple Silicon chip model string (e.g. `Apple M3 Max`) for recommendation tests. | probed hardware chip |
| `TURBOSPARK_PROBE_SLOTS` | `turbospark-model probe` | Override expert cache slot count used in probe arithmetic. | calculated / default |

---

## 6. Integration Test Oracles and Benchmark Installations

The integration tests and benchmark oracle suites (`turbospark-bench`, `turbospark-repack`, `turbospark-server`) use dedicated variables pointing to installed `.gturbo` model directories or fixture artifacts:

### Model Installation Directories
- `TURBOSPARK_GEMMA4_INSTALL_DIR`: Local Gemma 4 install directory.
- `TURBOSPARK_GEMMA4_GGUF_INSTALL_DIR`: Local Gemma 4 GGUF repack directory.
- `TURBOSPARK_GEMMA4_IQ_INSTALL_DIR`: Local Gemma 4 IQ-quantized install directory.
- `TURBOSPARK_QWEN35_INSTALL_DIR`: Local Qwen 3.5 install directory.
- `TURBOSPARK_QWEN36_INSTALL_DIR`: Local Qwen 3.6 install directory.
- `TURBOSPARK_QWEN36_GGUF_INSTALL_DIR`: Local Qwen 3.6 GGUF repack directory.
- `TURBOSPARK_QWEN38_INSTALL_DIR`: Local Qwen 3.8 install directory.
- `TURBOSPARK_QWEN38_MTP_INSTALL_DIR`: Local Qwen 3.8 MTP install directory.
- `TURBOSPARK_QWEN38_DFLASH2_INSTALL_DIR`: Local Qwen 3.8 DFlash2 install directory.
- `TURBOSPARK_QWEN38_VISION_INSTALL_DIR`: Local Qwen 3.8 Vision install directory.
- `TURBOSPARK_QWEN3MOE_INSTALL_DIR`: Local Qwen MoE install directory.
- `TURBOSPARK_MISTRAL_INSTALL_DIR`: Local Mistral install directory.
- `TURBOSPARK_MIXTRAL_INSTALL_DIR`: Local Mixtral install directory.
- `TURBOSPARK_DENSE_LLAMA_INSTALL_DIR`: Local Dense Llama install directory.
- `TURBOSPARK_GPTOSS_INSTALL_DIR`: Local GPT-OSS install directory.
- `TURBOSPARK_MUSEGLIMMER_INSTALL_DIR`: Local MuseGlimmer install directory.
- `TURBOSPARK_MTP_INSTALL_DIR`: General MTP test install directory.
- `TURBOSPARK_DFLASH2_INSTALL_DIR`: General DFlash2 test install directory.
- `TURBOSPARK_ORNITH35B_INSTALL_DIR`: Local Ornith 35B install directory.
- `TURBOSPARK_ORNITH35B_GGUF_INSTALL_DIR`: Local Ornith 35B GGUF install directory.
- `TURBOSPARK_ORNITH9B_INSTALL_DIR`: Local Ornith 9B install directory.
- `TURBOSPARK_TERNARY_INSTALL_DIR`: Local ternary quantized model install directory.
- `TURBOSPARK_IQ3_INSTALL_DIR`: Local IQ3 quantized model install directory.

### Logit Dump, Parity, and Vision Fixtures
- `TURBOSPARK_LOGIT_DUMP_DIR`: Directory containing baseline logit dumps for cross-engine KL divergence verification.
- `TURBOSPARK_LOGIT_DUMP_COLD`: Set to evaluate cold start logit consistency.
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

### Repack and Upstream Comparison
- `TURBOSPARK_LLAMACPP_DIR`: Path to local `llama.cpp` build directory for cross-engine parity tests.
- `TURBOSPARK_LLAMACPP_NGL`: Number of GPU layers offloaded in `llama.cpp` comparisons.
- `TURBOSPARK_LLAMA_ROPE_PATCH`: Set to test Llama RoPE frequency patching.
- `TURBOSPARK_QWEN_PATCH`: Set to test Qwen weight repacking adjustments.
