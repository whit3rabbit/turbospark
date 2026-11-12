# CTurboSpark (C ABI module)

## Overview
This directory is the C boundary between the Swift `TurboSpark` target and the Rust engine (`turbospark-ffi`). It holds a hand-written header, a clang module map that exposes the header to Swift as `import CTurboSpark`, and the staged static archive `libturbospark_ffi.a` (a binary build product, not source). The header and the archive are staged by `make swift-lib` and are gitignored copies, so they can drift apart; the ABI revision constant exists to detect that.

## Interface
- `module CTurboSpark` ([module.modulemap:CTurboSpark]) - clang module that maps `turbospark.h` with `export *`.
- `TS_ABI_VERSION` ([turbospark.h:TS_ABI_VERSION]) - ABI revision the header describes (currently 2); compared with `ts_abi_version()` by `TurboSparkRuntime.verifyABI`.
- Status codes `TS_OK` .. `TS_ERR_BUSY` ([turbospark.h:TS_ERR_INVALID_ARGUMENT]) - 0 to 8; mirrored by `TurboSparkError.Code`.
- Event kinds `TS_EVENT_PREFILL`, `TS_EVENT_CONTENT`, `TS_EVENT_REASONING`, `TS_EVENT_TOOL`, `TS_EVENT_FINISH` ([turbospark.h:TS_EVENT_FINISH]) - streaming generation kinds; an unknown kind must be a no-op.
- `TS_INSTALL_STAGE`, `TS_INSTALL_BYTES`, `TS_IMAGE_EVENT_STAGE`, `TS_IMAGE_EVENT_FINISH` ([turbospark.h:TS_INSTALL_BYTES]) - install and image progress kinds.
- Callback typedefs `TsEventCallback`, `TsImageEventCallback`, `TsInstallCallback`, `TsRecommendCallback`, `TsAudioCallback` ([turbospark.h:TsEventCallback]) - borrowed-text callbacks.
- Opaque handles `TsSession`, `TsServer`, `TsImageSession`, `TsTokenizer`, `TsOcrSession`, `TsSttModel`, `TsSttStream`, `TsAudioSession`, `TsAudioJob`, `TsAudioStream`, `TsAudioOpenToken`, `TsHeavyWorkPermit` ([turbospark.h:TsSession]).
- Errors and strings: `ts_last_error`, `ts_string_free` ([turbospark.h:ts_last_error]).
- Session lifecycle: `ts_session_open`, `ts_session_close`, `ts_session_cancel`, `ts_session_release_vision` ([turbospark.h:ts_session_open]).
- Generation: `ts_generate`, `ts_humanize` ([turbospark.h:ts_generate]).
- Session text tools: `ts_session_count_tokens`, `ts_session_render_prompt`, `ts_session_tokenize_json`, `ts_session_fit_window_json` and `_with_tools` variants ([turbospark.h:ts_session_fit_window_json]).
- Introspection: `ts_session_info_json`, `ts_session_phases_json`, `ts_system_info_json`, `ts_build_info_json`, `ts_abi_version`, `ts_peak_footprint_bytes` ([turbospark.h:ts_session_info_json]).
- Standalone tokenizer: `ts_tokenizer_open` and siblings ([turbospark.h:ts_tokenizer_open]).
- Image generation: `ts_image_session_open`, `ts_image_generate`, `ts_image_buffer_free` ([turbospark.h:ts_image_generate]).
- In-process HTTP server: `ts_server_start`, `ts_server_attach_session`, `ts_server_detach_model`, `ts_server_poll_events_json`, `ts_server_stop`, plus image, speech and audio attach/poll/complete calls ([turbospark.h:ts_server_start]).
- Model store and catalog: `ts_catalog_json`, `ts_installed_json`, `ts_install_with_options`, `ts_install_repo_with_options`, `ts_install_cancel`, `ts_install_pause`, `ts_model_delete_install`, `ts_model_move` ([turbospark.h:ts_install]).
- Model libraries: `ts_libraries_json`, `ts_library_add`, `ts_library_remove`, `ts_library_set_default`, `ts_model_forget`, `ts_store_import_moved_root` ([turbospark.h:ts_libraries_json]).
- Hardware fit: `ts_recommend_json`, `ts_probe_json`, `ts_context_ladder_json`, `ts_repo_variants_json`, `ts_hub_search_json`, `ts_hub_check_json` ([turbospark.h:ts_recommend_json]).
- Capability predicates: `ts_family_capabilities_json`, `ts_kv_quant_supported`, `ts_chip_supports_ane_prefill`, `ts_expert_residency_for_install_json`, `ts_control_vector_info_json` ([turbospark.h:ts_family_capabilities_json]).
- Hugging Face config: `ts_hf_token_*`, `ts_hf_endpoint_get`, `ts_hf_endpoint_set` ([turbospark.h:ts_hf_token_set]).
- Embeddings and OCR: `ts_embedding_*`, `ts_ocr_*`, `ts_cosine_similarity` ([turbospark.h:ts_embedding_encode_json]).
- Speech and audio: `ts_stt_*`, `ts_audio_*`, `ts_heavy_work_try_acquire`, `ts_heavy_work_release` ([turbospark.h:ts_audio_session_open]).
- Daemon: `ts_daemon_status_json`, `ts_daemon_start`, `ts_daemon_stop`, `ts_daemon_restart` ([turbospark.h:ts_daemon_status_json]).
- `libturbospark_ffi.a` - prebuilt Rust static library (binary, not read as source); it is the implementation of everything above.

## Invariants
- Every fallible call returns `TS_OK` or a non-zero code, and `ts_last_error()` on the SAME thread holds the message; read it before the next call on that thread ([turbospark.h:ts_last_error]).
- A `const char *` argument is borrowed for the call only; a `char **` out-parameter is an allocation the caller must release with `ts_string_free`; nothing hands back a pointer into library state ([turbospark.h:ts_string_free]).
- Options and results are JSON strings with camelCase keys, except the native audio block, the catalog and installed rows, and the OCR/embedding results, which are snake_case ([turbospark.h:ts_audio_catalog_json], [turbospark.h:ts_installed_json]).
- A session is single-threaded: one generation at a time, `ts_generate` blocks, and only `ts_session_cancel` is safe from any thread ([turbospark.h:ts_session_cancel]).
- The token callback runs while `ts_generate` holds the engine lock, so calling back into the same session (other than cancel) deadlocks ([turbospark.h:ts_generate]).
- `ts_install` byte callbacks arrive concurrently from worker threads, possibly out of order ([turbospark.h:ts_install]).
- `ts_session_close` must not be called while a generation is in flight on another thread ([turbospark.h:ts_session_close]).
- Callback `text` is not NUL-terminated and valid only during the call ([turbospark.h:TsEventCallback]).
- A host must treat any unknown event kind as a no-op ([turbospark.h:TS_EVENT_FINISH]).
- Library refusals put a stable lowercase code first in the error text as `<code>: <sentence>`; library changes are refused with `TS_ERR_BUSY` while an install or move runs ([turbospark.h:ts_model_move]).
- `ts_image_buffer_free` takes exactly the `out_png`/`out_png_len` pair from one call, once ([turbospark.h:ts_image_buffer_free]).
- Do not hold `TsHeavyWorkPermit` around a Rust generation call: the permit is non-reentrant ([turbospark.h:ts_heavy_work_try_acquire]).
- `TS_ABI_VERSION` is bumped when a wire shape, option key or function meaning changes; a purely additive export does not need a bump ([turbospark.h:TS_ABI_VERSION]).

## Side Effects
- Model store and registry: install, delete, move, forget and library calls read and write `~/.turbospark` (`installed.json`, library markers, model folders) ([turbospark.h:ts_model_delete_install], [turbospark.h:ts_library_add]).
- Network: installs, `ts_probe_json`, `ts_hub_search_json`, `ts_hub_check_json`, `ts_hf_token_validate_json` call Hugging Face; hub search results are cached on disk for 15 minutes ([turbospark.h:ts_hub_search_json]).
- Credentials: `ts_hf_token_set` and `ts_hf_token_clear` write `~/.turbospark/hf_token` ([turbospark.h:ts_hf_token_set]).
- Sockets and processes: `ts_server_start` binds a TCP socket and spawns a background thread; `ts_daemon_start` and `ts_daemon_restart` manage a background server process ([turbospark.h:ts_server_start], [turbospark.h:ts_daemon_start]).
- GPU and memory: `ts_session_open` maps model weights and compiles Metal pipelines; audio open hashes model files (SHA-256) before loading ([turbospark.h:ts_session_open], [turbospark.h:ts_audio_session_open_cancellable]).
- Library bookkeeping warnings are written to stderr by the engine ([turbospark.h:ts_libraries_json]).

## Verification
- Run: `swift test --package-path swift/TurboSpark`
