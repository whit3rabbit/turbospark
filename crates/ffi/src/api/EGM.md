# ffi api

## Overview
The `extern "C"` entry points of the C ABI, grouped by domain and re-exported from [mod.rs]. Each function returns a status code and leaves its message through the thread-local error slot.

## Interface
- Core: [core.rs:ts_last_error], [core.rs:ts_string_free], [core.rs:ts_abi_version], [core.rs:ts_build_info_json], [core.rs:ts_system_info_json], [core.rs:ts_peak_footprint_bytes].
- Session and generation: [session.rs:ts_session_open], [session.rs:ts_session_close], [session.rs:ts_session_cancel], [session.rs:ts_session_info_json]; [generate.rs:ts_generate], [generate.rs:ts_humanize], [generate.rs:ts_session_count_tokens], [generate.rs:ts_session_render_prompt], [generate.rs:ts_session_fit_window_json].
- Tokenizer: [tokenizer.rs:ts_tokenizer_open], [tokenizer.rs:ts_tokenizer_tokenize_json], [tokenizer.rs:ts_tokenizer_detokenize_json].
- Models and store: [models.rs:ts_catalog_json], [models.rs:ts_installed_json], [models.rs:ts_install], [models.rs:ts_probe_json], [models.rs:ts_hub_search_json], [models.rs:ts_recommend_json], [models.rs:ts_store_root_set]; libraries: [libraries.rs:ts_libraries_json], [libraries.rs:ts_library_add], [libraries.rs:ts_model_move].
- Server: [server.rs:ts_server_start], [server.rs:ts_server_stop], [server.rs:ts_server_attach_session], [server.rs:ts_server_poll_events_json]; daemon: [daemon.rs:ts_daemon_start], [daemon.rs:ts_daemon_stop], [daemon.rs:ts_daemon_status_json].
- Image: [image.rs:ts_image_session_open], [image.rs:ts_image_generate]. OCR: [ocr.rs:ts_ocr_open], [ocr.rs:ts_ocr_read_json]. Embedding: [embedding.rs:ts_embedding_encode_json], [embedding.rs:ts_cosine_similarity].
- Speech: [stt.rs:ts_stt_open], [stt.rs:ts_stt_stream_open]; [stt_unsupported.rs] keeps the same ABI off macOS; [audio.rs:ts_audio_session_open], [audio.rs:ts_audio_catalog_json], [audio.rs:ts_audio_install]; [audio_stream.rs:ts_audio_stream_open].

## Invariants
- Every entry point except `ts_last_error` and `ts_peak_footprint_bytes` (deliberately unguarded, see their comments) must run inside the ABI guard so no panic unwinds across `extern "C"` [../abi.rs:guard], [../abi.rs:guard_value], [../abi.rs:guard_result].
- `stt` is selected by cfg: real on macOS, [stt_unsupported.rs] elsewhere, with the same exported names [mod.rs].
- Strings returned through `char **` must be freed with `ts_string_free` [core.rs:ts_string_free].

## Side Effects
- Opens model sessions (Metal on macOS), spawns generation and server work, starts or stops the daemon [session.rs], [server.rs], [daemon.rs]. The daemon child gets `TURBOSPARK_API_KEY` and the process-local store root as `TURBOSPARK_HOME` so it resolves the same store as the app [daemon.rs:apply_child_env].
- Network and disk through installs and Hugging Face calls [models.rs], [audio.rs].
- Hugging Face token entry points. `ts_hf_token_set` is the only writer and is a deliberate host action: it writes plaintext file `hf_token` (mode 0600, temp file plus rename) in the model store root, not the Keychain, so nothing is stored unless the host calls it [models.rs:ts_hf_token_set]. `ts_hf_token_clear` removes only that file; a token from `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or the huggingface-cli cache still resolves afterward [models.rs:ts_hf_token_clear]. `ts_hf_token_get` and `ts_hf_token_info_json` return the RAW resolved token (env, store, or CLI cache; the JSON adds `source`) and NULL when none, so the host must not log either [models.rs:ts_hf_token_get], [models.rs:ts_hf_token_info_json].
- Other public surface not listed above: idle unload and image/embedding/audio attach, detach and poll on the server [server.rs:ts_server_set_idle_unload], [server.rs:ts_server_attach_image_model], [server.rs:ts_server_attach_embedding_model], shared heavy-work permit [audio.rs:ts_heavy_work_try_acquire], [audio.rs:ts_heavy_work_release], install control [models.rs:ts_install_pause], [models.rs:ts_install_resume], HF token validate (`ts_hf_token_validate_json`), HF endpoint get/set (`ts_hf_endpoint_get`, `ts_hf_endpoint_set`), `ts_audio_job_*`, `ts_ocr_add_page_*`.
- Moves or deletes model directories [libraries.rs:ts_model_move], [libraries.rs:ts_model_delete_install].
- Store location: [libraries.rs:ts_store_import_moved_root] (registers a moved store's `models` folder as a library in place and copies its embedding and OCR models into the home).

## Verification
- Run: `cargo test -p turbospark-ffi`
