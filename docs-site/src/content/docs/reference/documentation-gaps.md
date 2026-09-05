---
title: Documentation gaps
description: Public API symbols that lack a doc comment.
---
<!-- generated: doc-coverage, 238 undocumented public symbol(s) -->

These public API symbols were found WITHOUT a doc comment during reference generation. Documenting them (in the source) closes the gap.

## `crates/bench/src/model_mode.rs`

- `run_model_mode` (fn)

## `crates/catalog/src/probe/types.rs`

- `is_runnable` (fn)

## `crates/cli/src/chat.rs`

- `run` (fn)

## `crates/cli/src/generate/mod.rs`

- `try_generate` (fn)

## `crates/compute/src/tolerance.rs`

- `QUANT_INT8` (const)

## `crates/ffi/include/turbospark.h`

- `TS_OK` (macro)
- `TsSession` (type)
- `TsServer` (type)

## `crates/ffi/src/models.rs`

- `TS_INSTALL_BYTES` (const)

## `crates/gpu/src/attention_decode.rs`

- `new` (fn)

## `crates/gpu/src/context/device.rs`

- `new` (fn)

## `crates/gpu/src/moe_decode.rs`

- `new` (fn)
- `buffer` (fn)

## `crates/gpu/src/prefill_scratch.rs`

- `PrefillChunkScratchLayout` (struct)
- `allocate` (fn)

## `crates/gpu/src/resident_metal.rs`

- `ResidentGpuWeights` (struct)

## `crates/gpu/src/rope.rs`

- `encode_rope_proportional_neox` (fn)

## `crates/model-io/src/arch_baselines/mod.rs`

- `all_known_architectures` (fn)

## `crates/model-io/src/error.rs`

- `ModelError` (enum)

## `crates/model-io/src/manifest/mod.rs`

- `load` (fn)
- `validate` (fn)

## `crates/model-io/src/resident_buffer.rs`

- `map` (fn)

## `crates/model-io/src/sha256.rs`

- `hash_data` (fn)

## `crates/repack/src/gemma4_checkpoint/config.rs`

- `Gemma4Error` (enum)
- `bits_for` (fn)

## `crates/repack/src/gguf_checkpoint/plan.rs`

- `Plan` (struct)
- `classify` (fn)

## `crates/repack/src/gguf_checkpoint/types.rs`

- `GgufRepackError` (enum)
- `read_tensor` (fn)

## `crates/repack/src/gguf_config/masks.rs`

- `qwen_gdn_moe_layer_mask` (fn)

## `crates/repack/src/gguf_config/mod.rs`

- `GgufConfigError` (enum)

## `crates/repack/src/gguf_header/types.rs`

- `GgufHeaderError` (enum)

## `crates/repack/src/gguf_names/mod.rs`

- `GgufNameError` (enum)
- `GgufMapping` (enum)

## `crates/repack/src/gturbo_writer/types.rs`

- `WriterError` (enum)

## `crates/repack/src/hf_checkpoint.rs`

- `OrchestrateError` (enum)

## `crates/repack/src/lib.rs`

- `control_vector` (mod)

## `crates/repack/src/ranged_download/http.rs`

- `new` (fn)
- `with_progress` (fn)

## `crates/repack/src/ranged_download/mod.rs`

- `DownloadError` (enum)
- `new` (fn)

## `crates/repack/src/repack.rs`

- `RepackError` (enum)

## `crates/repack/src/synthetic_qwen/qwen4_decode.rs`

- `HIDDEN` (const)
- `HC_COUNT` (const)
- `HC_LOWRANK` (const)
- `NUM_HEADS` (const)
- `HEAD_DIM` (const)
- `NUM_KV_HEADS` (const)
- `NUM_EXPERTS` (const)
- `IDX_KV_HEADS` (const)
- `IDX_HEAD_DIM` (const)
- `IDX_COMPRESS` (const)
- `TOP_K` (const)
- `PLE_LAYER` (const)
- `NGRAM_EOS_TOKEN_ID` (const)

## `crates/runtime/src/config.rs`

- `GenerationConfig` (struct)

## `crates/runtime/src/error.rs`

- `RuntimeError` (enum)

## `crates/runtime/src/families/qwen/mod.rs`

- `off` (fn)
- `gdn_state_abs_max` (fn)

## `crates/runtime/src/families/qwen/mtp.rs`

- `mtp_draft_depth` (fn)

## `crates/runtime/src/lib.rs`

- `steering` (mod)
- `vision` (mod)

## `crates/runtime/src/producer.rs`

- `new` (fn)

## `crates/runtime/src/real_forward.rs`

- `RealForwardRunner` (struct)

## `crates/runtime/src/real_forward_types.rs`

- `RealForwardError` (enum)

## `crates/server/src/bind.rs`

- `host` (fn)

## `crates/server/src/handler/plan.rs`

- `AppState` (type)

## `crates/server/src/lib.rs`

- `observe` (mod)
- `registry` (mod)
- `vision` (mod)
- `new` (fn)

## `crates/server/src/model.rs`

- `ChatModel` (trait)
- `new` (fn)
- `with_default_reasoning` (fn)
- `with_default_system` (fn)

## `crates/server/src/observe.rs`

- `next` (fn)

## `crates/server/src/real_model.rs`

- `RealChatModel` (struct)

## `crates/server/src/registry.rs`

- `new` (fn)

## `crates/streaming/src/error.rs`

- `StreamerError` (enum)

## `crates/streaming/src/pread_streamer.rs`

- `PreadExpertStreamer` (struct)

## `crates/streaming/src/stream_layout.rs`

- `expert_offset` (fn)

## `crates/tokenizer/src/error.rs`

- `TokenizerError` (enum)
- `ToolCallParserError` (enum)

## `crates/tokenizer/src/json_value.rs`

- `JsonValue` (enum)
- `as_object` (fn)

## `crates/tokenizer/src/stop_matcher.rs`

- `new` (fn)
- `is_stopped` (fn)

## `crates/tokenizer/src/tool_call/mod.rs`

- `deepseek_dsml_mark` (fn)
- `ParsedToolCall` (struct)

## `crates/vision-io/src/error.rs`

- `VisionIoError` (enum)

## `crates/vision-io/src/lib.rs`

- `error` (mod)
- `mrope` (mod)
- `normalize` (mod)
- `params` (mod)
- `patchify` (mod)
- `pos_embed` (mod)
- `preprocess` (mod)
- `resize` (mod)
- `rope` (mod)
- `rounding` (mod)
- `smart_resize` (mod)

## `crates/vision-io/src/patchify.rs`

- `new` (fn)

## `crates/vision-io/src/pos_embed.rs`

- `is_empty` (fn)

## `crates/window-fit/src/lib.rs`

- `outcome` (mod)

## `scripts/extract_direction.py`

- `write_gguf` (function)
- `main` (function)

## `scripts/fetch_vision_tower.py`

- `main` (function)

## `scripts/ffn_sparsity.py`

- `load_group` (function)
- `main` (function)

## `scripts/generate_app_icons.py`

- `extract_and_generate_icons` (function)

## `scripts/kld.py`

- `snapshot_dir` (function)
- `main` (function)

## `scripts/kld_llamacpp.py`

- `build_harness` (function)
- `load_port_dump` (function)
- `main` (function)

## `scripts/kld_mlx_affine.py`

- `snapshot_dir` (function)
- `main` (function)

## `scripts/kld_mlx_vlm.py`

- `snapshot_dir` (function)
- `prepare` (function)
- `compare` (function)
- `main` (function)

## `scripts/make_vision_test_page.py`

- `render` (function)
- `main` (function)

## `scripts/mlx_1bit_oracle.py`

- `fetch` (function)
- `main` (function)

## `scripts/mlx_2bit_oracle.py`

- `fetch` (function)
- `main` (function)

## `scripts/mlx_prefill.py`

- `main` (function)

## `scripts/mlx_qmm_reference.py`

- `main` (function)

## `scripts/mtp_bisect.py`

- `f16` (function)
- `rms_norm` (function)
- `corr` (function)
- `main` (function)

## `scripts/pilot_ceiling.py`

- `analyse` (function)
- `main` (function)

## `scripts/qwen3vl_vision_oracle.py`

- `rust_u8_slice` (function)
- `rust_f32_slice` (function)
- `write` (function)
- `emit_smart_resize` (function)
- `emit_preprocess` (function)
- `emit_pos_embed` (function)
- `emit_rope_freqs` (function)
- `emit_mrope` (function)
- `main` (function)

## `scripts/router_hist.py`

- `load_group` (function)
- `main` (function)

## `scripts/router_window.py`

- `main` (function)

## `scripts/skill_state_probe.py`

- `empty_state` (function)
- `run` (function)
- `summarize` (function)
- `main` (function)
- `LocalAgent.reply` (method)
- `ServerAgent` (class)
- `ServerAgent.reply` (method)

## `scripts/vision_tower_probe.py`

- `load_vision_model` (function)
- `load_vision_config` (function)
- `build_model` (function)
- `preprocess` (function)
- `run_forward` (function)
- `mode_activation` (function)
- `mode_int4` (function)
- `main` (function)

## `swift/TurboSpark/Sources/TurboSpark/Catalog.swift`

- `CatalogEntry.id` (computed property)
- `CatalogEntry.alias` (property)
- `CatalogEntry.name` (property)
- `CatalogEntry.family` (property)
- `CatalogEntry.status` (property)
- `CatalogEntry.notes` (property)
- `InstalledModel.id` (computed property)
- `InstalledModel.alias` (property)
- `InstalledModel.path` (property)
- `InstalledModel.family` (property)
- `InstalledModel.installBytes` (property)
- `InstalledModel.init(alias:repo:revision:path:family:installBytes:installedOn:)` (initializer)
- `InstallCost.downloadBytes` (property)
- `InstallCost.installBytes` (property)
- `InstallEvent.stage` (enum case)
- `InstallEvent.bytes` (enum case)
- `InstallEvent.finished` (enum case)

## `swift/TurboSpark/Sources/TurboSpark/ChatTypes.swift`

- `ChatMessage.Role.system` (enum case)
- `ChatMessage.Role.developer` (enum case)
- `ChatMessage.Role.user` (enum case)
- `ChatMessage.Role.assistant` (enum case)
- `ChatMessage.Role.tool` (enum case)

## `swift/TurboSpark/Sources/TurboSpark/Errors.swift`

- `TurboSparkError.Code` (enum)
- `TurboSparkError.Code.invalidArgument` (enum case)
- `TurboSparkError.Code.open` (enum case)
- `TurboSparkError.Code.generate` (enum case)
- `TurboSparkError.Code.json` (enum case)
- `TurboSparkError.Code.unsupportedPlatform` (enum case)
- `TurboSparkError.Code.unknown` (enum case)
- `TurboSparkError.code` (property)
- `TurboSparkError.message` (property)
- `TurboSparkError.description` (computed property)

## `swift/TurboSpark/Sources/TurboSpark/GenerationTypes.swift`

- `GenerationResult.init(from:)` (initializer)
- `PhaseReport.calls` (property)
- `PhaseReport.totalMsPerCall` (property)
- `PhaseReport.gpuWaitMs` (property)
- `PhaseReport.finalWaitMs` (property)
- `PhaseReport.routerMs` (property)
- `PhaseReport.expertIoMs` (property)
- `PhaseReport.bindMs` (property)
- `PhaseReport.pipelineWaitMs` (property)
- `PhaseReport.cb1GpuMs` (property)
- `PhaseReport.routedCbGpuMs` (property)
- `PhaseReport.finalCbGpuMs` (property)
- `PhaseReport.expertRequests` (property)
- `PhaseReport.expertHits` (property)

## `swift/TurboSpark/Sources/TurboSpark/Options.swift`

- `OpenOptions.Sizing.encode(to:)` (method)
- `OpenOptions.Speculation.encode(to:)` (method)
- `OpenOptions.LoadGuard.encode(to:)` (method)
- `GenerateOptions.Reasoning.id` (computed property)
- `GenerateOptions.Reasoning.label` (computed property)
- `GenerateOptions.Reasoning.descriptionText` (computed property)

## `swift/TurboSpark/Sources/TurboSpark/SessionTypes.swift`

- `SessionInfo.ReasoningSupport` (enum)
- `SessionInfo.modelPath` (property)
- `SessionInfo.family` (property)
- `SessionInfo.trainedContext` (property)
- `SessionInfo.vocabSize` (property)
- `SessionInfo.dialect` (property)
- `SessionInfo.Speculation.Drafter` (enum)
- `SessionInfo.SpecialTokens.bosId` (property)
- `SessionInfo.SpecialTokens.eosId` (property)
- `SessionInfo.SpecialTokens.padId` (property)
- `SessionInfo.SpecialTokens.endOfTurnId` (property)
- `SessionInfo.SpecialTokens.stopTokenIds` (property)
- `SessionInfo.SpecialTokens.thinkStartId` (property)
- `SessionInfo.SpecialTokens.thinkEndId` (property)

## `swift/TurboSpark/Sources/TurboSpark/SystemTypes.swift`

- `SystemTelemetry.init(from:)` (initializer)

## `swift/TurboSpark/Sources/TurboSpark/TurboSparkServer.swift`

- `ServerOptions.init(port:apiKey:)` (initializer)
- `ServerEvent.requestStarted` (enum case)
- `ServerEvent.requestRouted` (enum case)
- `ServerEvent.generated` (enum case)
- `ServerEvent.requestFinished` (enum case)
- `ServerEvent.modelAttached` (enum case)
- `ServerEvent.modelDetached` (enum case)
- `ServerEvent.init(from:)` (initializer)
- `ServerEventBatch.events` (property)
