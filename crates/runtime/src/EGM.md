# TurboSpark Runtime Sources

## Overview
The top level of `turbospark-runtime`. It holds the Metal-backed `RealForwardRunner` (the `LogitProducer` for every supported model family), the generic raw-completion and speculative decode loops, steering and diagnostic capture, native audio/TTS/ASR runners, and power and startup policy. The crate is `#![forbid(unsafe_code)]`; Metal-dependent modules are gated on `target_os = "macos"` and only the ungated modules (config, error, producer, raw completion, speculative loops, dflash tree and lattice, pacing, token sink, turn stream, power, heavy, lab hooks, expert layout validation) build elsewhere; `speculation_policy` is macOS-only [lib.rs:crate]. Family-specific flow code lives in subdirectories.

## Interface

Real forward core
- `RealForwardRunner` [real_forward.rs:RealForwardRunner]: runner struct owning the Metal context, resident weights, `ArchConfig`, KV, scratch and expert streamers
- `RealForwardRunner::open` / `open_with_max_context` [real_forward_api.rs:open]: open ladder over a `.gturbo` dir; wider entry points add KV quant, session pool, steering, residency
- `OpenRole` [real_forward_open.rs:OpenRole]: Chat vs HeadlessTrunk open purpose
- `validate_arch_config`, `kv_quant_unsupported_reason` [real_forward_init.rs:validate_arch_config]: open-time dimension and KV-quant gates
- `encode_embed_any`, `encode_gemv_any` [real_forward_dispatch.rs:encode_gemv_any]: dtype-driven dispatchers; `encode_moe_phase1_any`, `encode_moe_phase2_any` [real_forward_dispatch_moe.rs:encode_moe_phase1_any] for routed experts
- `GGUF_BLOCK_DTYPES`, `EXECUTABLE_GGUF_DTYPES` [real_forward_layout.rs:EXECUTABLE_GGUF_DTYPES]: resident dtype tags and executability
- `RollbackPoint` [real_forward_rollback.rs:RollbackPoint]: checkpoint/rollback with lazy GDN tape
- `RealForwardError` [real_forward_types.rs:RealForwardError]: Model, Gpu, MissingTensor, Unsupported
- `LogitProducer`, `ChunkedPrefillRunner` [producer.rs:LogitProducer]: producer trait family; runner impls in [real_forward_traits.rs:produce]
- `run_raw_completion`, `run_raw_completion_cancellable` [raw_completion.rs:run_raw_completion_cancellable]; chunked variants [raw_completion_chunked.rs:run_raw_completion_chunked_cancellable]
- `GenerationConfig`, `RuntimeError`, `Pacer`, `TokenSink`, `TurnSplitter` [config.rs:GenerationConfig], [error.rs:RuntimeError], [pacing.rs:Pacer], [token_sink.rs:TokenSink], [turn_stream.rs:TurnSplitter]
- `KvPrefix` [kv_prefix.rs:KvPrefix], `SessionPool` [session_pool.rs:SessionPool], `kv_write_target` [kv_write.rs:kv_write_target]: prefix reuse, parked sessions, quantized KV writes
- `has_vision_tower`, `open_vision_tower` [real_forward_vision_api.rs:open_vision_tower]; `UnlimitedOcr2Reader` [unlimited_ocr2_reader.rs:UnlimitedOcr2Reader]
- `HiddenTrunk` [hidden_trunk.rs:HiddenTrunk]: headless Qwen GDN MoE trunk driven one row at a time
- `mapped_expert_count_matches` [expert_layout_validation.rs:mapped_expert_count_matches], `read_bounded` [model_dir_read.rs:read_bounded], `probe_embedding_dir` [embedding_probe.rs:probe_embedding_dir]

Speculation
- `Speculation`, `SpeculativeDrafter` [speculation_policy.rs:Speculation]: drafter choice and request policy
- `run_raw_completion_speculative`, `DEFAULT_SPECULATION_BLOCK` [speculative.rs:run_raw_completion_speculative]
- `AdaptiveParams` [speculative_adaptive.rs:AdaptiveParams]: hysteresis depth controller
- `dflash_best_first_tree`, `dflash_tree_accept` [dflash_tree.rs:dflash_best_first_tree]; `DflashLattice` [draft_lattice.rs:DflashLattice]
- `TURN_SNAPSHOT_MARGIN`, `turn_snapshot_enabled` [turn_snapshot.rs:turn_snapshot_enabled]: opt-in recurrent snapshot for turn rewind
- `RoutedSlot`, `routed_pipeline_banks` [moe_prefill_pipeline.rs:routed_pipeline_banks]: MoE prefill pipelining

Steering and capture
- `SteeringPolicy`, `SteeringVector` [steering.rs:SteeringPolicy]: directional steering policy and GPU state
- `ResidCapture` [resid_capture.rs:ResidCapture], `FfnActHist` [ffn_hist.rs:FfnActHist], `RouterHistogram` [router_hist.rs:RouterHistogram]: env-gated diagnostics
- `dispatch_probe_count`, `encode_dispatch_probe` [lab_hooks.rs:dispatch_probe_count]: lab measurement hooks

Audio, TTS and ASR
- `AudioSession`, `AudioRequest` [native_audio.rs:AudioSession]: shared worker-thread session for STT, TTS, music, diarization
- `MoonshineRunner` [moonshine.rs:MoonshineRunner], `Qwen3AsrRunner` [qwen3_asr.rs:Qwen3AsrRunner], `Qwen3AsrMetalEngine` [qwen3_asr_metal.rs:Qwen3AsrMetalEngine]
- `MultitalkerEncoderMetal` [multitalker_metal.rs:MultitalkerEncoderMetal], `NemotronDiarMetal` [nemotron_diar_metal.rs:NemotronDiarMetal]
- `KokoroRunner` [kokoro.rs:KokoroRunner], `MegaTts3Runner` [megatts3.rs:MegaTts3Runner], `OmniVoiceRunner` [omnivoice.rs:OmniVoiceRunner], `Music3Runner` [music3.rs:Music3Runner]
- Fish Drama 3: `TrunkBackbone` [fish_drama3.rs:TrunkBackbone], `MetalTower` [fish_drama3_tower.rs:MetalTower], `CodecWorker` [fish_drama3_codec.rs:CodecWorker]
- `HeavyWorkGuard` [heavy.rs:HeavyWorkGuard]: process-wide single permit for heavyweight Metal work

Power and startup
- `PowerProfile`, `ThermalLevel` [power.rs:PowerProfile]: profiles and thermal/memory ladders
- `StartupStats`, `ExpertPrefetchStats` [startup.rs:StartupStats]; `prepare_kernel_warmup` [kernel_warmup.rs:prepare_kernel_warmup]

Subdirectories
- [coreml](./coreml/EGM.md)
- [encoder](./encoder/EGM.md)
- [families](./families/EGM.md)
- [native_audio](./native_audio/EGM.md)
- [router_hist](./router_hist/EGM.md)
- [vision](./vision/EGM.md)
- [whisper](./whisper/EGM.md)

## Invariants
- Decode flow follows the manifest family, and exactly one family state is populated per install; unknown dtypes return Unsupported rather than defaulting to INT4 [real_forward.rs:RealForwardRunner], [real_forward_dispatch.rs:encode_gemv_any]
- Measuring open entry points pin speculation off, steering off, one session slot, KvQuant::Off and Streamed residency [real_forward_api.rs:open_with_options]; `open_with_kv_quant` takes all of them from the caller and defaults residency to Auto
- Admission requires a non-empty prompt, max_new_tokens > 0 (else ZeroBudget) and prompt_len + max_new <= max_context; [raw_completion.rs:check_admission]; prefix reuse is clamped to prompt_len - 1 so one token is always fed [raw_completion.rs:run_raw_completion_cancellable]
- Unservable speculation returns SpeculationUnavailable, never a silent sequential fallback; block 0 and sampled block-drafter runs are refused [error.rs:RuntimeError], [speculative.rs:run_speculative]
- `try_reuse_prefix` keys on fed token ids and returns 0 after non-id inputs (vision embeddings, host rows) [producer.rs:try_reuse_prefix], [hidden_trunk.rs:step]
- A quantized KV layer is written through staging then `encode_kv_commit`, never in place [kv_write.rs:kv_write_target]
- A routed sub-batch union must fit the expert cache slots because the plan asserts [moe_prefill_pipeline.rs:next_routed_sub_batch]
- Arch dimensions are validated against ceilings before any buffer is sized [real_forward_init.rs:validate_arch_config]
- Steering Off is byte-identical to no steering; bad alpha/target/zero-norm directions fail at open [steering.rs:build]
- Audio engines live on their worker thread only; Metal audio backends accept only F32 and refuse otherwise [native_audio.rs:AudioSession], [kokoro.rs:load_weight]
- Model-dir reads are size-capped [model_dir_read.rs:read_bounded]
- Env-gated diagnostics change no math; unwired families are ignored with a log [ffn_hist.rs:from_env], [resid_capture.rs:from_env]

## Side Effects
- Open mmaps `model_weights.bin`, allocates Metal KV/scratch buffers, and does expert pread I/O on streamed installs [real_forward_open.rs:open_inner_with_role], [moe_prefill_pipeline.rs:plan_and_stream_union]
- Env reads: TURBOSPARK_GDN_LAZY_TAPE, TURBOSPARK_GDN_TURN_SNAPSHOT, TURBOSPARK_ADAPTIVE_DEPTH, TURBOSPARK_SPEC_STATS, TURBOSPARK_EXPERT_RESIDENCY, TURBOSPARK_INT4_MMA8, TURBOSPARK_MOONSHINE_DEVICE, TURBOSPARK_METAL_KERNEL_WARMUP (opt-in JIT warmup, only "1" enables) [real_forward_rollback.rs:gdn_lazy_tape], [turn_snapshot.rs:turn_snapshot_enabled], [speculative_adaptive.rs:from_env], [real_forward_init.rs:mapped_residency_requested], [real_forward_dispatch.rs:int4_mma8_mode], [moonshine.rs:device_choice], [real_forward_open.rs:TURBOSPARK_METAL_KERNEL_WARMUP]
- Diagnostics gated by TURBOSPARK_FFN_HIST, TURBOSPARK_RESID_CAPTURE, TURBOSPARK_ROUTER_HIST (writes a JSON file on Drop), TURBOSPARK_DISPATCH_PROBE [ffn_hist.rs:from_env], [resid_capture.rs:from_env], [router_hist.rs:drop], [lab_hooks.rs:dispatch_probe_count]
- Process-global heavy-work permit (Mutex/Condvar) blocks or refuses concurrent heavy jobs [heavy.rs:STATE]
- Spawns worker threads `native-audio` and `fish-drama3-codec` [native_audio.rs:AudioSession], [fish_drama3_codec.rs:CodecWorker::open]
- Reads bounded model-dir files and resident indexes; writes the pipeline cache to disk [model_dir_read.rs:read_bounded], [startup.rs:flush_pipeline_cache]
- Power probes call into `crates/gpu` for thermal and memory state; the decode loop's single sleep lives in [raw_completion.rs:decode] [power.rs:thermal_level]

## Verification
- Run: `cargo test -p turbospark-runtime`
