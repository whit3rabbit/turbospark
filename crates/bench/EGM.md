# turbospark-bench

## Overview
Measurement harnesses: the frozen community-protocol throughput benchmark (scripted, synthetic `--real`, and real `--model` modes), a startup probe, speculative tree-replay tooling, logit comparison, and Frozen E meeting scoring. Library surface is in [src/lib.rs]; three binaries are declared in Cargo.toml (`turbospark-bench`, `turbospark-startup-probe`, `meeting_eval`).

## Interface
- Protocol constants and cases: `PROTOCOL_CASES`, `PROTOCOL_TEMPERATURE`, `PROTOCOL_TOP_K`, `PROTOCOL_TOP_P`, `PROTOCOL_MAX_NEW`, `PROTOCOL_MAX_CONTEXT`, `PROTOCOL_EXPERT_CACHE_SLOTS`, `swift_footer`, `swift_reason_name` [src/protocol.rs:PROTOCOL_CASES]
- Distribution comparison: `compare`, `log_softmax`, `top_k` [src/logit_compare.rs:compare]; corpus: `corpus_text`, `slices` [src/quality_corpus.rs:corpus_text]
- Tree vs chain verify replay: `replay`, `best_first_nodes`, `chain_nodes` [src/tree_replay.rs:replay]; recorders `record_mtp`, `record_dflash` [src/tree_replay_record.rs:record_mtp] (macOS)
- macOS only: `AppMemorySampler` [src/memory.rs:AppMemorySampler], `run_protocol_case` [src/real_model.rs:run_protocol_case], `open_model_runner` and its speculative, steered, vision-sidecar and KV-quant variants [src/real_model_open.rs:open_model_runner_speculative], `run_protocol_case_speculating`, `ProtocolShaping` [src/real_model.rs:ProtocolShaping], `protocol_parameters` [src/real_model_params.rs:protocol_parameters], startup probe `main_entry` [src/startup_probe.rs:main_entry]
- Binary modes: `run_scripted_mode`, `run_real_mode` [src/scripted.rs], `run_model_mode` [src/model_mode.rs:run_model_mode]
- Children: [src](./src/EGM.md) (library and binaries), [src/meeting](./src/meeting/EGM.md) (cpWER scoring), [scripts](./scripts/EGM.md) (Python fixture and scorer helpers)

## Invariants
- `--model` mode refuses to run when `TURBOSPARK_PREFILL_CHUNK` is set, so no unlabelled arm is inherited [src/main.rs:main].
- Scripted mode measures loop overhead only, not inference throughput [src/main.rs module doc].
- Memory, real-model, startup-probe and tree-replay-record modules are gated on `target_os = "macos"` [src/lib.rs].
- `--model` mode also takes `--case`, `--expert-cache-slots`, `--prefill-chunk`, `--power-profile`, `--max-tokens-per-sec`, `--kv-bits`, `--speculative`, `--speculative-drafter` and `--shaping`. KV quantization, prefill chunking and speculation default off so an unlabelled arm cannot be inherited; `--speculative` is refused unless `--shaping greedy` (acceptance is exact only at temperature 0), and `--speculative-drafter` only selects the drafter [src/model_mode.rs:run_model_mode, src/real_model.rs:run_protocol_case_speculating].
- `--model` mode echoes `TURBOSPARK_ROUTED_BATCH` and `TURBOSPARK_BATCHED_GEMV` in its banner, marked INERT when prefill is sequential, because the chunked driver alone reads them [src/model_mode.rs:run_model_mode].
- Startup probe is separate from the frozen protocol [src/lib.rs:startup_probe].

## Side Effects
- `--model` mode loads a real install, runs GPU inference, and prints results and `[stop=...]` footers [src/model_mode.rs:run_model_mode].
- Startup probe reads `TURBOSPARK_METAL_*`, `TURBOSPARK_QWEN_SHARED_READ_OVERLAP` and `TURBOSPARK_MAPPED_DEMAND_PREP` and reports them [src/startup_probe.rs:run].
- Memory sampler reads task VM info via Mach calls [src/memory.rs:read_process_footprint].

## Verification
- Run: `cargo test -p turbospark-bench`
