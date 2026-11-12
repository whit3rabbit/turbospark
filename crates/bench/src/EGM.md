# turbospark-bench source

## Overview
Library and binaries of the bench harness. The library lets the memory-oracle integration test drive the same real-install flow as the `turbospark-bench` binary: frozen community-protocol prompts, the Swift-parity `phys_footprint` sampler and per-case results [lib.rs module doc]. Binary entry points are `main.rs`, `startup_probe_main.rs` and `meeting_eval_main.rs`.

## Interface
- Protocol constants and cases: `PROTOCOL_CASES`, `PROTOCOL_TEMPERATURE`, `PROTOCOL_TOP_K`, `PROTOCOL_TOP_P`, `PROTOCOL_MAX_NEW`, `PROTOCOL_MAX_CONTEXT`, `PROTOCOL_EXPERT_CACHE_SLOTS`, `swift_footer`, `swift_reason_name` [protocol.rs].
- Distribution comparison: `compare`, `log_softmax`, `top_k` [logit_compare.rs]; corpus: `corpus_text`, `slices` [quality_corpus.rs].
- Tree versus chain verify replay: `replay`, `best_first_nodes`, `chain_nodes`, `accepted`, `oracle_accepted`, and the `edge_*` variants (`EdgeNode`, `edge_chain_nodes`, `edge_best_first_nodes`, `edge_accepted`, `edge_oracle_accepted`) [tree_replay.rs]; macOS recorders `record_mtp`, `record_dflash` [tree_replay_record.rs].
- Binary modes: `run_scripted_mode`, `run_real_mode`, `FIXED_PROMPTS`, `FIXED_SEED`, `FIXED_MAX_NEW_TOKENS`, `RunStats` [scripted.rs]; `run_model_mode` [model_mode.rs].
- macOS only: `AppMemorySampler`, `task_fault_counters`, `chip_brand_string` [memory.rs]; `run_protocol_case`, `run_protocol_case_with_context`, `run_protocol_case_with_budget`, `run_protocol_case_speculating`, `ProtocolShaping`, `CaseResult` [real_model.rs]; `open_model_runner` and its speculative, steered, context, vision-sidecar and KV-quant variants [real_model_open.rs]; `protocol_parameters` [real_model_params.rs]; `main_entry` [startup_probe.rs].
- Child: [meeting](./meeting/EGM.md) (cpWER scoring).

## Invariants
- `--model` mode refuses to run when `TURBOSPARK_PREFILL_CHUNK` is set [main.rs:main], so no unlabelled arm is inherited.
- Memory, real-model, startup-probe and tree-replay-record modules are gated on `target_os = "macos"` [lib.rs].

## Side Effects
- `run_model_mode` loads a real install and runs GPU inference [model_mode.rs:run_model_mode].
- `AppMemorySampler` reads process memory through Mach calls [memory.rs].

## Verification
- Run: `cargo test -p turbospark-bench`
