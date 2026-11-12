# tslab kernel benches

## Overview
`tslab kernel`: microbenches at shapes derived from an install's `ArchConfig` (text) or `config.json` (audio). Benches run in the orchestrator process (no worker; synthetic weights, except `vision_encode`, which opens the real model via `RealForwardRunner::open_with_max_context` [src/kernel/vision_encode.rs:VisionEncode]), take the machine lock and host guard, and ledger a `kernel_curve` record [src/kernel/mod.rs:kernel_cmd]. macOS only.

## Interface
- `KernelBench` trait, `registry()`, `by_name`, `arm_labels`, `coverage` [src/kernel/mod.rs:KernelBench] [src/kernel/mod.rs:registry].
- `Subject`, `MatShape`, `matmul_shapes`, `Case`, `BenchOpts`, `kernel_cmd` [src/kernel/mod.rs:Subject] [src/kernel/mod.rs:kernel_cmd].
- Benches: `AneHandoff` [src/kernel/ane_handoff.rs:AneHandoff], `AneMatmul` [src/kernel/ane_matmul.rs:AneMatmul], `AttnDecode` (arms `barrier`, `gqa`) [src/kernel/attn_decode.rs:AttnDecode], `GdnDelta` [src/kernel/gdn_delta.rs:GdnDelta], `Int4Matmul` [src/kernel/int4_matmul.rs:Int4Matmul], `Int8Matmul` [src/kernel/int8_matmul.rs:Int8Matmul], `MoeRouted` [src/kernel/moe_routed.rs:MoeRouted], `Music3Grid` [src/kernel/music3_grid.rs:Music3Grid], `QkvConcurrent` [src/kernel/qkv_concurrent.rs:QkvConcurrent], `RowsAttn` [src/kernel/rows_attn.rs:RowsAttn], `VerifyLadder` [src/kernel/verify_ladder.rs:VerifyLadder], `VisionEncode` [src/kernel/vision_encode.rs:VisionEncode], `WhisperLinear` [src/kernel/whisper_linear.rs:WhisperLinear].
- `support`: synthetic weight blobs and sampling helpers (`int4_blob`, `int8_blob`, `median`, `Primer`) [src/kernel/support.rs:int4_blob] [src/kernel/support.rs:Primer].

## Invariants
- Sample discipline: distinct weight copies per sample, arms alternated within each repetition, first repetition discarded [src/kernel/support.rs:copies_for] [src/kernel/int8_matmul.rs:Int8Matmul].
- The harness sets and clears no env knobs and calls kernels directly, but benches run in the caller's process, so env reads inside the gpu crate (for example `TURBOSPARK_ATTN_DECODE_KERNEL`, `TURBOSPARK_GDN_DK128`, `TURBOSPARK_METAL_PRECOMPILED`) still apply [src/kernel/mod.rs:kernel_cmd].
- `moe_routed` uses a uniform-routing expert union, an upper bound on the batched arm's cost [src/kernel/moe_routed.rs:uniform_union].
- `music3_grid` shares one corpus with the gpu crate by path include [src/kernel/music3_grid.rs:Music3Grid].

## Side Effects
- Takes `MachineLock::acquire` for the session [src/kernel/mod.rs:kernel_cmd].
- Appends `kernel_curve` records to the experiment ledger, plus a `vision_encode` record for each vision bench result [src/kernel/mod.rs:kernel_cmd].
- Runs Metal and Core ML work on the device; `ane_matmul` and `ane_handoff` always register but refuse to run without the `ane-prefill` feature [src/kernel/ane_handoff.rs:AneHandoff].

## Verification
- Run: `cargo test -p turbospark-lab`
