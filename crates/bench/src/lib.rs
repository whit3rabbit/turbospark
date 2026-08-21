//! Library surface of the bench harness, so the memory-oracle integration
//! test can drive the exact same real-install flow the `turbospark-bench`
//! binary runs (`--model` mode): frozen community-protocol prompts, the
//! Swift-parity `phys_footprint` sampler, and per-case results.

/// Benchmarking protocol case definitions and runner logic.
pub mod protocol;

/// Memory sampling helpers (macOS phys_footprint).
#[cfg(target_os = "macos")]
pub mod memory;
/// Real model benchmark harness and case execution.
#[cfg(target_os = "macos")]
pub mod real_model;
/// Real model protocol parameters.
pub mod real_model_params;
