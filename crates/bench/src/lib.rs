//! Library surface of the bench harness, so the memory-oracle integration
//! test can drive the exact same real-install flow the `turbospark-bench`
//! binary runs (`--model` mode): frozen community-protocol prompts, the
//! Swift-parity `phys_footprint` sampler, and per-case results.

pub mod protocol;

#[cfg(target_os = "macos")]
pub mod memory;
#[cfg(target_os = "macos")]
pub mod real_model;
