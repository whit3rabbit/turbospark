//! `tslab`: the evaluator behind kernel and method experiments.
//!
//! Every decision in this repo about a kernel arm (Steel against scalar
//! prefill, the MMA8 verify tile, the routed batch, the ANE split, slot counts,
//! residency) used to be measured by a one-off script that re-implemented arm
//! rotation, host checks, provenance and stats. This crate is the one copy:
//!
//! - `tslab profile`: per-family prefill and decode phase breakdown.
//! - `tslab ab`: interleaved fresh-process A/B with a statistical verdict and
//!   quality and memory gates.
//! - `tslab kernel`: microbenches at shapes derived from the install's
//!   `ArchConfig`, with c(M) fits.
//! - `tslab ane-plan`: where Core ML would place a program.
//! - `tslab run-experiment`: the fixed evaluator an autoresearch loop calls
//!   (`program.md` beside this crate is that loop's instructions).
//!
//! The orchestrator decides; `tslab-worker` only measures, one fresh process
//! per run, so process-cached knobs (`OnceLock` env reads) are safe by
//! construction. Results go to an append-only ledger (`ledger.rs`).
//!
//! The portable half (stats, ledger, spec, knob table, gates) builds on every
//! host so its tests run anywhere. Everything that touches a model, a Metal
//! device or `pmset` is macOS-only.

pub mod atomic;
pub mod gates;
pub mod knobs;
pub mod ledger;
pub mod lock;
pub mod paths;
pub mod spec;
pub mod stats;
pub mod workload;

// pending: ane
#[cfg(target_os = "macos")]
pub mod host;
// pending: integrity
// pending: kernel
// pending: orchestrate
#[cfg(target_os = "macos")]
pub mod phases;
#[cfg(target_os = "macos")]
pub mod provenance;
// pending: report
// pending: worker
