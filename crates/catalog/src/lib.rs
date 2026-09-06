//! The model catalog, the Hugging Face probe, and the install driver behind
//! `turbospark-model`.
//!
//! Three layers, and they answer three different questions:
//!
//! - [`Catalog`] -- **what has been run here.** A curated table whose every
//!   row names a repository and a revision that were streamed and generated
//!   on real hardware, with the gate targets that assert it. Small by design.
//! - [`probe`] -- **what COULD be run here.** Reads a header and decides:
//!   architecture, block types or affine width, expert granularity, tokenizer
//!   sidecars. Costs KB and seconds, never a download. This is the layer that
//!   scales past the table.
//! - [`install`] -- **the walk**, once, instead of thirteen times. The shape
//!   every `crates/repack/tests/*_network.rs` file repeats by hand, with the
//!   step order inverted so the cheap failures happen first.
//!
//! Nothing here decodes, dispatches or allocates a Metal buffer, so this
//! crate builds on every platform even though the models it installs only run
//! on macOS.

#![forbid(unsafe_code)]

mod catalog;
mod entry;
mod hf;
mod install;
mod probe;
mod recommend;
mod store;
mod stream;
mod vision;

pub use catalog::{Catalog, SCHEMA_VERSION};
pub use entry::{
    CatalogEntry, EntryKind, Measured, MtpSource, Sidecars, Source, SourceKind, Status,
};
pub use hf::{Client, PopularRepo, RepoFile, RepoRef};
pub use install::{
    gate, human_bytes, install, install_with_byte_progress, record, ByteProgressCallback,
    InstallPlan, Installed, VISION_SIDECAR_FILES,
};
pub use probe::{
    evaluate_config, evaluate_gguf, probe, ProbeReport, TypeShare, Verdict, KNOWN_SIDECARS,
};
pub use recommend::{
    context_ladder, discover, fit, from_entry, gguf_variants, name_params_hint, probe_entry,
    rank_recommendations, recommend_catalog, CountedSource, DiscoverOptions, Evidence, Fit,
    FitVerdict, GgufVariant, GgufVariants, LadderRung, Machine, Origin, Recommendation, Shape,
    ThroughputBand,
};
pub use store::{default_root, directory_bytes, resolve_model_arg, InstalledModel, Store};
pub use vision::resolve_vision_sidecar;

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate's tokenizer
// verification step produces.
pub use foundation::TokenId;
