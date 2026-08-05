//! Metal GPU backend: device/pipeline-cache context and per-kernel
//! dispatch, validated against `mrefrust-compute`'s CPU reference kernels.
//! Ported from `Infrastructure/Metal` and the `.metal` shader sources under
//! `Metal/*`.
//!
//! macOS-only (per the workspace's Phase 6 decision: reuse the existing MSL
//! shaders nearly as-is via the `metal` crate, matching how Mference itself
//! compiles them from source at startup). On other platforms this crate
//! exposes nothing, so `cargo build --workspace` / `cargo test --workspace`
//! still succeed on Linux CI; only macOS gets the real implementation.
//!
//! Status: the pipeline cache and five kernel dispatches
//! (`rmsnorm_no_scale`, `rope_proportional_neox`, `logit_softcap_softmax`,
//! `dequant_int4_gemv_simd`, `dequant_int8_gemv_simd`) are wired end-to-end
//! and parity-tested against `mrefrust_compute` on real hardware.
//! `KvCacheManager`, `GdnStateManager`, and `Dsv4StateManager` (real Metal
//! buffer allocation, position/capacity bookkeeping ported from the Swift
//! originals) are also wired, as is `PrefillChunkScratchLayout`/
//! `PrefillChunkScratchBuffers` (chunked-prefill scratch-space sizing and
//! allocation). What's not yet vendored or dispatched: attention, MoE, the
//! `sample` kernel (no CPU reference exists to verify a port against — see
//! `DEVIATIONS.md`), the chunked-prefill tile-pipeline kernel itself (the
//! scratch buffers it would use are allocated but nothing writes through
//! them yet), and the GDN/DSV4 compute kernels that would read and write
//! through their respective state managers.

#[cfg(target_os = "macos")]
mod attention_decode;
#[cfg(target_os = "macos")]
mod bytes;
#[cfg(target_os = "macos")]
mod context;
#[cfg(target_os = "macos")]
mod dequant_int4_gemv;
#[cfg(target_os = "macos")]
mod dequant_int8_gemv;
#[cfg(target_os = "macos")]
mod dsv4_state;
#[cfg(target_os = "macos")]
mod gdn_state;
#[cfg(target_os = "macos")]
mod kv_cache;
#[cfg(target_os = "macos")]
mod logit_softmax;
#[cfg(target_os = "macos")]
mod prefill_scratch;
#[cfg(target_os = "macos")]
mod rms_norm;
#[cfg(target_os = "macos")]
mod rope;

#[cfg(target_os = "macos")]
pub use attention_decode::attention_decode;
#[cfg(target_os = "macos")]
pub use context::{dispatch_one_threadgroup_per_row, dispatch_threads_3d, GpuError, MetalContext};
#[cfg(target_os = "macos")]
pub use dequant_int4_gemv::{dequant_int4_gemv, Int4AffineRowGpu};
#[cfg(target_os = "macos")]
pub use dequant_int8_gemv::{dequant_int8_gemv, Int8AffineRowGpu};
#[cfg(target_os = "macos")]
pub use dsv4_state::{Dsv4StateManager, LayerCounters};
#[cfg(target_os = "macos")]
pub use gdn_state::GdnStateManager;
#[cfg(target_os = "macos")]
pub use kv_cache::{KvCacheManager, KvView, LayerKind};
#[cfg(target_os = "macos")]
pub use logit_softmax::logit_softcap_softmax;
#[cfg(target_os = "macos")]
pub use prefill_scratch::{PrefillChunkScratchBuffers, PrefillChunkScratchLayout};
#[cfg(target_os = "macos")]
pub use rms_norm::rms_norm_no_scale;
#[cfg(target_os = "macos")]
pub use rope::rope_proportional_neox;

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
