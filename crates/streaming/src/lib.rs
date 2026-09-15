//! Routed-expert streaming: `pread`-based cache streamer, LFU/LRU eviction
//! policy, and readahead advice. Ported from `Infrastructure/Streaming`.
//!
//! Allowed a narrow amount of `unsafe` and platform `cfg` (the macOS
//! `F_RDADVISE` `fcntl` in `rdadvice`, and `disk_io`'s `proc_pid_rusage`
//! and `F_NOCACHE`), per the workspace's cross-cutting rule that streaming
//! is one of the few crates where that trade-off is made.

mod aligned_slot;
mod disk_io;
mod error;
mod expert_cache;
mod linux_uring;
mod mapped_experts;
mod pread_streamer;
mod rdadvice;
mod read_pool;
mod stream_layout;

pub use aligned_slot::AlignedSlot;
pub use disk_io::ExpertIoStats;
pub use error::StreamerError;
pub use expert_cache::{
    coalesced_adjacent_advice_ranges, ExpertCache, ExpertCachePlan, ExpertCachePolicy,
    ExpertIoAdviceResult,
};
pub use linux_uring::{dio_aligned, dio_round_up, Mode as LinuxIoMode, DIO_ALIGNMENT};
pub use mapped_experts::MappedExpertLayer;
pub use pread_streamer::PreadExpertStreamer;
pub use rdadvice::{call as rdadvice_call, clipped_byte_count, RdAdviceCallResult};
pub use stream_layout::StreamLayout;

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
