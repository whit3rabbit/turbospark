//! Allowed numeric sets and documented defaults for runtime knobs.
//!
//! This module is the single home for the allowed-value sets and fixed
//! defaults that `crates/invocation`, `crates/cli`, `crates/server` and
//! `crates/runtime` all validate against. It holds no builder and no
//! `Result`-returning constructor: a caller that wants a value outside an
//! allowed set has nowhere to go here but the const assertions below, which
//! catch a default drifting out of its own set at compile time.

/// Allowed cache-slot values.
///
/// `[8, 16, 24, 32]` until `qwen4_exp`'s Phase 4 (`docs/QWEN4_PHASE0.md`):
/// its 288 experts at top-10 cost 126.6 MiB per slot (48 layers x 2.7648 MB),
/// so 32 slots is only 3.95 GiB / 11% residency of the routed table. 48, 64,
/// 96 and 128 were added rather than replacing the old ceiling, because every
/// existing family's frozen memory-oracle row was measured against `Auto`
/// topping out at 32 and a wider set only raises that ceiling for an install
/// whose expert stride is small enough to afford it (AGENTS.md Gotcha 36).
pub const ALLOWED_CACHE_SLOTS: [u32; 8] = [8, 16, 24, 32, 48, 64, 96, 128];

/// The speculative-decoding block sizes a caller may name.
///
/// The upper bound is the batched verify's register-bound row cap (a round of
/// block B verifies `B + 1` rows against `MAX_BATCH_ROWS = 16`); the measured
/// optimum is 2 and everything above 4 loses on this engine (`docs/MTP.md`,
/// `docs/DFLASH2.md`), so the range is deliberately wider than the useful
/// part rather than pretending to be a recommendation.
///
/// **Here rather than in `crates/invocation` because it has two parsers to
/// serve.** `turbospark-check` reaches it through that crate's flat option
/// grammar and `turbospark-server` through its own; both depend on this one,
/// neither depends on the other, and a second copy of a numeric range is the
/// count-that-rots shape this repo has paid for before. Same reasoning as
/// [`ALLOWED_CACHE_SLOTS`] above, which both parsers already read for exactly
/// this purpose.
pub const ALLOWED_SPECULATION_BLOCKS: std::ops::RangeInclusive<u32> = 1..=15;

/// Documented default cache-slot count.
pub const DEFAULT_CACHE_SLOTS: u32 = 16;

/// Allowed prompt-processing chunk-size values.
pub const ALLOWED_CHUNK_SIZES: [u32; 8] = [32, 64, 128, 256, 512, 1024, 2048, 4096];

/// Documented default prompt-processing chunk size.
pub const DEFAULT_CHUNK_SIZE: u32 = 128;

/// Documented default context window, in tokens.
///
/// Has NO allowed set beside it, unlike the two knobs above: a context
/// window is a per-token KV allocation, so every positive value is legal and
/// the only real bound is what memory holds -- which `crates/runtime`'s
/// context policy checks, since it needs the machine and the install.
///
/// This is also what an `auto` window falls back to when the install
/// declares no trained context, which is every install written before that
/// field existed. Three crates read it (the argument parser, the context
/// policy, and the server's own flat parser), so it lives here rather than
/// as a literal in each.
pub const DEFAULT_MAX_CONTEXT: u32 = 4096;

/// Returns whether `value` is a member of `allowed`.
const fn contains(allowed: &[u32], value: u32) -> bool {
    let mut i = 0;
    while i < allowed.len() {
        if allowed[i] == value {
            return true;
        }
        i += 1;
    }
    false
}

/// Returns whether `values` is sorted in strictly increasing order.
/// [`crate::chunk_sizing::resolve_automatic_chunk_size`]'s smallest-covering
/// search depends on this: it takes the first entry that covers the
/// requested length, which is only the smallest such entry if the set is
/// sorted ascending.
const fn is_sorted_ascending(values: &[u32]) -> bool {
    let mut i = 1;
    while i < values.len() {
        if values[i] <= values[i - 1] {
            return false;
        }
        i += 1;
    }
    true
}

/// The largest value in `set`. `const fn` so [`crate::prefill::MAX_CHUNK_TOKENS`]
/// can derive from [`ALLOWED_CHUNK_SIZES`] instead of duplicating its
/// maximum as a separate literal.
pub const fn max_of(set: &[u32]) -> u32 {
    let mut max = set[0];
    let mut i = 1;
    while i < set.len() {
        if set[i] > max {
            max = set[i];
        }
        i += 1;
    }
    max
}

const _: () = assert!(
    !ALLOWED_CACHE_SLOTS.is_empty(),
    "ALLOWED_CACHE_SLOTS must not be empty"
);
const _: () = assert!(
    !ALLOWED_CHUNK_SIZES.is_empty(),
    "ALLOWED_CHUNK_SIZES must not be empty"
);
const _: () = assert!(
    is_sorted_ascending(&ALLOWED_CACHE_SLOTS),
    "ALLOWED_CACHE_SLOTS must be sorted strictly ascending"
);
const _: () = assert!(
    is_sorted_ascending(&ALLOWED_CHUNK_SIZES),
    "ALLOWED_CHUNK_SIZES must be sorted strictly ascending"
);
const _: () = assert!(
    contains(&ALLOWED_CACHE_SLOTS, DEFAULT_CACHE_SLOTS),
    "DEFAULT_CACHE_SLOTS must be one of ALLOWED_CACHE_SLOTS"
);
const _: () = assert!(
    contains(&ALLOWED_CHUNK_SIZES, DEFAULT_CHUNK_SIZE),
    "DEFAULT_CHUNK_SIZE must be one of ALLOWED_CHUNK_SIZES"
);
