//! Clamping the checkpoint's own declared `max_pixels` ceiling to what THIS
//! session's `LoadGuard` budget can actually afford (vision memory sidecar
//! Part B3; `docs/VISION.md`'s "What Part B started and did not finish").
//!
//! `PreprocessParams::max_pixels` comes from `preprocessor_config.json` and
//! is a hard ceiling declared by the checkpoint's publisher -- 16,777,216
//! (4096x4096) for the Qwen3.8 tower. Nothing before this module ever
//! clamped it further, so a page at that ceiling costs `VisionScratch` its
//! full formula on top of whatever KV cache and resident weights the
//! session already committed, invisible to `context_policy::
//! resolve_max_context`'s own arithmetic (which only knows about KV and
//! resident weights). On a memory-constrained machine a caller could commit
//! more than a `LoadGuard` was ever supposed to allow, with no warning.
//!
//! Two pure functions, no I/O and no globals, matching the
//! `context_policy`/`expert_cache_policy` precedent this crate's sizing
//! policies already follow: every input arrives as a parameter, and
//! [`resolve_max_pixels`]'s error mirrors [`model_io::ContextTooLarge`]'s
//! shape -- name every term of the subtraction rather than a bare "does not
//! fit".
//!
//! # The pixel-to-patch conversion is exact, not approximated
//!
//! `turbospark_vision_io::smart_resize::resized_dims`'s `max_pixels` branch
//! rescales the ORIGINAL dims by `beta = sqrt(orig_pixels / max_pixels)`
//! and then floors each edge to a multiple of `spatial_factor = patch_size *
//! merge_size`. The unrounded scaled product is exactly `max_pixels`
//! (`(orig_h/beta) * (orig_w/beta) == orig_pixels / beta^2 == max_pixels`),
//! and flooring each edge only ever shrinks it, so **no real resize under a
//! `pixels` ceiling can produce more than `floor(pixels / spatial_factor^2)
//! * spatial_factor^2` raw pixels**, i.e. `floor(pixels / spatial_factor^2)
//! * merge_size^2` patches. [`scratch_bytes_for_pixels`] uses exactly that
//! bound rather than a bare `pixels / patch_size^2` division, which is
//! deliberately conservative in the SAME direction a bare division would
//! be wrong in: it can only ever UNDER-count the patches a real image
//! produces at the ceiling, never over-count them, so a budget solved
//! against it is always safe to hand to `smart_resize` as `max_pixels`.

use model_io::LoadGuard;

use crate::vision::shape::VisionShape;

/// Scratch bytes a page of `pixels` raw pixels would cost, at this tower's
/// shape and the given MLP row tile.
///
/// Converts `pixels` to a patch count the way `smart_resize` would (see the
/// module doc), then defers to the shared B1/B2 formula
/// (`crate::vision::scratch::scratch_bytes`) rather than restating it --
/// the two must never independently drift about what a page of `N` patches
/// costs.
pub(crate) fn scratch_bytes_for_pixels(shape: &VisionShape, pixels: usize, tile: usize) -> u64 {
    crate::vision::scratch::scratch_bytes(shape, patches_for_pixel_ceiling(shape, pixels), tile)
}

/// The largest patch count any real `smart_resize` output could reach under
/// a `pixels` ceiling. See the module doc for why this is an exact
/// conservative bound rather than a division-based approximation.
fn patches_for_pixel_ceiling(shape: &VisionShape, pixels: usize) -> usize {
    let spatial_factor = shape.patch_size * shape.merge;
    let factor_sq = spatial_factor * spatial_factor;
    if factor_sq == 0 {
        // `VisionShape::resolve` already refuses a zero `patch_size` or
        // `merge`, so this is unreachable through any real shape; kept as a
        // defined answer rather than a division-by-zero panic.
        return 0;
    }
    let rounded_pixels = (pixels / factor_sq) * factor_sq;
    rounded_pixels / (shape.patch_size * shape.patch_size)
}

/// The result of resolving a checkpoint's declared `max_pixels` against a
/// session's memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelBudget {
    /// The ceiling to actually hand to `PreprocessParams::max_pixels`:
    /// `declared_max_pixels` unchanged when it already fits, or the largest
    /// value this budget affords otherwise.
    pub resolved_max_pixels: usize,
    /// True when [`Self::resolved_max_pixels`] is strictly smaller than
    /// what was declared -- the front ends' cue to report a resolved-at-open
    /// line, following the same "print only on a non-default outcome"
    /// convention `past_trained` warnings and sidecar-attach lines use.
    pub clamped: bool,
}

/// Why even `min_pixels` -- the checkpoint's own declared FLOOR -- does not
/// fit this session's memory budget.
///
/// Mirrors [`model_io::ContextTooLarge`]'s shape and reasoning: every term
/// of the subtraction is named, because a bare "vision budget too small"
/// sends the reader looking for a leak instead of at the arithmetic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionBudgetTooSmall {
    /// The ceiling that was asked for (the checkpoint's own declared value,
    /// or a caller-supplied override).
    pub declared_max_pixels: usize,
    /// The checkpoint's own declared floor -- the smallest page this budget
    /// was tested against.
    pub min_pixels: usize,
    /// Scratch bytes a page at `min_pixels` would cost.
    pub needs: u64,
    /// Bytes available for vision scratch after the guard's reserve and
    /// everything else this session already committed.
    pub available: u64,
    /// What the guard held back before `available` was computed. Carried
    /// rather than re-derived at format time, for `ContextTooLarge::
    /// reserve`'s own reason: a message quoting a module constant would name
    /// a number that did not produce this refusal under a non-default tier.
    pub reserve: u64,
    /// Physical memory on this machine.
    pub physical: u64,
    /// What the trunk already commits before vision scratch: its mapped
    /// weight region plus the routed-expert slot cache, i.e.
    /// `model_io::committed_bytes`. Same meaning as
    /// `ContextTooLarge::committed`.
    pub committed: u64,
    /// This session's own KV cache at its resolved `--max-context`, i.e.
    /// `ContextPlan::kv_bytes`. A separate field from `committed` rather
    /// than pre-summed, so the message can name each cost the way
    /// `ContextTooLarge` names weights and reserve as two separate terms.
    pub kv_bytes: u64,
}

impl std::fmt::Display for VisionBudgetTooSmall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "no page within the declared {}-pixel budget fits this session's memory: even the \
             {}-pixel floor needs {} of vision scratch; {} available ({} physical - {} \
             weights/expert cache - {} KV cache - {} reserve)",
            self.declared_max_pixels,
            self.min_pixels,
            gib(self.needs),
            gib(self.available),
            gib(self.physical),
            gib(self.committed),
            gib(self.kv_bytes),
            gib(self.reserve),
        )
    }
}

impl std::error::Error for VisionBudgetTooSmall {}

/// Bytes as GiB to one decimal, mirroring `context_policy`'s own private
/// helper of the same name (not importable -- that one is a free function
/// scoped to its own file).
fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// Resolve `declared_max_pixels` against this session's own memory budget.
///
/// `committed` and `kv_bytes` together are "everything else already spoken
/// for": the trunk's own `model_io::committed_bytes(dir)` and this
/// session's resolved `ContextPlan::kv_bytes`, the same pairing
/// `context_policy::resolve_max_context` computes into one `available`
/// figure via `guard.available(physical, committed)` -- summed here rather
/// than pre-added by the caller, so a caller passes the two numbers it
/// already has on hand rather than restating the addition.
///
/// [`LoadGuard::Off`] never clamps, matching `resolve_max_context`'s own
/// early return for that tier: the whole computation below is skipped.
///
/// Binary-searches the largest `pixels` in `[min_pixels, declared_max_pixels]`
/// whose scratch cost fits `available`, the same shape
/// `context_policy::largest_context_within` uses for a context window:
/// [`scratch_bytes_for_pixels`] is non-decreasing in `pixels` (more pixels
/// never yields fewer patches, and `scratch::scratch_bytes` is
/// non-decreasing in the patch count), so a bisection is exact rather than a
/// heuristic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_max_pixels(
    declared_max_pixels: usize,
    min_pixels: usize,
    shape: &VisionShape,
    tile: usize,
    guard: LoadGuard,
    physical: u64,
    committed: u64,
    kv_bytes: u64,
) -> Result<PixelBudget, VisionBudgetTooSmall> {
    if matches!(guard, LoadGuard::Off) {
        return Ok(PixelBudget {
            resolved_max_pixels: declared_max_pixels,
            clamped: false,
        });
    }

    let spoken_for = committed.saturating_add(kv_bytes);
    let available = guard.available(physical, spoken_for);

    if scratch_bytes_for_pixels(shape, declared_max_pixels, tile) <= available {
        return Ok(PixelBudget {
            resolved_max_pixels: declared_max_pixels,
            clamped: false,
        });
    }

    let needs_at_min = scratch_bytes_for_pixels(shape, min_pixels, tile);
    if needs_at_min > available {
        return Err(VisionBudgetTooSmall {
            declared_max_pixels,
            min_pixels,
            needs: needs_at_min,
            available,
            reserve: guard.budget().reserve_bytes,
            physical,
            committed,
            kv_bytes,
        });
    }

    // Invariant going in: `lo` (min_pixels) fits, `hi` (declared_max_pixels)
    // does not -- both checked above.
    let (mut lo, mut hi) = (min_pixels, declared_max_pixels);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if scratch_bytes_for_pixels(shape, mid, tile) <= available {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(PixelBudget {
        resolved_max_pixels: lo,
        clamped: true,
    })
}

/// The entry point every front end calls, over the public
/// [`model_io::VisionConfig`] `RealForwardRunner::vision_config()` already
/// hands out -- [`VisionShape`] itself stays crate-private, so this is what
/// lets `crates/cli`, `crates/ffi` and `crates/server` reach
/// [`resolve_max_pixels`] without a new dependency edge or a change to
/// `real_forward_api.rs`.
impl crate::real_forward::RealForwardRunner {
    /// Resolve `declared_max_pixels` against THIS session's own tower shape,
    /// MLP tile, and the load-guard tier it opened under.
    ///
    /// `guard`/`physical`/`committed`/`kv_bytes` are the same four values a
    /// caller already resolved for `--max-context` (`crates/model-io/
    /// CLAUDE.md`'s `LoadPolicy`/`LoadGuard` entry): the tier MUST be the
    /// one this session actually opened under, never a second, possibly
    /// different one (`crates/ffi/CLAUDE.md` Gotcha 12's rule, applied to a
    /// third call).
    ///
    /// The MLP tile is this session's OWN tower's, if one has already opened
    /// (a test may have overridden it via `set_vision_mlp_tile_rows`), or
    /// the shipped default otherwise -- the tower opens lazily on the first
    /// image, so a caller resolving this before that point cannot ask the
    /// tower directly.
    ///
    /// A `VisionConfig` that cannot resolve to a [`VisionShape`] (structurally
    /// invalid: a zero field, a `hidden_size` not divisible by `num_heads`,
    /// ...) cannot be budgeted against a formula that needs one. Rather than
    /// inventing a bound, this declines to clamp: the SAME install fails the
    /// same way, with the same message, the moment `open_vision_tower`
    /// actually tries to build the tower for a real image, so deferring to
    /// that refusal reports nothing new here.
    pub fn resolve_vision_pixel_budget(
        &self,
        declared_max_pixels: usize,
        min_pixels: usize,
        guard: LoadGuard,
        physical: u64,
        committed: u64,
        kv_bytes: u64,
    ) -> Result<PixelBudget, VisionBudgetTooSmall> {
        let tile = self
            .vision
            .as_ref()
            .map(|v| v.mlp_tile_rows)
            .unwrap_or(crate::vision::scratch::VISION_MLP_TILE_ROWS);
        match VisionShape::resolve(&self.arch.vision) {
            Ok(shape) => resolve_max_pixels(
                declared_max_pixels,
                min_pixels,
                &shape,
                tile,
                guard,
                physical,
                committed,
                kv_bytes,
            ),
            Err(_) => Ok(PixelBudget {
                resolved_max_pixels: declared_max_pixels,
                clamped: false,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::scratch::VISION_MLP_TILE_ROWS;

    /// The real Qwen3.8 tower's shape (`docs/VISION.md`'s "The forward
    /// pass"): depth 27, hidden 1152, intermediate 4304, 16 heads of 72,
    /// out_hidden 5120, patch 16, temporal 2, merge 2, position grid 48x48.
    fn qwen38_tower_shape() -> VisionShape {
        VisionShape {
            depth: 27,
            hidden: 1152,
            intermediate: 4304,
            heads: 16,
            head_dim: 72,
            merge: 2,
            out_hidden: 5120,
            patch_dim: 2 * 16 * 16 * 3,
            pos_rows: 48 * 48,
            pos_side: 48,
            patch_size: 16,
        }
    }

    const GIB: u64 = 1024 * 1024 * 1024;
    /// The real Qwen3.8 tower's declared budget
    /// (`crates/vision-io/CLAUDE.md` Gotcha 6): `size: {shortest_edge:
    /// 65536, longest_edge: 16777216}`.
    const REAL_MIN_PIXELS: usize = 65_536;
    const REAL_MAX_PIXELS: usize = 16_777_216;

    #[test]
    fn off_never_clamps_at_any_machine_size() {
        let shape = qwen38_tower_shape();
        for physical in [0, GIB, 4 * GIB, 36 * GIB] {
            let budget = resolve_max_pixels(
                REAL_MAX_PIXELS,
                REAL_MIN_PIXELS,
                &shape,
                VISION_MLP_TILE_ROWS,
                LoadGuard::Off,
                physical,
                0,
                0,
            )
            .expect("Off never refuses");
            assert_eq!(budget.resolved_max_pixels, REAL_MAX_PIXELS);
            assert!(!budget.clamped);
        }
    }

    /// `LoadGuard::Relaxed`'s reserve alone (`HEADROOM_RESERVE_BYTES`) is a
    /// full 4 GiB, so a "small machine" for this budget has to be bigger
    /// than that or nothing is left for even `min_pixels` (that shape is
    /// `below_min_pixels_refuses_with_the_subtraction_shown`'s case, not
    /// this one). 4.5 GiB leaves 0.5 GiB after the reserve, which is
    /// comfortably over what `min_pixels` costs (~7 MB) and well under the
    /// full ceiling's ~1.2 GB (`docs/VISION.md`'s "Memory" section).
    const SMALL_MACHINE: u64 = 4 * GIB + 512 * 1024 * 1024;

    #[test]
    fn a_small_machine_clamps_the_real_ceiling() {
        let shape = qwen38_tower_shape();
        let budget = resolve_max_pixels(
            REAL_MAX_PIXELS,
            REAL_MIN_PIXELS,
            &shape,
            VISION_MLP_TILE_ROWS,
            LoadGuard::Relaxed,
            SMALL_MACHINE,
            0,
            0,
        )
        .expect("min_pixels fits a 4.5 GiB machine");
        assert!(budget.clamped);
        assert!(budget.resolved_max_pixels < REAL_MAX_PIXELS);
        assert!(budget.resolved_max_pixels >= REAL_MIN_PIXELS);
        // The resolved ceiling must itself fit the same budget it was
        // solved against -- the whole point of the search.
        let available = LoadGuard::Relaxed.available(SMALL_MACHINE, 0);
        assert!(
            scratch_bytes_for_pixels(&shape, budget.resolved_max_pixels, VISION_MLP_TILE_ROWS)
                <= available
        );
    }

    /// `CLAUDE.local.md`'s actual hardware: an M4 Max, 36 GB unified memory.
    /// Whether the real tower's declared ceiling fits there under `Relaxed`
    /// with nothing else committed is a real finding, reported either way
    /// rather than assumed.
    #[test]
    fn a_36_gib_machine_and_the_real_tower_ceiling() {
        let shape = qwen38_tower_shape();
        let budget = resolve_max_pixels(
            REAL_MAX_PIXELS,
            REAL_MIN_PIXELS,
            &shape,
            VISION_MLP_TILE_ROWS,
            LoadGuard::Relaxed,
            36 * GIB,
            0,
            0,
        )
        .expect("min_pixels fits a 36 GiB machine");
        // `Relaxed`'s pool is a quarter of (36 GiB - HEADROOM_RESERVE_BYTES)
        // with nothing else committed, which is several GiB -- comfortably
        // past the ~1.3 GB the full ceiling costs. Asserted rather than
        // merely commented, so a change to either constant is caught here
        // instead of silently flipping this test's own premise.
        assert!(
            !budget.clamped,
            "expected the real tower's declared ceiling to fit a 36 GiB machine under Relaxed \
             with nothing else committed; it did not (resolved {} of {})",
            budget.resolved_max_pixels, REAL_MAX_PIXELS
        );
        assert_eq!(budget.resolved_max_pixels, REAL_MAX_PIXELS);
    }

    #[test]
    fn below_min_pixels_refuses_with_the_subtraction_shown() {
        let shape = qwen38_tower_shape();
        // A machine too small even for the checkpoint's own floor: pin
        // `committed` and `kv_bytes` high enough that nothing is left.
        let err = resolve_max_pixels(
            REAL_MAX_PIXELS,
            REAL_MIN_PIXELS,
            &shape,
            VISION_MLP_TILE_ROWS,
            LoadGuard::Relaxed,
            4 * GIB,
            4 * GIB, // already fully committed
            0,
        )
        .expect_err("min_pixels cannot fit a fully-committed machine");
        assert_eq!(err.declared_max_pixels, REAL_MAX_PIXELS);
        assert_eq!(err.min_pixels, REAL_MIN_PIXELS);
        assert!(err.needs > 0);
        let message = err.to_string();
        // The message must show the actual byte numbers, not just that it
        // errored.
        assert!(message.contains(&REAL_MAX_PIXELS.to_string()));
        assert!(message.contains(&REAL_MIN_PIXELS.to_string()));
        assert!(message.contains("physical"));
        assert!(message.contains("GiB"));
    }

    #[test]
    fn a_stricter_tier_clamps_at_least_as_hard() {
        let shape = qwen38_tower_shape();
        let relaxed = resolve_max_pixels(
            REAL_MAX_PIXELS,
            REAL_MIN_PIXELS,
            &shape,
            VISION_MLP_TILE_ROWS,
            LoadGuard::Relaxed,
            SMALL_MACHINE,
            0,
            0,
        )
        .expect("relaxed resolves on a 4.5 GiB machine");
        assert!(
            relaxed.clamped,
            "this test needs Relaxed itself to clamp, or the comparison below is vacuous"
        );
        // `Strict`'s reserve is 3x `Relaxed`'s (12 GiB against 4 GiB), which
        // on a machine this small can legitimately leave NO room at all --
        // refusing outright is at least as strict a clamp as any resolved
        // value, so that arm satisfies the monotonicity claim too rather
        // than being treated as a test failure.
        if let Ok(strict) = resolve_max_pixels(
            REAL_MAX_PIXELS,
            REAL_MIN_PIXELS,
            &shape,
            VISION_MLP_TILE_ROWS,
            LoadGuard::Strict,
            SMALL_MACHINE,
            0,
            0,
        ) {
            assert!(strict.resolved_max_pixels <= relaxed.resolved_max_pixels);
        }
    }

    /// AGENTS.md's mutation-check rule: drop the tile term from
    /// `scratch_bytes_for_pixels`'s shared formula (simulated here by
    /// calling the underlying `scratch::scratch_bytes` at `tile = usize::MAX`,
    /// which removes the row-tiling cap Part B1 added) and confirm the
    /// machine-size test that should catch a wrong cap actually reddens.
    /// This proves the clamp test is sensitive to the formula it is
    /// supposed to be checking, rather than passing for an unrelated reason.
    #[test]
    fn the_small_machine_test_is_sensitive_to_the_scratch_formula() {
        let shape = qwen38_tower_shape();
        // At `tile = seq` (i.e. no row tiling at all, Part B1 reverted),
        // `h1`'s term is much larger at the full 4096x4096 page -- the same
        // "-555 MB term at the extreme page" `docs/VISION.md`'s "Memory"
        // section names B1 as closing. A budget solved without the cap
        // would resolve to a LARGER ceiling than one solved with it, on the
        // same machine.
        let uncapped_tile = usize::MAX;
        let with_cap = resolve_max_pixels(
            REAL_MAX_PIXELS,
            REAL_MIN_PIXELS,
            &shape,
            VISION_MLP_TILE_ROWS,
            LoadGuard::Relaxed,
            SMALL_MACHINE,
            0,
            0,
        )
        .expect("min_pixels fits");
        let without_cap = resolve_max_pixels(
            REAL_MAX_PIXELS,
            REAL_MIN_PIXELS,
            &shape,
            uncapped_tile,
            LoadGuard::Relaxed,
            SMALL_MACHINE,
            0,
            0,
        )
        .expect("min_pixels fits");
        assert!(
            without_cap.resolved_max_pixels != with_cap.resolved_max_pixels,
            "the machine-size test must be sensitive to the tile term: dropping it moved \
             nothing, which means the test cannot see a broken formula"
        );
    }
}
