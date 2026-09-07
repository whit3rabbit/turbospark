//! The directional-steering edit's mode, shared by the CPU reference
//! (`turbospark_compute::steering`) and the Metal dispatch
//! (`turbospark_gpu::encode_steer_direction`).
//!
//! It lives in this leaf crate because those two are the only definitions of
//! what the kernel computes and both depend on `foundation`: one declaration
//! here is what keeps the parity test comparing two spellings of one
//! contract rather than two enums that happen to agree today.

/// Which edit [`crate::steering`]'s four-mode kernel applies to a residual
/// stream row, given a direction `d` and its precomputed `1 / ||d||`.
///
/// All four share one reduction pass and one strided write, and differ only
/// in what they do with the coefficient. Writing `c = d . x` for the raw dot
/// product and `c_hat = c / ||d||` for the coefficient along the unit
/// direction:
///
/// | mode | operation |
/// |---|---|
/// | [`Ablate`](Self::Ablate) | `x -= alpha * c_hat * d_hat` |
/// | [`Add`](Self::Add) | `x += alpha * d` |
/// | [`Clamp`](Self::Clamp) | `x += (target - c_hat) * d_hat` |
/// | [`Renorm`](Self::Renorm) | [`Ablate`](Self::Ablate), then rescale the row back to its original `\|\|x\|\|` |
///
/// [`Renorm`](Self::Renorm) is the only one that reads anything about `x`
/// beyond its coefficient, and it costs nothing extra to: `||x||^2`
/// accumulates in the same loop that computes `c`, from a value already in
/// register, and the POST-edit norm is analytic rather than a second pass
/// (see `turbospark_compute::steering`).
///
/// **`Add` is the only one that reads `d`'s magnitude**, and that asymmetry
/// is the file format's rather than a choice: a llama.cpp control vector
/// carries its strength IN the vector, so normalizing one at load would
/// silently rescale every published vector set. `Ablate` and `Clamp` are
/// defined against the unit direction because a projection that depended on
/// `||d||` would change meaning when a direction was re-extracted from a
/// larger corpus.
///
/// The discriminants are the wire values the MSL kernel switches on. They are
/// pinned by `steering_mode_codes_are_pinned` in
/// `crates/gpu/tests/utility_and_pass.rs`, because a reordering here would
/// silently swap two edits that both produce fluent output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum SteeringMode {
    /// Project the direction out of the stream. This is abliteration's
    /// operation, and at `alpha == 1.0` on a unit direction it is exactly the
    /// `x' = x - r_hat r_hat^T x` that the weight edit
    /// `W' = W - r_hat r_hat^T W` precomputes.
    ///
    /// The default because it cannot overflow: it removes a component of `x`
    /// and therefore never increases `|x|`, where [`Add`](Self::Add) and
    /// [`Clamp`](Self::Clamp) add to a stream stored in FP16 (see the module
    /// docs on [`crate::steering`]). [`Renorm`](Self::Renorm) cannot overflow
    /// either, for a different reason -- see [`Self::can_grow`].
    #[default]
    Ablate = 0,
    /// Add the direction, scaled. Turner et al.'s activation addition, and
    /// what llama.cpp's `--control-vector-scaled` applies.
    Add = 1,
    /// Pin the coefficient along the unit direction to `target`, whatever it
    /// was. Feature clamping: the edit that holds a concept on regardless of
    /// context, rather than nudging it.
    Clamp = 2,
    /// Norm-preserving projection: [`Ablate`](Self::Ablate), then rescale the
    /// row back to the magnitude it had before.
    ///
    /// It attacks the collapse `docs/OBLITERATION.md` records from the other
    /// side. Ablating a whole direction at every layer damages what the
    /// output head reads, and the ceiling that page derives AVOIDS that
    /// damage by bounding `alpha`; this REPAIRS it, so full ablation stays
    /// available. Whether that actually widens the usable band is a measured
    /// question rather than a claim -- see the alpha sweep on that page.
    ///
    /// The mechanism it can plausibly repair is NOT "the head reads a smaller
    /// vector", because the final RMS norm is scale-invariant and would undo
    /// that on its own. It is the RESIDUAL ADD, which is not: shrinking `x`
    /// at every layer amplifies each subsequent sublayer's relative
    /// contribution to the stream.
    Renorm = 3,
}

/// Every spelling [`SteeringMode::parse`] accepts, in declaration order.
///
/// Exists so a front end's rejection message can be SPELLED from the accepted
/// set rather than recalled from it. `turbospark-server`'s `--steering-mode`
/// error named three modes for a release after [`SteeringMode::Renorm`]
/// landed, which told a caller who had misspelled the fourth that it did not
/// exist -- the count-that-rots shape, on a string no test was reading.
/// `the_mode_names_are_exactly_what_parse_accepts` is what keeps this and
/// `parse` from drifting.
pub const STEERING_MODE_NAMES: &[&str] = &["ablate", "add", "clamp", "renorm"];

impl SteeringMode {
    /// The value the MSL kernel's `mode` uniform switches on. This is the
    /// variant's own `#[repr(u32)]` discriminant, so the wire value sits on
    /// the variant a reader is already looking at.
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Parses the spelling the CLI, the server and the direction file all use.
    ///
    /// `None` is a rejection and never a fallback to [`Self::Ablate`]: a
    /// caller who asked for one edit and silently got another would measure
    /// the wrong model and report it as the right one.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "ablate" => Some(Self::Ablate),
            "add" => Some(Self::Add),
            "clamp" => Some(Self::Clamp),
            "renorm" => Some(Self::Renorm),
            _ => None,
        }
    }

    /// The spelling [`Self::parse`] accepts, for diagnostics and for the
    /// startup line.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ablate => "ablate",
            Self::Add => "add",
            Self::Clamp => "clamp",
            Self::Renorm => "renorm",
        }
    }

    /// Whether this mode can increase `|x|`, and therefore whether a large
    /// `alpha` can push a residual stream stored in FP16 past 65,504.
    ///
    /// [`Self::Ablate`] removes a component and cannot. [`Self::Renorm`]
    /// cannot either, and for a reason worth stating rather than inferring:
    /// it restores the row to the L2 norm it already had, so no element can
    /// exceed `||x||`, and `||x||` was representable before the edit because
    /// the row was. Its per-element values DO grow -- the rescale multiplies
    /// by a factor at or above 1 -- but they are bounded by a magnitude the
    /// stream already carried. [`Self::Add`] and [`Self::Clamp`] have no such
    /// bound.
    ///
    /// Callers that dispatch an unbounded `alpha` on a growing mode should
    /// expect the overflow to arrive as `inf` and then as NaN, which reads as
    /// a PERFECT score on any rank or top-k instrument (AGENTS.md Gotcha 59).
    pub const fn can_grow(self) -> bool {
        match self {
            Self::Ablate | Self::Renorm => false,
            Self::Add | Self::Clamp => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [SteeringMode; 4] = [
        SteeringMode::Ablate,
        SteeringMode::Add,
        SteeringMode::Clamp,
        SteeringMode::Renorm,
    ];

    /// The discriminants are the wire values the MSL kernel switches on, so a
    /// reorder here silently swaps two edits that both decode fluently.
    /// `crates/gpu`'s `steering_mode_codes_are_pinned` pins them against
    /// the shader's own constants; this pins them on the side that declares
    /// them, so a reorder reddens without a Metal device.
    #[test]
    fn steering_mode_codes_are_pinned() {
        assert_eq!(SteeringMode::Ablate.as_u32(), 0);
        assert_eq!(SteeringMode::Add.as_u32(), 1);
        assert_eq!(SteeringMode::Clamp.as_u32(), 2);
        assert_eq!(SteeringMode::Renorm.as_u32(), 3);
    }

    /// [`STEERING_MODE_NAMES`] exists to be printed in a rejection message, so
    /// it is worth nothing unless it is exactly what `parse` accepts. Checked
    /// in BOTH directions: every name parses, and every mode's own
    /// `as_str` is in the list. One direction alone permits a list that has
    /// grown a name `parse` refuses, or lost one it accepts.
    #[test]
    fn the_mode_names_are_exactly_what_parse_accepts() {
        for name in STEERING_MODE_NAMES {
            assert!(
                SteeringMode::parse(name).is_some(),
                "{name} is advertised and does not parse"
            );
        }
        for mode in ALL {
            assert!(
                STEERING_MODE_NAMES.contains(&mode.as_str()),
                "{mode:?} parses as {} and is not advertised",
                mode.as_str()
            );
        }
        assert_eq!(STEERING_MODE_NAMES.len(), ALL.len());
    }

    /// A rejection is never a fallback to the default: a caller who asked for
    /// one edit and silently got another would measure the wrong model.
    #[test]
    fn an_unknown_mode_is_none_rather_than_the_default() {
        assert_eq!(SteeringMode::parse("Ablate"), None);
        assert_eq!(SteeringMode::parse("renrom"), None);
        assert_eq!(SteeringMode::parse(""), None);
    }
}
