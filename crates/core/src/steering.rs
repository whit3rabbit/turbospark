//! The directional-steering edit's mode, shared by the CPU reference
//! (`turbospark_compute::steering`) and the Metal dispatch
//! (`turbospark_gpu::encode_steer_direction`).
//!
//! It lives in this leaf crate because those two are the only definitions of
//! what the kernel computes and neither depends on the other:
//! `crates/gpu` carries `turbospark-compute` as a DEV-dependency only, so a
//! mode enum declared in `compute` would be unnameable from the dispatch
//! module it selects. Both crates depend on `foundation`, so one declaration
//! here is what keeps the parity test comparing two spellings of one contract
//! rather than two enums that happen to agree today.

/// Which edit [`crate::steering`]'s three-mode kernel applies to a residual
/// stream row, given a direction `d` and its precomputed `1 / ||d||`.
///
/// All three are one dot product plus one strided write, and differ only in
/// what they do with the coefficient. Writing `c = d . x` for the raw dot
/// product and `c_hat = c / ||d||` for the coefficient along the unit
/// direction:
///
/// | mode | operation |
/// |---|---|
/// | [`Ablate`](Self::Ablate) | `x -= alpha * c_hat * d_hat` |
/// | [`Add`](Self::Add) | `x += alpha * d` |
/// | [`Clamp`](Self::Clamp) | `x += (target - c_hat) * d_hat` |
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
/// pinned by `steering_mode_codes_match_the_shader` in
/// `crates/gpu/tests/utility_and_pass.rs`, because a reordering here would
/// silently swap two edits that both produce fluent output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SteeringMode {
    /// Project the direction out of the stream. This is abliteration's
    /// operation, and at `alpha == 1.0` on a unit direction it is exactly the
    /// `x' = x - r_hat r_hat^T x` that the weight edit
    /// `W' = W - r_hat r_hat^T W` precomputes.
    ///
    /// The default because it is the only mode that cannot overflow: it
    /// removes a component of `x` and therefore never increases `|x|`, where
    /// the other two add to a stream stored in FP16 (see the module docs on
    /// [`crate::steering`]).
    #[default]
    Ablate,
    /// Add the direction, scaled. Turner et al.'s activation addition, and
    /// what llama.cpp's `--control-vector-scaled` applies.
    Add,
    /// Pin the coefficient along the unit direction to `target`, whatever it
    /// was. Feature clamping: the edit that holds a concept on regardless of
    /// context, rather than nudging it.
    Clamp,
}

impl SteeringMode {
    /// The value the MSL kernel's `mode` uniform switches on.
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::Ablate => 0,
            Self::Add => 1,
            Self::Clamp => 2,
        }
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
        }
    }

    /// Whether this mode can increase `|x|`, and therefore whether a large
    /// `alpha` can push a residual stream stored in FP16 past 65,504.
    ///
    /// [`Self::Ablate`] removes a component and cannot; the other two can.
    /// Callers that dispatch an unbounded `alpha` on a growing mode should
    /// expect the overflow to arrive as `inf` and then as NaN, which reads as
    /// a PERFECT score on any rank or top-k instrument (AGENTS.md Gotcha 59).
    pub const fn can_grow(self) -> bool {
        match self {
            Self::Ablate => false,
            Self::Add | Self::Clamp => true,
        }
    }
}
