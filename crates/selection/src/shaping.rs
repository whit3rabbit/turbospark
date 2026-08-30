//! Shaping configuration for candidate selection, validated at construction.
//!
//! The configuration is a plain, caller-constructed value: it carries no
//! hidden singleton and no state between selections, so a fresh
//! configuration can be built per selection or reused freely.

/// A distinguishable, descriptive configuration or selection-input
/// validation error. Shared by [`ShapingConfig::new`] and by
/// [`crate::choose::select`] for the destination-decided input guards
/// (a non-finite score entry, an empty candidate domain).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionError {
    /// A human-readable reason. Wording is not contractual; only the fact
    /// that a distinguishable error was produced is.
    pub reason: String,
}

impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid selection input: {}", self.reason)
    }
}

impl std::error::Error for SelectionError {}

impl SelectionError {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

/// Documented default repetition penalty (identity, no attenuation).
pub const DEFAULT_REPETITION_PENALTY: f64 = 1.0;

/// Inclusive upper bound on the rank-based truncation count, aligned with
/// the argument-translation contract in this same slice so the two
/// surfaces cannot disagree about which counts are accepted.
pub const MAX_TOP_K: u32 = 256;

/// A plain, caller-constructed shaping configuration. Carries no state
/// between selections.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapingConfig {
    temperature: f64,
    top_k: u32,
    top_p: Option<f64>,
    repetition_penalty: f64,
    seed: Option<u64>,
    /// `0.0` is the identity value (no attenuation), matching OpenAI's own
    /// default. Set through [`Self::with_presence_penalty`] rather than
    /// [`Self::new`] -- see that constructor's doc for why.
    presence_penalty: f64,
    frequency_penalty: f64,
    /// `None` means min-p truncation is disabled. Set through
    /// [`Self::with_min_p`].
    min_p: Option<f64>,
}

impl ShapingConfig {
    /// Build and validate a shaping configuration.
    ///
    /// `top_k` of zero means rank-based truncation is disabled. `top_p` of
    /// `None` means probability-mass truncation is disabled.
    ///
    /// Rejected at construction time, with a distinguishable descriptive
    /// error and no selection attempted: a non-finite or negative
    /// temperature; a rank-based count outside `0..=MAX_TOP_K`; a
    /// cumulative-probability threshold that is non-finite, at or below
    /// zero, or above one; a non-finite or non-positive repetition penalty;
    /// and a sub-one threshold at a positive temperature with no
    /// rank-based count requested.
    pub fn new(
        temperature: f64,
        top_k: u32,
        top_p: Option<f64>,
        repetition_penalty: f64,
        seed: Option<u64>,
    ) -> Result<Self, SelectionError> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(SelectionError::new(format!(
                "temperature must be finite and non-negative, got {temperature}"
            )));
        }
        if top_k > MAX_TOP_K {
            return Err(SelectionError::new(format!(
                "top_k must be between 0 and {MAX_TOP_K}, got {top_k}"
            )));
        }
        if let Some(p) = top_p {
            if !p.is_finite() || p <= 0.0 || p > 1.0 {
                return Err(SelectionError::new(format!(
                    "top_p must be finite and in (0, 1], got {p}"
                )));
            }
        }
        if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
            return Err(SelectionError::new(format!(
                "repetition_penalty must be finite and greater than zero, got {repetition_penalty}"
            )));
        }
        if let Some(p) = top_p {
            if p < 1.0 && temperature > 0.0 && top_k == 0 {
                return Err(SelectionError::new(
                    "top_p below 1.0 at a positive temperature requires top_k to be enabled",
                ));
            }
        }
        Ok(Self {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: None,
        })
    }

    /// Set the presence penalty: a flat per-token subtraction applied ONCE
    /// per distinct candidate that appears anywhere in the GENERATED suffix
    /// of history (never the prompt), regardless of how many times it
    /// repeats there. `0.0` (the default from [`Self::new`]) is the
    /// identity value.
    ///
    /// Rejected at call time, with no change to `self`, for a non-finite
    /// value or one outside `[-2, 2]` (OpenAI's own documented range).
    pub fn with_presence_penalty(mut self, presence_penalty: f64) -> Result<Self, SelectionError> {
        if !presence_penalty.is_finite() || !(-2.0..=2.0).contains(&presence_penalty) {
            return Err(SelectionError::new(format!(
                "presence_penalty must be finite and in [-2, 2], got {presence_penalty}"
            )));
        }
        self.presence_penalty = presence_penalty;
        Ok(self)
    }

    /// Set the frequency penalty: a per-token subtraction SCALED by how many
    /// times a candidate appears in the GENERATED suffix of history (never
    /// the prompt). `0.0` (the default from [`Self::new`]) is the identity
    /// value.
    ///
    /// Rejected at call time, with no change to `self`, for a non-finite
    /// value or one outside `[-2, 2]` (OpenAI's own documented range).
    pub fn with_frequency_penalty(
        mut self,
        frequency_penalty: f64,
    ) -> Result<Self, SelectionError> {
        if !frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&frequency_penalty) {
            return Err(SelectionError::new(format!(
                "frequency_penalty must be finite and in [-2, 2], got {frequency_penalty}"
            )));
        }
        self.frequency_penalty = frequency_penalty;
        Ok(self)
    }

    /// Set the min-p threshold: a candidate survives truncation only if its
    /// unnormalized score is at least `min_p` times the ranked prefix's own
    /// top score. `0.0` (the default from [`Self::new`]) means DISABLED --
    /// stored as `None` rather than as a threshold of zero, since every
    /// score would trivially clear that bar and the two are observably the
    /// same thing.
    ///
    /// Rejected at call time, with no change to `self`, for a non-finite
    /// value or one outside `[0, 1)`. `1.0` is refused rather than accepted
    /// as "keep only the single most probable candidate": a caller asking
    /// for at least some FRACTION of the top probability almost certainly
    /// did not mean to collapse every request to greedy, and the greedy
    /// behavior is already reachable through `temperature: 0.0`.
    pub fn with_min_p(mut self, min_p: f64) -> Result<Self, SelectionError> {
        if !min_p.is_finite() || !(0.0..1.0).contains(&min_p) {
            return Err(SelectionError::new(format!(
                "min_p must be finite and in [0, 1), got {min_p}"
            )));
        }
        self.min_p = if min_p == 0.0 { None } else { Some(min_p) };
        Ok(self)
    }

    /// The configured sampling temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// The configured rank-based truncation count; zero means disabled.
    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    /// The configured cumulative-probability truncation threshold; `None`
    /// means disabled.
    pub fn top_p(&self) -> Option<f64> {
        self.top_p
    }

    /// The configured repetition penalty factor.
    pub fn repetition_penalty(&self) -> f64 {
        self.repetition_penalty
    }

    /// The configured determinism seed, if any.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Whether this configuration selects deterministically (temperature at
    /// its deterministic value).
    pub fn is_deterministic(&self) -> bool {
        self.temperature == 0.0
    }

    /// The configured presence penalty; `0.0` is the identity value.
    pub fn presence_penalty(&self) -> f64 {
        self.presence_penalty
    }

    /// The configured frequency penalty; `0.0` is the identity value.
    pub fn frequency_penalty(&self) -> f64 {
        self.frequency_penalty
    }

    /// The configured min-p threshold; `None` means disabled.
    pub fn min_p(&self) -> Option<f64> {
        self.min_p
    }
}
