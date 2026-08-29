//! Power policy for the decode loop (ROADMAP Phase P2): the profile set,
//! the thermal ladder, and the [`RateControl`] the loop reads.
//!
//! Everything here is pure except the two probes, which are cfg-paired
//! against `crates/gpu` on macOS and constant elsewhere -- this crate is
//! `#![forbid(unsafe_code)]` and portable, so the message sends live over
//! there and cross as an integer and a `bool`.
//!
//! DELIBERATELY NOT WIRED: read-pool QoS. ROADMAP Phase P2's text pairs
//! profiles with QoS, but `MFERENCE_READ_QOS=utility` was measured a null
//! result on AC and a loss on battery (`docs/POWER_BASELINE.md`), so a
//! profile that set it would ship a measured regression. Profiles map to
//! rate caps and thermal stepping only.

/// The "reading speed" preset: fast enough to read along with, slow
/// enough that the GPU idles between tokens. The `efficiency` profile's
/// cap.
pub const READING_SPEED_TOK_PER_SEC: f64 = 10.0;

/// Ceiling imposed once thermal pressure reaches `serious`.
pub const SERIOUS_TOK_PER_SEC: f64 = 10.0;

/// Ceiling imposed once thermal pressure reaches `critical`.
pub const CRITICAL_TOK_PER_SEC: f64 = 5.0;

/// macOS thermal pressure, as `NSProcessInfo.thermalState` reports it.
/// Constructed from the raw `NSInteger` in one place so the meaning of a
/// level is stated once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThermalLevel {
    Nominal,
    Fair,
    Serious,
    Critical,
}

impl ThermalLevel {
    /// Maps 0..=3 directly. Out-of-range values clamp toward the SAFE
    /// side in each direction: a future OS level above 3 means hotter
    /// still, and a negative reading is not a pressure report at all.
    pub fn from_raw(raw: i64) -> Self {
        match raw {
            1 => Self::Fair,
            2 => Self::Serious,
            n if n >= 3 => Self::Critical,
            _ => Self::Nominal,
        }
    }
}

/// macOS memory pressure, as `kern.memorystatus_vm_pressure_level` reports
/// it.
///
/// **A SIBLING OF [`ThermalLevel`] AND NOT A SECOND SPELLING OF IT.** The two
/// answer different questions and can move independently: a machine can be
/// cool and short of memory (another process just opened a model) or hot and
/// comfortable. [`stepped_cap`] therefore takes both and applies whichever is
/// worse, rather than one standing in for the other.
///
/// Three levels rather than four, because that is what the kernel publishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum MemoryPressure {
    /// Nothing to report. **Also what an ABSENT probe answers**, which is the
    /// right reading: a reading nobody could take must not pace a decode
    /// loop. That is the same rule `physical_memory()` follows when it
    /// answers 0 off macOS -- an unknown machine imposes no bound.
    #[default]
    Normal,
    /// The kernel is asking processes to release memory.
    Warn,
    /// The kernel is about to start killing them.
    Critical,
}

impl MemoryPressure {
    /// Maps the sysctl's 1/2/4 ladder. **The values are a BITMASK and not a
    /// sequence**, which is why this is a match on three literals rather than
    /// a range: there is no level 3, and reading `>= 2` as "warn or worse"
    /// happens to work today only because 4 is the sole value above 2.
    ///
    /// Everything unrecognized -- including the 0 an unavailable probe
    /// returns -- reads as [`Self::Normal`]. That is the safe direction here
    /// and the OPPOSITE of [`ThermalLevel::from_raw`]'s clamp, deliberately:
    /// an unknown thermal value above the range means hotter still, while an
    /// unknown memory value means the kernel did not answer, and capping a
    /// decode loop on the strength of a failed syscall would be pacing on no
    /// information.
    pub fn from_raw(raw: i64) -> Self {
        match raw {
            2 => Self::Warn,
            4 => Self::Critical,
            _ => Self::Normal,
        }
    }
}

/// The user-facing power profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PowerProfile {
    /// No rate cap and no thermal stepping: exactly the behavior every
    /// published benchmark in this repo was measured under.
    #[default]
    Performance,
    /// Uncapped, but steps down under thermal pressure.
    Balanced,
    /// Capped at reading speed, and steps down further under pressure.
    Efficiency,
}

impl PowerProfile {
    /// The CLI/server spelling. `None` for anything else, so a bad value
    /// is the caller's diagnostic to raise.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "performance" => Some(Self::Performance),
            "balanced" => Some(Self::Balanced),
            "efficiency" => Some(Self::Efficiency),
            _ => None,
        }
    }

    /// The inverse of [`Self::parse`], for echoing a resolved request.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Performance => "performance",
            Self::Balanced => "balanced",
            Self::Efficiency => "efficiency",
        }
    }

    /// The profile's own cap, before any explicit override.
    pub fn base_cap(self) -> Option<f64> {
        match self {
            Self::Efficiency => Some(READING_SPEED_TOK_PER_SEC),
            Self::Performance | Self::Balanced => None,
        }
    }

    /// Whether the loop polls thermal pressure at all.
    pub fn thermal_stepping(self) -> bool {
        !matches!(self, Self::Performance)
    }
}

/// The thermal ladder: nominal and fair leave the base cap alone, serious
/// and critical impose a ceiling. An UNCAPPED base still gets capped under
/// pressure, which is the whole point of `balanced`.
///
/// Kept as its own function rather than folded into [`stepped_cap`] because
/// it is what the five cases below assert and what every published power row
/// was measured under.
pub fn thermal_cap(base: Option<f64>, level: ThermalLevel) -> Option<f64> {
    let ceiling = match level {
        ThermalLevel::Nominal | ThermalLevel::Fair => return base,
        ThermalLevel::Serious => SERIOUS_TOK_PER_SEC,
        ThermalLevel::Critical => CRITICAL_TOK_PER_SEC,
    };
    Some(base.map_or(ceiling, |base| base.min(ceiling)))
}

/// The memory ladder. `Warn` is where the kernel starts ASKING processes to
/// release memory and `Critical` is where it starts killing them, so the two
/// map onto the same two ceilings thermal pressure uses rather than onto
/// invented ones -- there is no measurement here that would justify a third
/// pair of numbers, and a cap is a cap whatever drove it.
pub fn memory_cap(base: Option<f64>, level: MemoryPressure) -> Option<f64> {
    let ceiling = match level {
        MemoryPressure::Normal => return base,
        MemoryPressure::Warn => SERIOUS_TOK_PER_SEC,
        MemoryPressure::Critical => CRITICAL_TOK_PER_SEC,
    };
    Some(base.map_or(ceiling, |base| base.min(ceiling)))
}

/// Both ladders, whichever binds harder.
///
/// **THE MINIMUM AND NOT A PRECEDENCE.** The two signals are independent --
/// a machine can be cool and short of memory, or hot and comfortable -- so
/// asking which one "wins" is the wrong question: each states a ceiling that
/// is true on its own terms, and honouring the looser of two true ceilings
/// would ignore one of them.
pub fn stepped_cap(
    base: Option<f64>,
    thermal: ThermalLevel,
    memory: MemoryPressure,
) -> Option<f64> {
    match (thermal_cap(base, thermal), memory_cap(base, memory)) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// Current thermal pressure. Always `Nominal` off macOS.
#[cfg(target_os = "macos")]
pub fn thermal_level() -> ThermalLevel {
    ThermalLevel::from_raw(gpu::thermal_state_raw())
}

/// Current memory pressure. Always `Normal` off macOS, where there is no
/// probe -- absence of a reading, never a reading of "fine".
#[cfg(target_os = "macos")]
pub fn memory_pressure() -> MemoryPressure {
    MemoryPressure::from_raw(gpu::memory_pressure_raw())
}

/// Current memory pressure. Always `Normal` off macOS.
#[cfg(not(target_os = "macos"))]
pub fn memory_pressure() -> MemoryPressure {
    MemoryPressure::Normal
}

/// Current thermal pressure. Always `Nominal` off macOS.
#[cfg(not(target_os = "macos"))]
pub fn thermal_level() -> ThermalLevel {
    ThermalLevel::Nominal
}

/// Whether the OS reports Low Power Mode. Always `false` off macOS.
#[cfg(target_os = "macos")]
pub fn low_power_mode_enabled() -> bool {
    gpu::low_power_mode_enabled()
}

/// Whether the OS reports Low Power Mode. Always `false` off macOS.
#[cfg(not(target_os = "macos"))]
pub fn low_power_mode_enabled() -> bool {
    false
}

/// Installed physical memory in bytes, for the two sizing policies
/// ([`crate::ExpertCacheSlots`] and [`crate::MaxContext`]).
///
/// Here rather than in either policy module for the reason the two probes
/// above are: those modules are portable and take the machine as a
/// PARAMETER, so their whole test suites run anywhere. This is the one
/// place that asks the OS.
#[cfg(target_os = "macos")]
pub fn physical_memory() -> u64 {
    gpu::physical_memory()
}

/// Installed physical memory in bytes.
///
/// Off macOS there is no probe here and no `RealForwardRunner` to size
/// anything for, so this answers 0 -- which makes every budget empty and
/// every `Auto` fall back to its floor, rather than inventing a machine.
#[cfg(not(target_os = "macos"))]
pub fn physical_memory() -> u64 {
    0
}

/// What the Metal device says it will hold, and its name.
///
/// **Advisory only, and deliberately not what anything budgets from.** Both
/// sizing policies take [`physical_memory`], so this exists to report the
/// gap rather than to close it: on a unified-memory Mac the working set is
/// around 75% of installed RAM, and a machine where it sits below
/// `physical - CONTEXT_RESERVE_BYTES` is one whose binding constraint is the
/// driver rather than the arithmetic. Sizing from it instead would make a
/// recommendation disagree with the `open()` that follows it.
#[cfg(target_os = "macos")]
pub fn recommended_max_working_set() -> Option<(u64, String)> {
    gpu::recommended_max_working_set()
}

/// What the GPU says it will hold, and its name.
///
/// Off macOS there is no device to ask, so this answers `None` -- absent
/// rather than zero, since zero is a claim about a machine and this is the
/// absence of a probe.
#[cfg(not(target_os = "macos"))]
pub fn recommended_max_working_set() -> Option<(u64, String)> {
    None
}

/// An explicit profile wins; otherwise Low Power Mode selects
/// `efficiency` and its absence selects `performance`.
///
/// Call this ONCE, when a session or server opens -- never per token. The
/// LPM probe is a message send, and more importantly a profile that could
/// change mid-generation would make the same prompt decode at different
/// rates for reasons the caller never asked about.
pub fn resolve_profile(explicit: Option<PowerProfile>) -> PowerProfile {
    match explicit {
        Some(profile) => profile,
        None if low_power_mode_enabled() => PowerProfile::Efficiency,
        None => PowerProfile::Performance,
    }
}

/// Builds the loop's knob from a resolved profile plus an optional
/// explicit cap. The explicit cap OVERRIDES the profile's own; stepping
/// follows the profile either way, so `--power-profile performance
/// --max-tokens-per-sec 8` paces at 8 and never steps down.
pub fn rate_control_for(profile: PowerProfile, explicit_cap: Option<f64>) -> RateControl {
    RateControl {
        max_tokens_per_sec: explicit_cap.or_else(|| profile.base_cap()),
        thermal_probe: profile
            .thermal_stepping()
            .then_some(thermal_level as fn() -> ThermalLevel),
        // Paired with the thermal probe rather than given its own switch:
        // both are "step down when the machine is under strain", the profile
        // already answers whether this run does that, and a profile that
        // stepped for heat and not for memory would be a fourth policy
        // nobody asked for.
        memory_probe: profile
            .thermal_stepping()
            .then_some(memory_pressure as fn() -> MemoryPressure),
    }
}

/// What the decode loop reads. Both fields `None` (the [`Default`]) is
/// the uncapped path, and the loop then executes exactly the statement
/// sequence it did before Phase P2 existed.
///
/// The probe is a plain `fn` pointer rather than a trait object or a
/// generic parameter: it keeps this `Copy`, keeps `run_raw_completion`'s
/// signature unchanged for a feature that is off by default, and lets a
/// test inject a level sequence without a platform.
#[derive(Debug, Clone, Copy, Default)]
pub struct RateControl {
    /// Decode tokens per second, or `None` for uncapped.
    pub max_tokens_per_sec: Option<f64>,
    /// Polled every `THERMAL_POLL_TOKENS` tokens, or `None` for no
    /// stepping.
    pub thermal_probe: Option<fn() -> ThermalLevel>,
    /// The same, for memory pressure. Polled in the SAME block as
    /// `thermal_probe`, so the watcher costs no extra call site and no extra
    /// sleep.
    pub memory_probe: Option<fn() -> MemoryPressure>,
}

impl RateControl {
    /// Whether the loop needs a pacer at all.
    pub fn is_active(&self) -> bool {
        self.max_tokens_per_sec.is_some()
            || self.thermal_probe.is_some()
            || self.memory_probe.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_thermal_levels_map_and_clamp_toward_the_safe_side() {
        assert_eq!(ThermalLevel::from_raw(0), ThermalLevel::Nominal);
        assert_eq!(ThermalLevel::from_raw(1), ThermalLevel::Fair);
        assert_eq!(ThermalLevel::from_raw(2), ThermalLevel::Serious);
        assert_eq!(ThermalLevel::from_raw(3), ThermalLevel::Critical);
        assert_eq!(ThermalLevel::from_raw(7), ThermalLevel::Critical);
        assert_eq!(ThermalLevel::from_raw(-1), ThermalLevel::Nominal);
    }

    #[test]
    fn profiles_round_trip_through_their_spelling() {
        for profile in [
            PowerProfile::Performance,
            PowerProfile::Balanced,
            PowerProfile::Efficiency,
        ] {
            assert_eq!(PowerProfile::parse(profile.as_str()), Some(profile));
        }
        assert_eq!(PowerProfile::parse("turbo"), None);
        assert_eq!(PowerProfile::parse("Performance"), None);
        assert_eq!(PowerProfile::default(), PowerProfile::Performance);
    }

    #[test]
    fn only_efficiency_carries_a_base_cap_and_only_performance_skips_stepping() {
        assert_eq!(PowerProfile::Performance.base_cap(), None);
        assert_eq!(PowerProfile::Balanced.base_cap(), None);
        assert_eq!(
            PowerProfile::Efficiency.base_cap(),
            Some(READING_SPEED_TOK_PER_SEC)
        );
        assert!(!PowerProfile::Performance.thermal_stepping());
        assert!(PowerProfile::Balanced.thermal_stepping());
        assert!(PowerProfile::Efficiency.thermal_stepping());
    }

    #[test]
    fn the_thermal_ladder_only_ever_lowers_the_cap() {
        assert_eq!(thermal_cap(None, ThermalLevel::Nominal), None);
        assert_eq!(thermal_cap(None, ThermalLevel::Fair), None);
        assert_eq!(thermal_cap(None, ThermalLevel::Serious), Some(10.0));
        assert_eq!(thermal_cap(None, ThermalLevel::Critical), Some(5.0));
        assert_eq!(thermal_cap(Some(20.0), ThermalLevel::Fair), Some(20.0));
        assert_eq!(thermal_cap(Some(20.0), ThermalLevel::Serious), Some(10.0));
        assert_eq!(thermal_cap(Some(20.0), ThermalLevel::Critical), Some(5.0));
        // Already below the ceiling: pressure must not RAISE a cap.
        assert_eq!(thermal_cap(Some(4.0), ThermalLevel::Serious), Some(4.0));
        assert_eq!(thermal_cap(Some(4.0), ThermalLevel::Critical), Some(4.0));
    }

    #[test]
    fn raw_memory_levels_map_the_kernels_bitmask_and_default_to_normal() {
        assert_eq!(MemoryPressure::from_raw(1), MemoryPressure::Normal);
        assert_eq!(MemoryPressure::from_raw(2), MemoryPressure::Warn);
        assert_eq!(MemoryPressure::from_raw(4), MemoryPressure::Critical);
        // 0 is what an unavailable sysctl answers, and 3 is not a level the
        // kernel publishes. Both mean "no reading", never "critical" -- the
        // opposite of the thermal clamp, for the reason `from_raw` gives.
        assert_eq!(MemoryPressure::from_raw(0), MemoryPressure::Normal);
        assert_eq!(MemoryPressure::from_raw(3), MemoryPressure::Normal);
        assert_eq!(MemoryPressure::from_raw(-1), MemoryPressure::Normal);
        assert_eq!(MemoryPressure::from_raw(99), MemoryPressure::Normal);
        assert_eq!(MemoryPressure::default(), MemoryPressure::Normal);
    }

    #[test]
    fn the_memory_ladder_only_ever_lowers_the_cap() {
        assert_eq!(memory_cap(None, MemoryPressure::Normal), None);
        assert_eq!(memory_cap(None, MemoryPressure::Warn), Some(10.0));
        assert_eq!(memory_cap(None, MemoryPressure::Critical), Some(5.0));
        assert_eq!(memory_cap(Some(20.0), MemoryPressure::Warn), Some(10.0));
        assert_eq!(memory_cap(Some(4.0), MemoryPressure::Critical), Some(4.0));
    }

    /// **The two signals are independent and the combined ladder takes the
    /// MINIMUM.** Each case here is one a single-signal ladder gets wrong:
    /// cool-and-short and hot-and-comfortable both cap, and a run under both
    /// takes the harder of the two rather than whichever was checked last.
    #[test]
    fn the_combined_ladder_takes_whichever_signal_binds_harder() {
        // Neither: untouched, which is the default decode path.
        assert_eq!(
            stepped_cap(None, ThermalLevel::Nominal, MemoryPressure::Normal),
            None
        );
        // Cool and short of memory. A thermal-only ladder misses this.
        assert_eq!(
            stepped_cap(None, ThermalLevel::Nominal, MemoryPressure::Critical),
            Some(5.0)
        );
        // Hot and comfortable. A memory-only ladder misses this.
        assert_eq!(
            stepped_cap(None, ThermalLevel::Critical, MemoryPressure::Normal),
            Some(5.0)
        );
        // Both, disagreeing: the harder ceiling wins in either direction, so
        // neither argument order nor a last-writer-wins bug can pass.
        assert_eq!(
            stepped_cap(None, ThermalLevel::Serious, MemoryPressure::Critical),
            Some(5.0)
        );
        assert_eq!(
            stepped_cap(None, ThermalLevel::Critical, MemoryPressure::Warn),
            Some(5.0)
        );
        // And an explicit cap already below both is not raised by either.
        assert_eq!(
            stepped_cap(Some(2.0), ThermalLevel::Critical, MemoryPressure::Critical),
            Some(2.0)
        );
    }

    #[test]
    fn rate_control_pairs_the_cap_with_the_profiles_stepping() {
        let performance = rate_control_for(PowerProfile::Performance, None);
        assert_eq!(performance.max_tokens_per_sec, None);
        assert!(performance.thermal_probe.is_none());
        // The memory watcher follows the profile's stepping, so the default
        // decode path stays exactly the statement sequence it was: no probe,
        // not active, no pacer built.
        assert!(performance.memory_probe.is_none());
        assert!(!performance.is_active());

        let balanced = rate_control_for(PowerProfile::Balanced, None);
        assert_eq!(balanced.max_tokens_per_sec, None);
        assert!(balanced.thermal_probe.is_some());
        assert!(balanced.memory_probe.is_some());
        assert!(balanced.is_active());

        let efficiency = rate_control_for(PowerProfile::Efficiency, None);
        assert_eq!(
            efficiency.max_tokens_per_sec,
            Some(READING_SPEED_TOK_PER_SEC)
        );
        assert!(efficiency.thermal_probe.is_some());
        assert!(efficiency.memory_probe.is_some());

        // An explicit cap overrides the profile's own, in both directions.
        let overridden = rate_control_for(PowerProfile::Efficiency, Some(3.0));
        assert_eq!(overridden.max_tokens_per_sec, Some(3.0));
        let capped_performance = rate_control_for(PowerProfile::Performance, Some(8.0));
        assert_eq!(capped_performance.max_tokens_per_sec, Some(8.0));
        assert!(capped_performance.thermal_probe.is_none());
        assert!(capped_performance.memory_probe.is_none());
        assert!(capped_performance.is_active());
    }

    #[test]
    fn an_explicit_profile_wins_over_the_low_power_mode_default() {
        assert_eq!(
            resolve_profile(Some(PowerProfile::Performance)),
            PowerProfile::Performance
        );
        assert_eq!(
            resolve_profile(Some(PowerProfile::Efficiency)),
            PowerProfile::Efficiency
        );
        // Unset follows the machine's Low Power Mode, which is state, not
        // a constant: assert the mapping rather than one of the outcomes.
        let expected = if low_power_mode_enabled() {
            PowerProfile::Efficiency
        } else {
            PowerProfile::Performance
        };
        assert_eq!(resolve_profile(None), expected);
    }
}
