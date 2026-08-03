//! Public runtime configuration.
//!
//! Constrained numeric knobs and policy selectors with documented defaults.
//! Setting a numeric knob to a value outside its allowed set is a fatal
//! precondition failure that aborts construction by panicking; there is no
//! fallback or clamping to a nearby value.

/// Allowed cache-slot values.
pub const ALLOWED_CACHE_SLOTS: [u32; 4] = [8, 16, 24, 32];

/// Documented default cache-slot count.
pub const DEFAULT_CACHE_SLOTS: u32 = 16;

/// Allowed prompt-processing chunk-size values.
pub const ALLOWED_CHUNK_SIZES: [u32; 8] = [32, 64, 128, 256, 512, 1024, 2048, 4096];

/// Documented default prompt-processing chunk size.
pub const DEFAULT_CHUNK_SIZE: u32 = 128;

/// Cache replacement policy. Variant names are destination-selected and do
/// not mirror any source token; whether they are user-visible config keys is
/// an open decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CacheReplacement {
    /// Documented default mode.
    #[default]
    Primary,
    /// Alternate mode.
    Alternate,
}

/// Attention strategy selector. Variant names are destination-selected and do
/// not mirror any source token. The set is non-exhaustive because the full
/// documented variant set has not been pinned yet; later slices extend it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AttentionStrategy {
    /// Documented default strategy.
    #[default]
    Standard,
}

/// Head projection mode. The two modes are described by the approved spec.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum HeadProjection {
    /// Combined projection, the documented default.
    #[default]
    Combined,
    /// Separate projection, which can be forced on.
    Separate,
}

/// Immutable runtime configuration assembled from overrides plus defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    cache_slots: u32,
    chunk_size: u32,
    cache_replacement: CacheReplacement,
    prompt_processing_enabled: bool,
    attention_strategy: AttentionStrategy,
    head_projection: HeadProjection,
}

impl RuntimeConfig {
    /// Start a builder.
    pub fn builder() -> RuntimeConfigBuilder {
        RuntimeConfigBuilder::new()
    }

    /// Configured cache-slot count.
    pub fn cache_slots(&self) -> u32 {
        self.cache_slots
    }

    /// Configured prompt-processing chunk size.
    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    /// Configured cache replacement policy.
    pub fn cache_replacement(&self) -> CacheReplacement {
        self.cache_replacement
    }

    /// Whether prompt processing is enabled.
    pub fn prompt_processing_enabled(&self) -> bool {
        self.prompt_processing_enabled
    }

    /// Configured attention strategy.
    pub fn attention_strategy(&self) -> AttentionStrategy {
        self.attention_strategy
    }

    /// Configured head projection mode.
    pub fn head_projection(&self) -> HeadProjection {
        self.head_projection
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            cache_slots: DEFAULT_CACHE_SLOTS,
            chunk_size: DEFAULT_CHUNK_SIZE,
            cache_replacement: CacheReplacement::default(),
            prompt_processing_enabled: true,
            attention_strategy: AttentionStrategy::default(),
            head_projection: HeadProjection::default(),
        }
    }
}

fn is_allowed(value: u32, allowed: &[u32]) -> bool {
    allowed.contains(&value)
}

/// Builder for [`RuntimeConfig`]. Numeric setters panic on out-of-set values.
#[derive(Debug, Clone, Default)]
pub struct RuntimeConfigBuilder {
    cache_slots: Option<u32>,
    chunk_size: Option<u32>,
    cache_replacement: Option<CacheReplacement>,
    prompt_processing_enabled: Option<bool>,
    attention_strategy: Option<AttentionStrategy>,
    head_projection: Option<HeadProjection>,
}

impl RuntimeConfigBuilder {
    /// Create a builder with no overrides.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the cache-slot count. Panics unless `value` is 8, 16, 24, or 32.
    pub fn cache_slots(mut self, value: u32) -> Self {
        assert!(
            is_allowed(value, &ALLOWED_CACHE_SLOTS),
            "cache slots must be one of {:?}; got {value}",
            ALLOWED_CACHE_SLOTS,
        );
        self.cache_slots = Some(value);
        self
    }

    /// Set the prompt-processing chunk size. Panics unless `value` is one of
    /// the allowed sizes.
    pub fn chunk_size(mut self, value: u32) -> Self {
        assert!(
            is_allowed(value, &ALLOWED_CHUNK_SIZES),
            "chunk size must be one of {:?}; got {value}",
            ALLOWED_CHUNK_SIZES,
        );
        self.chunk_size = Some(value);
        self
    }

    /// Set the cache replacement policy.
    pub fn cache_replacement(mut self, value: CacheReplacement) -> Self {
        self.cache_replacement = Some(value);
        self
    }

    /// Set whether prompt processing is enabled.
    pub fn prompt_processing_enabled(mut self, value: bool) -> Self {
        self.prompt_processing_enabled = Some(value);
        self
    }

    /// Set the attention strategy.
    pub fn attention_strategy(mut self, value: AttentionStrategy) -> Self {
        self.attention_strategy = Some(value);
        self
    }

    /// Set the head projection mode.
    pub fn head_projection(mut self, value: HeadProjection) -> Self {
        self.head_projection = Some(value);
        self
    }

    /// Assemble the configuration, applying documented defaults for any knob
    /// or selector that was not set.
    pub fn build(self) -> RuntimeConfig {
        RuntimeConfig {
            cache_slots: self.cache_slots.unwrap_or(DEFAULT_CACHE_SLOTS),
            chunk_size: self.chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE),
            cache_replacement: self.cache_replacement.unwrap_or_default(),
            prompt_processing_enabled: self.prompt_processing_enabled.unwrap_or(true),
            attention_strategy: self.attention_strategy.unwrap_or_default(),
            head_projection: self.head_projection.unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Light inline checks; exhaustive value-set coverage lives in the
    //! integration test suite.
    use super::*;

    #[test]
    fn defaults_match_documented() {
        let cfg = RuntimeConfig::default();
        assert_eq!(cfg.cache_slots(), DEFAULT_CACHE_SLOTS);
        assert_eq!(cfg.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert!(cfg.prompt_processing_enabled());
    }

    #[test]
    fn builder_without_overrides_equals_default() {
        assert_eq!(
            RuntimeConfigBuilder::new().build(),
            RuntimeConfig::default()
        );
    }
}
