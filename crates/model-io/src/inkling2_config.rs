//! The `inkling2.json` sidecar: Inkling 2's architecture values that have no
//! `ArchConfig` field.
//!
//! A sidecar rather than `ArchConfig` fields because `ArchConfig` has about 50
//! constructor sites and none of these values means anything to another
//! family (`docs/INKLING2_PHASE0.md`, Stage A). The generic values (window,
//! softcap, layer mask, sconv taps, route scale, router function) stay in
//! `ArchConfig`; [`Inkling2Config::validate_against`] checks the two agree.
//!
//! Written by `turbospark-repack`'s Inkling intake, read by the runtime's
//! Inkling flow at open. Every field is required: an omitted value must fail
//! the parse, never default (AGENTS.md Gotcha 24's lesson, applied here).

use serde::{Deserialize, Serialize};

use crate::{ArchConfig, ModelFamily};

/// File name of the sidecar inside an install directory.
pub const INKLING2_CONFIG_FILE: &str = "inkling2.json";

/// Inkling-only architecture values, from `config.json -> text_config`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Inkling2Config {
    /// First looped layer (`loop_config.core_start`).
    pub core_start: u32,
    /// One past the last looped layer (`loop_config.core_end`).
    pub core_end: u32,
    /// Passes over the core (`loop_config.num_passes`).
    pub passes: u32,
    /// Controller rank (`loop_config.gate_rank`).
    pub gate_rank: u32,
    /// Relative-state width per head (`d_rel`).
    pub d_rel: u32,
    /// Distances that get a relative bias (`rel_extent`).
    pub rel_extent: u32,
    /// L2 clip radius for q and k (`qk_l2_clip_threshold`).
    pub qk_clip: f32,
    /// L2 clip radius for the relative states (`rel_l2_clip_threshold`).
    pub rel_clip: f32,
    /// Full-attention log scale floor (`log_scaling_n_floor`).
    pub log_floor: f32,
    /// Full-attention log scale slope (`log_scaling_alpha`).
    pub log_alpha: f32,
    /// Expert gate softcap (`mlp_gate_softcap`).
    pub gate_softcap: f32,
    /// Expert up softcap (`mlp_up_softcap`).
    pub up_softcap: f32,
    /// Layers per attention-residual block (`n_layers_per_attn_res_block`).
    pub res_block: u32,
    /// Real vocabulary; logits past it are padding (`unpadded_vocab_size`).
    pub unpadded_vocab: u32,
    /// RMS norm epsilon (`rms_norm_eps`).
    pub rms_eps: f32,
    /// End-of-turn token (`eos_token_id`).
    pub eos_token_id: u32,
}

/// Why a sidecar was refused.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2ConfigError(pub String);

impl std::fmt::Display for Inkling2ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "inkling2.json: {}", self.0)
    }
}

impl std::error::Error for Inkling2ConfigError {}

impl Inkling2Config {
    /// `thinkingmachines/Inkling-2-Mini` at `ed40e7da`.
    pub fn mini() -> Self {
        Inkling2Config {
            core_start: 6,
            core_end: 18,
            passes: 3,
            gate_rank: 64,
            d_rel: 16,
            rel_extent: 1024,
            qk_clip: 22.627_417,
            rel_clip: 8.0,
            log_floor: 8192.0,
            log_alpha: 0.1,
            gate_softcap: 4.0,
            up_softcap: 25.0,
            res_block: 6,
            unpadded_vocab: 200_058,
            rms_eps: 1e-6,
            eos_token_id: 200_006,
        }
    }

    /// Cache slots of the unrolled loop (24 + 2 x 12 = 48 on the Mini).
    pub fn slots(&self, num_layers: u32) -> u32 {
        num_layers + (self.passes - 1) * (self.core_end - self.core_start)
    }

    /// Refuses values the flow does not implement, and any disagreement with
    /// `arch`. Every rule here has a reason in the reference graph
    /// (llama.cpp `inkling2-loop.cpp` validates the same loop bounds).
    pub fn validate_against(&self, arch: &ArchConfig) -> Result<(), Inkling2ConfigError> {
        let e = |m: String| Err(Inkling2ConfigError(m));
        if arch.family != ModelFamily::Inkling2Loop {
            return e(format!(
                "family is {}, not inkling2_loop",
                arch.family.as_str()
            ));
        }
        let layers = arch.num_layers as u32;
        if self.res_block == 0 {
            return e("resBlock must be positive".into());
        }
        if !(0 < self.core_start
            && self.core_start < self.core_end
            && self.core_end <= layers
            && self.core_start % self.res_block == 0
            && self.core_end % self.res_block == 0)
        {
            return e(format!(
                "loop core [{}, {}) must be non-empty, start after layer 0, end by layer {layers} \
                 and align to {}-layer blocks",
                self.core_start, self.core_end, self.res_block
            ));
        }
        // A fixed loop needs at least two passes (one pass is no loop).
        if self.passes < 2 || self.gate_rank == 0 {
            return e(format!(
                "passes {} must be at least 2 and gateRank {} positive",
                self.passes, self.gate_rank
            ));
        }
        if self.d_rel == 0 || self.rel_extent == 0 {
            return e("dRel and relExtent must be positive".into());
        }
        // The SWA layers bias every in-window distance; a window past the
        // extent would leave distances the table cannot index.
        if arch.sliding_window as u64 > u64::from(self.rel_extent) {
            return e(format!(
                "sliding window {} exceeds the relative extent {}",
                arch.sliding_window, self.rel_extent
            ));
        }
        for (name, v) in [
            ("qkClip", self.qk_clip),
            ("relClip", self.rel_clip),
            ("logFloor", self.log_floor),
            ("gateSoftcap", self.gate_softcap),
            ("upSoftcap", self.up_softcap),
            ("rmsEps", self.rms_eps),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return e(format!("{name} {v} must be finite and positive"));
            }
        }
        if !(self.log_alpha.is_finite() && self.log_alpha >= 0.0) {
            return e(format!(
                "logAlpha {} must be finite and non-negative",
                self.log_alpha
            ));
        }
        if self.unpadded_vocab == 0 || i64::from(self.unpadded_vocab) > arch.vocab_size {
            return e(format!(
                "unpaddedVocab {} must be in 1..={}",
                self.unpadded_vocab, arch.vocab_size
            ));
        }
        if self.eos_token_id >= self.unpadded_vocab {
            return e(format!(
                "eosTokenId {} is past the vocabulary",
                self.eos_token_id
            ));
        }
        Ok(())
    }

    /// Reads and validates `dir/inkling2.json`.
    pub fn load(dir: &std::path::Path, arch: &ArchConfig) -> Result<Self, Inkling2ConfigError> {
        let path = dir.join(INKLING2_CONFIG_FILE);
        let text = std::fs::read_to_string(&path)
            .map_err(|err| Inkling2ConfigError(format!("{}: {err}", path.display())))?;
        let cfg: Inkling2Config =
            serde_json::from_str(&text).map_err(|err| Inkling2ConfigError(err.to_string()))?;
        cfg.validate_against(arch)?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_validates_against_its_baseline() {
        let arch = crate::inkling2_mini();
        Inkling2Config::mini().validate_against(&arch).unwrap();
        assert_eq!(Inkling2Config::mini().slots(24), 48);
    }

    #[test]
    fn round_trips_through_json_exactly() {
        let cfg = Inkling2Config::mini();
        let text = serde_json::to_string(&cfg).unwrap();
        let back: Inkling2Config = serde_json::from_str(&text).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn an_omitted_or_unknown_field_is_refused() {
        let mut v = serde_json::to_value(Inkling2Config::mini()).unwrap();
        v.as_object_mut().unwrap().remove("relClip");
        assert!(serde_json::from_value::<Inkling2Config>(v.clone()).is_err());
        let mut v = serde_json::to_value(Inkling2Config::mini()).unwrap();
        v["surprise"] = serde_json::json!(1);
        assert!(serde_json::from_value::<Inkling2Config>(v).is_err());
    }

    #[test]
    fn unimplemented_loops_and_mismatches_are_refused() {
        let arch = crate::inkling2_mini();
        let bad = |f: fn(&mut Inkling2Config)| {
            let mut c = Inkling2Config::mini();
            f(&mut c);
            c.validate_against(&arch).is_err()
        };
        assert!(bad(|c| c.passes = 1));
        assert!(bad(|c| c.core_start = 0));
        assert!(bad(|c| c.core_end = 17));
        assert!(bad(|c| c.core_end = 30));
        assert!(bad(|c| c.rel_extent = 512));
        assert!(bad(|c| c.qk_clip = f32::NAN));
        assert!(bad(|c| c.unpadded_vocab = 300_000));
        assert!(bad(|c| c.eos_token_id = 200_058));
        let mut other = crate::inkling2_mini();
        other.family = ModelFamily::Gemma4;
        assert!(Inkling2Config::mini().validate_against(&other).is_err());
    }
}
