//! Built-in trait packs: contrastive prompt corpora and evaluation markers
//! for `turbospark-steer`.
//!
//! A pack ships PROMPTS and WORD LISTS, never a direction. The direction is
//! extracted at run time from the user's own checkpoint, so the provenance
//! rule `swift/TurboSparkApp` states ("nothing ships a direction") holds for
//! these files too: a pack labels a contrast, not a behavior.
//!
//! The corpus design mirrors the manual pairs `docs/OBLITERATION.md` records
//! (ocean/mountain, formal/casual): two prompt sets whose difference IS the
//! trait, reduced per layer to a difference of means. Held-out prompts and
//! keyword markers support the `check` verb's before/after readout.

use serde::Deserialize;

/// One side of the contrast.
#[derive(Debug, Deserialize)]
pub struct PackSide {
    /// Optional system prompt applied to every prompt on this side (the
    /// sycophancy pack conditions both sides this way).
    #[serde(default)]
    pub system: Option<String>,
    pub prompts: Vec<String>,
}

/// Held-out evaluation prompts plus the keyword markers the `check` verb
/// counts, case-insensitively.
#[derive(Debug, Deserialize)]
pub struct PackEval {
    pub prompts: Vec<String>,
    pub markers: Vec<String>,
}

/// A trait pack.
#[derive(Debug, Deserialize)]
pub struct TraitPack {
    pub id: String,
    pub title: String,
    pub summary: String,
    pub positive: PackSide,
    pub negative: PackSide,
    pub recommended_mode: String,
    pub recommended_scale: f32,
    pub advice: String,
    pub eval: PackEval,
}

impl TraitPack {
    /// Parse and validate an embedded pack's JSON.
    fn parse(id: &'static str, json: &'static str) -> Result<TraitPack, String> {
        let pack: TraitPack =
            serde_json::from_str(json).map_err(|e| format!("built-in pack {id}: {e}"))?;
        if pack.id != id {
            return Err(format!(
                "built-in pack file {id}.json declares id '{}'; the file name is the id",
                pack.id
            ));
        }
        for (side, prompts) in [
            ("positive", &pack.positive.prompts),
            ("negative", &pack.negative.prompts),
        ] {
            if prompts.len() < 4 {
                return Err(format!(
                    "pack {}: {side} side holds {} prompts; at least 4 are required",
                    pack.id,
                    prompts.len()
                ));
            }
        }
        if pack.eval.prompts.len() < 3 {
            return Err(format!(
                "pack {}: eval holds {} prompts; at least 3 are required",
                pack.id,
                pack.eval.prompts.len()
            ));
        }
        if pack.eval.markers.len() < 3 {
            return Err(format!(
                "pack {}: eval holds {} markers; at least 3 are required",
                pack.id,
                pack.eval.markers.len()
            ));
        }
        if foundation::SteeringMode::parse(&pack.recommended_mode).is_none() {
            return Err(format!(
                "pack {}: recommended_mode {} is not one of {}",
                pack.id,
                pack.recommended_mode,
                foundation::STEERING_MODE_NAMES.join(", ")
            ));
        }
        if !(0.0..=2.0).contains(&pack.recommended_scale) {
            return Err(format!(
                "pack {}: recommended_scale {} is outside 0.0..=2.0",
                pack.id, pack.recommended_scale
            ));
        }
        Ok(pack)
    }
}

/// Every built-in pack, in display order.
pub fn all() -> Result<Vec<TraitPack>, String> {
    let built_in: &[(&str, &str)] = &[
        ("refusals", include_str!("../steer_data/refusals.json")),
        ("slop", include_str!("../steer_data/slop.json")),
        ("humor", include_str!("../steer_data/humor.json")),
        ("sycophancy", include_str!("../steer_data/sycophancy.json")),
        ("emotions", include_str!("../steer_data/emotions.json")),
    ];
    let mut packs = Vec::with_capacity(built_in.len());
    for (id, json) in built_in {
        packs.push(TraitPack::parse(id, json)?);
    }
    Ok(packs)
}

/// One pack by id, or an error naming the built-ins.
pub fn by_id(id: &str) -> Result<TraitPack, String> {
    let known: &[(&str, &str)] = &[
        ("refusals", include_str!("../steer_data/refusals.json")),
        ("slop", include_str!("../steer_data/slop.json")),
        ("humor", include_str!("../steer_data/humor.json")),
        ("sycophancy", include_str!("../steer_data/sycophancy.json")),
        ("emotions", include_str!("../steer_data/emotions.json")),
    ];
    let (id, json) = known
        .iter()
        .find(|(known_id, _)| *known_id == id)
        .ok_or_else(|| {
            format!(
                "unknown trait pack '{id}'; built-ins are: {}",
                known
                    .iter()
                    .map(|(k, _)| *k)
                    .collect::<Vec<&str>>()
                    .join(", ")
            )
        })?;
    TraitPack::parse(id, json)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every embedded pack must parse, validate, and carry a recommended
    /// mode the engine's parser accepts.
    #[test]
    fn every_builtin_pack_parses_and_validates() {
        let packs = all().expect("all packs valid");
        assert_eq!(packs.len(), 5, "five built-in packs");
        let ids: Vec<&str> = packs.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["refusals", "slop", "humor", "sycophancy", "emotions"]
        );
        for pack in &packs {
            assert!(
                foundation::SteeringMode::parse(&pack.recommended_mode).is_some(),
                "pack {}: bad mode",
                pack.id
            );
            assert!(
                pack.eval
                    .markers
                    .iter()
                    .all(|m| m.is_ascii() && !m.is_empty()),
                "pack {}: markers must be non-empty ASCII",
                pack.id
            );
            for prompt in pack
                .positive
                .prompts
                .iter()
                .chain(&pack.negative.prompts)
                .chain(&pack.eval.prompts)
            {
                assert!(prompt.is_ascii(), "pack {}: prompts must be ASCII", pack.id);
            }
        }
    }

    /// The id lookup must resolve every id the list exposes and refuse
    /// anything else, naming the valid set.
    #[test]
    fn by_id_resolves_every_id_and_refuses_unknowns() {
        for pack in all().expect("packs") {
            let got = by_id(&pack.id).expect("resolvable id");
            assert_eq!(got.id, pack.id);
        }
        let err = by_id("nope").expect_err("unknown id refused");
        assert!(err.contains("refusals"), "error names the built-ins: {err}");
    }

    /// The sycophancy pack is the one pack whose sides are conditioned by
    /// system prompts, and both must be present: without the contrast the
    /// extraction would measure the questions rather than the flattery.
    #[test]
    fn sycophancy_conditions_both_sides() {
        let pack = by_id("sycophancy").expect("pack");
        assert!(pack.positive.system.as_deref().unwrap_or_default().len() > 40);
        assert!(pack.negative.system.as_deref().unwrap_or_default().len() > 40);
        assert_eq!(
            pack.positive.prompts.len(),
            pack.negative.prompts.len(),
            "paired design: same questions on both sides"
        );
    }
}
