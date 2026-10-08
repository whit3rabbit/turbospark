//! What an install or a family can do, answered once in Rust.
//!
//! A host used to restate these predicates in its own language (the macOS app
//! kept a Swift copy of the steering family set and of the KV-quant rule),
//! and a copy is correct only until the next family lands. The engine opens a
//! session with the Rust answer and REFUSES by name when it disagrees, so a
//! stale host copy either hides a working control or offers one that fails at
//! open.

use model_io::{layer_is_quantized, rht_supported, KvQuant, ModelFamily};
use serde_json::json;

/// Whether `--kv-bits` would be accepted for an install with these facts:
/// `model_io::rht_supported` on the full head width, and at least one layer
/// that `layer_is_quantized` would pick. Width 4 stands in for any width,
/// because eligibility does not depend on which one is requested.
pub(crate) fn kv_quant_supported(full_head_dim: i64, mask: &[u8], num_layers: usize) -> bool {
    if !rht_supported(full_head_dim) {
        return false;
    }
    let quant = KvQuant::TurboQuant {
        k_bits: 4,
        v_bits: 4,
    };
    mask.iter()
        .enumerate()
        .any(|(layer, value)| layer_is_quantized(quant, *value, layer, num_layers))
}

/// Capabilities of a family named by its persisted `ModelFamily::as_str`
/// spelling (what `installed.json` and `manifest.json` carry).
///
/// An unrecognised family is reported as `known: false` with every capability
/// false, because the safe answer for a family nothing has wired is "no" --
/// the engine itself fails an unlisted family loudly at open.
pub(crate) fn family_capabilities_json(family: &str) -> String {
    let parsed = ModelFamily::parse(family);
    json!({
        "family": family,
        "known": parsed.is_some(),
        "steeringSupported": parsed.map(steering_supported).unwrap_or(false),
    })
    .to_string()
}

#[cfg(target_os = "macos")]
fn steering_supported(family: ModelFamily) -> bool {
    runtime::family_dispatches_steering(family)
}

/// The decode flows that dispatch steering exist only in the macOS engine.
#[cfg(not(target_os = "macos"))]
fn steering_supported(_family: ModelFamily) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_quant_follows_head_width_and_layer_mask() {
        // Power-of-two width in 32..=512 with a full-attention layer that is
        // not the last one.
        assert!(kv_quant_supported(128, &[1, 0, 1, 1], 4));
        // Not a power of two.
        assert!(!kv_quant_supported(96, &[1, 1, 1, 1], 4));
        // Out of range on both sides.
        assert!(!kv_quant_supported(16, &[1, 1, 1], 3));
        assert!(!kv_quant_supported(1024, &[1, 1, 1], 3));
        // Sliding-window layers only: nothing to quantize.
        assert!(!kv_quant_supported(128, &[0, 0, 0, 0], 4));
        // The last full-attention layer alone is excluded in a deep stack...
        assert!(!kv_quant_supported(128, &[0, 0, 0, 1], 4));
        // ...but a two-layer stack counts every layer.
        assert!(kv_quant_supported(128, &[1, 1], 2));
    }

    #[test]
    fn family_capabilities_report_unknown_as_unsupported() {
        let v: serde_json::Value =
            serde_json::from_str(&family_capabilities_json("not-a-family")).unwrap();
        assert_eq!(v["known"], false);
        assert_eq!(v["steeringSupported"], false);
        assert_eq!(v["family"], "not-a-family");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn family_capabilities_match_the_runtime_predicate_for_every_family() {
        for family in ModelFamily::ALL {
            let v: serde_json::Value =
                serde_json::from_str(&family_capabilities_json(family.as_str())).unwrap();
            assert_eq!(v["known"], true, "{}", family.as_str());
            assert_eq!(
                v["steeringSupported"],
                runtime::family_dispatches_steering(family),
                "{}",
                family.as_str()
            );
        }
    }
}
