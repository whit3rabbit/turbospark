//! What an install or a family can do, answered once in Rust.
//!
//! A host used to restate these predicates in its own language (the macOS app
//! kept a Swift copy of the steering family set and of the KV-quant rule),
//! and a copy is correct only until the next family lands. The engine opens a
//! session with the Rust answer and REFUSES by name when it disagrees, so a
//! stale host copy either hides a working control or offers one that fails at
//! open.

use model_io::{layer_is_quantized, rht_supported, KvQuant, ModelFamily, ResolvedExpertResidency};
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
        "anePrefillSupported": parsed.map(ane_prefill_supported).unwrap_or(false),
        "mappedResidencySupported": parsed.map(mapped_residency_supported).unwrap_or(false),
        "mappedResidencyMeasured": parsed.map(mapped_residency_measured).unwrap_or(false),
    })
    .to_string()
}

/// The family and this build only. The chip is deliberately not part of it:
/// an explicit split runs on any chip, and M4+ is only `Auto`'s threshold
/// (`ts_chip_supports_ane_prefill` answers that half).
fn ane_prefill_supported(family: ModelFamily) -> bool {
    runtime::ANE_PREFILL_BUILT && family == ModelFamily::QwenGdnDense
}

#[cfg(target_os = "macos")]
fn mapped_residency_supported(family: ModelFamily) -> bool {
    runtime::mapped_residency_supported(family)
}

/// Mapped residency is served only by the macOS engine.
#[cfg(not(target_os = "macos"))]
fn mapped_residency_supported(_family: ModelFamily) -> bool {
    false
}

/// Families where mapped residency has a real-install memory and speed
/// measurement, not only a correctness check (`docs/EXPERT_RESIDENCY.md`:
/// Gemma 4, 84% less counted memory and 35% more decode; Qwen3Moe and GptOss
/// are verified byte-identical but not measured). A host marks mapped as
/// unmeasured elsewhere. Widen only with a measurement.
fn mapped_residency_measured(family: ModelFamily) -> bool {
    mapped_residency_supported(family) && family == ModelFamily::Gemma4
}

/// What an open would do with a residency request on one install, decided
/// from facts the caller already read, so the rule is testable without one.
///
/// `routed` is whether the install has a routed expert layout: a dense
/// install has nothing to place, so residency is not applicable there and a
/// host should send `auto`. `resolved` is
/// `runtime::resolve_expert_residency_for_install`'s answer, which already
/// refuses a mapped request past Metal's working set. The family refusal is
/// applied later in the open (`open_expert_streamers`), so it is checked here
/// too, or a host would offer mapped on qwen4exp and the open would fail.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn residency_preflight_json(
    routed: bool,
    requested: &str,
    family_mapped: bool,
    resolved: Result<ResolvedExpertResidency, String>,
    family_refusal: Option<String>,
) -> serde_json::Value {
    let (resolved, refusal) = if !routed {
        (None, None)
    } else {
        match resolved {
            Err(reason) => (None, Some(reason)),
            Ok(ResolvedExpertResidency::Mapped) if !family_mapped => (None, family_refusal),
            Ok(mode) => (Some(mode.as_str()), None),
        }
    };
    json!({
        "applicable": routed,
        "requested": requested,
        "resolved": resolved,
        "refusal": refusal,
    })
}

/// `residency_preflight_json` for the install at `path`, with the same
/// resolver and family rule the open uses. `requested` is parsed like the
/// `expertResidency` open option, so a typo is refused by name.
#[cfg(target_os = "macos")]
pub(crate) fn expert_residency_for_install_json(
    path: &str,
    requested: &str,
) -> Result<String, String> {
    let dir = std::path::Path::new(path);
    let request = crate::wire::expert_residency(&Some(requested.to_string()))?;
    let arch = repack::peek_manifest_arch(dir)?;
    let routed = model_io::load_packed_experts_layout(
        dir,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .map(|layout| layout.num_layers > 0)
    .unwrap_or(false);
    let resolved = runtime::resolve_expert_residency_for_install(
        dir,
        arch.family,
        request,
        runtime::physical_memory(),
    );
    Ok(residency_preflight_json(
        routed,
        &requested.trim().to_ascii_lowercase(),
        runtime::mapped_residency_supported(arch.family),
        resolved,
        runtime::mapped_residency_refusal(arch.family)
            .err()
            .map(|e| e.to_string()),
    )
    .to_string())
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
        assert_eq!(v["anePrefillSupported"], false);
        assert_eq!(v["mappedResidencySupported"], false);
        assert_eq!(v["mappedResidencyMeasured"], false);
        assert_eq!(v["family"], "not-a-family");
    }

    #[test]
    fn residency_preflight_follows_the_open() {
        use ResolvedExpertResidency::{Mapped, Streamed};
        let refusal = || Some("not wired".to_string());
        // Dense: nothing to place, whatever was asked.
        let v = residency_preflight_json(false, "mapped", true, Ok(Mapped), refusal());
        assert_eq!(v["applicable"], false);
        assert!(v["resolved"].is_null() && v["refusal"].is_null());
        // Mapped on a family with a mapped arm, inside the working set.
        let v = residency_preflight_json(true, "mapped", true, Ok(Mapped), refusal());
        assert_eq!(v["resolved"], "mapped");
        assert!(v["refusal"].is_null());
        // Mapped on a family without one: the open's family refusal.
        let v = residency_preflight_json(true, "mapped", false, Ok(Mapped), refusal());
        assert!(v["resolved"].is_null());
        assert_eq!(v["refusal"], "not wired");
        // Past the working set: the resolver's own refusal.
        let v = residency_preflight_json(true, "mapped", true, Err("too big".into()), refusal());
        assert!(v["resolved"].is_null());
        assert_eq!(v["refusal"], "too big");
        // Streamed is never refused, on any family.
        let v = residency_preflight_json(true, "streamed", false, Ok(Streamed), refusal());
        assert_eq!(v["resolved"], "streamed");
        assert!(v["refusal"].is_null());
        assert_eq!(v["requested"], "streamed");
    }

    /// The real resolver on a real install: `TURBOSPARK_RESIDENCY_INSTALL`
    /// names an MoE install directory. Streamed is always taken, and mapped
    /// is taken or refused exactly as the family's mapped arm says (a
    /// working-set refusal on a supported family is also a refusal).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs TURBOSPARK_RESIDENCY_INSTALL, an MoE install directory"]
    fn residency_preflight_on_a_real_install() {
        let path = std::env::var("TURBOSPARK_RESIDENCY_INSTALL").expect("set the install path");
        let family = repack::peek_manifest_arch(std::path::Path::new(&path))
            .unwrap()
            .family;
        let ask = |requested: &str| -> serde_json::Value {
            let json = expert_residency_for_install_json(&path, requested).unwrap();
            eprintln!("{requested}: {json}");
            serde_json::from_str(&json).unwrap()
        };
        let streamed = ask("streamed");
        assert_eq!(streamed["applicable"], true, "not an MoE install");
        assert_eq!(streamed["resolved"], "streamed");
        let mapped = ask("mapped");
        if !runtime::mapped_residency_supported(family) {
            // Refused either way: the working-set guard runs first, as in the
            // open (qwen4exp's 35.65 GiB table trips it before the family
            // check is reached), and the family refusal otherwise.
            assert!(mapped["resolved"].is_null());
            assert!(mapped["refusal"].is_string());
        } else {
            assert!(mapped["resolved"] == "mapped" || mapped["refusal"].is_string());
        }
        assert!(ask("auto")["resolved"].is_string());
        assert!(expert_residency_for_install_json(&path, "mmap").is_err());
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
            assert_eq!(
                v["anePrefillSupported"],
                runtime::ANE_PREFILL_BUILT && family == ModelFamily::QwenGdnDense,
                "{}",
                family.as_str()
            );
            assert_eq!(
                v["mappedResidencySupported"],
                runtime::mapped_residency_supported(family),
                "{}",
                family.as_str()
            );
            // Measured implies supported: never flag a family as measured
            // for a mode its open refuses.
            assert!(
                v["mappedResidencyMeasured"] == false || v["mappedResidencySupported"] == true,
                "{}",
                family.as_str()
            );
        }
    }
}
