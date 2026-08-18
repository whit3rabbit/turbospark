//! The checkpoint's own context length: reading it out of either intake
//! format, recording it in an install, and reading it back.
//!
//! **Deliberately NOT an `ArchConfig` field**, and the reason is what that
//! struct is for rather than where the value would be tidiest. `ArchConfig`
//! describes an ARCHITECTURE -- `arch_validation` compares one field by
//! field against a per-family baseline, and `crates/repack`'s network tests
//! assert whole-struct equality between a GGUF-derived config and a shipped
//! baseline. A trained context is a per-CHECKPOINT number: two files of one
//! architecture can legitimately differ, because a YaRN-extended release
//! declares a longer one than the base it was built from. Putting it there
//! would make every such pair a baseline mismatch, and would require a
//! per-checkpoint claim inside a per-architecture table.
//!
//! It is also read by no kernel. Nothing about the forward pass changes
//! with it; it feeds the context-window policy in `crates/runtime` and the
//! warning the two binaries print. So it is install METADATA, sitting
//! beside `modelID` in spirit, and [`record`] annotates the manifest after
//! the walk has written it rather than threading a parameter through six
//! public install writers and their sixty-odd call sites.
//!
//! **An install written before this existed declares nothing, and that
//! reads as UNKNOWN rather than as a value.** Every install currently on
//! disk is in that state; `crates/runtime`'s policy answers an unknown
//! trained context with the documented default rather than inventing a
//! window (AGENTS.md Gotcha 39).

use std::path::Path;

use crate::gguf_header::GgufHeader;

/// The trained context an HF `config.json` declares, or `None`.
///
/// `max_position_embeddings` is the key every family here uses. It is read
/// from `text_config` first and the root second, because the multimodal
/// checkpoints this port ingests (Gemma 4, both Qwen halves, Muse Glimmer)
/// nest their text architecture one level down and put a VISION config
/// beside it -- and a root-first lookup on those files would find the
/// wrapper's own value where one exists, which is not the text model's.
pub fn from_config_json(json: &str) -> Option<u32> {
    let value: serde_json::Value = serde_json::from_str(json).ok()?;
    let read = |v: &serde_json::Value| v.get("max_position_embeddings").and_then(|n| n.as_u64());
    value
        .get("text_config")
        .and_then(read)
        .or_else(|| read(&value))
        .and_then(sane)
}

/// The trained context a GGUF header declares, or `None`.
///
/// `<arch>.context_length` is llama.cpp's spelling and it writes one for
/// every architecture it converts, so unlike the safetensors side this is
/// rarely absent. Still optional: a hand-built or truncated header is not
/// worth refusing an install over, since the value is advisory.
pub fn from_gguf(header: &GgufHeader) -> Option<u32> {
    let architecture = header.architecture()?;
    header
        .metadata
        .get(&format!("{architecture}.context_length"))
        .and_then(crate::gguf_header::GgufValue::as_u64)
        .and_then(sane)
}

/// A declared context length this port will act on.
///
/// Zero is what an absent key looks like after a cast and means "not
/// declared", never "a window of zero tokens". The upper bound is
/// `u32::MAX` because the value crosses into the policy as a `u32`; a file
/// declaring more than four billion positions is describing something other
/// than a context length.
fn sane(raw: u64) -> Option<u32> {
    match raw {
        0 => None,
        n if n > u32::MAX as u64 => None,
        n => Some(n as u32),
    }
}

/// Annotate an install's `manifest.json` with the checkpoint's trained
/// context, in place.
///
/// Safe to run after the walk has finished: `manifest.json` hashes
/// `model_weights.bin` and the packed-expert files and never itself, so
/// rewriting it invalidates nothing. Idempotent -- re-running overwrites
/// the same key with the same value.
pub fn record(model_dir: &Path, trained_context: u32) -> Result<(), String> {
    let path = model_dir.join("manifest.json");
    let bytes =
        std::fs::read(&path).map_err(|e| format!("no manifest.json at {}: {e}", path.display()))?;
    let mut value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("manifest.json: {e}"))?;
    let arch = value
        .get_mut("arch")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| format!("manifest.json at {} has no arch object", path.display()))?;
    arch.insert(
        "trainedContext".to_string(),
        serde_json::Value::from(trained_context),
    );
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap())
        .map_err(|e| format!("writing {}: {e}", path.display()))
}

/// The trained context an installed `.gturbo` declares, or `None` for one
/// written before the field existed.
///
/// A sibling of [`crate::peek_manifest_arch`] rather than part of it: that
/// one returns an `ArchConfig` and this value is deliberately not in it.
pub fn peek(model_dir: &Path) -> Option<u32> {
    let bytes = std::fs::read(model_dir.join("manifest.json")).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    value
        .get("arch")?
        .get("trainedContext")?
        .as_u64()
        .and_then(sane)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real shape: a multimodal wrapper with the text model's window in
    /// `text_config` and a DIFFERENT one on the vision tower beside it.
    /// Reading the root first finds neither reliably.
    #[test]
    fn the_text_configs_window_wins_over_the_wrapper() {
        let json = r#"{
            "model_type": "gemma4",
            "max_position_embeddings": 8192,
            "text_config": { "max_position_embeddings": 131072 },
            "vision_config": { "max_position_embeddings": 4096 }
        }"#;
        assert_eq!(from_config_json(json), Some(131_072));
    }

    /// A flat config (no `text_config`) still resolves, which is what a
    /// text-only checkpoint looks like.
    #[test]
    fn a_flat_config_falls_back_to_the_root() {
        assert_eq!(
            from_config_json(r#"{"max_position_embeddings": 32768}"#),
            Some(32_768)
        );
    }

    /// Absent, zero, malformed and absurd all read as UNKNOWN rather than
    /// as a number. Zero is the one that matters: it is what a missing key
    /// becomes after a cast, and a window of zero would refuse every prompt.
    #[test]
    fn nothing_usable_reads_as_unknown() {
        assert_eq!(from_config_json("{}"), None);
        assert_eq!(from_config_json(r#"{"max_position_embeddings": 0}"#), None);
        assert_eq!(
            from_config_json(r#"{"max_position_embeddings": 999999999999}"#),
            None
        );
        assert_eq!(from_config_json("not json at all"), None);
        assert_eq!(
            from_config_json(r#"{"max_position_embeddings": "8k"}"#),
            None
        );
    }

    /// `record` then `peek` round-trips, and leaves everything else in the
    /// manifest untouched -- it is annotating a file another writer owns.
    #[test]
    fn recording_round_trips_and_disturbs_nothing_else() {
        let dir =
            std::env::temp_dir().join(format!("turbospark-trained-context-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let original = serde_json::json!({
            "magic": "GTURBO",
            "modelID": "fixture",
            "arch": { "hiddenSize": 64, "numLayers": 2 },
            "quant": { "attention": "q8_0" },
        });
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(&original).unwrap(),
        )
        .unwrap();

        assert_eq!(peek(&dir), None, "nothing declared before recording");
        record(&dir, 131_072).unwrap();
        assert_eq!(peek(&dir), Some(131_072));

        let after: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(after["magic"], original["magic"]);
        assert_eq!(after["modelID"], original["modelID"]);
        assert_eq!(after["quant"], original["quant"]);
        assert_eq!(after["arch"]["hiddenSize"], 64);
        assert_eq!(after["arch"]["numLayers"], 2);

        // Idempotent: the walk may be re-run over an existing directory.
        record(&dir, 8192).unwrap();
        assert_eq!(peek(&dir), Some(8192));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A directory with no manifest is an error rather than a panic, and a
    /// peek at one is simply unknown.
    #[test]
    fn a_missing_manifest_is_an_error_to_write_and_unknown_to_read() {
        let dir = std::env::temp_dir().join("turbospark-no-such-install");
        assert!(record(&dir, 4096).is_err());
        assert_eq!(peek(&dir), None);
    }
}
