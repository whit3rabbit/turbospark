//! The prism Hadamard contract's REPACK half (the Bonsai-2 line,
//! `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`): parse the contract out of the
//! checkpoint's `config.json`, read its per-module sign vectors off the
//! safetensors, and emit the two things an install carries --
//! `hadamard.bin` (the sign vectors, one per distinct input width) and the
//! manifest's `hadamard` section the runtime plan is built from.
//!
//! THE CONTRACT (`docs/BONSAI2.md`): every quantized matrix is stored in a
//! signed block-Hadamard-rotated basis, so the RUNTIME transforms
//! activations -- forward on a folded entry's input, inverse on the
//! embedding's dequantized rows. Weights are never un-rotated at repack
//! time: dequantize-fold-requantize would either add a second quantization
//! error (2-bit) or blow the footprint (4-bit/BF16), and the activation-side
//! transform prism's own runtime applies is the cheaper and exact-by-
//! construction choice.

use super::shards::{shape4, Gemma4Shards};
use super::Gemma4Error;

/// The contract as parsed from `config.json`: the butterfly block width and
/// the folded/inverse module lists, already translated to ENGINE tensor
/// names (`language_model.model.layers.N.*`, the names the resident index
/// and the runtime's folded set both key on).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrismHadamard {
    /// Butterfly width in elements. The real checkpoint declares 1024 for
    /// every folded module; the parser admits exactly the four widths the
    /// bundled runtime validates.
    pub block: u32,
    /// Engine names of the quantized matrices whose INPUT the runtime must
    /// forward-transform.
    pub folded: Vec<String>,
    /// Engine names whose OUTPUT the runtime must inverse-transform. The
    /// embedding, on the one real checkpoint -- mirroring the bundled GGUF
    /// runtime, which refuses any inverse set beyond `token_embd.weight`.
    pub inverse: Vec<String>,
}

/// Config module path (`model.layers.3.mlp.gate_proj`) to the engine tensor
/// stem the resident index carries.
fn engine_name(path: &str) -> String {
    if path == "lm_head" {
        "language_model.lm_head".to_string()
    } else {
        format!("language_model.{path}")
    }
}

/// Parses `config.json -> modules` for the Hadamard contract. Returns `None`
/// when the config carries no packed modules at all, which is every
/// pre-contract checkpoint; a config that mixes transformed and untransformed
/// packed modules is refused, because the runtime's per-width sign sharing
/// (and every call site this port wires) assumes the all-or-nothing contract
/// the real checkpoint ships.
pub fn parse_prism_hadamard(config_json: &str) -> Result<Option<PrismHadamard>, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let Some(modules) = root.get("modules").and_then(|v| v.as_array()) else {
        return Ok(None);
    };
    let config_error = |detail: String| {
        Err(Gemma4Error::Config(format!(
            "config.json hadamard contract: {detail}"
        )))
    };
    let mut block: Option<u32> = None;
    let mut folded = Vec::new();
    let mut inverse = Vec::new();
    let mut packed = 0usize;
    for m in modules {
        let path = m
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Gemma4Error::Config("hadamard module without a path".to_string()))?;
        let module_block = m.get("block").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let embedding = m
            .get("embedding")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if module_block == 0 {
            // A packed module with no transform would read the RAW activation
            // while its folded siblings read the rotated one -- the one shape
            // the shared-transform wiring cannot express. See the fn doc.
            return config_error(format!(
                "packed module {path} declares block 0 amid transformed modules; this port \
                 implements the all-or-nothing contract"
            ));
        }
        packed += 1;
        match block {
            Some(b) if b != module_block => {
                return config_error(format!(
                    "module {path} declares block {module_block} against the contract's {b}; \
                     one block per checkpoint"
                ))
            }
            Some(_) => {}
            None => block = Some(module_block),
        }
        let engine = format!("{}.weight", engine_name(path));
        if embedding {
            inverse.push(engine);
        } else {
            folded.push(engine);
        }
    }
    if packed == 0 {
        return Ok(None);
    }
    let block = block.expect("set when packed > 0");
    if !matches!(block, 512 | 1024 | 2048 | 4096) {
        return config_error(format!(
            "block {block} is not one of the widths the bundled runtime validates \
             (512/1024/2048/4096)"
        ));
    }
    Ok(Some(PrismHadamard {
        block,
        folded,
        inverse,
    }))
}

/// Reads every folded module's `.signs` tensor and builds the install's
/// `hadamard.bin` blob plus the manifest section. Sign vectors are shared
/// per input width on the real contract: every vector of a given width is
/// asserted BYTE-EQUAL to the first vector seen at that width, so a future
/// checkpoint with genuinely per-module signs fails here (loudly, before any
/// install is written) instead of silently running on the wrong vector.
pub fn read_hadamard(
    shards: &Gemma4Shards<'_>,
    contract: &PrismHadamard,
) -> Result<(Vec<u8>, serde_json::Value), Gemma4Error> {
    let mut blob: Vec<u8> = Vec::new();
    let mut signs_json: Vec<serde_json::Value> = Vec::new();
    // width -> (offset, length) of that width's first vector inside `blob`.
    let mut seen: std::collections::BTreeMap<u64, (u64, u64)> = std::collections::BTreeMap::new();
    let width_of = |name: &str| -> Result<u64, Gemma4Error> {
        let info = shards.info(name)?;
        let shape = shape4(&info.shape);
        if shape.2 != 0 || shape.3 != 0 || shape.1 != 0 {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!("a sign vector is rank 1, got shape {:?}", info.shape),
            });
        }
        Ok(shape.0 as u64)
    };
    for name in contract.folded.iter().chain(contract.inverse.iter()) {
        let signs_name = name
            .strip_suffix(".weight")
            .ok_or_else(|| Gemma4Error::Config(format!("folded name {name} has no .weight")))?
            .to_string()
            + ".signs";
        if !shards.contains(&signs_name) {
            return Err(Gemma4Error::MissingTensor(signs_name));
        }
        let width = width_of(&signs_name)?;
        let bytes = shards.read(&signs_name)?;
        if bytes.len() as u64 != width * 4 {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: signs_name.clone(),
                detail: format!(
                    "sign vector of width {width} carries {} bytes; F32 is width*4",
                    bytes.len()
                ),
            });
        }
        match seen.get(&width) {
            Some(&(offset, length)) => {
                let start = offset as usize;
                if blob[start..start + length as usize] != bytes[..] {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: signs_name,
                        detail: format!(
                            "sign vector disagrees with the first vector at width {width}; \
                             per-module signs are a contract this port does not implement"
                        ),
                    });
                }
            }
            None => {
                let offset = blob.len() as u64;
                blob.extend_from_slice(&bytes);
                seen.insert(width, (offset, bytes.len() as u64));
                signs_json.push(serde_json::json!({
                    "width": width,
                    "offset": offset,
                    "bytes": bytes.len() as u64,
                }));
            }
        }
    }
    // The section the runtime plan is parsed from; `model_io` validates its
    // shape again at open, this is the writer agreeing with the reader.
    let section = serde_json::json!({
        "block": contract.block,
        "signs": signs_json,
        "folded": contract.folded,
        "inverse": contract.inverse,
    });
    Ok((blob, section))
}
