//! Reads a llama.cpp-layout control vector into a [`SteeringSet`].
//!
//! The format is deliberately borrowed rather than invented, and it is small:
//! a GGUF whose tensors are named `direction.N`, each one-dimensional F32 of
//! `hidden` elements. llama.cpp's loader reads nothing else -- it looks at no
//! metadata key at all -- so a file this module writes or reads is a file the
//! wider ecosystem (`repeng`, the published community vector sets) can also
//! use, which is the entire reason for the choice.
//!
//! # The indexing, which is 1-based and is the trap
//!
//! llama.cpp rejects `direction.0` BY NAME ("invalid (zero) direction tensor
//! layer index") and its apply loop runs `for il = 1; il < n_layer`, so its
//! layer 0 never receives a direction. This reader maps `direction.N` to
//! 0-based layer `N - 1`, which is what `scripts/extract_direction.py` writes
//! and what `turbospark.layer_base = 0` in the file records.
//!
//! **Whether that agrees with llama.cpp's own layer numbering is UNVERIFIED**
//! (`docs/OBLITERATION.md`). The two could differ by one layer, and an
//! off-by-one direction is a plausible wrong answer rather than a failure --
//! it steers, it just steers the wrong place. Read a vector written here with
//! this port; do not assume it is positioned identically elsewhere until
//! someone measures it.
//!
//! # Why this lives in `crates/repack`
//!
//! Because the GGUF parser does. `crates/runtime` cannot reach this crate
//! (AGENTS.md Gotcha 8), so the shape is the one `resolve_drafter` already
//! uses for the resident index: the FRONT END parses the file before open and
//! hands the runner a plain `SteeringSet`.

use std::collections::BTreeMap;
use std::path::Path;

use foundation::SteeringMode;
use model_io::{LayerDirection, SteeringSet};

use crate::gguf_header::{parse_header, GgufHeader};

/// The tensor-name prefix llama.cpp's loader splits on.
const DIRECTION_PREFIX: &str = "direction";

/// ggml's F32 type id. The only element type a control vector may use;
/// llama.cpp refuses anything else by name.
const GGML_TYPE_F32: u32 = 0;

/// Metadata key this port stamps with the mode a file was built for.
/// llama.cpp ignores it, as it ignores every metadata key here.
const MODE_KEY: &str = "turbospark.steering_mode";

/// What went wrong reading a control vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlVectorError {
    /// The file could not be opened or read.
    Io { path: String, detail: String },
    /// The GGUF container itself is malformed.
    Header { detail: String },
    /// A tensor is not named `direction.<positive integer>`.
    BadTensorName { name: String },
    /// llama.cpp rejects a zero layer index by name; so does this.
    ZeroLayerIndex,
    /// A direction is not one-dimensional F32.
    BadTensorShape { name: String, detail: String },
    /// Two directions disagree on their element count.
    RaggedWidths { first: usize, then: usize },
    /// The file carries no `direction.*` tensor at all.
    NoDirections,
    /// A tensor's data range runs past the end of the file.
    Truncated { name: String, wanted: u64, len: u64 },
}

impl std::fmt::Display for ControlVectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, detail } => write!(f, "reading {path}: {detail}"),
            Self::Header { detail } => write!(f, "not a readable GGUF: {detail}"),
            Self::BadTensorName { name } => write!(
                f,
                "tensor {name} is not named direction.<n>; a control vector carries \
                 nothing else"
            ),
            Self::ZeroLayerIndex => write!(
                f,
                "direction.0: the layer index is ONE-based in this format and a zero \
                 index is rejected, by llama.cpp as well as here"
            ),
            Self::BadTensorShape { name, detail } => {
                write!(f, "{name}: {detail}; a direction is 1-D F32")
            }
            Self::RaggedWidths { first, then } => write!(
                f,
                "directions disagree on width: {first} then {then}; every layer's \
                 direction must have the model's hidden size"
            ),
            Self::NoDirections => write!(
                f,
                "no direction.<n> tensors; this is a GGUF but not a control vector"
            ),
            Self::Truncated { name, wanted, len } => {
                write!(f, "{name}: data runs to byte {wanted} of a {len}-byte file")
            }
        }
    }
}

impl std::error::Error for ControlVectorError {}

/// Parses `direction.N` into a 0-based layer index.
fn layer_index(name: &str) -> Result<usize, ControlVectorError> {
    let rest = name
        .strip_prefix(DIRECTION_PREFIX)
        .and_then(|r| r.strip_prefix('.'))
        .ok_or_else(|| ControlVectorError::BadTensorName {
            name: name.to_string(),
        })?;
    let n: usize = rest
        .parse()
        .map_err(|_| ControlVectorError::BadTensorName {
            name: name.to_string(),
        })?;
    // Refused rather than clamped: a file numbering from zero is a file
    // written against a different convention, and silently shifting it by one
    // would steer every layer one place off.
    n.checked_sub(1).ok_or(ControlVectorError::ZeroLayerIndex)
}

fn read_set(header: &GgufHeader, bytes: &[u8]) -> Result<SteeringSet, ControlVectorError> {
    let mut found: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    let mut width: Option<usize> = None;

    for (name, info) in &header.tensors {
        let layer = layer_index(name)?;
        if info.ggml_type != GGML_TYPE_F32 {
            return Err(ControlVectorError::BadTensorShape {
                name: name.clone(),
                detail: format!("ggml type {} is not F32", info.ggml_type),
            });
        }
        if info.dims.len() != 1 {
            return Err(ControlVectorError::BadTensorShape {
                name: name.clone(),
                detail: format!("{} dimensions", info.dims.len()),
            });
        }
        let n = info.dims[0] as usize;
        match width {
            None => width = Some(n),
            Some(w) if w != n => {
                return Err(ControlVectorError::RaggedWidths { first: w, then: n })
            }
            Some(_) => {}
        }

        let start = (header.data_region_start + info.offset) as usize;
        let end = start + n * 4;
        if end > bytes.len() {
            return Err(ControlVectorError::Truncated {
                name: name.clone(),
                wanted: end as u64,
                len: bytes.len() as u64,
            });
        }
        let values = bytes[start..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        found.insert(layer, values);
    }

    let hidden = width.ok_or(ControlVectorError::NoDirections)?;
    let highest = *found.keys().next_back().expect("non-empty, width was set");
    let mut layers: Vec<Option<LayerDirection>> = (0..=highest).map(|_| None).collect();
    for (layer, values) in found {
        layers[layer] = Some(LayerDirection::new(values));
    }

    Ok(SteeringSet {
        layers,
        hidden,
        declared_mode: header.metadata_str(MODE_KEY).and_then(SteeringMode::parse),
        declared_arch: header.architecture().map(str::to_string),
    })
}

/// Reads a control vector from `bytes`.
///
/// The whole file is taken in memory rather than ranged: a full per-layer set
/// for the widest model here is 64 x 5120 x 4 = 1.25 MB, against the 14 GB
/// install it steers. There is nothing to stream.
pub fn parse_control_vector(bytes: &[u8]) -> Result<SteeringSet, ControlVectorError> {
    let header =
        parse_header(bytes, bytes.len() as u64).map_err(|e| ControlVectorError::Header {
            detail: format!("{e:?}"),
        })?;
    read_set(&header, bytes)
}

/// Reads a control vector from a path.
pub fn load_control_vector(path: &Path) -> Result<SteeringSet, ControlVectorError> {
    let bytes = std::fs::read(path).map_err(|e| ControlVectorError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    parse_control_vector(&bytes)
}

/// Serializes per-layer directions into a llama.cpp-layout control vector.
///
/// `directions` is indexed by 0-BASED layer and written as `direction.{l+1}`,
/// which is this port's convention and the one
/// `scripts/extract_direction.py` writes; see the module header on why the
/// off-by-one against llama.cpp's own numbering is unverified.
///
/// It exists as much for the tests as for callers: a reader whose only
/// fixtures come from the writer beside it can agree with that writer while
/// both are wrong, so the round-trip case below is paired with one that
/// checks the BYTES against the layout llama.cpp documents (1-based names, F32,
/// one dimension) rather than against this function.
pub fn write_control_vector(
    directions: &[Vec<f32>],
    arch: &str,
    mode: Option<SteeringMode>,
) -> Vec<u8> {
    let mut builder = crate::GgufBuilder::new()
        .metadata_str("general.architecture", arch)
        .metadata_str("controlvector.model_hint", arch)
        .metadata_u32("controlvector.layer_count", directions.len() as u32)
        .metadata_u32("turbospark.layer_base", 0);
    if let Some(m) = mode {
        builder = builder.metadata_str(MODE_KEY, m.as_str());
    }
    for (l, values) in directions.iter().enumerate() {
        let mut data = Vec::with_capacity(values.len() * 4);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        builder = builder.tensor(
            &format!("{DIRECTION_PREFIX}.{}", l + 1),
            GGML_TYPE_F32,
            &[values.len() as u64],
            data,
        );
    }
    builder.build().0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dirs(layers: usize, hidden: usize) -> Vec<Vec<f32>> {
        (0..layers)
            .map(|l| {
                (0..hidden)
                    .map(|i| ((l * hidden + i) as f32 * 0.37).sin())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_written_vector_reads_back_with_the_same_values() {
        let want = dirs(4, 16);
        let bytes = write_control_vector(&want, "qwen35", Some(SteeringMode::Ablate));
        let set = parse_control_vector(&bytes).expect("parses");

        assert_eq!(set.hidden, 16);
        assert_eq!(set.layers.len(), 4);
        assert_eq!(set.covered_layers(), 4);
        assert_eq!(set.declared_mode, Some(SteeringMode::Ablate));
        assert_eq!(set.declared_arch.as_deref(), Some("qwen35"));
        for (l, w) in want.iter().enumerate() {
            let got = set.layer(l).expect("layer present");
            assert_eq!(&got.values, w, "layer {l}");
        }
    }

    /// The reader and the writer beside it share an author, so a round trip
    /// alone cannot say the file matches llama.cpp's layout. This checks the
    /// three things that layout actually requires, against the BYTES.
    #[test]
    fn the_written_bytes_match_the_documented_layout() {
        let bytes = write_control_vector(&dirs(3, 8), "qwen35", None);
        let header = parse_header(&bytes, bytes.len() as u64).expect("valid GGUF");

        let names: Vec<&str> = header.tensors.keys().map(String::as_str).collect();
        assert!(
            names.contains(&"direction.1") && names.contains(&"direction.3"),
            "names are ONE-based: {names:?}"
        );
        assert!(
            !names.contains(&"direction.0"),
            "a zero index is rejected by llama.cpp by name"
        );
        for (name, info) in &header.tensors {
            assert_eq!(info.ggml_type, GGML_TYPE_F32, "{name} must be F32");
            assert_eq!(info.dims.len(), 1, "{name} must be one-dimensional");
        }
    }

    /// 1-based on the wire, 0-based in the struct. Getting this backwards
    /// steers every layer one place off, which is fluent and wrong.
    #[test]
    fn direction_one_is_layer_zero() {
        assert_eq!(layer_index("direction.1").unwrap(), 0);
        assert_eq!(layer_index("direction.64").unwrap(), 63);
    }

    #[test]
    fn a_zero_index_is_refused_rather_than_shifted() {
        assert_eq!(
            layer_index("direction.0"),
            Err(ControlVectorError::ZeroLayerIndex)
        );
    }

    #[test]
    fn a_foreign_tensor_name_is_refused() {
        assert!(matches!(
            layer_index("blk.0.attn_q.weight"),
            Err(ControlVectorError::BadTensorName { .. })
        ));
        assert!(matches!(
            layer_index("direction.middle"),
            Err(ControlVectorError::BadTensorName { .. })
        ));
    }

    /// A sparse file is the NORMAL case, not a damaged one: steering a narrow
    /// band of layers is what the research recommends, so a vector covering
    /// only layers 30-32 must load with the rest absent rather than zeroed.
    #[test]
    fn a_sparse_file_leaves_uncovered_layers_absent() {
        let mut builder = crate::GgufBuilder::new().metadata_str("general.architecture", "qwen35");
        for l in [3usize, 5] {
            let data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
            builder = builder.tensor(&format!("direction.{}", l + 1), GGML_TYPE_F32, &[8], data);
        }
        let set = parse_control_vector(&builder.build().0).expect("parses");

        assert_eq!(set.layers.len(), 6, "sized to the highest covered layer");
        assert_eq!(set.covered_layers(), 2);
        assert!(set.layer(3).is_some() && set.layer(5).is_some());
        for l in [0usize, 1, 2, 4] {
            assert!(set.layer(l).is_none(), "layer {l} should be absent");
        }
    }

    /// An F16 direction is the same width as nothing else here and would
    /// decode to garbage; llama.cpp refuses it by name and so does this.
    #[test]
    fn a_non_f32_direction_is_refused() {
        let builder = crate::GgufBuilder::new().tensor("direction.1", 1, &[8], vec![0u8; 16]);
        assert!(matches!(
            parse_control_vector(&builder.build().0),
            Err(ControlVectorError::BadTensorShape { .. })
        ));
    }

    #[test]
    fn directions_of_different_widths_are_refused() {
        let builder = crate::GgufBuilder::new()
            .tensor("direction.1", GGML_TYPE_F32, &[8], vec![0u8; 32])
            .tensor("direction.2", GGML_TYPE_F32, &[16], vec![0u8; 64]);
        assert!(matches!(
            parse_control_vector(&builder.build().0),
            Err(ControlVectorError::RaggedWidths { .. })
        ));
    }

    /// An ordinary GGUF is a readable container and NOT a control vector.
    /// It has to fail by name rather than load as an empty set.
    #[test]
    fn a_gguf_that_is_not_a_control_vector_is_refused() {
        let builder = crate::GgufBuilder::new().metadata_str("general.architecture", "llama");
        assert_eq!(
            parse_control_vector(&builder.build().0),
            Err(ControlVectorError::NoDirections)
        );
    }

    /// The mode is a HINT the file may omit. Absent must mean "the caller
    /// decides", never a silent default to one of the three edits.
    #[test]
    fn an_absent_mode_is_none_rather_than_a_default() {
        let bytes = write_control_vector(&dirs(2, 8), "qwen35", None);
        let set = parse_control_vector(&bytes).expect("parses");
        assert_eq!(set.declared_mode, None);
    }
}
