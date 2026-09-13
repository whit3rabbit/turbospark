//! Reads a llama.cpp-layout control vector into a [`SteeringSet`].
//!
//! The format is deliberately borrowed rather than invented, and it is small:
//! a GGUF whose tensors are named `direction.N`, each one-dimensional F32 of
//! `hidden` elements. llama.cpp's loader reads nothing else -- it looks at no
//! metadata key at all -- so a file this module writes or reads is a file the
//! wider ecosystem (`repeng`, the published community vector sets) can also
//! use, which is the entire reason for the choice.
//!
//! # The indexing, which is 1-based and WAS off by one here
//!
//! `direction.N` names llama.cpp's 0-based block `N`, so `direction.1` is the
//! SECOND block and block 0 cannot be addressed at all. That is settled by
//! reading llama.cpp rather than by measuring it, from two places that agree:
//!
//! - its loader writes `direction.N` to buffer offset `n_embd * (N - 1)`
//!   (`common.cpp`, `common_control_vector_load_one`), and
//! - its applier reads block `il` from offset `n_embd * (il - 1)`, looping
//!   from `il = 1` (`llama-adapter.cpp`). Chaining the two gives `il = N`.
//!   `llama.h` says the same independently, in a comment: the buffer "should
//!   point to an n_embd x n_layers buffer starting from layer 1".
//!
//! **This module used to map `direction.N` to layer `N - 1`**, i.e. one block
//! early, and the reasoning that produced it is worth keeping because it is
//! plausible: llama.cpp's block 0 never receives a direction, so the indices
//! look like they must shift down by one. They do not. Block 0 goes unsteered
//! precisely BECAUSE the lowest direction lands on block 1, and the old
//! mapping steered block 0 first -- contradicting the very invariant it cited.
//!
//! The apply SITE was right and is unchanged: llama.cpp adds the direction
//! after the FFN residual add, on the block output that feeds the next block
//! (`build_cvec` between the residual add and `l_out`), which is the boundary
//! `families/qwen/produce.rs` uses.
//!
//! # Reading both conventions, which is what the stamped key is for
//!
//! `turbospark.layer_base` records the 0-based block that `direction.1`
//! refers to. It was stamped from the first commit and READ BY NOTHING, which
//! is AGENTS.md Gotcha 45's shape (a writer may only record a tag some reader
//! honours); giving it a consumer is what lets the correction land without
//! reinterpreting the vectors already on disk.
//!
//! - `1`, or the key ABSENT: `direction.N` is block `N`. llama.cpp's
//!   convention, every foreign vector, and everything this port writes now.
//! - `0`: `direction.N` is block `N - 1`. Files this port wrote before the
//!   correction, which keep meaning what they meant when they were measured.
//!
//! Any other value is REFUSED rather than clamped: it is a file written
//! against a convention nothing here implements, and guessing at it would
//! steer every block some unknown distance off. So is a value that is not an
//! integer at all -- GGUF metadata carries no schema, so a string, a float or
//! an array is a real thing a writer can put here, and reading one as ABSENT
//! would hand llama.cpp's convention to a file that was declaring a different
//! one. Presence and readability are separate questions, and only the first
//! has a safe default.
//!
//! Because llama.cpp ignores the key, a file written now is positioned
//! identically in both engines. A legacy `layer_base = 0` file is NOT, and
//! never was; that is the bug, recorded rather than silently rewritten.
//!
//! # Why this lives in `crates/repack`
//!
//! Because the GGUF parser does. `crates/runtime` cannot reach this crate
//! (AGENTS.md Gotcha 8), so the shape is the one `resolve_drafter` already
//! uses for the resident index: the FRONT END parses the file before open and
//! hands the runner a plain `SteeringSet`.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use foundation::SteeringMode;
use model_io::{LayerDirection, SteeringSet};

use crate::gguf_header::{parse_header, GgufHeader, GgufValue};

/// The tensor-name prefix llama.cpp's loader splits on.
const DIRECTION_PREFIX: &str = "direction";

/// ggml's F32 type id. The only element type a control vector may use;
/// llama.cpp refuses anything else by name.
const GGML_TYPE_F32: u32 = 0;

/// Metadata key this port stamps with the mode a file was built for.
/// llama.cpp ignores it, as it ignores every metadata key here.
const MODE_KEY: &str = "turbospark.steering_mode";

/// Metadata key recording which 0-based block `direction.1` refers to. See
/// the module header: 1 (or absent) is llama.cpp's convention, 0 is this
/// port's pre-correction one.
const LAYER_BASE_KEY: &str = "turbospark.layer_base";

/// What `direction.1` means when the file does not say. llama.cpp reads no
/// metadata at all, so every foreign vector lands here, and the default has
/// to be ITS convention rather than this port's old one.
const DEFAULT_LAYER_BASE: u64 = 1;

/// Upper bound on a resolved 0-based block index. `read_set` allocates a
/// `Vec<Option<LayerDirection>>` of `highest + 1`, so an unbounded
/// `direction.N` (or a `turbospark.layer_base` chosen to push a small N past
/// this) is an attacker-controlled allocation size; 4096 sits far above any
/// model this port runs (the largest is a few hundred blocks).
const MAX_LAYER_INDEX: usize = 4096;

/// Maximum number of bytes accepted from a control-vector file.
///
/// Published vectors are around 1.3 MiB. Keeping a generous fixed ceiling
/// makes the standalone inspector safe for caller-selected paths without
/// changing the format accepted by real vectors.
const MAX_CONTROL_VECTOR_BYTES: u64 = 16 * 1024 * 1024;

/// What went wrong reading a control vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlVectorError {
    /// The file could not be opened or read.
    Io { path: String, detail: String },
    /// The file exceeds the resource ceiling for a control vector.
    FileTooLarge { path: String, max_bytes: u64 },
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
    /// `turbospark.layer_base` names a convention this reader does not know.
    UnknownLayerBase { base: u64 },
    /// `turbospark.layer_base` is PRESENT but is not an integer, so the file
    /// states a convention this reader cannot read. Distinct from the key
    /// being absent, which is llama.cpp's convention and is fine.
    MalformedLayerBase { kind: String },
    /// The caller asked to write a direction for block 0, which the format
    /// cannot name (`direction.0` is refused by llama.cpp and here).
    LayerZeroNotExpressible,
    /// `direction.N`'s block index is absurdly large. No model this port
    /// runs has more than a few hundred blocks; refusing early avoids
    /// sizing a `Vec` off an attacker-controlled tensor name.
    LayerIndexTooLarge { name: String, index: usize },
}

impl std::fmt::Display for ControlVectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, detail } => write!(f, "reading {path}: {detail}"),
            Self::FileTooLarge { path, max_bytes } => write!(
                f,
                "reading {path}: control vector exceeds the {max_bytes}-byte limit"
            ),
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
            Self::UnknownLayerBase { base } => write!(
                f,
                "{LAYER_BASE_KEY} = {base}: this reader knows 1 (llama.cpp's, and the \
                 default when the key is absent) and 0 (this port before the numbering \
                 was corrected). Guessing at another would steer every block off"
            ),
            Self::MalformedLayerBase { kind } => write!(
                f,
                "{LAYER_BASE_KEY} is present but is {kind} rather than an integer, so \
                 this file states a block numbering that cannot be read. Treating it as \
                 absent would silently apply llama.cpp's convention to a file that was \
                 trying to declare a different one"
            ),
            Self::LayerZeroNotExpressible => write!(
                f,
                "a direction for block 0 cannot be written: the format names blocks from \
                 direction.1 = block 1, and llama.cpp never applies a direction at block \
                 0 at all"
            ),
            Self::LayerIndexTooLarge { name, index } => write!(
                f,
                "{name}: block index {index} exceeds {MAX_LAYER_INDEX}, far past any real \
                 model's block count"
            ),
        }
    }
}

impl std::error::Error for ControlVectorError {}

/// Resolves the file's numbering convention from its metadata.
///
/// Absent means llama.cpp's, because llama.cpp stamps nothing and a foreign
/// vector is the case the default exists to serve.
/// Names what an unreadable `layer_base` value IS, short and from a fixed
/// set. `{value:?}` would print a whole array into an error message.
fn unreadable_kind(value: &GgufValue) -> &'static str {
    use GgufValue as V;
    match value {
        V::I8(_) | V::I16(_) | V::I32(_) | V::I64(_) => "a negative integer",
        V::F32(_) | V::F64(_) => "a float",
        V::Bool(_) => "a bool",
        V::String(_) => "a string",
        V::Array(_) => "an array",
        // Unreachable: `as_u64` widens every unsigned variant. Named rather
        // than `unreachable!`d, because a panic here would abort a load over
        // a metadata key the reader is already refusing.
        V::U8(_) | V::U16(_) | V::U32(_) | V::U64(_) => "an integer out of range",
    }
}

fn layer_base(header: &GgufHeader) -> Result<u64, ControlVectorError> {
    // Keyed on PRESENCE first, not on `metadata_u64`'s `None`. That `None`
    // merges "no such key" with "the key is a string / a float / an array",
    // and the two mean opposite things here: absent is a foreign vector under
    // llama.cpp's convention, present-and-unreadable is a file declaring a
    // convention this reader cannot resolve. Merging them would place every
    // direction one block off in silence, which is the failure this whole
    // key exists to prevent.
    let Some(value) = header.metadata.get(LAYER_BASE_KEY) else {
        return Ok(DEFAULT_LAYER_BASE);
    };
    match value.as_u64() {
        Some(b @ (0 | 1)) => Ok(b),
        Some(base) => Err(ControlVectorError::UnknownLayerBase { base }),
        None => Err(ControlVectorError::MalformedLayerBase {
            kind: unreadable_kind(value).to_string(),
        }),
    }
}

/// Parses `direction.N` into a 0-based layer index under `base`.
///
/// `base` is the 0-based block `direction.1` names, so the map is
/// `N - 1 + base`. A zero index is refused under BOTH conventions: llama.cpp
/// rejects it by name, and this port's older files never wrote one.
fn layer_index(name: &str, base: u64) -> Result<usize, ControlVectorError> {
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
    let zero_based = n.checked_sub(1).ok_or(ControlVectorError::ZeroLayerIndex)?;
    let index = zero_based
        .checked_add(base as usize)
        .filter(|i| *i <= MAX_LAYER_INDEX)
        .ok_or_else(|| ControlVectorError::LayerIndexTooLarge {
            name: name.to_string(),
            index: zero_based.saturating_add(base as usize),
        })?;
    Ok(index)
}

fn read_set(header: &GgufHeader, bytes: &[u8]) -> Result<SteeringSet, ControlVectorError> {
    let mut found: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    let mut width: Option<usize> = None;
    let base = layer_base(header)?;

    for (name, info) in &header.tensors {
        let layer = layer_index(name, base)?;
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

        // `info.offset` is an untrusted header field; add and multiply
        // checked rather than wrapping toward a plausible-looking range
        // that then under-reads or panics on the slice below. An overflow
        // is reported as `Truncated` with a sentinel `wanted` -- there is
        // no real byte offset to name, and the file is unreadable either
        // way.
        let overflow = || ControlVectorError::Truncated {
            name: name.clone(),
            wanted: u64::MAX,
            len: bytes.len() as u64,
        };
        let byte_len = (n as u64).checked_mul(4).ok_or_else(overflow)?;
        let start_u64 = header
            .data_region_start
            .checked_add(info.offset)
            .ok_or_else(overflow)?;
        let end_u64 = start_u64.checked_add(byte_len).ok_or_else(overflow)?;
        if end_u64 > bytes.len() as u64 {
            return Err(ControlVectorError::Truncated {
                name: name.clone(),
                wanted: end_u64,
                len: bytes.len() as u64,
            });
        }
        let start = start_u64 as usize;
        let end = end_u64 as usize;
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
    let io_error = |e: std::io::Error| ControlVectorError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    };
    let file = std::fs::File::open(path).map_err(&io_error)?;
    let mut bytes = Vec::new();
    file.take(MAX_CONTROL_VECTOR_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    if bytes.len() as u64 > MAX_CONTROL_VECTOR_BYTES {
        return Err(ControlVectorError::FileTooLarge {
            path: path.display().to_string(),
            max_bytes: MAX_CONTROL_VECTOR_BYTES,
        });
    }
    parse_control_vector(&bytes)
}

/// Serializes per-block directions into a llama.cpp-layout control vector.
///
/// Keyed by 0-BASED block and written as `direction.{l}`, which is
/// llama.cpp's own numbering, so a file this writes is positioned identically
/// in both engines.
///
/// # Why a map and not a slice
///
/// It took `&[Vec<f32>]`, dense from block 0, and that signature cannot
/// express this format: block 0 has no name here, and a dense 0-based slice
/// always has a block 0. The reader has always modelled a sparse set as the
/// NORMAL case (steering a narrow band is what the research recommends), so
/// the map is the writer finally mirroring it.
///
/// A block-0 entry is REFUSED rather than dropped. Dropping it would write a
/// file quietly missing an edit the caller asked for, which is the silent
/// no-op this whole surface is built to avoid; llama.cpp cannot apply one at
/// block 0 in any case.
///
/// It exists as much for the tests as for callers: a reader whose only
/// fixtures come from the writer beside it can agree with that writer while
/// both are wrong, so the round-trip case below is paired with one that
/// checks the BYTES against the layout llama.cpp documents (1-based names, F32,
/// one dimension) rather than against this function.
pub fn write_control_vector(
    directions: &BTreeMap<usize, Vec<f32>>,
    arch: &str,
    mode: Option<SteeringMode>,
) -> Result<Vec<u8>, ControlVectorError> {
    if directions.contains_key(&0) {
        return Err(ControlVectorError::LayerZeroNotExpressible);
    }
    let spanned = directions.keys().next_back().map_or(0, |l| l + 1);
    let mut builder = crate::GgufBuilder::new()
        .metadata_str("general.architecture", arch)
        .metadata_str("controlvector.model_hint", arch)
        .metadata_u32("controlvector.layer_count", spanned as u32)
        // 1, not 0: this file names blocks the way llama.cpp does. The key is
        // stamped rather than omitted so the file says so itself, and reads
        // the same either way.
        .metadata_u32(LAYER_BASE_KEY, DEFAULT_LAYER_BASE as u32);
    if let Some(m) = mode {
        builder = builder.metadata_str(MODE_KEY, m.as_str());
    }
    for (l, values) in directions {
        let mut data = Vec::with_capacity(values.len() * 4);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        builder = builder.tensor(
            &format!("{DIRECTION_PREFIX}.{l}"),
            GGML_TYPE_F32,
            &[values.len() as u64],
            data,
        );
    }
    Ok(builder.build().0)
}

#[cfg(test)]
#[path = "control_vector_tests.rs"]
mod control_vector_tests;
