//! GGUF repack error types, basic data structures, and type tag mappings.

use model_io::ArchConfig;

use crate::gguf_config::GgufConfigError;
use crate::gguf_header::{ggml_type_name, GgufHeader, GgufHeaderError};
use crate::gguf_names::GgufNameError;
use crate::gturbo_writer::{LayerBlobs, WriterError};
use crate::ranged_download::{DownloadError, RangeSource};
use crate::resident_writer::ResidentEntrySpec;

/// ggml's F32 type id. The one source type this walk does not carry through
/// verbatim; see [`transcode_f32`].
pub const GGML_TYPE_F32: u32 = 0;

/// Which half of a fused `ffn_gate_up_exps` tensor is the gate.
pub const FUSED_GATE_FIRST: bool = true;

#[derive(Debug)]
pub enum GgufRepackError {
    Config(GgufConfigError),
    Header(GgufHeaderError),
    Name(GgufNameError),
    Download(DownloadError),
    Writer(WriterError),
    UnsupportedFamily {
        family: &'static str,
    },
    /// A ggml type with no `.gturbo` dtype tag.
    UnsupportedType {
        tensor: String,
        ggml_type: u32,
    },
    ShapeMismatch {
        tensor: String,
        detail: String,
    },
    MissingTensor {
        name: String,
    },
    Io {
        path: String,
        detail: String,
    },
}

impl std::fmt::Display for GgufRepackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufRepackError::Config(e) => write!(f, "{e}"),
            GgufRepackError::Header(e) => write!(f, "{e}"),
            GgufRepackError::Name(e) => write!(f, "{e}"),
            GgufRepackError::Download(e) => write!(f, "{e}"),
            GgufRepackError::Writer(e) => write!(f, "{e}"),
            GgufRepackError::UnsupportedFamily { family } => {
                write!(f, "no GGUF repack path for family {family}")
            }
            GgufRepackError::UnsupportedType { tensor, ggml_type } => write!(
                f,
                "tensor {tensor}: ggml type {} (id {ggml_type}) has no .gturbo dtype tag",
                ggml_type_name(*ggml_type).unwrap_or("?")
            ),
            GgufRepackError::ShapeMismatch { tensor, detail } => {
                write!(f, "tensor {tensor}: {detail}")
            }
            GgufRepackError::MissingTensor { name } => write!(f, "missing tensor {name}"),
            GgufRepackError::Io { path, detail } => write!(f, "{path}: {detail}"),
        }
    }
}

impl std::error::Error for GgufRepackError {}

macro_rules! from_error {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        $(impl From<$ty> for GgufRepackError {
            fn from(e: $ty) -> Self {
                GgufRepackError::$variant(e)
            }
        })*
    };
}
from_error! {
    GgufConfigError => Config,
    GgufHeaderError => Header,
    GgufNameError => Name,
    DownloadError => Download,
    WriterError => Writer,
}

/// `.gturbo` resident-index dtype tag for a ggml type.
pub fn dtype_tag_for_ggml_type(ggml_type: u32) -> Option<u8> {
    use crate::resident_writer::{
        DTYPE_BF16, DTYPE_FP16, DTYPE_FP32, DTYPE_GGUF_IQ3_XXS, DTYPE_GGUF_IQ4_NL,
        DTYPE_GGUF_IQ4_XS, DTYPE_GGUF_Q4_0, DTYPE_GGUF_Q4_K, DTYPE_GGUF_Q6_K, DTYPE_GGUF_Q8_0,
    };
    Some(match ggml_type {
        0 => DTYPE_FP32,
        1 => DTYPE_FP16,
        30 => DTYPE_BF16,
        2 => DTYPE_GGUF_Q4_0,
        8 => DTYPE_GGUF_Q8_0,
        12 => DTYPE_GGUF_Q4_K,
        14 => DTYPE_GGUF_Q6_K,
        18 => DTYPE_GGUF_IQ3_XXS,
        20 => DTYPE_GGUF_IQ4_NL,
        23 => DTYPE_GGUF_IQ4_XS,
        _ => return None,
    })
}

/// The `manifest.json -> quant` slot name for a ggml type, for the manifest
/// this walk writes.
pub fn ggml_scheme_name(ggml_type: u32) -> &'static str {
    ggml_type_name(ggml_type).unwrap_or("unknown")
}

/// GGUF stores dims fastest-varying first; the resident index stores logical
/// shape. Reversing is the whole conversion.
pub fn logical_shape(dims: &[u64]) -> (u32, u32, u32, u32) {
    let mut out = [0u32; 4];
    for (slot, d) in out.iter_mut().zip(dims.iter().rev()) {
        *slot = *d as u32;
    }
    (out[0], out[1], out[2], out[3])
}

pub fn read_tensor(
    header: &GgufHeader,
    source: &dyn RangeSource,
    name: &str,
) -> Result<Vec<u8>, GgufRepackError> {
    let (start, end) =
        header
            .absolute_range(name)
            .ok_or_else(|| GgufRepackError::MissingTensor {
                name: name.to_string(),
            })??;
    Ok(source.read_range(start, end)?)
}

/// Everything the writer needs, for callers that want to inspect the plan
/// before it hits disk.
pub struct GgufRepackOutput {
    pub arch: ArchConfig,
    pub resident: Vec<ResidentEntrySpec>,
    pub layers: Vec<LayerBlobs>,
    pub expert_stride: u64,
    /// Tensors recognized and deliberately not carried, with the reason.
    pub ignored: Vec<String>,
    pub lossy_narrowing: Vec<(String, usize)>,
}
