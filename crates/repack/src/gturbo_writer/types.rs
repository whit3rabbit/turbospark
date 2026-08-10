//! Error and blob data types for `.gturbo` install writing.

use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum WriterError {
    Io {
        path: String,
        detail: String,
    },
    ExpertOversized {
        layer: usize,
        expert: usize,
        used: u64,
        stride: u64,
    },
    WrongExpertCount {
        layer: usize,
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for WriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriterError::Io { path, detail } => write!(f, "{path}: {detail}"),
            WriterError::ExpertOversized {
                layer,
                expert,
                used,
                stride,
            } => write!(
                f,
                "layer {layer} expert {expert} uses {used} bytes, exceeding the {stride}-byte expert stride"
            ),
            WriterError::WrongExpertCount {
                layer,
                expected,
                actual,
            } => {
                write!(f, "layer {layer} has {actual} experts, expected {expected}")
            }
        }
    }
}

impl std::error::Error for WriterError {}

/// One named sub-tensor inside an expert's blob (e.g. `"gate"`,
/// `"gate_scales"`, `"gate_biases"`).
#[derive(Debug, Clone)]
pub struct SubTensor {
    pub role: String,
    pub bytes: Vec<u8>,
    pub dtype: String,
    pub shape: Vec<u64>,
}

/// One expert's full set of sub-tensors, written back to back (zero-padded
/// to `expert_stride`) inside its layer file.
#[derive(Debug, Clone)]
pub struct ExpertBlob {
    pub expert: usize,
    pub sub_tensors: Vec<SubTensor>,
}

/// One `packed_experts/layer_NN.bin` file's worth of experts.
#[derive(Debug, Clone)]
pub struct LayerBlobs {
    pub layer: usize,
    pub experts: Vec<ExpertBlob>,
}

pub(crate) fn io_err(path: &Path, e: std::io::Error) -> WriterError {
    WriterError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    }
}
