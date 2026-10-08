//! Safe shared audio device contract. Model sequencing stays portable.
//! Music 3 retains its original module path for source compatibility.
pub use crate::music::minimax_music3::backend::{
    AttentionShape, ComputeBackend, ConvShape, DeviceWeight, NormalizationLayout, RopeShape,
    WeightData, WeightEncoding,
};
pub use crate::music::minimax_music3::precision::DType;
