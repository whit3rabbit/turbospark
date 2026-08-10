//! Assembles a byte-exact `.gturbo` install directory: `packed_experts/
//! layer_NN.bin` blobs (matching `turbospark_model_io::PackedExpertsLayout`'s
//! per-expert sub-tensor byte layout), `packed_experts/layout.json`, a
//! minimal `model_weights.bin` (a valid, empty `ResidentIndex` header
//! followed by a raw tensor-data region), and `manifest.json` with computed
//! per-file SHA-256 checksums. This is the piece Phase 8's "streaming
//! installer" work item was missing: given already-quantized tensor bytes
//! (from `repack::quantize_matrix_int4`/`_int8` or any other source), it
//! writes an install that `turbospark_model_io::load_manifest`/
//! `load_packed_experts_layout`/`load_resident_index` can read back.
//!
//! Building the tensors themselves from a downloaded HF checkpoint (walking
//! `safetensors` files, slicing by role, quantizing every projection) is
//! still the caller's job; this module is the on-disk assembly step that
//! sits after it.

pub mod layers;
pub mod manifest;
pub mod streaming;
pub mod types;

pub use layers::{
    write_gturbo_install, write_gturbo_install_with_resident_index,
    write_gturbo_install_with_resident_index_and_experts,
};
pub use streaming::StreamingGturboWriter;
pub use types::{ExpertBlob, LayerBlobs, SubTensor, WriterError};
