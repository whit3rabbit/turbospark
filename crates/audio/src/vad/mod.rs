//! Voice Activity Detection (VAD) models.
//!
//! Aligned with `mlx_audio/vad/models/`.

pub mod nemotron_diarization;
pub mod silero_vad;
pub mod sortformer;

pub use nemotron_diarization::NemotronDiarization;
pub use silero_vad::{SileroVad, VadStreamState, VadTiming};
pub use sortformer::{DiarizationOutput, DiarizationSegment, Sortformer};
