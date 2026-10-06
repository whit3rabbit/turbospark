//! MOSS audio-text VQ adaptor: Linear -> SiLU -> Linear -> LayerNorm.
//!
//! Reference: `VQAdaptor` in
//! `mlx_audio/stt/models/moss_transcribe_diarize/moss_transcribe_diarize.py`
//! at mlx-audio 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//! Input rows are the time-merge of four consecutive encoder frames
//! (`adaptor_input_dim = d_model * audio_merge_size`); the final layer norm
//! reuses the text config's `rms_norm_eps`, exactly like the reference.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::quant::{is_quantized, QuantScheme};
use crate::stt::qwen3_asr::decoder::Linear;
use crate::{Result, SpeechError};

/// Placeholder scheme for [`Linear::load`]; every adaptor tensor is
/// unquantized in the verified profile, so the plain path never reads it.
const UNUSED_SCHEME: QuantScheme = QuantScheme {
    bits: 4,
    group_size: 64,
};

pub struct VqAdaptor {
    fc: Linear,
    proj: Linear,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    hidden_size: usize,
    input_dim: usize,
    norm_eps: f32,
}

impl VqAdaptor {
    pub fn load(
        file: &SafetensorsFile,
        input_dim: usize,
        hidden_size: usize,
        norm_eps: f32,
    ) -> Result<Self> {
        // The saved keys double the "layers" segment: nn.Sequential nests a
        // layer list inside the adaptor's own "layers" attribute. Index 1 is
        // the parameter-free SiLU.
        let prefix = "model.vq_adaptor.layers.layers";
        for base in [
            format!("{prefix}.0"),
            format!("{prefix}.2"),
            format!("{prefix}.3"),
        ] {
            if is_quantized(file, &base) {
                return Err(SpeechError::Unsupported {
                    why: format!(
                        "{base} carries quantization scales; the verified MOSS profile keeps \
                         the VQ adaptor unquantized"
                    ),
                });
            }
        }
        let norm_weight = load_exact(file, &format!("{prefix}.3.weight"), &[hidden_size])?;
        let norm_bias = load_exact(file, &format!("{prefix}.3.bias"), &[hidden_size])?;
        Ok(Self {
            fc: Linear::load(
                file,
                &format!("{prefix}.0"),
                input_dim,
                hidden_size,
                UNUSED_SCHEME,
            )?,
            proj: Linear::load(
                file,
                &format!("{prefix}.2"),
                hidden_size,
                hidden_size,
                UNUSED_SCHEME,
            )?,
            norm_weight,
            norm_bias,
            hidden_size,
            input_dim,
            norm_eps,
        })
    }

    /// Projects `[rows, adaptor_input_dim]` merged frames to decoder
    /// embeddings `[rows, hidden_size]`.
    pub fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
        debug_assert_eq!(input.len(), rows * self.input_dim);
        let mut hidden = self.fc.forward(input, rows);
        ops::silu(&mut hidden);
        let mut hidden = self.proj.forward(&hidden, rows);
        ops::layernorm(
            &mut hidden,
            rows,
            self.hidden_size,
            &self.norm_weight,
            Some(&self.norm_bias),
            self.norm_eps,
        );
        hidden
    }
}

fn load_exact(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "required tensor is missing".into(),
    })?;
    if descriptor.shape != shape {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected shape {shape:?}, got {:?}", descriptor.shape),
        });
    }
    Ok(file.load_as_f32(name)?)
}
