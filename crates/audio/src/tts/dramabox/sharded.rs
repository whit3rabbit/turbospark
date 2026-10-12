//! Shared sharded-tensor loaders for the dramabox family modules.

use crate::nn::Linear;
use crate::{Result, SpeechError};
use turbospark_model_io::safetensors::SafetensorsFile;

/// Load a tensor from the first shard that carries it, with a shape check.
pub(crate) fn load_sharded(
    files: &[SafetensorsFile],
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>> {
    let file = files
        .iter()
        .find(|file| file.contains_tensor(name))
        .ok_or_else(|| SpeechError::Tensor {
            name: name.to_owned(),
            why: "tensor is missing".into(),
        })?;
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape != shape {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected shape {shape:?}, got {:?}", descriptor.shape),
        });
    }
    Ok(file.load_as_f32(name)?)
}

/// Loads one biased linear `[out, in]` plus its bias.
pub(crate) fn load_linear_sharded(
    files: &[SafetensorsFile],
    name: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    let weight = load_sharded(files, &format!("{name}.weight"), &[output, input])?;
    let bias = load_sharded(files, &format!("{name}.bias"), &[output])?;
    Ok(Linear::new(weight, Some(bias), input, output))
}
