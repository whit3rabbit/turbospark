//! OmniVoice voice cloning on the native audio session.
//!
//! With [`Device::Metal`], the bidirectional Qwen3 backbone's projections and
//! non-causal GQA attention plus the 8 audio prediction heads run on
//! `gpu::Music3Device` (tiled float GEMM, batched non-causal attention);
//! remaining operations execute on the CPU. With [`Device::Cpu`], execution
//! stays portable f32.

use std::path::Path;
use std::rc::Rc;

use audio::backend::{
    AttentionShape, ComputeBackend, ConvShape, DType, DeviceWeight, WeightData, WeightEncoding,
};
use audio::tts::omnivoice::pipeline::{GenerateParams, OmniVoicePipeline};
use audio::{Result as AudioResult, SpeechError};
use gpu::{Music3Device, Music3Encoding, Music3Weight};

/// Where the backbone and audio heads run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal,
}

fn metal_error(e: gpu::GpuError) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("OmniVoice Metal: {e}"),
    }
}

/// The device: F32 weights only, tiled float GEMM, and batched non-causal
/// attention.
struct MetalBackend {
    device: Music3Device,
}

struct MetalWeight {
    inner: Music3Weight,
}

impl ComputeBackend for MetalBackend {
    fn load_weight(&self, data: WeightData<'_>) -> AudioResult<Rc<dyn DeviceWeight>> {
        if data.encoding != WeightEncoding::F32 || data.dtype != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "OmniVoice Metal loads its F32 checkpoint only".into(),
            });
        }
        let inner = self
            .device
            .load_weight(data.shape, data.bytes, Music3Encoding::F32, &[], &[], &[])
            .map_err(metal_error)?;
        Ok(Rc::new(MetalWeight { inner }))
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: AttentionShape,
        d: DType,
    ) -> AudioResult<Vec<f32>> {
        if d != DType::F32 || s.causal || s.kv_time_major || s.offset != 0 || s.queries != s.keys {
            return Err(SpeechError::Unsupported {
                why: "OmniVoice attention is non-causal F32 self-attention".into(),
            });
        }
        self.device
            .attention_typed(
                q,
                k,
                v,
                s.batch,
                s.queries,
                s.keys,
                s.heads,
                s.kv_heads,
                s.dim,
                false,
                false,
                0,
                gpu::Music3DType::F32,
            )
            .map_err(metal_error)
    }
}

impl DeviceWeight for MetalWeight {
    fn linear(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input: usize,
        output: usize,
        d: DType,
    ) -> AudioResult<Vec<f32>> {
        if d != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "OmniVoice linear must remain F32".into(),
            });
        }
        self.inner
            .linear_f32(x, bias, rows, input, output)
            .map_err(metal_error)
    }

    fn embedding(&self, _ids: &[i32], _width: usize) -> AudioResult<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "OmniVoice Metal has no embeddings".into(),
        })
    }

    fn convolution(
        &self,
        _x: &[f32],
        _bias: Option<&[f32]>,
        _s: ConvShape,
        _d: DType,
    ) -> AudioResult<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "OmniVoice Metal has no convolutions".into(),
        })
    }
}

/// Persisted family string of the OmniVoice engine (the checkpoint
/// config's `model_type`).
pub const FAMILY: &str = "omnivoice";

/// The bounds the engine places on the instruction block (the
/// `<|instruct_start|>` string); the reference passes it through
/// unvalidated, the session refuses clearly pathological input.
const MAX_INSTRUCTION_BYTES: usize = 512;

/// One synthesis request, validated by
/// [`crate::native_audio::omnivoice_request`].
pub struct OmniVoiceRequest<'a> {
    pub text: &'a str,
    /// ISO language code for the `<|lang_*|>` tag; `None` uses the
    /// reference's unset `"None"` marker.
    pub language: Option<&'a str>,
    /// Voice-design instruction block, when the caller gave one.
    pub instruction: Option<&'a str>,
    /// Mono reference audio at `reference_rate`, for cloning.
    pub reference: Option<&'a [f32]>,
    pub reference_rate: u32,
    /// The reference transcript prepended to the text (`_combine_text`).
    pub reference_text: Option<&'a str>,
    pub steps: usize,
    pub speed: f32,
    pub seed: u64,
}

/// OmniVoice runner with reference voice caching and optional Metal acceleration.
pub struct OmniVoiceRunner {
    pipeline: OmniVoicePipeline,
    device: Device,
    /// Content hash of the last reference and its encoded token rows.
    voice: Option<(u64, Vec<Vec<i32>>)>,
}

impl OmniVoiceRunner {
    /// Opens an OmniVoice install (`config.json`, `mlx_manifest.json`,
    /// `model.safetensors`, `tokenizer.json`, `audio_tokenizer/`).
    pub fn open(install: &Path, device: Device) -> Result<Self, String> {
        let mut pipeline = OmniVoicePipeline::open(install).map_err(|e| e.to_string())?;
        if device == Device::Metal {
            let dev = Music3Device::new().map_err(|e| e.to_string())?;
            let backend = Rc::new(MetalBackend { device: dev });
            pipeline.set_backend(Some(backend));
        }
        Ok(OmniVoiceRunner {
            pipeline,
            device,
            voice: None,
        })
    }

    /// The execution device selected at open.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The checkpoint's output rate (24 kHz).
    pub fn sample_rate(&self) -> u32 {
        self.pipeline.sample_rate()
    }

    /// The encoded reference code rows through the voice cache.
    pub fn encode_reference(
        &mut self,
        samples: &[f32],
        rate: u32,
    ) -> Result<Vec<Vec<i32>>, String> {
        self.reference(samples, rate)
    }

    /// The encoded reference code rows, from the cache when the same
    /// audio was encoded last. The session hands the reference in at
    /// `reference_rate`; it is resampled to the checkpoint rate with the
    /// crate's sinc resampler before the Higgs encode.
    fn reference(&mut self, samples: &[f32], rate: u32) -> Result<Vec<Vec<i32>>, String> {
        if samples.is_empty() {
            return Err("reference audio cannot be empty".into());
        }
        if rate == 0 {
            return Err("reference audio needs a sample rate".into());
        }
        if samples.iter().any(|v| !v.is_finite()) {
            return Err("reference audio must contain finite samples".into());
        }
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        rate.hash(&mut hasher);
        samples.len().hash(&mut hasher);
        for value in samples {
            value.to_bits().hash(&mut hasher);
        }
        let key = hasher.finish();
        if let Some((cached, tokens)) = &self.voice {
            if *cached == key {
                return Ok(tokens.clone());
            }
        }
        let target = self.pipeline.sample_rate();
        let mono = if rate == target {
            samples.to_vec()
        } else {
            let wave = audio::Waveform {
                sample_rate: rate,
                channels: 1,
                samples: samples.to_vec(),
            };
            audio::to_mono_resampled(
                &wave,
                target,
                &audio::MonoResampleStrategy::SincHann(Default::default()),
            )
            .map_err(|e| e.to_string())?
        };
        let tokens = self
            .pipeline
            .encode_reference(&mono)
            .map_err(|e| e.to_string())?;
        self.voice = Some((key, tokens.clone()));
        Ok(tokens)
    }

    /// Synthesizes one utterance, emitting the whole waveform when the
    /// diffusion finishes (the pipeline is non-streaming) and reporting
    /// per-step progress through `on_progress`. `on_progress` returns
    /// false to cancel.
    pub fn synthesize(
        &mut self,
        request: &OmniVoiceRequest,
        on_progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<(usize, Vec<f32>), String> {
        if request.text.trim().is_empty() {
            return Err("speech text cannot be empty".into());
        }
        if request.reference.is_none()
            && request
                .reference_text
                .as_deref()
                .is_some_and(|s| !s.trim().is_empty())
        {
            return Err("reference_text needs reference audio".into());
        }
        if let Some(instruction) = request.instruction {
            validate_instruction(instruction)?;
        }
        let reference = match (request.reference, request.reference_rate) {
            (Some(s), _) if s.is_empty() => return Err("reference audio cannot be empty".into()),
            (Some(_), 0) => return Err("reference audio needs a sample rate".into()),
            (Some(samples), rate) => Some(self.reference(samples, rate)?),
            (None, _) => None,
        };
        let params = GenerateParams {
            language: request.language.unwrap_or("None").to_string(),
            instruct: request.instruction.unwrap_or("None").to_string(),
            steps: request.steps,
            speed: request.speed,
            seed: request.seed,
            ..GenerateParams::default()
        };
        let (tokens, audio) = self
            .pipeline
            .generate(
                request.text,
                request.reference_text,
                reference.as_deref(),
                None,
                &params,
                Some(on_progress),
            )
            .map_err(|e| e.to_string())?;
        Ok((tokens.len(), audio))
    }
}

/// The safetensors byte count of an install plus a fixed overhead, for
/// open admission. The 8-bit checkpoint widens to f32 at load (roughly
/// four times the packed bytes) and the released tokenizer file is
/// already f32; counting everything four times is an upper bound.
pub fn open_bytes(install: &Path) -> Result<u64, String> {
    fn sum_safetensors(path: &std::path::Path) -> Result<u64, String> {
        let mut total = 0u64;
        for entry in std::fs::read_dir(path).map_err(|e| e.to_string())? {
            let file = entry.map_err(|e| e.to_string())?.path();
            if file.extension().is_some_and(|ext| ext == "safetensors") {
                if let Ok(meta) = file.metadata() {
                    total = total.saturating_add(meta.len());
                }
            }
        }
        Ok(total)
    }
    let root = sum_safetensors(install)?;
    let tokenizer_dir = install.join("audio_tokenizer");
    let tokenizer = if tokenizer_dir.is_dir() {
        sum_safetensors(&tokenizer_dir)?
    } else {
        0
    };
    Ok(root
        .saturating_mul(4)
        .saturating_add(tokenizer)
        .saturating_add(512 << 20))
}

/// The engine-side instruction bound, applied to the raw request before
/// the pipeline assembles the style prefix.
pub fn validate_instruction(instruction: &str) -> Result<(), String> {
    if instruction.len() > MAX_INSTRUCTION_BYTES {
        return Err(format!(
            "instruction {} bytes is outside 0 through {MAX_INSTRUCTION_BYTES}",
            instruction.len()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_constant_is_omnivoice() {
        assert_eq!(FAMILY, "omnivoice");
    }

    #[test]
    fn validate_instruction_bounds() {
        assert!(validate_instruction("hello").is_ok());
        assert!(validate_instruction(&"a".repeat(MAX_INSTRUCTION_BYTES)).is_ok());
        assert!(validate_instruction(&"a".repeat(MAX_INSTRUCTION_BYTES + 1)).is_err());
    }
}
