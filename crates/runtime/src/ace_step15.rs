//! ACE-Step 1.5 turbo text2music on the native audio worker.
//!
//! The pipeline is `audio::music::ace_step15`. With Metal, the DiT and
//! condition-encoder projections live on `gpu::Music3Device` as resident
//! bf16 weights (tiled float32 GEMM), unmasked attention runs there, and
//! the VAE convolutions run there as F32 weights; norms, RoPE, residuals,
//! windowed attention, Snake and the text encoder stay on the host. The
//! portable path runs everything on the host (slow; correct).
//!
//! Defaults follow upstream's turbo recipe: 8 steps at shift 3 and the
//! DCW sampler correction upstream enables for turbo.

use std::path::Path;
use std::rc::Rc;

use audio::backend::{
    AttentionShape, ComputeBackend, ConvShape, DType, DeviceWeight, WeightData, WeightEncoding,
};
use audio::music::ace_step15::{AceStep15, Dcw, Stage, Text2Music, SAMPLE_RATE};
use audio::{Result as AudioResult, SpeechError};
use gpu::{Music3DType, Music3Device, Music3Encoding, Music3Weight};

/// The persisted family string.
pub const FAMILY: &str = "ace_step15";
/// Turbo runs a fixed 8-step schedule.
pub const STEPS: usize = 8;
/// Upstream's recommended turbo shift.
pub const SHIFT: f32 = 3.0;

/// Whether `path` is an ACE-Step 1.5 checkpoint root: the turbo DiT
/// (with its converted silence latent), the VAE and the text encoder.
pub fn detect(path: &Path) -> bool {
    let dit = path.join("acestep-v15-turbo");
    dit.join("config.json").is_file()
        && dit.join("model.safetensors").is_file()
        && path.join("vae").join("config.json").is_file()
        && path
            .join("Qwen3-Embedding-0.6B")
            .join("tokenizer.json")
            .is_file()
}

fn metal_error(e: gpu::GpuError) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("ACE-Step Metal: {e}"),
    }
}

/// Dense 16- or 32-bit weights computed in F32, F32 non-causal
/// attention, F32 convolutions. Anything else is refused, not
/// approximated.
struct MetalBackend {
    device: Music3Device,
}
struct MetalWeight(Music3Weight);

impl ComputeBackend for MetalBackend {
    fn load_weight(&self, data: WeightData<'_>) -> AudioResult<Rc<dyn DeviceWeight>> {
        let encoding = match data.encoding {
            WeightEncoding::Bf16 => Music3Encoding::Bf16,
            WeightEncoding::F16 => Music3Encoding::F16,
            WeightEncoding::F32 => Music3Encoding::F32,
            other => {
                return Err(SpeechError::Unsupported {
                    why: format!("ACE-Step Metal loads dense weights only, got {other:?}"),
                })
            }
        };
        if data.dtype != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "ACE-Step Metal computes in F32".into(),
            });
        }
        let w = self
            .device
            .load_weight(data.shape, data.bytes, encoding, &[], &[], &[])
            .map_err(metal_error)?;
        Ok(Rc::new(MetalWeight(if data.tiled_f32 {
            w.tiled_for_f32()
        } else {
            w
        })))
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: AttentionShape,
        d: DType,
    ) -> AudioResult<Vec<f32>> {
        if d != DType::F32 || s.causal || s.kv_time_major || s.offset != 0 {
            return Err(SpeechError::Unsupported {
                why: "ACE-Step attention is non-causal F32".into(),
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
                Music3DType::F32,
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
                why: "ACE-Step linear is F32".into(),
            });
        }
        self.0
            .linear_typed(x, bias, rows, input, output, Music3DType::F32)
            .map_err(metal_error)
    }
    fn embedding(&self, _ids: &[i32], _width: usize) -> AudioResult<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "ACE-Step has no device embeddings".into(),
        })
    }
    fn convolution(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        d: DType,
    ) -> AudioResult<Vec<f32>> {
        if d != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "ACE-Step VAE convolution is F32".into(),
            });
        }
        self.0
            .convolution_f32_grouped(
                x,
                bias,
                s.input_channels,
                s.output_channels,
                s.kernel,
                s.stride,
                s.padding,
                s.dilation,
                s.transpose,
                1,
            )
            .map_err(metal_error)
    }
}

/// Where the heavy operations run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal,
}

/// One text2music request as the audio session hands it over.
#[derive(Debug, Clone)]
pub struct Request<'a> {
    pub caption: &'a str,
    pub lyrics: &'a str,
    /// Vocal language code for the lyric prompt; upstream defaults to `en`.
    pub language: &'a str,
    pub duration_seconds: f64,
    pub seed: u64,
}

/// The rendered song, interleaved stereo at [`SAMPLE_RATE`].
#[derive(Debug, Clone)]
pub struct Rendered {
    pub interleaved: Vec<f32>,
    pub sample_rate: u32,
    pub caption_truncated: bool,
    pub lyrics_truncated: bool,
}

pub struct AceStep15Runner {
    model: AceStep15,
    device: Device,
}

impl AceStep15Runner {
    /// Opens `path` on Metal, or on the host when `device` is
    /// [`Device::Cpu`].
    pub fn open(path: &Path, device: Device) -> AudioResult<Self> {
        if !detect(path) {
            return Err(SpeechError::BadConfig {
                field: path.display().to_string(),
                why: "not an ACE-Step 1.5 checkpoint root (acestep-v15-turbo/, vae/, \
                      Qwen3-Embedding-0.6B/)"
                    .into(),
            });
        }
        let backend: Option<Rc<dyn ComputeBackend>> = match device {
            Device::Cpu => None,
            Device::Metal => Some(Rc::new(MetalBackend {
                device: Music3Device::new().map_err(metal_error)?,
            })),
        };
        Ok(Self {
            model: AceStep15::load(path, backend)?,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Renders one request. `progress` sees each stage first and cancels
    /// by returning false (`Ok(None)`).
    pub fn generate(
        &mut self,
        req: &Request<'_>,
        progress: impl FnMut(Stage) -> bool,
    ) -> AudioResult<Option<Rendered>> {
        let request = Text2Music {
            caption: req.caption,
            lyrics: req.lyrics,
            language: req.language,
            bpm: None,
            key_scale: "",
            time_signature: "",
            duration_seconds: req.duration_seconds,
            shift: SHIFT,
            seed: req.seed,
            noise: None,
            dcw: Some(Dcw::TURBO),
        };
        let Some(out) = self.model.text2music(&request, progress)? else {
            return Ok(None);
        };
        let n = out.samples;
        let mut interleaved = Vec::with_capacity(2 * n);
        for i in 0..n {
            interleaved.push(out.audio[i]);
            interleaved.push(out.audio[n + i]);
        }
        Ok(Some(Rendered {
            interleaved,
            sample_rate: SAMPLE_RATE,
            caption_truncated: out.caption_truncated,
            lyrics_truncated: out.lyrics_truncated,
        }))
    }
}

/// Open-time memory estimate for admission: the DiT checkpoint (its
/// projections become device bf16, the rest host f32), the text encoder
/// widened to f32 on the host, the VAE as host f32 plus its device copy and
/// phase-split transposed weights, and a 512 MiB reserve. Checked against a
/// measured session footprint in the runtime tests.
pub fn open_bytes(path: &Path) -> Result<u64, String> {
    let size = |p: std::path::PathBuf| -> Result<u64, String> {
        std::fs::metadata(&p)
            .map(|m| m.len())
            .map_err(|e| format!("{}: {e}", p.display()))
    };
    let dit = size(path.join("acestep-v15-turbo").join("model.safetensors"))?;
    let text = size(path.join("Qwen3-Embedding-0.6B").join("model.safetensors"))?;
    let vae = size(path.join("vae").join("diffusion_pytorch_model.safetensors"))?;
    Ok(dit
        .saturating_add(text.saturating_mul(2))
        .saturating_add(vae.saturating_mul(3))
        .saturating_add(512 << 20))
}
