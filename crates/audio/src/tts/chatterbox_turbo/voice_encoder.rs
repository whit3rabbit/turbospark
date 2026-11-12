//! The chatterbox_turbo voice encoder (`voice_encoder/`): the 40-mel
//! front end, the partial-utterance windowing, the three-layer LSTM,
//! the projection head with the final ReLU, and the L2 normalization
//! that produces the 256-dim speaker embedding consumed by the T3
//! conditioning.
//!
//! Reference: `chatterbox_turbo/models/voice_encoder/`
//! (`voice_encoder.py`, `melspec.py`, `config.py`) at the pinned commit
//! recorded in the parent module. The forward math is the sibling
//! chatterbox port's (`crate::tts::chatterbox::voice_encoder`, whose
//! mel front end and trim are reused here); this fork exists because
//! the turbo module stacks the LSTMs as named attributes
//! (`lstm1`/`lstm2`/`lstm3` checkpoints) where the sibling nests them
//! in a list (`lstm.layers.N`), and because the reference normalization
//! divides by `norm + 1e-8`, which the port keeps explicitly. The
//! `similarity_weight`/`similarity_bias` checkpoint arrays are unused
//! at inference and not loaded.
//!
//! Scope: single-utterance `embeds_from_wavs` (the reference mean over
//! one utterance's partials is a single mean; multi-utterance batches
//! reduce to repeated calls). The caller passes the 16 kHz waveform;
//! resampling stays outside the port like every other family. The
//! energy trim (`trim_top_db = 20`) is portable RMS math and is
//! included.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::load_tensor;
use crate::{Result, SpeechError};

/// Reference `VoiceEncConfig` defaults (the constructor arguments the
/// fixture may shrink are the hidden and embedding widths).
#[derive(Debug, Clone)]
pub struct VoiceEncConfig {
    pub num_mels: usize,
    pub sample_rate: u32,
    pub speaker_embed_size: usize,
    pub ve_hidden_size: usize,
    /// Frames per partial utterance (1.6 s at 100 frames/s).
    pub ve_partial_frames: usize,
    pub ve_final_relu: bool,
}

impl Default for VoiceEncConfig {
    fn default() -> Self {
        VoiceEncConfig {
            num_mels: 40,
            sample_rate: 16_000,
            speaker_embed_size: 256,
            ve_hidden_size: 256,
            ve_partial_frames: 160,
            ve_final_relu: true,
        }
    }
}

/// One LSTM cell (`nn.LSTM` semantics: sigmoid gates in the PyTorch
/// i, f, g, o order over a tanh cell). `x` is one frame `[input]`,
/// `h` and `c` are the running `[hidden]` states, updated in place.
struct Lstm {
    wx: Vec<f32>,
    wh: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    hidden: usize,
}

impl Lstm {
    fn load(file: &SafetensorsFile, prefix: &str, input: usize, hidden: usize) -> Result<Self> {
        Ok(Lstm {
            wx: load_tensor(file, &format!("{prefix}.Wx"), &[hidden * 4, input])?,
            wh: load_tensor(file, &format!("{prefix}.Wh"), &[hidden * 4, hidden])?,
            bias: load_tensor(file, &format!("{prefix}.bias"), &[hidden * 4])?,
            input,
            hidden,
        })
    }

    fn step(&self, x: &[f32], h: &mut [f32], c: &mut [f32]) {
        let hidden = self.hidden;
        let mut gates = vec![0.0f32; hidden * 4];
        for (gate_index, slot) in gates.iter_mut().enumerate() {
            let wx_row = &self.wx[gate_index * self.input..(gate_index + 1) * self.input];
            let wh_row = &self.wh[gate_index * hidden..(gate_index + 1) * hidden];
            let mut acc = self.bias[gate_index];
            for (i, &w) in wx_row.iter().enumerate() {
                acc += w * x[i];
            }
            for (i, &w) in wh_row.iter().enumerate() {
                acc += w * h[i];
            }
            *slot = acc;
        }
        for i in 0..hidden {
            let i_gate = sigmoid(gates[i]);
            let f_gate = sigmoid(gates[hidden + i]);
            let g = gates[2 * hidden + i].tanh();
            let o_gate = sigmoid(gates[3 * hidden + i]);
            c[i] = f_gate * c[i] + i_gate * g;
            h[i] = o_gate * c[i].tanh();
        }
    }
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// The loaded voice encoder (`ve.` subtree): three stacked LSTMs and
/// the projection head.
pub struct VoiceEncoder {
    config: VoiceEncConfig,
    lstm: Vec<Lstm>,
    proj: Vec<f32>,
    proj_bias: Option<Vec<f32>>,
}

impl VoiceEncoder {
    pub fn load(file: &SafetensorsFile, prefix: &str, config: VoiceEncConfig) -> Result<Self> {
        // The turbo module names the three LSTMs lstm1/lstm2/lstm3.
        let names = ["lstm1", "lstm2", "lstm3"];
        let mut lstm = Vec::new();
        for (layer, name) in names.iter().enumerate() {
            let input = if layer == 0 {
                config.num_mels
            } else {
                config.ve_hidden_size
            };
            lstm.push(Lstm::load(
                file,
                &format!("{prefix}{name}"),
                input,
                config.ve_hidden_size,
            )?);
        }
        let proj = load_tensor(
            file,
            &format!("{prefix}proj.weight"),
            &[config.speaker_embed_size, config.ve_hidden_size],
        )?;
        let proj_bias = if file.contains_tensor(&format!("{prefix}proj.bias")) {
            Some(load_tensor(
                file,
                &format!("{prefix}proj.bias"),
                &[config.speaker_embed_size],
            )?)
        } else {
            None
        };
        Ok(VoiceEncoder {
            config,
            lstm,
            proj,
            proj_bias,
        })
    }

    /// The batch forward (`VoiceEncoder.__call__` over a stack of
    /// partial mel windows): each layer runs the whole sequence with its
    /// own zero-initialized state, feeding the next layer; the last
    /// layer's final hidden state goes through proj, the optional ReLU,
    /// and the `norm + 1e-8` L2 normalization. Returns one embedding per
    /// window.
    fn embed_partials(&self, partials: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let hidden = self.config.ve_hidden_size;
        let mels = self.config.num_mels;
        let layers = self.lstm.len();
        let mut out = Vec::with_capacity(partials.len());
        for partial in partials {
            let steps = partial.len() / mels;
            // Layer 0 consumes the mel frames; deeper layers consume the
            // previous layer's per-frame hidden states.
            let mut input = partial.clone();
            let mut width = mels;
            let mut last_hidden = vec![0.0f32; hidden];
            for (layer_index, layer) in self.lstm.iter().enumerate() {
                let mut h = vec![0.0f32; hidden];
                let mut c = vec![0.0f32; hidden];
                let mut sequence = Vec::with_capacity(steps * hidden);
                for t in 0..steps {
                    let frame = &input[t * width..(t + 1) * width];
                    layer.step(frame, &mut h, &mut c);
                    sequence.extend_from_slice(&h);
                }
                input = sequence;
                width = hidden;
                if layer_index == layers - 1 {
                    last_hidden = h;
                }
            }
            let mut raw = vec![0.0f32; self.config.speaker_embed_size];
            for (o, w_row) in self.proj.chunks_exact(hidden).enumerate() {
                let mut acc = self.proj_bias.as_ref().map_or(0.0, |b| b[o]);
                for (i, &w) in w_row.iter().enumerate() {
                    acc += w * last_hidden[i];
                }
                raw[o] = acc;
            }
            if self.config.ve_final_relu {
                for v in &mut raw {
                    *v = v.max(0.0);
                }
            }
            let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
            for v in &mut raw {
                *v /= norm;
            }
            out.push(raw);
        }
        out
    }

    /// `embeds_from_wavs` for one utterance: trim (default on), mel, the
    /// partial window grid, the LSTM stack, the partial-embedding mean,
    /// and the final `norm + 1e-8` L2 normalization. Returns the speaker
    /// embedding.
    pub fn embeds_from_wav(&self, wav: &[f32], trim_top_db: Option<f32>) -> Result<Vec<f32>> {
        let trimmed = match trim_top_db {
            None => wav.to_vec(),
            Some(top_db) => crate::tts::chatterbox::voice_encoder::trim_silence(wav, top_db),
        };
        let frames = crate::tts::chatterbox::voice_encoder::melspectrogram(&trimmed)?;
        let n_frames = frames.len();
        if n_frames == 0 {
            return Err(SpeechError::Input {
                why: "waveform is shorter than one mel frame".to_string(),
            });
        }
        let step = frame_step(&self.config, 0.5, Some(1.3))?;
        let (n_partials, target_len) = num_wins(n_frames, step, &self.config);
        // Pad the mel rows with zeros to the target length.
        let mut mel = vec![0.0f32; target_len * self.config.num_mels];
        for (t, frame) in frames.iter().take(target_len).enumerate() {
            mel[t * self.config.num_mels..(t + 1) * self.config.num_mels]
                .copy_from_slice(&frame[..self.config.num_mels]);
        }
        // Extract the partials.
        let mut partials = Vec::with_capacity(n_partials);
        for p in 0..n_partials {
            let start = p * step;
            let mut window = vec![0.0f32; self.config.ve_partial_frames * self.config.num_mels];
            for t in 0..self.config.ve_partial_frames {
                let src = (start + t) * self.config.num_mels;
                let dst = t * self.config.num_mels;
                window[dst..dst + self.config.num_mels]
                    .copy_from_slice(&mel[src..src + self.config.num_mels]);
            }
            partials.push(window);
        }
        let embeds = self.embed_partials(&partials);
        // Mean over the per-partial L2-normalized embeddings, then a
        // final L2 normalization.
        let dim = self.config.speaker_embed_size;
        let mut speaker = vec![0.0f32; dim];
        for embed in &embeds {
            for (s, v) in speaker.iter_mut().zip(embed) {
                *s += v;
            }
        }
        for v in &mut speaker {
            *v /= embeds.len() as f32;
        }
        let norm = speaker.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
        for v in &mut speaker {
            *v /= norm;
        }
        Ok(speaker)
    }
}

/// `get_frame_step` (`overlap = 0.5`, `rate = 1.3`, the
/// `embeds_from_wavs` defaults): `int(np.round((sample_rate / rate) /
/// ve_partial_frames))`.
fn frame_step(config: &VoiceEncConfig, overlap: f32, rate: Option<f32>) -> Result<usize> {
    if !(0.0..1.0).contains(&overlap) {
        return Err(SpeechError::Input {
            why: format!("overlap must be in [0, 1), got {overlap}"),
        });
    }
    let step = match rate {
        None => (config.ve_partial_frames as f32 * (1.0 - overlap)).round() as usize,
        Some(rate) => {
            ((config.sample_rate as f32 / rate) / config.ve_partial_frames as f32).round() as usize
        }
    };
    if step == 0 || step > config.ve_partial_frames {
        return Err(SpeechError::Input {
            why: format!(
                "frame step {step} outside (0, {}]",
                config.ve_partial_frames
            ),
        });
    }
    Ok(step)
}

/// `get_num_wins`: partials covering `n_frames` at `step` with the
/// `min_coverage = 0.8` tail rule:
/// `divmod(max(n_frames - win_size + step, 0), step)`; one more window
/// when the remainder tail covers at least 80 percent of a window.
fn num_wins(n_frames: usize, step: usize, config: &VoiceEncConfig) -> (usize, usize) {
    let win_size = config.ve_partial_frames;
    let total = (n_frames + step).saturating_sub(win_size);
    let mut n_wins = total / step;
    let remainder = total % step;
    let coverage = (remainder + (win_size - step)) as f32 / win_size as f32;
    if n_wins == 0 || coverage >= 0.8 {
        n_wins += 1;
    }
    let target_n = win_size + step * (n_wins - 1);
    (n_wins, target_n)
}
