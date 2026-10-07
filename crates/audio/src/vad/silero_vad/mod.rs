//! Silero VAD: speech/non-speech probabilities and speech timestamps.
//!
//! Reference: `mlx_audio/vad/models/silero_vad/` (silero_vad.py,
//! config.py) at the cloned v0.5.7 revision. One branch per sample rate
//! (16 kHz and 8 kHz): a learned-STFT convolution (kernel = filter
//! length, stride = hop) producing `2 * cutoff` channels that split into
//! real and imaginary halves and magnitude-compress, four ReLU convs
//! downsampling to one time step, a 128-wide LSTM whose hidden state is
//! the streaming state, and a 1x1 conv + sigmoid whose time mean is the
//! speech probability.
//!
//! The checkpoint is the mlx-community/silero-vad conversion (f32,
//! tensor names under `vad_16k.` / `vad_8k.`). All layer order, eps-free
//! math, and the timestamp hysteresis below are transcribed from the
//! reference; the reflect padding starts at the second-to-last sample
//! (`n - 2`), which is the reference's own convention, not a typo.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::{Result, SpeechError};

/// Streaming state for one branch: LSTM hidden and cell, each `[128]`
/// for a batch of one, plus the trailing context samples.
pub struct VadStreamState {
    pub hidden: Vec<f32>,
    pub cell: Vec<f32>,
    pub context: Vec<f32>,
    pub sample_rate: u32,
}

/// Tunables the timestamp conversion applies on top of the raw
/// probabilities; defaults mirror the reference config.
#[derive(Debug, Clone)]
pub struct VadTiming {
    pub threshold: f32,
    pub min_speech_duration_ms: u32,
    pub min_silence_duration_ms: u32,
    pub speech_pad_ms: u32,
}

impl Default for VadTiming {
    fn default() -> Self {
        VadTiming {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
        }
    }
}

/// One sample-rate branch's weights, all f32.
struct Branch {
    #[allow(
        dead_code,
        reason = "mirrors the reference config; kept for loader symmetry"
    )]
    sample_rate: u32,
    filter_length: usize,
    hop_length: usize,
    pad: usize,
    cutoff: usize,
    context_size: usize,
    chunk_size: usize,
    stft_conv: Vec<f32>,
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    conv2_w: Vec<f32>,
    conv2_b: Vec<f32>,
    conv3_w: Vec<f32>,
    conv3_b: Vec<f32>,
    conv4_w: Vec<f32>,
    conv4_b: Vec<f32>,
    lstm_ih: Vec<f32>,
    lstm_hh: Vec<f32>,
    lstm_ih_bias: Vec<f32>,
    lstm_hh_bias: Vec<f32>,
    final_conv_w: Vec<f32>,
    final_conv_b: Vec<f32>,
}

impl Branch {
    /// The 16 kHz defaults from the reference config.
    fn config_16k() -> (u32, usize, usize, usize, usize, usize, usize) {
        (16_000, 256, 128, 64, 129, 64, 512)
    }

    /// The 8 kHz defaults from the reference config.
    fn config_8k() -> (u32, usize, usize, usize, usize, usize, usize) {
        (8_000, 128, 64, 32, 65, 32, 256)
    }

    fn load(file: &SafetensorsFile, prefix: &str) -> Result<Self> {
        let (sample_rate, filter_length, hop_length, pad, cutoff, context_size, chunk_size) =
            if prefix.ends_with("vad_16k") {
                Self::config_16k()
            } else {
                Self::config_8k()
            };
        let lstm_dim = 128usize;
        // MLX conv weights store [out_ch, kernel, in_ch]; the kernels read
        // PyTorch's [out_ch, in_ch, kernel], so every conv load transposes
        // the last two dims.
        let conv_shape_check =
            |name: &str, got: &[f32], out_ch: usize, in_ch: usize, kernel: usize| -> Result<()> {
                if got.len() != out_ch * in_ch * kernel {
                    return Err(SpeechError::Tensor {
                        name: name.to_string(),
                        why: format!(
                            "expected {out_ch}x{in_ch}x{kernel} = {}, got {}",
                            out_ch * in_ch * kernel,
                            got.len()
                        ),
                    });
                }
                Ok(())
            };
        let transpose_oki_to_oik =
            |flat: &[f32], out_ch: usize, in_ch: usize, kernel: usize| -> Vec<f32> {
                let mut out = vec![0.0f32; flat.len()];
                for o in 0..out_ch {
                    for i in 0..in_ch {
                        for k in 0..kernel {
                            out[o * in_ch * kernel + i * kernel + k] =
                                flat[o * kernel * in_ch + k * in_ch + i];
                        }
                    }
                }
                out
            };
        let stft_raw = file.load_as_f32(&format!("{prefix}.stft_conv.weight"))?;
        conv_shape_check(
            &format!("{prefix}.stft_conv.weight"),
            &stft_raw,
            cutoff * 2,
            1,
            filter_length,
        )?;
        let stft_conv = transpose_oki_to_oik(&stft_raw, cutoff * 2, 1, filter_length);
        let load_conv = |name: &str, out_ch: usize, in_ch: usize| -> Result<(Vec<f32>, Vec<f32>)> {
            let raw = file.load_as_f32(name)?;
            conv_shape_check(name, &raw, out_ch, in_ch, 3)?;
            let w = transpose_oki_to_oik(&raw, out_ch, in_ch, 3);
            let base = name.strip_suffix(".weight").unwrap_or(name);
            let b = file.load_as_f32(&format!("{base}.bias"))?;
            if b.len() != out_ch {
                return Err(SpeechError::Tensor {
                    name: format!("{base}.bias"),
                    why: format!("expected {out_ch} values, got {}", b.len()),
                });
            }
            Ok((w, b))
        };
        let (conv1_w, conv1_b) = load_conv(&format!("{prefix}.conv1.weight"), 128, cutoff)?;
        let (conv2_w, conv2_b) = load_conv(&format!("{prefix}.conv2.weight"), 64, 128)?;
        let (conv3_w, conv3_b) = load_conv(&format!("{prefix}.conv3.weight"), 64, 64)?;
        let (conv4_w, conv4_b) = load_conv(&format!("{prefix}.conv4.weight"), 128, 64)?;
        // The conversion stores MLX nn.LSTM parameters: Wx (input), Wh
        // (hidden), one combined bias, gate order i, f, g, o.
        let lstm_ih = file.load_as_f32(&format!("{prefix}.lstm.Wx"))?;
        if lstm_ih.len() != 4 * lstm_dim * lstm_dim {
            return Err(SpeechError::Tensor {
                name: format!("{prefix}.lstm.Wx"),
                why: format!("expected 4x128x128, got {}", lstm_ih.len()),
            });
        }
        let lstm_hh = file.load_as_f32(&format!("{prefix}.lstm.Wh"))?;
        if lstm_hh.len() != 4 * lstm_dim * lstm_dim {
            return Err(SpeechError::Tensor {
                name: format!("{prefix}.lstm.Wh"),
                why: format!("expected 4x128x128, got {}", lstm_hh.len()),
            });
        }
        let combined_bias = file.load_as_f32(&format!("{prefix}.lstm.bias"))?;
        if combined_bias.len() != 4 * lstm_dim {
            return Err(SpeechError::Tensor {
                name: format!("{prefix}.lstm.bias"),
                why: format!("expected 512, got {}", combined_bias.len()),
            });
        }
        let lstm_ih_bias = combined_bias;
        let lstm_hh_bias = vec![0.0f32; 4 * lstm_dim];
        let final_conv_raw = file.load_as_f32(&format!("{prefix}.final_conv.weight"))?;
        conv_shape_check(
            &format!("{prefix}.final_conv.weight"),
            &final_conv_raw,
            1,
            128,
            1,
        )?;
        let final_conv_w = transpose_oki_to_oik(&final_conv_raw, 1, 128, 1);
        let final_conv_b = file.load_as_f32(&format!("{prefix}.final_conv.bias"))?;
        if final_conv_b.len() != 1 {
            return Err(SpeechError::Tensor {
                name: format!("{prefix}.final_conv.bias"),
                why: "expected one output bias".to_string(),
            });
        }
        Ok(Branch {
            sample_rate,
            filter_length,
            hop_length,
            pad,
            cutoff,
            context_size,
            chunk_size,
            stft_conv,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            conv3_w,
            conv3_b,
            conv4_w,
            conv4_b,
            lstm_ih,
            lstm_hh,
            lstm_ih_bias,
            lstm_hh_bias,
            final_conv_w,
            final_conv_b,
        })
    }

    /// Forward pass for one window of `context + chunk` samples (576 at
    /// 16 kHz). Returns `(probability, new_hidden, new_cell)`.
    fn forward(
        &self,
        window: &[f32],
        hidden: &[f32],
        cell: &[f32],
    ) -> Result<(f32, Vec<f32>, Vec<f32>)> {
        let expect = self.context_size + self.chunk_size;
        if window.len() != expect {
            return Err(SpeechError::Input {
                why: format!(
                    "window must be context {} + chunk {} = {} samples, got {}",
                    self.context_size,
                    self.chunk_size,
                    expect,
                    window.len()
                ),
            });
        }
        // Reflect pad right: the reflected tail walks backwards from the
        // second-to-last sample (reference `_reflect_pad_right`).
        let n = window.len();
        let pad = self.pad;
        if n <= pad + 2 {
            return Err(SpeechError::Input {
                why: format!("reflect padding {pad} needs more than {pad} samples"),
            });
        }
        let mut padded = Vec::with_capacity(n + pad);
        padded.extend_from_slice(window);
        for i in 0..pad {
            padded.push(window[n - 2 - i]);
        }
        // Learned STFT conv: [1, seq] input, kernel = filter_length,
        // stride = hop, no bias.
        let frames = ops::conv1d(
            &padded,
            &self.stft_conv,
            None,
            1,
            self.cutoff * 2,
            self.filter_length,
            self.hop_length,
            0,
            1,
            1,
        );
        let t = frames.len() / (self.cutoff * 2);
        // Magnitude from the real/imag channel split.
        let mut mag = vec![0.0f32; self.cutoff * t];
        for ch in 0..self.cutoff {
            for f in 0..t {
                let re = frames[ch * t + f];
                let im = frames[(self.cutoff + ch) * t + f];
                mag[ch * t + f] = (re * re + im * im).sqrt();
            }
        }
        let mut x = ops::conv1d(
            &mag,
            &self.conv1_w,
            Some(&self.conv1_b),
            self.cutoff,
            128,
            3,
            1,
            1,
            1,
            1,
        );
        for v in x.iter_mut() {
            *v = v.max(0.0);
        }
        let mut x = ops::conv1d(
            &x,
            &self.conv2_w,
            Some(&self.conv2_b),
            128,
            64,
            3,
            2,
            1,
            1,
            1,
        );
        for v in x.iter_mut() {
            *v = v.max(0.0);
        }
        let mut x = ops::conv1d(
            &x,
            &self.conv3_w,
            Some(&self.conv3_b),
            64,
            64,
            3,
            2,
            1,
            1,
            1,
        );
        for v in x.iter_mut() {
            *v = v.max(0.0);
        }
        let mut x = ops::conv1d(
            &x,
            &self.conv4_w,
            Some(&self.conv4_b),
            64,
            128,
            3,
            1,
            1,
            1,
            1,
        );
        for v in x.iter_mut() {
            *v = v.max(0.0);
        }
        let seq = x.len() / 128;
        let mut h = hidden.to_vec();
        let mut c = cell.to_vec();
        let mut last_hidden_out = self.lstm_steps(&x, seq, &mut h, &mut c);
        // relu(hidden_seq) -> final 1x1 conv -> sigmoid -> time mean.
        for v in last_hidden_out.iter_mut() {
            *v = v.max(0.0);
        }
        let mut out = ops::conv1d(
            &last_hidden_out,
            &self.final_conv_w,
            Some(&self.final_conv_b),
            128,
            1,
            1,
            1,
            0,
            1,
            1,
        );
        let mut sum = 0.0f32;
        for v in out.iter_mut() {
            *v = ops::sigmoid(*v);
            sum += *v;
        }
        let prob = sum / out.len() as f32;
        Ok((prob, h, c))
    }
}

impl Branch {
    /// LSTM over the (usually length 1) time axis of `x [seq, 128]`, gate
    /// order i, f, g, o. Advances `h` and `c` in place and returns the
    /// per-step hidden outputs `[seq, 128]`.
    fn lstm_steps(&self, x: &[f32], seq: usize, h: &mut [f32], c: &mut [f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; 128 * seq];
        for s in 0..seq {
            let x_t = &x[s * 128..(s + 1) * 128];
            let gates = lstm_gates(
                x_t,
                h,
                &self.lstm_ih,
                &self.lstm_hh,
                &self.lstm_ih_bias,
                &self.lstm_hh_bias,
            );
            // `gates` already holds every use of the old `h`, so the new
            // hidden values go straight into this step's output row.
            let row = &mut out[s * 128..(s + 1) * 128];
            for d in 0..128 {
                let i = gates[d];
                let f = gates[128 + d];
                let g = gates[256 + d].tanh();
                let o = gates[384 + d];
                c[d] = f * c[d] + i * g;
                row[d] = o * c[d].tanh();
            }
            h.copy_from_slice(row);
        }
        out
    }
}

/// LSTM gate pre-activations, ifgo order, for one time step: gate `i`
/// occupies the first `dim` outputs, then `f`, `g` (linear), `o`.
fn lstm_gates(
    x: &[f32],
    h: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
) -> Vec<f32> {
    let dim = x.len();
    let mut gates = vec![0.0f32; 4 * dim];
    for (g, gate) in gates.iter_mut().enumerate() {
        let w_row = &w_ih[g * dim..(g + 1) * dim];
        let h_row = &w_hh[g * dim..(g + 1) * dim];
        let mut acc = b_ih[g] + b_hh[g];
        for d in 0..dim {
            acc += x[d] * w_row[d] + h[d] * h_row[d];
        }
        *gate = acc;
    }
    for d in 0..dim {
        gates[d] = ops::sigmoid(gates[d]);
        gates[dim + d] = ops::sigmoid(gates[dim + d]);
        gates[3 * dim + d] = ops::sigmoid(gates[3 * dim + d]);
        // g gate stays linear.
    }
    gates
}

/// The loaded Silero VAD model: both sample-rate branches.
pub struct SileroVad {
    vad_16k: Branch,
    vad_8k: Branch,
    pub timing: VadTiming,
}

impl SileroVad {
    /// Loads from a directory holding `model.safetensors` (and an
    /// optional `config.json`; the branch shapes are validated against
    /// the weights themselves).
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        let vad_16k = Branch::load(&file, "vad_16k")?;
        let vad_8k = Branch::load(&file, "vad_8k")?;
        Ok(SileroVad {
            vad_16k,
            vad_8k,
            timing: VadTiming::default(),
        })
    }

    fn branch(&self, sample_rate: u32) -> Result<&Branch> {
        match sample_rate {
            16_000 => Ok(&self.vad_16k),
            8_000 => Ok(&self.vad_8k),
            other => Err(SpeechError::Input {
                why: format!("Silero VAD supports 8000 Hz and 16000 Hz, got {other}"),
            }),
        }
    }

    /// Fresh streaming state for `sample_rate`.
    pub fn initial_state(&self, sample_rate: u32) -> Result<VadStreamState> {
        let branch = self.branch(sample_rate)?;
        Ok(VadStreamState {
            hidden: vec![0.0; 128],
            cell: vec![0.0; 128],
            context: vec![0.0; branch.context_size],
            sample_rate,
        })
    }

    /// Feeds one chunk (`512` samples at 16 kHz, `256` at 8 kHz) and
    /// returns the speech probability for it, advancing the state.
    pub fn feed(&self, chunk: &[f32], state: &mut VadStreamState) -> Result<f32> {
        let branch = self.branch(state.sample_rate)?;
        if state.hidden.len() != 128
            || state.cell.len() != 128
            || state.context.len() != branch.context_size
        {
            return Err(SpeechError::Input {
                why: "VAD state must have 128 hidden/cell values and the branch context length"
                    .to_string(),
            });
        }
        if chunk
            .iter()
            .chain(&state.context)
            .chain(&state.hidden)
            .chain(&state.cell)
            .any(|v| !v.is_finite())
        {
            return Err(SpeechError::Input {
                why: "VAD samples and state must be finite".to_string(),
            });
        }
        if chunk.len() != branch.chunk_size {
            return Err(SpeechError::Input {
                why: format!(
                    "expected {} samples at {} Hz, got {}",
                    branch.chunk_size,
                    state.sample_rate,
                    chunk.len()
                ),
            });
        }
        let mut window = Vec::with_capacity(branch.context_size + chunk.len());
        window.extend_from_slice(&state.context);
        window.extend_from_slice(chunk);
        let (prob, h, c) = branch.forward(&window, &state.hidden, &state.cell)?;
        state.hidden = h;
        state.cell = c;
        state
            .context
            .copy_from_slice(&chunk[chunk.len() - branch.context_size..]);
        Ok(prob)
    }

    /// Whole-file inference: per-chunk probabilities, transcribed from
    /// `_predict_proba_array` (zero-pad to a chunk multiple, lead with
    /// one context of zeros, step chunk_size).
    pub fn predict_proba(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let branch = self.branch(sample_rate)?;
        if samples.iter().any(|v| !v.is_finite()) {
            return Err(SpeechError::Input {
                why: "VAD samples must be finite".to_string(),
            });
        }
        let mut probs = Vec::new();
        let mut state = self.initial_state(sample_rate)?;
        if samples.is_empty() {
            return Ok(probs);
        }
        let pad = (branch.chunk_size - samples.len() % branch.chunk_size) % branch.chunk_size;
        let mut audio = samples.to_vec();
        audio.extend(std::iter::repeat_n(0.0, pad));
        let mut padded = vec![0.0f32; branch.context_size];
        padded.extend_from_slice(&audio);
        let mut pos = branch.context_size;
        while pos < padded.len() {
            let window = &padded[pos - branch.context_size..pos + branch.chunk_size];
            let (prob, h, c) = branch.forward(window, &state.hidden, &state.cell)?;
            state.hidden = h;
            state.cell = c;
            probs.push(prob);
            pos += branch.chunk_size;
        }
        Ok(probs)
    }

    /// Speech timestamps from probabilities: the reference hysteresis
    /// (`threshold` to trigger, `threshold - 0.15` to release, minimum
    /// durations, padding, merge of overlapping segments). Returns
    /// `(start_sample, end_sample)` pairs.
    pub fn timestamps(
        &self,
        probs: &[f32],
        audio_len: usize,
        sample_rate: u32,
    ) -> Vec<(usize, usize)> {
        let t = &self.timing;
        let chunk_size = if sample_rate == 16_000 { 512 } else { 256 };
        let min_speech_samples = sample_rate as f32 * t.min_speech_duration_ms as f32 / 1000.0;
        let min_silence_samples = sample_rate as f32 * t.min_silence_duration_ms as f32 / 1000.0;
        let speech_pad_samples = (sample_rate as f32 * t.speech_pad_ms as f32 / 1000.0) as usize;
        let neg_threshold = (t.threshold - 0.15).max(0.01);

        let mut speeches: Vec<(usize, usize)> = Vec::new();
        let mut triggered = false;
        let mut current_start = 0usize;
        let mut temp_end = 0usize;
        for (idx, &prob) in probs.iter().enumerate() {
            let chunk_start = idx * chunk_size;
            if prob >= t.threshold && !triggered {
                triggered = true;
                current_start = chunk_start;
                temp_end = 0;
                continue;
            }
            if triggered && prob >= t.threshold {
                temp_end = 0;
                continue;
            }
            if triggered && prob < neg_threshold {
                if temp_end == 0 {
                    temp_end = chunk_start;
                }
                if chunk_start - temp_end >= min_silence_samples as usize {
                    if temp_end - current_start >= min_speech_samples as usize {
                        speeches.push((current_start, temp_end));
                    }
                    triggered = false;
                    temp_end = 0;
                }
            }
        }
        if triggered {
            let end = audio_len.min(probs.len() * chunk_size);
            if end >= current_start && end - current_start >= min_speech_samples as usize {
                speeches.push((current_start, end));
            }
        }
        // Pad and merge overlapping segments.
        let mut padded_segments: Vec<(usize, usize)> = Vec::new();
        for (start, end) in speeches {
            let start = start.saturating_sub(speech_pad_samples);
            let end = (end + speech_pad_samples).min(audio_len);
            if let Some(last) = padded_segments.last_mut() {
                if start <= last.1 {
                    last.1 = last.1.max(end);
                    continue;
                }
            }
            padded_segments.push((start, end));
        }
        padded_segments
    }

    /// Convenience: probabilities then timestamps in one call.
    pub fn detect(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<(usize, usize)>> {
        let probs = self.predict_proba(samples, sample_rate)?;
        Ok(self.timestamps(&probs, samples.len(), sample_rate))
    }
}

#[cfg(test)]
impl Branch {
    fn config_16k_defaults_for_test() -> Branch {
        Self::empty_for_test(Self::config_16k())
    }

    fn config_8k_defaults_for_test() -> Branch {
        Self::empty_for_test(Self::config_8k())
    }

    fn empty_for_test(
        (sample_rate, filter_length, hop_length, pad, cutoff, context_size, chunk_size): (
            u32,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
        ),
    ) -> Branch {
        Branch {
            sample_rate,
            filter_length,
            hop_length,
            pad,
            cutoff,
            context_size,
            chunk_size,
            stft_conv: Vec::new(),
            conv1_w: Vec::new(),
            conv1_b: Vec::new(),
            conv2_w: Vec::new(),
            conv2_b: Vec::new(),
            conv3_w: Vec::new(),
            conv3_b: Vec::new(),
            conv4_w: Vec::new(),
            conv4_b: Vec::new(),
            lstm_ih: Vec::new(),
            lstm_hh: Vec::new(),
            lstm_ih_bias: Vec::new(),
            lstm_hh_bias: Vec::new(),
            final_conv_w: Vec::new(),
            final_conv_b: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feed_refuses_malformed_public_state_before_compute() {
        let model = SileroVad {
            vad_16k: Branch::config_16k_defaults_for_test(),
            vad_8k: Branch::config_8k_defaults_for_test(),
            timing: VadTiming::default(),
        };
        for field in 0..3 {
            let mut state = model.initial_state(16000).unwrap();
            match field {
                0 => {
                    state.hidden.pop();
                }
                1 => {
                    state.cell.pop();
                }
                _ => {
                    state.context.pop();
                }
            }
            assert!(model.feed(&[0.0; 512], &mut state).is_err());
        }
        let mut state = model.initial_state(16000).unwrap();
        state.hidden[0] = f32::NAN;
        assert!(model.feed(&[0.0; 512], &mut state).is_err());
        assert!(model.predict_proba(&[f32::NAN], 16000).is_err());
    }

    #[test]
    fn reflect_pad_starts_at_second_to_last() {
        // The branch forward validates and pads internally; here we pin
        // the convention via a synthetic check on the pad math.
        let window: Vec<f32> = (0..576).map(|i| i as f32).collect();
        let n = window.len();
        let pad = 64;
        let reflected_last = window[n - 2]; // first reflected value
        let mut padded = window.clone();
        for i in 0..pad {
            padded.push(window[n - 2 - i]);
        }
        assert_eq!(padded.len(), n + pad);
        assert_eq!(padded[n], reflected_last);
        assert_eq!(padded[n + pad - 1], window[n - 2 - (pad - 1)]);
    }

    #[test]
    fn lstm_gates_ifgo_order_and_sigmoid() {
        // Identity weights, zero biases: gates are sigmoid(x + h) for i/f/o
        // and linear (x + h) for g.
        let dim = 4;
        let x = vec![0.1f32; dim];
        let h = vec![0.2f32; dim];
        let mut w_ih = vec![0.0f32; 4 * dim * dim];
        let mut w_hh = vec![0.0f32; 4 * dim * dim];
        for g in 0..4 {
            for d in 0..dim {
                w_ih[g * dim * dim + d * dim + d] = 1.0;
                w_hh[g * dim * dim + d * dim + d] = 1.0;
            }
        }
        let b = vec![0.0f32; 4 * dim];
        let gates = lstm_gates(&x, &h, &w_ih, &w_hh, &b, &b);
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        for d in 0..dim {
            assert!((gates[d] - sig(0.3)).abs() < 1e-6, "i gate");
            assert!((gates[dim + d] - sig(0.3)).abs() < 1e-6, "f gate");
            assert!((gates[2 * dim + d] - 0.3).abs() < 1e-6, "g gate linear");
            assert!((gates[3 * dim + d] - sig(0.3)).abs() < 1e-6, "o gate");
        }
    }

    #[test]
    fn timestamps_hysteresis_pads_and_merges() {
        let model = SileroVad {
            vad_16k: Branch::config_16k_defaults_for_test(),
            vad_8k: Branch::config_8k_defaults_for_test(),
            timing: VadTiming {
                threshold: 0.5,
                min_speech_duration_ms: 250,
                min_silence_duration_ms: 100,
                speech_pad_ms: 30,
            },
        };
        // 16 kHz: chunk 512 (32 ms). Speech chunks 0..31, silence chunks
        // 31..38, speech chunks 38..69. The 192 ms silence exceeds the
        // 100 ms minimum, so two segments survive; 30 ms padding keeps a
        // gap between them.
        let mut probs = vec![0.9f32; 31];
        probs.extend(vec![0.2f32; 7]);
        probs.extend(vec![0.9f32; 31]);
        let audio_len = 69 * 512;
        let ts = model.timestamps(&probs, audio_len, 16_000);
        assert_eq!(
            ts,
            vec![(0, 16352), (18976, 35328)],
            "padded segments: {ts:?}"
        );
    }

    /// Deterministic xorshift stream with ~1 in 6 exact zeros; avoids a
    /// dev-dependency.
    struct Rng(u64);

    impl Rng {
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    let r = self.0;
                    if r % 6 == 0 {
                        0.0
                    } else {
                        (((r >> 8) % 20001) as f32 / 10000.0 - 1.0) * scale
                    }
                })
                .collect()
        }
    }

    fn random_branch(rng: &mut Rng) -> Branch {
        let mut b = Branch::config_16k_defaults_for_test();
        b.stft_conv = rng.vec(b.cutoff * 2 * b.filter_length, 0.1);
        b.conv1_w = rng.vec(128 * b.cutoff * 3, 0.05);
        b.conv1_b = rng.vec(128, 0.1);
        b.conv2_w = rng.vec(64 * 128 * 3, 0.05);
        b.conv2_b = rng.vec(64, 0.1);
        b.conv3_w = rng.vec(64 * 64 * 3, 0.05);
        b.conv3_b = rng.vec(64, 0.1);
        b.conv4_w = rng.vec(128 * 64 * 3, 0.05);
        b.conv4_b = rng.vec(128, 0.1);
        b.lstm_ih = rng.vec(4 * 128 * 128, 0.1);
        b.lstm_hh = rng.vec(4 * 128 * 128, 0.1);
        b.lstm_ih_bias = rng.vec(4 * 128, 0.1);
        b.lstm_hh_bias = rng.vec(4 * 128, 0.1);
        b.final_conv_w = rng.vec(128, 0.1);
        b.final_conv_b = rng.vec(1, 0.1);
        b
    }

    fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}: element {i}: got {g} want {w}"
            );
        }
    }

    /// The original LSTM step loop (fresh `new_h` per step, `h` rebuilt
    /// by clone), kept verbatim as the parity reference.
    fn lstm_steps_reference(
        b: &Branch,
        x: &[f32],
        seq: usize,
        hidden: &[f32],
        cell: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut h = hidden.to_vec();
        let mut c = cell.to_vec();
        let mut last_hidden_out = vec![0.0f32; 128 * seq];
        for s in 0..seq {
            let x_t = &x[s * 128..(s + 1) * 128];
            let gates = lstm_gates(
                x_t,
                &h,
                &b.lstm_ih,
                &b.lstm_hh,
                &b.lstm_ih_bias,
                &b.lstm_hh_bias,
            );
            let mut new_h = vec![0.0f32; 128];
            for d in 0..128 {
                let i = gates[d];
                let f = gates[128 + d];
                let g = gates[256 + d].tanh();
                let o = gates[384 + d];
                c[d] = f * c[d] + i * g;
                new_h[d] = o * c[d].tanh();
            }
            h = new_h.clone();
            last_hidden_out[s * 128..(s + 1) * 128].copy_from_slice(&new_h);
        }
        (last_hidden_out, h, c)
    }

    #[test]
    fn lstm_steps_matches_original_over_many_steps_bitwise() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let b = random_branch(&mut rng);
        for &seq in &[1usize, 2, 7] {
            let x = rng.vec(128 * seq, 1.0);
            let hidden = rng.vec(128, 0.5);
            let cell = rng.vec(128, 0.5);
            let (want_out, want_h, want_c) = lstm_steps_reference(&b, &x, seq, &hidden, &cell);
            let (mut h, mut c) = (hidden.clone(), cell.clone());
            let got_out = b.lstm_steps(&x, seq, &mut h, &mut c);
            assert_bits("out", &got_out, &want_out);
            assert_bits("h", &h, &want_h);
            assert_bits("c", &c, &want_c);
        }
    }

    /// Streams several chunks through `feed` and through the original
    /// bookkeeping (new context/hidden/cell vectors each step) and
    /// compares every probability and the carried state bit for bit.
    #[test]
    fn feed_streaming_state_matches_original_bookkeeping_bitwise() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let model = SileroVad {
            vad_16k: random_branch(&mut rng),
            vad_8k: Branch::config_8k_defaults_for_test(),
            timing: VadTiming::default(),
        };
        let branch = &model.vad_16k;
        let mut state = model.initial_state(16_000).unwrap();
        let (mut ref_h, mut ref_c, mut ref_ctx) = (
            state.hidden.clone(),
            state.cell.clone(),
            state.context.clone(),
        );
        for step in 0..5 {
            let chunk = rng.vec(branch.chunk_size, 0.5);
            let prob = model.feed(&chunk, &mut state).unwrap();
            let mut window = Vec::with_capacity(branch.context_size + chunk.len());
            window.extend_from_slice(&ref_ctx);
            window.extend_from_slice(&chunk);
            let (ref_prob, h, c) = branch.forward(&window, &ref_h, &ref_c).unwrap();
            ref_h = h;
            ref_c = c;
            ref_ctx = chunk[chunk.len() - branch.context_size..].to_vec();
            assert_eq!(prob.to_bits(), ref_prob.to_bits(), "step {step} prob");
            assert_bits("hidden", &state.hidden, &ref_h);
            assert_bits("cell", &state.cell, &ref_c);
            assert_bits("context", &state.context, &ref_ctx);
        }
    }
}
