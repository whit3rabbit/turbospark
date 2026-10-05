use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::{Result, SpeechError};

use super::config::FireRedAsr2Config;

fn tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
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
    file.load_as_f32(name).map_err(Into::into)
}

struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    input: usize,
    output: usize,
}

impl Linear {
    fn load(file: &SafetensorsFile, name: &str, input: usize, output: usize) -> Result<Self> {
        let weight_name = format!("{name}.weight");
        let weight = if file.contains_tensor(&weight_name) {
            tensor(file, &weight_name, &[output, input])?
        } else if name == "decoder.tgt_word_prj" {
            tensor(file, "decoder.tgt_word_emb.weight", &[output, input])?
        } else {
            return Err(SpeechError::Tensor {
                name: weight_name,
                why: "tensor is missing".into(),
            });
        };
        let bias_name = format!("{name}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(tensor(file, &bias_name, &[output])?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            input,
            output,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.weight,
            self.bias.as_deref(),
            rows,
            self.input,
            self.output,
        )
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    width: usize,
}

impl LayerNorm {
    fn load(file: &SafetensorsFile, name: &str, width: usize) -> Result<Self> {
        Ok(Self {
            weight: tensor(file, &format!("{name}.weight"), &[width])?,
            bias: tensor(file, &format!("{name}.bias"), &[width])?,
            width,
        })
    }

    fn apply(&self, x: &mut [f32], rows: usize) {
        ops::layernorm(x, rows, self.width, &self.weight, Some(&self.bias), 1e-5);
    }
}

struct ProjectedKeys {
    keys: Vec<f32>,
    values: Vec<f32>,
    rows: usize,
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
    width: usize,
}

impl Attention {
    fn load(file: &SafetensorsFile, prefix: &str, width: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            query: Linear::load(file, &format!("{prefix}.w_qs"), width, width)?,
            key: Linear::load(file, &format!("{prefix}.w_ks"), width, width)?,
            value: Linear::load(file, &format!("{prefix}.w_vs"), width, width)?,
            output: Linear::load(file, &format!("{prefix}.fc"), width, width)?,
            heads,
            head_dim: width / heads,
            width,
        })
    }

    fn project_keys(&self, input: &[f32], rows: usize) -> ProjectedKeys {
        ProjectedKeys {
            keys: to_head_major(
                &self.key.forward(input, rows),
                rows,
                self.heads,
                self.head_dim,
            ),
            values: to_head_major(
                &self.value.forward(input, rows),
                rows,
                self.heads,
                self.head_dim,
            ),
            rows,
        }
    }

    fn self_last(&self, normalized: &[f32], rows: usize) -> Vec<f32> {
        let all_keys = self.project_keys(normalized, rows);
        self.attend_last(normalized, rows, &all_keys)
    }

    fn cross_last(&self, normalized_query: &[f32], projected: &ProjectedKeys) -> Vec<f32> {
        self.attend_last(normalized_query, 1, projected)
    }

    fn attend_last(
        &self,
        query_input: &[f32],
        query_rows: usize,
        projected: &ProjectedKeys,
    ) -> Vec<f32> {
        let query = self
            .query
            .forward(&query_input[(query_rows - 1) * self.width..], 1);
        let mut context = vec![0.0; self.width];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut scores = vec![0.0; projected.rows];
        for head in 0..self.heads {
            let query_head = &query[head * self.head_dim..(head + 1) * self.head_dim];
            let base = head * projected.rows * self.head_dim;
            for row in 0..projected.rows {
                let key =
                    &projected.keys[base + row * self.head_dim..base + (row + 1) * self.head_dim];
                scores[row] = dot(query_head, key) * scale;
            }
            ops::softmax_row(&mut scores);
            for feature in 0..self.head_dim {
                let mut sum = 0.0;
                for row in 0..projected.rows {
                    sum += scores[row] * projected.values[base + row * self.head_dim + feature];
                }
                context[head * self.head_dim + feature] = sum;
            }
        }
        self.output.forward(&context, 1)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&a, &b)| a * b).sum()
}

fn to_head_major(input: &[f32], rows: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for head in 0..heads {
            let source = (row * heads + head) * head_dim;
            let target = (head * rows + row) * head_dim;
            output[target..target + head_dim].copy_from_slice(&input[source..source + head_dim]);
        }
    }
    output
}

struct DecoderLayer {
    self_norm: LayerNorm,
    self_attention: Attention,
    cross_norm: LayerNorm,
    cross_attention: Attention,
    mlp_norm: LayerNorm,
    mlp_in: Linear,
    mlp_out: Linear,
    width: usize,
}

impl DecoderLayer {
    fn load(file: &SafetensorsFile, index: usize, config: &FireRedAsr2Config) -> Result<Self> {
        let prefix = format!("decoder.layer_stack.{index}");
        let width = config.model_dim;
        Ok(Self {
            self_norm: LayerNorm::load(file, &format!("{prefix}.self_attn_norm"), width)?,
            self_attention: Attention::load(
                file,
                &format!("{prefix}.self_attn"),
                width,
                config.decoder_heads,
            )?,
            cross_norm: LayerNorm::load(file, &format!("{prefix}.cross_attn_norm"), width)?,
            cross_attention: Attention::load(
                file,
                &format!("{prefix}.cross_attn"),
                width,
                config.decoder_heads,
            )?,
            mlp_norm: LayerNorm::load(file, &format!("{prefix}.mlp_norm"), width)?,
            mlp_in: Linear::load(file, &format!("{prefix}.mlp.w_1"), width, width * 4)?,
            mlp_out: Linear::load(file, &format!("{prefix}.mlp.w_2"), width * 4, width)?,
            width,
        })
    }

    fn cross_keys(&self, encoder: &[f32], rows: usize) -> ProjectedKeys {
        self.cross_attention.project_keys(encoder, rows)
    }

    fn step(&self, input: &[f32], rows: usize, cross: &ProjectedKeys) -> Vec<f32> {
        let offset = (rows - 1) * self.width;
        let mut normalized = input.to_vec();
        self.self_norm.apply(&mut normalized, rows);
        let self_update = self.self_attention.self_last(&normalized, rows);
        let mut hidden: Vec<f32> = input[offset..offset + self.width]
            .iter()
            .zip(self_update)
            .map(|(&x, update)| x + update)
            .collect();

        let mut normalized = hidden.clone();
        self.cross_norm.apply(&mut normalized, 1);
        let cross_update = self.cross_attention.cross_last(&normalized, cross);
        for (x, update) in hidden.iter_mut().zip(cross_update) {
            *x += update;
        }

        let residual = hidden.clone();
        let mut normalized = hidden.clone();
        self.mlp_norm.apply(&mut normalized, 1);
        let mut feed_forward = self.mlp_in.forward(&normalized, 1);
        ops::gelu_erf(&mut feed_forward);
        let feed_forward = self.mlp_out.forward(&feed_forward, 1);
        for ((x, residual), update) in hidden.iter_mut().zip(residual).zip(feed_forward) {
            *x = residual + update;
        }
        hidden
    }
}

#[derive(Clone)]
struct Beam {
    tokens: Vec<usize>,
    score: f32,
    finished: bool,
    layer_outputs: Vec<Vec<f32>>,
}

struct Candidate {
    score: f32,
    parent: usize,
    token: usize,
    finished: bool,
}

pub(super) struct Decoder {
    config: FireRedAsr2Config,
    embeddings: Vec<f32>,
    layers: Vec<DecoderLayer>,
    output_norm: LayerNorm,
    output_projection: Linear,
}

impl Decoder {
    pub(super) fn load(file: &SafetensorsFile, config: FireRedAsr2Config) -> Result<Self> {
        let width = config.model_dim;
        let embeddings = tensor(
            file,
            "decoder.tgt_word_emb.weight",
            &[config.vocab_size, width],
        )?;
        let layers = (0..config.decoder_layers)
            .map(|index| DecoderLayer::load(file, index, &config))
            .collect::<Result<Vec<_>>>()?;
        let output_norm = LayerNorm::load(file, "decoder.layer_norm_out", width)?;
        let output_projection =
            Linear::load(file, "decoder.tgt_word_prj", width, config.vocab_size)?;
        Ok(Self {
            config,
            embeddings,
            layers,
            output_norm,
            output_projection,
        })
    }

    pub(super) fn decode(
        &self,
        encoder: &[f32],
        encoder_rows: usize,
        dictionary: &[String],
        beam_size: usize,
        max_len: usize,
        softmax_smoothing: f32,
        length_penalty: f32,
        eos_penalty: f32,
    ) -> Result<String> {
        if beam_size == 0 || beam_size > self.config.vocab_size {
            return Err(SpeechError::Input {
                why: "FireRedASR2 beam_size must be within the vocabulary size".into(),
            });
        }
        if !softmax_smoothing.is_finite() || softmax_smoothing <= 0.0 {
            return Err(SpeechError::Input {
                why: "FireRedASR2 softmax_smoothing must be finite and positive".into(),
            });
        }
        if !length_penalty.is_finite() || length_penalty < 0.0 || !eos_penalty.is_finite() {
            return Err(SpeechError::Input {
                why: "FireRedASR2 decoding penalties must be finite and valid".into(),
            });
        }
        if max_len == 0 || max_len > self.config.decoder_position_limit {
            return Err(SpeechError::Input {
                why: "FireRedASR2 decode length is outside the configured position limit".into(),
            });
        }

        let cross_keys: Vec<_> = self
            .layers
            .iter()
            .map(|layer| layer.cross_keys(encoder, encoder_rows))
            .collect();
        let mut beams: Vec<Beam> = (0..beam_size)
            .map(|index| Beam {
                tokens: vec![self.config.start_token],
                score: if index == 0 { 0.0 } else { -1e10 },
                finished: false,
                layer_outputs: vec![Vec::new(); self.layers.len()],
            })
            .collect();
        let mut positions = Vec::with_capacity(max_len * self.config.model_dim);
        positional_encoding(&mut positions, max_len, self.config.model_dim);

        for _ in 0..max_len {
            let mut candidates = Vec::with_capacity(beam_size * beam_size);
            for (parent, beam) in beams.iter_mut().enumerate() {
                if beam.finished {
                    candidates.push(Candidate {
                        score: beam.score,
                        parent,
                        token: self.config.end_token,
                        finished: true,
                    });
                    for _ in 1..beam_size {
                        candidates.push(Candidate {
                            score: beam.score - 1e10,
                            parent,
                            token: self.config.end_token,
                            finished: true,
                        });
                    }
                    continue;
                }

                let rows = beam.tokens.len();
                let embeddings = embed_positions(
                    &self.embeddings,
                    &beam.tokens,
                    &positions,
                    self.config.model_dim,
                );
                for index in 0..self.layers.len() {
                    let layer_input = if index == 0 {
                        embeddings.as_slice()
                    } else {
                        beam.layer_outputs[index - 1].as_slice()
                    };
                    let hidden = self.layers[index].step(layer_input, rows, &cross_keys[index]);
                    beam.layer_outputs[index].extend_from_slice(&hidden);
                }
                let mut hidden = beam.layer_outputs.last().expect("decoder has layers")
                    [(rows - 1) * self.config.model_dim..rows * self.config.model_dim]
                    .to_vec();
                self.output_norm.apply(&mut hidden, 1);
                let logits = self.output_projection.forward(&hidden, 1);
                let mut probabilities = logits
                    .iter()
                    .map(|logit| logit / softmax_smoothing)
                    .collect::<Vec<_>>();
                ops::softmax_row(&mut probabilities);
                let mut log_probs = probabilities
                    .iter()
                    .map(|probability| (probability + 1e-10).ln())
                    .collect::<Vec<_>>();
                log_probs[self.config.end_token] *= eos_penalty;
                for (log_probability, token) in top_k(&log_probs, beam_size) {
                    candidates.push(Candidate {
                        score: beam.score + log_probability,
                        parent,
                        token,
                        finished: token == self.config.end_token,
                    });
                }
            }

            candidates.sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| a.parent.cmp(&b.parent))
                    .then_with(|| a.token.cmp(&b.token))
            });
            candidates.truncate(beam_size);
            let mut next = Vec::with_capacity(beam_size);
            for candidate in candidates {
                let mut beam = beams[candidate.parent].clone();
                beam.tokens.push(candidate.token);
                beam.score = candidate.score;
                beam.finished = candidate.finished;
                next.push(beam);
            }
            beams = next;
            if beams.iter().all(|beam| beam.finished) {
                break;
            }
        }

        let best = beams
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                final_score(a, length_penalty, self.config.end_token).total_cmp(&final_score(
                    b,
                    length_penalty,
                    self.config.end_token,
                ))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        let mut ids = beams[best]
            .tokens
            .iter()
            .skip(1)
            .copied()
            .collect::<Vec<_>>();
        if let Some(end) = ids.iter().position(|&token| token == self.config.end_token) {
            ids.truncate(end);
        }
        let mut text = String::new();
        for id in ids {
            if let Some(token) = dictionary.get(id) {
                if token == "<space>" {
                    text.push(' ');
                } else {
                    text.push_str(token);
                }
            }
        }
        text = text.replace('\u{2581}', " ");
        text = text.replace("<blank>", "").replace("<sil>", "");
        Ok(text.trim().to_lowercase())
    }
}

fn top_k(values: &[f32], count: usize) -> Vec<(f32, usize)> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]).then_with(|| a.cmp(&b)));
    indices
        .into_iter()
        .take(count)
        .map(|index| (values[index], index))
        .collect()
}

fn final_score(beam: &Beam, penalty: f32, end_token: usize) -> f32 {
    let length = beam
        .tokens
        .iter()
        .filter(|&&token| token != end_token)
        .count() as f32;
    if penalty > 0.0 {
        beam.score / ((5.0 + length) / 6.0).powf(penalty)
    } else {
        beam.score
    }
}

fn positional_encoding(output: &mut Vec<f32>, rows: usize, width: usize) {
    for position in 0..rows {
        for pair in 0..width / 2 {
            let angle = position as f32 * 10_000.0f32.powf(-((2 * pair) as f32) / width as f32);
            output.push(angle.sin());
            output.push(angle.cos());
        }
    }
}

fn embed_positions(
    embeddings: &[f32],
    tokens: &[usize],
    positions: &[f32],
    width: usize,
) -> Vec<f32> {
    let scale = (width as f32).sqrt();
    let mut output = vec![0.0; tokens.len() * width];
    for (row, &token) in tokens.iter().enumerate() {
        for col in 0..width {
            output[row * width + col] =
                embeddings[token * width + col] * scale + positions[row * width + col];
        }
    }
    output
}
