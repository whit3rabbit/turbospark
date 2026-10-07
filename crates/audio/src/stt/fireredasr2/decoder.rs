use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::{Result, SpeechError};

use super::config::FireRedAsr2Config;

const LAYER_NORM_EPS: f32 = 1e-5;

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

// The tied-embedding fallback for the output projection stays local to this
// family, so the shared layers are built here rather than with `nn`'s loader.
fn load_linear(file: &SafetensorsFile, name: &str, input: usize, output: usize) -> Result<Linear> {
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
    Ok(Linear::new(weight, bias, input, output))
}

fn load_layer_norm(file: &SafetensorsFile, name: &str, width: usize) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        tensor(file, &format!("{name}.weight"), &[width])?,
        Some(tensor(file, &format!("{name}.bias"), &[width])?),
        LAYER_NORM_EPS,
    ))
}

/// Head-major keys and values `[heads][stride][head_dim]` of which the first
/// `rows` rows per head are filled. Cross-attention keys are built once with
/// `stride == rows`; the self-attention cache is preallocated to the decode
/// length (`stride > rows`) and grows one row per step.
struct ProjectedKeys {
    keys: Vec<f32>,
    values: Vec<f32>,
    rows: usize,
    stride: usize,
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
            query: load_linear(file, &format!("{prefix}.w_qs"), width, width)?,
            key: load_linear(file, &format!("{prefix}.w_ks"), width, width)?,
            value: load_linear(file, &format!("{prefix}.w_vs"), width, width)?,
            output: load_linear(file, &format!("{prefix}.fc"), width, width)?,
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
            stride: rows,
        }
    }

    fn empty_cache(&self, capacity: usize) -> ProjectedKeys {
        ProjectedKeys {
            keys: vec![0.0; self.heads * capacity * self.head_dim],
            values: vec![0.0; self.heads * capacity * self.head_dim],
            rows: 0,
            stride: capacity,
        }
    }

    /// Projects one normalized row and appends it to `cache`. K and V rows
    /// are independent per input row, so a cached row equals the row a full
    /// prefix reprojection would produce.
    fn append_row(&self, normalized: &[f32], cache: &mut ProjectedKeys) {
        debug_assert!(cache.rows < cache.stride);
        let key = self.key.forward(normalized, 1);
        let value = self.value.forward(normalized, 1);
        for head in 0..self.heads {
            let source = head * self.head_dim..(head + 1) * self.head_dim;
            let target = (head * cache.stride + cache.rows) * self.head_dim;
            cache.keys[target..target + self.head_dim].copy_from_slice(&key[source.clone()]);
            cache.values[target..target + self.head_dim].copy_from_slice(&value[source]);
        }
        cache.rows += 1;
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
            let base = head * projected.stride * self.head_dim;
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
}

impl DecoderLayer {
    fn load(file: &SafetensorsFile, index: usize, config: &FireRedAsr2Config) -> Result<Self> {
        let prefix = format!("decoder.layer_stack.{index}");
        let width = config.model_dim;
        Ok(Self {
            self_norm: load_layer_norm(file, &format!("{prefix}.self_attn_norm"), width)?,
            self_attention: Attention::load(
                file,
                &format!("{prefix}.self_attn"),
                width,
                config.decoder_heads,
            )?,
            cross_norm: load_layer_norm(file, &format!("{prefix}.cross_attn_norm"), width)?,
            cross_attention: Attention::load(
                file,
                &format!("{prefix}.cross_attn"),
                width,
                config.decoder_heads,
            )?,
            mlp_norm: load_layer_norm(file, &format!("{prefix}.mlp_norm"), width)?,
            mlp_in: load_linear(file, &format!("{prefix}.mlp.w_1"), width, width * 4)?,
            mlp_out: load_linear(file, &format!("{prefix}.mlp.w_2"), width * 4, width)?,
        })
    }

    fn cross_keys(&self, encoder: &[f32], rows: usize) -> ProjectedKeys {
        self.cross_attention.project_keys(encoder, rows)
    }

    /// One decode row through the layer. `input` is the current row only;
    /// earlier rows live in `cache` as projected keys and values.
    fn step(&self, input: &[f32], cache: &mut ProjectedKeys, cross: &ProjectedKeys) -> Vec<f32> {
        let mut normalized = input.to_vec();
        self.self_norm.apply(&mut normalized, 1);
        self.self_attention.append_row(&normalized, cache);
        let self_update = self.self_attention.attend_last(&normalized, 1, cache);
        let mut hidden: Vec<f32> = input
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

struct Beam {
    tokens: Vec<usize>,
    score: f32,
    finished: bool,
    /// Per-layer self-attention K/V of every row decoded so far. Empty once
    /// the beam finishes, since a finished beam never runs another step.
    cache: Vec<ProjectedKeys>,
}

impl ProjectedKeys {
    /// Copy of the filled rows only; the rest of the preallocated
    /// capacity stays zero and is never read.
    fn duplicate(&self, heads: usize, head_dim: usize) -> Self {
        let mut copy = Self {
            keys: vec![0.0; self.keys.len()],
            values: vec![0.0; self.values.len()],
            rows: self.rows,
            stride: self.stride,
        };
        let used = self.rows * head_dim;
        for head in 0..heads {
            let base = head * self.stride * head_dim;
            copy.keys[base..base + used].copy_from_slice(&self.keys[base..base + used]);
            copy.values[base..base + used].copy_from_slice(&self.values[base..base + used]);
        }
        copy
    }
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
        let output_norm = load_layer_norm(file, "decoder.layer_norm_out", width)?;
        let output_projection =
            load_linear(file, "decoder.tgt_word_prj", width, config.vocab_size)?;
        Ok(Self {
            config,
            embeddings,
            layers,
            output_norm,
            output_projection,
        })
    }

    /// Logits for the next token of an unfinished beam, appending its newest
    /// row to the beam's per-layer caches.
    fn beam_logits(
        &self,
        beam: &mut Beam,
        cross_keys: &[ProjectedKeys],
        positions: &[f32],
    ) -> Vec<f32> {
        let row = beam.tokens.len() - 1;
        let width = self.config.model_dim;
        let mut hidden = embed_positions(
            &self.embeddings,
            &beam.tokens[row..],
            &positions[row * width..],
            width,
        );
        for index in 0..self.layers.len() {
            hidden = self.layers[index].step(&hidden, &mut beam.cache[index], &cross_keys[index]);
        }
        self.output_norm.apply(&mut hidden, 1);
        self.output_projection.forward(&hidden, 1)
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
                cache: self
                    .layers
                    .iter()
                    .map(|layer| layer.self_attention.empty_cache(max_len))
                    .collect(),
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

                let logits = self.beam_logits(beam, &cross_keys, &positions);
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
            // Children of one parent need their own cache, but the last child
            // takes the parent's, so only extra children copy.
            let mut uses = vec![0usize; beam_size];
            for candidate in &candidates {
                uses[candidate.parent] += 1;
            }
            let mut parents: Vec<Option<Beam>> = beams.into_iter().map(Some).collect();
            let mut next = Vec::with_capacity(beam_size);
            for candidate in candidates {
                uses[candidate.parent] -= 1;
                let parent = parents[candidate.parent]
                    .as_ref()
                    .expect("parent beam present");
                let mut tokens = parent.tokens.clone();
                tokens.push(candidate.token);
                let cache = if candidate.finished {
                    Vec::new()
                } else if uses[candidate.parent] == 0 {
                    std::mem::take(&mut parents[candidate.parent].as_mut().unwrap().cache)
                } else {
                    parent
                        .cache
                        .iter()
                        .zip(&self.layers)
                        .map(|(cache, layer)| {
                            cache.duplicate(
                                layer.self_attention.heads,
                                layer.self_attention.head_dim,
                            )
                        })
                        .collect()
                };
                next.push(Beam {
                    tokens,
                    score: candidate.score,
                    finished: candidate.finished,
                    cache,
                });
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

#[cfg(test)]
mod tests {
    //! Retained-reference parity: `reference_*` below is the original
    //! full-prefix decoder (re-norm and re-project every row each step, deep
    //! clone of per-beam layer outputs on reorder), kept verbatim. The cached
    //! decoder must reproduce its logits bitwise and its tokens exactly.
    #![allow(clippy::needless_range_loop)]
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn value(&mut self, scale: f32) -> f32 {
            ((self.next() >> 8) % 20001) as f32 / 10000.0 * scale - scale
        }

        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.value(scale)).collect()
        }

        fn linear(&mut self, input: usize, output: usize, bias: bool) -> Linear {
            let weight = self.vec(input * output, 0.9 / (input as f32).sqrt());
            let bias = bias.then(|| self.vec(output, 0.2));
            Linear::new(weight, bias, input, output)
        }

        fn norm(&mut self, width: usize) -> LayerNorm {
            let weight = self.vec(width, 0.3).iter().map(|w| w + 1.0).collect();
            LayerNorm::new(weight, Some(self.vec(width, 0.1)), LAYER_NORM_EPS)
        }
    }

    const WIDTH: usize = 16;
    const HEADS: usize = 4;
    const VOCAB: usize = 9;

    fn attention(rng: &mut Rng) -> Attention {
        Attention {
            query: rng.linear(WIDTH, WIDTH, true),
            key: rng.linear(WIDTH, WIDTH, true),
            value: rng.linear(WIDTH, WIDTH, true),
            output: rng.linear(WIDTH, WIDTH, true),
            heads: HEADS,
            head_dim: WIDTH / HEADS,
            width: WIDTH,
        }
    }

    fn decoder(seed: u64) -> Decoder {
        let mut rng = Rng(seed);
        let layers = (0..3)
            .map(|_| DecoderLayer {
                self_norm: rng.norm(WIDTH),
                self_attention: attention(&mut rng),
                cross_norm: rng.norm(WIDTH),
                cross_attention: attention(&mut rng),
                mlp_norm: rng.norm(WIDTH),
                mlp_in: rng.linear(WIDTH, WIDTH * 4, true),
                mlp_out: rng.linear(WIDTH * 4, WIDTH, true),
            })
            .collect();
        Decoder {
            config: FireRedAsr2Config {
                input_dim: 80,
                vocab_size: VOCAB,
                model_dim: WIDTH,
                start_token: 1,
                end_token: 2,
                pad_token: 0,
                encoder_layers: 1,
                encoder_heads: HEADS,
                encoder_kernel: 33,
                encoder_position_limit: 5000,
                decoder_layers: 3,
                decoder_heads: HEADS,
                decoder_position_limit: 5000,
            },
            embeddings: rng.vec(VOCAB * WIDTH, 1.0),
            layers,
            output_norm: rng.norm(WIDTH),
            output_projection: rng.linear(WIDTH, VOCAB, false),
        }
    }

    // ---- original implementation, verbatim apart from the reference_ names ----

    fn reference_attend_last(
        attention: &Attention,
        query_input: &[f32],
        query_rows: usize,
        projected: &ProjectedKeys,
    ) -> Vec<f32> {
        let query = attention
            .query
            .forward(&query_input[(query_rows - 1) * attention.width..], 1);
        let mut context = vec![0.0; attention.width];
        let scale = 1.0 / (attention.head_dim as f32).sqrt();
        let mut scores = vec![0.0; projected.rows];
        for head in 0..attention.heads {
            let query_head = &query[head * attention.head_dim..(head + 1) * attention.head_dim];
            let base = head * projected.rows * attention.head_dim;
            for row in 0..projected.rows {
                let key = &projected.keys
                    [base + row * attention.head_dim..base + (row + 1) * attention.head_dim];
                scores[row] = dot(query_head, key) * scale;
            }
            ops::softmax_row(&mut scores);
            for feature in 0..attention.head_dim {
                let mut sum = 0.0;
                for row in 0..projected.rows {
                    sum +=
                        scores[row] * projected.values[base + row * attention.head_dim + feature];
                }
                context[head * attention.head_dim + feature] = sum;
            }
        }
        attention.output.forward(&context, 1)
    }

    fn reference_step(
        layer: &DecoderLayer,
        input: &[f32],
        rows: usize,
        cross: &ProjectedKeys,
    ) -> Vec<f32> {
        let width = layer.self_attention.width;
        let offset = (rows - 1) * width;
        let mut normalized = input.to_vec();
        layer.self_norm.apply(&mut normalized, rows);
        let all_keys = layer.self_attention.project_keys(&normalized, rows);
        let self_update =
            reference_attend_last(&layer.self_attention, &normalized, rows, &all_keys);
        let mut hidden: Vec<f32> = input[offset..offset + width]
            .iter()
            .zip(self_update)
            .map(|(&x, update)| x + update)
            .collect();

        let mut normalized = hidden.clone();
        layer.cross_norm.apply(&mut normalized, 1);
        let cross_update = reference_attend_last(&layer.cross_attention, &normalized, 1, cross);
        for (x, update) in hidden.iter_mut().zip(cross_update) {
            *x += update;
        }

        let residual = hidden.clone();
        let mut normalized = hidden.clone();
        layer.mlp_norm.apply(&mut normalized, 1);
        let mut feed_forward = layer.mlp_in.forward(&normalized, 1);
        ops::gelu_erf(&mut feed_forward);
        let feed_forward = layer.mlp_out.forward(&feed_forward, 1);
        for ((x, residual), update) in hidden.iter_mut().zip(residual).zip(feed_forward) {
            *x = residual + update;
        }
        hidden
    }

    #[derive(Clone)]
    struct ReferenceBeam {
        tokens: Vec<usize>,
        score: f32,
        finished: bool,
        layer_outputs: Vec<Vec<f32>>,
    }

    fn reference_logits(
        decoder: &Decoder,
        beam: &mut ReferenceBeam,
        cross_keys: &[ProjectedKeys],
        positions: &[f32],
    ) -> Vec<f32> {
        let rows = beam.tokens.len();
        let embeddings = embed_positions(
            &decoder.embeddings,
            &beam.tokens,
            positions,
            decoder.config.model_dim,
        );
        for index in 0..decoder.layers.len() {
            let layer_input = if index == 0 {
                embeddings.as_slice()
            } else {
                beam.layer_outputs[index - 1].as_slice()
            };
            let hidden = reference_step(
                &decoder.layers[index],
                layer_input,
                rows,
                &cross_keys[index],
            );
            beam.layer_outputs[index].extend_from_slice(&hidden);
        }
        let mut hidden = beam.layer_outputs.last().expect("decoder has layers")
            [(rows - 1) * decoder.config.model_dim..rows * decoder.config.model_dim]
            .to_vec();
        decoder.output_norm.apply(&mut hidden, 1);
        decoder.output_projection.forward(&hidden, 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_decode(
        decoder: &Decoder,
        encoder: &[f32],
        encoder_rows: usize,
        dictionary: &[String],
        beam_size: usize,
        max_len: usize,
        softmax_smoothing: f32,
        length_penalty: f32,
        eos_penalty: f32,
    ) -> String {
        let cross_keys: Vec<_> = decoder
            .layers
            .iter()
            .map(|layer| layer.cross_keys(encoder, encoder_rows))
            .collect();
        let mut beams: Vec<ReferenceBeam> = (0..beam_size)
            .map(|index| ReferenceBeam {
                tokens: vec![decoder.config.start_token],
                score: if index == 0 { 0.0 } else { -1e10 },
                finished: false,
                layer_outputs: vec![Vec::new(); decoder.layers.len()],
            })
            .collect();
        let mut positions = Vec::with_capacity(max_len * decoder.config.model_dim);
        positional_encoding(&mut positions, max_len, decoder.config.model_dim);

        for _ in 0..max_len {
            let mut candidates = Vec::with_capacity(beam_size * beam_size);
            for (parent, beam) in beams.iter_mut().enumerate() {
                if beam.finished {
                    candidates.push(Candidate {
                        score: beam.score,
                        parent,
                        token: decoder.config.end_token,
                        finished: true,
                    });
                    for _ in 1..beam_size {
                        candidates.push(Candidate {
                            score: beam.score - 1e10,
                            parent,
                            token: decoder.config.end_token,
                            finished: true,
                        });
                    }
                    continue;
                }
                let logits = reference_logits(decoder, beam, &cross_keys, &positions);
                let mut probabilities = logits
                    .iter()
                    .map(|logit| logit / softmax_smoothing)
                    .collect::<Vec<_>>();
                ops::softmax_row(&mut probabilities);
                let mut log_probs = probabilities
                    .iter()
                    .map(|probability| (probability + 1e-10).ln())
                    .collect::<Vec<_>>();
                log_probs[decoder.config.end_token] *= eos_penalty;
                for (log_probability, token) in top_k(&log_probs, beam_size) {
                    candidates.push(Candidate {
                        score: beam.score + log_probability,
                        parent,
                        token,
                        finished: token == decoder.config.end_token,
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
                let score = |beam: &ReferenceBeam| {
                    let length = beam
                        .tokens
                        .iter()
                        .filter(|&&token| token != decoder.config.end_token)
                        .count() as f32;
                    if length_penalty > 0.0 {
                        beam.score / ((5.0 + length) / 6.0).powf(length_penalty)
                    } else {
                        beam.score
                    }
                };
                score(a).total_cmp(&score(b))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        let mut ids = beams[best]
            .tokens
            .iter()
            .skip(1)
            .copied()
            .collect::<Vec<_>>();
        if let Some(end) = ids
            .iter()
            .position(|&token| token == decoder.config.end_token)
        {
            ids.truncate(end);
        }
        // One distinct string per id, so equal text means equal tokens.
        ids.iter().map(|&id| dictionary[id].clone()).collect()
    }

    fn dictionary() -> Vec<String> {
        (0..VOCAB).map(|id| format!("<t{id}>")).collect()
    }

    fn encoder_states(rng: &mut Rng, rows: usize) -> Vec<f32> {
        rng.vec(rows * WIDTH, 1.5)
    }

    #[test]
    fn cached_step_logits_match_full_prefix_bitwise() {
        for seed in [0x9e37_79b9_u64, 0x1234_5678, 0xdead_beef] {
            let decoder = decoder(seed);
            let mut rng = Rng(seed ^ 0x55aa);
            let encoder = encoder_states(&mut rng, 7);
            let cross_keys: Vec<_> = decoder
                .layers
                .iter()
                .map(|layer| layer.cross_keys(&encoder, 7))
                .collect();
            let max_len = 14;
            let mut positions = Vec::new();
            positional_encoding(&mut positions, max_len, WIDTH);
            let mut reference = ReferenceBeam {
                tokens: vec![1],
                score: 0.0,
                finished: false,
                layer_outputs: vec![Vec::new(); decoder.layers.len()],
            };
            let mut beam = Beam {
                tokens: vec![1],
                score: 0.0,
                finished: false,
                cache: decoder
                    .layers
                    .iter()
                    .map(|layer| layer.self_attention.empty_cache(max_len))
                    .collect(),
            };
            for step in 0..max_len {
                let want = reference_logits(&decoder, &mut reference, &cross_keys, &positions);
                let got = decoder.beam_logits(&mut beam, &cross_keys, &positions);
                assert_eq!(got.len(), want.len());
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "seed {seed} step {step} logit {i}"
                    );
                }
                let token = (rng.next() % VOCAB as u64) as usize;
                reference.tokens.push(token);
                beam.tokens.push(token);
            }
        }
    }

    #[test]
    fn duplicated_cache_continues_identically() {
        // Beam reorder gives two children one parent: the copy must behave
        // exactly like the original from the shared prefix onward.
        let decoder = decoder(0xabcdef);
        let mut rng = Rng(77);
        let encoder = encoder_states(&mut rng, 5);
        let cross_keys: Vec<_> = decoder
            .layers
            .iter()
            .map(|layer| layer.cross_keys(&encoder, 5))
            .collect();
        let max_len = 10;
        let mut positions = Vec::new();
        positional_encoding(&mut positions, max_len, WIDTH);
        let mut parent = Beam {
            tokens: vec![1],
            score: 0.0,
            finished: false,
            cache: decoder
                .layers
                .iter()
                .map(|layer| layer.self_attention.empty_cache(max_len))
                .collect(),
        };
        for token in [4, 6, 3] {
            decoder.beam_logits(&mut parent, &cross_keys, &positions);
            parent.tokens.push(token);
        }
        let mut copy = Beam {
            tokens: parent.tokens.clone(),
            score: 0.0,
            finished: false,
            cache: parent
                .cache
                .iter()
                .zip(&decoder.layers)
                .map(|(cache, layer)| {
                    cache.duplicate(layer.self_attention.heads, layer.self_attention.head_dim)
                })
                .collect(),
        };
        for token in [5, 0, 8] {
            let a = decoder.beam_logits(&mut parent, &cross_keys, &positions);
            let b = decoder.beam_logits(&mut copy, &cross_keys, &positions);
            assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
            parent.tokens.push(token);
            copy.tokens.push(token);
        }
    }

    #[test]
    fn cached_beam_search_matches_full_prefix_tokens() {
        let dictionary = dictionary();
        let mut finished_early = 0;
        for seed in [3u64, 11, 29, 101, 0xfeed, 0xbeef_cafe, 0x1357_9bdf] {
            let decoder = decoder(seed);
            let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
            let encoder = encoder_states(&mut rng, 6);
            for (beam_size, max_len, smoothing, length_penalty, eos_penalty) in [
                (1, 12, 1.0, 0.0, 1.0),
                (3, 12, 1.25, 0.6, 1.0),
                (4, 16, 0.8, 0.6, 1.5),
                (VOCAB, 10, 1.0, 1.0, 0.5),
            ] {
                let want = reference_decode(
                    &decoder,
                    &encoder,
                    6,
                    &dictionary,
                    beam_size,
                    max_len,
                    smoothing,
                    length_penalty,
                    eos_penalty,
                );
                let got = decoder
                    .decode(
                        &encoder,
                        6,
                        &dictionary,
                        beam_size,
                        max_len,
                        smoothing,
                        length_penalty,
                        eos_penalty,
                    )
                    .unwrap();
                assert_eq!(got, want, "seed {seed} beam {beam_size}");
                if want.matches('<').count() < max_len {
                    finished_early += 1;
                }
            }
        }
        // Guard against a vacuous test: some searches must hit EOS and
        // exercise the finished-beam path.
        assert!(finished_early > 0, "no search ended on EOS");
    }
}
