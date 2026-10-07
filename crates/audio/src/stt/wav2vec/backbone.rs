//! Pieces shared by the Wav2Vec2 CTC families (`wav2vec` and `mms`): the
//! rows/channels transposes, the weight-normalized positional convolution,
//! plain multi-head attention, the `vocab.json` reader, and greedy CTC text
//! decoding.
//!
//! The two families differ in the feature-extractor normalization (group
//! versus layer norm), the encoder layer (post-norm versus stable pre-norm
//! plus adapters), and the config type, so those stay in each family and the
//! helpers here take plain dimensions instead of a config.

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use super::ctc::CtcCollapse;
use crate::nn::{argmax, bad_config, load_tensor, Linear};
use crate::ops;
use crate::{Result, SpeechError};

pub(crate) fn positive(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .filter(|&number| number > 0)
        .ok_or_else(|| bad_config(field, "must be a positive integer"))
}

pub(crate) fn positive_array(value: &Value, field: &str) -> Result<Vec<usize>> {
    let values = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| bad_config(field, "must be an integer array"))?;
    values
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .filter(|&number| number > 0)
                .ok_or_else(|| bad_config(field, "contains a non-positive integer"))
        })
        .collect()
}

pub(crate) fn channels_first_to_rows(x: &[f32], channels: usize, steps: usize) -> Vec<f32> {
    let mut rows = vec![0.0f32; x.len()];
    for channel in 0..channels {
        for step in 0..steps {
            rows[step * channels + channel] = x[channel * steps + step];
        }
    }
    rows
}

pub(crate) fn rows_to_channels_first(x: &[f32], steps: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; x.len()];
    for step in 0..steps {
        for channel in 0..channels {
            output[channel * steps + step] = x[step * channels + channel];
        }
    }
    output
}

pub(crate) struct PositionalConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    width: usize,
    kernel: usize,
    groups: usize,
}

impl PositionalConv {
    /// `width` is the model width, `kernel` and `groups` the positional
    /// convolution geometry from the checkpoint config.
    pub(crate) fn load(
        file: &SafetensorsFile,
        width: usize,
        kernel: usize,
        groups: usize,
    ) -> Result<Self> {
        let base = "wav2vec2.encoder.pos_conv_embed.conv";
        let weight_g = load_tensor_alias(
            file,
            &[
                &format!("{base}.weight_g"),
                &format!("{base}.parametrizations.weight.original0"),
            ],
        )?;
        let weight_v = load_tensor_alias(
            file,
            &[
                &format!("{base}.weight_v"),
                &format!("{base}.parametrizations.weight.original1"),
            ],
        )?;
        let in_per_group = width / groups;
        let gain_count = weight_g.len();
        if weight_v.len() != width * in_per_group * kernel
            || !(gain_count == 1
                || gain_count == kernel
                || gain_count == width
                || gain_count == width * kernel)
        {
            return Err(SpeechError::Tensor {
                name: base.to_owned(),
                why: format!(
                    "unsupported positional convolution weights: g={}, v={}, expected v={} and broadcast g",
                    weight_g.len(),
                    weight_v.len(),
                    width * in_per_group * kernel
                ),
            });
        }

        let mut norm = vec![0.0f32; kernel];
        for out in 0..width {
            for input in 0..in_per_group {
                for (position, norm_slot) in norm.iter_mut().enumerate() {
                    let source = (out * in_per_group + input) * kernel + position;
                    *norm_slot += weight_v[source] * weight_v[source];
                }
            }
        }
        for value in &mut norm {
            *value = value.sqrt().max(1e-12);
        }

        let mut weight = vec![0.0f32; weight_v.len()];
        for out in 0..width {
            for input in 0..in_per_group {
                for position in 0..kernel {
                    let source = (out * in_per_group + input) * kernel + position;
                    let gain = match weight_g.len() {
                        1 => weight_g[0],
                        n if n == kernel => weight_g[position],
                        n if n == width => weight_g[out],
                        _ => weight_g[out * kernel + position],
                    };
                    weight[source] = gain * weight_v[source] / norm[position];
                }
            }
        }
        Ok(Self {
            weight,
            bias: load_tensor(file, &format!("{base}.bias"), &[width])?,
            width,
            kernel,
            groups,
        })
    }

    pub(crate) fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let channels_first = rows_to_channels_first(x, steps, self.width);
        let convolved = ops::conv1d(
            &channels_first,
            &self.weight,
            Some(&self.bias),
            self.width,
            self.width,
            self.kernel,
            1,
            self.kernel / 2,
            1,
            self.groups,
        );
        let padded_steps = steps + usize::from(self.kernel % 2 == 0);
        let mut rows = channels_first_to_rows(&convolved, self.width, padded_steps);
        rows.truncate(steps * self.width);
        ops::gelu_erf(&mut rows);
        rows
    }
}

pub(crate) struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    width: usize,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    pub(crate) fn load(
        file: &SafetensorsFile,
        prefix: &str,
        width: usize,
        heads: usize,
    ) -> Result<Self> {
        Ok(Self {
            query: Linear::load(file, &format!("{prefix}.q_proj"), width, width, true)?,
            key: Linear::load(file, &format!("{prefix}.k_proj"), width, width, true)?,
            value: Linear::load(file, &format!("{prefix}.v_proj"), width, width, true)?,
            output: Linear::load(file, &format!("{prefix}.out_proj"), width, width, true)?,
            width,
            heads,
            head_dim: width / heads,
        })
    }

    pub(crate) fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let query = self.query.forward(x, steps);
        let key = self.key.forward(x, steps);
        let value = self.value.forward(x, steps);
        let mut attended = vec![0.0f32; steps * self.width];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; steps];
        for head in 0..self.heads {
            for query_step in 0..steps {
                let q_offset = query_step * self.width + head * self.head_dim;
                for (key_step, score_slot) in scores.iter_mut().enumerate() {
                    let k_offset = key_step * self.width + head * self.head_dim;
                    let mut score = 0.0f32;
                    for dim in 0..self.head_dim {
                        score += query[q_offset + dim] * key[k_offset + dim];
                    }
                    *score_slot = score * scale;
                }
                ops::softmax_row(&mut scores);
                let out_offset = query_step * self.width + head * self.head_dim;
                for dim in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for key_step in 0..steps {
                        sum += scores[key_step]
                            * value[key_step * self.width + head * self.head_dim + dim];
                    }
                    attended[out_offset + dim] = sum;
                }
            }
        }
        self.output.forward(&attended, steps)
    }
}

pub(crate) fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    debug_assert_eq!(left.len(), right.len());
    left.iter().zip(right).map(|(&a, &b)| a + b).collect()
}

pub(crate) fn load_tensor_alias(file: &SafetensorsFile, names: &[&str]) -> Result<Vec<f32>> {
    let name = names
        .iter()
        .find(|name| file.contains_tensor(name))
        .ok_or_else(|| SpeechError::Tensor {
            name: names.join(" or "),
            why: "tensor is missing".into(),
        })?;
    Ok(file.load_as_f32(name)?)
}

/// Reads `vocab.json` as a token-to-id map into an id-indexed vector.
///
/// With `per_language` set (the MMS adapter checkpoints), a map of language
/// maps is accepted and the `eng`, then `en`, then first language is used.
pub(crate) fn parse_vocab(
    json: &str,
    vocab_size: usize,
    per_language: bool,
) -> Result<Vec<String>> {
    let value: Value =
        serde_json::from_str(json).map_err(|error| bad_config("vocab.json", error.to_string()))?;
    let entries = value
        .as_object()
        .ok_or_else(|| bad_config("vocab.json", "must be a token-to-id object"))?;
    let token_ids = if per_language && entries.values().any(Value::is_object) {
        entries
            .get("eng")
            .or_else(|| entries.get("en"))
            .and_then(Value::as_object)
            .or_else(|| entries.values().find_map(Value::as_object))
            .ok_or_else(|| bad_config("vocab.json", "has no language vocabulary object"))?
    } else {
        entries
    };
    let mut vocab = vec![None; vocab_size];
    for (token, id) in token_ids {
        let id = id
            .as_u64()
            .and_then(|id| usize::try_from(id).ok())
            .filter(|&id| id < vocab_size)
            .ok_or_else(|| bad_config("vocab.json", "token id is outside vocab_size"))?;
        if vocab[id].replace(token.clone()).is_some() {
            return Err(bad_config("vocab.json", "contains duplicate token ids"));
        }
    }
    vocab
        .into_iter()
        .enumerate()
        .map(|(id, token)| {
            token.ok_or_else(|| bad_config("vocab.json", format!("missing token id {id}")))
        })
        .collect()
}

/// Greedy CTC over `[steps, vocab_size]` logits: blank is id 0, `|` is the
/// word delimiter.
pub(crate) fn decode_ctc(
    logits: &[f32],
    steps: usize,
    vocab_size: usize,
    vocab: &[String],
) -> Result<String> {
    if logits.len() != steps * vocab_size || vocab.len() != vocab_size {
        return Err(SpeechError::Tensor {
            name: "lm_head.logits".into(),
            why: "logit or vocabulary dimensions do not match the configured CTC head".into(),
        });
    }
    let mut collapse = CtcCollapse::new(0);
    for row in logits.chunks_exact(vocab_size) {
        collapse.push(argmax(row));
    }
    let pieces: Vec<&str> = collapse
        .finish()
        .into_iter()
        .map(|token| vocab[token].as_str())
        .collect();
    Ok(pieces.join("").replace('|', " ").trim().to_owned())
}

#[cfg(test)]
mod tests {
    // The retired MMS copies are kept verbatim, including their index loops.
    #![allow(dead_code, clippy::needless_range_loop)]

    use super::*;
    use crate::stt::mms::MmsConfig;
    use crate::stt::wav2vec::Wav2VecConfig;
    use serde_json::json;

    // ---- retired wav2vec implementation (verbatim) ----
    mod wav2vec_ref {
        use crate::nn::{bad_config, load_tensor, Linear};
        use crate::ops;
        use crate::stt::wav2vec::Wav2VecConfig;
        use crate::{Result, SpeechError};
        use serde_json::Value;
        use turbospark_model_io::safetensors::SafetensorsFile;

        pub(super) fn channels_first_to_rows(x: &[f32], channels: usize, steps: usize) -> Vec<f32> {
            let mut rows = vec![0.0f32; x.len()];
            for channel in 0..channels {
                for step in 0..steps {
                    rows[step * channels + channel] = x[channel * steps + step];
                }
            }
            rows
        }

        pub(super) fn rows_to_channels_first(x: &[f32], steps: usize, channels: usize) -> Vec<f32> {
            let mut output = vec![0.0f32; x.len()];
            for step in 0..steps {
                for channel in 0..channels {
                    output[channel * steps + step] = x[step * channels + channel];
                }
            }
            output
        }

        pub(super) struct PositionalConv {
            pub(super) weight: Vec<f32>,
            pub(super) bias: Vec<f32>,
            pub(super) width: usize,
            pub(super) kernel: usize,
            pub(super) groups: usize,
        }

        impl PositionalConv {
            pub(super) fn load(file: &SafetensorsFile, config: &Wav2VecConfig) -> Result<Self> {
                let width = config.hidden_size;
                let kernel = config.num_conv_pos_embeddings;
                let groups = config.num_conv_pos_embedding_groups;
                let base = "wav2vec2.encoder.pos_conv_embed.conv";
                let weight_g = load_tensor_alias(
                    file,
                    &[
                        &format!("{base}.weight_g"),
                        &format!("{base}.parametrizations.weight.original0"),
                    ],
                )?;
                let weight_v = load_tensor_alias(
                    file,
                    &[
                        &format!("{base}.weight_v"),
                        &format!("{base}.parametrizations.weight.original1"),
                    ],
                )?;
                let in_per_group = width / groups;
                let gain_count = weight_g.len();
                if weight_v.len() != width * in_per_group * kernel
                    || !(gain_count == 1
                        || gain_count == kernel
                        || gain_count == width
                        || gain_count == width * kernel)
                {
                    return Err(SpeechError::Tensor {
                        name: base.to_owned(),
                        why: format!(
                            "unsupported positional convolution weights: g={}, v={}, expected v={} and broadcast g",
                            weight_g.len(),
                            weight_v.len(),
                            width * in_per_group * kernel
                        ),
                    });
                }

                let mut norm = vec![0.0f32; kernel];
                for out in 0..width {
                    for input in 0..in_per_group {
                        for (position, norm_slot) in norm.iter_mut().enumerate() {
                            let source = (out * in_per_group + input) * kernel + position;
                            *norm_slot += weight_v[source] * weight_v[source];
                        }
                    }
                }
                for value in &mut norm {
                    *value = value.sqrt().max(1e-12);
                }

                let mut weight = vec![0.0f32; weight_v.len()];
                for out in 0..width {
                    for input in 0..in_per_group {
                        for position in 0..kernel {
                            let source = (out * in_per_group + input) * kernel + position;
                            let gain = match weight_g.len() {
                                1 => weight_g[0],
                                n if n == kernel => weight_g[position],
                                n if n == width => weight_g[out],
                                _ => weight_g[out * kernel + position],
                            };
                            weight[source] = gain * weight_v[source] / norm[position];
                        }
                    }
                }
                Ok(Self {
                    weight,
                    bias: load_tensor(file, &format!("{base}.bias"), &[width])?,
                    width,
                    kernel,
                    groups,
                })
            }

            pub(super) fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
                let channels_first = rows_to_channels_first(x, steps, self.width);
                let convolved = ops::conv1d(
                    &channels_first,
                    &self.weight,
                    Some(&self.bias),
                    self.width,
                    self.width,
                    self.kernel,
                    1,
                    self.kernel / 2,
                    1,
                    self.groups,
                );
                let padded_steps = steps + usize::from(self.kernel % 2 == 0);
                let mut rows = channels_first_to_rows(&convolved, self.width, padded_steps);
                rows.truncate(steps * self.width);
                ops::gelu_erf(&mut rows);
                rows
            }
        }

        pub(super) struct Attention {
            pub(super) query: Linear,
            pub(super) key: Linear,
            pub(super) value: Linear,
            pub(super) output: Linear,
            pub(super) width: usize,
            pub(super) heads: usize,
            pub(super) head_dim: usize,
        }

        impl Attention {
            pub(super) fn load(
                file: &SafetensorsFile,
                prefix: &str,
                config: &Wav2VecConfig,
            ) -> Result<Self> {
                let width = config.hidden_size;
                Ok(Self {
                    query: Linear::load(file, &format!("{prefix}.q_proj"), width, width, true)?,
                    key: Linear::load(file, &format!("{prefix}.k_proj"), width, width, true)?,
                    value: Linear::load(file, &format!("{prefix}.v_proj"), width, width, true)?,
                    output: Linear::load(file, &format!("{prefix}.out_proj"), width, width, true)?,
                    width,
                    heads: config.num_attention_heads,
                    head_dim: width / config.num_attention_heads,
                })
            }

            pub(super) fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
                let query = self.query.forward(x, steps);
                let key = self.key.forward(x, steps);
                let value = self.value.forward(x, steps);
                let mut attended = vec![0.0f32; steps * self.width];
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                let mut scores = vec![0.0f32; steps];
                for head in 0..self.heads {
                    for query_step in 0..steps {
                        let q_offset = query_step * self.width + head * self.head_dim;
                        for (key_step, score_slot) in scores.iter_mut().enumerate() {
                            let k_offset = key_step * self.width + head * self.head_dim;
                            let mut score = 0.0f32;
                            for dim in 0..self.head_dim {
                                score += query[q_offset + dim] * key[k_offset + dim];
                            }
                            *score_slot = score * scale;
                        }
                        ops::softmax_row(&mut scores);
                        let out_offset = query_step * self.width + head * self.head_dim;
                        for dim in 0..self.head_dim {
                            let mut sum = 0.0f32;
                            for key_step in 0..steps {
                                sum += scores[key_step]
                                    * value[key_step * self.width + head * self.head_dim + dim];
                            }
                            attended[out_offset + dim] = sum;
                        }
                    }
                }
                self.output.forward(&attended, steps)
            }
        }

        pub(super) fn load_tensor_alias(
            file: &SafetensorsFile,
            names: &[&str],
        ) -> Result<Vec<f32>> {
            let name = names
                .iter()
                .find(|name| file.contains_tensor(name))
                .ok_or_else(|| SpeechError::Tensor {
                    name: names.join(" or "),
                    why: "tensor is missing".into(),
                })?;
            Ok(file.load_as_f32(name)?)
        }

        pub(super) fn parse_vocab(json: &str, vocab_size: usize) -> Result<Vec<String>> {
            let value: Value = serde_json::from_str(json)
                .map_err(|error| bad_config("vocab.json", error.to_string()))?;
            let entries = value
                .as_object()
                .ok_or_else(|| bad_config("vocab.json", "must be a token-to-id object"))?;
            let mut vocab = vec![None; vocab_size];
            for (token, id) in entries {
                let id = id
                    .as_u64()
                    .and_then(|id| usize::try_from(id).ok())
                    .filter(|&id| id < vocab_size)
                    .ok_or_else(|| bad_config("vocab.json", "token id is outside vocab_size"))?;
                if vocab[id].replace(token.clone()).is_some() {
                    return Err(bad_config("vocab.json", "contains duplicate token ids"));
                }
            }
            vocab
                .into_iter()
                .enumerate()
                .map(|(id, token)| {
                    token.ok_or_else(|| bad_config("vocab.json", format!("missing token id {id}")))
                })
                .collect()
        }

        pub(super) fn decode_ctc(
            logits: &[f32],
            steps: usize,
            vocab_size: usize,
            vocab: &[String],
        ) -> Result<String> {
            if logits.len() != steps * vocab_size || vocab.len() != vocab_size {
                return Err(SpeechError::Tensor {
                    name: "lm_head.logits".into(),
                    why: "logit or vocabulary dimensions do not match the configured CTC head"
                        .into(),
                });
            }
            let mut pieces = Vec::new();
            let mut previous = None;
            for row in logits.chunks_exact(vocab_size) {
                let mut token = 0;
                let mut best = f32::NEG_INFINITY;
                for (index, &score) in row.iter().enumerate() {
                    if score > best {
                        token = index;
                        best = score;
                    }
                }
                if token != previous.unwrap_or(usize::MAX) && token != 0 {
                    pieces.push(vocab[token].as_str());
                }
                previous = Some(token);
            }
            Ok(pieces.join("").replace('|', " ").trim().to_owned())
        }
    }

    // ---- retired MMS implementation (verbatim) ----
    mod mms_ref {
        use crate::nn::{bad_config, load_tensor, Linear};
        use crate::ops;
        use crate::stt::mms::MmsConfig;
        use crate::{Result, SpeechError};
        use serde_json::Value;
        use turbospark_model_io::safetensors::SafetensorsFile;

        pub(super) fn channels_first_to_rows(x: &[f32], channels: usize, steps: usize) -> Vec<f32> {
            let mut rows = vec![0.0f32; x.len()];
            for channel in 0..channels {
                for step in 0..steps {
                    rows[step * channels + channel] = x[channel * steps + step];
                }
            }
            rows
        }

        pub(super) fn rows_to_channels_first(x: &[f32], steps: usize, channels: usize) -> Vec<f32> {
            let mut output = vec![0.0f32; x.len()];
            for step in 0..steps {
                for channel in 0..channels {
                    output[channel * steps + step] = x[step * channels + channel];
                }
            }
            output
        }

        pub(super) struct PositionalConv {
            pub(super) weight: Vec<f32>,
            pub(super) bias: Vec<f32>,
            pub(super) width: usize,
            pub(super) kernel: usize,
            pub(super) groups: usize,
        }

        impl PositionalConv {
            pub(super) fn load(file: &SafetensorsFile, config: &MmsConfig) -> Result<Self> {
                let width = config.hidden_size;
                let kernel = config.num_conv_pos_embeddings;
                let groups = config.num_conv_pos_embedding_groups;
                let base = "wav2vec2.encoder.pos_conv_embed.conv";
                let weight_g = load_tensor_alias(
                    file,
                    &[
                        &format!("{base}.weight_g"),
                        &format!("{base}.parametrizations.weight.original0"),
                    ],
                )?;
                let weight_v = load_tensor_alias(
                    file,
                    &[
                        &format!("{base}.weight_v"),
                        &format!("{base}.parametrizations.weight.original1"),
                    ],
                )?;
                let in_per_group = width / groups;
                let gain_count = weight_g.len();
                if weight_v.len() != width * in_per_group * kernel
                    || !(gain_count == 1
                        || gain_count == kernel
                        || gain_count == width
                        || gain_count == width * kernel)
                {
                    return Err(SpeechError::Tensor {
                        name: base.to_owned(),
                        why: format!(
                            "unsupported positional convolution weights: g={}, v={}, expected v={} and broadcast g",
                            weight_g.len(),
                            weight_v.len(),
                            width * in_per_group * kernel
                        ),
                    });
                }

                let mut norm = vec![0.0f32; kernel];
                for out in 0..width {
                    for input in 0..in_per_group {
                        for position in 0..kernel {
                            let source = (out * in_per_group + input) * kernel + position;
                            norm[position] += weight_v[source] * weight_v[source];
                        }
                    }
                }
                for value in &mut norm {
                    *value = value.sqrt().max(1e-12);
                }

                let mut weight = vec![0.0f32; weight_v.len()];
                for out in 0..width {
                    for input in 0..in_per_group {
                        for position in 0..kernel {
                            let source = (out * in_per_group + input) * kernel + position;
                            let gain = match weight_g.len() {
                                1 => weight_g[0],
                                n if n == kernel => weight_g[position],
                                n if n == width => weight_g[out],
                                _ => weight_g[out * kernel + position],
                            };
                            weight[source] = gain * weight_v[source] / norm[position];
                        }
                    }
                }
                Ok(Self {
                    weight,
                    bias: load_tensor(file, &format!("{base}.bias"), &[width])?,
                    width,
                    kernel,
                    groups,
                })
            }

            pub(super) fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
                let channels_first = rows_to_channels_first(x, steps, self.width);
                let convolved = ops::conv1d(
                    &channels_first,
                    &self.weight,
                    Some(&self.bias),
                    self.width,
                    self.width,
                    self.kernel,
                    1,
                    self.kernel / 2,
                    1,
                    self.groups,
                );
                let padded_steps = steps + usize::from(self.kernel % 2 == 0);
                let mut rows = channels_first_to_rows(&convolved, self.width, padded_steps);
                rows.truncate(steps * self.width);
                ops::gelu_erf(&mut rows);
                rows
            }
        }

        pub(super) struct Attention {
            pub(super) query: Linear,
            pub(super) key: Linear,
            pub(super) value: Linear,
            pub(super) output: Linear,
            pub(super) width: usize,
            pub(super) heads: usize,
            pub(super) head_dim: usize,
        }

        impl Attention {
            pub(super) fn load(
                file: &SafetensorsFile,
                prefix: &str,
                config: &MmsConfig,
            ) -> Result<Self> {
                let width = config.hidden_size;
                Ok(Self {
                    query: Linear::load(file, &format!("{prefix}.q_proj"), width, width, true)?,
                    key: Linear::load(file, &format!("{prefix}.k_proj"), width, width, true)?,
                    value: Linear::load(file, &format!("{prefix}.v_proj"), width, width, true)?,
                    output: Linear::load(file, &format!("{prefix}.out_proj"), width, width, true)?,
                    width,
                    heads: config.num_attention_heads,
                    head_dim: width / config.num_attention_heads,
                })
            }

            pub(super) fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
                let query = self.query.forward(x, steps);
                let key = self.key.forward(x, steps);
                let value = self.value.forward(x, steps);
                let mut attended = vec![0.0f32; steps * self.width];
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                let mut scores = vec![0.0f32; steps];
                for head in 0..self.heads {
                    for query_step in 0..steps {
                        let q_offset = query_step * self.width + head * self.head_dim;
                        for key_step in 0..steps {
                            let k_offset = key_step * self.width + head * self.head_dim;
                            let mut score = 0.0f32;
                            for dim in 0..self.head_dim {
                                score += query[q_offset + dim] * key[k_offset + dim];
                            }
                            scores[key_step] = score * scale;
                        }
                        ops::softmax_row(&mut scores);
                        let out_offset = query_step * self.width + head * self.head_dim;
                        for dim in 0..self.head_dim {
                            let mut sum = 0.0f32;
                            for key_step in 0..steps {
                                sum += scores[key_step]
                                    * value[key_step * self.width + head * self.head_dim + dim];
                            }
                            attended[out_offset + dim] = sum;
                        }
                    }
                }
                self.output.forward(&attended, steps)
            }
        }

        pub(super) fn load_tensor_alias(
            file: &SafetensorsFile,
            names: &[&str],
        ) -> Result<Vec<f32>> {
            let name = names
                .iter()
                .find(|name| file.contains_tensor(name))
                .ok_or_else(|| SpeechError::Tensor {
                    name: names.join(" or "),
                    why: "tensor is missing".into(),
                })?;
            Ok(file.load_as_f32(name)?)
        }

        pub(super) fn parse_vocab(json: &str, vocab_size: usize) -> Result<Vec<String>> {
            let value: Value = serde_json::from_str(json)
                .map_err(|error| bad_config("vocab.json", error.to_string()))?;
            let entries = value
                .as_object()
                .ok_or_else(|| bad_config("vocab.json", "must be a token-to-id object"))?;
            let token_ids = if entries.values().any(Value::is_object) {
                entries
                    .get("eng")
                    .or_else(|| entries.get("en"))
                    .and_then(Value::as_object)
                    .or_else(|| entries.values().find_map(Value::as_object))
                    .ok_or_else(|| bad_config("vocab.json", "has no language vocabulary object"))?
            } else {
                entries
            };
            let mut vocab = vec![None; vocab_size];
            for (token, id) in token_ids {
                let id = id
                    .as_u64()
                    .and_then(|id| usize::try_from(id).ok())
                    .filter(|&id| id < vocab_size)
                    .ok_or_else(|| bad_config("vocab.json", "token id is outside vocab_size"))?;
                if vocab[id].replace(token.clone()).is_some() {
                    return Err(bad_config("vocab.json", "contains duplicate token ids"));
                }
            }
            vocab
                .into_iter()
                .enumerate()
                .map(|(id, token)| {
                    token.ok_or_else(|| bad_config("vocab.json", format!("missing token id {id}")))
                })
                .collect()
        }

        pub(super) fn decode_ctc(
            logits: &[f32],
            steps: usize,
            vocab_size: usize,
            vocab: &[String],
        ) -> Result<String> {
            if logits.len() != steps * vocab_size || vocab.len() != vocab_size {
                return Err(SpeechError::Tensor {
                    name: "lm_head.logits".into(),
                    why: "logit or vocabulary dimensions do not match the configured CTC head"
                        .into(),
                });
            }
            let mut pieces = Vec::new();
            let mut previous = None;
            for row in logits.chunks_exact(vocab_size) {
                let mut token = 0;
                let mut best = f32::NEG_INFINITY;
                for (index, &score) in row.iter().enumerate() {
                    if score > best {
                        token = index;
                        best = score;
                    }
                }
                if token != previous.unwrap_or(usize::MAX) && token != 0 {
                    pieces.push(vocab[token].as_str());
                }
                previous = Some(token);
            }
            Ok(pieces.join("").replace('|', " ").trim().to_owned())
        }
    }

    struct Rng(u32);

    impl Rng {
        fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
            (0..len)
                .map(|_| {
                    self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
                })
                .collect()
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn write_safetensors(
        tag: &str,
        tensors: &[(String, Vec<usize>, Vec<f32>)],
    ) -> std::path::PathBuf {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                name.clone(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let header = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data);
        let path = std::env::temp_dir().join(format!(
            "turbospark_backbone_{tag}_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, out).unwrap();
        path
    }

    const WIDTH: usize = 16;
    const GROUPS: usize = 4;
    const HEADS: usize = 4;

    fn wav2vec_config(kernel: usize) -> Wav2VecConfig {
        Wav2VecConfig {
            vocab_size: 5,
            hidden_size: WIDTH,
            num_hidden_layers: 1,
            num_attention_heads: HEADS,
            intermediate_size: 24,
            layer_norm_eps: 1e-5,
            conv_dim: vec![4],
            conv_stride: vec![2],
            conv_kernel: vec![3],
            conv_bias: false,
            num_conv_pos_embeddings: kernel,
            num_conv_pos_embedding_groups: GROUPS,
        }
    }

    fn mms_config(kernel: usize) -> MmsConfig {
        MmsConfig {
            vocab_size: 5,
            hidden_size: WIDTH,
            num_hidden_layers: 1,
            num_attention_heads: HEADS,
            intermediate_size: 24,
            layer_norm_eps: 1e-5,
            conv_dim: vec![4],
            conv_stride: vec![2],
            conv_kernel: vec![3],
            conv_bias: false,
            num_conv_pos_embeddings: kernel,
            num_conv_pos_embedding_groups: GROUPS,
            adapter_attn_dim: 6,
        }
    }

    #[test]
    fn positional_conv_matches_both_retired_copies_bitwise() {
        let base = "wav2vec2.encoder.pos_conv_embed.conv";
        let in_per_group = WIDTH / GROUPS;
        for kernel in [4usize, 5] {
            for (gain_len, legacy_names) in [
                (1usize, true),
                (kernel, false),
                (WIDTH, true),
                (WIDTH * kernel, false),
            ] {
                let mut rng = Rng(kernel as u32 * 31 + gain_len as u32);
                let (g_name, v_name) = if legacy_names {
                    (format!("{base}.weight_g"), format!("{base}.weight_v"))
                } else {
                    (
                        format!("{base}.parametrizations.weight.original0"),
                        format!("{base}.parametrizations.weight.original1"),
                    )
                };
                let path = write_safetensors(
                    "posconv",
                    &[
                        (g_name, vec![gain_len], rng.vec(gain_len, 2.0)),
                        (
                            v_name,
                            vec![WIDTH, in_per_group, kernel],
                            rng.vec(WIDTH * in_per_group * kernel, 1.0),
                        ),
                        (format!("{base}.bias"), vec![WIDTH], rng.vec(WIDTH, 0.5)),
                    ],
                );
                let file = SafetensorsFile::open(&path).unwrap();
                let shared = PositionalConv::load(&file, WIDTH, kernel, GROUPS).unwrap();
                let old_w =
                    wav2vec_ref::PositionalConv::load(&file, &wav2vec_config(kernel)).unwrap();
                let old_m = mms_ref::PositionalConv::load(&file, &mms_config(kernel)).unwrap();
                assert_eq!(bits(&shared.weight), bits(&old_w.weight), "gain {gain_len}");
                assert_eq!(bits(&shared.weight), bits(&old_m.weight));
                assert_eq!(bits(&shared.bias), bits(&old_w.bias));
                for steps in [1usize, 6, 11] {
                    let x = rng.vec(steps * WIDTH, 2.0);
                    let got = shared.forward(&x, steps);
                    assert_eq!(bits(&got), bits(&old_w.forward(&x, steps)));
                    assert_eq!(bits(&got), bits(&old_m.forward(&x, steps)));
                }
                let _ = std::fs::remove_file(path);
            }
        }
    }

    #[test]
    fn unsupported_positional_weights_report_the_retired_error() {
        let base = "wav2vec2.encoder.pos_conv_embed.conv";
        let path = write_safetensors(
            "posconv_bad",
            &[
                (format!("{base}.weight_g"), vec![3], vec![1.0; 3]),
                (
                    format!("{base}.weight_v"),
                    vec![WIDTH, WIDTH / GROUPS, 4],
                    vec![0.5; WIDTH * (WIDTH / GROUPS) * 4],
                ),
                (format!("{base}.bias"), vec![WIDTH], vec![0.0; WIDTH]),
            ],
        );
        let file = SafetensorsFile::open(&path).unwrap();
        let new = PositionalConv::load(&file, WIDTH, 4, GROUPS).err().unwrap();
        let old = wav2vec_ref::PositionalConv::load(&file, &wav2vec_config(4))
            .err()
            .unwrap();
        assert_eq!(new.to_string(), old.to_string());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn attention_matches_both_retired_copies_bitwise() {
        let mut rng = Rng(77);
        let prefix = "layer.attention";
        let mut tensors = Vec::new();
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            tensors.push((
                format!("{prefix}.{name}.weight"),
                vec![WIDTH, WIDTH],
                rng.vec(WIDTH * WIDTH, 0.8),
            ));
            tensors.push((
                format!("{prefix}.{name}.bias"),
                vec![WIDTH],
                rng.vec(WIDTH, 0.3),
            ));
        }
        let path = write_safetensors("attention", &tensors);
        let file = SafetensorsFile::open(&path).unwrap();
        let shared = Attention::load(&file, prefix, WIDTH, HEADS).unwrap();
        let old_w = wav2vec_ref::Attention::load(&file, prefix, &wav2vec_config(4)).unwrap();
        let old_m = mms_ref::Attention::load(&file, prefix, &mms_config(4)).unwrap();
        for steps in [1usize, 2, 9, 17] {
            let x = rng.vec(steps * WIDTH, 2.0);
            let got = shared.forward(&x, steps);
            assert_eq!(bits(&got), bits(&old_w.forward(&x, steps)));
            assert_eq!(bits(&got), bits(&old_m.forward(&x, steps)));
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn transposes_and_vocab_and_ctc_text_match_the_retired_copies() {
        let mut rng = Rng(5);
        let x = rng.vec(6 * 7, 1.0);
        assert_eq!(
            bits(&channels_first_to_rows(&x, 6, 7)),
            bits(&wav2vec_ref::channels_first_to_rows(&x, 6, 7))
        );
        assert_eq!(
            bits(&rows_to_channels_first(&x, 7, 6)),
            bits(&wav2vec_ref::rows_to_channels_first(&x, 7, 6))
        );

        // Flat vocab: both retired families agree; the language-map form
        // only the MMS copy accepted.
        let flat = r#"{"<pad>": 0, "|": 1, "E": 2, "T": 3}"#;
        assert_eq!(
            parse_vocab(flat, 4, false).unwrap(),
            wav2vec_ref::parse_vocab(flat, 4).unwrap()
        );
        assert_eq!(
            parse_vocab(flat, 4, true).unwrap(),
            mms_ref::parse_vocab(flat, 4).unwrap()
        );
        let nested = r#"{"fra":{"<pad>":0,"z":1},"eng":{"<pad>":0,"h":1}}"#;
        assert_eq!(
            parse_vocab(nested, 2, true).unwrap(),
            mms_ref::parse_vocab(nested, 2).unwrap()
        );
        for (bad, size) in [
            ("[]", 2usize),
            (r#"{"a": 0, "b": 0}"#, 2),
            (r#"{"a": 5}"#, 2),
            (r#"{"a": 0}"#, 2),
            ("not json", 2),
        ] {
            assert_eq!(
                parse_vocab(bad, size, false).err().map(|e| e.to_string()),
                wav2vec_ref::parse_vocab(bad, size)
                    .err()
                    .map(|e| e.to_string()),
                "{bad}"
            );
            assert_eq!(
                parse_vocab(bad, size, true).err().map(|e| e.to_string()),
                mms_ref::parse_vocab(bad, size).err().map(|e| e.to_string()),
                "{bad}"
            );
        }

        let vocab: Vec<String> = ["<pad>", "|", "E", "T"].map(String::from).to_vec();
        let mut state = 3u32;
        for steps in [0usize, 1, 4, 25] {
            let logits: Vec<f32> = (0..steps * 4)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    match (state >> 24) % 9 {
                        0 => f32::NAN,
                        v => (v % 3) as f32,
                    }
                })
                .collect();
            assert_eq!(
                decode_ctc(&logits, steps, 4, &vocab).unwrap(),
                wav2vec_ref::decode_ctc(&logits, steps, 4, &vocab).unwrap()
            );
        }
        let mismatch = decode_ctc(&[0.0; 3], 1, 4, &vocab).err().unwrap();
        let retired = wav2vec_ref::decode_ctc(&[0.0; 3], 1, 4, &vocab)
            .err()
            .unwrap();
        assert_eq!(mismatch.to_string(), retired.to_string());
    }
}
