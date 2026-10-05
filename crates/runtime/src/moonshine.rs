//! Moonshine Tiny Metal runner. The portable audio crate remains the CPU
//! reference; this module owns resident GPU weights and request-sized scratch.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use audio::stt::moonshine::{Moonshine, MoonshineConfig, Tokenizer};
use gpu::{
    autorelease_pool, encode_moonshine_conv1d, encode_moonshine_embed, encode_moonshine_groupnorm,
    encode_moonshine_rope, encode_moonshine_swiglu, encode_moonshine_tanh, encode_whisper_add,
    encode_whisper_attn_step, encode_whisper_gelu_erf, encode_whisper_gemv,
    encode_whisper_layer_norm, encode_whisper_matmul_bias, encode_whisper_softmax_rows,
    encode_whisper_transpose, read_f32_buffer, write_buffer_bytes, F32View, MetalBuffer,
    MetalContext, PassEncoder, MAX_ATTN_STEP,
};
use model_io::safetensors::SafetensorsFile;

const MAX_DECODE_TOKENS: usize = 448;

fn view(buffer: &MetalBuffer) -> F32View<'_> {
    F32View::new(buffer)
}

fn bytes(values: &[f32]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    raw
}

fn elems(count: usize) -> Result<u64, String> {
    count
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| "Moonshine buffer size overflows".to_string())
}

fn output(context: &MetalContext, count: usize) -> Result<MetalBuffer, String> {
    Ok(context.new_output_buffer(elems(count)?))
}

fn as_u32(value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| "Moonshine tensor dimension exceeds u32".to_string())
}

fn gpu_result<T>(result: Result<T, gpu::GpuError>) -> Result<T, String> {
    result.map_err(|error| error.to_string())
}

struct LayerCache {
    self_k: MetalBuffer,
    self_v: MetalBuffer,
    cross_k: MetalBuffer,
    cross_v: MetalBuffer,
    cross_v_transposed: MetalBuffer,
}

/// Every activation buffer is sized for one request and reused for requests
/// that fit. The encoder score store is one sequence-square plane reused
/// head by head, rather than a persistent heads-times-sequence-square tensor.
struct Work {
    samples_capacity: usize,
    sequence_capacity: usize,
    input: MetalBuffer,
    conv1: MetalBuffer,
    conv1_norm: MetalBuffer,
    conv2: MetalBuffer,
    conv3: MetalBuffer,
    hidden: MetalBuffer,
    normed: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    mix: MetalBuffer,
    scores: MetalBuffer,
    ffn: MetalBuffer,
    tmp: MetalBuffer,
    encoder_out: MetalBuffer,
    decoder_hidden: MetalBuffer,
    decoder_next: MetalBuffer,
    decoder_normed: MetalBuffer,
    decoder_qkv: MetalBuffer,
    decoder_attention: MetalBuffer,
    decoder_cross_q: MetalBuffer,
    decoder_cross_mix: MetalBuffer,
    decoder_ffn: MetalBuffer,
    decoder_gate: MetalBuffer,
    logits: MetalBuffer,
    caches: Vec<LayerCache>,
}

impl Work {
    fn new(
        context: &MetalContext,
        config: &MoonshineConfig,
        samples: usize,
        seq: usize,
    ) -> Result<Self, String> {
        let d = config.hidden_size;
        let mid = config.intermediate_size;
        let seq1 = (samples - 127) / 64 + 1;
        let seq2 = (seq1 - 7) / 3 + 1;
        let caches = (0..config.decoder_layers)
            .map(|_| {
                Ok(LayerCache {
                    self_k: output(context, MAX_DECODE_TOKENS * d)?,
                    self_v: output(context, MAX_DECODE_TOKENS * d)?,
                    cross_k: output(context, seq * d)?,
                    cross_v: output(context, seq * d)?,
                    cross_v_transposed: output(context, seq * d)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            samples_capacity: samples,
            sequence_capacity: seq,
            input: output(context, samples)?,
            conv1: output(context, d * seq1)?,
            conv1_norm: output(context, d * seq1)?,
            conv2: output(context, 2 * d * seq2)?,
            conv3: output(context, d * seq)?,
            hidden: output(context, seq * d)?,
            normed: output(context, seq * d)?,
            q: output(context, seq * d)?,
            k: output(context, seq * d)?,
            v: output(context, seq * d)?,
            mix: output(context, seq * d)?,
            scores: output(context, seq * seq)?,
            ffn: output(context, seq * mid)?,
            tmp: output(context, seq * d)?,
            encoder_out: output(context, seq * d)?,
            decoder_hidden: output(context, d)?,
            decoder_next: output(context, d)?,
            decoder_normed: output(context, d)?,
            decoder_qkv: output(context, 3 * d)?,
            decoder_attention: output(context, d)?,
            decoder_cross_q: output(context, d)?,
            decoder_cross_mix: output(context, d)?,
            decoder_ffn: output(context, 2 * mid)?,
            decoder_gate: output(context, mid)?,
            logits: output(context, config.vocab_size)?,
            caches,
        })
    }
}

pub struct MoonshineMetalEngine {
    context: RefCell<MetalContext>,
    config: MoonshineConfig,
    tokenizer: Tokenizer,
    weights: HashMap<String, MetalBuffer>,
    zero_bias: MetalBuffer,
    decode_cos: MetalBuffer,
    decode_sin: MetalBuffer,
    scratch: Option<Work>,
}

impl MoonshineMetalEngine {
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let config_text = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|error| error.to_string())?;
        let config_value: serde_json::Value =
            serde_json::from_str(&config_text).map_err(|error| error.to_string())?;
        let config =
            MoonshineConfig::from_json(&config_value).map_err(|error| error.to_string())?;
        // This first Metal profile uses the Tiny checkpoint's equal Q/KV
        // heads. Larger or grouped-query variants stay on the CPU reference.
        if config.hidden_size != 288
            || config.intermediate_size != 1152
            || config.encoder_heads != config.encoder_kv_heads
            || config.decoder_heads != config.decoder_kv_heads
        {
            return Err("Moonshine Metal currently supports the Tiny equal-head profile".into());
        }
        let tokenizer = Tokenizer::load(&model_dir.join("tokenizer.json"))
            .map_err(|error| error.to_string())?;
        let file = SafetensorsFile::open(&model_dir.join("model.safetensors"))
            .map_err(|error| error.to_string())?;
        let context = MetalContext::new().map_err(|error| error.to_string())?;
        let mut weights = HashMap::new();
        let d = config.hidden_size;
        let mid = config.intermediate_size;
        let mut expected: Vec<(String, Vec<usize>)> = vec![
            ("encoder.conv1.weight".into(), vec![d, 1, 127]),
            ("encoder.groupnorm.weight".into(), vec![d]),
            ("encoder.groupnorm.bias".into(), vec![d]),
            ("encoder.conv2.weight".into(), vec![2 * d, d, 7]),
            ("encoder.conv2.bias".into(), vec![2 * d]),
            ("encoder.conv3.weight".into(), vec![d, 2 * d, 3]),
            ("encoder.conv3.bias".into(), vec![d]),
            ("encoder.layer_norm.weight".into(), vec![d]),
            (
                "decoder.embed_tokens.weight".into(),
                vec![config.vocab_size, d],
            ),
            ("decoder.norm.weight".into(), vec![d]),
        ];
        for layer in 0..config.encoder_layers {
            let base = format!("encoder.layers.{layer}");
            for projection in ["q", "k", "v", "o"] {
                expected.push((
                    format!("{base}.self_attn.{projection}_proj.weight"),
                    vec![d, d],
                ));
            }
            expected.extend([
                (format!("{base}.input_layernorm.weight"), vec![d]),
                (format!("{base}.post_attention_layernorm.weight"), vec![d]),
                (format!("{base}.mlp.fc1.weight"), vec![mid, d]),
                (format!("{base}.mlp.fc1.bias"), vec![mid]),
                (format!("{base}.mlp.fc2.weight"), vec![d, mid]),
                (format!("{base}.mlp.fc2.bias"), vec![d]),
            ]);
        }
        for layer in 0..config.decoder_layers {
            let base = format!("decoder.layers.{layer}");
            for projection in ["q", "k", "v", "o"] {
                expected.push((
                    format!("{base}.self_attn.{projection}_proj.weight"),
                    vec![d, d],
                ));
                expected.push((
                    format!("{base}.encoder_attn.{projection}_proj.weight"),
                    vec![d, d],
                ));
            }
            expected.extend([
                (format!("{base}.input_layernorm.weight"), vec![d]),
                (format!("{base}.post_attention_layernorm.weight"), vec![d]),
                (format!("{base}.final_layernorm.weight"), vec![d]),
                (format!("{base}.mlp.fc1.weight"), vec![2 * mid, d]),
                (format!("{base}.mlp.fc1.bias"), vec![2 * mid]),
                (format!("{base}.mlp.fc2.weight"), vec![d, mid]),
                (format!("{base}.mlp.fc2.bias"), vec![d]),
            ]);
        }
        if file.contains_tensor("model.proj_out.weight") || file.contains_tensor("proj_out.weight")
        {
            expected.push(("proj_out.weight".into(), vec![config.vocab_size, d]));
        }
        for (key, shape) in expected {
            let full = if file.contains_tensor(&format!("model.{key}")) {
                format!("model.{key}")
            } else {
                key.clone()
            };
            let descriptor = file
                .descriptor(&full)
                .ok_or_else(|| format!("Moonshine checkpoint missing {full}"))?;
            if descriptor.shape != shape {
                return Err(format!(
                    "Moonshine {full} has shape {:?}, expected {shape:?}",
                    descriptor.shape
                ));
            }
            if (0..config.decoder_layers).any(|layer| {
                ["q", "k", "v"].iter().any(|projection| {
                    key == format!("decoder.layers.{layer}.self_attn.{projection}_proj.weight")
                })
            }) {
                continue;
            }
            let values = file.load_as_f32(&full).map_err(|error| error.to_string())?;
            weights.insert(key, context.new_buffer_with_data(&values));
        }
        for layer in 0..config.decoder_layers {
            let mut packed = Vec::with_capacity(3 * config.hidden_size * config.hidden_size);
            for projection in ["q", "k", "v"] {
                let key = format!("decoder.layers.{layer}.self_attn.{projection}_proj.weight");
                let full = if file.contains_tensor(&format!("model.{key}")) {
                    format!("model.{key}")
                } else {
                    key
                };
                let values = file.load_as_f32(&full).map_err(|error| error.to_string())?;
                packed.extend_from_slice(&values);
            }
            if packed.len() != 3 * config.hidden_size * config.hidden_size {
                return Err(format!(
                    "Moonshine decoder layer {layer} has wrong QKV shape"
                ));
            }
            weights.insert(
                format!("decoder.layers.{layer}.self_attn.qkv_packed"),
                context.new_buffer_with_data(&packed),
            );
        }
        let rotary = Self::rotary_width(&config);
        let (cos, sin) = audio::ops::rope_tables(MAX_DECODE_TOKENS, rotary, config.rope_theta);
        let zero_bias = context.new_buffer_with_data(&vec![0.0f32; config.hidden_size]);
        let decode_cos = context.new_buffer_with_data(&cos);
        let decode_sin = context.new_buffer_with_data(&sin);
        Ok(Self {
            context: RefCell::new(context),
            config,
            tokenizer,
            weights,
            zero_bias,
            decode_cos,
            decode_sin,
            scratch: None,
        })
    }

    fn rotary_width(config: &MoonshineConfig) -> usize {
        let head_dim = config.hidden_size / config.encoder_heads;
        let width = (head_dim as f32 * config.partial_rotary_factor) as usize;
        width - width % 2
    }

    fn weight(&self, name: &str) -> Result<&MetalBuffer, String> {
        self.weights
            .get(name)
            .ok_or_else(|| format!("Moonshine checkpoint missing {name}"))
    }

    fn matmul(
        &self,
        pass: &PassEncoder,
        input: F32View,
        weight: &str,
        bias: Option<&str>,
        output: F32View,
        rows: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<(), String> {
        let weight = self.weight(weight)?;
        let bias = bias.map(|name| self.weight(name)).transpose()?;
        gpu_result(encode_whisper_matmul_bias(
            &mut self.context.borrow_mut(),
            pass,
            input,
            view(weight),
            bias.map(view),
            output,
            as_u32(rows)?,
            as_u32(input_width)?,
            as_u32(output_width)?,
            as_u32(input_width)?,
            as_u32(input_width)?,
            as_u32(output_width)?,
            false,
            1.0,
        ))
    }

    fn norm(
        &self,
        pass: &PassEncoder,
        input: F32View,
        weight: &str,
        output: F32View,
        rows: usize,
    ) -> Result<(), String> {
        gpu_result(encode_whisper_layer_norm(
            &mut self.context.borrow_mut(),
            pass,
            input,
            view(self.weight(weight)?),
            view(&self.zero_bias),
            output,
            as_u32(rows)?,
            as_u32(self.config.hidden_size)?,
            1e-5,
        ))
    }

    fn add(
        &self,
        pass: &PassEncoder,
        src: F32View,
        dst: F32View,
        count: usize,
    ) -> Result<(), String> {
        gpu_result(encode_whisper_add(
            &mut self.context.borrow_mut(),
            pass,
            src,
            dst,
            dst,
            as_u32(count)?,
        ))
    }

    fn encode_frontend(
        &self,
        pass: &PassEncoder,
        work: &Work,
        samples: usize,
        seq: usize,
    ) -> Result<(), String> {
        let d = self.config.hidden_size;
        let seq1 = (samples - 127) / 64 + 1;
        let seq2 = (seq1 - 7) / 3 + 1;
        gpu_result(encode_moonshine_conv1d(
            &mut self.context.borrow_mut(),
            pass,
            view(&work.input),
            view(self.weight("encoder.conv1.weight")?),
            None,
            view(&work.conv1),
            as_u32(samples)?,
            1,
            as_u32(d)?,
            127,
            64,
        ))?;
        gpu_result(encode_moonshine_tanh(
            &mut self.context.borrow_mut(),
            pass,
            view(&work.conv1),
            view(&work.conv1),
            as_u32(d * seq1)?,
        ))?;
        gpu_result(encode_moonshine_groupnorm(
            &mut self.context.borrow_mut(),
            pass,
            view(&work.conv1),
            view(self.weight("encoder.groupnorm.weight")?),
            view(self.weight("encoder.groupnorm.bias")?),
            view(&work.conv1_norm),
            as_u32(d)?,
            as_u32(seq1)?,
            1e-5,
        ))?;
        gpu_result(encode_moonshine_conv1d(
            &mut self.context.borrow_mut(),
            pass,
            view(&work.conv1_norm),
            view(self.weight("encoder.conv2.weight")?),
            Some(view(self.weight("encoder.conv2.bias")?)),
            view(&work.conv2),
            as_u32(seq1)?,
            as_u32(d)?,
            as_u32(2 * d)?,
            7,
            3,
        ))?;
        gpu_result(encode_whisper_gelu_erf(
            &mut self.context.borrow_mut(),
            pass,
            view(&work.conv2),
            view(&work.conv2),
            as_u32(2 * d * seq2)?,
        ))?;
        gpu_result(encode_moonshine_conv1d(
            &mut self.context.borrow_mut(),
            pass,
            view(&work.conv2),
            view(self.weight("encoder.conv3.weight")?),
            Some(view(self.weight("encoder.conv3.bias")?)),
            view(&work.conv3),
            as_u32(seq2)?,
            as_u32(2 * d)?,
            as_u32(d)?,
            3,
            2,
        ))?;
        gpu_result(encode_whisper_gelu_erf(
            &mut self.context.borrow_mut(),
            pass,
            view(&work.conv3),
            view(&work.conv3),
            as_u32(d * seq)?,
        ))?;
        gpu_result(encode_whisper_transpose(
            &mut self.context.borrow_mut(),
            pass,
            view(&work.conv3),
            view(&work.hidden),
            as_u32(d)?,
            as_u32(seq)?,
        ))
    }

    fn gemv(
        &self,
        pass: &PassEncoder,
        input: F32View,
        weight: &str,
        bias: Option<&str>,
        residual: Option<F32View>,
        out: F32View,
        input_width: usize,
        output_width: usize,
    ) -> Result<(), String> {
        gpu_result(encode_whisper_gemv(
            &mut self.context.borrow_mut(),
            pass,
            input,
            view(self.weight(weight)?),
            bias.map(|name| self.weight(name).map(view)).transpose()?,
            residual,
            out,
            as_u32(input_width)?,
            as_u32(output_width)?,
            as_u32(input_width)?,
            false,
            1.0,
        ))
    }

    fn rope(
        &self,
        pass: &PassEncoder,
        values: F32View,
        cos: &MetalBuffer,
        sin: &MetalBuffer,
        rows: usize,
        heads: usize,
        offset: usize,
    ) -> Result<(), String> {
        gpu_result(encode_moonshine_rope(
            &mut self.context.borrow_mut(),
            pass,
            values,
            view(cos),
            view(sin),
            as_u32(rows)?,
            as_u32(heads)?,
            as_u32(self.config.hidden_size / heads)?,
            as_u32(Self::rotary_width(&self.config))?,
            as_u32(offset)?,
        ))
    }

    fn encode_transformer(
        &self,
        pass: &PassEncoder,
        work: &Work,
        seq: usize,
        cos: &MetalBuffer,
        sin: &MetalBuffer,
    ) -> Result<(), String> {
        let d = self.config.hidden_size;
        let mid = self.config.intermediate_size;
        let heads = self.config.encoder_heads;
        let hd = d / heads;
        for layer in 0..self.config.encoder_layers {
            let base = format!("encoder.layers.{layer}");
            self.norm(
                pass,
                view(&work.hidden),
                &format!("{base}.input_layernorm.weight"),
                view(&work.normed),
                seq,
            )?;
            for (projection, dst) in [("q", &work.q), ("k", &work.k), ("v", &work.v)] {
                self.matmul(
                    pass,
                    view(&work.normed),
                    &format!("{base}.self_attn.{projection}_proj.weight"),
                    None,
                    view(dst),
                    seq,
                    d,
                    d,
                )?;
            }
            self.rope(pass, view(&work.q), cos, sin, seq, heads, 0)?;
            self.rope(pass, view(&work.k), cos, sin, seq, heads, 0)?;
            for head in 0..heads {
                let off = (head * hd) as u64;
                gpu_result(encode_whisper_matmul_bias(
                    &mut self.context.borrow_mut(),
                    pass,
                    F32View::at(&work.q, off),
                    F32View::at(&work.k, off),
                    None,
                    view(&work.scores),
                    as_u32(seq)?,
                    as_u32(hd)?,
                    as_u32(seq)?,
                    as_u32(d)?,
                    as_u32(d)?,
                    as_u32(seq)?,
                    false,
                    (hd as f32).powf(-0.5),
                ))?;
                gpu_result(encode_whisper_softmax_rows(
                    &mut self.context.borrow_mut(),
                    pass,
                    view(&work.scores),
                    view(&work.scores),
                    as_u32(seq)?,
                    as_u32(seq)?,
                ))?;
                gpu_result(encode_whisper_matmul_bias(
                    &mut self.context.borrow_mut(),
                    pass,
                    view(&work.scores),
                    F32View::at(&work.v, off),
                    None,
                    F32View::at(&work.mix, off),
                    as_u32(seq)?,
                    as_u32(seq)?,
                    as_u32(hd)?,
                    as_u32(seq)?,
                    as_u32(d)?,
                    as_u32(d)?,
                    true,
                    1.0,
                ))?;
            }
            self.matmul(
                pass,
                view(&work.mix),
                &format!("{base}.self_attn.o_proj.weight"),
                None,
                view(&work.tmp),
                seq,
                d,
                d,
            )?;
            self.add(pass, view(&work.tmp), view(&work.hidden), seq * d)?;
            self.norm(
                pass,
                view(&work.hidden),
                &format!("{base}.post_attention_layernorm.weight"),
                view(&work.normed),
                seq,
            )?;
            self.matmul(
                pass,
                view(&work.normed),
                &format!("{base}.mlp.fc1.weight"),
                Some(&format!("{base}.mlp.fc1.bias")),
                view(&work.ffn),
                seq,
                d,
                mid,
            )?;
            gpu_result(encode_whisper_gelu_erf(
                &mut self.context.borrow_mut(),
                pass,
                view(&work.ffn),
                view(&work.ffn),
                as_u32(seq * mid)?,
            ))?;
            self.matmul(
                pass,
                view(&work.ffn),
                &format!("{base}.mlp.fc2.weight"),
                Some(&format!("{base}.mlp.fc2.bias")),
                view(&work.tmp),
                seq,
                mid,
                d,
            )?;
            self.add(pass, view(&work.tmp), view(&work.hidden), seq * d)?;
        }
        self.norm(
            pass,
            view(&work.hidden),
            "encoder.layer_norm.weight",
            view(&work.encoder_out),
            seq,
        )
    }

    fn build_cross_caches(
        &self,
        pass: &PassEncoder,
        work: &Work,
        seq: usize,
    ) -> Result<(), String> {
        let d = self.config.hidden_size;
        for (layer, cache) in work.caches.iter().enumerate() {
            let base = format!("decoder.layers.{layer}.encoder_attn");
            self.matmul(
                pass,
                view(&work.encoder_out),
                &format!("{base}.k_proj.weight"),
                None,
                view(&cache.cross_k),
                seq,
                d,
                d,
            )?;
            self.matmul(
                pass,
                view(&work.encoder_out),
                &format!("{base}.v_proj.weight"),
                None,
                view(&cache.cross_v),
                seq,
                d,
                d,
            )?;
            gpu_result(encode_whisper_transpose(
                &mut self.context.borrow_mut(),
                pass,
                view(&cache.cross_v),
                view(&cache.cross_v_transposed),
                as_u32(seq)?,
                as_u32(d)?,
            ))?;
        }
        Ok(())
    }

    fn decode_step(
        &self,
        pass: &PassEncoder,
        work: &mut Work,
        token: u32,
        pos: usize,
        seq: usize,
    ) -> Result<(), String> {
        let d = self.config.hidden_size;
        let mid = self.config.intermediate_size;
        let hd = d / self.config.decoder_heads;
        gpu_result(encode_moonshine_embed(
            &mut self.context.borrow_mut(),
            pass,
            view(self.weight("decoder.embed_tokens.weight")?),
            view(&work.decoder_hidden),
            token,
            as_u32(d)?,
        ))?;
        for (layer, cache) in work.caches.iter().enumerate() {
            let base = format!("decoder.layers.{layer}");
            self.norm(
                pass,
                view(&work.decoder_hidden),
                &format!("{base}.input_layernorm.weight"),
                view(&work.decoder_normed),
                1,
            )?;
            self.gemv(
                pass,
                view(&work.decoder_normed),
                &format!("{base}.self_attn.qkv_packed"),
                None,
                None,
                view(&work.decoder_qkv),
                d,
                3 * d,
            )?;
            self.rope(
                pass,
                view(&work.decoder_qkv),
                &self.decode_cos,
                &self.decode_sin,
                1,
                self.config.decoder_heads,
                pos,
            )?;
            self.rope(
                pass,
                F32View::at(&work.decoder_qkv, d as u64),
                &self.decode_cos,
                &self.decode_sin,
                1,
                self.config.decoder_heads,
                pos,
            )?;
            gpu_result(encode_whisper_attn_step(
                &mut self.context.borrow_mut(),
                pass,
                view(&work.decoder_qkv),
                F32View::at(&work.decoder_qkv, d as u64),
                F32View::at(&work.decoder_qkv, (2 * d) as u64),
                view(&cache.self_k),
                view(&cache.self_v),
                view(&work.decoder_attention),
                as_u32(pos)?,
                as_u32(pos + 1)?,
                as_u32(d)?,
                as_u32(hd)?,
                true,
                (hd as f32).powf(-0.5),
                false,
            ))?;
            self.gemv(
                pass,
                view(&work.decoder_attention),
                &format!("{base}.self_attn.o_proj.weight"),
                None,
                Some(view(&work.decoder_hidden)),
                view(&work.decoder_next),
                d,
                d,
            )?;
            std::mem::swap(&mut work.decoder_hidden, &mut work.decoder_next);

            self.norm(
                pass,
                view(&work.decoder_hidden),
                &format!("{base}.post_attention_layernorm.weight"),
                view(&work.decoder_normed),
                1,
            )?;
            self.gemv(
                pass,
                view(&work.decoder_normed),
                &format!("{base}.encoder_attn.q_proj.weight"),
                None,
                None,
                view(&work.decoder_cross_q),
                d,
                d,
            )?;
            gpu_result(encode_whisper_attn_step(
                &mut self.context.borrow_mut(),
                pass,
                view(&work.decoder_cross_q),
                view(&work.decoder_cross_q),
                view(&work.decoder_cross_q),
                view(&cache.cross_k),
                view(&cache.cross_v_transposed),
                view(&work.decoder_cross_mix),
                0,
                as_u32(seq)?,
                as_u32(d)?,
                as_u32(hd)?,
                false,
                (hd as f32).powf(-0.5),
                true,
            ))?;
            self.gemv(
                pass,
                view(&work.decoder_cross_mix),
                &format!("{base}.encoder_attn.o_proj.weight"),
                None,
                Some(view(&work.decoder_hidden)),
                view(&work.decoder_next),
                d,
                d,
            )?;
            std::mem::swap(&mut work.decoder_hidden, &mut work.decoder_next);

            self.norm(
                pass,
                view(&work.decoder_hidden),
                &format!("{base}.final_layernorm.weight"),
                view(&work.decoder_normed),
                1,
            )?;
            self.gemv(
                pass,
                view(&work.decoder_normed),
                &format!("{base}.mlp.fc1.weight"),
                Some(&format!("{base}.mlp.fc1.bias")),
                None,
                view(&work.decoder_ffn),
                d,
                2 * mid,
            )?;
            gpu_result(encode_moonshine_swiglu(
                &mut self.context.borrow_mut(),
                pass,
                view(&work.decoder_ffn),
                view(&work.decoder_gate),
                as_u32(mid)?,
            ))?;
            self.gemv(
                pass,
                view(&work.decoder_gate),
                &format!("{base}.mlp.fc2.weight"),
                Some(&format!("{base}.mlp.fc2.bias")),
                Some(view(&work.decoder_hidden)),
                view(&work.decoder_next),
                mid,
                d,
            )?;
            std::mem::swap(&mut work.decoder_hidden, &mut work.decoder_next);
        }
        self.norm(
            pass,
            view(&work.decoder_hidden),
            "decoder.norm.weight",
            view(&work.decoder_normed),
            1,
        )?;
        let head = if self.weights.contains_key("proj_out.weight") {
            "proj_out.weight"
        } else {
            "decoder.embed_tokens.weight"
        };
        self.gemv(
            pass,
            view(&work.decoder_normed),
            head,
            None,
            None,
            view(&work.logits),
            d,
            self.config.vocab_size,
        )
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, String> {
        if samples.len() < 895 || samples.iter().any(|value| !value.is_finite()) {
            return Err("Moonshine requires at least 895 finite 16 kHz samples".into());
        }
        let seq1 = (samples.len() - 127) / 64 + 1;
        let seq2 = (seq1 - 7) / 3 + 1;
        let seq = (seq2 - 3) / 2 + 1;
        if seq > MAX_ATTN_STEP || samples.len() > u32::MAX as usize {
            return Err("Moonshine Metal clip exceeds attention limit".into());
        }
        let mut work = match self.scratch.take() {
            Some(work)
                if work.samples_capacity >= samples.len() && work.sequence_capacity >= seq =>
            {
                work
            }
            _ => Work::new(&self.context.borrow(), &self.config, samples.len(), seq)?,
        };
        let result = (|| {
            write_buffer_bytes(&work.input, 0, &bytes(samples));
            let (cos, sin) = audio::ops::rope_tables(
                seq,
                Self::rotary_width(&self.config),
                self.config.rope_theta,
            );
            let (cos, sin) = {
                let context = self.context.borrow();
                (
                    context.new_buffer_with_data(&cos),
                    context.new_buffer_with_data(&sin),
                )
            };
            let profile = std::env::var("TURBOSPARK_MOONSHINE_PROFILE").as_deref() == Ok("1");
            let mut frontend_ms = 0.0;
            let mut encoder_ms = 0.0;
            let mut cross_ms = 0.0;
            if profile {
                let started = Instant::now();
                autorelease_pool(|| -> Result<(), String> {
                    let pass = self.context.borrow_mut().begin_pass();
                    self.encode_frontend(&pass, &work, samples.len(), seq)?;
                    pass.commit_and_wait();
                    Ok(())
                })?;
                frontend_ms = started.elapsed().as_secs_f64() * 1_000.0;
                let started = Instant::now();
                autorelease_pool(|| -> Result<(), String> {
                    let pass = self.context.borrow_mut().begin_pass();
                    self.encode_transformer(&pass, &work, seq, &cos, &sin)?;
                    pass.commit_and_wait();
                    Ok(())
                })?;
                encoder_ms = started.elapsed().as_secs_f64() * 1_000.0;
                let started = Instant::now();
                autorelease_pool(|| -> Result<(), String> {
                    let pass = self.context.borrow_mut().begin_pass();
                    self.build_cross_caches(&pass, &work, seq)?;
                    pass.commit_and_wait();
                    Ok(())
                })?;
                cross_ms = started.elapsed().as_secs_f64() * 1_000.0;
            } else {
                autorelease_pool(|| -> Result<(), String> {
                    let pass = self.context.borrow_mut().begin_pass();
                    self.encode_frontend(&pass, &work, samples.len(), seq)?;
                    self.encode_transformer(&pass, &work, seq, &cos, &sin)?;
                    self.build_cross_caches(&pass, &work, seq)?;
                    pass.commit_and_wait();
                    Ok(())
                })?;
            }
            let decode_started = Instant::now();
            let mut token = self.config.decoder_start_token_id;
            let mut generated = Vec::new();
            for pos in 0..MAX_DECODE_TOKENS {
                autorelease_pool(|| -> Result<(), String> {
                    let pass = self.context.borrow_mut().begin_pass();
                    self.decode_step(&pass, &mut work, token, pos, seq)?;
                    pass.commit_and_wait();
                    Ok(())
                })?;
                let logits = read_f32_buffer(&work.logits, self.config.vocab_size);
                let best = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .ok_or_else(|| "Moonshine Metal produced no logits".to_string())?
                    .0 as u32;
                if best == self.config.eos_token_id {
                    break;
                }
                token = best;
                generated.push(token);
            }
            if profile {
                let stats = self.context.borrow().compilation_stats();
                eprintln!(
                    "moonshine_metal frontend_ms={frontend_ms:.2} encoder_ms={encoder_ms:.2} cross_ms={cross_ms:.2} decode_ms={:.2} tokens={} libraries_loaded={} libraries_compiled={} pipelines_created={} buffers_allocated={}",
                    decode_started.elapsed().as_secs_f64() * 1_000.0,
                    generated.len(),
                    stats.library_loads,
                    stats.library_compiles,
                    stats.pipeline_creations,
                    stats.buffer_allocations,
                );
            }
            Ok(self.tokenizer.decode(&generated))
        })();
        self.scratch = Some(work);
        result
    }
}

/// Device choice is made at open. An unsupported GPU profile uses the
/// portable CPU reference without keeping duplicate weights resident.
pub enum MoonshineBackend {
    Cpu(Moonshine),
    Metal(Mutex<MoonshineMetalEngine>),
}

pub struct MoonshineRunner {
    pub backend: MoonshineBackend,
}

#[derive(Debug, PartialEq, Eq)]
enum DeviceChoice {
    Cpu,
    Metal,
}

fn device_choice(value: Option<&str>) -> Result<DeviceChoice, String> {
    match value {
        None | Some("cpu") => Ok(DeviceChoice::Cpu),
        Some("metal") => Ok(DeviceChoice::Metal),
        Some(other) => Err(format!(
            "unsupported TURBOSPARK_MOONSHINE_DEVICE={other:?}; expected cpu or metal"
        )),
    }
}

impl MoonshineRunner {
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let selected = std::env::var("TURBOSPARK_MOONSHINE_DEVICE");
        let choice = match selected.as_deref() {
            Ok(value) => device_choice(Some(value)),
            Err(std::env::VarError::NotPresent) => device_choice(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("TURBOSPARK_MOONSHINE_DEVICE must be UTF-8".into())
            }
        }?;
        let backend = match choice {
            DeviceChoice::Cpu => MoonshineBackend::Cpu(
                Moonshine::open(model_dir).map_err(|error| error.to_string())?,
            ),
            DeviceChoice::Metal => {
                MoonshineBackend::Metal(Mutex::new(MoonshineMetalEngine::open(model_dir)?))
            }
        };
        Ok(Self { backend })
    }

    pub fn using_metal(&self) -> bool {
        matches!(self.backend, MoonshineBackend::Metal(_))
    }

    pub fn transcribe(&self, samples: &[f32]) -> Result<String, String> {
        match &self.backend {
            MoonshineBackend::Cpu(model) => {
                model.transcribe(samples).map_err(|error| error.to_string())
            }
            MoonshineBackend::Metal(engine) => {
                let mut engine = engine
                    .lock()
                    .map_err(|_| "Moonshine engine lock poisoned".to_string())?;
                engine.transcribe(samples)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn backend_choice_refuses_unknown_value() {
        assert_eq!(device_choice(None), Ok(DeviceChoice::Cpu));
        assert_eq!(device_choice(Some("cpu")), Ok(DeviceChoice::Cpu));
        assert_eq!(device_choice(Some("metal")), Ok(DeviceChoice::Metal));
        assert!(device_choice(Some("metla")).is_err());
    }

    /// Run with TURBOSPARK_MOONSHINE_TEST_MODEL and TEST_WAV set to a pinned
    /// Tiny checkpoint and a known 16 kHz WAV. This loads both backends only
    /// in the test process so production never retains duplicate weights.
    #[test]
    #[ignore = "requires a pinned Moonshine Tiny checkpoint and a Metal device"]
    fn pinned_checkpoint_encoder_and_transcript_parity() {
        let model_dir = PathBuf::from(std::env::var("TURBOSPARK_MOONSHINE_TEST_MODEL").unwrap());
        let wav = PathBuf::from(std::env::var("TURBOSPARK_MOONSHINE_TEST_WAV").unwrap());
        let input = audio::read_wav_f32(&wav).unwrap();
        let samples = audio::to_mono_resampled(
            &input,
            16_000,
            &audio::MonoResampleStrategy::SincHann(Default::default()),
        )
        .unwrap();
        let cpu = Moonshine::open(&model_dir).unwrap();
        let gpu = MoonshineMetalEngine::open(&model_dir).unwrap();
        let expected = cpu.encode(&samples).unwrap();
        let seq = expected.len() / gpu.config.hidden_size;
        let work = Work::new(&gpu.context.borrow(), &gpu.config, samples.len(), seq).unwrap();
        write_buffer_bytes(&work.input, 0, &bytes(&samples));
        let (cos, sin) = audio::ops::rope_tables(
            seq,
            MoonshineMetalEngine::rotary_width(&gpu.config),
            gpu.config.rope_theta,
        );
        let (cos, sin) = {
            let context = gpu.context.borrow();
            (
                context.new_buffer_with_data(&cos),
                context.new_buffer_with_data(&sin),
            )
        };
        let pass = gpu.context.borrow_mut().begin_pass();
        gpu.encode_frontend(&pass, &work, samples.len(), seq)
            .unwrap();
        gpu.encode_transformer(&pass, &work, seq, &cos, &sin)
            .unwrap();
        pass.commit_and_wait();
        let actual = read_f32_buffer(&work.encoder_out, expected.len());
        let max_abs = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Moonshine encoder max_abs={max_abs}");
        assert!(
            max_abs < 1e-3,
            "Moonshine encoder diverged: max_abs={max_abs}"
        );
        let mut gpu = gpu;
        let cpu_text = cpu.transcribe(&samples).unwrap();
        let gpu_text = gpu.transcribe(&samples).unwrap();
        assert_eq!(gpu_text, cpu_text);
        let shorter = &samples[..samples.len() / 2];
        let cpu_short = cpu.transcribe(shorter).unwrap();
        let gpu_short = gpu.transcribe(shorter).unwrap();
        assert_eq!(
            gpu_short, cpu_short,
            "reused scratch changed the transcript"
        );
    }
}
