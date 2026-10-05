//! Resident packed 8-bit Metal decoder for Qwen3-ASR.
//!
//! The mel frontend and the audio transformer stay on the portable CPU path
//! in `turbospark-audio` (the audio tower is plain BF16 and unquantized);
//! this module uploads the text decoder in its checkpoint-packed form, U32
//! affine-INT8 words plus BF16 scales and biases in groups of 64, and
//! dequantizes inside every GEMV. Prompt rows prefill one row at a time into
//! growing per-layer K/V pairs, the same incremental-cache structure the CPU
//! reference uses, and every generated token runs one command pass.
//!
//! Opt-in only. The kernel-parity, synthetic full-layer parity, pinned
//! transcript, and interleaved CPU/Metal latency and footprint gates
//! recorded in `crates/audio/src/stt/qwen3_asr/README.md` all pass, and the
//! CPU path stays the default until the still-open quality evaluation and
//! default-on decision say otherwise.

use std::cell::RefCell;
use std::path::Path;

use audio::stt::qwen3_asr::encoder::AudioEncoder;
use audio::stt::qwen3_asr::frontend::compute_features;
use audio::stt::qwen3_asr::Qwen3Config;
use audio::stt::qwen3_asr::{load_tokenizer, prompt_token_ids, transcript_from_tokens};
use gpu::{
    autorelease_pool, encode_attention_decode, encode_dequant_int8_gemv_resident,
    encode_residual_add, encode_rms_norm_bf16w, encode_rms_norm_bf16w_perhead,
    encode_rope_neox_subdim, encode_silu_mul, half_slice_to_le_bytes, read_half_buffer,
    write_buffer_bytes, AttentionScratch, Int8ResidentMatrix, MetalBuffer, MetalContext,
    PassEncoder,
};
use half::{bf16, f16};
use model_io::safetensors::SafetensorsFile;
use tokenizer::Tokenizer;

const GROUP_SIZE: usize = 64;

/// One affine-INT8 linear resident as `[weight bytes | BF16 scales | BF16
/// biases]` in a single buffer, addressed by offset inside every GEMV.
struct PackedLinear {
    buffer: MetalBuffer,
    rows: usize,
    cols: usize,
}

impl PackedLinear {
    fn resident(&self) -> Int8ResidentMatrix<'_> {
        let companion = self.rows * (self.cols / GROUP_SIZE) * 2;
        Int8ResidentMatrix {
            buffer: &self.buffer,
            weights_offset: 0,
            scales_offset: (self.rows * self.cols) as u64,
            biases_offset: (self.rows * self.cols + companion) as u64,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

struct DecoderLayerWeights {
    input_norm: MetalBuffer,
    post_attention_norm: MetalBuffer,
    q_norm: MetalBuffer,
    k_norm: MetalBuffer,
    q_proj: PackedLinear,
    k_proj: PackedLinear,
    v_proj: PackedLinear,
    o_proj: PackedLinear,
    gate_proj: PackedLinear,
    up_proj: PackedLinear,
    down_proj: PackedLinear,
}

struct LayerKv {
    k: MetalBuffer,
    v: MetalBuffer,
}

/// Request-sized scratch: one residual stream, per-layer K/V at the
/// request's row capacity, and the decode-attention split-softmax scratch.
struct Work {
    capacity: usize,
    x: MetalBuffer,
    normed: MetalBuffer,
    q: MetalBuffer,
    attn: MetalBuffer,
    delta: MetalBuffer,
    gate: MetalBuffer,
    up: MetalBuffer,
    logits: MetalBuffer,
    caches: Vec<LayerKv>,
    attention: AttentionScratch,
}

/// Host copy of the tied embedding table for per-token row dequantization;
/// the same packed bytes stay resident on the GPU for the LM-head GEMV.
struct EmbedTable {
    packed: PackedLinear,
    weight: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
}

/// Resident packed 8-bit Metal decoder for one Qwen3-ASR checkpoint.
pub struct Qwen3AsrMetalEngine {
    context: RefCell<MetalContext>,
    config: Qwen3Config,
    tokenizer: Tokenizer,
    encoder: AudioEncoder,
    layers: Vec<DecoderLayerWeights>,
    final_norm: MetalBuffer,
    embed: EmbedTable,
    scratch: RefCell<Option<Work>>,
}

fn resolve_base(file: &SafetensorsFile, base: &str) -> String {
    for candidate in [
        base.to_owned(),
        format!("thinker.{base}"),
        format!("llm.{base}"),
    ] {
        if file.contains_tensor(&format!("{candidate}.weight")) {
            return candidate;
        }
    }
    base.to_owned()
}

fn raw(file: &SafetensorsFile, base: &str, suffix: &str) -> Result<Vec<u8>, String> {
    file.raw_bytes(&format!("{base}.{suffix}"))
        .map(|bytes| bytes.to_vec())
        .map_err(|error| error.to_string())
}

fn bf16_norm(
    file: &SafetensorsFile,
    base: &str,
    context: &MetalContext,
) -> Result<MetalBuffer, String> {
    let name = format!("{}.weight", resolve_base(file, base));
    let desc = file
        .descriptor(&name)
        .ok_or_else(|| format!("Qwen3-ASR Metal missing {name}"))?;
    if desc.dtype != "BF16" {
        return Err(format!(
            "Qwen3-ASR Metal norm {name} is {}, expected BF16",
            desc.dtype
        ));
    }
    Ok(context.new_buffer_with_data(file.raw_bytes(&name).map_err(|error| error.to_string())?))
}

fn load_packed_linear(
    file: &SafetensorsFile,
    base: &str,
    rows: usize,
    cols: usize,
    context: &MetalContext,
) -> Result<PackedLinear, String> {
    let base = resolve_base(file, base);
    let weight_name = format!("{base}.weight");
    let desc = file
        .descriptor(&weight_name)
        .ok_or_else(|| format!("Qwen3-ASR Metal missing {weight_name}"))?;
    if desc.dtype != "U32" || desc.shape != [rows, cols / 4] {
        return Err(format!(
            "Qwen3-ASR Metal {weight_name} is {:?} {:?}, expected U32 [{rows}, {}]",
            desc.dtype,
            desc.shape,
            cols / 4
        ));
    }
    if file.contains_tensor(&format!("{base}.bias")) {
        return Err(format!(
            "Qwen3-ASR Metal refuses biased linear {base}: the packed GEMV has no bias input"
        ));
    }
    let scales_name = format!("{base}.scales");
    let biases_name = format!("{base}.biases");
    for name in [&scales_name, &biases_name] {
        let desc = file
            .descriptor(name)
            .ok_or_else(|| format!("Qwen3-ASR Metal missing {name}"))?;
        if desc.dtype != "BF16" || desc.shape != [rows, cols / GROUP_SIZE] {
            return Err(format!(
                "Qwen3-ASR Metal {name} is {:?} {:?}, expected BF16 [{rows}, {}]",
                desc.dtype,
                desc.shape,
                cols / GROUP_SIZE
            ));
        }
    }
    let weight = file
        .raw_bytes(&weight_name)
        .map_err(|error| error.to_string())?;
    let scales = file
        .raw_bytes(&scales_name)
        .map_err(|error| error.to_string())?;
    let biases = file
        .raw_bytes(&biases_name)
        .map_err(|error| error.to_string())?;
    if weight.len() != rows * cols {
        return Err(format!(
            "Qwen3-ASR Metal {weight_name} carries {} bytes for {rows}x{cols}",
            weight.len()
        ));
    }
    let companion_bytes = rows * (cols / GROUP_SIZE) * 2;
    if scales.len() != companion_bytes || biases.len() != companion_bytes {
        return Err(format!(
            "Qwen3-ASR Metal {base} scales and biases must each carry {companion_bytes} bytes"
        ));
    }
    let mut blob = weight.to_vec();
    blob.extend_from_slice(scales);
    blob.extend_from_slice(biases);
    Ok(PackedLinear {
        buffer: context.new_buffer_with_data(&blob),
        rows,
        cols,
    })
}

impl Qwen3AsrMetalEngine {
    /// Opens one local checkpoint directory at the pinned profile layout:
    /// an unquantized BF16 audio tower the CPU encoder reads, and a grouped
    /// 8-bit affine text decoder this module keeps packed on the GPU.
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let config_json: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("config.json"))
                .map_err(|error| format!("config.json: {error}"))?,
        )
        .map_err(|error| format!("config.json: {error}"))?;
        let config = Qwen3Config::from_json(&config_json).map_err(|error| error.to_string())?;
        if !config.text.tie_word_embeddings {
            return Err("Qwen3-ASR Metal requires tied token embeddings".into());
        }
        if config.quant_bits != 8 || config.quant_group_size != GROUP_SIZE {
            return Err(format!(
                "Qwen3-ASR Metal supports 8-bit groups of {GROUP_SIZE}, checkpoint says {}-bit groups of {}",
                config.quant_bits, config.quant_group_size
            ));
        }
        let tokenizer = load_tokenizer(model_dir).map_err(|error| error.to_string())?;
        let file = SafetensorsFile::open(&model_dir.join("model.safetensors"))
            .map_err(|error| error.to_string())?;
        let context = MetalContext::new().map_err(|error| error.to_string())?;
        let encoder =
            AudioEncoder::load(&file, &config.audio).map_err(|error| error.to_string())?;

        let text = &config.text;
        let hidden = text.hidden_size;
        let q_width = text.num_attention_heads * text.head_dim;
        let kv_width = text.num_key_value_heads * text.head_dim;
        let layers = (0..text.num_hidden_layers)
            .map(|index| {
                let prefix = format!("model.layers.{index}");
                let attn = format!("{prefix}.self_attn");
                let mlp = format!("{prefix}.mlp");
                Ok(DecoderLayerWeights {
                    input_norm: bf16_norm(&file, &format!("{prefix}.input_layernorm"), &context)?,
                    post_attention_norm: bf16_norm(
                        &file,
                        &format!("{prefix}.post_attention_layernorm"),
                        &context,
                    )?,
                    q_norm: bf16_norm(&file, &format!("{attn}.q_norm"), &context)?,
                    k_norm: bf16_norm(&file, &format!("{attn}.k_norm"), &context)?,
                    q_proj: load_packed_linear(
                        &file,
                        &format!("{attn}.q_proj"),
                        q_width,
                        hidden,
                        &context,
                    )?,
                    k_proj: load_packed_linear(
                        &file,
                        &format!("{attn}.k_proj"),
                        kv_width,
                        hidden,
                        &context,
                    )?,
                    v_proj: load_packed_linear(
                        &file,
                        &format!("{attn}.v_proj"),
                        kv_width,
                        hidden,
                        &context,
                    )?,
                    o_proj: load_packed_linear(
                        &file,
                        &format!("{attn}.o_proj"),
                        hidden,
                        q_width,
                        &context,
                    )?,
                    gate_proj: load_packed_linear(
                        &file,
                        &format!("{mlp}.gate_proj"),
                        text.intermediate_size,
                        hidden,
                        &context,
                    )?,
                    up_proj: load_packed_linear(
                        &file,
                        &format!("{mlp}.up_proj"),
                        text.intermediate_size,
                        hidden,
                        &context,
                    )?,
                    down_proj: load_packed_linear(
                        &file,
                        &format!("{mlp}.down_proj"),
                        hidden,
                        text.intermediate_size,
                        &context,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let final_norm = bf16_norm(&file, "model.norm", &context)?;
        let embed_base = resolve_base(&file, "model.embed_tokens");
        let embed = EmbedTable {
            packed: load_packed_linear(
                &file,
                "model.embed_tokens",
                text.vocab_size,
                hidden,
                &context,
            )?,
            weight: raw(&file, &embed_base, "weight")?,
            scales: raw(&file, &embed_base, "scales")?,
            biases: raw(&file, &embed_base, "biases")?,
        };
        Ok(Self {
            context: RefCell::new(context),
            config,
            tokenizer,
            encoder,
            layers,
            final_norm,
            embed,
            scratch: RefCell::new(None),
        })
    }

    /// Dequantizes one tied-embedding row on the host: the GEMV kernels take
    /// a dense activation vector, and a one-hot GEMV over 151936 rows would
    /// be the wrong shape for a lookup.
    fn embed_row_bytes(&self, token_id: u32) -> Vec<u8> {
        let hidden = self.config.text.hidden_size;
        let groups = hidden / GROUP_SIZE;
        let row = token_id as usize;
        let mut out = vec![0u8; hidden * 2];
        for (index, chunk) in out.chunks_mut(2).enumerate() {
            let q = self.embed.weight[row * hidden + index] as f32;
            let group = row * groups + index / GROUP_SIZE;
            let scale = bf16::from_bits(u16::from_le_bytes([
                self.embed.scales[group * 2],
                self.embed.scales[group * 2 + 1],
            ]))
            .to_f32();
            let bias = bf16::from_bits(u16::from_le_bytes([
                self.embed.biases[group * 2],
                self.embed.biases[group * 2 + 1],
            ]))
            .to_f32();
            chunk.copy_from_slice(f16::from_f32(q * scale + bias).to_le_bytes().as_slice());
        }
        out
    }

    /// Allocates or reuses scratch sized for at least `capacity` rows.
    fn ensure_work(&self, capacity: usize) -> Result<(), String> {
        let mut slot = self.scratch.borrow_mut();
        if slot.as_ref().is_some_and(|work| work.capacity >= capacity) {
            return Ok(());
        }
        let context = self.context.borrow();
        let text = &self.config.text;
        let hidden = text.hidden_size;
        let q_width = text.num_attention_heads * text.head_dim;
        let kv_width = text.num_key_value_heads * text.head_dim;
        let word = |count: usize| -> Result<MetalBuffer, String> {
            Ok(context.new_output_buffer((count * 2) as u64))
        };
        let caches = (0..text.num_hidden_layers)
            .map(|_| {
                Ok(LayerKv {
                    k: word(capacity * kv_width)?,
                    v: word(capacity * kv_width)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        *slot = Some(Work {
            capacity,
            x: word(hidden)?,
            normed: word(hidden)?,
            q: word(q_width)?,
            attn: word(q_width)?,
            delta: word(hidden)?,
            gate: word(text.intermediate_size)?,
            up: word(text.intermediate_size)?,
            logits: word(text.vocab_size)?,
            caches,
            attention: AttentionScratch::new(
                &context,
                text.num_attention_heads as u32,
                text.head_dim as u32,
            ),
        });
        Ok(())
    }

    fn gemv(
        &self,
        context: &mut MetalContext,
        pass: &PassEncoder,
        linear: &PackedLinear,
        x: (&MetalBuffer, u64),
        y: (&MetalBuffer, u64),
    ) -> Result<(), String> {
        encode_dequant_int8_gemv_resident(context, pass, &linear.resident(), x, y)
            .map_err(|error| error.to_string())
    }

    /// Runs one row through every layer at `position`, appending its rotated
    /// K/V, and optionally returns the next-token logits.
    fn step(
        &self,
        row_bytes: &[u8],
        position: usize,
        want_logits: bool,
    ) -> Result<Option<Vec<f16>>, String> {
        let mut slot = self.scratch.borrow_mut();
        let work = slot.as_mut().expect("ensure_work ran before step");
        let text = &self.config.text;
        let hidden = text.hidden_size;
        let heads = text.num_attention_heads;
        let kv_heads = text.num_key_value_heads;
        let head_dim = text.head_dim;
        let kv_width = kv_heads * head_dim;
        let inter = text.intermediate_size;
        let eps = text.rms_norm_eps;
        let theta = text.rope_theta;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let context = &mut self.context.borrow_mut();
        let x = &work.x;
        write_buffer_bytes(x, 0, row_bytes);
        let kv_row = (position * kv_width * 2) as u64;
        autorelease_pool(|| -> Result<(), String> {
            let pass = context.begin_pass();
            for (layer, cache) in self.layers.iter().zip(&work.caches) {
                encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (x, 0),
                    (&layer.input_norm, 0),
                    (&work.normed, 0),
                    hidden as u32,
                    eps,
                )
                .map_err(|error| error.to_string())?;
                self.gemv(
                    context,
                    &pass,
                    &layer.q_proj,
                    (&work.normed, 0),
                    (&work.q, 0),
                )?;
                self.gemv(
                    context,
                    &pass,
                    &layer.k_proj,
                    (&work.normed, 0),
                    (&cache.k, kv_row),
                )?;
                self.gemv(
                    context,
                    &pass,
                    &layer.v_proj,
                    (&work.normed, 0),
                    (&cache.v, kv_row),
                )?;
                encode_rms_norm_bf16w_perhead(
                    context,
                    &pass,
                    (&work.q, 0),
                    (&layer.q_norm, 0),
                    (&work.q, 0),
                    heads as u32,
                    head_dim as u32,
                    eps,
                )
                .map_err(|error| error.to_string())?;
                encode_rms_norm_bf16w_perhead(
                    context,
                    &pass,
                    (&cache.k, kv_row),
                    (&layer.k_norm, 0),
                    (&cache.k, kv_row),
                    kv_heads as u32,
                    head_dim as u32,
                    eps,
                )
                .map_err(|error| error.to_string())?;
                encode_rope_neox_subdim(
                    context,
                    &pass,
                    (&work.q, 0),
                    position as u32,
                    heads as u32,
                    head_dim as u32,
                    head_dim as u32,
                    theta,
                )
                .map_err(|error| error.to_string())?;
                encode_rope_neox_subdim(
                    context,
                    &pass,
                    (&cache.k, kv_row),
                    position as u32,
                    kv_heads as u32,
                    head_dim as u32,
                    head_dim as u32,
                    theta,
                )
                .map_err(|error| error.to_string())?;
                encode_attention_decode(
                    context,
                    &pass,
                    (&work.q, 0),
                    &cache.k,
                    &cache.v,
                    &work.attention,
                    (&work.attn, 0),
                    head_dim as u32,
                    heads as u32,
                    kv_heads as u32,
                    (position + 1) as u32,
                    // kv_start is the attended window start, not the write
                    // row: Qwen3-ASR has no sliding window, so every row
                    // attends the full causal prefix [0, position].
                    0,
                    0,
                    scale,
                    None,
                )
                .map_err(|error| error.to_string())?;
                self.gemv(
                    context,
                    &pass,
                    &layer.o_proj,
                    (&work.attn, 0),
                    (&work.delta, 0),
                )?;
                encode_residual_add(context, &pass, (x, 0), (&work.delta, 0), hidden as u32)
                    .map_err(|error| error.to_string())?;
                encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (x, 0),
                    (&layer.post_attention_norm, 0),
                    (&work.normed, 0),
                    hidden as u32,
                    eps,
                )
                .map_err(|error| error.to_string())?;
                self.gemv(
                    context,
                    &pass,
                    &layer.gate_proj,
                    (&work.normed, 0),
                    (&work.gate, 0),
                )?;
                self.gemv(
                    context,
                    &pass,
                    &layer.up_proj,
                    (&work.normed, 0),
                    (&work.up, 0),
                )?;
                encode_silu_mul(
                    context,
                    &pass,
                    (&work.gate, 0),
                    (&work.up, 0),
                    (&work.gate, 0),
                    inter as u32,
                )
                .map_err(|error| error.to_string())?;
                self.gemv(
                    context,
                    &pass,
                    &layer.down_proj,
                    (&work.gate, 0),
                    (&work.delta, 0),
                )?;
                encode_residual_add(context, &pass, (x, 0), (&work.delta, 0), hidden as u32)
                    .map_err(|error| error.to_string())?;
            }
            if want_logits {
                encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (x, 0),
                    (&self.final_norm, 0),
                    (&work.normed, 0),
                    hidden as u32,
                    eps,
                )
                .map_err(|error| error.to_string())?;
                self.gemv(
                    context,
                    &pass,
                    &self.embed.packed,
                    (&work.normed, 0),
                    (&work.logits, 0),
                )?;
            }
            pass.commit_and_wait();
            Ok(())
        })?;
        Ok(if want_logits {
            Some(read_half_buffer(&work.logits, self.config.text.vocab_size))
        } else {
            None
        })
    }

    fn argmax(logits: &[f16]) -> u32 {
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (id, &value)| {
                let value = value.to_f32();
                if value > best.1 {
                    (id, value)
                } else {
                    best
                }
            })
            .0 as u32
    }

    /// Transcribes mono 16 kHz PCM with automatic language detection.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, String> {
        self.transcribe_with_options(samples, None, 512)
    }

    /// Transcribes one mono 16 kHz clip, optionally prompting a supported
    /// output language. `max_tokens` bounds greedy decoding.
    pub fn transcribe_with_options(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        max_tokens: usize,
    ) -> Result<String, String> {
        if max_tokens == 0 {
            return Err("Qwen3-ASR max_tokens must be positive".into());
        }
        let language = language
            .map(|requested| {
                self.config
                    .supported_languages
                    .iter()
                    .find(|known| known.eq_ignore_ascii_case(requested))
                    .map(String::as_str)
                    .ok_or_else(|| format!("unsupported Qwen3-ASR language: {requested}"))
            })
            .transpose()?;
        // TURBOSPARK_QWEN3_ASR_PROFILE=1 prints the same per-phase line the
        // CPU path prints; the clock reads are negligible next to the phases
        // they bracket.
        let profile = std::env::var("TURBOSPARK_QWEN3_ASR_PROFILE").as_deref() == Ok("1");
        let started = std::time::Instant::now();
        let features = compute_features(samples).map_err(|error| error.to_string())?;
        let features_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let started = std::time::Instant::now();
        let audio_embeddings = self
            .encoder
            .forward(&features)
            .map_err(|error| error.to_string())?;
        let encode_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let hidden = self.config.text.hidden_size;
        if audio_embeddings.len() % hidden != 0 {
            return Err("Qwen3 audio encoder output is not whole decoder rows".into());
        }
        let audio_rows = audio_embeddings.len() / hidden;
        let (token_ids, audio_positions) =
            prompt_token_ids(&self.config, &self.tokenizer, audio_rows, language)
                .map_err(|error| error.to_string())?;

        let prompt_rows = token_ids.len();
        self.ensure_work(prompt_rows + max_tokens)?;
        let stop_ids = ["<|im_end|>", "<|endoftext|>"]
            .into_iter()
            .filter_map(|token| self.tokenizer.token_to_id(token))
            .collect::<Vec<_>>();

        let mut generated: Vec<u32> = Vec::new();
        let mut next: Option<u32> = None;
        let mut prefill_started: Option<std::time::Instant> = None;
        for (position, &token_id) in token_ids.iter().enumerate() {
            let row = if token_id == self.config.audio_token_id {
                let frame = audio_positions
                    .iter()
                    .position(|&placeholder| placeholder == position)
                    .ok_or("Qwen3 prompt audio placeholder lost its frame index")?;
                half_slice_to_le_bytes(
                    &audio_embeddings[frame * hidden..(frame + 1) * hidden]
                        .iter()
                        .map(|&value| f16::from_f32(value))
                        .collect::<Vec<_>>(),
                )
            } else {
                self.embed_row_bytes(token_id as u32)
            };
            let last = position + 1 == prompt_rows;
            if last {
                prefill_started.get_or_insert(std::time::Instant::now());
                next = Some(Self::argmax(
                    &self
                        .step(&row, position, true)?
                        .expect("final prefill row carries logits"),
                ));
            } else {
                prefill_started.get_or_insert(std::time::Instant::now());
                self.step(&row, position, false)?;
            }
        }
        let prefill_ms = prefill_started
            .map(|started| started.elapsed().as_secs_f64() * 1_000.0)
            .unwrap_or(0.0);
        let decode_started = std::time::Instant::now();
        for _ in 0..max_tokens {
            let token = next.take().ok_or("Qwen3-ASR Metal lost the greedy token")?;
            if stop_ids.contains(&token)
                || (stop_ids.is_empty() && matches!(token, 151_645 | 151_643))
            {
                break;
            }
            generated.push(token);
            if generated.len() == max_tokens {
                break;
            }
            let position = prompt_rows + generated.len() - 1;
            let row = self.embed_row_bytes(token);
            next = Some(Self::argmax(
                &self
                    .step(&row, position, true)?
                    .expect("decode step carries logits"),
            ));
        }
        if profile {
            eprintln!(
                "qwen3_asr_metal features_ms={features_ms:.2} encode_ms={encode_ms:.2} \
                 prefill_ms={prefill_ms:.2} decode_ms={:.2} audio_rows={audio_rows} tokens={}",
                decode_started.elapsed().as_secs_f64() * 1_000.0,
                generated.len()
            );
        }
        transcript_from_tokens(&self.tokenizer, &generated, language)
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen3AsrMetalEngine;
    use super::GROUP_SIZE;
    use half::bf16;
    use std::path::PathBuf;

    const PINNED_TRANSCRIPT: &str = "The quick brown fox jumps over the lazy dog.";

    fn pinned_model_dir() -> PathBuf {
        PathBuf::from(
            std::env::var("TURBOSPARK_QWEN3_ASR_DIR")
                .expect("set TURBOSPARK_QWEN3_ASR_DIR to the pinned checkpoint directory"),
        )
    }

    fn reference_wav() -> Vec<f32> {
        let path = std::env::var("TURBOSPARK_QWEN3_ASR_WAV")
            .expect("set TURBOSPARK_QWEN3_ASR_WAV to the reference speech clip");
        let audio = audio::read_wav_f32(std::path::Path::new(&path)).unwrap();
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        audio.samples
    }

    /// The packed Metal decoder must reproduce the CPU reference transcript
    /// end to end on the real checkpoint: this is the full-stack gate over
    /// the packed GEMVs, per-head norms, RoPE, GQA decode attention, and the
    /// incremental K/V path. Timing prints support the interleaved latency
    /// runs; a debug build number is not a performance measurement.
    #[test]
    #[ignore = "requires the pinned Qwen3-ASR checkpoint, the reference WAV, and a Metal device"]
    fn pinned_checkpoint_transcript_matches_cpu_reference() {
        let model_dir = pinned_model_dir();
        let samples = reference_wav();

        let started = std::time::Instant::now();
        let mut engine = Qwen3AsrMetalEngine::open(&model_dir).unwrap();
        eprintln!(
            "qwen3_asr_metal open_ms={:.2}",
            started.elapsed().as_secs_f64() * 1_000.0
        );
        let started = std::time::Instant::now();
        let metal = engine.transcribe_with_options(&samples, None, 32).unwrap();
        eprintln!(
            "qwen3_asr_metal transcribe_ms={:.2} text={metal:?}",
            started.elapsed().as_secs_f64() * 1_000.0
        );

        let reference = audio::stt::qwen3_asr::Qwen3Asr::load(&model_dir).unwrap();
        let cpu = reference
            .transcribe_with_options(&samples, None, 32)
            .unwrap();
        eprintln!("qwen3_asr_metal cpu_text={cpu:?}");
        assert_eq!(metal, PINNED_TRANSCRIPT);
        assert_eq!(metal, cpu);
    }

    /// One greedy step on a one-second prefix: the cheapest end-to-end check
    /// that prefill, the first logits, and the detokenization seam agree
    /// with the CPU reference before the full transcript runs.
    #[test]
    #[ignore = "requires the pinned Qwen3-ASR checkpoint, the reference WAV, and a Metal device"]
    fn pinned_checkpoint_one_step_matches_cpu_reference() {
        let model_dir = pinned_model_dir();
        let samples = reference_wav();
        let mut engine = Qwen3AsrMetalEngine::open(&model_dir).unwrap();
        let metal = engine
            .transcribe_with_options(&samples[..16_000], None, 1)
            .unwrap();
        eprintln!("qwen3_asr_metal one_token={metal:?}");
        assert!(!metal.is_empty(), "checkpoint stopped before emitting text");
        let reference = audio::stt::qwen3_asr::Qwen3Asr::load(&model_dir).unwrap();
        let cpu = reference
            .transcribe_with_options(&samples[..16_000], None, 1)
            .unwrap();
        assert_eq!(metal, cpu);
    }

    /// Metal-only transcript for the interleaved benchmark: unlike the
    /// parity tests this never loads the f32 CPU reference model, so a
    /// `/usr/bin/time -l` peak over this process is the serving-shape
    /// footprint of the packed Metal path plus the CPU audio encoder.
    #[test]
    #[ignore = "requires the pinned Qwen3-ASR checkpoint, the reference WAV, and a Metal device"]
    fn pinned_checkpoint_transcribes_metal_only() {
        let model_dir = pinned_model_dir();
        let samples = reference_wav();
        let started = std::time::Instant::now();
        let mut engine = Qwen3AsrMetalEngine::open(&model_dir).unwrap();
        eprintln!(
            "qwen3_asr_metal open_ms={:.2}",
            started.elapsed().as_secs_f64() * 1_000.0
        );
        let started = std::time::Instant::now();
        let text = engine.transcribe_with_options(&samples, None, 32).unwrap();
        eprintln!(
            "qwen3_asr_metal transcribe_ms={:.2}",
            started.elapsed().as_secs_f64() * 1_000.0
        );
        assert_eq!(text, PINNED_TRANSCRIPT);
    }

    // ---- synthetic full-layer parity gate ----
    //
    // The pinned-checkpoint tests above prove the packed decoder against one
    // real weight set; this gate generates every weight byte here, so a
    // layout, shape, or cache regression cannot hide behind weights that
    // happen to work. Both engines load the same synthetic on-disk
    // checkpoint and greedy decode the same clip, and the generated text can
    // only match if the packed GEMV layout, per-head norms, RoPE, GQA
    // attention, incremental K/V, and prompt assembly agree step for step.

    /// Deterministic byte/uniform generator; the checkpoint must be
    /// byte-identical across runs so failures are reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn byte(&mut self) -> u8 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u8
        }

        fn unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / 16_777_216.0
        }
    }

    struct SyntheticTensor {
        dtype: &'static str,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl SyntheticTensor {
        fn f32(shape: &[usize], values: &[f32]) -> Self {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Self {
                dtype: "F32",
                shape: shape.to_vec(),
                bytes,
            }
        }

        fn bf16(shape: &[usize], bits: &[u16]) -> Self {
            let mut bytes = Vec::with_capacity(bits.len() * 2);
            for bits in bits {
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
            Self {
                dtype: "BF16",
                shape: shape.to_vec(),
                bytes,
            }
        }
    }

    fn random_f32(
        out: &mut Vec<(String, SyntheticTensor)>,
        rng: &mut Lcg,
        name: impl Into<String>,
        shape: &[usize],
        spread: f32,
    ) {
        let count: usize = shape.iter().product();
        let values: Vec<f32> = (0..count).map(|_| (rng.unit() - 0.5) * spread).collect();
        out.push((name.into(), SyntheticTensor::f32(shape, &values)));
    }

    fn constant_f32(
        out: &mut Vec<(String, SyntheticTensor)>,
        name: impl Into<String>,
        shape: &[usize],
        value: f32,
    ) {
        let count: usize = shape.iter().product();
        out.push((
            name.into(),
            SyntheticTensor::f32(shape, &vec![value; count]),
        ));
    }

    fn constant_bf16(
        out: &mut Vec<(String, SyntheticTensor)>,
        name: impl Into<String>,
        shape: &[usize],
        value: f32,
    ) {
        let count: usize = shape.iter().product();
        out.push((
            name.into(),
            SyntheticTensor::bf16(shape, &vec![bf16::from_f32(value).to_bits(); count]),
        ));
    }

    /// One affine-INT8 linear in exactly the pinned checkpoint layout: U32
    /// packed bytes `[rows, cols / 4]` plus BF16 scales and biases
    /// `[rows, cols / 64]`. The byte range times the companion constants
    /// keeps quantized weights near [-0.51, 0.51] so activations stay well
    /// inside f16 while remaining decisive for greedy argmax.
    fn packed_linear(
        out: &mut Vec<(String, SyntheticTensor)>,
        rng: &mut Lcg,
        base: &str,
        rows: usize,
        cols: usize,
    ) {
        let mut weight = vec![0u8; rows * cols];
        for byte in weight.iter_mut() {
            *byte = rng.byte();
        }
        let groups = rows * (cols / GROUP_SIZE);
        out.push((
            format!("{base}.weight"),
            SyntheticTensor {
                dtype: "U32",
                shape: vec![rows, cols / 4],
                bytes: weight,
            },
        ));
        out.push((
            format!("{base}.scales"),
            SyntheticTensor::bf16(
                &[rows, cols / GROUP_SIZE],
                &vec![bf16::from_f32(0.004).to_bits(); groups],
            ),
        ));
        out.push((
            format!("{base}.biases"),
            SyntheticTensor::bf16(
                &[rows, cols / GROUP_SIZE],
                &vec![bf16::from_f32(-0.508).to_bits(); groups],
            ),
        ));
    }

    fn synthetic_config_json() -> String {
        serde_json::json!({
            "model_type": "qwen3_asr",
            "audio_token_id": 259,
            "audio_start_token_id": 258,
            "audio_end_token_id": 260,
            "support_languages": [],
            "audio_config": {
                "num_mel_bins": 128,
                "encoder_layers": 1,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 64,
                "d_model": 64,
                "max_source_positions": 64,
                "n_window": 2,
                "n_window_infer": 4,
                "downsample_hidden_size": 8,
                "output_dim": 64,
                "scale_embedding": false,
                "activation_function": "gelu"
            },
            "text_config": {
                "vocab_size": 261,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": true,
                "attention_bias": false,
                "hidden_act": "silu"
            },
            "quantization_config": {"bits": 8, "group_size": 64, "mode": "affine"}
        })
        .to_string()
    }

    /// Byte-level BPE over the full GPT-2 byte alphabet with no merges and
    /// five added special tokens, so the prompt encodes deterministically
    /// and every byte piece resolves below the added ids.
    fn synthetic_tokenizer_json() -> String {
        let mut vocab = serde_json::Map::new();
        let mut offset = 0u32;
        for byte in 0u32..256 {
            let printable = (33..=126).contains(&byte)
                || (161..=172).contains(&byte)
                || (174..=255).contains(&byte);
            let mapped = if printable {
                byte
            } else {
                let mapped = 256 + offset;
                offset += 1;
                mapped
            };
            vocab.insert(
                char::from_u32(mapped).unwrap().to_string(),
                serde_json::json!(byte),
            );
        }
        let added = [
            ("<|im_start|>", 256u32),
            ("<|im_end|>", 257),
            ("<|audio_start|>", 258),
            ("<|audio_pad|>", 259),
            ("<|audio_end|>", 260),
        ]
        .map(|(content, id)| {
            serde_json::json!({
                "id": id,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        });
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": vocab,
                "merges": []
            }
        })
        .to_string()
    }

    fn write_safetensors(path: &std::path::Path, tensors: &[(String, SyntheticTensor)]) {
        let mut header = serde_json::Map::new();
        let mut blob = Vec::<u8>::new();
        for (name, tensor) in tensors {
            let start = blob.len() as u64;
            blob.extend_from_slice(&tensor.bytes);
            while blob.len() % 8 != 0 {
                blob.push(0);
            }
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": [start, start + tensor.bytes.len() as u64],
                }),
            );
        }
        let header_json = serde_json::Value::Object(header).to_string();
        let mut file = (header_json.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header_json.as_bytes());
        file.extend_from_slice(&blob);
        std::fs::write(path, file).unwrap();
    }

    fn write_synthetic_checkpoint(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), synthetic_tokenizer_json()).unwrap();
        std::fs::write(dir.join("config.json"), synthetic_config_json()).unwrap();

        let mut rng = Lcg(0x5EED_1234_ABCD_0001);
        let mut tensors: Vec<(String, SyntheticTensor)> = Vec::new();
        // The portable CPU audio tower feeds both engines, so plain F32
        // random weights keep the fixture small without weakening the
        // decoder parity this gate exists for.
        const WIDTH: usize = 8;
        const D_MODEL: usize = 64;
        const CONV_FREQ: usize = 16; // three stride-2 convs turn 128 bands into 16
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.conv2d1.weight",
            &[WIDTH, 1, 3, 3],
            0.2,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.conv2d1.bias",
            &[WIDTH],
            0.05,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.conv2d2.weight",
            &[WIDTH, WIDTH, 3, 3],
            0.2,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.conv2d2.bias",
            &[WIDTH],
            0.05,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.conv2d3.weight",
            &[WIDTH, WIDTH, 3, 3],
            0.2,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.conv2d3.bias",
            &[WIDTH],
            0.05,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.conv_out.weight",
            &[D_MODEL, WIDTH * CONV_FREQ],
            0.2,
        );
        let attention = "audio_tower.layers.0.self_attn";
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            random_f32(
                &mut tensors,
                &mut rng,
                format!("{attention}.{projection}.weight"),
                &[D_MODEL, D_MODEL],
                0.2,
            );
            random_f32(
                &mut tensors,
                &mut rng,
                format!("{attention}.{projection}.bias"),
                &[D_MODEL],
                0.05,
            );
        }
        constant_f32(
            &mut tensors,
            "audio_tower.layers.0.self_attn_layer_norm.weight",
            &[D_MODEL],
            1.0,
        );
        constant_f32(
            &mut tensors,
            "audio_tower.layers.0.self_attn_layer_norm.bias",
            &[D_MODEL],
            0.0,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.layers.0.fc1.weight",
            &[64, D_MODEL],
            0.2,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.layers.0.fc1.bias",
            &[64],
            0.05,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.layers.0.fc2.weight",
            &[D_MODEL, 64],
            0.2,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.layers.0.fc2.bias",
            &[D_MODEL],
            0.05,
        );
        constant_f32(
            &mut tensors,
            "audio_tower.layers.0.final_layer_norm.weight",
            &[D_MODEL],
            1.0,
        );
        constant_f32(
            &mut tensors,
            "audio_tower.layers.0.final_layer_norm.bias",
            &[D_MODEL],
            0.0,
        );
        constant_f32(&mut tensors, "audio_tower.ln_post.weight", &[D_MODEL], 1.0);
        constant_f32(&mut tensors, "audio_tower.ln_post.bias", &[D_MODEL], 0.0);
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.proj1.weight",
            &[D_MODEL, D_MODEL],
            0.2,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.proj1.bias",
            &[D_MODEL],
            0.05,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.proj2.weight",
            &[D_MODEL, D_MODEL],
            0.2,
        );
        random_f32(
            &mut tensors,
            &mut rng,
            "audio_tower.proj2.bias",
            &[D_MODEL],
            0.05,
        );

        const HIDDEN: usize = 64;
        const INTER: usize = 128;
        const HEADS: usize = 2;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 64;
        const VOCAB: usize = 261;
        packed_linear(&mut tensors, &mut rng, "model.embed_tokens", VOCAB, HIDDEN);
        for layer in 0..2usize {
            let prefix = format!("model.layers.{layer}");
            let attn = format!("{prefix}.self_attn");
            let mlp = format!("{prefix}.mlp");
            constant_bf16(
                &mut tensors,
                format!("{prefix}.input_layernorm.weight"),
                &[HIDDEN],
                1.0,
            );
            constant_bf16(
                &mut tensors,
                format!("{prefix}.post_attention_layernorm.weight"),
                &[HIDDEN],
                1.0,
            );
            constant_bf16(
                &mut tensors,
                format!("{attn}.q_norm.weight"),
                &[HEAD_DIM],
                1.0,
            );
            constant_bf16(
                &mut tensors,
                format!("{attn}.k_norm.weight"),
                &[HEAD_DIM],
                1.0,
            );
            packed_linear(
                &mut tensors,
                &mut rng,
                &format!("{attn}.q_proj"),
                HEADS * HEAD_DIM,
                HIDDEN,
            );
            packed_linear(
                &mut tensors,
                &mut rng,
                &format!("{attn}.k_proj"),
                KV_HEADS * HEAD_DIM,
                HIDDEN,
            );
            packed_linear(
                &mut tensors,
                &mut rng,
                &format!("{attn}.v_proj"),
                KV_HEADS * HEAD_DIM,
                HIDDEN,
            );
            packed_linear(
                &mut tensors,
                &mut rng,
                &format!("{attn}.o_proj"),
                HIDDEN,
                HEADS * HEAD_DIM,
            );
            packed_linear(
                &mut tensors,
                &mut rng,
                &format!("{mlp}.gate_proj"),
                INTER,
                HIDDEN,
            );
            packed_linear(
                &mut tensors,
                &mut rng,
                &format!("{mlp}.up_proj"),
                INTER,
                HIDDEN,
            );
            packed_linear(
                &mut tensors,
                &mut rng,
                &format!("{mlp}.down_proj"),
                HIDDEN,
                INTER,
            );
        }
        constant_bf16(&mut tensors, "model.norm.weight", &[HIDDEN], 1.0);
        write_safetensors(&dir.join("model.safetensors"), &tensors);
    }

    #[test]
    fn synthetic_checkpoint_metal_matches_cpu_reference() {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-qwen3-asr-synthetic-{}",
            std::process::id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        write_synthetic_checkpoint(&dir);
        let samples: Vec<f32> = (0..6400)
            .map(|index| {
                let seconds = index as f32 / 16_000.0;
                0.1 * (core::f32::consts::TAU * 440.0 * seconds).sin()
                    + 0.03 * (core::f32::consts::TAU * 997.0 * seconds).sin()
            })
            .collect();
        let cpu = audio::stt::qwen3_asr::Qwen3Asr::load(&dir)
            .unwrap()
            .transcribe_with_options(&samples, None, 8)
            .unwrap();
        let metal = Qwen3AsrMetalEngine::open(&dir)
            .unwrap()
            .transcribe_with_options(&samples, None, 8)
            .unwrap();
        eprintln!("qwen3_asr synthetic cpu_text={cpu:?} metal_text={metal:?}");
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(
            metal, cpu,
            "Metal decode diverged from the CPU reference on synthetic weights"
        );
    }
}
