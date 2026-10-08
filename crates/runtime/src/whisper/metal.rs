//! The whisper Metal execution path: resident f32 weight buffers, the
//! encoder stack, the cross-attention caches, and the incremental decoder,
//! all sequenced through `gpu`'s whisper kernels (`shaders/
//! whisper_encoder.metal` plus the front-end conv).
//!
//! The CPU reference path in `super` (`compute::whisper` kernels via
//! `encode_window` and `WindowDecoder`) is the parity partner: `tests/
//! whisper_encoder_parity.rs` in the gpu crate bounds every kernel, and
//! `whisper::tests` pins same-tokens agreement between the two paths on
//! synthetic fixtures. Numerics are f32 end to end on both sides, matching
//! the reference's own accumulation precision.
//!
//! Device gating: `WhisperRunner::open` builds an engine when a Metal
//! device initializes and `TURBOSPARK_WHISPER_DEVICE` does not force
//! `cpu`; otherwise (or on any engine error at open) the runner keeps the
//! CPU reference path. Transcription runs entirely on one device path per
//! call; the same device produces the same segments for the same PCM.
//!
//! Buffering: every window is zero-padded to the full 30 seconds before
//! the mel, so the encoder sequence length is a constant
//! (`max_source_positions`) and all scratch is allocated once, at open.
//! The largest scratch is the encoder attention score store,
//! `heads * seq * seq` f32 (~54 MB at tiny.en's 6 heads of 384, ~180 MB at
//! large-v3's 20 heads of 1280) -- unified memory, resident for the
//! runner's lifetime.

use gpu::{
    autorelease_pool, encode_whisper_add, encode_whisper_attn_step, encode_whisper_conv1d3_gelu,
    encode_whisper_gelu_erf, encode_whisper_gemv, encode_whisper_layer_norm,
    encode_whisper_matmul_bias, encode_whisper_softmax_rows, encode_whisper_transpose,
    encode_whisper_transpose_pos, read_f32_buffer, write_buffer_bytes, F32View, MetalBuffer,
    MetalContext, PassEncoder,
};
use model_io::whisper_config::{WhisperConfig, WhisperSpecialTokens};

use super::decode::{argmax_suppressed, DecodedWindow, StopReason};
use super::weights::{WhisperDecoderLayerOwned, WhisperEncoderLayerOwned, WhisperWeights};

/// One encoder layer's resident weights. Attention projections carry no k
/// bias (openai convention); q, v, and out carry one.
struct EncLayerGpu {
    q_w: MetalBuffer,
    q_b: MetalBuffer,
    k_w: MetalBuffer,
    v_w: MetalBuffer,
    v_b: MetalBuffer,
    out_w: MetalBuffer,
    out_b: MetalBuffer,
    ln1_w: MetalBuffer,
    ln1_b: MetalBuffer,
    fc1_w: MetalBuffer,
    fc1_b: MetalBuffer,
    fc2_w: MetalBuffer,
    fc2_b: MetalBuffer,
    ln2_w: MetalBuffer,
    ln2_b: MetalBuffer,
}

/// One decoder layer's resident weights plus its two KV caches. The self
/// caches fill incrementally during the prompt warm-up and greedy loop;
/// the cross caches are projected once per encoded window.
struct DecLayerGpu {
    /// Self-attention Q, K, V packed as one `[3d, d]` weight and `[3d]`
    /// bias (rows 0..d are Q, d..2d K, 2d..3d V; the checkpoint's K
    /// projection carries no bias, so its slot is zeros). One GEMV serves
    /// all three projections, which matters in the decode loop where
    /// every dispatch is latency the token pays.
    self_qkv_w: MetalBuffer,
    self_qkv_b: MetalBuffer,
    self_out_w: MetalBuffer,
    self_out_b: MetalBuffer,
    ln_self_w: MetalBuffer,
    ln_self_b: MetalBuffer,
    cross_q_w: MetalBuffer,
    cross_q_b: MetalBuffer,
    cross_k_w: MetalBuffer,
    cross_v_w: MetalBuffer,
    cross_v_b: MetalBuffer,
    cross_out_w: MetalBuffer,
    cross_out_b: MetalBuffer,
    ln_cross_w: MetalBuffer,
    ln_cross_b: MetalBuffer,
    fc1_w: MetalBuffer,
    fc1_b: MetalBuffer,
    fc2_w: MetalBuffer,
    fc2_b: MetalBuffer,
    ln_fc_w: MetalBuffer,
    ln_fc_b: MetalBuffer,
    /// Self-attention K/V cache, `[max_target_positions, d_model]`. Rows
    /// past `self_len` hold stale data by design: the attention kernel
    /// attends over `0..=pos` only, so a reset is a host-side counter,
    /// not a buffer clear.
    self_k_cache: MetalBuffer,
    self_v_cache: MetalBuffer,
    /// Cross-attention K/V cache, `[seq, d_model]`, filled once per window.
    cross_k: MetalBuffer,
    cross_v: MetalBuffer,
    /// The value cache transposed to `[d_model, seq]`: the attention
    /// step's weighted value sum walks one output dim across every key,
    /// contiguous here and strided by d_model in `cross_v` -- measured at
    /// 77% of the whole decode step before this transpose went in.
    cross_v_t: MetalBuffer,
}

/// Resident Metal whisper model: weights uploaded once at open, activation
/// scratch sized for a full 30-second window, and the decoder caches.
pub struct WhisperMetalEngine {
    context: std::cell::RefCell<MetalContext>,
    config: WhisperConfig,
    /// The encoder sequence length: `max_source_positions`, constant
    /// because every window arrives zero-padded to the full 30 seconds.
    seq: usize,

    // Front end and encoder weights.
    conv1_w: MetalBuffer,
    conv1_b: MetalBuffer,
    conv2_w: MetalBuffer,
    conv2_b: MetalBuffer,
    enc_positions: MetalBuffer,
    enc_layers: Vec<EncLayerGpu>,
    enc_ln_w: MetalBuffer,
    enc_ln_b: MetalBuffer,

    // Decoder weights and caches. The embedding is tied with the logits
    // projection, so one resident table serves both.
    embed_tokens: MetalBuffer,
    dec_layers: Vec<DecLayerGpu>,
    dec_ln_w: MetalBuffer,
    dec_ln_b: MetalBuffer,
    /// Filled self-cache rows (host-side counter; see `DecLayerGpu`).
    self_len: usize,

    // Encoder activation scratch, all sized for one full window.
    mel_in: MetalBuffer,
    conv1_out: MetalBuffer,
    conv2_out: MetalBuffer,
    hidden: MetalBuffer,
    normed: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attn_mix: MetalBuffer,
    scores: MetalBuffer,
    ffn: MetalBuffer,
    tmp: MetalBuffer,
    enc_out: MetalBuffer,

    // Decoder scratch (single-token streams) and logits.
    dec_in: MetalBuffer,
    dec_hidden: MetalBuffer,
    dec_normed: MetalBuffer,
    dec_qkv: MetalBuffer,
    dec_attn: MetalBuffer,
    dec_h1: MetalBuffer,
    dec_h2: MetalBuffer,
    dec_tmp: MetalBuffer,
    dec_normed2: MetalBuffer,
    dec_cq: MetalBuffer,
    dec_crossmix: MetalBuffer,
    dec_co: MetalBuffer,
    dec_fc1: MetalBuffer,
    logits: MetalBuffer,
}

fn upload(context: &MetalContext, data: &[f32]) -> MetalBuffer {
    context.new_buffer_with_data(data)
}

fn f32_le_bytes(slice: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(slice));
    for v in slice {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

impl WhisperMetalEngine {
    /// Uploads the weights and allocates the scratch for a full window.
    /// Fails when no Metal device initializes or a shader fails to build;
    /// the caller falls back to the CPU reference path.
    pub fn new(weights: &WhisperWeights, config: &WhisperConfig) -> Result<Self, String> {
        let context = MetalContext::new().map_err(|e| e.to_string())?;
        let d = config.d_model;
        let seq = config.max_source_positions;
        let dec_len = config.max_target_positions;
        let enc_heads = config.num_attention_heads;
        let vocab = config.vocab_size;

        // The kernels walk one threadgroup per head over head_dim-wide
        // slices; a non-integral head geometry must not reach them.
        if d % enc_heads != 0 || d % config.decoder_attention_heads() != 0 {
            return Err(format!(
                "whisper Metal path needs d_model {d} divisible by both head counts"
            ));
        }

        let enc_layers: Vec<EncLayerGpu> = weights
            .enc_layers
            .iter()
            .map(|layer: &WhisperEncoderLayerOwned| EncLayerGpu {
                q_w: upload(&context, &layer.q),
                q_b: upload(&context, &layer.q_bias),
                k_w: upload(&context, &layer.k),
                v_w: upload(&context, &layer.v),
                v_b: upload(&context, &layer.v_bias),
                out_w: upload(&context, &layer.out),
                out_b: upload(&context, &layer.out_bias),
                ln1_w: upload(&context, &layer.ln1_weight),
                ln1_b: upload(&context, &layer.ln1_bias),
                fc1_w: upload(&context, &layer.fc1),
                fc1_b: upload(&context, &layer.fc1_bias),
                fc2_w: upload(&context, &layer.fc2),
                fc2_b: upload(&context, &layer.fc2_bias),
                ln2_w: upload(&context, &layer.ln2_weight),
                ln2_b: upload(&context, &layer.ln2_bias),
            })
            .collect();

        let dec_layers: Vec<DecLayerGpu> = weights
            .dec_layers
            .iter()
            .map(|layer: &WhisperDecoderLayerOwned| {
                let mut qkv_w = Vec::with_capacity(3 * d * d);
                qkv_w.extend_from_slice(&layer.self_q);
                qkv_w.extend_from_slice(&layer.self_k);
                qkv_w.extend_from_slice(&layer.self_v);
                let mut qkv_b = Vec::with_capacity(3 * d);
                qkv_b.extend_from_slice(&layer.self_q_bias);
                qkv_b.resize(2 * d, 0.0);
                qkv_b.extend_from_slice(&layer.self_v_bias);
                DecLayerGpu {
                    self_qkv_w: upload(&context, &qkv_w),
                    self_qkv_b: upload(&context, &qkv_b),
                    self_out_w: upload(&context, &layer.self_out),
                    self_out_b: upload(&context, &layer.self_out_bias),
                    ln_self_w: upload(&context, &layer.ln_self_weight),
                    ln_self_b: upload(&context, &layer.ln_self_bias),
                    cross_q_w: upload(&context, &layer.cross_q),
                    cross_q_b: upload(&context, &layer.cross_q_bias),
                    cross_k_w: upload(&context, &layer.cross_k),
                    cross_v_w: upload(&context, &layer.cross_v),
                    cross_v_b: upload(&context, &layer.cross_v_bias),
                    cross_out_w: upload(&context, &layer.cross_out),
                    cross_out_b: upload(&context, &layer.cross_out_bias),
                    ln_cross_w: upload(&context, &layer.ln_cross_weight),
                    ln_cross_b: upload(&context, &layer.ln_cross_bias),
                    fc1_w: upload(&context, &layer.fc1),
                    fc1_b: upload(&context, &layer.fc1_bias),
                    fc2_w: upload(&context, &layer.fc2),
                    fc2_b: upload(&context, &layer.fc2_bias),
                    ln_fc_w: upload(&context, &layer.ln_fc_weight),
                    ln_fc_b: upload(&context, &layer.ln_fc_bias),
                    self_k_cache: context.new_output_buffer((dec_len * d * 4) as u64),
                    self_v_cache: context.new_output_buffer((dec_len * d * 4) as u64),
                    cross_k: context.new_output_buffer((seq * d * 4) as u64),
                    cross_v: context.new_output_buffer((seq * d * 4) as u64),
                    cross_v_t: context.new_output_buffer((seq * d * 4) as u64),
                }
            })
            .collect();

        let elems = |n: usize| (n * 4) as u64;
        Ok(Self {
            mel_in: context.new_output_buffer(elems(config.n_mels * seq * 2)),
            conv1_out: context.new_output_buffer(elems(d * seq * 2)),
            conv2_out: context.new_output_buffer(elems(d * seq)),
            hidden: context.new_output_buffer(elems(seq * d)),
            normed: context.new_output_buffer(elems(seq * d)),
            q: context.new_output_buffer(elems(seq * d)),
            k: context.new_output_buffer(elems(seq * d)),
            v: context.new_output_buffer(elems(seq * d)),
            attn_mix: context.new_output_buffer(elems(seq * d)),
            scores: context.new_output_buffer(elems(enc_heads * seq * seq)),
            ffn: context.new_output_buffer(elems(seq * config.encoder_ffn_dim)),
            tmp: context.new_output_buffer(elems(seq * d)),
            enc_out: context.new_output_buffer(elems(seq * d)),
            dec_in: context.new_output_buffer(elems(d)),
            dec_hidden: context.new_output_buffer(elems(d)),
            dec_normed: context.new_output_buffer(elems(d)),
            dec_qkv: context.new_output_buffer(elems(3 * d)),
            dec_attn: context.new_output_buffer(elems(d)),
            dec_h1: context.new_output_buffer(elems(d)),
            dec_h2: context.new_output_buffer(elems(d)),
            dec_tmp: context.new_output_buffer(elems(d)),
            dec_normed2: context.new_output_buffer(elems(d)),
            dec_cq: context.new_output_buffer(elems(d)),
            dec_crossmix: context.new_output_buffer(elems(d)),
            dec_co: context.new_output_buffer(elems(d)),
            dec_fc1: context.new_output_buffer(elems(config.decoder_ffn_dim())),
            logits: context.new_output_buffer(elems(vocab)),
            config: config.clone(),
            seq,
            conv1_w: upload(&context, &weights.conv1),
            conv1_b: upload(&context, &weights.conv1_bias),
            conv2_w: upload(&context, &weights.conv2),
            conv2_b: upload(&context, &weights.conv2_bias),
            enc_positions: upload(&context, &weights.enc_positions),
            enc_ln_w: upload(&context, &weights.enc_ln_weight),
            enc_ln_b: upload(&context, &weights.enc_ln_bias),
            embed_tokens: upload(&context, &weights.embed_tokens),
            dec_ln_w: upload(&context, &weights.dec_ln_weight),
            dec_ln_b: upload(&context, &weights.dec_ln_bias),
            dec_layers,
            enc_layers,
            self_len: 0,
            context: std::cell::RefCell::new(context),
        })
    }

    /// The engine's `max_source_positions`; the runner's window loop
    /// already fixed `seq` from the config, and this documents the two
    /// agree.
    pub fn seq(&self) -> usize {
        self.seq
    }

    /// Clears the self-attention caches. Stale rows are never read (the
    /// attention kernel attends over `0..=pos`), so the reset is the
    /// counter alone; each window's decode starts from position 0.
    pub fn begin_decode(&mut self) {
        self.self_len = 0;
    }

    /// Runs the encoder over one band-major log-mel window
    /// `[n_mels * frames]`, leaving the final-normed stream in `enc_out`.
    /// `frames` must be exactly `2 * max_source_positions`: the runner
    /// zero-pads tail windows to the full 30 seconds before the mel, so
    /// this holds for every window it feeds.
    pub fn encode(&mut self, mel_band_major: &[f32], frames: usize) -> Result<(), String> {
        self.encode_partial(mel_band_major, frames, None)
    }

    /// Encoder forward optionally bounded to the first `stop_after`
    /// layers, for parity bisection; `read_hidden` pairs with it.
    pub fn encode_partial(
        &mut self,
        mel_band_major: &[f32],
        frames: usize,
        stop_after: Option<usize>,
    ) -> Result<(), String> {
        let d = self.config.d_model;
        let n_mels = self.config.n_mels;
        let eps = self.config.layer_norm_eps;
        let expected = 2 * self.seq;
        if frames != expected || mel_band_major.len() != n_mels * frames {
            return Err(format!(
                "mel window must carry {n_mels} x {expected} values, got {} per frame x {frames}",
                mel_band_major.len() / frames.max(1)
            ));
        }
        write_buffer_bytes(&self.mel_in, 0, &f32_le_bytes(mel_band_major));
        let heads = self.config.num_attention_heads;
        let head_dim = d / heads;
        let scale = (head_dim as f32).powf(-0.5);
        let d_u = d as u32;
        let seq_u = self.seq as u32;

        autorelease_pool(|| -> Result<(), String> {
            let pass = self.context.borrow().begin_pass_labeled("whisper_encode");
            // Conv front end: mel -> conv1 (stride 1 pad 1) -> conv2
            // (stride 2 pad 1), exact-erf GELU, band-major throughout.
            encode_whisper_conv1d3_gelu(
                &mut self.context.borrow_mut(),
                &pass,
                F32View::new(&self.mel_in),
                F32View::new(&self.conv1_w),
                F32View::new(&self.conv1_b),
                F32View::new(&self.conv1_out),
                n_mels as u32,
                frames as u32,
                d_u,
                1,
                1,
            )
            .map_err(|e| e.to_string())?;
            // Dispatches execute in order but their writes are not
            // automatically visible to the next dispatch: conv2 reads
            // conv1's output and transpose_pos reads conv2's.
            encode_whisper_conv1d3_gelu(
                &mut self.context.borrow_mut(),
                &pass,
                F32View::new(&self.conv1_out),
                F32View::new(&self.conv2_w),
                F32View::new(&self.conv2_b),
                F32View::new(&self.conv2_out),
                d_u,
                frames as u32,
                d_u,
                2,
                1,
            )
            .map_err(|e| e.to_string())?;
            // [d, seq] band-major -> [seq, d] rows plus the positional row.
            encode_whisper_transpose_pos(
                &mut self.context.borrow_mut(),
                &pass,
                F32View::new(&self.conv2_out),
                F32View::new(&self.enc_positions),
                F32View::new(&self.hidden),
                seq_u,
                d_u,
            )
            .map_err(|e| e.to_string())?;

            let layers = match stop_after {
                Some(n) => &self.enc_layers[..n.min(self.enc_layers.len())],
                None => &self.enc_layers[..],
            };
            for layer in layers {
                self.encode_layer(&pass, layer, heads, head_dim, scale, d_u, seq_u, eps)?;
            }

            // When bounded, stop at the layer output (no final norm): the
            // bisection compares per-layer hidden states.
            if stop_after.is_some() {
                pass.commit_and_wait();
                return Ok(());
            }

            // Final encoder layer norm straight into enc_out.
            encode_whisper_layer_norm(
                &mut self.context.borrow_mut(),
                &pass,
                F32View::new(&self.hidden),
                F32View::new(&self.enc_ln_w),
                F32View::new(&self.enc_ln_b),
                F32View::new(&self.enc_out),
                seq_u,
                d_u,
                eps,
            )
            .map_err(|e| e.to_string())?;
            pass.commit_and_wait();
            Ok(())
        })
    }

    /// One Pre-LN encoder block on the caller's pass: Pre-LN attention and
    /// MLP with residual adds. The barriers make each dispatch's writes
    /// visible to its consumers.
    #[allow(clippy::too_many_arguments)]
    fn encode_layer(
        &self,
        pass: &PassEncoder,
        layer: &EncLayerGpu,
        heads: usize,
        head_dim: usize,
        scale: f32,
        d_u: u32,
        seq_u: u32,
        eps: f32,
    ) -> Result<(), String> {
        let ffn_dim = self.config.encoder_ffn_dim as u32;

        // Pre-LN attention.
        encode_whisper_layer_norm(
            &mut self.context.borrow_mut(),
            pass,
            F32View::new(&self.hidden),
            F32View::new(&layer.ln1_w),
            F32View::new(&layer.ln1_b),
            F32View::new(&self.normed),
            seq_u,
            d_u,
            eps,
        )
        .map_err(|e| e.to_string())?;
        let mm = |a: F32View,
                  w: F32View,
                  bias: Option<F32View>,
                  out: F32View,
                  m: u32,
                  k: u32,
                  n: u32,
                  a_stride: u32,
                  w_stride: u32,
                  out_stride: u32,
                  w_t: bool|
         -> Result<(), String> {
            encode_whisper_matmul_bias(
                &mut self.context.borrow_mut(),
                pass,
                a,
                w,
                bias,
                out,
                m,
                k,
                n,
                a_stride,
                w_stride,
                out_stride,
                w_t,
                1.0,
            )
            .map_err(|e| e.to_string())
        };
        mm(
            F32View::new(&self.normed),
            F32View::new(&layer.q_w),
            Some(F32View::new(&layer.q_b)),
            F32View::new(&self.q),
            seq_u,
            d_u,
            d_u,
            d_u,
            d_u,
            d_u,
            false,
        )?;
        mm(
            F32View::new(&self.normed),
            F32View::new(&layer.k_w),
            None,
            F32View::new(&self.k),
            seq_u,
            d_u,
            d_u,
            d_u,
            d_u,
            d_u,
            false,
        )?;
        mm(
            F32View::new(&self.normed),
            F32View::new(&layer.v_w),
            Some(F32View::new(&layer.v_b)),
            F32View::new(&self.v),
            seq_u,
            d_u,
            d_u,
            d_u,
            d_u,
            d_u,
            false,
        )?;

        // Per head: scores = Q_h x K_h^T scaled by head_dim^-0.5 (the
        // reference scales the dot products) into the head's score plane
        // (head_dim-wide reduction over d_model-strided rows), one
        // row-softmax over every head's planes, then attn_h = scores_h x
        // V_h with the transposed-B read (V's rows are keys) written into
        // the head's column slice of the mixed stream.
        let scores_per_head = self.seq * self.seq;
        for h in 0..heads {
            let head_off = (h * head_dim) as u64;
            let plane_off = (h * scores_per_head) as u64;
            encode_whisper_matmul_bias(
                &mut self.context.borrow_mut(),
                pass,
                F32View::at(&self.q, head_off),
                F32View::at(&self.k, head_off),
                None,
                F32View::at(&self.scores, plane_off),
                seq_u,
                head_dim as u32,
                seq_u,
                d_u,
                d_u,
                seq_u,
                false,
                scale,
            )
            .map_err(|e| e.to_string())?;
        }
        encode_whisper_softmax_rows(
            &mut self.context.borrow_mut(),
            pass,
            F32View::new(&self.scores),
            F32View::new(&self.scores),
            (heads * self.seq) as u32,
            seq_u,
        )
        .map_err(|e| e.to_string())?;
        for h in 0..heads {
            let head_off = (h * head_dim) as u64;
            let plane_off = (h * scores_per_head) as u64;
            mm(
                F32View::at(&self.scores, plane_off),
                F32View::at(&self.v, head_off),
                None,
                F32View::at(&self.attn_mix, head_off),
                seq_u,
                seq_u,
                head_dim as u32,
                seq_u,
                d_u,
                d_u,
                true,
            )?;
        }

        mm(
            F32View::new(&self.attn_mix),
            F32View::new(&layer.out_w),
            Some(F32View::new(&layer.out_b)),
            F32View::new(&self.tmp),
            seq_u,
            d_u,
            d_u,
            d_u,
            d_u,
            d_u,
            false,
        )?;
        encode_whisper_add(
            &mut self.context.borrow_mut(),
            pass,
            F32View::new(&self.hidden),
            F32View::new(&self.tmp),
            F32View::new(&self.hidden),
            seq_u * d_u,
        )
        .map_err(|e| e.to_string())?;

        // MLP.
        encode_whisper_layer_norm(
            &mut self.context.borrow_mut(),
            pass,
            F32View::new(&self.hidden),
            F32View::new(&layer.ln2_w),
            F32View::new(&layer.ln2_b),
            F32View::new(&self.normed),
            seq_u,
            d_u,
            eps,
        )
        .map_err(|e| e.to_string())?;
        mm(
            F32View::new(&self.normed),
            F32View::new(&layer.fc1_w),
            Some(F32View::new(&layer.fc1_b)),
            F32View::new(&self.ffn),
            seq_u,
            d_u,
            ffn_dim,
            d_u,
            d_u,
            ffn_dim,
            false,
        )?;
        encode_whisper_gelu_erf(
            &mut self.context.borrow_mut(),
            pass,
            F32View::new(&self.ffn),
            F32View::new(&self.ffn),
            seq_u * ffn_dim,
        )
        .map_err(|e| e.to_string())?;
        mm(
            F32View::new(&self.ffn),
            F32View::new(&layer.fc2_w),
            Some(F32View::new(&layer.fc2_b)),
            F32View::new(&self.tmp),
            seq_u,
            ffn_dim,
            d_u,
            ffn_dim,
            ffn_dim,
            d_u,
            false,
        )?;
        encode_whisper_add(
            &mut self.context.borrow_mut(),
            pass,
            F32View::new(&self.hidden),
            F32View::new(&self.tmp),
            F32View::new(&self.hidden),
            seq_u * d_u,
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Projects the encoded window through each decoder layer's cross K/V
    /// weights, once per window. Must run after `encode`.
    pub fn build_cross_caches(&mut self) -> Result<(), String> {
        let d = self.config.d_model;
        let d_u = d as u32;
        let seq_u = self.seq as u32;
        autorelease_pool(|| -> Result<(), String> {
            let pass = self.context.borrow().begin_pass_labeled("whisper_cross");
            for layer in &self.dec_layers {
                encode_whisper_matmul_bias(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.enc_out),
                    F32View::new(&layer.cross_k_w),
                    None,
                    F32View::new(&layer.cross_k),
                    seq_u,
                    d_u,
                    d_u,
                    d_u,
                    d_u,
                    d_u,
                    false,
                    1.0,
                )
                .map_err(|e| e.to_string())?;
                encode_whisper_matmul_bias(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.enc_out),
                    F32View::new(&layer.cross_v_w),
                    Some(F32View::new(&layer.cross_v_b)),
                    F32View::new(&layer.cross_v),
                    seq_u,
                    d_u,
                    d_u,
                    d_u,
                    d_u,
                    d_u,
                    false,
                    1.0,
                )
                .map_err(|e| e.to_string())?;
                encode_whisper_transpose(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&layer.cross_v),
                    F32View::new(&layer.cross_v_t),
                    seq_u,
                    d_u,
                )
                .map_err(|e| e.to_string())?;
            }
            pass.commit_and_wait();
            Ok(())
        })
    }

    /// One incremental decoder step for `token` at decoder position `pos`:
    /// every layer consumes the current stream row, appends its self-attention
    /// K/V, cross-attends over the encoded window, and hands the result on.
    /// `embed_table` is the runner's resident token-embedding table
    /// (`[vocab, d_model]` row-major; the input row is computed host-side
    /// because it is one `d_model`-wide vector and the embedding doubles as
    /// the logits weight) and `dec_positions` the decoder's sinusoidal table
    /// (`[max_target_positions, d_model]`). Leaves the updated stream in
    /// `dec_hidden`; call [`Self::logits`] for the vocab projection.
    pub fn step_with_logits(
        &mut self,
        embed_table: &[f32],
        dec_positions: &[f32],
        token: u32,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        if pos != self.self_len {
            return Err(format!(
                "decoder position {pos} does not match the filled cache length {}",
                self.self_len
            ));
        }
        let d = self.config.d_model;
        let row = token as usize * d;
        if row + d > embed_table.len() {
            return Err(format!(
                "token id {token} exceeds the embedding table ({} rows)",
                embed_table.len() / d
            ));
        }
        if (pos + 1) * d > dec_positions.len() {
            return Err(format!(
                "decoder position {pos} exceeds the positional table ({} rows)",
                dec_positions.len() / d
            ));
        }
        let embed_scale = self.config.embed_scale();
        let pos_row = &dec_positions[pos * d..pos * d + d];
        let input: Vec<f32> = embed_table[row..row + d]
            .iter()
            .zip(pos_row.iter())
            .map(|(&e, &p)| e * embed_scale + p)
            .collect();
        write_buffer_bytes(&self.dec_in, 0, &f32_le_bytes(&input));

        let heads = self.config.decoder_attention_heads();
        let head_dim = d / heads;
        let scale = (head_dim as f32).powf(-0.5);
        let d_u = d as u32;
        let eps = self.config.layer_norm_eps;
        let attend_self = (pos + 1) as u32;
        let attend_cross = self.seq as u32;

        autorelease_pool(|| -> Result<Vec<f32>, String> {
            let pass = self.context.borrow().begin_pass_labeled("whisper_step");
            let cpu_encode_started = if super::whisper_profile() {
                Some(std::time::Instant::now())
            } else {
                None
            };
            for (li, layer) in self.dec_layers.iter().enumerate() {
                // Pre-LN self attention with KV cache append. Layer 0's
                // residual base is dec_in, the current token's embedding
                // plus positional row; every later layer's base is the
                // previous layer's output (dec_hidden).
                let base = if li == 0 {
                    &self.dec_in
                } else {
                    &self.dec_hidden
                };
                encode_whisper_layer_norm(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(base),
                    F32View::new(&layer.ln_self_w),
                    F32View::new(&layer.ln_self_b),
                    F32View::new(&self.dec_normed),
                    1,
                    d_u,
                    eps,
                )
                .map_err(|e| e.to_string())?;
                // One packed GEMV projects Q, K, and V (rows 0..d, d..2d,
                // 2d..3d of dec_qkv); the attention step reads the slices.
                encode_whisper_gemv(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.dec_normed),
                    F32View::new(&layer.self_qkv_w),
                    Some(F32View::new(&layer.self_qkv_b)),
                    None,
                    F32View::new(&self.dec_qkv),
                    d_u,
                    3 * d_u,
                    d_u,
                    false,
                    1.0,
                )
                .map_err(|e| e.to_string())?;
                encode_whisper_attn_step(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.dec_qkv),
                    F32View::at(&self.dec_qkv, d as u64),
                    F32View::at(&self.dec_qkv, 2 * d as u64),
                    F32View::new(&layer.self_k_cache),
                    F32View::new(&layer.self_v_cache),
                    F32View::new(&self.dec_attn),
                    pos as u32,
                    attend_self,
                    d_u,
                    head_dim as u32,
                    true,
                    scale,
                    false,
                )
                .map_err(|e| e.to_string())?;
                // out projection with the residual fused: h1 = base + attn_out.
                encode_whisper_gemv(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.dec_attn),
                    F32View::new(&layer.self_out_w),
                    Some(F32View::new(&layer.self_out_b)),
                    Some(F32View::new(base)),
                    F32View::new(&self.dec_h1),
                    d_u,
                    d_u,
                    d_u,
                    false,
                    1.0,
                )
                .map_err(|e| e.to_string())?;

                // Cross attention over the encoded window.
                encode_whisper_layer_norm(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.dec_h1),
                    F32View::new(&layer.ln_cross_w),
                    F32View::new(&layer.ln_cross_b),
                    F32View::new(&self.dec_normed2),
                    1,
                    d_u,
                    eps,
                )
                .map_err(|e| e.to_string())?;
                encode_whisper_gemv(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.dec_normed2),
                    F32View::new(&layer.cross_q_w),
                    Some(F32View::new(&layer.cross_q_b)),
                    None,
                    F32View::new(&self.dec_cq),
                    d_u,
                    d_u,
                    d_u,
                    false,
                    1.0,
                )
                .map_err(|e| e.to_string())?;
                encode_whisper_attn_step(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.dec_cq),
                    F32View::new(&self.dec_cq),
                    F32View::new(&self.dec_cq),
                    F32View::new(&layer.cross_k),
                    F32View::new(&layer.cross_v_t),
                    F32View::new(&self.dec_crossmix),
                    0,
                    attend_cross,
                    d_u,
                    head_dim as u32,
                    false,
                    scale,
                    true,
                )
                .map_err(|e| e.to_string())?;
                encode_whisper_gemv(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.dec_crossmix),
                    F32View::new(&layer.cross_out_w),
                    Some(F32View::new(&layer.cross_out_b)),
                    Some(F32View::new(&self.dec_h1)),
                    F32View::new(&self.dec_h2),
                    d_u,
                    d_u,
                    d_u,
                    false,
                    1.0,
                )
                .map_err(|e| e.to_string())?;

                // FFN.
                encode_whisper_layer_norm(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.dec_h2),
                    F32View::new(&layer.ln_fc_w),
                    F32View::new(&layer.ln_fc_b),
                    F32View::new(&self.dec_normed),
                    1,
                    d_u,
                    eps,
                )
                .map_err(|e| e.to_string())?;
                let ffn_dim = self.config.decoder_ffn_dim() as u32;
                encode_whisper_gemv(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.dec_normed),
                    F32View::new(&layer.fc1_w),
                    Some(F32View::new(&layer.fc1_b)),
                    None,
                    F32View::new(&self.dec_fc1),
                    d_u,
                    ffn_dim,
                    d_u,
                    false,
                    1.0,
                )
                .map_err(|e| e.to_string())?;
                encode_whisper_gelu_erf(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.dec_fc1),
                    F32View::new(&self.dec_fc1),
                    ffn_dim,
                )
                .map_err(|e| e.to_string())?;
                encode_whisper_gemv(
                    &mut self.context.borrow_mut(),
                    &pass,
                    F32View::new(&self.dec_fc1),
                    F32View::new(&layer.fc2_w),
                    Some(F32View::new(&layer.fc2_b)),
                    Some(F32View::new(&self.dec_h2)),
                    F32View::new(&self.dec_hidden),
                    ffn_dim,
                    d_u,
                    ffn_dim,
                    false,
                    1.0,
                )
                .map_err(|e| e.to_string())?;
            }

            // Final layer norm then the tied-embedding projection, in the
            // same pass: one GEMM row against every embedding row.
            encode_whisper_layer_norm(
                &mut self.context.borrow_mut(),
                &pass,
                F32View::new(&self.dec_hidden),
                F32View::new(&self.dec_ln_w),
                F32View::new(&self.dec_ln_b),
                F32View::new(&self.dec_normed2),
                1,
                d_u,
                eps,
            )
            .map_err(|e| e.to_string())?;
            encode_whisper_gemv(
                &mut self.context.borrow_mut(),
                &pass,
                F32View::new(&self.dec_normed2),
                F32View::new(&self.embed_tokens),
                None,
                None,
                F32View::new(&self.logits),
                d_u,
                self.config.vocab_size as u32,
                d_u,
                false,
                1.0,
            )
            .map_err(|e| e.to_string())?;
            if let Some(t0) = cpu_encode_started {
                eprintln!(
                    "[whisper] cpu encode of step pass: {:.3} ms",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            let wall = std::time::Instant::now();
            let gpu_seconds = if super::whisper_profile() {
                pass.commit_and_wait_with_gpu_time()
            } else {
                pass.commit_and_wait();
                0.0
            };
            if super::whisper_profile() {
                eprintln!(
                    "[whisper] step pass: wall {:.3} ms, gpu busy {:.3} ms",
                    wall.elapsed().as_secs_f64() * 1e3,
                    gpu_seconds * 1e3
                );
            }
            self.self_len += 1;
            let vocab = self.config.vocab_size;
            Ok(read_f32_buffer(&self.logits, vocab))
        })
    }
}

/// The per-window decoder over the Metal engine. The cross caches must be
/// built (`build_cross_caches`) before construction; the SOT prompt warms
/// the self caches one fused step at a time, then greedy decoding runs
/// one step-with-logits pass per token -- layer stack, final norm, and
/// the vocab projection in a single command pass, reading the logits
/// back once. Mirrors `decode::WindowDecoder`'s stop conditions exactly
/// (budget, repetition limit, end-of-text) so the two device paths make
/// the same decisions on the same logits.
pub struct MetalWindowDecoder<'a> {
    engine: &'a mut WhisperMetalEngine,
    embed_table: &'a [f32],
    dec_positions: &'a [f32],
    tokens: WhisperSpecialTokens,
    prompt_len: usize,
    /// Decoder position the next emitted token occupies.
    next_pos: usize,
    emitted: Vec<u32>,
    /// Logits of the stream state after the most recently fed token; the
    /// greedy argmax reads this, one pass behind the emissions.
    current_logits: Vec<f32>,
}

impl<'a> MetalWindowDecoder<'a> {
    /// Resets the self caches and warms the SOT prompt through the stack.
    pub fn new(
        engine: &'a mut WhisperMetalEngine,
        embed_table: &'a [f32],
        dec_positions: &'a [f32],
        tokens: WhisperSpecialTokens,
        prompt: &[u32],
    ) -> Result<Self, String> {
        if prompt.is_empty() || prompt.len() > engine.config.max_target_positions {
            return Err("decoder prompt must be nonempty and fit max_target_positions".to_string());
        }
        engine.begin_decode();
        let mut current_logits = Vec::new();
        for (pos, &token) in prompt.iter().enumerate() {
            current_logits = engine.step_with_logits(embed_table, dec_positions, token, pos)?;
        }
        Ok(Self {
            engine,
            embed_table,
            dec_positions,
            tokens,
            prompt_len: prompt.len(),
            next_pos: prompt.len(),
            emitted: Vec::new(),
            current_logits,
        })
    }

    /// Greedy decode until `<|endoftext|>`, the token budget, or the
    /// repetition limit. Returns the emitted text tokens (specials and the
    /// eot excluded).
    pub fn decode(self) -> Result<DecodedWindow, String> {
        self.decode_cancellable(&|| false)
    }
    pub fn decode_cancellable(self, cancelled: &dyn Fn() -> bool) -> Result<DecodedWindow, String> {
        self.decode_with_progress(cancelled, &mut |_, _| true)
    }

    pub fn decode_with_progress(
        mut self,
        cancelled: &dyn Fn() -> bool,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<DecodedWindow, String> {
        let vocab = self.engine.config.vocab_size;
        let budget = self
            .engine
            .config
            .max_target_positions
            .saturating_sub(self.prompt_len)
            .min(440);

        let profile = super::whisper_profile();
        let mut step_ns = 0u128;
        let mut argmax_ns = 0u128;
        let stop = loop {
            if cancelled() || !progress(self.emitted.len(), budget) {
                return Err("audio job cancelled".into());
            }
            if self.emitted.len() >= budget {
                break StopReason::TokenBudget;
            }
            let t0 = std::time::Instant::now();
            let next = argmax_suppressed(&self.current_logits, &self.tokens, vocab);
            let t2 = std::time::Instant::now();
            argmax_ns += (t2 - t0).as_nanos();
            if next == self.tokens.eot {
                break StopReason::EndOfText;
            }
            self.emitted.push(next);
            if self.emitted.len() >= 8
                && self.emitted[self.emitted.len() - 8..]
                    .iter()
                    .all(|&t| t == next)
            {
                break StopReason::RepetitionLimit;
            }
            if self.emitted.len() >= budget {
                break StopReason::TokenBudget;
            }
            self.current_logits = self.engine.step_with_logits(
                self.embed_table,
                self.dec_positions,
                next,
                self.next_pos,
            )?;
            step_ns += (std::time::Instant::now() - t2).as_nanos();
            self.next_pos += 1;
        };
        let _ = step_ns;
        if profile {
            let n = self.emitted.len().max(1) as f64;
            eprintln!(
                "[whisper] decode/token: step {:.3} ms (incl. logits), argmax {:.3} ms",
                step_ns as f64 / n / 1e6,
                argmax_ns as f64 / n / 1e6
            );
        }

        Ok(DecodedWindow {
            text_tokens: std::mem::take(&mut self.emitted),
            stop,
        })
    }
}

/// Language auto-detection on the Metal path: one decoder step over the
/// first window with only `<|startoftranscript|>` fed, argmax restricted
/// to the language token range. Mirrors `decode::detect_language`;
/// deterministic by construction. The self caches are left one token
/// deep -- callers begin a fresh window decode afterwards.
pub fn detect_language_metal(
    engine: &mut WhisperMetalEngine,
    embed_table: &[f32],
    dec_positions: &[f32],
    tokens: &WhisperSpecialTokens,
) -> Result<u32, String> {
    engine.begin_decode();
    let logits = engine.step_with_logits(embed_table, dec_positions, tokens.sot, 0)?;
    let mut best: Option<(f32, u32)> = None;
    for id in tokens.first_language..=tokens.last_language {
        if id as usize >= logits.len() {
            break;
        }
        let score = logits[id as usize];
        if best.is_none_or(|(b, _)| score > b) {
            best = Some((score, id));
        }
    }
    best.map(|(_, id)| id).ok_or_else(|| {
        "no language tokens in the vocabulary; this looks like an English-only model, pass \"en\" explicitly".to_string()
    })
}

impl WhisperMetalEngine {
    /// Reads the encoded window back for parity checks and diagnostics;
    /// the transcription path itself never round-trips the encoder output.
    pub fn read_enc_out(&self) -> Vec<f32> {
        read_f32_buffer(&self.enc_out, self.seq * self.config.d_model)
    }

    /// Reads the current hidden stream (post-transpose-pos, or the last
    /// layer's residual output when `encode_partial` bounded the run).
    pub fn read_hidden(&self) -> Vec<f32> {
        read_f32_buffer(&self.hidden, self.seq * self.config.d_model)
    }

    /// Reads the decoder stream row for diagnostics.
    pub fn read_dec_hidden(&self) -> Vec<f32> {
        read_f32_buffer(&self.dec_hidden, self.config.d_model)
    }

    /// Reads one named decoder scratch buffer for diagnostics: one of
    /// "normed", "q", "k", "v", "attn", "h1", "normed2", "cq", "crossmix",
    /// "co", "fc1", "h2", "tmp", or a layer-0 cross cache "cross_k"/
    /// "cross_v". Diagnostics only; the transcription path never reads
    /// these back.
    pub fn read_dec_scratch(&self, which: &str) -> Vec<f32> {
        let d = self.config.d_model;
        let len = match which {
            "fc1" => self.config.decoder_ffn_dim(),
            "cross_k" | "cross_v" | "cross_v_t" => self.seq * d,
            _ => d,
        };
        let buf = match which {
            "normed" => &self.dec_normed,
            "q" => &self.dec_qkv,
            "k" => &self.dec_qkv,
            "v" => &self.dec_qkv,
            "attn" => &self.dec_attn,
            "h1" => &self.dec_h1,
            "normed2" => &self.dec_normed2,
            "cq" => &self.dec_cq,
            "crossmix" => &self.dec_crossmix,
            "co" => &self.dec_co,
            "fc1" => &self.dec_fc1,
            "h2" => &self.dec_h2,
            "tmp" => &self.dec_tmp,
            "cross_k" => &self.dec_layers[0].cross_k,
            "cross_v" => &self.dec_layers[0].cross_v,
            "cross_v_t" => &self.dec_layers[0].cross_v_t,
            other => panic!("unknown debug buffer {other}"),
        };
        read_f32_buffer(buf, len)
    }

    /// Reads the uploaded mel window back for diagnostics.
    pub fn read_mel_in(&self) -> Vec<f32> {
        read_f32_buffer(&self.mel_in, self.config.n_mels * self.seq * 2)
    }

    /// Reads the conv front end's band-major outputs for diagnostics.
    pub fn read_conv_out(&self) -> (Vec<f32>, Vec<f32>) {
        (
            read_f32_buffer(&self.conv1_out, self.config.d_model * self.seq * 2),
            read_f32_buffer(&self.conv2_out, self.config.d_model * self.seq),
        )
    }
}
