//! Metal execution of the Nemotron-3 diarization encoder
//! ([`NemotronDiarization::encoder_forward`]), the transformer stack the
//! coupled Frozen E diarizer re-runs over its spkcache + FIFO + chunk window
//! every step. Plugs into [`CoupledDiarizer::with_executor`].
//!
//! Numerics are f32 with f32 accumulation, like the CPU path, but the
//! reduction order differs (Metal tiled GEMM, fast-math shaders), so the
//! result is tolerance-gated, not bit-exact: `tests/nemotron_diar_metal.rs`
//! holds the encoder output and the coupled chunk predictions to the CPU.
//!
//! Kernels: the whisper f32 set (`shaders/whisper_encoder.metal`) plus the
//! strided NeoX rotary in `shaders/nemotron_diar.metal`. One command buffer
//! per call. Keys at or past `lengths` are masked on the CPU with `-inf`;
//! here the score GEMM, softmax and value mix simply stop at `lengths`
//! keys, which is the same function. Input scaling and the embed norm run
//! on the host with the CPU kernels (this checkpoint uses neither).
//!
//! [`CoupledDiarizer::with_executor`]: audio::vad::nemotron_diarization::CoupledDiarizer::with_executor

use audio::vad::nemotron_diarization::{EncoderExecutor, NemotronDiarization};
use audio::{ops, Result, SpeechError};
use gpu::{
    autorelease_pool, encode_nemotron_rope_neox, encode_whisper_add, encode_whisper_gelu_erf,
    encode_whisper_layer_norm, encode_whisper_matmul_bias, encode_whisper_softmax_rows,
    read_f32_buffer, write_buffer_bytes, F32View, MetalBuffer, MetalContext, PassEncoder,
};

const EPS: f32 = 1e-5;

fn device_error(why: impl std::fmt::Display) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("Nemotron diarization Metal: {why}"),
    }
}

fn f32_le_bytes(slice: &[f32]) -> Vec<u8> {
    slice.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// One layer's resident weights (linear weights `[out, in]`).
struct LayerGpu {
    norm1: (MetalBuffer, MetalBuffer),
    qkv_w: MetalBuffer,
    qkv_b: Option<MetalBuffer>,
    out_w: MetalBuffer,
    out_b: MetalBuffer,
    norm2: (MetalBuffer, MetalBuffer),
    ff1_w: MetalBuffer,
    ff1_b: MetalBuffer,
    ff2_w: MetalBuffer,
    ff2_b: MetalBuffer,
}

/// Activation scratch for up to `rows` rows, plus the RoPE tables.
struct Scratch {
    rows: usize,
    hidden: MetalBuffer,
    normed: MetalBuffer,
    qkv: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    mix: MetalBuffer,
    /// `heads` planes of `[rows, keys]` scores, keys at most `rows`.
    scores: MetalBuffer,
    ffn: MetalBuffer,
    tmp: MetalBuffer,
    cos: MetalBuffer,
    sin: MetalBuffer,
}

/// The diarization encoder resident on the Metal device.
pub struct NemotronDiarMetal<'m> {
    model: &'m NemotronDiarization,
    context: MetalContext,
    d: usize,
    heads: usize,
    ff: usize,
    rotary: usize,
    rope_base: f32,
    layers: Vec<LayerGpu>,
    final_norm: (MetalBuffer, MetalBuffer),
    scratch: Option<Scratch>,
}

impl<'m> NemotronDiarMetal<'m> {
    /// Uploads `model`'s encoder weights. Fails when no Metal device
    /// initializes; the caller keeps the CPU encoder.
    pub fn new(model: &'m NemotronDiarization) -> Result<Self> {
        let context = MetalContext::new().map_err(device_error)?;
        let w = model.encoder_weights();
        if w.heads == 0 || w.d_model % w.heads != 0 {
            return Err(device_error("d_model must divide into the heads"));
        }
        let upload = |data: &[f32]| context.new_buffer_with_data(data);
        let pair = |(a, b): (&[f32], &[f32])| (upload(a), upload(b));
        let layers = w
            .layers
            .iter()
            .map(|l| LayerGpu {
                norm1: pair(l.norm1),
                qkv_w: upload(l.qkv_w),
                qkv_b: l.qkv_b.map(upload),
                out_w: upload(l.out_w),
                out_b: upload(l.out_b),
                norm2: pair(l.norm2),
                ff1_w: upload(l.ff1_w),
                ff1_b: upload(l.ff1_b),
                ff2_w: upload(l.ff2_w),
                ff2_b: upload(l.ff2_b),
            })
            .collect();
        let final_norm = pair(w.final_norm);
        Ok(Self {
            model,
            d: w.d_model,
            heads: w.heads,
            ff: w.ff_hidden,
            rotary: w.rotary_dim,
            rope_base: w.rope_base,
            layers,
            final_norm,
            scratch: None,
            context,
        })
    }

    /// Scratch for at least `rows` rows, grown (never shrunk) on demand.
    fn ensure_scratch(&mut self, rows: usize) {
        if self.scratch.as_ref().is_some_and(|s| s.rows >= rows) {
            return;
        }
        let (d, ff, heads) = (self.d, self.ff, self.heads);
        let c = &self.context;
        let buf = |elements: usize| c.new_output_buffer((elements.max(1) * 4) as u64);
        let half = self.rotary / 2;
        let (cos, sin) = ops::rope_tables(rows, self.rotary, self.rope_base);
        debug_assert_eq!(cos.len(), rows * half);
        self.scratch = Some(Scratch {
            rows,
            hidden: buf(rows * d),
            normed: buf(rows * d),
            qkv: buf(rows * 3 * d),
            q: buf(rows * d),
            k: buf(rows * d),
            mix: buf(rows * d),
            scores: buf(heads * rows * rows),
            ffn: buf(rows * ff),
            tmp: buf(rows * d),
            cos: c.new_buffer_with_data(&cos),
            sin: c.new_buffer_with_data(&sin),
        });
    }

    /// Records the stack for `rows` rows with `keys` valid keys and returns
    /// once the device finishes; the result is in `scratch.normed`.
    fn run(&mut self, rows: usize, keys: usize) -> std::result::Result<(), gpu::GpuError> {
        let s = self.scratch.as_ref().expect("scratch allocated");
        let (d, ff, heads, rotary) = (self.d, self.ff, self.heads, self.rotary);
        let head_dim = d / heads;
        let scale = (head_dim as f32).powf(-0.5);
        let (r, du) = (rows as u32, d as u32);
        let context = &mut self.context;
        autorelease_pool(|| {
            let pass = context.begin_pass_labeled("nemotron_diar_encoder");
            for layer in &self.layers {
                let geometry = Geometry {
                    rows,
                    keys,
                    d,
                    ff,
                    heads,
                    rotary,
                    scale,
                };
                encode_layer(context, &pass, s, layer, geometry)?;
            }
            encode_whisper_layer_norm(
                context,
                &pass,
                F32View::new(&s.hidden),
                F32View::new(&self.final_norm.0),
                F32View::new(&self.final_norm.1),
                F32View::new(&s.normed),
                r,
                du,
                EPS,
            )?;
            pass.commit_and_wait();
            Ok(())
        })
    }
}

/// One call's shape: `rows` query rows, `keys` valid keys, and the model's
/// widths and attention scale.
#[derive(Clone, Copy)]
struct Geometry {
    rows: usize,
    keys: usize,
    d: usize,
    ff: usize,
    heads: usize,
    rotary: usize,
    scale: f32,
}

/// One pre-norm transformer layer on the caller's pass, in
/// `encoder_forward`'s order.
fn encode_layer(
    c: &mut MetalContext,
    pass: &PassEncoder,
    s: &Scratch,
    layer: &LayerGpu,
    g: Geometry,
) -> std::result::Result<(), gpu::GpuError> {
    let Geometry {
        rows,
        keys,
        d,
        ff,
        heads,
        rotary,
        scale,
    } = g;
    let head_dim = d / heads;
    let (r, du, kv) = (rows as u32, d as u32, keys as u32);
    let (hd, d3, ffu) = (head_dim as u32, 3 * d as u32, ff as u32);
    let mm = |c: &mut MetalContext,
              a: F32View,
              w: F32View,
              bias: Option<F32View>,
              out: F32View,
              (m, k, n): (u32, u32, u32),
              (a_stride, w_stride, out_stride): (u32, u32, u32),
              w_t: bool,
              out_scale: f32| {
        encode_whisper_matmul_bias(
            c, pass, a, w, bias, out, m, k, n, a_stride, w_stride, out_stride, w_t, out_scale,
        )
    };

    // Attention block.
    encode_whisper_layer_norm(
        c,
        pass,
        F32View::new(&s.hidden),
        F32View::new(&layer.norm1.0),
        F32View::new(&layer.norm1.1),
        F32View::new(&s.normed),
        r,
        du,
        EPS,
    )?;
    mm(
        c,
        F32View::new(&s.normed),
        F32View::new(&layer.qkv_w),
        layer.qkv_b.as_ref().map(F32View::new),
        F32View::new(&s.qkv),
        (r, du, d3),
        (du, du, d3),
        false,
        1.0,
    )?;
    for (offset, out) in [(0u64, &s.q), (d as u64, &s.k)] {
        encode_nemotron_rope_neox(
            c,
            pass,
            F32View::at(&s.qkv, offset),
            F32View::new(&s.cos),
            F32View::new(&s.sin),
            F32View::new(out),
            r,
            heads as u32,
            hd,
            rotary as u32,
            d3,
            du,
        )?;
    }
    let plane = (rows * keys) as u64;
    for h in 0..heads {
        let head = (h * head_dim) as u64;
        mm(
            c,
            F32View::at(&s.q, head),
            F32View::at(&s.k, head),
            None,
            F32View::at(&s.scores, h as u64 * plane),
            (r, hd, kv),
            (du, du, kv),
            false,
            scale,
        )?;
    }
    encode_whisper_softmax_rows(
        c,
        pass,
        F32View::new(&s.scores),
        F32View::new(&s.scores),
        (heads * rows) as u32,
        kv,
    )?;
    for h in 0..heads {
        let head = (h * head_dim) as u64;
        mm(
            c,
            F32View::at(&s.scores, h as u64 * plane),
            F32View::at(&s.qkv, 2 * d as u64 + head),
            None,
            F32View::at(&s.mix, head),
            (r, kv, hd),
            (kv, d3, du),
            true,
            1.0,
        )?;
    }
    mm(
        c,
        F32View::new(&s.mix),
        F32View::new(&layer.out_w),
        Some(F32View::new(&layer.out_b)),
        F32View::new(&s.tmp),
        (r, du, du),
        (du, du, du),
        false,
        1.0,
    )?;
    encode_whisper_add(
        c,
        pass,
        F32View::new(&s.hidden),
        F32View::new(&s.tmp),
        F32View::new(&s.hidden),
        r * du,
    )?;

    // Feed-forward block.
    encode_whisper_layer_norm(
        c,
        pass,
        F32View::new(&s.hidden),
        F32View::new(&layer.norm2.0),
        F32View::new(&layer.norm2.1),
        F32View::new(&s.normed),
        r,
        du,
        EPS,
    )?;
    mm(
        c,
        F32View::new(&s.normed),
        F32View::new(&layer.ff1_w),
        Some(F32View::new(&layer.ff1_b)),
        F32View::new(&s.ffn),
        (r, du, ffu),
        (du, du, ffu),
        false,
        1.0,
    )?;
    encode_whisper_gelu_erf(c, pass, F32View::new(&s.ffn), F32View::new(&s.ffn), r * ffu)?;
    mm(
        c,
        F32View::new(&s.ffn),
        F32View::new(&layer.ff2_w),
        Some(F32View::new(&layer.ff2_b)),
        F32View::new(&s.tmp),
        (r, ffu, du),
        (ffu, ffu, du),
        false,
        1.0,
    )?;
    encode_whisper_add(
        c,
        pass,
        F32View::new(&s.hidden),
        F32View::new(&s.tmp),
        F32View::new(&s.hidden),
        r * du,
    )
}

impl EncoderExecutor for NemotronDiarMetal<'_> {
    fn encoder_forward(&mut self, x: &mut [f32], rows: usize, lengths: usize) -> Result<()> {
        let d = self.d;
        if rows == 0 || x.len() != rows * d || lengths > rows {
            return Err(SpeechError::Input {
                why: format!(
                    "diarization encoder input of {} values does not hold {rows} rows of {d} \
                     ({lengths} valid)",
                    x.len()
                ),
            });
        }
        if lengths == 0 {
            // Every key masked: the CPU's own (degenerate) result.
            self.model.encoder_forward(x, rows, lengths);
            return Ok(());
        }
        let w = self.model.encoder_weights();
        if w.xscaling {
            let scale = (d as f64).sqrt() as f32;
            for v in x.iter_mut() {
                *v *= scale;
            }
        }
        if let Some((nw, nb)) = w.embed_norm {
            ops::layernorm(x, rows, d, nw, Some(nb), EPS);
        }
        self.ensure_scratch(rows);
        let s = self.scratch.as_ref().expect("scratch allocated");
        write_buffer_bytes(&s.hidden, 0, &f32_le_bytes(x));
        self.run(rows, lengths).map_err(device_error)?;
        let s = self.scratch.as_ref().expect("scratch allocated");
        x.copy_from_slice(&read_f32_buffer(&s.normed, rows * d));
        Ok(())
    }
}
