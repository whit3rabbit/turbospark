//! Metal execution of the Multitalker Parakeet streaming encoder layers
//! (`NemotronEncoder::stream_layers` in `crates/audio`), batched over the
//! active speakers of a Frozen E step. Plugs into
//! [`CoupledSession::with_encoder`]; the speaker kernels before it and the
//! RNN-T decode after it stay on the CPU.
//!
//! Each speaker slot keeps its per-layer attention cache (`[attn_frames,
//! hidden]`, the last `norm_self_att` outputs) and conv cache (`[kernel -
//! 1, hidden]`, time-major here) resident on the device; a slot's caches
//! start zeroed and change only when it is stepped. One command buffer per
//! step runs every layer for every active speaker: row-wise work (norms,
//! feed-forward, projections, pointwise convs) is one dispatch over all
//! speakers' rows, and attention and the depthwise conv read each
//! speaker's own caches.
//!
//! Numerics are f32 with f32 accumulation, like the CPU path, but the
//! reduction order differs, so the result is tolerance-gated:
//! `tests/multitalker_metal.rs` holds the encoder output to NeMo and to the
//! CPU path. The feed-forward half residual uses the GEMM's `out_scale` of
//! 0.5, an exact power-of-two scaling like the CPU's `x * 0.5`.
//!
//! [`CoupledSession::with_encoder`]: audio::models::stt::multitalker_asr::scheduler::CoupledSession::with_encoder

use audio::models::stt::multitalker_asr::{MultitalkerAsr, SpeakerEncoder};
use audio::{Result, SpeechError};
use gpu::{
    autorelease_pool, encode_conformer_copy, encode_conformer_depthwise_causal,
    encode_conformer_glu_extend, encode_conformer_relpos_attention, encode_conformer_silu,
    encode_whisper_add, encode_whisper_layer_norm, encode_whisper_matmul_bias, read_f32_buffer,
    write_buffer_bytes, ConformerRelPosAttention, F32View, MetalBuffer, MetalContext, PassEncoder,
    MAX_STREAM_HEAD_DIM, MAX_STREAM_KEYS,
};

fn device_error(why: impl std::fmt::Display) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("Multitalker encoder Metal: {why}"),
    }
}

fn f32_le_bytes(slice: &[f32]) -> Vec<u8> {
    slice.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A layer norm's resident `(weight, bias)`; a bias-free norm gets zeros.
struct Norm(MetalBuffer, MetalBuffer);

struct LayerGpu {
    norm_ff1: Norm,
    ff1_in: MetalBuffer,
    ff1_out: MetalBuffer,
    norm_attn: Norm,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    out: MetalBuffer,
    pos: MetalBuffer,
    bias_u: MetalBuffer,
    bias_v: MetalBuffer,
    norm_conv: Norm,
    pointwise_in: MetalBuffer,
    depthwise: MetalBuffer,
    depthwise_bias: Option<MetalBuffer>,
    conv_norm: Norm,
    pointwise_out: MetalBuffer,
    norm_ff2: Norm,
    ff2_in: MetalBuffer,
    ff2_out: MetalBuffer,
    norm_out: Norm,
}

/// One speaker slot's caches, one buffer per layer.
struct SlotCaches {
    attn: Vec<MetalBuffer>,
    conv: Vec<MetalBuffer>,
}

/// Activation scratch for `speakers` speakers of `rows` rows.
struct Scratch {
    speakers: usize,
    rows: usize,
    cur: MetalBuffer,
    alt: MetalBuffer,
    nrm: MetalBuffer,
    nrm2: MetalBuffer,
    ffh: MetalBuffer,
    tmp: MetalBuffer,
    q: MetalBuffer,
    kvin: MetalBuffer,
    kk: MetalBuffer,
    vv: MetalBuffer,
    mix: MetalBuffer,
    pw: MetalBuffer,
    ext: MetalBuffer,
    dw: MetalBuffer,
    firsts: MetalBuffer,
}

/// The Multitalker streaming encoder resident on the Metal device.
pub struct MultitalkerEncoderMetal {
    context: MetalContext,
    hidden: usize,
    heads: usize,
    ff: usize,
    kernel: usize,
    attn_frames: usize,
    window: usize,
    eps: f32,
    layers: Vec<LayerGpu>,
    slots: Vec<Option<SlotCaches>>,
    scratch: Option<Scratch>,
}

/// One call's shape.
#[derive(Clone, Copy)]
struct Shape {
    speakers: usize,
    rows: usize,
    valid: usize,
}

impl MultitalkerEncoderMetal {
    /// Uploads `asr`'s streaming encoder weights. Fails when no Metal
    /// device initializes; the caller keeps the CPU encoder.
    pub fn new(asr: &MultitalkerAsr) -> Result<Self> {
        let w = asr.stream_weights();
        if w.heads == 0 || w.hidden % w.heads != 0 || w.conv_kernel == 0 || w.chunk == 0 {
            return Err(device_error("invalid encoder geometry"));
        }
        // The attention kernel keeps one score per key and one query vector
        // per head in threadgroup memory. Refuse a checkpoint that does not
        // fit here, so the caller keeps the CPU encoder instead of failing
        // on the first step.
        if w.attn_frames + w.chunk > MAX_STREAM_KEYS || w.hidden / w.heads > MAX_STREAM_HEAD_DIM {
            return Err(device_error(format!(
                "attention of {} cache + {} chunk rows and head size {} exceeds the kernel's \
                 {MAX_STREAM_KEYS} keys and {MAX_STREAM_HEAD_DIM} lanes",
                w.attn_frames,
                w.chunk,
                w.hidden / w.heads
            )));
        }
        let context = MetalContext::new().map_err(device_error)?;
        let up = |data: &[f32]| context.new_buffer_with_data(data);
        let norm = |(weight, bias): (&[f32], Option<&[f32]>)| {
            let zeros;
            let bias = match bias {
                Some(b) => b,
                None => {
                    zeros = vec![0.0f32; weight.len()];
                    &zeros
                }
            };
            Norm(up(weight), up(bias))
        };
        let layers = w
            .layers
            .iter()
            .map(|l| LayerGpu {
                norm_ff1: norm(l.norm_ff1),
                ff1_in: up(l.ff1_in),
                ff1_out: up(l.ff1_out),
                norm_attn: norm(l.norm_attn),
                q: up(l.q),
                k: up(l.k),
                v: up(l.v),
                out: up(l.out),
                pos: up(l.pos),
                bias_u: up(l.bias_u),
                bias_v: up(l.bias_v),
                norm_conv: norm(l.norm_conv),
                pointwise_in: up(l.pointwise_in),
                depthwise: up(l.depthwise),
                depthwise_bias: l.depthwise_bias.map(up),
                conv_norm: norm(l.conv_norm),
                pointwise_out: up(l.pointwise_out),
                norm_ff2: norm(l.norm_ff2),
                ff2_in: up(l.ff2_in),
                ff2_out: up(l.ff2_out),
                norm_out: norm(l.norm_out),
            })
            .collect();
        Ok(Self {
            hidden: w.hidden,
            heads: w.heads,
            ff: w.ff_hidden,
            kernel: w.conv_kernel,
            attn_frames: w.attn_frames,
            window: w.attn_frames + w.chunk,
            eps: w.eps,
            layers,
            slots: Vec::new(),
            scratch: None,
            context,
        })
    }

    /// Drops every slot's caches for a new recording; the weights stay
    /// resident.
    pub fn reset(&mut self) {
        self.slots.clear();
    }

    /// Zeroed caches for `slot` the first time it is stepped.
    fn ensure_slot(&mut self, slot: usize) {
        if self.slots.len() <= slot {
            self.slots.resize_with(slot + 1, || None);
        }
        if self.slots[slot].is_some() {
            return;
        }
        let zeros = |n: usize| self.context.new_buffer_with_data(&vec![0.0f32; n]);
        let attn = (0..self.layers.len())
            .map(|_| zeros(self.attn_frames * self.hidden))
            .collect();
        let conv = (0..self.layers.len())
            .map(|_| zeros((self.kernel - 1) * self.hidden))
            .collect();
        self.slots[slot] = Some(SlotCaches { attn, conv });
    }

    fn ensure_scratch(&mut self, speakers: usize, rows: usize) {
        if self
            .scratch
            .as_ref()
            .is_some_and(|s| s.speakers >= speakers && s.rows >= rows)
        {
            return;
        }
        let (d, ff, a, ctx) = (self.hidden, self.ff, self.attn_frames, self.kernel - 1);
        let c = &self.context;
        let buf = |n: usize| c.new_output_buffer((n.max(1) * 4) as u64);
        let n = speakers * rows;
        self.scratch = Some(Scratch {
            speakers,
            rows,
            cur: buf(n * d),
            alt: buf(n * d),
            nrm: buf(n * d),
            nrm2: buf(n * d),
            ffh: buf(n * ff),
            tmp: buf(n * d),
            q: buf(n * d),
            kvin: buf(speakers * (a + rows) * d),
            kk: buf(speakers * (a + rows) * d),
            vv: buf(speakers * (a + rows) * d),
            mix: buf(n * d),
            pw: buf(n * 2 * d),
            ext: buf(speakers * (ctx + rows) * d),
            dw: buf(n * d),
            firsts: c.new_buffer_with_data(&vec![0u32; speakers]),
        });
    }

    /// Records every layer for the speakers in `slots` and waits; the
    /// output is in the returned buffer (`cur` or `alt`).
    fn run(&mut self, slots: &[usize], shape: Shape) -> std::result::Result<bool, gpu::GpuError> {
        let s = self.scratch.as_ref().expect("scratch allocated");
        let context = &mut self.context;
        let layers = &self.layers;
        let caches = &self.slots;
        let geometry = Geometry {
            hidden: self.hidden,
            heads: self.heads,
            ff: self.ff,
            kernel: self.kernel,
            attn_frames: self.attn_frames,
            window: self.window,
            eps: self.eps,
        };
        autorelease_pool(|| {
            let pass = context.begin_pass_labeled("multitalker_encoder");
            let mut in_cur = true;
            for (index, layer) in layers.iter().enumerate() {
                let (x, y) = if in_cur {
                    (&s.cur, &s.alt)
                } else {
                    (&s.alt, &s.cur)
                };
                let slot_caches: Vec<(&MetalBuffer, &MetalBuffer)> = slots
                    .iter()
                    .map(|&slot| {
                        let c = caches[slot].as_ref().expect("slot allocated");
                        (&c.attn[index], &c.conv[index])
                    })
                    .collect();
                encode_layer(
                    context,
                    &pass,
                    s,
                    layer,
                    &slot_caches,
                    x,
                    y,
                    geometry,
                    shape,
                )?;
                in_cur = !in_cur;
            }
            // A faulted buffer completes "normally" and leaves stale rows and
            // half-updated slot caches behind, so check the status.
            pass.commit_and_wait_checked()?;
            Ok(in_cur)
        })
    }
}

#[derive(Clone, Copy)]
struct Geometry {
    hidden: usize,
    heads: usize,
    ff: usize,
    kernel: usize,
    attn_frames: usize,
    window: usize,
    eps: f32,
}

/// One Conformer layer in `forward_cached`'s order, from `x` (updated in
/// place through the residuals) into `y` (the layer's output norm).
#[allow(clippy::too_many_arguments)]
fn encode_layer(
    c: &mut MetalContext,
    pass: &PassEncoder,
    s: &Scratch,
    l: &LayerGpu,
    caches: &[(&MetalBuffer, &MetalBuffer)],
    x: &MetalBuffer,
    y: &MetalBuffer,
    g: Geometry,
    shape: Shape,
) -> std::result::Result<(), gpu::GpuError> {
    let Shape {
        speakers,
        rows,
        valid,
    } = shape;
    let (d, a, ctx) = (g.hidden, g.attn_frames, g.kernel - 1);
    let n = (speakers * rows) as u32;
    let du = d as u32;
    let kv = (a + rows) as u32;
    let ln = |c: &mut MetalContext, src: &MetalBuffer, norm: &Norm, dst: &MetalBuffer| {
        encode_whisper_layer_norm(
            c,
            pass,
            F32View::new(src),
            F32View::new(&norm.0),
            F32View::new(&norm.1),
            F32View::new(dst),
            n,
            du,
            g.eps,
        )
    };
    // `out = scale * a @ w.T` over `m` rows; no layer here has a bias.
    let mm = |c: &mut MetalContext,
              a_buf: &MetalBuffer,
              w: &MetalBuffer,
              out: &MetalBuffer,
              (m, k, cols): (u32, u32, u32),
              scale: f32| {
        encode_whisper_matmul_bias(
            c,
            pass,
            F32View::new(a_buf),
            F32View::new(w),
            None,
            F32View::new(out),
            m,
            k,
            cols,
            k,
            k,
            cols,
            false,
            scale,
        )
    };
    let add = |c: &mut MetalContext, other: &MetalBuffer| {
        encode_whisper_add(
            c,
            pass,
            F32View::new(x),
            F32View::new(other),
            F32View::new(x),
            n * du,
        )
    };
    let copy = |c: &mut MetalContext, src: F32View, dst: F32View, count: usize| {
        encode_conformer_copy(c, pass, src, dst, count as u32)
    };
    let ff = g.ff as u32;
    let feed_forward = |c: &mut MetalContext,
                        norm: &Norm,
                        w_in: &MetalBuffer,
                        w_out: &MetalBuffer|
     -> std::result::Result<(), gpu::GpuError> {
        ln(c, x, norm, &s.nrm)?;
        mm(c, &s.nrm, w_in, &s.ffh, (n, du, ff), 1.0)?;
        encode_conformer_silu(c, pass, F32View::new(&s.ffh), F32View::new(&s.ffh), n * ff)?;
        mm(c, &s.ffh, w_out, &s.tmp, (n, ff, du), 0.5)?;
        add(c, &s.tmp)
    };

    // Macaron feed-forward 1.
    feed_forward(c, &l.norm_ff1, &l.ff1_in, &l.ff1_out)?;

    // Relative-position attention against [cache ; chunk] per speaker.
    ln(c, x, &l.norm_attn, &s.nrm)?;
    mm(c, &s.nrm, &l.q, &s.q, (n, du, du), 1.0)?;
    let block = (a + rows) * d;
    for (b, (attn_cache, _)) in caches.iter().enumerate() {
        copy(
            c,
            F32View::new(attn_cache),
            F32View::at(&s.kvin, (b * block) as u64),
            a * d,
        )?;
        copy(
            c,
            F32View::at(&s.nrm, (b * rows * d) as u64),
            F32View::at(&s.kvin, (b * block + a * d) as u64),
            rows * d,
        )?;
    }
    let kv_rows = speakers as u32 * kv;
    mm(c, &s.kvin, &l.k, &s.kk, (kv_rows, du, du), 1.0)?;
    mm(c, &s.kvin, &l.v, &s.vv, (kv_rows, du, du), 1.0)?;
    encode_conformer_relpos_attention(
        c,
        pass,
        F32View::new(&s.q),
        F32View::new(&s.kk),
        F32View::new(&s.vv),
        F32View::new(&l.pos),
        F32View::new(&l.bias_u),
        F32View::new(&l.bias_v),
        &s.firsts,
        F32View::new(&s.mix),
        ConformerRelPosAttention {
            speakers: speakers as u32,
            rows: rows as u32,
            valid: valid as u32,
            heads: g.heads as u32,
            head_dim: (d / g.heads) as u32,
            attn_frames: a as u32,
            window: g.window as u32,
        },
    )?;
    // The new cache is the last `attn_frames` rows of [cache ; chunk].
    for (b, (attn_cache, _)) in caches.iter().enumerate() {
        copy(
            c,
            F32View::at(&s.kvin, (b * block + rows * d) as u64),
            F32View::new(attn_cache),
            a * d,
        )?;
    }
    mm(c, &s.mix, &l.out, &s.tmp, (n, du, du), 1.0)?;
    add(c, &s.tmp)?;

    // Convolution module: pointwise in, GLU, causal depthwise over [cache ;
    // GLU], norm, SiLU, pointwise out.
    ln(c, x, &l.norm_conv, &s.nrm)?;
    mm(c, &s.nrm, &l.pointwise_in, &s.pw, (n, du, 2 * du), 1.0)?;
    let ext_block = (ctx + rows) * d;
    for (b, (_, conv_cache)) in caches.iter().enumerate() {
        copy(
            c,
            F32View::new(conv_cache),
            F32View::at(&s.ext, (b * ext_block) as u64),
            ctx * d,
        )?;
    }
    encode_conformer_glu_extend(
        c,
        pass,
        F32View::new(&s.pw),
        F32View::new(&s.ext),
        speakers as u32,
        rows as u32,
        valid as u32,
        du,
        ctx as u32,
    )?;
    for (b, (_, conv_cache)) in caches.iter().enumerate() {
        copy(
            c,
            F32View::at(&s.ext, (b * ext_block + rows * d) as u64),
            F32View::new(conv_cache),
            ctx * d,
        )?;
    }
    encode_conformer_depthwise_causal(
        c,
        pass,
        F32View::new(&s.ext),
        F32View::new(&l.depthwise),
        l.depthwise_bias.as_ref().map(F32View::new),
        F32View::new(&s.dw),
        speakers as u32,
        rows as u32,
        du,
        g.kernel as u32,
    )?;
    ln(c, &s.dw, &l.conv_norm, &s.nrm2)?;
    encode_conformer_silu(
        c,
        pass,
        F32View::new(&s.nrm2),
        F32View::new(&s.nrm2),
        n * du,
    )?;
    mm(c, &s.nrm2, &l.pointwise_out, &s.tmp, (n, du, du), 1.0)?;
    add(c, &s.tmp)?;

    // Macaron feed-forward 2, then the layer's output norm.
    feed_forward(c, &l.norm_ff2, &l.ff2_in, &l.ff2_out)?;
    ln(c, x, &l.norm_out, y)
}

impl SpeakerEncoder for MultitalkerEncoderMetal {
    fn stream_layers(
        &mut self,
        slots: &[usize],
        inputs: &[&[f32]],
        rows: usize,
        valid: usize,
        cache_lens: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let d = self.hidden;
        let speakers = slots.len();
        let chunk = self.window - self.attn_frames;
        if speakers == 0
            || inputs.len() != speakers
            || cache_lens.len() != speakers
            || rows == 0
            || rows > chunk
            || valid > rows
            || inputs.iter().any(|x| x.len() != rows * d)
        {
            return Err(SpeechError::Input {
                why: format!(
                    "device encoder step of {speakers} speakers x {rows} rows ({valid} valid) \
                     does not fit {chunk}-row chunks of {d}"
                ),
            });
        }
        let mut sorted = slots.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != speakers {
            return Err(SpeechError::Input {
                why: "a speaker slot appears twice in one step".into(),
            });
        }
        for &slot in slots {
            self.ensure_slot(slot);
        }
        self.ensure_scratch(speakers, rows);
        let s = self.scratch.as_ref().expect("scratch allocated");
        let packed: Vec<f32> = inputs.iter().flat_map(|x| x.iter().copied()).collect();
        write_buffer_bytes(&s.cur, 0, &f32_le_bytes(&packed));
        let firsts: Vec<u8> = cache_lens
            .iter()
            .map(|&len| (self.attn_frames - len.min(self.attn_frames)) as u32)
            .flat_map(|f| f.to_le_bytes())
            .collect();
        write_buffer_bytes(&s.firsts, 0, &firsts);
        let in_cur = self
            .run(
                slots,
                Shape {
                    speakers,
                    rows,
                    valid,
                },
            )
            .map_err(device_error)?;
        let s = self.scratch.as_ref().expect("scratch allocated");
        let out = read_f32_buffer(if in_cur { &s.cur } else { &s.alt }, speakers * rows * d);
        Ok(out.chunks_exact(rows * d).map(<[f32]>::to_vec).collect())
    }
}
