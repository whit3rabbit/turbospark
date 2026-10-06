//! Resident Music 3 matrices and typed tensor operations in f32 storage. Quantized matrices
//! remain in checkpoint packing and decode inside the projection kernel.
use crate::{autorelease_pool, read_f32_buffer, GpuError, MetalBuffer, MetalContext};
use half::{bf16, f16};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

pub(crate) static SOURCE: &str = include_str!("shaders/music3.metal");
// Native MLX arithmetic needs precise division; the FP32 diagnostic keeps its
// established compiler policy. Both sources are bundled and cache separately.
static NATIVE_SOURCE: &str = concat!(
    "// turbospark: precise-math\n",
    include_str!("shaders/music3.metal"),
    include_str!("shaders/music3_attention_fallback.metal")
);
fn source(dtype: Music3DType) -> &'static str {
    if dtype == Music3DType::F32 {
        SOURCE
    } else {
        NATIVE_SOURCE
    }
}

/// Logical activation dtype. Storage stays f32; every operation rounds its result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Music3DType {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Music3Encoding {
    F32,
    F16,
    Bf16,
    Affine { bits: u32, group_size: usize },
    MxFp4,
    MxFp8,
    NvFp4,
}
impl Music3Encoding {
    fn parameters(self) -> (u32, u32, usize) {
        match self {
            Self::F32 => (0, 32, 1),
            Self::F16 => (1, 16, 1),
            Self::Bf16 => (2, 16, 1),
            Self::Affine { bits, group_size } => (3, bits, group_size),
            Self::MxFp4 => (4, 4, 32),
            Self::MxFp8 => (5, 8, 32),
            Self::NvFp4 => (6, 4, 16),
        }
    }
}

struct State {
    context: RefCell<MetalContext>,
    resident: Cell<usize>,
}
#[derive(Clone)]
pub struct Music3Device {
    state: Rc<State>,
}
pub struct Music3Weight {
    state: Rc<State>,
    bytes: MetalBuffer,
    scales: MetalBuffer,
    offsets: MetalBuffer,
    block_scales: MetalBuffer,
    shape: Vec<usize>,
    encoding: Music3Encoding,
    resident: usize,
}
impl Drop for Music3Weight {
    fn drop(&mut self) {
        self.state
            .resident
            .set(self.state.resident.get() - self.resident);
    }
}
fn bad(message: impl Into<String>) -> GpuError {
    GpuError::InvalidInput(message.into())
}
fn product(values: &[usize]) -> Result<usize, GpuError> {
    values.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| bad("Music 3 shape exceeds device indexing"))
    })
}
fn params(values: &[usize]) -> Result<Vec<u32>, GpuError> {
    values
        .iter()
        .map(|&v| u32::try_from(v).map_err(|_| bad("Music 3 dimension overflows u32")))
        .collect()
}
fn bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn e4m3(c: u8) -> f32 {
    let e = (c >> 3) & 15;
    let m = c & 7;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let v = if e == 0 {
        m as f32 * 2f32.powi(-9)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    };
    if c & 128 != 0 {
        -v
    } else {
        v
    }
}

impl Music3Device {
    pub fn new() -> Result<Self, GpuError> {
        Ok(Self {
            state: Rc::new(State {
                context: RefCell::new(MetalContext::new()?),
                resident: Cell::new(0),
            }),
        })
    }
    pub fn resident_weight_bytes(&self) -> usize {
        self.state.resident.get()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_weight(
        &self,
        shape: &[usize],
        data: &[u8],
        encoding: Music3Encoding,
        scales: &[f32],
        offsets: &[f32],
        block_scales: &[u8],
    ) -> Result<Music3Weight, GpuError> {
        let elements = product(shape)?;
        if shape.len() < 2 || shape.len() > 3 || shape.contains(&0) {
            return Err(bad(
                "Music 3 weights need non-empty matrices or convolution tensors",
            ));
        }
        let (mode, bits, group) = encoding.parameters();
        let expected = if mode <= 2 {
            elements * (bits as usize / 8)
        } else {
            if shape.len() != 2
                || group == 0
                || shape[1] % group != 0
                || !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8)
            {
                return Err(bad("Music 3 invalid quantized geometry"));
            }
            shape[0] * (shape[1] * bits as usize).div_ceil(32) * 4
        };
        if data.len() != expected {
            return Err(bad("Music 3 weight byte count differs from its shape"));
        }
        if mode == 3 {
            let groups = elements / group;
            if scales.len() != groups
                || offsets.len() != groups
                || !block_scales.is_empty()
                || scales.iter().chain(offsets).any(|v| !v.is_finite())
            {
                return Err(bad("Music 3 affine companions do not match weights"));
            }
        } else if mode >= 4 {
            if block_scales.len() != elements / group || !scales.is_empty() || !offsets.is_empty() {
                return Err(bad("Music 3 block scales do not match weights"));
            }
            for &s in block_scales {
                let v = if mode == 6 {
                    e4m3(s)
                } else {
                    2f32.powi(s as i32 - 127)
                };
                if (mode != 6 && s == 255) || !v.is_finite() {
                    return Err(bad("Music 3 non-finite block scale"));
                }
            }
            if mode == 5 && data.iter().any(|&c| c & 127 == 127) {
                return Err(bad("Music 3 MXFP8 NaN weight"));
            }
        } else {
            if !scales.is_empty() || !offsets.is_empty() || !block_scales.is_empty() {
                return Err(bad("Music 3 dense weights carry packed companions"));
            }
            let finite = match encoding {
                Music3Encoding::F32 => data
                    .chunks_exact(4)
                    .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
                Music3Encoding::F16 => data
                    .chunks_exact(2)
                    .all(|v| f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).is_finite()),
                _ => data.chunks_exact(2).all(|v| {
                    bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).is_finite()
                }),
            };
            if !finite {
                return Err(bad("Music 3 non-finite dense weight"));
            }
        }
        autorelease_pool(|| {
            let c = self.state.context.borrow();
            // Metal refuses zero-length buffers; unused companions bind a single zero.
            let weight = c.new_buffer_with_data(data);
            let sc = c.new_buffer_with_data(if scales.is_empty() { &[0.0] } else { scales });
            let off = c.new_buffer_with_data(if offsets.is_empty() { &[0.0] } else { offsets });
            let bs = c.new_buffer_with_data(if block_scales.is_empty() {
                &[0u8]
            } else {
                block_scales
            });
            let resident = weight.length() as usize
                + sc.length() as usize
                + off.length() as usize
                + bs.length() as usize;
            self.state
                .resident
                .set(self.state.resident.get() + resident);
            Ok(Music3Weight {
                state: self.state.clone(),
                bytes: weight,
                scales: sc,
                offsets: off,
                block_scales: bs,
                shape: shape.to_vec(),
                encoding,
                resident,
            })
        })
    }

    /// Batched row-major SDPA, with a time-major KV option for AR decode.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        queries: usize,
        keys: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        time_major: bool,
        causal: bool,
        offset: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.attention_typed(
            q,
            k,
            v,
            batch,
            queries,
            keys,
            heads,
            kv_heads,
            dim,
            time_major,
            causal,
            offset,
            Music3DType::F32,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn attention_typed(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        queries: usize,
        keys: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        time_major: bool,
        causal: bool,
        offset: usize,
        dtype: Music3DType,
    ) -> Result<Vec<f32>, GpuError> {
        let qcount = product(&[batch, queries, heads, dim])?;
        let kcount = product(&[batch, keys, kv_heads, dim])?;
        if batch == 0
            || queries == 0
            || keys == 0
            || kv_heads == 0
            || heads == 0
            || heads % kv_heads != 0
            || dim == 0
            || dim > 256
            || q.len() != qcount
            || k.len() != kcount
            || v.len() != kcount
            || offset.checked_add(queries).is_none()
        {
            return Err(bad("Music 3 attention geometry mismatch"));
        }
        let vector =
            dtype != Music3DType::F32 && queries <= 8 && matches!(dim, 64 | 96 | 128 | 256);
        let mma = dtype != Music3DType::F32 && queries > 8 && matches!(dim, 64 | 128);
        let fallback = dtype != Music3DType::F32 && !vector && !mma;
        let threads = if vector {
            1024
        } else if mma {
            32
        } else {
            dim.next_power_of_two().max(32)
        };
        let p = bytes(&params(&[
            batch,
            queries,
            keys,
            heads,
            kv_heads,
            dim,
            time_major as usize,
            causal as usize,
            offset,
            threads,
            dtype as usize,
            ((1.0 / (dim as f64).sqrt()) as f32).to_bits() as usize,
        ])?);
        autorelease_pool(|| {
            let mut c = self.state.context.borrow_mut();
            if fallback {
                let threads = keys.div_ceil(4).div_ceil(32).min(32) * 32;
                let mut fp = p.clone();
                fp[36..40].copy_from_slice(&(threads as u32).to_le_bytes());
                let constants = crate::rms_norm::unused_function_constants();
                let qk = c.pipeline(
                    NATIVE_SOURCE,
                    "music3_attention_fallback_qk",
                    &constants,
                    b"",
                )?;
                let softmax = c.pipeline(
                    NATIVE_SOURCE,
                    "music3_attention_fallback_softmax",
                    &constants,
                    b"",
                )?;
                let pv = c.pipeline(
                    NATIVE_SOURCE,
                    "music3_attention_fallback_pv",
                    &constants,
                    b"",
                )?;
                let qb = c.new_buffer_with_data(q);
                let kb = c.new_buffer_with_data(k);
                let vb = c.new_buffer_with_data(v);
                let out = c.new_output_buffer((qcount * 4) as u64);
                let size = (product(&[batch, heads, queries, keys])? * 4) as u64;
                let scores = c.new_output_buffer(size);
                let probabilities = c.new_output_buffer(size);
                let pass = c.begin_pass();
                pass.encode_threadgroups(
                    &qk,
                    &[(&qb, 0, 0), (&kb, 1, 0), (&scores, 2, 0)],
                    &[(&fp, 3)],
                    (batch * heads * queries.div_ceil(8) * keys.div_ceil(8)) as u64,
                    32,
                );
                pass.encode_threadgroups(
                    &softmax,
                    &[(&scores, 0, 0), (&probabilities, 1, 0)],
                    &[(&fp, 2)],
                    (batch * heads * queries) as u64,
                    threads as u64,
                );
                pass.encode_threadgroups(
                    &pv,
                    &[(&probabilities, 0, 0), (&vb, 1, 0), (&out, 2, 0)],
                    &[(&fp, 3)],
                    (batch * heads * queries.div_ceil(8) * dim.div_ceil(8)) as u64,
                    32,
                );
                pass.commit_and_wait_checked()?;
                return finite_output(&out, qcount);
            }
            let pipeline = c.pipeline(
                source(dtype),
                if vector {
                    "music3_attention_vector"
                } else if mma {
                    "music3_attention_mma"
                } else {
                    "music3_attention"
                },
                &crate::rms_norm::unused_function_constants(),
                b"",
            )?;
            let qb = c.new_buffer_with_data(q);
            let kb = c.new_buffer_with_data(k);
            let vb = c.new_buffer_with_data(v);
            let out = c.new_output_buffer((qcount * 4) as u64);
            let scratch =
                c.new_output_buffer((product(&[batch, queries, heads, keys])? * 4) as u64);
            let pass = c.begin_pass();
            pass.encode_threadgroups(
                &pipeline,
                &[
                    (&qb, 0, 0),
                    (&kb, 1, 0),
                    (&vb, 2, 0),
                    (&out, 3, 0),
                    (&scratch, 5, 0),
                ],
                &[(&p, 4)],
                if mma {
                    (batch * queries.div_ceil(8) * heads) as u64
                } else {
                    (batch * queries * heads) as u64
                },
                threads as u64,
            );
            pass.commit_and_wait_checked()?;
            finite_output(&out, qcount)
        })
    }
    pub fn rms_norm(
        &self,
        x: &[f32],
        w: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        dtype: Music3DType,
    ) -> Result<Vec<f32>, GpuError> {
        self.norm(x, w, None, rows, cols, eps, dtype, false)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &self,
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
        dtype: Music3DType,
    ) -> Result<Vec<f32>, GpuError> {
        self.norm(x, w, bias, rows, cols, eps, dtype, true)
    }
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
        dtype: Music3DType,
        layer: bool,
    ) -> Result<Vec<f32>, GpuError> {
        let count = product(&[rows, cols])?;
        if rows == 0
            || cols == 0
            || cols > 8192
            || x.len() != count
            || w.len() != cols
            || bias.is_some_and(|b| b.len() != cols)
            || !eps.is_finite()
            || eps < 0.0
        {
            return Err(bad("Music 3 norm geometry mismatch"));
        }
        let reads = if layer { 8 } else { 4 };
        let threads = cols.div_ceil(reads).div_ceil(32).clamp(1, 32) * 32;
        let p = bytes(&[
            cols as u32,
            dtype as u32,
            layer as u32,
            bias.is_some() as u32,
            reads as u32,
            threads as u32,
            eps.to_bits(),
        ]);
        autorelease_pool(|| {
            let mut c = self.state.context.borrow_mut();
            let pipeline = c.pipeline(
                source(dtype),
                "music3_norm",
                &crate::rms_norm::unused_function_constants(),
                b"",
            )?;
            let xb = c.new_buffer_with_data(x);
            let wb = c.new_buffer_with_data(w);
            let bb = c.new_buffer_with_data(bias.unwrap_or(&[0.0]));
            let out = c.new_output_buffer((count * 4) as u64);
            let pass = c.begin_pass();
            pass.encode_threadgroups(
                &pipeline,
                &[(&xb, 0, 0), (&wb, 1, 0), (&bb, 2, 0), (&out, 3, 0)],
                &[(&p, 4)],
                rows as u64,
                threads as u64,
            );
            pass.commit_and_wait_checked()?;
            finite_output(&out, count)
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        dim: usize,
        offset: usize,
        theta: f32,
        dtype: Music3DType,
    ) -> Result<Vec<f32>, GpuError> {
        let count = product(&[batch, seq, heads, dim])?;
        if count == 0
            || dim % 2 != 0
            || x.len() != count
            || !theta.is_finite()
            || theta <= 0.0
            || offset.checked_add(seq).is_none()
        {
            return Err(bad("Music 3 rotary geometry mismatch"));
        }
        // Kernel positions are u32; reject truncation before encoding.
        if u32::try_from(offset + seq).is_err() {
            return Err(bad("Music 3 rotary positions exceed Metal indexing"));
        }
        let p = bytes(&[
            seq as u32,
            heads as u32,
            dim as u32,
            offset as u32,
            theta.log2().to_bits(),
            dtype as u32,
        ]);
        self.unary("music3_rope", x, &p, count / 2, dtype)
    }
    pub fn normal_from_uniform(&self, x: &[f32], dtype: Music3DType) -> Result<Vec<f32>, GpuError> {
        if x.is_empty() || x.iter().any(|v| !v.is_finite() || *v <= -1.0 || *v >= 1.0) {
            return Err(bad("Music 3 normal transform expects uniforms in (-1,1)"));
        }
        self.unary(
            "music3_normal",
            x,
            &bytes(&[x.len() as u32, dtype as u32]),
            x.len(),
            dtype,
        )
    }
    /// DiT computes tables in FP32 using inverse positive powers before
    /// casting them at the separate partial-rotary multiplication boundary.
    pub fn rotary_tables(
        &self,
        seq: usize,
        dim: usize,
        theta: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        let total = product(&[seq, dim])?;
        if total == 0 || dim % 2 != 0 || !theta.is_finite() || theta <= 0.0 {
            return Err(bad("Music 3 rotary table geometry mismatch"));
        }
        let mut values = self.unary(
            "music3_rotary_tables",
            &vec![0.0; total],
            &bytes(&[seq as u32, dim as u32, theta.to_bits()]),
            total / 2,
            Music3DType::Bf16,
        )?;
        let sin = values.split_off(total / 2);
        Ok((values, sin))
    }
    pub fn snake(
        &self,
        x: &[f32],
        alpha: &[f32],
        channels: usize,
        frames: usize,
        dtype: Music3DType,
    ) -> Result<Vec<f32>, GpuError> {
        let count = product(&[channels, frames])?;
        if count == 0 || x.len() != count || alpha.len() != channels {
            return Err(bad("Music 3 Snake geometry mismatch"));
        }
        let p = bytes(&[frames as u32, dtype as u32, count as u32, 2.0f32.to_bits()]);
        autorelease_pool(|| {
            let mut c = self.state.context.borrow_mut();
            let pipeline = c.pipeline(
                source(dtype),
                "music3_snake",
                &crate::rms_norm::unused_function_constants(),
                b"",
            )?;
            let xb = c.new_buffer_with_data(x);
            let ab = c.new_buffer_with_data(alpha);
            let out = c.new_output_buffer((count * 4) as u64);
            let pass = c.begin_pass();
            pass.encode_threads_3d(
                &pipeline,
                &[(&xb, 0, 0), (&ab, 1, 0), (&out, 2, 0)],
                &[(&p, 3)],
                (count as u64, 1, 1),
                (128, 1, 1),
            );
            pass.commit_and_wait_checked()?;
            finite_output(&out, count)
        })
    }
    fn unary(
        &self,
        name: &'static str,
        x: &[f32],
        p: &[u8],
        threads: usize,
        dtype: Music3DType,
    ) -> Result<Vec<f32>, GpuError> {
        autorelease_pool(|| {
            let mut c = self.state.context.borrow_mut();
            let pipeline = c.pipeline(
                source(dtype),
                name,
                &crate::rms_norm::unused_function_constants(),
                b"",
            )?;
            let xb = c.new_buffer_with_data(x);
            let out = c.new_output_buffer((x.len() * 4) as u64);
            let pass = c.begin_pass();
            pass.encode_threads_3d(
                &pipeline,
                &[(&xb, 0, 0), (&out, 1, 0)],
                &[(p, 2)],
                (threads as u64, 1, 1),
                (128, 1, 1),
            );
            pass.commit_and_wait_checked()?;
            finite_output(&out, x.len())
        })
    }
}
fn finite_output(out: &MetalBuffer, count: usize) -> Result<Vec<f32>, GpuError> {
    let values = read_f32_buffer(out, count);
    if values.iter().any(|v| !v.is_finite()) {
        return Err(bad("Music 3 device produced non-finite values"));
    }
    Ok(values)
}
impl Music3Weight {
    fn parameters(&self) -> Vec<u32> {
        let (mode, bits, group) = self.encoding.parameters();
        vec![mode, self.shape[1] as u32, bits, group as u32]
    }
    pub fn linear(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.linear_typed(input, bias, rows, input_dim, output_dim, Music3DType::F32)
    }
    pub fn linear_typed(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        dtype: Music3DType,
    ) -> Result<Vec<f32>, GpuError> {
        let count = product(&[rows, output_dim])?;
        if self.shape != [output_dim, input_dim]
            || input.len() != product(&[rows, input_dim])?
            || rows == 0
            || bias.is_some_and(|b| b.len() != output_dim)
        {
            return Err(bad("Music 3 linear geometry mismatch"));
        }
        let mut p = self.parameters();
        p.extend([
            output_dim as u32,
            bias.is_some() as u32,
            dtype as u32,
            rows as u32,
        ]);
        // MLX stores split-K partials in the activation dtype before reduction.
        let name = self.state.context.borrow().device().name().to_owned();
        let medium = input_dim <= 4096 && output_dim <= 4096;
        let small = input_dim <= 2048 && output_dim <= 2048;
        let limit = if name.contains("Ultra") {
            if small {
                32
            } else if medium {
                18
            } else {
                12
            }
        } else if name.contains("M5") {
            if small {
                33
            } else if medium {
                25
            } else {
                13
            }
        } else if name.contains("M3") || name.contains("M4") {
            if small {
                13
            } else if medium {
                15
            } else {
                13
            }
        } else {
            if small {
                14
            } else if medium {
                10
            } else {
                6
            }
        };
        let mut split = 1;
        if dtype != Music3DType::F32 && self.encoding.parameters().0 >= 3 && rows >= limit {
            let align = self.encoding.parameters().2.max(32);
            split = (512 / (rows.div_ceil(32) * output_dim.div_ceil(32)))
                .max(1)
                .min(input_dim / align);
            while split > 1 && input_dim % (split * align) != 0 {
                split -= 1;
            }
        }
        p.push(split as u32);
        p.push((rows >= limit) as u32);
        // MLX's wide vector kernel dequantizes in FP32. Its small-K quad
        // kernel instead uses typed four-element input sums.
        let quad = matches!(input_dim, 64 | 128) && self.encoding.parameters().1.is_power_of_two();
        let wide = self.encoding.parameters().0 >= 3
            && rows >= 2
            && !quad
            && (self.encoding.parameters().0 >= 4
                || ["M3", "M4", "M5"]
                    .iter()
                    .any(|generation| name.contains(generation)));
        p.push(wide as u32);
        let values_per_lane = if self.encoding.parameters().0 >= 4 && !quad {
            let pack = 32 / self.encoding.parameters().1;
            if output_dim % 8 == 0 && input_dim % (2 * pack as usize * 32) == 0 {
                2 * pack
            } else {
                pack
            }
        } else {
            0
        };
        p.push(values_per_lane);
        let mma = dtype != Music3DType::F32
            && rows > 1
            && (self.encoding.parameters().0 < 3 || rows >= limit);
        // The affine wide branch of `music3_linear`, repacked sixteen pairs
        // per threadgroup. Same conditions as that branch, plus the group
        // geometry the packed kernel assumes.
        let (mode, _, group) = self.encoding.parameters();
        let packed_wide = mode == 3
            && dtype != Music3DType::F32
            && rows < limit
            && wide
            && group % 8 == 0
            && input_dim % group == 0;
        let p = bytes(&p);
        autorelease_pool(|| {
            let mut c = self.state.context.borrow_mut();
            let pipeline = c.pipeline(
                source(dtype),
                if mma && rows <= 16 {
                    "music3_linear_tiled_16x64"
                } else if mma {
                    "music3_linear_tiled_32x32"
                } else if packed_wide {
                    "music3_linear_wide"
                } else {
                    "music3_linear"
                },
                &crate::rms_norm::unused_function_constants(),
                b"",
            )?;
            let x = c.new_buffer_with_data(input);
            let b = c.new_buffer_with_data(bias.unwrap_or(&[0.0]));
            let out = c.new_output_buffer((count * 4) as u64);
            let pass = c.begin_pass();
            pass.encode_threadgroups(
                &pipeline,
                &[
                    (&self.bytes, 0, 0),
                    (&self.scales, 1, 0),
                    (&self.offsets, 2, 0),
                    (&self.block_scales, 3, 0),
                    (&x, 4, 0),
                    (&b, 5, 0),
                    (&out, 6, 0),
                ],
                &[(&p, 7)],
                if mma && rows <= 16 {
                    (rows.div_ceil(16) * output_dim.div_ceil(64)) as u64
                } else if mma {
                    (rows.div_ceil(32) * output_dim.div_ceil(32)) as u64
                } else if packed_wide {
                    count.div_ceil(16) as u64
                } else {
                    count as u64
                },
                128,
            );
            pass.commit_and_wait_checked()?;
            finite_output(&out, count)
        })
    }
    pub fn embedding(&self, ids: &[i32], width: usize) -> Result<Vec<f32>, GpuError> {
        if self.shape.len() != 2
            || self.shape[1] != width
            || ids.is_empty()
            || ids.iter().any(|&i| i < 0 || i as usize >= self.shape[0])
        {
            return Err(bad("Music 3 embedding geometry mismatch"));
        }
        let count = product(&[ids.len(), width])?;
        let p = bytes(&self.parameters());
        autorelease_pool(|| {
            let mut c = self.state.context.borrow_mut();
            let pipeline = c.pipeline(
                SOURCE,
                "music3_embedding",
                &crate::rms_norm::unused_function_constants(),
                b"",
            )?;
            let ib = c.new_buffer_with_data(ids);
            let out = c.new_output_buffer((count * 4) as u64);
            let pass = c.begin_pass();
            pass.encode_threads_3d(
                &pipeline,
                &[
                    (&self.bytes, 0, 0),
                    (&self.scales, 1, 0),
                    (&self.offsets, 2, 0),
                    (&self.block_scales, 3, 0),
                    (&ib, 4, 0),
                    (&out, 5, 0),
                ],
                &[(&p, 6)],
                (count as u64, 1, 1),
                (128, 1, 1),
            );
            pass.commit_and_wait_checked()?;
            finite_output(&out, count)
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn convolution(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        ic: usize,
        oc: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        transpose: bool,
    ) -> Result<Vec<f32>, GpuError> {
        self.convolution_typed(
            input,
            bias,
            ic,
            oc,
            kernel,
            stride,
            pad,
            dilation,
            transpose,
            Music3DType::F32,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn convolution_typed(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        ic: usize,
        oc: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        transpose: bool,
        dtype: Music3DType,
    ) -> Result<Vec<f32>, GpuError> {
        let expected = if transpose {
            vec![ic, oc, kernel]
        } else {
            vec![oc, ic, kernel]
        };
        if !matches!(
            self.encoding,
            Music3Encoding::F32 | Music3Encoding::F16 | Music3Encoding::Bf16
        ) || self.shape != expected
            || ic == 0
            || stride == 0
            || dilation == 0
            || (transpose && dilation != 1)
            || input.is_empty()
            || input.len() % ic != 0
            || bias.is_some_and(|b| b.len() != oc)
        {
            return Err(bad("Music 3 convolution geometry mismatch"));
        }
        let len = input.len() / ic;
        let padding = pad
            .checked_mul(2)
            .ok_or_else(|| bad("Music 3 padding overflows"))?;
        let reach = kernel
            .checked_sub(1)
            .and_then(|n| n.checked_mul(dilation))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| bad("Music 3 kernel reach overflows"))?;
        let outlen = if transpose {
            (len - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(kernel))
                .and_then(|n| n.checked_sub(padding))
        } else {
            len.checked_add(padding)
                .and_then(|n| n.checked_sub(reach))
                .map(|n| n / stride + 1)
        }
        .filter(|&n| n > 0)
        .ok_or_else(|| bad("Music 3 convolution collapsed or overflowed"))?;
        let count = product(&[oc, outlen])?;
        let mma = dtype != Music3DType::F32;
        let p = bytes(&params(&[
            ic,
            oc,
            kernel,
            stride,
            pad,
            dilation,
            transpose as usize,
            len,
            outlen,
            bias.is_some() as usize,
            dtype as usize,
            self.encoding.parameters().0 as usize,
        ])?);
        autorelease_pool(|| {
            let mut c = self.state.context.borrow_mut();
            let pipeline = c.pipeline(
                source(dtype),
                if mma {
                    "music3_conv_tiled_32x32"
                } else {
                    "music3_conv"
                },
                &crate::rms_norm::unused_function_constants(),
                b"",
            )?;
            let x = c.new_buffer_with_data(input);
            let b = c.new_buffer_with_data(bias.unwrap_or(&[0.0]));
            let out = c.new_output_buffer((count * 4) as u64);
            let pass = c.begin_pass();
            pass.encode_threadgroups(
                &pipeline,
                &[(&self.bytes, 0, 0), (&x, 1, 0), (&b, 2, 0), (&out, 3, 0)],
                &[(&p, 4)],
                if mma {
                    (outlen.div_ceil(32) * oc.div_ceil(32)) as u64
                } else {
                    count as u64
                },
                128,
            );
            pass.commit_and_wait_checked()?;
            finite_output(&out, count)
        })
    }
}
