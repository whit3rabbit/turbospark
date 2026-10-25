//! Incremental decode for the Fish S1 DAC: codes in, samples out, a chunk
//! at a time, matching [`FishS1Dac::decode`] over the whole sequence bit for
//! bit on the host.
//!
//! The decode path has three kinds of stage, and each streams its own way:
//!
//! - **Code lookup** (`quantized_from_codes`) is pointwise per frame, so a
//!   chunk is decoded on its own.
//! - **The post transformer** attends over a window of past frames
//!   (`window_size`, 128 on the real codec). It runs incrementally: each
//!   block keeps the keys and values of the last `window - 1` frames, rotated
//!   at their absolute positions, and every new row attends over exactly the
//!   keys the batch mask admits, through the same `sdpa_strided_into` in the
//!   same order. Masked keys add exact zeros in the batch path, so the two
//!   agree bit for bit.
//! - **The upsample stack and the decoder** are causal stride-1 convolutions,
//!   transposed convolutions and pointwise work. Each causal conv carries its
//!   last `kernel_eff - 1` input columns between chunks (zeros at the start,
//!   the batch path's causal padding), and each transposed conv the
//!   `ceil(kernel / stride) - 1` input columns its next outputs still read.
//!   Every output column then sees exactly the inputs the batch path gives
//!   it, so a chunk costs what its own frames cost and nothing is decoded
//!   twice.

use super::*;

/// An incremental decoder over one [`FishS1Dac`] (see the module doc).
pub struct FishS1DacStream<'a> {
    dac: &'a FishS1Dac,
    /// Where the conv stack runs; `None` is `host`.
    exec: Option<&'a DacExec>,
    host: DacExec,
    /// Per post block: keys and values per head, rows of `head_dim`.
    keys: Vec<Vec<Vec<f32>>>,
    values: Vec<Vec<Vec<f32>>>,
    /// Absolute position of the first cached row.
    cache_start: usize,
    /// Absolute position of the next frame.
    position: usize,
    convs: ConvState,
}

/// Input columns each streamed convolution carries, channel-major.
struct ConvState {
    /// Per upsample stage: the transposed conv's, then the depthwise conv's.
    upsample: Vec<(Vec<f32>, Vec<f32>)>,
    decoder_in: Vec<f32>,
    /// Per decoder block: the transposed conv's, then each residual unit's
    /// dilated conv's.
    blocks: Vec<(Vec<f32>, [Vec<f32>; 3])>,
    decoder_out: Vec<f32>,
}

impl FishCausalConv {
    /// The causal padding as carried state.
    fn stream_state(&self) -> Vec<f32> {
        vec![0.0; self.conv.in_ch * self.pad]
    }

    /// The next `n` output columns: `[state | x]` through the conv without
    /// padding, then the last `pad` columns become the state.
    fn forward_stream(
        &self,
        exec: &DacExec,
        state: &mut [f32],
        x: &[f32],
        n: usize,
    ) -> Result<Vec<f32>> {
        if self.conv.stride != 1 {
            return Err(SpeechError::Unsupported {
                why: "streamed Fish S1 DAC convolutions must have stride 1".into(),
            });
        }
        let (ch, pad) = (self.conv.in_ch, self.pad);
        let len = pad + n;
        let mut joined = vec![0.0f32; ch * len];
        for c in 0..ch {
            let row = &mut joined[c * len..][..len];
            row[..pad].copy_from_slice(&state[c * pad..][..pad]);
            row[pad..].copy_from_slice(&x[c * n..][..n]);
            state[c * pad..][..pad].copy_from_slice(&row[n..]);
        }
        exec.conv(&self.conv, &joined)
    }
}

impl FishCausalConvTr {
    /// Earlier input columns the next outputs still read: output `t` reads
    /// inputs `i` with `i * stride <= t < i * stride + kernel`.
    fn stream_context(&self) -> usize {
        self.conv.kernel.div_ceil(self.conv.stride) - 1
    }

    /// The `n * stride` output columns of `n` new inputs. `state` holds the
    /// carried input columns (none at the start).
    fn forward_stream(
        &self,
        exec: &DacExec,
        state: &mut Vec<f32>,
        x: &[f32],
        n: usize,
    ) -> Result<Vec<f32>> {
        let c = &self.conv;
        let (ch, stride) = (c.in_ch, c.stride);
        let have = state.len() / ch;
        let len = have + n;
        let mut joined = vec![0.0f32; ch * len];
        for i in 0..ch {
            joined[i * len..][..have].copy_from_slice(&state[i * have..][..have]);
            joined[i * len + have..][..n].copy_from_slice(&x[i * n..][..n]);
        }
        let y = exec.conv_transpose(c, &joined)?;
        let out_len = (len - 1) * stride + c.kernel;
        let (skip, keep) = (have * stride, n * stride);
        let mut out = vec![0.0f32; c.out_ch * keep];
        for o in 0..c.out_ch {
            out[o * keep..][..keep].copy_from_slice(&y[o * out_len + skip..][..keep]);
        }
        let carry = self.stream_context().min(len);
        *state = (0..ch)
            .flat_map(|i| joined[i * len + len - carry..][..carry].iter().copied())
            .collect();
        Ok(out)
    }
}

impl FishConvNeXtBlock {
    fn forward_stream(
        &self,
        exec: &DacExec,
        state: &mut [f32],
        x: &mut Vec<f32>,
        n: usize,
    ) -> Result<()> {
        let h = self.dwconv.forward_stream(exec, state, x, n)?;
        self.after_dwconv(exec, x, h, n)
    }
}

impl FishResidualUnit {
    fn forward_stream(
        &self,
        exec: &DacExec,
        state: &mut [f32],
        x: &[f32],
        n: usize,
    ) -> Result<Vec<f32>> {
        let mut y = x.to_vec();
        exec.snake(&mut y, &self.alpha1, self.channels, n)?;
        let mut y = self.conv1.forward_stream(exec, state, &y, n)?;
        exec.snake(&mut y, &self.alpha2, self.channels, n)?;
        // A 1x1 conv carries nothing.
        let y = self.conv2.forward(exec, &y, n)?;
        Ok(add_cropped_residual(x, n, y, self.channels, 0))
    }
}

impl FishS1Dac {
    /// An incremental decoder on the host.
    pub fn stream(&self) -> FishS1DacStream<'_> {
        self.stream_inner(None)
    }

    /// [`Self::stream`] with the conv stack on `exec`. The post transformer
    /// runs on the host either way (a few rows per chunk).
    pub fn stream_on<'a>(&'a self, exec: &'a DacExec) -> FishS1DacStream<'a> {
        self.stream_inner(Some(exec))
    }

    fn stream_inner<'a>(&'a self, exec: Option<&'a DacExec>) -> FishS1DacStream<'a> {
        let blocks = &self.quantizer.post.blocks;
        let heads = |b: &FishBlock| vec![Vec::new(); b.n_local];
        let convs = ConvState {
            upsample: self
                .quantizer
                .upsample
                .iter()
                .map(|(_, block)| (Vec::new(), block.dwconv.stream_state()))
                .collect(),
            decoder_in: self.decoder_conv_in.stream_state(),
            blocks: self
                .decoder_blocks
                .iter()
                .map(|b| {
                    (
                        Vec::new(),
                        b.res_units.each_ref().map(|r| r.conv1.stream_state()),
                    )
                })
                .collect(),
            decoder_out: self.decoder_conv_out.stream_state(),
        };
        FishS1DacStream {
            dac: self,
            exec,
            host: DacExec::host(),
            keys: blocks.iter().map(heads).collect(),
            values: blocks.iter().map(heads).collect(),
            cache_start: 0,
            position: 0,
            convs,
        }
    }
}

impl FishS1DacStream<'_> {
    /// Frames decoded so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Decodes the next frames (`codes[book][frame]`, every book the same
    /// length) and returns their samples.
    pub fn push(&mut self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let books = self.dac.config.total_books();
        if codes.len() != books {
            return Err(SpeechError::Input {
                why: format!(
                    "expected {books} books (semantic + acoustic), got {}",
                    codes.len()
                ),
            });
        }
        let n = codes[0].len();
        if codes.iter().any(|b| b.len() != n) {
            return Err(SpeechError::Input {
                why: "codebooks differ in length".into(),
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let dac = self.dac;
        let quantizer = &dac.quantizer;
        let in_dim = quantizer.semantic.input_dim;
        let z = quantizer.quantized_from_codes(codes, n)?;
        let mut rows = vec![0.0f32; n * in_dim];
        for t in 0..n {
            for d in 0..in_dim {
                rows[t * in_dim + d] = z[d * n + t];
            }
        }
        let post = self.post_incremental(&rows, n);
        let mut z = vec![0.0f32; n * in_dim];
        for t in 0..n {
            for d in 0..in_dim {
                z[d * n + t] = post[t * in_dim + d];
            }
        }

        let exec = self.exec.unwrap_or(&self.host);
        let convs = &mut self.convs;
        for ((conv, block), (tr, dw)) in quantizer.upsample.iter().zip(&mut convs.upsample) {
            let seq = z.len() / conv.conv.in_ch;
            z = conv.forward_stream(exec, tr, &z, seq)?;
            let seq = z.len() / conv.conv.out_ch;
            block.forward_stream(exec, dw, &mut z, seq)?;
        }
        let seq = z.len() / in_dim;
        let mut h = dac
            .decoder_conv_in
            .forward_stream(exec, &mut convs.decoder_in, &z, seq)?;
        for (block, (tr, units)) in dac.decoder_blocks.iter().zip(&mut convs.blocks) {
            let seq = h.len() / block.in_ch;
            exec.snake(&mut h, &block.alpha, block.in_ch, seq)?;
            h = block.conv_tr.forward_stream(exec, tr, &h, seq)?;
            let out_ch = block.conv_tr.conv.out_ch;
            for (unit, state) in block.res_units.iter().zip(units.iter_mut()) {
                let seq = h.len() / out_ch;
                h = unit.forward_stream(exec, state, &h, seq)?;
            }
        }
        let final_dim = dac.config.decoder_dim >> dac.config.decoder_rates.len();
        let seq = h.len() / final_dim;
        exec.snake(&mut h, &dac.decoder_alpha, final_dim, seq)?;
        let mut audio =
            dac.decoder_conv_out
                .forward_stream(exec, &mut convs.decoder_out, &h, seq)?;
        for v in &mut audio {
            *v = v.tanh();
        }
        if audio.len() != n * dac.config.frame_length() {
            return Err(SpeechError::Input {
                why: format!(
                    "{n} frames decoded to {} samples, not {}",
                    audio.len(),
                    n * dac.config.frame_length()
                ),
            });
        }
        self.position += n;
        Ok(audio)
    }

    /// The post transformer over `n` new rows, attending over the cache.
    fn post_incremental(&mut self, rows: &[f32], n: usize) -> Vec<f32> {
        let transformer = &self.dac.quantizer.post;
        let mut h = match &transformer.input_proj {
            Some((i, o, w, b)) => ops::linear(rows, w, Some(b), n, *i, *o),
            None => rows.to_vec(),
        };
        let dim = transformer.dim;
        let hd = transformer.head_dim;
        let window = transformer.window_size.unwrap_or(usize::MAX);
        let (cos, sin) = ops::rope_tables_range(self.position, n, hd, transformer.rope_base);
        for (index, block) in transformer.blocks.iter().enumerate() {
            let mut normed = h.clone();
            block.attention_norm.apply(&mut normed, n, dim);
            let qkv = ops::linear(&normed, &block.wqkv, None, n, dim, 3 * dim);
            let kv_size = block.n_local * hd;
            let mut qh = ops::split_heads_strided(&qkv, 3 * dim, 0, n, block.n_head, hd);
            let mut kh = ops::split_heads_strided(&qkv, 3 * dim, kv_size, n, block.n_local, hd);
            let vh = ops::split_heads_strided(&qkv, 3 * dim, 2 * kv_size, n, block.n_local, hd);
            ops::rope_interleaved(&mut qh, block.n_head, n, hd, &cos, &sin);
            ops::rope_interleaved(&mut kh, block.n_local, n, hd, &cos, &sin);
            for head in 0..block.n_local {
                self.keys[index][head].extend_from_slice(&kh[head * n * hd..][..n * hd]);
                self.values[index][head].extend_from_slice(&vh[head * n * hd..][..n * hd]);
            }

            let scale = (hd as f32).powf(-0.5);
            let group = block.n_head / block.n_local;
            let mut out = vec![0.0f32; block.n_head * n * hd];
            for i in 0..n {
                let p = self.position + i;
                // The batch mask admits j with j <= p and p < j + window.
                let first = (p + 1).saturating_sub(window).max(self.cache_start);
                let count = p + 1 - first;
                let base = (first - self.cache_start) * hd;
                for head in 0..block.n_head {
                    let kv = head / group;
                    ops::sdpa_strided_into(
                        &mut out[(head * n + i) * hd..][..hd],
                        &qh[(head * n + i) * hd..][..hd],
                        &self.keys[index][kv],
                        base,
                        hd,
                        &self.values[index][kv],
                        base,
                        hd,
                        None,
                        1,
                        count,
                        hd,
                        hd,
                        scale,
                    );
                }
            }
            let merged = ops::merge_heads(&out, n, block.n_head, hd);
            let attn = ops::linear(&merged, &block.wo, None, n, dim, dim);
            for (x, a) in h.chunks_exact_mut(dim).zip(attn.chunks_exact(dim)) {
                for (d, v) in x.iter_mut().enumerate() {
                    *v += a[d] * block.attention_gamma[d];
                }
            }
            let mut normed = h.clone();
            block.ffn_norm.apply(&mut normed, n, dim);
            let h1 = ops::linear(&normed, &block.w1, None, n, dim, block.ffn_dim);
            let h3 = ops::linear(&normed, &block.w3, None, n, dim, block.ffn_dim);
            let mut gated = vec![0.0f32; h1.len()];
            for ((g, &a), &b) in gated.iter_mut().zip(&h1).zip(&h3) {
                let mut tmp = [a];
                ops::silu(&mut tmp);
                *g = tmp[0] * b;
            }
            let ffn = ops::linear(&gated, &block.w2, None, n, block.ffn_dim, dim);
            for (x, f) in h.chunks_exact_mut(dim).zip(ffn.chunks_exact(dim)) {
                for (d, v) in x.iter_mut().enumerate() {
                    *v += f[d] * block.ffn_gamma[d];
                }
            }
        }
        // Keep only the rows a future frame can still attend to.
        let end = self.position + n;
        let keep_from = (end + 1).saturating_sub(window).max(self.cache_start);
        let drop = (keep_from - self.cache_start) * hd;
        if drop > 0 {
            for cache in self.keys.iter_mut().chain(self.values.iter_mut()) {
                for head in cache.iter_mut() {
                    head.drain(..drop);
                }
            }
            self.cache_start = keep_from;
        }
        transformer.final_norm.apply(&mut h, n, dim);
        match &transformer.output_proj {
            Some((i, o, w, b)) => ops::linear(&h, w, Some(b), n, *i, *o),
            None => h,
        }
    }
}
