//! oMLX's oQ layout of Inkling 2 (`mlx-community/Inkling-2-Mini-oQ4e-mtp`),
//! presented under the HF names by inverting oQ's sanitize
//! (`omlx/patches/inkling2/oq_layout.py`). It differs from mlx-lm's layout:
//! the attention projections are stacked into `in_proj` (or `q_proj`,
//! `k_proj`, `v_proj` on the static-conv layers), the loop controllers are
//! transposed and merged, and the routed experts of the MXFP8 layers are
//! requantized to affine 4-bit.
//!
//! Affine rows are quantized per group of 64 columns, so slicing a stacked
//! projection by rows is exact. An affine expert row is repacked as GGUF
//! Q4_1 (`w = q * d + m` per 32), which holds the same values: nothing is
//! requantized, and a BF16 scale or bias that FP16 cannot hold is refused.

use compute::quant_1bit::{f16_to_f32, f32_to_f16};

use super::{Inkling2MlxSource, Mapped, MLX_NVFP4_GLOBAL_FACTOR};
use crate::inkling2_checkpoint::Inkling2RepackError;

fn err<T>(msg: impl Into<String>) -> Result<T, Inkling2RepackError> {
    Err(Inkling2RepackError(msg.into()))
}

/// Order of the loop message inputs in oQ's merged controller tensors.
const LOOP_INPUTS: [&str; 4] = ["attn", "k", "v", "mlp"];
/// Columns one affine scale covers.
const AFFINE_GROUP: usize = 64;
/// Elements and bytes of one Q4_1 block: f16 `d`, f16 `m`, 16 nibble bytes.
const Q4_1_ELEMS: usize = 32;
pub(crate) const Q4_1_BLOCK_BYTES: usize = 20;

/// A BF16 value as FP16 bits, refused unless exact.
fn bf16_as_f16(bytes: &[u8]) -> Result<[u8; 2], Inkling2RepackError> {
    let v = f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
    let h = f32_to_f16(v);
    if f16_to_f32(h).to_bits() != v.to_bits() {
        return err(format!("affine scale or bias {v:e} is not exact in FP16"));
    }
    Ok(h.to_le_bytes())
}

/// One MLX affine 4-bit row (`cols` values in u32 words, element `i` in
/// nibble `i`, BF16 scale and bias per 64) as Q4_1 blocks.
pub(crate) fn affine4_row_to_q4_1(
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
    cols: usize,
) -> Result<Vec<u8>, Inkling2RepackError> {
    if cols % AFFINE_GROUP != 0
        || packed.len() != cols / 2
        || scales.len() != cols / AFFINE_GROUP * 2
        || biases.len() != scales.len()
    {
        return err(format!(
            "affine 4-bit row of {cols} columns has wrong sizes"
        ));
    }
    let nibble = |i: usize| (packed[i / 2] >> (4 * (i % 2))) & 0xF;
    let mut out = Vec::with_capacity(cols / Q4_1_ELEMS * Q4_1_BLOCK_BYTES);
    for block in 0..cols / Q4_1_ELEMS {
        let first = block * Q4_1_ELEMS;
        let g = first / AFFINE_GROUP * 2;
        out.extend_from_slice(&bf16_as_f16(&scales[g..g + 2])?);
        out.extend_from_slice(&bf16_as_f16(&biases[g..g + 2])?);
        // ggml's Q4_1 pairs element j (low nibble) with j + 16 (high).
        for j in 0..16 {
            out.push(nibble(first + j) | (nibble(first + j + 16) << 4));
        }
    }
    Ok(out)
}

/// Whether the source is oQ's layout rather than mlx-lm's.
pub(super) fn is_oq(src: &Inkling2MlxSource) -> bool {
    src.at.contains_key("model.layers.0.attn.o_proj.weight")
}

impl Inkling2MlxSource {
    /// Takes `mlx` as a BF16/F32 tensor under the HF name `hf`.
    fn oq_plain(
        &mut self,
        hf: String,
        mlx: &str,
        shape: Option<Vec<usize>>,
        narrow: bool,
    ) -> Result<(), Inkling2RepackError> {
        if self.desc(mlx).is_none() {
            return err(format!("MLX tensor {mlx} missing"));
        }
        self.map.insert(
            hf,
            Mapped::Plain {
                mlx: mlx.into(),
                dtype: None,
                shape,
                narrow,
            },
        );
        Ok(())
    }

    /// The affine tensor `base`'s row count and bit width.
    fn oq_affine_shape(&mut self, base: &str) -> Result<(usize, u8), Inkling2RepackError> {
        let (w, s) = (format!("{base}.weight"), format!("{base}.scales"));
        let (Some(w), Some(s)) = (self.desc(&w), self.desc(&s)) else {
            return err(format!("{base}: expected an affine weight and scales"));
        };
        let (words, groups) = (*w.shape.last().unwrap_or(&0), *s.shape.last().unwrap_or(&0));
        if w.dtype != "U32" || s.dtype != "BF16" || groups == 0 {
            return err(format!(
                "{base}: expected U32 weights and BF16 scales, got {} and {}",
                w.dtype, s.dtype
            ));
        }
        let rows = w.shape[..w.shape.len() - 1].iter().product();
        let bits = (words * 32 / (groups * AFFINE_GROUP)) as u8;
        Ok((rows, bits))
    }

    /// Splits a stacked affine projection into HF matrices by row count.
    fn oq_split(
        &mut self,
        base: &str,
        parts: &[(String, usize)],
    ) -> Result<(), Inkling2RepackError> {
        let (rows, bits) = self.oq_affine_shape(base)?;
        let want: usize = parts.iter().map(|p| p.1).sum();
        if rows != want {
            return err(format!("{base}: {rows} rows, the split needs {want}"));
        }
        self.affine_bits.insert(bits);
        let mut at = 0;
        for (hf, n) in parts {
            self.map.insert(
                hf.clone(),
                Mapped::Affine {
                    base: base.into(),
                    bits,
                    rows: Some((at, *n)),
                },
            );
            at += n;
        }
        Ok(())
    }

    fn oq_synth(&mut self, hf: String, dtype: &'static str, shape: Vec<usize>, bytes: Vec<u8>) {
        self.map.insert(
            hf,
            Mapped::Synth {
                dtype,
                shape,
                bytes,
            },
        );
    }

    pub(super) fn build_oq_map(
        &mut self,
        layers: usize,
        core: usize,
        hidden: usize,
    ) -> Result<(), Inkling2RepackError> {
        for (hf, mlx) in [
            ("model.llm.embed_norm.weight", "model.embed_norm.weight"),
            ("model.llm.norm.weight", "model.norm.weight"),
        ] {
            self.oq_plain(hf.into(), mlx, None, false)?;
        }
        for (hf, base) in [
            ("model.llm.embed.weight", "model.embed"),
            ("model.llm.unembed.weight", "lm_head"),
        ] {
            let (rows, _) = self.oq_affine_shape(base)?;
            self.oq_split(base, &[(hf.into(), rows)])?;
        }
        self.oq_plain(
            "model.llm.attn_res.output_query".into(),
            "model.final_input.query",
            Some(vec![hidden]),
            false,
        )?;
        let core_start = self.oq_core_start(layers, core)?;
        for l in 0..layers {
            self.oq_layer(l, hidden)?;
            if (core_start..core_start + core).contains(&l) {
                self.oq_controllers(l, hidden)?;
            }
        }
        self.oq_message_queries(core_start, core, hidden)
    }

    fn oq_core_start(&self, layers: usize, core: usize) -> Result<usize, Inkling2RepackError> {
        let start = (0..layers)
            .find(|l| {
                self.at
                    .contains_key(&format!("model.layers.{l}.loop_message.query"))
            })
            .ok_or_else(|| Inkling2RepackError("no loop_message.query tensor".into()))?;
        if start + core > layers {
            return err(format!("loop core at {start} runs past {layers} layers"));
        }
        Ok(start)
    }

    fn oq_layer(&mut self, l: usize, hidden: usize) -> Result<(), Inkling2RepackError> {
        let m = format!("model.layers.{l}");
        let h = format!("model.llm.layers.{l}");
        let hf = |s: &str| format!("{h}.{s}");
        let mx = |s: &str| format!("{m}.{s}");
        let (o_rows, _) = self.oq_affine_shape(&mx("attn.o_proj"))?;
        self.oq_split(&mx("attn.o_proj"), &[(hf("attn.wo_ud.weight"), o_rows)])?;
        // HF row counts of the stacked projections, fixed by the Mini's heads
        // (checked: they must add up to each stacked tensor's rows).
        let (q, k, r, gate, sink) = (3072, 1024, 384, 3072, 24);
        let static_layer = !self.at.contains_key(&mx("attn.in_proj.weight"));
        if static_layer {
            self.oq_split(
                &mx("attn.q_proj"),
                &[
                    (hf("attn.wq_du.weight"), q),
                    (hf("attn.wr_du.weight"), r),
                    (hf("attn.gate_attn.weight"), gate),
                    (hf("attn.attn_sink_proj.weight"), sink),
                ],
            )?;
            for (n, w) in [("k", "wk_dv"), ("v", "wv_dv")] {
                let base = mx(&format!("attn.{n}_proj"));
                let (rows, _) = self.oq_affine_shape(&base)?;
                self.oq_split(
                    &base,
                    &[
                        (hf(&format!("attn.{w}.weight")), k),
                        (
                            hf(&format!("attn.{n}_sconv.weight_proj")),
                            rows.saturating_sub(k),
                        ),
                    ],
                )?;
                self.oq_plain(
                    hf(&format!("attn.{n}_sconv.static_w")),
                    &mx(&format!("attn.{n}_static")),
                    None,
                    false,
                )?;
            }
            for n in ["q", "k", "v"] {
                self.oq_plain(
                    hf(&format!("{n}_input_norm.weight")),
                    &mx(&format!("{n}_input_norm.weight")),
                    None,
                    false,
                )?;
            }
        } else {
            self.oq_split(
                &mx("attn.in_proj"),
                &[
                    (hf("attn.wq_du.weight"), q),
                    (hf("attn.wk_dv.weight"), k),
                    (hf("attn.wv_dv.weight"), k),
                    (hf("attn.wr_du.weight"), r),
                    (hf("attn.gate_attn.weight"), gate),
                    (hf("attn.attn_sink_proj.weight"), sink),
                ],
            )?;
            for n in ["k", "v"] {
                let conv = mx(&format!("attn.{n}_conv"));
                let shape = self.oq_conv_shape(&conv)?;
                self.oq_plain(
                    hf(&format!("attn.{n}_sconv.weight")),
                    &conv,
                    Some(shape),
                    false,
                )?;
            }
            self.oq_plain(hf("attn_norm.weight"), &mx("attn_norm.weight"), None, false)?;
        }
        for (to, from) in [
            ("attn.q_l2_clip.scale", "attn.q_scale"),
            ("attn.rel_l2_clip.scale", "attn.r_scale"),
            ("attn.rel_logits_proj.proj", "attn.rel_proj"),
            ("mlp_norm.weight", "mlp_norm.weight"),
            ("mlp.gate.bias", "mlp.router_bias"),
        ] {
            self.oq_plain(hf(to), &mx(from), None, false)?;
        }
        // oQ widens the router to F32; the intake expects the HF BF16.
        self.oq_plain(hf("mlp.gate.weight"), &mx("mlp.router_weight"), None, true)?;
        for (to, from) in [
            ("attn_sconv.weight", "attn_conv"),
            ("mlp_sconv.weight", "mlp_conv"),
        ] {
            let shape = self.oq_conv_shape(&mx(from))?;
            self.oq_plain(hf(to), &mx(from), Some(shape), false)?;
        }
        if l > 0 {
            self.oq_query(
                format!("model.llm.attn_res.attn_queries.{}", l - 1),
                &mx("attn_input.query"),
                hidden,
            )?;
        }
        self.oq_query(
            format!("model.llm.attn_res.mlp_queries.{l}"),
            &mx("mlp_input.query"),
            hidden,
        )?;
        self.oq_experts(l)
    }

    /// `[C, taps]` as the HF `[C, 1, taps]`.
    fn oq_conv_shape(&self, mlx: &str) -> Result<Vec<usize>, Inkling2RepackError> {
        match self.desc(mlx).map(|d| d.shape.as_slice()) {
            Some([c, taps]) => Ok(vec![*c, 1, *taps]),
            other => err(format!("{mlx}: expected [C, taps], got {other:?}")),
        }
    }

    /// An attention-residual query `[n, H]`: `[H]` in HF when `n` is 1.
    fn oq_query(
        &mut self,
        hf: String,
        mlx: &str,
        hidden: usize,
    ) -> Result<(), Inkling2RepackError> {
        let shape = match self.desc(mlx).map(|d| d.shape.as_slice()) {
            Some([1, h]) if *h == hidden => vec![hidden],
            Some([n, h]) if *h == hidden => vec![*n, hidden],
            other => return err(format!("{mlx}: expected [n, {hidden}], got {other:?}")),
        };
        self.oq_plain(hf, mlx, Some(shape), false)
    }

    fn oq_experts(&mut self, l: usize) -> Result<(), Inkling2RepackError> {
        let m = format!("model.layers.{l}.mlp.experts");
        let h = format!("model.llm.layers.{l}.mlp.experts");
        for p in ["w13", "w2"] {
            let base = format!("{m}.{p}");
            let (w, s, g) = (
                format!("{base}.weight"),
                format!("{base}.scales"),
                format!("{base}.global_scale"),
            );
            let (Some(wd), Some(sd)) = (self.desc(&w).cloned(), self.desc(&s).cloned()) else {
                return err(format!("{base}: expert weight or scales missing"));
            };
            if wd.dtype != "U32" || wd.shape.len() != 3 {
                return err(format!(
                    "{w}: expected 3-D U32, got {} {:?}",
                    wd.dtype, wd.shape
                ));
            }
            let nvfp4 = self.at.contains_key(&g);
            if !nvfp4 {
                if sd.dtype != "BF16" {
                    return err(format!(
                        "{s}: an expert layer without global_scale must be affine 4-bit"
                    ));
                }
                let (_, bits) = self.oq_affine_shape(&base)?;
                if bits != 4 {
                    return err(format!("{base}: affine {bits}-bit experts have no kernel"));
                }
                // Not counted in `affine_bits`, which names the dense width.
                self.map.insert(
                    format!("{h}.{p}_weight"),
                    Mapped::Affine {
                        base,
                        bits,
                        rows: None,
                    },
                );
                continue;
            }
            let row_bytes = wd.shape[2] * 4;
            self.map.insert(
                format!("{h}.{p}_weight"),
                Mapped::Plain {
                    mlx: w,
                    dtype: Some("U8"),
                    shape: Some(vec![wd.shape[0], wd.shape[1], row_bytes]),
                    narrow: false,
                },
            );
            self.map.insert(
                format!("{h}.{p}_weight_scale"),
                Mapped::Plain {
                    mlx: s,
                    dtype: Some("F8_E4M3"),
                    shape: None,
                    narrow: false,
                },
            );
            self.oq_plain(format!("{h}.{p}_weight_amax"), &g, None, false)?;
            self.oq_synth(
                format!("{h}.{p}_weight_outer_scale_denominator"),
                "F32",
                vec![1],
                MLX_NVFP4_GLOBAL_FACTOR.to_le_bytes().to_vec(),
            );
        }
        Ok(())
    }

    fn oq_f32(&self, mlx: &str, shape: &[usize]) -> Result<Vec<u8>, Inkling2RepackError> {
        let d = self
            .desc(mlx)
            .ok_or_else(|| Inkling2RepackError(format!("MLX tensor {mlx} missing")))?;
        if d.dtype != "F32" || d.shape != shape {
            return err(format!(
                "{mlx}: expected F32 {shape:?}, got {} {:?}",
                d.dtype, d.shape
            ));
        }
        Ok(self.bytes(mlx)?.to_vec())
    }

    /// `loop_down [4, 2, H, r]` back to HF `down [r, 2H]`, and
    /// `loop_gain_forget [4, r, 2H]` back to `gain [H, r]`, `forget [H, r]`.
    fn oq_controllers(&mut self, l: usize, hidden: usize) -> Result<(), Inkling2RepackError> {
        let down_name = format!("model.layers.{l}.loop_down");
        let rank = match self.desc(&down_name).map(|d| d.shape.as_slice()) {
            Some([4, 2, h, r]) if *h == hidden => *r,
            other => return err(format!("{down_name}: unexpected shape {other:?}")),
        };
        let down = self.oq_f32(&down_name, &[4, 2, hidden, rank])?;
        let gf = self.oq_f32(
            &format!("model.layers.{l}.loop_gain_forget"),
            &[4, rank, 2 * hidden],
        )?;
        for (i, input) in LOOP_INPUTS.iter().enumerate() {
            let at = format!("model.llm.layers.{l}.controllers.{input}");
            let mut d = Vec::with_capacity(rank * 2 * hidden * 4);
            for r in 0..rank {
                for part in 0..2 {
                    for j in 0..hidden {
                        let e = ((i * 2 + part) * hidden + j) * rank + r;
                        d.extend_from_slice(&down[e * 4..e * 4 + 4]);
                    }
                }
            }
            self.oq_synth(
                format!("{at}.down_weight"),
                "BF16",
                vec![rank, 2 * hidden],
                super::narrow_f32(&down_name, &d)?,
            );
            for (half, name) in ["gain_weight", "forget_weight"].iter().enumerate() {
                let mut v = Vec::with_capacity(hidden * rank * 4);
                for j in 0..hidden {
                    for r in 0..rank {
                        let e = (i * rank + r) * 2 * hidden + half * hidden + j;
                        v.extend_from_slice(&gf[e * 4..e * 4 + 4]);
                    }
                }
                self.oq_synth(
                    format!("{at}.{name}"),
                    "BF16",
                    vec![hidden, rank],
                    super::narrow_f32(name, &v)?,
                );
            }
        }
        Ok(())
    }

    /// Rows `[attn, k, v, mlp]` per core layer back to the two HF tables.
    fn oq_message_queries(
        &mut self,
        core_start: usize,
        core: usize,
        hidden: usize,
    ) -> Result<(), Inkling2RepackError> {
        let (mut q, mut kv) = (Vec::new(), Vec::new());
        for i in 0..core {
            let name = format!("model.layers.{}.loop_message.query", core_start + i);
            let b = self.oq_f32(&name, &[4, hidden])?;
            let row = |j: usize| &b[j * hidden * 4..(j + 1) * hidden * 4];
            q.extend_from_slice(row(0));
            q.extend_from_slice(row(3));
            kv.extend_from_slice(row(1));
            kv.extend_from_slice(row(2));
        }
        for (name, bytes) in [
            ("model.llm.middle_cycle_cross_pass_queries", q),
            ("model.llm.middle_cycle_cross_pass_kv_queries", kv),
        ] {
            self.oq_synth(name.into(), "F32", vec![2 * core, hidden], bytes);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The repack must hold exactly MLX's values: decode each Q4_1 element
    /// the way the kernel does and compare with `q * scale + bias`.
    #[test]
    fn a_q4_1_repack_holds_the_affine_values() {
        let cols = 128;
        let q: Vec<u8> = (0..cols).map(|i| ((i * 7 + i / 32) % 16) as u8).collect();
        let packed: Vec<u8> = q.chunks(2).map(|p| p[0] | (p[1] << 4)).collect();
        let bf = |v: f32| ((v.to_bits() >> 16) as u16).to_le_bytes();
        let scales = [bf(0.5), bf(0.015625)].concat();
        let biases = [bf(-2.0), bf(0.25)].concat();
        let out = affine4_row_to_q4_1(&packed, &scales, &biases, cols).unwrap();
        assert_eq!(out.len(), cols / 32 * Q4_1_BLOCK_BYTES);
        for i in 0..cols {
            let block = &out[i / 32 * Q4_1_BLOCK_BYTES..][..Q4_1_BLOCK_BYTES];
            let e = i % 32;
            let byte = block[4 + (e & 15)];
            let n = if e < 16 { byte & 0xF } else { byte >> 4 };
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (s, b) = [(0.5, -2.0), (0.015625, 0.25)][i / 64];
            assert_eq!(f32::from(n) * d + m, f32::from(q[i]) * s + b, "column {i}");
        }
    }

    #[test]
    fn a_scale_fp16_cannot_hold_is_refused() {
        let tiny = ((1.0e-9f32.to_bits() >> 16) as u16).to_le_bytes();
        let zero = [0u8; 2];
        assert!(affine4_row_to_q4_1(&[0; 32], &tiny, &zero, 64).is_err());
    }
}
