//! IQ4_NL and IQ4_XS encoders, ported from ggml's `quantize_row_iq4_nl_impl`
//! (one function serves both: a 32-element block for NL, eight sub-blocks of
//! 32 under a 6-bit scale for XS). No grid tables are involved, only the
//! non-linear 16-entry value table.
//!
//! `ggml_quantize_chunk` always runs this with `ntry = 7`, with or without an
//! importance matrix (without one the weights are `x * x`). The older
//! `ntry = -1` reference path is not what `llama-quantize` uses, so it is not
//! ported.

// Index-based loops mirror ggml's C one for one so the port can be diffed
// against ggml-quants.c; iterator rewrites would hide a transcription error.
#![allow(clippy::needless_range_loop, clippy::needless_late_init)]

use foundation::LogitValue as F16;

const VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
const GROUP_MAX_EPS: f32 = 1e-15;
const NTRY: i32 = 7;

fn nearest_int(v: f32) -> i32 {
    let bits = (v + 12582912.0).to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

fn best_index(mid: &[f32; 15], x: f32) -> u8 {
    let mut l = 15i32;
    for m in mid {
        l -= i32::from(x < *m);
    }
    l as u8
}

/// Encode one super block of `sb` elements (32 for NL, 256 for XS).
fn block(sb: usize, x: &[f32], qw: Option<&[f32]>, out: &mut Vec<u8>) {
    let mut sigma2 = 0f32;
    for v in &x[..sb] {
        sigma2 = (*v).mul_add(*v, sigma2);
    }
    sigma2 *= 2.0 / sb as f32;

    let mut mid = [0f32; 15];
    for k in 0..15 {
        mid[k] = 0.5 * (i32::from(VALUES[k]) + i32::from(VALUES[k + 1])) as f32;
    }
    let fval: [f32; 16] = std::array::from_fn(|k| f32::from(VALUES[k]));

    let nblk = sb / 32;
    let mut l = [0u8; 256];
    let mut scales = [0f32; 8];
    let (mut max_scale, mut amax_scale) = (0f32, 0f32);
    let mut weight = [0f32; 32];
    for ib in 0..nblk {
        let xb = &x[ib * 32..ib * 32 + 32];
        match qw {
            Some(q) => {
                for j in 0..32 {
                    weight[j] = q[ib * 32 + j] * (xb[j]).mul_add(xb[j], sigma2).sqrt();
                }
            }
            None => {
                for j in 0..32 {
                    weight[j] = xb[j] * xb[j];
                }
            }
        }
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        let v0 = f32::from(VALUES[0]);
        let mut d = -max / v0; // ntry > 0
                               // Lane 0 is the initial guess, lanes 1..nt are itry = -ntry..ntry.
        let nt = (2 * NTRY + 2) as usize;
        let (mut idt, mut sqx, mut sq2) = ([0f32; 16], [0f32; 16], [0f32; 16]);
        idt[0] = 1.0 / d;
        for t in 1..nt {
            idt[t] = ((t as i32 - 1 - NTRY + i32::from(VALUES[0])) as f32) / max;
        }
        for j in 0..32 {
            l[ib * 32 + j] = best_index(&mid, idt[0] * xb[j]);
        }
        for j in 0..32 {
            let w = weight[j];
            for t in 0..nt {
                let q = fval[best_index(&mid, idt[t] * xb[j]) as usize];
                sqx[t] = (w * q).mul_add(xb[j], sqx[t]);
                sq2[t] = (w * q).mul_add(q, sq2[t]);
            }
        }
        d = if sq2[0] > 0.0 { sqx[0] / sq2[0] } else { 0.0 };
        let mut best = d * sqx[0];
        for t in 1..nt {
            if sq2[t] > 0.0 && sqx[t] * sqx[t] > best * sq2[t] {
                d = sqx[t] / sq2[t];
                best = d * sqx[t];
            }
        }
        scales[ib] = d;
        if d.abs() > amax_scale {
            amax_scale = d.abs();
            max_scale = d;
        }
    }

    let dh;
    let (mut scales_h, mut scales_l) = (0u16, [0u8; 4]);
    if nblk > 1 {
        let d = -max_scale / 32.0;
        dh = F16::from_f32(d).to_bits();
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for ib in 0..nblk {
            let mut lq = (-32i32).max(31i32.min(nearest_int(id * scales[ib])));
            let dl = d * lq as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            for j in 0..32 {
                l[ib * 32 + j] = best_index(&mid, idl * x[ib * 32 + j]);
            }
            lq += 32;
            let (l_l, l_h) = ((lq & 0xf) as u8, (lq >> 4) as u16);
            if ib % 2 == 0 {
                scales_l[ib / 2] = l_l;
            } else {
                scales_l[ib / 2] |= l_l << 4;
            }
            scales_h |= l_h << (2 * (ib % 8));
        }
    } else {
        dh = F16::from_f32(scales[0]).to_bits();
        let id = if scales[0] != 0.0 {
            1.0 / scales[0]
        } else {
            0.0
        };
        for j in 0..sb {
            l[j] = best_index(&mid, id * x[j]);
        }
    }

    out.extend_from_slice(&dh.to_le_bytes());
    if nblk > 1 {
        out.extend_from_slice(&scales_h.to_le_bytes());
        out.extend_from_slice(&scales_l);
    }
    for i in 0..sb / 32 {
        for j in 0..16 {
            out.push(l[32 * i + j] | (l[32 * i + 16 + j] << 4));
        }
    }
}

/// `xs` selects IQ4_XS (256-element blocks) over IQ4_NL (32).
pub fn encode_rows(xs: bool, x: &[f32], row_len: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    let sb = if xs { 256 } else { 32 };
    assert!(
        row_len % sb == 0 && x.len() % row_len == 0,
        "IQ4 rows must be whole blocks"
    );
    if let Some(q) = imatrix {
        assert!(q.len() >= row_len, "imatrix shorter than a row");
    }
    let mut out = Vec::with_capacity(x.len() / sb * if xs { 136 } else { 18 });
    for row in x.chunks_exact(row_len) {
        for (i, blk) in row.chunks_exact(sb).enumerate() {
            block(sb, blk, imatrix.map(|q| &q[sb * i..sb * (i + 1)]), &mut out);
        }
    }
    out
}
