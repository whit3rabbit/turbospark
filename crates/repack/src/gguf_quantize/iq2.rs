//! IQ2_XXS, IQ2_XS and IQ2_S encoders, ported from ggml's
//! `quantize_row_iq2_{xxs,xs,s}_impl`. They search the grids built in
//! `iqgrid.rs` (nearest grid point per 8 values, then a scale refit).
//!
//! IQ2_XXS and IQ2_XS require an importance matrix in ggml (they assert on a
//! missing one); IQ2_S works without, using a `0.25 * sigma2 + x * x` weight.

// Index-based loops mirror ggml's C one for one so the port can be diffed
// against ggml-quants.c; iterator rewrites would hide a transcription error.
#![allow(clippy::needless_range_loop, clippy::needless_late_init)]

use foundation::LogitValue as F16;

use super::iqgrid::{grid, IqGrid, Kind};
use super::kquants::{make_qp_quants, nearest_int};

const QK_K: usize = 256;
const K_MAX_Q: i32 = 3;

fn h(v: f32) -> u16 {
    F16::from_f32(v).to_bits()
}

/// ggml `iq2_find_best_neighbour`: best of the listed grid points under the
/// weighted squared error at `scale`. Writes the point's levels into `l`.
pub(super) fn find_best_neighbour(
    g: &IqGrid,
    m: i32,
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    l: &mut [u8],
) -> usize {
    let nb = g.neighbours_of(m);
    let n = nb[0] as usize;
    assert!(n > 0);
    let (mut best_d2, mut best) = (f32::MAX, usize::MAX);
    for &j in &nb[1..=n] {
        let pg = g.point(j as usize);
        let mut d2 = 0f32;
        for i in 0..g.dim {
            let q = f32::from(pg[i]);
            let diff = scale.mul_add(q, -xval[i]);
            // ggml has two copies of this loop (8-value points for IQ2, 4-value
            // for IQ3). Measured against llama-quantize on real weights, the
            // IQ2 loop accumulates with a fused multiply-add and the IQ3 loop
            // does not (the compiler vectorized it), so the two differ here.
            d2 = if g.dim == 8 {
                (weight[i] * diff).mul_add(diff, d2)
            } else {
                d2 + weight[i] * diff * diff
            };
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best = j as usize;
        }
    }
    let pg = g.point(best);
    for i in 0..g.dim {
        l[i] = ((pg[i] - 1) / 2) as u8;
    }
    best
}

fn pack_u(l: &[u8]) -> usize {
    let mut u = 0usize;
    for i in 0..8 {
        u |= (l[i] as usize) << (2 * i);
    }
    u
}

fn grid_index(g: &IqGrid, l: &[u8]) -> usize {
    let idx = g.map[pack_u(l)];
    assert!(idx >= 0, "found a point not on the grid");
    idx as usize
}

/// The `kMaxQ` levels nearest `id * xval` for one 8-value group.
fn levels(id: f32, xval: &[f32], out: &mut [u8]) {
    for i in 0..8 {
        let l = nearest_int(0.5 * (id).mul_add(xval[i], -1.0));
        out[i] = 0i32.max((K_MAX_Q - 1).min(l)) as u8;
    }
}

/// Unfused on purpose: against `llama-quantize` on real weights these two
/// accumulations only match without fused multiply-adds (the compiler
/// vectorized ggml's loop). The synthetic goldens cannot tell the difference.
fn sums(weight: &[f32], xval: &[f32], l: &[u8]) -> (f32, f32) {
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for i in 0..weight.len() {
        let w = weight[i];
        let q = (2 * i32::from(l[i]) + 1) as f32;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
    }
    (sumqx, sumq2)
}

/// Sign bits and absolute values for one 8-group. `parity` is the IQ2_XXS and
/// IQ2_XS rule: an odd number of negatives is fixed by flipping the cheapest
/// element, so only 7 sign bits need storing.
fn signs8(xb: &[f32], weight: &[f32], xval: &mut [f32], parity: bool) -> u8 {
    let (mut nflip, mut s) = (0, 0u8);
    for i in 0..8 {
        if xb[i] >= 0.0 {
            xval[i] = xb[i];
        } else {
            xval[i] = -xb[i];
            nflip += 1;
            s |= 1 << i;
        }
    }
    if parity && nflip % 2 == 1 {
        let mut imin = 0;
        let mut min = weight[imin] * xb[imin] * xb[imin];
        for i in 1..8 {
            let ax = weight[i] * xb[i] * xb[i];
            if ax < min {
                min = ax;
                imin = i;
            }
        }
        xval[imin] = -xval[imin];
        s ^= 1 << imin;
    }
    s
}

fn sigma2_of(xbl: &[f32], factor: f32) -> f32 {
    let mut sumx2 = 0f32;
    for v in &xbl[..QK_K] {
        sumx2 = (*v).mul_add(*v, sumx2);
    }
    factor * sumx2 / QK_K as f32
}

pub fn encode_xxs(x: &[f32], qw: &[f32], out: &mut Vec<u8>) {
    let g = grid(Kind::Iq2Xxs);
    let mut q2 = [0u32; 16];
    let mut scales = [0f32; 8];
    let mut max_scale = 0f32;
    let sigma2 = sigma2_of(x, 1.0);
    let (mut weight, mut xval, mut waux) = ([0f32; 32], [0f32; 32], [0f32; 32]);
    let (mut l, mut laux) = ([0u8; 32], [0u8; 32]);
    for ib in 0..8 {
        let xb = &x[32 * ib..];
        let qwb = &qw[32 * ib..];
        for i in 0..32 {
            weight[i] = qwb[i] * (xb[i]).mul_add(xb[i], sigma2).sqrt();
        }
        for i in 0..32 {
            waux[i] = weight[i].sqrt();
        }
        let mut block_signs = [0u8; 4];
        for k in 0..4 {
            let s = signs8(&xb[8 * k..], &weight[8 * k..], &mut xval[8 * k..], true);
            block_signs[k] = s & 127;
        }
        let mut max = xval[0];
        for i in 1..32 {
            max = max.max(xval[i]);
        }
        if max < 1e-15 {
            scales[ib] = 0.0;
            l.fill(0);
            continue;
        }
        let mut scale = make_qp_quants(32, K_MAX_Q + 1, &xval, &mut l, &weight);
        let eff_max = scale * K_MAX_Q as f32;
        if eff_max <= 0.0 {
            scales[ib] = 0.0;
            l.fill(0);
            continue;
        }
        let mut best = 0f32;
        for is in -6i32..=6 {
            let id = ((is as f32).mul_add(0.1, (2 * K_MAX_Q - 1) as f32)) / eff_max;
            let this_scale = 1.0 / id;
            for k in 0..4 {
                levels(id, &xval[8 * k..], &mut laux[8 * k..]);
                let m = g.map[pack_u(&laux[8 * k..])];
                if m < 0 {
                    find_best_neighbour(
                        g,
                        m,
                        &xval[8 * k..],
                        &waux[8 * k..],
                        this_scale,
                        &mut laux[8 * k..],
                    );
                }
            }
            let (sumqx, sumq2) = sums(&weight, &xval, &laux);
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l.copy_from_slice(&laux);
            }
        }
        if scale > 0.0 {
            let id = 1.0 / scale;
            for k in 0..4 {
                let mut lk = [0u8; 8];
                levels(id, &xval[8 * k..], &mut lk);
                let m = g.map[pack_u(&lk)];
                let gi = if m < 0 {
                    find_best_neighbour(g, m, &xval[8 * k..], &waux[8 * k..], scale, &mut lk)
                } else {
                    m as usize
                };
                let pg = g.point(gi);
                for i in 0..8 {
                    l[8 * k + i] = ((pg[i] - 1) / 2) as u8;
                }
            }
            let (sumqx, sumq2) = sums(&weight, &xval, &l);
            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for k in 0..4 {
                block_signs[k] = !block_signs[k] & 127;
            }
        }
        for k in 0..4 {
            let gi = grid_index(g, &l[8 * k..]);
            q2[2 * ib] |= (gi as u32) << (8 * k);
            q2[2 * ib + 1] |= u32::from(block_signs[k]) << (7 * k);
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        out.extend_from_slice(&h(0.0).to_le_bytes());
        out.extend_from_slice(&[0u8; 64]);
        return;
    }
    let d = max_scale / 31.0;
    let id = 1.0 / d;
    for ib in 0..8 {
        let lq = 0i32.max(15i32.min(nearest_int(0.5 * (id).mul_add(scales[ib], -1.0))));
        q2[2 * ib + 1] |= (lq as u32) << 28;
    }
    out.extend_from_slice(&h(d).to_le_bytes());
    q2.iter()
        .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
}

/// Shared 16-value sub-block search of IQ2_XS and IQ2_S. Returns `None` for an
/// all-zero sub-block. `L` is the chosen levels and the scale may be negative.
fn search16(
    g: &IqGrid,
    xval: &[f32],
    weight: &[f32],
    waux: &[f32],
    eps: f32,
) -> Option<([u8; 16], f32)> {
    let mut max = xval[0];
    for i in 1..16 {
        max = max.max(xval[i]);
    }
    let mut l = [0u8; 16];
    if max < eps {
        return None;
    }
    let mut best = 0f32;
    let mut scale = max / (2 * K_MAX_Q - 1) as f32;
    let (mut on_grid, mut on_grid_aux) = ([true; 2], [true; 2]);
    let mut laux = [0u8; 16];
    for is in -9i32..=9 {
        let id = ((is as f32).mul_add(0.1, (2 * K_MAX_Q - 1) as f32)) / max;
        let this_scale = 1.0 / id;
        for k in 0..2 {
            levels(id, &xval[8 * k..], &mut laux[8 * k..]);
            let m = g.map[pack_u(&laux[8 * k..])];
            on_grid_aux[k] = true;
            if m < 0 {
                on_grid_aux[k] = false;
                find_best_neighbour(
                    g,
                    m,
                    &xval[8 * k..],
                    &waux[8 * k..],
                    this_scale,
                    &mut laux[8 * k..],
                );
            }
        }
        let (sumqx, sumq2) = sums(weight, xval, &laux);
        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
            scale = sumqx / sumq2;
            best = scale * sumqx;
            l = laux;
            on_grid = on_grid_aux;
        }
    }
    let n_not_ongrid = on_grid.iter().filter(|b| !**b).count();
    if n_not_ongrid > 0 && scale > 0.0 {
        let id = 1.0 / scale;
        for k in 0..2 {
            if on_grid[k] {
                continue;
            }
            let mut u = 0usize;
            for i in 0..8 {
                let q = 0i32
                    .max((K_MAX_Q - 1).min(nearest_int(0.5 * (id).mul_add(xval[8 * k + i], -1.0))));
                u |= (q as usize) << (2 * i);
                l[8 * k + i] = q as u8;
            }
            let m = g.map[u];
            if m < 0 {
                find_best_neighbour(g, m, &xval[8 * k..], &waux[8 * k..], scale, &mut l[8 * k..]);
            }
        }
        let (sumqx, sumq2) = sums(weight, xval, &l);
        if sumq2 > 0.0 {
            scale = sumqx / sumq2;
        }
    }
    Some((l, scale))
}

pub fn encode_xs(x: &[f32], qw: &[f32], out: &mut Vec<u8>) {
    let g = grid(Kind::Iq2Xs);
    let mut q2 = [0u16; 32];
    let mut sc_bytes = [0u8; 8];
    let mut scales = [0f32; 16];
    let mut max_scale = 0f32;
    let sigma2 = sigma2_of(x, 1.0);
    let (mut weight, mut xval, mut waux) = ([0f32; 16], [0f32; 16], [0f32; 16]);
    for ib in 0..16 {
        let xb = &x[16 * ib..];
        let qwb = &qw[16 * ib..];
        for i in 0..16 {
            weight[i] = qwb[i] * (xb[i]).mul_add(xb[i], sigma2).sqrt();
        }
        for i in 0..16 {
            waux[i] = weight[i].sqrt();
        }
        let mut block_signs = [0u8; 2];
        for k in 0..2 {
            let s = signs8(&xb[8 * k..], &weight[8 * k..], &mut xval[8 * k..], true);
            block_signs[k] = s & 127;
        }
        let Some((l, mut scale)) = search16(g, &xval, &weight, &waux, 1e-15) else {
            scales[ib] = 0.0;
            continue;
        };
        if scale < 0.0 {
            scale = -scale;
            for k in 0..2 {
                block_signs[k] = !block_signs[k] & 127;
            }
        }
        for k in 0..2 {
            let gi = grid_index(g, &l[8 * k..]);
            q2[2 * ib + k] = (gi | (usize::from(block_signs[k]) << 9)) as u16;
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        out.extend_from_slice(&h(0.0).to_le_bytes());
        out.extend_from_slice(&[0u8; 72]);
        return;
    }
    let d = max_scale / 31.0;
    let id = 1.0 / d;
    for ib in 0..16 {
        let lq = 0i32.max(15i32.min(nearest_int(0.5 * (id).mul_add(scales[ib], -1.0)))) as u8;
        if ib % 2 == 0 {
            sc_bytes[ib / 2] = lq;
        } else {
            sc_bytes[ib / 2] |= lq << 4;
        }
    }
    out.extend_from_slice(&h(d).to_le_bytes());
    q2.iter()
        .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
    out.extend_from_slice(&sc_bytes);
}

pub fn encode_s(x: &[f32], qw: Option<&[f32]>, out: &mut Vec<u8>) {
    let g = grid(Kind::Iq2S);
    let mut qs = [0u8; 64];
    let mut qh = [0u8; 8];
    let mut sc_bytes = [0u8; 8];
    let mut scales = [0f32; 16];
    let mut max_scale = 0f32;
    let sigma2 = sigma2_of(x, 2.0);
    let (mut weight, mut xval, mut waux) = ([0f32; 16], [0f32; 16], [0f32; 16]);
    for ib in 0..16 {
        let xb = &x[16 * ib..];
        match qw {
            Some(q) => {
                for i in 0..16 {
                    weight[i] = q[16 * ib + i] * (xb[i]).mul_add(xb[i], sigma2).sqrt();
                }
            }
            None => {
                for i in 0..16 {
                    weight[i] = xb[i].mul_add(xb[i], 0.25 * sigma2);
                }
            }
        }
        for i in 0..16 {
            waux[i] = weight[i].sqrt();
        }
        let mut block_signs = [0u8; 2];
        for k in 0..2 {
            block_signs[k] = signs8(&xb[8 * k..], &weight[8 * k..], &mut xval[8 * k..], false);
        }
        let Some((l, mut scale)) = search16(g, &xval, &weight, &waux, 1e-8) else {
            scales[ib] = 0.0;
            continue;
        };
        if scale < 0.0 {
            scale = -scale;
            for k in 0..2 {
                block_signs[k] = !block_signs[k];
            }
        }
        for k in 0..2 {
            let gi = grid_index(g, &l[8 * k..]);
            let i8_ = 2 * ib + k;
            qs[i8_] = (gi & 255) as u8;
            qh[i8_ / 4] |= ((gi >> 8) as u8) << (2 * (i8_ % 4));
            qs[QK_K / 8 + i8_] = block_signs[k];
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        out.extend_from_slice(&h(0.0).to_le_bytes());
        out.extend_from_slice(&[0u8; 80]);
        return;
    }
    let d = max_scale / 31.0;
    let id = 1.0 / d;
    for ib in 0..16 {
        let lq = 0i32.max(15i32.min(nearest_int(0.5 * (id).mul_add(scales[ib], -1.0)))) as u8;
        if ib % 2 == 0 {
            sc_bytes[ib / 2] = lq;
        } else {
            sc_bytes[ib / 2] |= lq << 4;
        }
    }
    out.extend_from_slice(&h(d * 0.9875).to_le_bytes());
    out.extend_from_slice(&qs);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&sc_bytes);
}

/// Rows of IQ2 data. `qw` is required for XXS and XS (see module docs).
pub fn encode_rows(which: Kind, x: &[f32], row_len: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(
        row_len % QK_K == 0 && x.len() % row_len == 0,
        "IQ2 rows must be whole blocks"
    );
    let mut out = Vec::new();
    for row in x.chunks_exact(row_len) {
        for (i, blk) in row.chunks_exact(QK_K).enumerate() {
            let qw = imatrix.map(|q| &q[QK_K * i..QK_K * (i + 1)]);
            match which {
                Kind::Iq2Xxs => encode_xxs(blk, qw.expect("IQ2_XXS needs an imatrix"), &mut out),
                Kind::Iq2Xs => encode_xs(blk, qw.expect("IQ2_XS needs an imatrix"), &mut out),
                Kind::Iq2S => encode_s(blk, qw, &mut out),
                _ => unreachable!(),
            }
        }
    }
    out
}
