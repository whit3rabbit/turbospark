//! IQ3_XXS and IQ3_S encoders, ported from ggml's `quantize_row_iq3_xxs_impl`
//! and `quantize_row_iq3_s_impl`. Both search 4-value grid points on a 3-bit
//! lattice (`iqgrid.rs`), with or without an importance matrix.
//!
//! One quirk is reproduced on purpose: in IQ3_S an all-zero 32-element
//! sub-block `continue`s before ggml advances its `qs` and `signs` write
//! pointers, so the next sub-block overwrites that slot. That is what
//! `llama-quantize` writes, so matching it is the point.

// Index-based loops mirror ggml's C one for one so the port can be diffed
// against ggml-quants.c; iterator rewrites would hide a transcription error.
#![allow(clippy::needless_range_loop, clippy::needless_late_init)]

use foundation::LogitValue as F16;

use super::iq2::find_best_neighbour;
use super::iqgrid::{grid, IqGrid, Kind};
use super::kquants::nearest_int;

const QK_K: usize = 256;
const K_MAX_Q: i32 = 8;

fn h(v: f32) -> u16 {
    F16::from_f32(v).to_bits()
}

fn pack4(l: &[u8]) -> usize {
    (0..4).fold(0usize, |u, i| u | (l[i] as usize) << (3 * i))
}

/// Unfused on purpose: against `llama-quantize` on real weights these two
/// accumulations only match without fused multiply-adds (the compiler
/// vectorized ggml's loop). The synthetic goldens cannot tell the difference.
fn sums(weight: &[f32], xval: &[f32], l: &[u8]) -> (f32, f32) {
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for i in 0..32 {
        let w = weight[i];
        let q = (2 * i32::from(l[i]) + 1) as f32;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
    }
    (sumqx, sumq2)
}

struct Search {
    /// Levels of the 32 values.
    l: [u8; 32],
    scale: f32,
}

/// The 32-value search shared by both types. `None` means skip the sub-block.
#[allow(clippy::too_many_arguments)]
fn search32(
    g: &IqGrid,
    xval: &[f32; 32],
    weight: &[f32; 32],
    waux: &[f32; 32],
    skip_below: Option<f32>,
    is_range: i32,
    initial_on_grid: bool,
    refit_only_off_grid: bool,
) -> Option<Search> {
    let mut max = xval[0];
    for i in 1..32 {
        max = max.max(xval[i]);
    }
    let skip = match skip_below {
        Some(eps) => max < eps,
        None => max == 0.0,
    };
    if skip {
        return None;
    }
    let mut l = [0u8; 32];
    let mut laux = [0u8; 32];
    let mut best = 0f32;
    let mut scale = max / (2 * K_MAX_Q - 1) as f32;
    let mut on_grid = [initial_on_grid; 8];
    let mut on_grid_aux = [true; 8];
    for is in -is_range..=is_range {
        let id = ((is as f32).mul_add(0.2, (2 * K_MAX_Q - 1) as f32)) / max;
        let this_scale = 1.0 / id;
        for k in 0..8 {
            for i in 0..4 {
                let q = nearest_int(0.5 * (id).mul_add(xval[4 * k + i], -1.0));
                laux[4 * k + i] = 0i32.max((K_MAX_Q - 1).min(q)) as u8;
            }
            let m = g.map[pack4(&laux[4 * k..])];
            on_grid_aux[k] = true;
            if m < 0 {
                on_grid_aux[k] = false;
                find_best_neighbour(
                    g,
                    m,
                    &xval[4 * k..],
                    &waux[4 * k..],
                    this_scale,
                    &mut laux[4 * k..],
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
        for k in 0..8 {
            if refit_only_off_grid && on_grid[k] {
                continue;
            }
            let mut lk = [0u8; 4];
            for i in 0..4 {
                let q = nearest_int(0.5 * (id).mul_add(xval[4 * k + i], -1.0));
                lk[i] = 0i32.max((K_MAX_Q - 1).min(q)) as u8;
            }
            let m = g.map[pack4(&lk)];
            let gi = if m < 0 {
                find_best_neighbour(g, m, &xval[4 * k..], &waux[4 * k..], scale, &mut lk)
            } else {
                m as usize
            };
            let pg = g.point(gi);
            for i in 0..4 {
                l[4 * k + i] = ((pg[i] - 1) / 2) as u8;
            }
        }
        let (sumqx, sumq2) = sums(weight, xval, &l);
        if sumq2 > 0.0 {
            scale = sumqx / sumq2;
        }
    }
    Some(Search { l, scale })
}

fn grid_index(g: &IqGrid, l: &[u8]) -> usize {
    let idx = g.map[pack4(l)];
    assert!(idx >= 0, "found a point not on the grid");
    idx as usize
}

fn signs_and_xval(xb: &[f32], weight: &[f32], xval: &mut [f32; 32], parity: bool) -> [u8; 4] {
    let mut out = [0u8; 4];
    for k in 0..4 {
        let (mut nflip, mut s) = (0, 0u8);
        for i in 0..8 {
            if xb[8 * k + i] >= 0.0 {
                xval[8 * k + i] = xb[8 * k + i];
            } else {
                xval[8 * k + i] = -xb[8 * k + i];
                nflip += 1;
                s |= 1 << i;
            }
        }
        if parity && nflip % 2 == 1 {
            let mut imin = 0;
            let mut min = weight[8 * k] * xb[8 * k] * xb[8 * k];
            for i in 1..8 {
                let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                if ax < min {
                    min = ax;
                    imin = i;
                }
            }
            xval[8 * k + imin] = -xval[8 * k + imin];
            s ^= 1 << imin;
        }
        out[k] = if parity { s & 127 } else { s };
    }
    out
}

fn weights(xb: &[f32], qw: Option<&[f32]>, sigma2: f32, w: &mut [f32; 32], waux: &mut [f32; 32]) {
    for i in 0..32 {
        w[i] = match qw {
            Some(q) => q[i] * (xb[i]).mul_add(xb[i], sigma2).sqrt(),
            None => xb[i] * xb[i],
        };
    }
    for i in 0..32 {
        waux[i] = w[i].sqrt();
    }
}

fn sigma2_of(x: &[f32]) -> f32 {
    let mut sumx2 = 0f32;
    for v in &x[..QK_K] {
        sumx2 = (*v).mul_add(*v, sumx2);
    }
    2.0 * sumx2 / QK_K as f32
}

pub fn encode_xxs(x: &[f32], qw: Option<&[f32]>, out: &mut Vec<u8>) {
    let g = grid(Kind::Iq3Xxs);
    let sigma2 = sigma2_of(x);
    let mut q3 = [0u8; 64];
    let mut scales_and_signs = [0u32; 8];
    let mut scales = [0f32; 8];
    let mut max_scale = 0f32;
    let (mut weight, mut xval, mut waux) = ([0f32; 32], [0f32; 32], [0f32; 32]);
    for ib in 0..8 {
        let xb = &x[32 * ib..];
        weights(
            xb,
            qw.map(|q| &q[32 * ib..]),
            sigma2,
            &mut weight,
            &mut waux,
        );
        let mut block_signs = signs_and_xval(xb, &weight, &mut xval, true);
        let Some(Search { l, mut scale }) =
            search32(g, &xval, &weight, &waux, Some(1e-8), 15, true, true)
        else {
            scales[ib] = 0.0;
            continue;
        };
        if scale < 0.0 {
            scale = -scale;
            for s in &mut block_signs {
                *s = !*s & 127;
            }
        }
        for k in 0..8 {
            q3[8 * ib + k] = grid_index(g, &l[4 * k..]) as u8;
        }
        scales_and_signs[ib] = u32::from(block_signs[0])
            | (u32::from(block_signs[1]) << 7)
            | (u32::from(block_signs[2]) << 14)
            | (u32::from(block_signs[3]) << 21);
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        out.extend_from_slice(&h(0.0).to_le_bytes());
        out.extend_from_slice(&[0u8; 96]);
        return;
    }
    let d = max_scale / 31.0;
    let id = 1.0 / d;
    for ib in 0..8 {
        let lq = 0i32.max(15i32.min(nearest_int(0.5 * (id).mul_add(scales[ib], -1.0))));
        scales_and_signs[ib] |= (lq as u32) << 28;
    }
    out.extend_from_slice(&h(d * 1.0125).to_le_bytes());
    out.extend_from_slice(&q3);
    scales_and_signs
        .iter()
        .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
}

pub fn encode_s(x: &[f32], qw: Option<&[f32]>, out: &mut Vec<u8>) {
    let g = grid(Kind::Iq3S);
    let sigma2 = sigma2_of(x);
    let (mut qs, mut qh, mut signs, mut sc) = ([0u8; 64], [0u8; 8], [0u8; 32], [0u8; 4]);
    let (mut qs_pos, mut signs_pos) = (0usize, 0usize);
    let mut scales = [0f32; 8];
    let mut max_scale = 0f32;
    let (mut weight, mut xval, mut waux) = ([0f32; 32], [0f32; 32], [0f32; 32]);
    for ib in 0..8 {
        let xb = &x[32 * ib..];
        weights(
            xb,
            qw.map(|q| &q[32 * ib..]),
            sigma2,
            &mut weight,
            &mut waux,
        );
        let mut block_signs = signs_and_xval(xb, &weight, &mut xval, false);
        let Some(Search { l, mut scale }) =
            search32(g, &xval, &weight, &waux, None, 9, false, false)
        else {
            // ggml `continue`s here without advancing qs/signs (see module docs).
            scales[ib] = 0.0;
            continue;
        };
        if scale < 0.0 {
            scale = -scale;
            for s in &mut block_signs {
                *s = !*s;
            }
        }
        for k in 0..8 {
            let gi = grid_index(g, &l[4 * k..]);
            qs[qs_pos + k] = (gi & 255) as u8;
            qh[(ib * 8 + k) / 8] |= ((gi >> 8) as u8) << ((ib * 8 + k) % 8);
        }
        qs_pos += 8;
        signs[signs_pos..signs_pos + 4].copy_from_slice(&block_signs);
        signs_pos += 4;
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        out.extend_from_slice(&h(0.0).to_le_bytes());
        out.extend_from_slice(&[0u8; 108]);
        return;
    }
    let d = max_scale / 31.0;
    let id = 1.0 / d;
    for ib in (0..8).step_by(2) {
        let l1 = 0i32.max(15i32.min(nearest_int(0.5 * (id).mul_add(scales[ib], -1.0)))) as u8;
        let l2 = 0i32.max(15i32.min(nearest_int(0.5 * (id).mul_add(scales[ib + 1], -1.0)))) as u8;
        sc[ib / 2] = l1 | (l2 << 4);
    }
    out.extend_from_slice(&h(d * 1.033).to_le_bytes());
    out.extend_from_slice(&qs);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&signs);
    out.extend_from_slice(&sc);
}

pub fn encode_rows(which: Kind, x: &[f32], row_len: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(
        row_len % QK_K == 0 && x.len() % row_len == 0,
        "IQ3 rows must be whole blocks"
    );
    let mut out = Vec::new();
    for row in x.chunks_exact(row_len) {
        for (i, blk) in row.chunks_exact(QK_K).enumerate() {
            let qw = imatrix.map(|q| &q[QK_K * i..QK_K * (i + 1)]);
            match which {
                Kind::Iq3Xxs => encode_xxs(blk, qw, &mut out),
                Kind::Iq3S => encode_s(blk, qw, &mut out),
                _ => unreachable!(),
            }
        }
    }
    out
}
