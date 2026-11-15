//! IQ1_S and IQ1_M encoders, ported from ggml's `quantize_row_iq1_s_impl` and
//! `quantize_row_iq1_m_impl`.
//!
//! Each block of 32 (IQ1_S) or 16 (IQ1_M) values is first split exactly into
//! three groups (levels -1, 0, +1 shifted by +/- `DELTA`) by an exhaustive
//! search over the two sorted boundaries, then each 8-value group is snapped
//! to the nearest grid point.
//!
//! Known difference: ggml sorts with libc `qsort`, which is not stable, so two
//! exactly equal weights may be ordered differently from this crate's
//! index-stable sort. Real data rarely has such ties inside a block.
//!
//! Imatrix: required by ggml for IQ1_S, optional for IQ1_M.

// Index-based loops mirror ggml's C one for one so the port can be diffed
// against ggml-quants.c; iterator rewrites would hide a transcription error.
#![allow(clippy::needless_range_loop, clippy::needless_late_init)]

use foundation::LogitValue as F16;

use super::iqgrid::{grid, IqGrid, Kind};
use super::kquants::nearest_int;

const QK_K: usize = 256;
const DELTA: f32 = 0.125;
const X_P: [f32; 3] = [-1.0 + DELTA, DELTA, 1.0 + DELTA];
const X_M: [f32; 3] = [-1.0 - DELTA, -DELTA, 1.0 - DELTA];

fn h(v: f32) -> u16 {
    F16::from_f32(v).to_bits()
}

/// ggml `iq1_find_best_neighbour2`: nearest listed grid point under the
/// weighted squared error with level values `xg`.
fn find_best_neighbour2(
    g: &IqGrid,
    m: i32,
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    xg: &[f32; 3],
    l: &mut [u8],
) -> usize {
    let nb = g.neighbours_of(m);
    let n = nb[0] as usize;
    assert!(n > 0);
    let (mut best_score, mut best) = (f32::MAX, usize::MAX);
    for &j in &nb[1..=n] {
        let pg = g.point(j as usize);
        let mut d2 = 0f32;
        for i in 0..8 {
            let q = xg[((pg[i] - 1) / 2) as usize];
            let diff = scale.mul_add(q, -xval[i]);
            d2 = (weight[i] * diff).mul_add(diff, d2);
        }
        if d2 < best_score {
            best_score = d2;
            best = j as usize;
        }
    }
    // ggml falls back to a scan of the whole grid here, and that scan reads
    // `xval[i]` with the grid index (an out-of-bounds read in C). It is
    // reachable only for non-finite input, so refuse instead.
    assert!(
        best != usize::MAX,
        "IQ1: no grid point found (non-finite input?)"
    );
    let pg = g.point(best);
    for i in 0..8 {
        l[i] = ((pg[i] - 1) / 2) as u8;
    }
    best
}

fn pack(l: &[u8]) -> usize {
    (0..8).fold(0usize, |u, j| u | (l[j] as usize) << (2 * j))
}

/// How exactly equal values inside an IQ1 block are ordered before ggml's
/// block search. ggml uses libc `qsort`, which is not stable, so this depends
/// on the platform whose output you want to match (see `libc_sort.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TieOrder {
    /// Ties keep their original index order (what a stable sort, such as
    /// glibc's mergesort-based `qsort`, produces).
    Stable,
    /// Apple libc `qsort`, as in Homebrew builds of ggml on macOS.
    Apple,
}

static TIE_ORDER: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Select the tie order for subsequent IQ1 encoding (process-wide).
pub fn set_tie_order(o: TieOrder) {
    TIE_ORDER.store(
        u8::from(o == TieOrder::Apple),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Indices of `xb` in ascending value order.
fn sorted_idx(xb: &[f32]) -> Vec<usize> {
    if TIE_ORDER.load(std::sync::atomic::Ordering::Relaxed) == 1 {
        let mut v: Vec<(f32, u32)> = xb.iter().enumerate().map(|(i, &x)| (x, i as u32)).collect();
        super::libc_sort::apple_qsort(&mut v);
        return v.into_iter().map(|(_, i)| i as usize).collect();
    }
    let mut idx: Vec<usize> = (0..xb.len()).collect();
    idx.sort_by(|&a, &b| {
        xb[a]
            .partial_cmp(&xb[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

fn weights(xb: &[f32], qw: Option<&[f32]>, sigma2: f32, w: &mut [f32]) {
    for i in 0..xb.len() {
        w[i] = match qw {
            Some(q) => q[i] * (xb[i]).mul_add(xb[i], sigma2).sqrt(),
            None => xb[i] * xb[i],
        };
    }
}

fn sigma2_of(x: &[f32]) -> f32 {
    let mut sumx2 = 0f32;
    for v in &x[..QK_K] {
        sumx2 = (*v).mul_add(*v, sumx2);
    }
    2.0 * sumx2 / QK_K as f32
}

/// What the IQ1_S block search decides, before any format packs it.
struct SBlock {
    /// Grid index of every 8-element group (`4 * ib + k`).
    index: [usize; 32],
    scales: [f32; 8],
    shifts: [i8; 8],
    max_scale: f32,
}

/// ggml's `quantize_row_iq1_s_impl` search over `g`, shared verbatim by the
/// IQ1_S encoder and by Unsloth's IQ1_XS / IQ1_XXS / IQ1_XXXS (which differ
/// only in the grid and in how the result is packed).
fn search_s(g: &IqGrid, x: &[f32], qw: &[f32]) -> SBlock {
    let sigma2 = sigma2_of(x);
    let mut all_index = [0usize; 32];
    let (mut scales, mut shifts) = ([0f32; 8], [0i8; 8]);
    let mut max_scale = 0f32;
    let mut weight = [0f32; 32];
    for ib in 0..8 {
        let xb = &x[32 * ib..32 * ib + 32];
        weights(xb, Some(&qw[32 * ib..]), sigma2, &mut weight);
        let mut max = xb[0].abs();
        for v in &xb[1..] {
            max = max.max(v.abs());
        }
        if max < 1e-12 {
            scales[ib] = 0.0;
            shifts[ib] = 1;
            continue;
        }
        let idx = sorted_idx(xb);
        let (mut sumx, mut sumw) = ([0f32; 33], [0f32; 33]);
        for j in 0..32 {
            let i = idx[j];
            sumx[j + 1] = weight[i].mul_add(xb[i], sumx[j]);
            sumw[j + 1] = sumw[j] + weight[i];
        }
        let (mut best_score, mut scale) = (-f32::MAX, max);
        let (mut besti1, mut besti2, mut best_shift) = (-1i32, -1i32, 0i32);
        for i1 in 0..=32usize {
            for i2 in i1..=32usize {
                for (xs, shift) in [(&X_P, 1), (&X_M, -1)] {
                    let sumqx = (sumx[32] - sumx[i2]).mul_add(
                        xs[2],
                        (sumx[i1] - sumx[0]).mul_add(xs[0], (sumx[i2] - sumx[i1]) * xs[1]),
                    );
                    let sumq2 = ((sumw[32] - sumw[i2]) * xs[2]).mul_add(
                        xs[2],
                        ((sumw[i1] - sumw[0]) * xs[0])
                            .mul_add(xs[0], (sumw[i2] - sumw[i1]) * xs[1] * xs[1]),
                    );
                    if sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
                        scale = sumqx / sumq2;
                        best_score = scale * sumqx;
                        besti1 = i1 as i32;
                        besti2 = i2 as i32;
                        best_shift = shift;
                    }
                }
            }
        }
        if besti1 < 0 || besti2 < 0 || best_shift == 0 {
            scales[ib] = 0.0;
            shifts[ib] = 1;
            continue;
        }
        let mut l = [0u8; 32];
        for j in 0..besti1 as usize {
            l[idx[j]] = 0;
        }
        for j in besti1 as usize..besti2 as usize {
            l[idx[j]] = 1;
        }
        for j in besti2 as usize..32 {
            l[idx[j]] = 2;
        }
        if scale < 0.0 {
            for v in &mut l {
                *v = 2 - *v;
            }
            scale = -scale;
            best_shift = -best_shift;
        }
        let xx = if best_shift == 1 { &X_P } else { &X_M };
        let mut index = [0usize; 4];
        let mut all_on_grid = true;
        for k in 0..4 {
            let m = g.map[pack(&l[8 * k..])];
            index[k] = if m < 0 {
                all_on_grid = false;
                find_best_neighbour2(
                    g,
                    m,
                    &xb[8 * k..],
                    &weight[8 * k..],
                    scale,
                    xx,
                    &mut l[8 * k..],
                )
            } else {
                m as usize
            };
        }
        if !all_on_grid {
            let (mut sumqx, mut sumq2) = (0f32, 0f32);
            for k in 0..4 {
                let pg = g.point(index[k]);
                for j in 0..8 {
                    let w = weight[8 * k + j];
                    let q = xx[((pg[j] - 1) / 2) as usize];
                    sumqx = (w * q).mul_add(xb[8 * k + j], sumqx);
                    sumq2 = (w * q).mul_add(q, sumq2);
                }
            }
            if sumqx > 0.0 && sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        all_index[4 * ib..4 * ib + 4].copy_from_slice(&index);
        scales[ib] = scale;
        shifts[ib] = best_shift as i8;
        max_scale = max_scale.max(scale);
    }
    SBlock {
        index: all_index,
        scales,
        shifts,
        max_scale,
    }
}

/// The 3-bit scale code plus delta-sign bit for sub-block `ib`, as every IQ1
/// format stores it (sign in bit 3).
fn scale_nibble(id: f32, scale: f32, shift: i8) -> u8 {
    let mut l = 0i32.max(7i32.min(nearest_int(0.5 * (id).mul_add(scale, -1.0))));
    if shift == -1 {
        l |= 8;
    }
    l as u8
}

pub fn encode_s(x: &[f32], qw: &[f32], out: &mut Vec<u8>) {
    let g = grid(Kind::Iq1);
    let b = search_s(g, x, qw);
    let (mut qs, mut qh) = ([0u8; 32], [0u16; 8]);
    for ib in 0..8 {
        let mut hh = 0u16;
        for k in 0..4 {
            let i = b.index[4 * ib + k];
            qs[4 * ib + k] = (i & 255) as u8;
            hh |= ((i >> 8) as u16) << (3 * k);
        }
        qh[ib] = hh;
    }
    if b.max_scale == 0.0 {
        out.extend_from_slice(&h(0.0).to_le_bytes());
    } else {
        let d = b.max_scale / 15.0;
        out.extend_from_slice(&h(d * 1.125).to_le_bytes());
        let id = 1.0 / d;
        for ib in 0..8 {
            qh[ib] |= u16::from(scale_nibble(id, b.scales[ib], b.shifts[ib])) << 12;
        }
    }
    out.extend_from_slice(&qs);
    qh.iter()
        .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
}

/// Unsloth's narrow-index IQ1 formats (ggml ids 64 to 66).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Narrow {
    Xs,
    Xxs,
    Xxxs,
}

/// `quantize_row_iq1_narrow_impl` from the fork: IQ1_S's search over the
/// narrower grid, then format-specific packing. The index fields are written
/// whatever the scales turn out to be, and the scale codes stay zero when the
/// whole super-block is degenerate, exactly as `encode_s` does.
pub fn encode_narrow(kind: Narrow, x: &[f32], qw: &[f32], out: &mut Vec<u8>) {
    let g = grid(match kind {
        Narrow::Xs => Kind::Iq1Xs,
        Narrow::Xxs => Kind::Iq1Xxs,
        Narrow::Xxxs => Kind::Iq1Xxxs,
    });
    let ngrid = g.points.len() / 8;
    let b = search_s(g, x, qw);
    for &i in &b.index {
        assert!(
            i < ngrid,
            "IQ1 narrow index {i} outside a {ngrid}-entry grid"
        );
    }
    let mut qs = [0u8; 32];
    for (slot, &i) in qs.iter_mut().zip(&b.index) {
        *slot = (i & 255) as u8;
    }
    // XS: 4 x 2-bit high fields per sub-block; XXS: 1 bit per group in the low
    // nibble (scale in bits 4-6, sign in bit 7); XXXS: none.
    let mut qh = [0u8; 8];
    for ib in 0..8 {
        for k in 0..4 {
            let hi = (b.index[4 * ib + k] >> 8) as u8;
            match kind {
                Narrow::Xs => qh[ib] |= hi << (2 * k),
                Narrow::Xxs => qh[ib] |= hi << k,
                Narrow::Xxxs => {}
            }
        }
    }
    let mut sc = [0u8; 4];
    let mut d_bits = h(0.0);
    if b.max_scale != 0.0 {
        let d = b.max_scale / 15.0;
        let id = 1.0 / d;
        for ib in 0..8 {
            let l = scale_nibble(id, b.scales[ib], b.shifts[ib]);
            match kind {
                Narrow::Xxs => qh[ib] |= l << 4,
                _ => sc[ib / 2] |= l << (4 * (ib & 1)),
            }
        }
        d_bits = h(d * 1.125);
    }
    out.extend_from_slice(&d_bits.to_le_bytes());
    out.extend_from_slice(&qs);
    match kind {
        Narrow::Xs => {
            out.extend_from_slice(&qh);
            out.extend_from_slice(&sc);
        }
        Narrow::Xxs => out.extend_from_slice(&qh),
        Narrow::Xxxs => out.extend_from_slice(&sc),
    }
}

pub fn encode_m(x: &[f32], qw: Option<&[f32]>, out: &mut Vec<u8>) {
    let g = grid(Kind::Iq1);
    let sigma2 = sigma2_of(x);
    let masks = [0x00u8, 0x80, 0x08, 0x88];
    let (mut qs, mut qh, mut sc) = ([0u8; 32], [0u8; 16], [0u16; 4]);
    let (mut scales, mut shifts) = ([0f32; 16], [0usize; 16]);
    let mut max_scale = 0f32;
    let mut weight = [0f32; 16];
    for ib in 0..16 {
        let xb = &x[16 * ib..16 * ib + 16];
        weights(xb, qw.map(|q| &q[16 * ib..]), sigma2, &mut weight);
        let mut max = xb[0].abs();
        for v in &xb[1..] {
            max = max.max(v.abs());
        }
        if max < 1e-7 {
            scales[ib] = 0.0;
            shifts[ib] = 0;
            continue;
        }
        let idx = sorted_idx(xb);
        let (mut best_score, mut scale) = (-f32::MAX, max);
        let (mut besti1, mut besti2, mut best_k) = (-1i32, -1i32, -1i32);
        {
            let (mut px, mut pw) = ([[0f32; 17]; 2], [[0f32; 17]; 2]);
            for j in 0..16 {
                let i = idx[j];
                let hh = usize::from(i >= 8);
                px[hh][j + 1] = weight[i].mul_add(xb[i], px[hh][j]);
                px[1 - hh][j + 1] = px[1 - hh][j];
                pw[hh][j + 1] = pw[hh][j] + weight[i];
                pw[1 - hh][j + 1] = pw[1 - hh][j];
            }
            let txs = [px[0][16], px[1][16]];
            let tws = [pw[0][16], pw[1][16]];
            let (p2, m2) = (X_P[2], X_M[2]);
            let cp1 = X_P[0] * X_P[0] - X_P[1] * X_P[1];
            let cp2 = X_P[1] * X_P[1] - X_P[2] * X_P[2];
            let cm1 = X_M[0] * X_M[0] - X_M[1] * X_M[1];
            let cm2 = X_M[1] * X_M[1] - X_M[2] * X_M[2];
            for i1 in 0..=16usize {
                for i2 in i1..=16usize {
                    let (mut qx_p, mut qx_m, mut q2_p, mut q2_m) =
                        ([0f32; 2], [0f32; 2], [0f32; 2], [0f32; 2]);
                    for hh in 0..2 {
                        let sx = px[hh][i1] + px[hh][i2];
                        qx_p[hh] = txs[hh].mul_add(p2, -sx);
                        qx_m[hh] = txs[hh].mul_add(m2, -sx);
                        q2_p[hh] =
                            pw[hh][i2].mul_add(cp2, (tws[hh] * p2).mul_add(p2, pw[hh][i1] * cp1));
                        q2_m[hh] =
                            pw[hh][i2].mul_add(cm2, (tws[hh] * m2).mul_add(m2, pw[hh][i1] * cm1));
                    }
                    let sumqx = [
                        qx_p[0] + qx_p[1],
                        qx_p[0] + qx_m[1],
                        qx_m[0] + qx_p[1],
                        qx_m[0] + qx_m[1],
                    ];
                    let sumq2 = [
                        q2_p[0] + q2_p[1],
                        q2_p[0] + q2_m[1],
                        q2_m[0] + q2_p[1],
                        q2_m[0] + q2_m[1],
                    ];
                    for k in 0..4 {
                        if sumq2[k] > 0.0 && sumqx[k] * sumqx[k] > best_score * sumq2[k] {
                            scale = sumqx[k] / sumq2[k];
                            best_score = scale * sumqx[k];
                            besti1 = i1 as i32;
                            besti2 = i2 as i32;
                            best_k = k as i32;
                        }
                    }
                }
            }
        }
        if besti1 < 0 || besti2 < 0 || best_k < 0 {
            scales[ib] = 0.0;
            shifts[ib] = 0;
            continue;
        }
        let mut l = [0u8; 16];
        for j in 0..besti1 as usize {
            l[idx[j]] = 0;
        }
        for j in besti1 as usize..besti2 as usize {
            l[idx[j]] = 1;
        }
        for j in besti2 as usize..16 {
            l[idx[j]] = 2;
        }
        if scale < 0.0 {
            for v in &mut l {
                *v = 2 - *v;
            }
            scale = -scale;
            best_k = match best_k {
                0 => 3,
                1 => 2,
                2 => 1,
                _ => 0,
            };
        }
        let xx_for = |k: usize, bk: i32| -> &'static [f32; 3] {
            let plus = if k == 0 { bk < 2 } else { bk % 2 == 0 };
            if plus {
                &X_P
            } else {
                &X_M
            }
        };
        let mut index = [0usize; 2];
        let mut all_on_grid = true;
        for k in 0..2 {
            let xx = xx_for(k, best_k);
            let m = g.map[pack(&l[8 * k..])];
            index[k] = if m < 0 {
                all_on_grid = false;
                find_best_neighbour2(
                    g,
                    m,
                    &xb[8 * k..],
                    &weight[8 * k..],
                    scale,
                    xx,
                    &mut l[8 * k..],
                )
            } else {
                m as usize
            };
        }
        if !all_on_grid {
            let (mut sumqx, mut sumq2) = (0f32, 0f32);
            for k in 0..2 {
                let xx = xx_for(k, best_k);
                let pg = g.point(index[k]);
                for j in 0..8 {
                    let w = weight[8 * k + j];
                    let q = xx[((pg[j] - 1) / 2) as usize];
                    sumqx = (w * q).mul_add(xb[8 * k + j], sumqx);
                    sumq2 = (w * q).mul_add(q, sumq2);
                }
            }
            if sumqx > 0.0 && sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        qs[2 * ib] = (index[0] & 255) as u8;
        qs[2 * ib + 1] = (index[1] & 255) as u8;
        qh[ib] = ((index[0] >> 8) | ((index[1] >> 8) << 4)) as u8;
        scales[ib] = scale;
        shifts[ib] = best_k as usize;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        out.extend_from_slice(&qs);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&[0u8; 8]);
        return;
    }
    let mut d = max_scale / 15.0;
    let id = 1.0 / d;
    let (mut sumqx_f, mut sumq2_f) = (0f32, 0f32);
    for ib in 0..16 {
        let l = 0i32.max(7i32.min(nearest_int(0.5 * (id).mul_add(scales[ib], -1.0))));
        sc[ib / 4] |= (l as u16) << (3 * (ib % 4));
        qh[ib] |= masks[shifts[ib]];
        let xb = &x[16 * ib..16 * ib + 16];
        weights(xb, qw.map(|q| &q[16 * ib..]), sigma2, &mut weight);
        for k in 0..2 {
            let plus = if k == 0 {
                shifts[ib] < 2
            } else {
                shifts[ib] % 2 == 0
            };
            let xx = if plus { &X_P } else { &X_M };
            let gi = qs[2 * ib + k] as usize + ((usize::from(qh[ib]) << (8 - 4 * k)) & 0x700);
            let pg = g.point(gi);
            for j in 0..8 {
                let w = weight[8 * k + j];
                let q = xx[((pg[j] - 1) / 2) as usize] * (2 * l + 1) as f32;
                sumqx_f = (w * q).mul_add(xb[8 * k + j], sumqx_f);
                sumq2_f = (w * q).mul_add(q, sumq2_f);
            }
        }
    }
    if sumq2_f > 0.0 {
        d = sumqx_f / sumq2_f;
    }
    let s = h(d * 1.1125);
    sc[0] |= (s & 0x000f) << 12;
    sc[1] |= (s & 0x00f0) << 8;
    sc[2] |= (s & 0x0f00) << 4;
    sc[3] |= s & 0xf000;
    out.extend_from_slice(&qs);
    out.extend_from_slice(&qh);
    sc.iter()
        .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
}

/// Rows of IQ1 data. `qw` is required for IQ1_S (see module docs).
pub fn encode_rows(s: bool, x: &[f32], row_len: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    encode_rows_fmt(if s { Fmt::S } else { Fmt::M }, x, row_len, imatrix)
}

/// Which IQ1 block format [`encode_rows_fmt`] writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fmt {
    S,
    M,
    Narrow(Narrow),
}

pub fn encode_rows_fmt(fmt: Fmt, x: &[f32], row_len: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(
        row_len % QK_K == 0 && x.len() % row_len == 0,
        "IQ1 rows must be whole blocks"
    );
    let mut out = Vec::new();
    for row in x.chunks_exact(row_len) {
        for (i, blk) in row.chunks_exact(QK_K).enumerate() {
            let qw = imatrix.map(|q| &q[QK_K * i..QK_K * (i + 1)]);
            match fmt {
                Fmt::S => encode_s(blk, qw.expect("IQ1_S needs an imatrix"), &mut out),
                Fmt::M => encode_m(blk, qw, &mut out),
                Fmt::Narrow(k) => encode_narrow(
                    k,
                    blk,
                    qw.expect("IQ1 narrow types need an imatrix"),
                    &mut out,
                ),
            }
        }
    }
    out
}
