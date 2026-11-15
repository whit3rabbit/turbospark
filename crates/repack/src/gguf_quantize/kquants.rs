//! K-quant encoders (Q2_K..Q6_K), ported from ggml's `quantize_row_q*_K_ref`
//! (no importance matrix) and `quantize_row_q*_K_impl` (with one).
//!
//! ggml picks the path in `quantize_q*_K`: no weights -> `_ref`, weights ->
//! `_impl`, and the two are different algorithms, not one with a flag. The same
//! split is kept here. Float expressions keep ggml's operand order so results
//! can be compared byte for byte (`tests/gguf_quantize_golden.rs`).
//!
//! Source: ggml/src/ggml-quants.c (ggml-org/llama.cpp master as of the port).

// The port mirrors ggml's index-based C loops one for one so it can be diffed
// against ggml-quants.c. Iterator rewrites would hide a transcription error.
#![allow(clippy::needless_range_loop, clippy::needless_late_init)]

use foundation::LogitValue as F16;

const QK_K: usize = 256;
const GROUP_MAX_EPS: f32 = 1e-15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KKind {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl KKind {
    pub fn block_bytes(self) -> usize {
        match self {
            KKind::Q2K => 84,
            KKind::Q3K => 110,
            KKind::Q4K => 144,
            KKind::Q5K => 176,
            KKind::Q6K => 210,
        }
    }
}

fn h(v: f32) -> u16 {
    F16::from_f32(v).to_bits()
}

fn f(b: u16) -> f32 {
    f32::from(F16::from_bits(b))
}

/// ggml's `nearest_int`: round to nearest even via the 1.5 * 2^23 trick.
pub(super) fn nearest_int(v: f32) -> i32 {
    let bits = (v + 12582912.0).to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

pub(super) fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [i8],
    rmse_type: i32,
    qw: Option<&[f32]>,
) -> f32 {
    let (mut max, mut amax) = (0f32, 0f32);
    for &v in &x[..n] {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l[..n].fill(0);
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let q = nearest_int(iscale * x[i]);
            l[i] = (nmax + (-nmax).max((nmax - 1).min(q))) as i8;
        }
        return 1.0 / iscale;
    }
    let mut rt = rmse_type;
    let mut return_early = false;
    if rt < 0 {
        rt = -rt;
        return_early = true;
    }
    let weight = |i: usize| -> f32 {
        match qw {
            Some(q) => q[i],
            None => match rt {
                1 => x[i] * x[i],
                2 => 1.0,
                3 => x[i].abs(),
                _ => x[i].abs().sqrt(),
            },
        }
    };
    let (mut sumlx, mut suml2) = (0f32, 0f32);
    for i in 0..n {
        let q = (-nmax).max((nmax - 1).min(nearest_int(iscale * x[i])));
        l[i] = (q + nmax) as i8;
        let w = weight(i);
        sumlx = (w * x[i]).mul_add(q as f32, sumlx);
        suml2 = (w * q as f32).mul_add(q as f32, suml2);
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9i32..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(0.1f32.mul_add(is as f32, nmax as f32)) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let q = (-nmax).max((nmax - 1).min(nearest_int(iscale * x[i])));
            let w = weight(i);
            sumlx = (w * x[i]).mul_add(q as f32, sumlx);
            suml2 = (w * q as f32).mul_add(q as f32, suml2);
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let q = nearest_int(iscale * x[i]);
                l[i] = (nmax + (-nmax).max((nmax - 1).min(q))) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

fn make_q3_quants(n: usize, nmax: i32, x: &[f32], l: &mut [i8], do_rmse: bool) -> f32 {
    let (mut max, mut amax) = (0f32, 0f32);
    for &v in &x[..n] {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l[..n].fill(0);
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    if do_rmse {
        let (mut sumlx, mut suml2) = (0f32, 0f32);
        for i in 0..n {
            let q = (-nmax).max((nmax - 1).min(nearest_int(iscale * x[i])));
            l[i] = q as i8;
            let w = x[i] * x[i];
            sumlx = (w * x[i]).mul_add(q as f32, sumlx);
            suml2 = (w * q as f32).mul_add(q as f32, suml2);
        }
        for _ in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let mut slx = (-(w * x[i])).mul_add(l[i] as f32, sumlx);
                if slx > 0.0 {
                    let mut sl2 = (-(w * l[i] as f32)).mul_add(l[i] as f32, suml2);
                    let new_l = (-nmax).max((nmax - 1).min(nearest_int(x[i] * sl2 / slx)));
                    if new_l != l[i] as i32 {
                        slx = (w * x[i]).mul_add(new_l as f32, slx);
                        sl2 = (w * new_l as f32).mul_add(new_l as f32, sl2);
                        if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            l[i] = new_l as i8;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        for v in &mut l[..n] {
            *v += nmax as i8;
        }
        return if suml2 > 0.0 { sumlx / suml2 } else { 0.0 };
    }
    for i in 0..n {
        let q = (-nmax).max((nmax - 1).min(nearest_int(iscale * x[i])));
        l[i] = (q + nmax) as i8;
    }
    1.0 / iscale
}

/// `make_qkx2_quants` (weights required) and `make_qkx3_quants` differ only in
/// where they start the search and in edge handling, so they stay separate.
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let (mut min, mut max) = (x[0], x[0]);
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        sum_w += weights[i];
        sum_x = (weights[i]).mul_add(x[i], sum_x);
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l[..n].fill(0);
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min));
        l[i] = 0i32.max(nmax.min(q)) as u8;
        let mut diff = scale.mul_add(l[i] as f32, min) - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_error = (weights[i]).mul_add(diff, best_error);
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rdelta.mul_add(is as f32, rmin) + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let q = 0i32.max(nmax.min(nearest_int(iscale * (x[i] - min))));
            laux[i] = q as u8;
            let w = weights[i];
            sum_l = (w).mul_add(q as f32, sum_l);
            sum_l2 = (w * q as f32).mul_add(q as f32, sum_l2);
            sum_xl = (w * q as f32).mul_add(x[i], sum_xl);
        }
        let d = sum_w.mul_add(sum_l2, -(sum_l * sum_l));
        if d > 0.0 {
            let mut this_scale = (sum_w.mul_add(sum_xl, -(sum_x * sum_l))) / d;
            let mut this_min = (sum_l2.mul_add(sum_x, -(sum_l * sum_xl))) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0f32;
            for i in 0..n {
                let mut diff = this_scale.mul_add(laux[i] as f32, this_min) - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error = (weights[i]).mul_add(diff, cur_error);
            }
            if cur_error < best_error {
                l[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

#[allow(clippy::too_many_arguments)]
pub(super) fn make_qkx3_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let (mut min, mut max) = (x[0], x[0]);
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        sum_w += weights[i];
        sum_x = (weights[i]).mul_add(x[i], sum_x);
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        l[..n].fill(0);
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min));
        l[i] = 0i32.max(nmax.min(q)) as u8;
        let mut diff = scale.mul_add(l[i] as f32, min) - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_mad = (weights[i]).mul_add(diff, best_mad);
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rdelta.mul_add(is as f32, rmin) + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            // v can be inf/nan/out of range when min sits at max (see ggml).
            let v = iscale * (x[i] - min);
            let q = if v > 0.0 {
                nearest_int(v.min(nmax as f32))
            } else {
                0
            };
            laux[i] = q as u8;
            let w = weights[i];
            sum_l = (w).mul_add(q as f32, sum_l);
            sum_l2 = (w * q as f32).mul_add(q as f32, sum_l2);
            sum_xl = (w * q as f32).mul_add(x[i], sum_xl);
        }
        let d = sum_w.mul_add(sum_l2, -(sum_l * sum_l));
        if d > 0.0 {
            let mut this_scale = (sum_w.mul_add(sum_xl, -(sum_x * sum_l))) / d;
            let mut this_min = (sum_l2.mul_add(sum_x, -(sum_l * sum_xl))) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0f32;
            for i in 0..n {
                let mut diff = this_scale.mul_add(laux[i] as f32, this_min) - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                mad = (weights[i]).mul_add(diff, mad);
            }
            if mad < best_mad {
                l[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

pub(super) fn make_qp_quants(n: usize, nmax: i32, x: &[f32], l: &mut [u8], qw: &[f32]) -> f32 {
    let mut max = 0f32;
    for &v in &x[..n] {
        max = max.max(v);
    }
    if max < GROUP_MAX_EPS {
        l[..n].fill(0);
        return 0.0;
    }
    let mut iscale = nmax as f32 / max;
    for i in 0..n {
        l[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0f32;
    for i in 0..n {
        let diff = (-scale).mul_add(l[i] as f32, x[i]);
        best_mse = (qw[i] * diff).mul_add(diff, best_mse);
    }
    for is in -4i32..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1f32.mul_add(is as f32, nmax as f32)) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0f32;
        for i in 0..n {
            let q = nmax.min(nearest_int(iscale_is * x[i]));
            let diff = (-scale_is).mul_add(q as f32, x[i]);
            mse = (qw[i] * diff).mul_add(diff, mse);
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let (mut sumlx, mut suml2) = (0f32, 0f32);
    for i in 0..n {
        let q = nmax.min(nearest_int(iscale * x[i]));
        l[i] = q as u8;
        let w = qw[i];
        sumlx = (w * x[i]).mul_add(q as f32, sumlx);
        suml2 = (w * q as f32).mul_add(q as f32, suml2);
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = qw[i];
            let mut slx = (-(w * x[i])).mul_add(l[i] as f32, sumlx);
            let mut sl2 = (-(w * l[i] as f32)).mul_add(l[i] as f32, suml2);
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nmax.min(nearest_int(x[i] * sl2 / slx));
                if new_l != l[i] as i32 {
                    slx = (w * x[i]).mul_add(new_l as f32, slx);
                    sl2 = (w * new_l as f32).mul_add(new_l as f32, sl2);
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Pack 6-bit scale/min pair `j` into the 12-byte K4 scale area.
fn put_scale_min_k4(scales: &mut [u8; 12], j: usize, ls: u8, lm: u8) {
    if j < 4 {
        scales[j] = ls;
        scales[j + 4] = lm;
    } else {
        scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        scales[j - 4] |= (ls >> 4) << 6;
        scales[j] |= (lm >> 4) << 6;
    }
}

// ---- Q2_K -------------------------------------------------------------------

fn q2k_pack(out: &mut Vec<u8>, scales: &[u8; 16], l: &[u8; QK_K], d: u16, dmin: u16) {
    out.extend_from_slice(scales);
    for j in (0..QK_K).step_by(128) {
        for k in 0..32 {
            out.push(l[j + k] | (l[j + k + 32] << 2) | (l[j + k + 64] << 4) | (l[j + k + 96] << 6));
        }
    }
    out.extend_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&dmin.to_le_bytes());
}

fn q2k_requant(x: &[f32], scales: &[u8; 16], d: f32, dmin: f32, l: &mut [u8; QK_K]) {
    for j in 0..16 {
        let dd = d * (scales[j] & 0xF) as f32;
        if dd == 0.0 {
            continue;
        }
        let m = dmin * (scales[j] >> 4) as f32;
        for ii in 0..16 {
            let q = nearest_int((x[16 * j + ii] + m) / dd);
            l[16 * j + ii] = 0i32.max(3i32.min(q)) as u8;
        }
    }
}

fn q2k_block_ref(x: &[f32], out: &mut Vec<u8>) {
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 16];
    let mut weights = [0f32; 16];
    let (mut mins, mut scales) = ([0f32; 16], [0f32; 16]);
    let q4scale = 15.0f32;
    let (mut max_scale, mut max_min) = (0f32, 0f32);
    for j in 0..16 {
        for k in 0..16 {
            weights[k] = x[16 * j + k].abs();
        }
        scales[j] = make_qkx2_quants(
            16,
            3,
            &x[16 * j..],
            &weights,
            &mut l[16 * j..],
            &mut mins[j],
            &mut laux,
            -0.5,
            0.1,
            15,
            true,
        );
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }
    let mut sc = [0u8; 16];
    let (d, dmin);
    if max_scale > 0.0 {
        let iscale = q4scale / max_scale;
        for j in 0..16 {
            sc[j] = nearest_int(iscale * scales[j]) as u8;
        }
        d = h(max_scale / q4scale);
    } else {
        d = h(0.0);
    }
    if max_min > 0.0 {
        let iscale = q4scale / max_min;
        for j in 0..16 {
            sc[j] |= (nearest_int(iscale * mins[j]) << 4) as u8;
        }
        dmin = h(max_min / q4scale);
    } else {
        dmin = h(0.0);
    }
    q2k_requant(x, &sc, f(d), f(dmin), &mut l);
    q2k_pack(out, &sc, &l, d, dmin);
}

fn q2k_block_impl(x: &[f32], qw_blk: &[f32], out: &mut Vec<u8>) {
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 16];
    let (mut mins, mut scales, mut sw) = ([0f32; 16], [0f32; 16], [0f32; 16]);
    let mut weight = [0f32; 16];
    let (mut ls, mut lm) = ([0u8; 16], [0u8; 16]);
    let mut sumx2 = 0f32;
    for v in &x[..QK_K] {
        sumx2 = (*v).mul_add(*v, sumx2);
    }
    let sigma2 = sumx2 / QK_K as f32;
    for j in 0..16 {
        let qw = &qw_blk[16 * j..];
        for k in 0..16 {
            weight[k] = qw[k] * (x[16 * j + k]).mul_add(x[16 * j + k], sigma2).sqrt();
        }
        for k in 0..16 {
            sw[j] += weight[k];
        }
        scales[j] = make_qkx3_quants(
            16,
            3,
            &x[16 * j..],
            &weight,
            &mut l[16 * j..],
            &mut mins[j],
            &mut laux,
            -0.9,
            0.05,
            36,
            false,
        );
    }
    let dm = make_qp_quants(16, 15, &scales, &mut ls, &sw);
    let mm = make_qp_quants(16, 15, &mins, &mut lm, &sw);
    let (d, dmin) = (h(dm), h(mm));
    let mut sc = [0u8; 16];
    for j in 0..16 {
        sc[j] = ls[j] | (lm[j] << 4);
    }
    q2k_requant(x, &sc, f(d), f(dmin), &mut l);
    q2k_pack(out, &sc, &l, d, dmin);
}

// ---- Q3_K -------------------------------------------------------------------

fn q3k_decode_sc(scales: &[u8; 12], j: usize) -> i32 {
    let lo = if j < 8 {
        scales[j] & 0xF
    } else {
        scales[j - 8] >> 4
    } as i32;
    (lo | ((((scales[8 + j % 4] >> (2 * (j / 4))) & 3) as i32) << 4)) - 32
}

fn q3k_pack(out: &mut Vec<u8>, x: &[f32], scales: &[u8; 12], d: u16, l: &mut [i8; QK_K]) {
    for j in 0..16 {
        let dd = f(d) * q3k_decode_sc(scales, j) as f32;
        if dd == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let q = nearest_int(x[16 * j + ii] / dd);
            l[16 * j + ii] = ((-4i32).max(3i32.min(q)) + 4) as i8;
        }
    }
    let mut hmask = [0u8; QK_K / 8];
    let (mut m, mut hm) = (0usize, 1u8);
    for j in 0..QK_K {
        if l[j] > 3 {
            hmask[m] |= hm;
            l[j] -= 4;
        }
        m += 1;
        if m == QK_K / 8 {
            m = 0;
            hm <<= 1;
        }
    }
    out.extend_from_slice(&hmask);
    for j in (0..QK_K).step_by(128) {
        for k in 0..32 {
            out.push(
                (l[j + k] | (l[j + k + 32] << 2) | (l[j + k + 64] << 4) | (l[j + k + 96] << 6))
                    as u8,
            );
        }
    }
    out.extend_from_slice(scales);
    out.extend_from_slice(&d.to_le_bytes());
}

fn q3k_put_scale(scales: &mut [u8; 12], j: usize, mut l: i32) {
    if j < 8 {
        scales[j] = (l & 0xF) as u8;
    } else {
        scales[j - 8] |= ((l & 0xF) << 4) as u8;
    }
    l >>= 4;
    scales[j % 4 + 8] |= (l << (2 * (j / 4))) as u8;
}

fn q3k_block_ref(x: &[f32], out: &mut Vec<u8>) {
    let mut l = [0i8; QK_K];
    let mut scales = [0f32; 16];
    let (mut max_scale, mut amax) = (0f32, 0f32);
    for j in 0..16 {
        scales[j] = make_q3_quants(16, 4, &x[16 * j..], &mut l[16 * j..], true);
        let s = scales[j].abs();
        if s > amax {
            amax = s;
            max_scale = scales[j];
        }
    }
    let mut sc = [0u8; 12];
    let d;
    if max_scale != 0.0 {
        let iscale = -32.0 / max_scale;
        for j in 0..16 {
            let q = ((-32i32).max(31i32.min(nearest_int(iscale * scales[j])))) + 32;
            q3k_put_scale(&mut sc, j, q);
        }
        d = h(1.0 / iscale);
    } else {
        d = h(0.0);
    }
    q3k_pack(out, x, &sc, d, &mut l);
}

fn q3k_block_impl(x: &[f32], qw_blk: &[f32], out: &mut Vec<u8>) {
    let mut l = [0i8; QK_K];
    let (mut scales, mut sw, mut weight) = ([0f32; 16], [0f32; 16], [0f32; 16]);
    let mut ls = [0i8; 16];
    let mut sumx2 = 0f32;
    for v in &x[..QK_K] {
        sumx2 = (*v).mul_add(*v, sumx2);
    }
    let sigma2 = 2.0 * sumx2 / QK_K as f32;
    for j in 0..16 {
        let qw = &qw_blk[16 * j..];
        for k in 0..16 {
            weight[k] = qw[k] * (x[16 * j + k]).mul_add(x[16 * j + k], sigma2).sqrt();
        }
        let mut sumw = 0f32;
        for k in 0..16 {
            sumw += weight[k];
        }
        sw[j] = sumw;
        scales[j] = make_qx_quants(16, 4, &x[16 * j..], &mut l[16 * j..], 1, Some(&weight));
    }
    let mut sc = [0u8; 12];
    let d_block = make_qx_quants(16, 32, &scales, &mut ls, 1, Some(&sw));
    for j in 0..16 {
        q3k_put_scale(&mut sc, j, ls[j] as i32);
    }
    q3k_pack(out, x, &sc, h(d_block), &mut l);
}

// ---- Q4_K / Q5_K ------------------------------------------------------------

/// Shared tail: derive `L` from the packed scales, then pack nibbles (and for
/// Q5_K the fifth bit).
fn k45_finish(out: &mut Vec<u8>, x: &[f32], sc: &[u8; 12], d: u16, dmin: u16, five: bool) {
    let mut l = [0u8; QK_K];
    let maxq = if five { 31 } else { 15 };
    for j in 0..8 {
        let (s, m) = get_scale_min_k4(j, sc);
        let dd = f(d) * s as f32;
        if dd == 0.0 {
            continue;
        }
        let dm = f(dmin) * m as f32;
        for ii in 0..32 {
            let q = nearest_int((x[32 * j + ii] + dm) / dd);
            l[32 * j + ii] = 0i32.max(maxq.min(q)) as u8;
        }
    }
    out.extend_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&dmin.to_le_bytes());
    out.extend_from_slice(sc);
    if !five {
        for j in (0..QK_K).step_by(64) {
            for k in 0..32 {
                out.push(l[j + k] | (l[j + k + 32] << 4));
            }
        }
        return;
    }
    let mut qh = [0u8; 32];
    let mut ql = Vec::with_capacity(128);
    let (mut m1, mut m2) = (1u8, 2u8);
    for n in (0..QK_K).step_by(64) {
        for j in 0..32 {
            let mut l1 = l[n + j];
            if l1 > 15 {
                l1 -= 16;
                qh[j] |= m1;
            }
            let mut l2 = l[n + j + 32];
            if l2 > 15 {
                l2 -= 16;
                qh[j] |= m2;
            }
            ql.push(l1 | (l2 << 4));
        }
        m1 <<= 2;
        m2 <<= 2;
    }
    out.extend_from_slice(&qh);
    out.extend_from_slice(&ql);
}

fn k45_block_ref(x: &[f32], out: &mut Vec<u8>, five: bool) {
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut weights = [0f32; 32];
    let (mut mins, mut scales) = ([0f32; 8], [0f32; 8]);
    let (mut max_scale, mut max_min) = (0f32, 0f32);
    for j in 0..8 {
        let mut sum_x2 = 0f32;
        for k in 0..32 {
            sum_x2 = (x[32 * j + k]).mul_add(x[32 * j + k], sum_x2);
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        for k in 0..32 {
            weights[k] = av_x + x[32 * j + k].abs();
        }
        let (nmax, rmin, nstep) = if five { (31, -0.5, 15) } else { (15, -1.0, 20) };
        scales[j] = make_qkx2_quants(
            32,
            nmax,
            &x[32 * j..],
            &weights,
            &mut l[32 * j..],
            &mut mins[j],
            &mut laux,
            rmin,
            0.1,
            nstep,
            false,
        );
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }
    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut sc = [0u8; 12];
    for j in 0..8 {
        let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
        let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
        put_scale_min_k4(&mut sc, j, ls, lm);
    }
    k45_finish(out, x, &sc, h(max_scale / 63.0), h(max_min / 63.0), five);
}

fn k45_block_impl(x: &[f32], qw_blk: &[f32], out: &mut Vec<u8>, five: bool) {
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let (mut ls, mut lm) = ([0u8; 8], [0u8; 8]);
    let (mut mins, mut scales, mut sw) = ([0f32; 8], [0f32; 8], [0f32; 8]);
    let mut weights = [0f32; 32];
    let mut sum_x2 = 0f32;
    for v in &x[..QK_K] {
        sum_x2 = (*v).mul_add(*v, sum_x2);
    }
    let sigma2 = 2.0 * sum_x2 / QK_K as f32;
    for j in 0..8 {
        let qw = &qw_blk[32 * j..];
        for k in 0..32 {
            weights[k] = qw[k] * (x[32 * j + k]).mul_add(x[32 * j + k], sigma2).sqrt();
        }
        let mut sumw = 0f32;
        for k in 0..32 {
            sumw += weights[k];
        }
        sw[j] = sumw;
        let nmax = if five { 31 } else { 15 };
        scales[j] = make_qkx3_quants(
            32,
            nmax,
            &x[32 * j..],
            &weights,
            &mut l[32 * j..],
            &mut mins[j],
            &mut laux,
            -0.9,
            0.05,
            36,
            false,
        );
    }
    let d_block = make_qp_quants(8, 63, &scales, &mut ls, &sw);
    let m_block = make_qp_quants(8, 63, &mins, &mut lm, &sw);
    let mut sc = [0u8; 12];
    for j in 0..8 {
        put_scale_min_k4(&mut sc, j, ls[j].min(63), lm[j].min(63));
    }
    k45_finish(out, x, &sc, h(d_block), h(m_block), five);
}

// ---- Q6_K -------------------------------------------------------------------

fn q6k_block(x: &[f32], qw_blk: Option<&[f32]>, out: &mut Vec<u8>) {
    let mut l = [0i8; QK_K];
    let mut scales = [0f32; 16];
    let (mut max_scale, mut max_abs_scale) = (0f32, 0f32);
    for ib in 0..16 {
        let qw = qw_blk.map(|q| &q[16 * ib..]);
        let scale = make_qx_quants(16, 32, &x[16 * ib..], &mut l[16 * ib..], 1, qw);
        scales[ib] = scale;
        if scale.abs() > max_abs_scale {
            max_abs_scale = scale.abs();
            max_scale = scale;
        }
    }
    if max_abs_scale < GROUP_MAX_EPS {
        out.extend_from_slice(&[0u8; 208]);
        out.extend_from_slice(&h(0.0).to_le_bytes());
        return;
    }
    let iscale = -128.0 / max_scale;
    let d = h(1.0 / iscale);
    let mut sc = [0i8; 16];
    for ib in 0..16 {
        sc[ib] = 127i32.min(nearest_int(iscale * scales[ib])) as i8;
    }
    let mut lq = [0u8; QK_K];
    for j in 0..16 {
        let dd = f(d) * sc[j] as f32;
        if dd == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let q = nearest_int(x[16 * j + ii] / dd);
            lq[16 * j + ii] = ((-32i32).max(31i32.min(q)) + 32) as u8;
        }
    }
    let (mut ql, mut qh) = (Vec::with_capacity(128), Vec::with_capacity(64));
    for j in (0..QK_K).step_by(128) {
        let mut ql_part = [0u8; 64];
        for k in 0..32 {
            let (q1, q2, q3, q4) = (
                lq[j + k] & 0xF,
                lq[j + k + 32] & 0xF,
                lq[j + k + 64] & 0xF,
                lq[j + k + 96] & 0xF,
            );
            ql_part[k] = q1 | (q3 << 4);
            ql_part[k + 32] = q2 | (q4 << 4);
            qh.push(
                (lq[j + k] >> 4)
                    | ((lq[j + k + 32] >> 4) << 2)
                    | ((lq[j + k + 64] >> 4) << 4)
                    | ((lq[j + k + 96] >> 4) << 6),
            );
        }
        ql.extend_from_slice(&ql_part);
    }
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    out.extend(sc.iter().map(|&s| s as u8));
    out.extend_from_slice(&d.to_le_bytes());
}

/// Encode whole rows of `row_len` (a multiple of 256) elements. `imatrix` is
/// one weight per column, reused for every row, as `ggml_quantize_chunk` does.
pub fn encode_rows(kind: KKind, x: &[f32], row_len: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(
        row_len % QK_K == 0 && x.len() % row_len == 0,
        "K-quant rows must be whole blocks"
    );
    if let Some(q) = imatrix {
        assert!(q.len() >= row_len, "imatrix shorter than a row");
    }
    let mut out = Vec::with_capacity(x.len() / QK_K * kind.block_bytes());
    for row in x.chunks_exact(row_len) {
        for (i, blk) in row.chunks_exact(QK_K).enumerate() {
            let qw = imatrix.map(|q| &q[QK_K * i..QK_K * (i + 1)]);
            match (kind, qw) {
                (KKind::Q2K, None) => q2k_block_ref(blk, &mut out),
                (KKind::Q2K, Some(w)) => q2k_block_impl(blk, w, &mut out),
                (KKind::Q3K, None) => q3k_block_ref(blk, &mut out),
                (KKind::Q3K, Some(w)) => q3k_block_impl(blk, w, &mut out),
                (KKind::Q4K, None) => k45_block_ref(blk, &mut out, false),
                (KKind::Q4K, Some(w)) => k45_block_impl(blk, w, &mut out, false),
                (KKind::Q5K, None) => k45_block_ref(blk, &mut out, true),
                (KKind::Q5K, Some(w)) => k45_block_impl(blk, w, &mut out, true),
                (KKind::Q6K, w) => q6k_block(blk, w, &mut out),
            }
        }
    }
    out
}
