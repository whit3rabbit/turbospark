//! Block encoders. Each follows ggml's `quantize_row_*_ref` so the bytes can be
//! compared against `llama-quantize` output, and is checked against the
//! `compute` dequantizers in `tests.rs`.

use foundation::LogitValue as F16;

/// Target element types this phase can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q1_0,
    Q2_0,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Nvfp4,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Iq4Nl,
    Iq4Xs,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq1S,
    Iq1M,
    Iq1Xs,
    Iq1Xxs,
    Iq1Xxxs,
}

impl Target {
    /// Every producible target, in a stable order.
    pub const ALL: [Target; 31] = [
        Target::F32,
        Target::F16,
        Target::Bf16,
        Target::Q4_0,
        Target::Q4_1,
        Target::Q5_0,
        Target::Q5_1,
        Target::Q8_0,
        Target::Q1_0,
        Target::Q2_0,
        Target::Tq1_0,
        Target::Tq2_0,
        Target::Mxfp4,
        Target::Nvfp4,
        Target::Q2K,
        Target::Q3K,
        Target::Q4K,
        Target::Q5K,
        Target::Q6K,
        Target::Iq4Nl,
        Target::Iq4Xs,
        Target::Iq2Xxs,
        Target::Iq2Xs,
        Target::Iq2S,
        Target::Iq3Xxs,
        Target::Iq3S,
        Target::Iq1S,
        Target::Iq1M,
        Target::Iq1Xs,
        Target::Iq1Xxs,
        Target::Iq1Xxxs,
    ];

    pub fn parse(s: &str) -> Option<Target> {
        Some(match s.to_ascii_lowercase().as_str() {
            "f32" => Target::F32,
            "f16" => Target::F16,
            "bf16" => Target::Bf16,
            "q4_0" => Target::Q4_0,
            "q4_1" => Target::Q4_1,
            "q5_0" => Target::Q5_0,
            "q5_1" => Target::Q5_1,
            "q8_0" => Target::Q8_0,
            "q1_0" => Target::Q1_0,
            "q2_0" => Target::Q2_0,
            "tq1_0" => Target::Tq1_0,
            "tq2_0" => Target::Tq2_0,
            "mxfp4" => Target::Mxfp4,
            "nvfp4" => Target::Nvfp4,
            "q2_k" => Target::Q2K,
            "q3_k" => Target::Q3K,
            "q4_k" => Target::Q4K,
            "q5_k" => Target::Q5K,
            "q6_k" => Target::Q6K,
            "iq4_nl" => Target::Iq4Nl,
            "iq4_xs" => Target::Iq4Xs,
            "iq2_xxs" => Target::Iq2Xxs,
            "iq2_xs" => Target::Iq2Xs,
            "iq2_s" => Target::Iq2S,
            "iq3_xxs" => Target::Iq3Xxs,
            "iq3_s" => Target::Iq3S,
            "iq1_s" => Target::Iq1S,
            "iq1_m" => Target::Iq1M,
            "iq1_xs" => Target::Iq1Xs,
            "iq1_xxs" => Target::Iq1Xxs,
            "iq1_xxxs" => Target::Iq1Xxxs,
            _ => return None,
        })
    }

    /// Lower-case recipe name (inverse of [`Target::parse`]).
    pub fn name(self) -> &'static str {
        match self {
            Target::F32 => "f32",
            Target::F16 => "f16",
            Target::Bf16 => "bf16",
            Target::Q4_0 => "q4_0",
            Target::Q4_1 => "q4_1",
            Target::Q5_0 => "q5_0",
            Target::Q5_1 => "q5_1",
            Target::Q8_0 => "q8_0",
            Target::Q1_0 => "q1_0",
            Target::Q2_0 => "q2_0",
            Target::Tq1_0 => "tq1_0",
            Target::Tq2_0 => "tq2_0",
            Target::Mxfp4 => "mxfp4",
            Target::Nvfp4 => "nvfp4",
            Target::Q2K => "q2_k",
            Target::Q3K => "q3_k",
            Target::Q4K => "q4_k",
            Target::Q5K => "q5_k",
            Target::Q6K => "q6_k",
            Target::Iq4Nl => "iq4_nl",
            Target::Iq4Xs => "iq4_xs",
            Target::Iq2Xxs => "iq2_xxs",
            Target::Iq2Xs => "iq2_xs",
            Target::Iq2S => "iq2_s",
            Target::Iq3Xxs => "iq3_xxs",
            Target::Iq3S => "iq3_s",
            Target::Iq1S => "iq1_s",
            Target::Iq1M => "iq1_m",
            Target::Iq1Xs => "iq1_xs",
            Target::Iq1Xxs => "iq1_xxs",
            Target::Iq1Xxxs => "iq1_xxxs",
        }
    }

    /// The target for a ggml type id, if this crate can produce it.
    pub fn from_ggml_id(id: u32) -> Option<Target> {
        Target::ALL.into_iter().find(|t| t.ggml_id() == id)
    }

    /// ggml type id (`enum ggml_type`).
    pub fn ggml_id(self) -> u32 {
        match self {
            Target::F32 => 0,
            Target::F16 => 1,
            Target::Q4_0 => 2,
            Target::Q4_1 => 3,
            Target::Q5_0 => 6,
            Target::Q5_1 => 7,
            Target::Q8_0 => 8,
            Target::Tq1_0 => 34,
            Target::Tq2_0 => 35,
            Target::Mxfp4 => 39,
            Target::Nvfp4 => 40,
            Target::Q1_0 => 41,
            Target::Q2_0 => 42,
            Target::Q2K => 10,
            Target::Q3K => 11,
            Target::Q4K => 12,
            Target::Q5K => 13,
            Target::Q6K => 14,
            Target::Iq4Nl => 20,
            Target::Iq4Xs => 23,
            Target::Iq2Xxs => 16,
            Target::Iq2Xs => 17,
            Target::Iq2S => 22,
            Target::Iq3Xxs => 18,
            Target::Iq3S => 21,
            Target::Iq1S => 19,
            Target::Iq1M => 29,
            // Unsloth fork ids, clear of upstream's reserved 42..63.
            Target::Iq1Xs => 64,
            Target::Iq1Xxs => 65,
            Target::Iq1Xxxs => 66,
            Target::Bf16 => 30,
        }
    }

    /// `(elements, bytes)` per block.
    pub fn block(self) -> (usize, usize) {
        match self {
            Target::F32 => (1, 4),
            Target::F16 | Target::Bf16 => (1, 2),
            Target::Q4_0 => (32, 18),
            Target::Q4_1 => (32, 20),
            Target::Q5_0 => (32, 22),
            Target::Q5_1 => (32, 24),
            Target::Q8_0 => (32, 34),
            Target::Q1_0 => (128, 18),
            Target::Q2_0 => (64, 18),
            Target::Tq1_0 => (256, 54),
            Target::Tq2_0 => (256, 66),
            Target::Mxfp4 => (32, 17),
            Target::Nvfp4 => (64, 36),
            Target::Q2K => (256, 84),
            Target::Q3K => (256, 110),
            Target::Q4K => (256, 144),
            Target::Q5K => (256, 176),
            Target::Q6K => (256, 210),
            Target::Iq4Nl => (32, 18),
            Target::Iq4Xs => (256, 136),
            Target::Iq2Xxs => (256, 66),
            Target::Iq2Xs => (256, 74),
            Target::Iq2S => (256, 82),
            Target::Iq3Xxs => (256, 98),
            Target::Iq3S => (256, 110),
            Target::Iq1S => (256, 50),
            Target::Iq1M => (256, 56),
            Target::Iq1Xs => (256, 46),
            Target::Iq1Xxs => (256, 42),
            Target::Iq1Xxxs => (256, 38),
        }
    }

    /// `general.file_type` (`llama_ftype`) for a file made wholly of this type.
    pub fn file_type(self) -> u32 {
        match self {
            Target::F32 => 0,
            Target::F16 => 1,
            Target::Q4_0 => 2,
            Target::Q4_1 => 3,
            Target::Q8_0 => 7,
            // llama_ftype values from llama.h (Homebrew b-build), whole-type files.
            Target::Tq1_0 => 36,
            Target::Tq2_0 => 37,
            Target::Mxfp4 => 38,
            Target::Nvfp4 => 39,
            Target::Q1_0 => 40,
            Target::Q2_0 => 41,
            Target::Q5_0 => 8,
            Target::Q5_1 => 9,
            Target::Bf16 => 32,
            // llama_ftype MOSTLY_Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K. Cosmetic
            // for a mixed file (see module docs in mod.rs).
            Target::Q2K => 10,
            Target::Q3K => 12,
            Target::Q4K => 15,
            Target::Q5K => 17,
            Target::Q6K => 18,
            // MOSTLY_IQ4_NL, MOSTLY_IQ4_XS.
            Target::Iq4Nl => 25,
            Target::Iq4Xs => 30,
            // MOSTLY_IQ2_XXS, IQ2_XS, IQ2_S.
            Target::Iq2Xxs => 19,
            Target::Iq2Xs => 20,
            Target::Iq2S => 28,
            // MOSTLY_IQ3_XXS, IQ3_S.
            Target::Iq3Xxs => 23,
            Target::Iq3S => 26,
            // MOSTLY_IQ1_S, IQ1_M.
            Target::Iq1S => 24,
            Target::Iq1M => 31,
            // LLAMA_FTYPE_MOSTLY_IQ1_XS/XXS/XXXS in the Unsloth fork's llama.h.
            Target::Iq1Xs => 64,
            Target::Iq1Xxs => 65,
            Target::Iq1Xxxs => 66,
        }
    }

    /// ggml's encoders assert on a missing imatrix for these.
    pub fn requires_imatrix(self) -> bool {
        matches!(
            self,
            Target::Iq2Xxs
                | Target::Iq2Xs
                | Target::Iq1S
                | Target::Iq1Xs
                | Target::Iq1Xxs
                | Target::Iq1Xxxs
        )
    }

    /// llama.cpp's stricter policy (`tensor_requires_imatrix`): these very
    /// low-bit types are refused without one, except on `token_embd` and
    /// `output.weight`.
    pub fn policy_requires_imatrix(self) -> bool {
        matches!(
            self,
            Target::Iq3Xxs
                | Target::Iq2Xxs
                | Target::Iq2Xs
                | Target::Iq2S
                | Target::Iq1M
                | Target::Iq1S
                | Target::Iq1Xs
                | Target::Iq1Xxs
                | Target::Iq1Xxxs
        )
    }

    /// Encode whole rows of `row_len` elements. `imatrix` (one weight per
    /// column) only affects the K-quants; other types ignore it.
    pub fn encode_rows(self, src: &[f32], row_len: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
        use super::iqgrid::Kind;
        use super::kquants::{encode_rows, KKind};
        if let (Target::Q4_0 | Target::Q5_0, Some(im)) = (self, imatrix) {
            return self.legacy_weighted(src, row_len, im);
        }
        if let (Target::Q4_1 | Target::Q5_1, Some(im)) = (self, imatrix) {
            return self.legacy_weighted_min(src, row_len, im);
        }
        match self {
            Target::Iq4Nl => return super::iq4::encode_rows(false, src, row_len, imatrix),
            Target::Iq4Xs => return super::iq4::encode_rows(true, src, row_len, imatrix),
            Target::Iq2Xxs => return super::iq2::encode_rows(Kind::Iq2Xxs, src, row_len, imatrix),
            Target::Iq2Xs => return super::iq2::encode_rows(Kind::Iq2Xs, src, row_len, imatrix),
            Target::Iq2S => return super::iq2::encode_rows(Kind::Iq2S, src, row_len, imatrix),
            Target::Iq3Xxs => return super::iq3::encode_rows(Kind::Iq3Xxs, src, row_len, imatrix),
            Target::Iq3S => return super::iq3::encode_rows(Kind::Iq3S, src, row_len, imatrix),
            Target::Iq1S => return super::iq1::encode_rows(true, src, row_len, imatrix),
            Target::Iq1M => return super::iq1::encode_rows(false, src, row_len, imatrix),
            Target::Iq1Xs => {
                return super::iq1::encode_rows_fmt(
                    super::iq1::Fmt::Narrow(super::iq1::Narrow::Xs),
                    src,
                    row_len,
                    imatrix,
                )
            }
            Target::Iq1Xxs => {
                return super::iq1::encode_rows_fmt(
                    super::iq1::Fmt::Narrow(super::iq1::Narrow::Xxs),
                    src,
                    row_len,
                    imatrix,
                )
            }
            Target::Iq1Xxxs => {
                return super::iq1::encode_rows_fmt(
                    super::iq1::Fmt::Narrow(super::iq1::Narrow::Xxxs),
                    src,
                    row_len,
                    imatrix,
                )
            }
            _ => {}
        }
        let k = match self {
            Target::Q2K => KKind::Q2K,
            Target::Q3K => KKind::Q3K,
            Target::Q4K => KKind::Q4K,
            Target::Q5K => KKind::Q5K,
            Target::Q6K => KKind::Q6K,
            _ => return self.encode(src),
        };
        encode_rows(k, src, row_len, imatrix)
    }

    /// ggml's `quantize_row_q{4,5}_0_impl`: with an importance matrix these
    /// types switch to a weighted scale search, with `sigma2` taken over the
    /// whole row (unlike the K-quants, which take it per block).
    fn legacy_weighted(self, src: &[f32], row_len: usize, im: &[f32]) -> Vec<u8> {
        use super::kquants::make_qx_quants;
        assert!(im.len() >= row_len && src.len() % row_len == 0);
        let nmax = if self == Target::Q4_0 { 8 } else { 16 };
        let mut out = Vec::new();
        for row in src.chunks_exact(row_len) {
            let mut sum_x2 = 0f32;
            for v in row {
                sum_x2 = (*v).mul_add(*v, sum_x2);
            }
            let sigma2 = sum_x2 / row_len as f32;
            for (ib, xb) in row.chunks_exact(32).enumerate() {
                let (mut w, mut l) = ([0f32; 32], [0i8; 32]);
                for j in 0..32 {
                    w[j] = im[32 * ib + j] * (xb[j]).mul_add(xb[j], sigma2).sqrt();
                }
                let d = make_qx_quants(32, nmax, xb, &mut l, 1, Some(&w));
                out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
                if self == Target::Q4_0 {
                    for j in 0..16 {
                        out.push((l[j] as u8) | ((l[j + 16] as u8) << 4));
                    }
                } else {
                    let (mut qh, mut qs) = (0u32, [0u8; 16]);
                    for j in 0..16 {
                        let (x0, x1) = (l[j] as u8, l[j + 16] as u8);
                        qs[j] = (x0 & 0x0f) | ((x1 & 0x0f) << 4);
                        qh |= u32::from((x0 & 0x10) >> 4) << j;
                        qh |= u32::from((x1 & 0x10) >> 4) << (j + 16);
                    }
                    out.extend_from_slice(&qh.to_le_bytes());
                    out.extend_from_slice(&qs);
                }
            }
        }
        out
    }

    /// ggml's `quantize_row_q{4,5}_1_impl`: the imatrix variant of the
    /// scale-and-min legacy types. Same row-wide `sigma2` as the `_0` types,
    /// but the search is `make_qkx3_quants` and the block stores `-min`.
    fn legacy_weighted_min(self, src: &[f32], row_len: usize, im: &[f32]) -> Vec<u8> {
        use super::kquants::make_qkx3_quants;
        assert!(im.len() >= row_len && src.len() % row_len == 0);
        let nmax = if self == Target::Q4_1 { 15 } else { 31 };
        let mut out = Vec::new();
        for row in src.chunks_exact(row_len) {
            let mut sum_x2 = 0f32;
            for v in row {
                sum_x2 = (*v).mul_add(*v, sum_x2);
            }
            let sigma2 = sum_x2 / row_len as f32;
            for (ib, xb) in row.chunks_exact(32).enumerate() {
                let (mut w, mut l, mut laux) = ([0f32; 32], [0u8; 32], [0u8; 32]);
                for j in 0..32 {
                    w[j] = im[32 * ib + j] * (xb[j]).mul_add(xb[j], sigma2).sqrt();
                }
                let mut min = 0f32;
                let d = make_qkx3_quants(
                    32, nmax, xb, &w, &mut l, &mut min, &mut laux, -0.9, 0.05, 36, false,
                );
                out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
                out.extend_from_slice(&F16::from_f32(-min).to_bits().to_le_bytes());
                push_low_bits(self == Target::Q5_1, &l, &mut out);
            }
        }
        out
    }

    /// Encode `src`, whose length must be a whole number of blocks. K-quants
    /// need row structure, so use [`Target::encode_rows`] for those.
    pub fn encode(self, src: &[f32]) -> Vec<u8> {
        let (be, bb) = self.block();
        assert!(src.len() % be == 0, "encode: ragged input for {self:?}");
        let mut out = Vec::with_capacity(src.len() / be * bb);
        match self {
            Target::F32 => src
                .iter()
                .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
            Target::F16 => src
                .iter()
                .for_each(|v| out.extend_from_slice(&F16::from_f32(*v).to_bits().to_le_bytes())),
            Target::Bf16 => src
                .iter()
                .for_each(|v| out.extend_from_slice(&bf16_bits(*v).to_le_bytes())),
            Target::Q4_0 => src.chunks_exact(32).for_each(|b| q4_0_block(b, &mut out)),
            Target::Q4_1 => src.chunks_exact(32).for_each(|b| q4_1_block(b, &mut out)),
            Target::Q5_0 => src.chunks_exact(32).for_each(|b| q5_0_block(b, &mut out)),
            Target::Q5_1 => src.chunks_exact(32).for_each(|b| q5_1_block(b, &mut out)),
            Target::Q8_0 => src.chunks_exact(32).for_each(|b| q8_0_block(b, &mut out)),
            Target::Q1_0 => src.chunks_exact(128).for_each(|b| q1_0_block(b, &mut out)),
            Target::Q2_0 => src.chunks_exact(64).for_each(|b| q2_0_block(b, &mut out)),
            Target::Tq1_0 => src.chunks_exact(256).for_each(|b| tq1_0_block(b, &mut out)),
            Target::Tq2_0 => src.chunks_exact(256).for_each(|b| tq2_0_block(b, &mut out)),
            Target::Mxfp4 => src.chunks_exact(32).for_each(|b| mxfp4_block(b, &mut out)),
            Target::Nvfp4 => src.chunks_exact(64).for_each(|b| nvfp4_block(b, &mut out)),
            Target::Q2K
            | Target::Q3K
            | Target::Q4K
            | Target::Q5K
            | Target::Q6K
            | Target::Iq4Nl
            | Target::Iq4Xs
            | Target::Iq2Xxs
            | Target::Iq2Xs
            | Target::Iq2S
            | Target::Iq3Xxs
            | Target::Iq3S
            | Target::Iq1S
            | Target::Iq1M
            | Target::Iq1Xs
            | Target::Iq1Xxs
            | Target::Iq1Xxxs => return self.encode_rows(src, src.len().max(256), None),
        }
        out
    }
}

/// Round-to-nearest-even f32 -> bf16, matching ggml's `ggml_fp32_to_bf16`
/// (NaN keeps a quiet payload).
fn bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    if v.is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16
}

/// Signed value of the largest magnitude in the block, as ggml picks it.
fn signed_max(block: &[f32]) -> f32 {
    let (mut amax, mut max) = (0f32, 0f32);
    for &v in block {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    max
}

fn q4_0_block(b: &[f32], out: &mut Vec<u8>) {
    let d = signed_max(b) / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
    for j in 0..16 {
        let x0 = b[j] * id;
        let x1 = b[16 + j] * id;
        let q0 = ((x0 + 8.5) as i8).min(15) as u8;
        let q1 = ((x1 + 8.5) as i8).min(15) as u8;
        out.push(q0 | (q1 << 4));
    }
}

fn q5_0_block(b: &[f32], out: &mut Vec<u8>) {
    let d = signed_max(b) / -16.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
    let mut qh = 0u32;
    let mut qs = [0u8; 16];
    for j in 0..16 {
        let q0 = ((b[j] * id + 16.5) as i8).min(31) as u8;
        let q1 = ((b[16 + j] * id + 16.5) as i8).min(31) as u8;
        qs[j] = (q0 & 0x0f) | ((q1 & 0x0f) << 4);
        qh |= u32::from((q0 & 0x10) >> 4) << j;
        qh |= u32::from((q1 & 0x10) >> 4) << (j + 16);
    }
    out.extend_from_slice(&qh.to_le_bytes());
    out.extend_from_slice(&qs);
}

/// Min and max of a block as ggml's `_1` reference finds them (`v < min`,
/// `v > max`, so a NaN never wins).
fn min_max(b: &[f32]) -> (f32, f32) {
    let (mut min, mut max) = (f32::MAX, -f32::MAX);
    for &v in b {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (min, max)
}

/// Packs 32 quants (low nibbles, then for Q5 the fifth bits as a u32) in
/// ggml's layout: element `j` and `j + 16` share byte `j`.
fn push_low_bits(five_bit: bool, l: &[u8; 32], out: &mut Vec<u8>) {
    let (mut qh, mut qs) = (0u32, [0u8; 16]);
    for j in 0..16 {
        let (x0, x1) = (l[j], l[j + 16]);
        qs[j] = (x0 & 0x0f) | ((x1 & 0x0f) << 4);
        qh |= u32::from((x0 & 0x10) >> 4) << j;
        qh |= u32::from((x1 & 0x10) >> 4) << (j + 16);
    }
    if five_bit {
        out.extend_from_slice(&qh.to_le_bytes());
    }
    out.extend_from_slice(&qs);
}

fn q4_1_block(b: &[f32], out: &mut Vec<u8>) {
    let (min, max) = min_max(b);
    let d = (max - min) / 15.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
    out.extend_from_slice(&F16::from_f32(min).to_bits().to_le_bytes());
    for j in 0..16 {
        let q0 = (((b[j] - min) * id + 0.5) as i8).min(15) as u8;
        let q1 = (((b[16 + j] - min) * id + 0.5) as i8).min(15) as u8;
        out.push(q0 | (q1 << 4));
    }
}

fn q5_1_block(b: &[f32], out: &mut Vec<u8>) {
    let (min, max) = min_max(b);
    let d = (max - min) / 31.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
    out.extend_from_slice(&F16::from_f32(min).to_bits().to_le_bytes());
    let mut l = [0u8; 32];
    for j in 0..16 {
        l[j] = ((b[j] - min) * id + 0.5) as u8;
        l[j + 16] = ((b[16 + j] - min) * id + 0.5) as u8;
    }
    push_low_bits(true, &l, out);
}

fn q8_0_block(b: &[f32], out: &mut Vec<u8>) {
    let amax = b.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
    for &v in b {
        out.push((v * id).round() as i8 as u8);
    }
}

// ---- types ggml added after the K and IQ families ---------------------------
//
// Each follows ggml's `quantize_row_*_ref`. The rules below were recovered
// from ggml's OUTPUT (the fixtures in `tests/fixtures/gguf_quantize`), not read
// off a spec page, and the ties and clamps are the ones ggml's bytes showed.

/// Q1_0: `d` is the mean magnitude, a set bit is `x >= 0` (so zero is `+d`).
fn q1_0_block(b: &[f32], out: &mut Vec<u8>) {
    let mut sum_abs = 0f32;
    for &v in b {
        sum_abs += v.abs();
    }
    let d = sum_abs / 128.0;
    out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
    for byte in b.chunks_exact(8) {
        let mut bits = 0u8;
        for (i, &v) in byte.iter().enumerate() {
            bits |= u8::from(v >= 0.0) << i;
        }
        out.push(bits);
    }
}

/// Largest magnitude of a block (NaN never wins, as in ggml's `fmaxf` loops).
fn abs_max(b: &[f32]) -> f32 {
    b.iter().fold(0f32, |a, &v| a.max(v.abs()))
}

/// Q2_0: `d = amax` stored as f16 but applied UNROUNDED (`id = 1 / amax`),
/// quant `round(x * id) + 1` with ties away from zero, clamped to 0..=3, four
/// to a byte, element `i` at bits `2 * (i % 4)` of byte `i / 4`.
fn q2_0_block(b: &[f32], out: &mut Vec<u8>) {
    let d = abs_max(b);
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
    for quad in b.chunks_exact(4) {
        let mut byte = 0u8;
        for (n, &v) in quad.iter().enumerate() {
            let q = ((v * id).round() as i32 + 1).clamp(0, 3) as u8;
            byte |= q << (2 * n);
        }
        out.push(byte);
    }
}

/// TQ2_0: `d = amax`, trit `round(x * id) + 1` (ties away from zero) packed
/// four to a byte; the block is two 128-element halves, and within one half
/// element `n * 32 + m` is bits `2n` of byte `m`.
fn tq2_0_block(b: &[f32], out: &mut Vec<u8>) {
    let d = abs_max(b);
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut qs = [0u8; 64];
    for j in [0usize, 32] {
        for m in 0..32 {
            let mut q = 0u8;
            for n in 0..4 {
                let xi = ((b[j * 4 + n * 32 + m] * id).round() as i32 + 1) as u8;
                q += (xi & 3) << (2 * n);
            }
            qs[j + m] = q;
        }
    }
    out.extend_from_slice(&qs);
    out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
}

/// Five (or four, for `qh`) trits to one byte: base-3 accumulate, then ggml's
/// ceiling division by 243 scaled to 8 bits. `extra_shift` is the `q *= 3`
/// ggml applies to the four-trit `qh` bytes only.
fn tq1_0_pack(trits: &[i32], extra_shift: bool) -> u8 {
    let mut q = 0u32;
    for &t in trits {
        q = q * 3 + t as u32;
    }
    if extra_shift {
        q *= 3;
    }
    (q * 256).div_ceil(243) as u8
}

/// TQ1_0: `d = amax`, trit `round(x * id) + 1` (ties away from zero). Layout:
/// 32 bytes of five trit-planes of 32, 16 bytes of five planes of 16, then 4
/// `qh` bytes of four planes of 4 (the same split the decoder reads).
// Index loops mirror ggml's `quantize_row_tq1_0_ref` one for one.
#[allow(clippy::needless_range_loop)]
fn tq1_0_block(b: &[f32], out: &mut Vec<u8>) {
    let d = abs_max(b);
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let trit = |i: usize| (b[i] * id).round() as i32 + 1;
    let mut qs = [0u8; 48];
    let mut qh = [0u8; 4];
    for m in 0..32 {
        let t: Vec<i32> = (0..5).map(|n| trit(m + n * 32)).collect();
        qs[m] = tq1_0_pack(&t, false);
    }
    for m in 0..16 {
        let t: Vec<i32> = (0..5).map(|n| trit(160 + m + n * 16)).collect();
        qs[32 + m] = tq1_0_pack(&t, false);
    }
    for m in 0..4 {
        let t: Vec<i32> = (0..4).map(|n| trit(240 + m + n * 4)).collect();
        qh[m] = tq1_0_pack(&t, true);
    }
    out.extend_from_slice(&qs);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
}

/// ggml's `kvalues_mxfp4` (the second zero at index 8 is real).
const MXFP4_KVALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// ggml's E8M0-to-fp32 "half" scale: `2^(e - 128)`, with the two subnormal
/// bytes built by shifting a fixed pattern.
fn e8m0_half(e: u8) -> f32 {
    f32::from_bits(if e < 2 {
        0x0020_0000u32 << e
    } else {
        u32::from(e - 1) << 23
    })
}

/// Index of the code whose scaled value is nearest `x` (first wins a tie).
fn best_mxfp4_index(x: f32, d: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = (f32::from(MXFP4_KVALUES[0]) * d - x).abs();
    for (i, &k) in MXFP4_KVALUES.iter().enumerate().skip(1) {
        let err = (f32::from(k) * d - x).abs();
        if err < best_err {
            best_err = err;
            best = i as u8;
        }
    }
    best
}

/// MXFP4: shared exponent `floor(log2(amax)) - 2 + 127` (zero block: 0), the
/// scale is `e8m0_half(e)`, and each element takes the nearest codebook entry.
fn mxfp4_block(b: &[f32], out: &mut Vec<u8>) {
    let amax = abs_max(b);
    let e: u8 = if amax > 0.0 {
        (amax.log2().floor() as i32 - 2 + 127) as u8
    } else {
        0
    };
    let d = e8m0_half(e);
    out.push(e);
    for j in 0..16 {
        out.push(best_mxfp4_index(b[j], d) | (best_mxfp4_index(b[16 + j], d) << 4));
    }
}

/// NVFP4 scale byte to value: unsigned E4M3, bit 7 ignored, `0x7F` is zero.
fn ue4m3_value(b: u8) -> f32 {
    if b & 0x7F == 0x7F {
        return 0.0;
    }
    let e = i32::from((b >> 3) & 15);
    let m = f32::from(b & 7);
    if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

const NVFP4_E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// NVFP4: per 16-element sub-block the scale byte is the E4M3 value nearest
/// `amax / 6` (rules measured from ggml in the loop below), and each element
/// takes the nearest E2M1 code under that scale.
fn nvfp4_block(b: &[f32], out: &mut Vec<u8>) {
    let mut scales = [0u8; 4];
    let mut qs = [0u8; 32];
    for s in 0..4 {
        let sub = &b[s * 16..(s + 1) * 16];
        let target = abs_max(sub) / 6.0;
        // Measured from ggml's bytes, not assumed: a tie goes to the HIGHER
        // code, the search only ever lands on 0..=0x77 (240 is the largest
        // value it picks by nearness), and a target at or above 248 (the
        // midpoint of 240 and 256) jumps straight to 0x7E (448), skipping
        // 0x78..=0x7D. The golden fixture pins all three on outlier blocks.
        let mut best = 0u8;
        if target >= 248.0 {
            best = 0x7E;
        } else {
            let mut best_err = (ue4m3_value(0) - target).abs();
            for c in 1u8..=0x77 {
                let err = (ue4m3_value(c) - target).abs();
                if err <= best_err {
                    best_err = err;
                    best = c;
                }
            }
        }
        scales[s] = best;
        let d = ue4m3_value(best);
        for (j, &v) in sub.iter().enumerate() {
            let mut bi = 0u8;
            let mut bd = (NVFP4_E2M1[0] * d - v).abs();
            for (i, &k) in NVFP4_E2M1.iter().enumerate().skip(1) {
                let err = (k * d - v).abs();
                if err < bd {
                    bd = err;
                    bi = i as u8;
                }
            }
            if j < 8 {
                qs[s * 8 + j] |= bi;
            } else {
                qs[s * 8 + j - 8] |= bi << 4;
            }
        }
    }
    out.extend_from_slice(&scales);
    out.extend_from_slice(&qs);
}
