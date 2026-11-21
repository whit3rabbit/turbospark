//! CPU reference for LookAtMyQuant (LAMQ) weight codes: trellis (`ht2`,
//! `ht3`, `rtr1`..`rtr4`) and Lloyd-Max level (`lm6`, `lm8`) rows under a
//! randomized Kronecker Hadamard rotation. Spec: `docs/LAMQ.md`.
//!
//! PORT-LOCAL: transcribed from the upstream reference decoders
//! (`runtime_mlx/inkling/kernels.py`, MIT). Accumulation is f64 so this is an
//! oracle for the Metal kernels, not a copy of their rounding order.
//!
//! Easy to get fluently wrong:
//! * Column `j`'s 12-bit state is read at bit `(n - 1 - c) * k` where `c` is
//!   the LAST column of its group (symbols are stored last column first), and
//!   words past the row end read as zero.
//! * `H = kron(A, B)` with A the Sylvester (outer) factor. Paley blocks are
//!   not symmetric, so the output rotation needs the transposes.
//! * Signs go on BEFORE the input transform and AFTER the output transform.

/// Golden-ratio multiplier of the hashed (`ht*`) codes.
pub const HASH_MULT: u32 = 0x9E37_79B1;
/// Trellis state width in bits.
pub const STATE_BITS: u32 = 12;
const STATE_MASK: u32 = (1 << STATE_BITS) - 1;

/// One LAMQ weight code. `ht1` exists upstream as a format id but ships no
/// table and no tensor uses it, so it is not representable here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LamqCode {
    Ht2,
    Ht3,
    Rtr1,
    Rtr2,
    Rtr3,
    Rtr4,
    Lm6,
    Lm8,
}

impl LamqCode {
    /// The upstream name (`fmt` / `opt` in the metas).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "ht2" => Self::Ht2,
            "ht3" => Self::Ht3,
            "rtr1" => Self::Rtr1,
            "rtr2" => Self::Rtr2,
            "rtr3" => Self::Rtr3,
            "rtr4" => Self::Rtr4,
            "lm6" => Self::Lm6,
            "lm8" => Self::Lm8,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Ht2 => "ht2",
            Self::Ht3 => "ht3",
            Self::Rtr1 => "rtr1",
            Self::Rtr2 => "rtr2",
            Self::Rtr3 => "rtr3",
            Self::Rtr4 => "rtr4",
            Self::Lm6 => "lm6",
            Self::Lm8 => "lm8",
        }
    }

    /// Upstream expert format id (`manifest.json` `fmt_ids`). Level codes
    /// have none: they never appear as expert formats.
    pub fn from_fid(fid: i32) -> Option<Self> {
        Some(match fid {
            1 => Self::Ht2,
            2 => Self::Ht3,
            3 => Self::Rtr1,
            4 => Self::Rtr2,
            5 => Self::Rtr3,
            6 => Self::Rtr4,
            _ => return None,
        })
    }

    /// Bits per weight in the stream.
    pub fn k(self) -> u32 {
        match self {
            Self::Rtr1 => 1,
            Self::Ht2 | Self::Rtr2 => 2,
            Self::Ht3 | Self::Rtr3 => 3,
            Self::Rtr4 => 4,
            Self::Lm6 => 6,
            Self::Lm8 => 8,
        }
    }

    /// Weights produced per table lookup.
    pub fn g(self) -> usize {
        match self {
            Self::Ht2 => 4,
            Self::Ht3 => 2,
            _ => 1,
        }
    }

    pub fn is_trellis(self) -> bool {
        !matches!(self, Self::Lm6 | Self::Lm8)
    }

    /// Table rows (each `g` wide): 512 / 1024 for the hashed codes, 4096 for
    /// row trellis, the level count for `lm*`.
    pub fn table_rows(self) -> usize {
        match self {
            Self::Ht2 => 512,
            Self::Ht3 => 1024,
            Self::Lm6 => 64,
            Self::Lm8 => 256,
            _ => 1 << STATE_BITS,
        }
    }

    /// Right shift of the hashed index, or None for a direct index.
    pub fn hash_shift(self) -> Option<u32> {
        match self {
            Self::Ht2 => Some(32 - 9),
            Self::Ht3 => Some(32 - 10),
            _ => None,
        }
    }

    /// Bytes one row of `n` columns takes, or None when `n` does not fit the
    /// code's packing (trellis rows are whole u32 words, `ht*` whole groups,
    /// `lm6` whole 3-byte quads).
    pub fn row_bytes(self, n: usize) -> Option<usize> {
        if n == 0 || n % self.g() != 0 {
            return None;
        }
        let bits = n.checked_mul(self.k() as usize)?;
        match self {
            Self::Lm8 => Some(n),
            Self::Lm6 => (n % 4 == 0).then_some(n / 4 * 3),
            _ => (bits % 32 == 0).then_some(bits / 8),
        }
    }
}

impl LamqCode {
    /// The code's id in an install's aux header. Trellis ids equal the
    /// upstream expert `fmt_ids`; level codes take upstream's 10 and 11.
    pub fn id(self) -> u8 {
        match self {
            Self::Ht2 => 1,
            Self::Ht3 => 2,
            Self::Rtr1 => 3,
            Self::Rtr2 => 4,
            Self::Rtr3 => 5,
            Self::Rtr4 => 6,
            Self::Lm6 => 10,
            Self::Lm8 => 11,
        }
    }

    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            10 => Some(Self::Lm6),
            11 => Some(Self::Lm8),
            _ => Self::from_fid(i32::from(id)),
        }
    }
}

/// What one LAMQ resident entry holds (the `kind` byte of its aux header).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LamqKind {
    /// A coded `m x n` matrix: payload = codes, scales = f16 per row, aux =
    /// header, `signs_in` (n), `signs_out` (m) when `out_rot`, zero pad to 4,
    /// then F32 levels for `lm*`.
    Matrix,
    /// A bucket permutation: payload = i32 token id per packed position,
    /// `n` positions; no scales; aux = header only.
    Order,
    /// Two sign vectors: payload = `n` then `m` int8 signs; aux = header.
    Signs,
}

/// The 16-byte header at the start of a LAMQ entry's aux (bias) region:
/// `b"LQ"`, version, kind, code id, flags (bit 0 = out_rot), 2 reserved
/// bytes, then `n` and `m` as u32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LamqAuxHeader {
    pub kind: LamqKind,
    pub code: Option<LamqCode>,
    pub out_rot: bool,
    pub n: u32,
    pub m: u32,
}

pub const LAMQ_AUX_VERSION: u8 = 1;
pub const LAMQ_AUX_HEADER_BYTES: usize = 16;

impl LamqAuxHeader {
    pub fn encode(&self) -> [u8; LAMQ_AUX_HEADER_BYTES] {
        let mut h = [0u8; LAMQ_AUX_HEADER_BYTES];
        h[0] = b'L';
        h[1] = b'Q';
        h[2] = LAMQ_AUX_VERSION;
        h[3] = match self.kind {
            LamqKind::Matrix => 1,
            LamqKind::Order => 2,
            LamqKind::Signs => 3,
        };
        h[4] = self.code.map_or(0, LamqCode::id);
        h[5] = u8::from(self.out_rot);
        h[8..12].copy_from_slice(&self.n.to_le_bytes());
        h[12..16].copy_from_slice(&self.m.to_le_bytes());
        h
    }

    /// Refuses an unknown magic, version, kind, code or flag, a matrix with
    /// no code, and a code or flag on a non-matrix entry.
    pub fn parse(b: &[u8]) -> Result<Self, String> {
        if b.len() < LAMQ_AUX_HEADER_BYTES || &b[..2] != b"LQ" {
            return Err("LAMQ aux header missing".into());
        }
        if b[2] != LAMQ_AUX_VERSION {
            return Err(format!("LAMQ aux version {} unknown", b[2]));
        }
        let kind = match b[3] {
            1 => LamqKind::Matrix,
            2 => LamqKind::Order,
            3 => LamqKind::Signs,
            k => return Err(format!("LAMQ entry kind {k} unknown")),
        };
        let code = match b[4] {
            0 => None,
            id => Some(LamqCode::from_id(id).ok_or_else(|| format!("LAMQ code id {id} unknown"))?),
        };
        if b[5] > 1 || b[6] != 0 || b[7] != 0 {
            return Err("LAMQ aux flags or reserved bytes set".into());
        }
        let out_rot = b[5] == 1;
        if (kind == LamqKind::Matrix) != code.is_some() || (out_rot && kind != LamqKind::Matrix) {
            return Err(format!(
                "LAMQ {kind:?} entry with code {code:?} out_rot {out_rot}"
            ));
        }
        let n = u32::from_le_bytes(b[8..12].try_into().unwrap());
        let m = u32::from_le_bytes(b[12..16].try_into().unwrap());
        Ok(Self {
            kind,
            code,
            out_rot,
            n,
            m,
        })
    }

    /// Exact aux length this header implies.
    pub fn aux_len(&self) -> usize {
        let (n, m) = (self.n as usize, self.m as usize);
        match (self.kind, self.code) {
            (LamqKind::Matrix, Some(code)) => {
                let signs = n + if self.out_rot { m } else { 0 };
                let base = (LAMQ_AUX_HEADER_BYTES + signs).div_ceil(4) * 4;
                base + if code.is_trellis() {
                    0
                } else {
                    code.table_rows() * 4
                }
            }
            _ => LAMQ_AUX_HEADER_BYTES,
        }
    }
}

/// Builds a matrix entry's aux region.
pub fn encode_matrix_aux(
    code: LamqCode,
    n: usize,
    m: usize,
    signs_in: &[i8],
    signs_out: Option<&[i8]>,
    levels: Option<&[f32]>,
) -> Vec<u8> {
    let h = LamqAuxHeader {
        kind: LamqKind::Matrix,
        code: Some(code),
        out_rot: signs_out.is_some(),
        n: n as u32,
        m: m as u32,
    };
    let mut out = h.encode().to_vec();
    out.extend(signs_in.iter().map(|&s| s as u8));
    if let Some(s) = signs_out {
        out.extend(s.iter().map(|&v| v as u8));
    }
    out.resize(out.len().div_ceil(4) * 4, 0);
    if let Some(l) = levels {
        for v in l {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    debug_assert_eq!(out.len(), h.aux_len());
    out
}

/// A matrix entry's aux, parsed.
#[derive(Clone, Debug, PartialEq)]
pub struct LamqMatrixAux {
    pub header: LamqAuxHeader,
    pub signs_in: Vec<i8>,
    pub signs_out: Option<Vec<i8>>,
    /// The matrix's own levels, for `lm*` codes.
    pub levels: Option<Vec<f32>>,
}

/// Parses and checks a matrix entry's aux region.
pub fn parse_matrix_aux(aux: &[u8]) -> Result<LamqMatrixAux, String> {
    let h = LamqAuxHeader::parse(aux)?;
    let code = match (h.kind, h.code) {
        (LamqKind::Matrix, Some(c)) => c,
        _ => return Err(format!("LAMQ entry is {:?}, not a matrix", h.kind)),
    };
    if aux.len() != h.aux_len() {
        return Err(format!(
            "LAMQ aux is {} bytes, header implies {}",
            aux.len(),
            h.aux_len()
        ));
    }
    let (n, m) = (h.n as usize, h.m as usize);
    let at = LAMQ_AUX_HEADER_BYTES;
    let signs = |b: &[u8]| -> Result<Vec<i8>, String> {
        b.iter()
            .map(|&v| match v as i8 {
                1 => Ok(1),
                -1 => Ok(-1),
                o => Err(format!("LAMQ sign {o} is not +1/-1")),
            })
            .collect()
    };
    let s_in = signs(&aux[at..at + n])?;
    let s_out = if h.out_rot {
        Some(signs(&aux[at + n..at + n + m])?)
    } else {
        None
    };
    let lv_at = (at + n + if h.out_rot { m } else { 0 }).div_ceil(4) * 4;
    if aux[at + n + if h.out_rot { m } else { 0 }..lv_at]
        .iter()
        .any(|&b| b != 0)
    {
        return Err("LAMQ aux pad bytes are not zero".into());
    }
    let levels = (!code.is_trellis()).then(|| {
        aux[lv_at..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    });
    Ok(LamqMatrixAux {
        header: h,
        signs_in: s_in,
        signs_out: s_out,
        levels,
    })
}

/// The 12-bit trellis state at bit `bp` of a row's little-endian u32 stream.
/// Words past the end read as zero.
pub fn window(words: &[u32], bp: usize) -> u32 {
    let wi = bp >> 5;
    let off = (bp & 31) as u32;
    let lo = words.get(wi).copied().unwrap_or(0) >> off;
    let hi = if off == 0 {
        0
    } else {
        words.get(wi + 1).copied().unwrap_or(0) << (32 - off)
    };
    (lo | hi) & STATE_MASK
}

/// Table row for a state under `code`.
pub fn table_index(code: LamqCode, state: u32) -> usize {
    match code.hash_shift() {
        Some(shift) => (state.wrapping_mul(HASH_MULT) >> shift) as usize,
        None => state as usize,
    }
}

/// Decodes one trellis row (before the per-row scale). `table` is
/// `table_rows * g` values, row-major. Panics on a length mismatch: callers
/// validate shapes at intake.
pub fn decode_trellis_row(code: LamqCode, words: &[u32], n: usize, table: &[f32]) -> Vec<f32> {
    assert!(code.is_trellis(), "{} is not a trellis code", code.name());
    let g = code.g();
    let k = code.k() as usize;
    assert_eq!(
        table.len(),
        code.table_rows() * g,
        "{} table length",
        code.name()
    );
    assert_eq!(
        words.len() * 4,
        code.row_bytes(n).expect("row width fits the code"),
        "{} row words",
        code.name()
    );
    let mut out = vec![0.0f32; n];
    for j in (0..n).step_by(g) {
        let last = j + g - 1;
        let s = window(words, (n - 1 - last) * k);
        let e = table_index(code, s) * g;
        out[j..j + g].copy_from_slice(&table[e..e + g]);
    }
    out
}

/// Level indices of one `lm6` / `lm8` row.
pub fn level_indices(code: LamqCode, bytes: &[u8], n: usize) -> Vec<usize> {
    assert_eq!(
        bytes.len(),
        code.row_bytes(n).expect("row width fits the code")
    );
    match code {
        LamqCode::Lm8 => bytes.iter().map(|&b| b as usize).collect(),
        LamqCode::Lm6 => {
            let mut out = Vec::with_capacity(n);
            for q in bytes.chunks_exact(3) {
                let v = u32::from(q[0]) | u32::from(q[1]) << 8 | u32::from(q[2]) << 16;
                for t in 0..4 {
                    out.push(((v >> (6 * t)) & 63) as usize);
                }
            }
            out
        }
        _ => panic!("{} is not a level code", code.name()),
    }
}

/// Decodes one level row (before the per-row scale).
pub fn decode_level_row(code: LamqCode, bytes: &[u8], n: usize, levels: &[f32]) -> Vec<f32> {
    assert_eq!(
        levels.len(),
        code.table_rows(),
        "{} levels length",
        code.name()
    );
    level_indices(code, bytes, n)
        .into_iter()
        .map(|i| levels[i])
        .collect()
}

/// A row's packed codes: u32 words for trellis, bytes for levels.
#[derive(Clone, Copy, Debug)]
pub enum RowCodes<'a> {
    Words(&'a [u32]),
    Bytes(&'a [u8]),
}

/// Decodes one row of either kind against its table or levels.
pub fn decode_row(code: LamqCode, row: RowCodes<'_>, n: usize, table: &[f32]) -> Vec<f32> {
    match row {
        RowCodes::Words(w) => decode_trellis_row(code, w, n, table),
        RowCodes::Bytes(b) => decode_level_row(code, b, n, table),
    }
}

/// Sylvester Hadamard of order `n` (a power of two), row-major.
pub fn sylvester(n: usize) -> Vec<f64> {
    assert!(n.is_power_of_two());
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = if (i & j).count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
        }
    }
    h
}

fn is_prime(q: usize) -> bool {
    q >= 2 && (2..).take_while(|d| d * d <= q).all(|d| q % d != 0)
}

/// Paley-I Hadamard of order `q + 1` for a prime `q = 3 mod 4`, row-major:
/// `S + I` with `S[0][1..] = 1`, `S[1..][0] = -1`, `S[1+i][1+j] = chi(j - i)`.
pub fn paley(q: usize) -> Vec<f64> {
    assert!(
        is_prime(q) && q % 4 == 3,
        "Paley-I needs a prime q = 3 mod 4, got {q}"
    );
    let mut square = vec![false; q];
    for x in 1..q {
        square[x * x % q] = true;
    }
    let chi = |a: usize| -> f64 {
        if a == 0 {
            0.0
        } else if square[a] {
            1.0
        } else {
            -1.0
        }
    };
    let p = q + 1;
    let mut h = vec![0.0; p * p];
    for j in 1..p {
        h[j] = 1.0;
        h[j * p] = -1.0;
    }
    for i in 0..q {
        for j in 0..q {
            h[(i + 1) * p + (j + 1)] = chi((j + q - i) % q);
        }
    }
    for d in 0..p {
        h[d * p + d] += 1.0;
    }
    h
}

/// The two Kronecker factors of `H_n`: `(A, |A|, B, |B|)`.
fn factors(n: usize) -> Option<(Vec<f64>, usize, Vec<f64>, usize)> {
    if n == 0 {
        return None;
    }
    let top = usize::BITS - 1 - n.leading_zeros();
    for a in (0..=top).rev() {
        let pa = 1usize << a;
        if n % pa != 0 {
            continue;
        }
        let p = n / pa;
        if p == 1 {
            let (x, y) = (1usize << (a / 2), 1usize << (a - a / 2));
            return Some((sylvester(x), x, sylvester(y), y));
        }
        // Upstream also admits p == 2, which this descending search never
        // reaches (n = 2^(a+1) matches p == 1 one step earlier).
        if p > 2 && is_prime(p - 1) && (p - 1) % 4 == 3 {
            return Some((sylvester(pa), pa, paley(p - 1), p));
        }
    }
    None
}

/// Randomized Hadamard transform of width `n`, orthonormal.
#[derive(Clone, Debug)]
pub struct Rht {
    pub n: usize,
    a: Vec<f64>,
    na: usize,
    b: Vec<f64>,
    nb: usize,
}

impl Rht {
    /// None when no Sylvester x Paley-I factorization of `n` exists.
    pub fn new(n: usize) -> Option<Self> {
        let (a, na, b, nb) = factors(n)?;
        Some(Self { n, a, na, b, nb })
    }

    /// The dense `H_n = kron(A, B)` (unnormalized), for tests and small n.
    pub fn dense(&self) -> Vec<f64> {
        let n = self.n;
        let mut h = vec![0.0; n * n];
        for i in 0..self.na {
            for p in 0..self.na {
                let av = self.a[i * self.na + p];
                for j in 0..self.nb {
                    for q in 0..self.nb {
                        h[(i * self.nb + j) * n + p * self.nb + q] = av * self.b[j * self.nb + q];
                    }
                }
            }
        }
        h
    }

    /// `H (s * x) / sqrt(n)`: `A X B^T` on `x` reshaped `|A| x |B|`.
    pub fn apply_in(&self, x: &[f32], signs: &[i8]) -> Vec<f64> {
        assert_eq!(x.len(), self.n);
        assert_eq!(signs.len(), self.n);
        let xs: Vec<f64> = x
            .iter()
            .zip(signs)
            .map(|(&v, &s)| f64::from(v) * f64::from(s))
            .collect();
        let (na, nb) = (self.na, self.nb);
        // T = X B^T, then Y = A T.
        let mut t = vec![0.0; self.n];
        for p in 0..na {
            for j in 0..nb {
                t[p * nb + j] = (0..nb).map(|q| xs[p * nb + q] * self.b[j * nb + q]).sum();
            }
        }
        let inv = 1.0 / (self.n as f64).sqrt();
        let mut y = vec![0.0; self.n];
        for i in 0..na {
            for j in 0..nb {
                y[i * nb + j] = (0..na)
                    .map(|p| self.a[i * na + p] * t[p * nb + j])
                    .sum::<f64>()
                    * inv;
            }
        }
        y
    }

    /// `s * (H^T y) / sqrt(n)`: `A^T Y B`, signs applied last.
    pub fn apply_out(&self, y: &[f64], signs: &[i8]) -> Vec<f64> {
        assert_eq!(y.len(), self.n);
        assert_eq!(signs.len(), self.n);
        let (na, nb) = (self.na, self.nb);
        let mut t = vec![0.0; self.n];
        for p in 0..na {
            for j in 0..nb {
                t[p * nb + j] = (0..nb).map(|q| y[p * nb + q] * self.b[q * nb + j]).sum();
            }
        }
        let inv = 1.0 / (self.n as f64).sqrt();
        let mut z = vec![0.0; self.n];
        for i in 0..na {
            for j in 0..nb {
                let v: f64 = (0..na).map(|p| self.a[p * na + i] * t[p * nb + j]).sum();
                z[i * nb + j] = v * inv * f64::from(signs[i * nb + j]);
            }
        }
        z
    }
}

/// One coded `m x n` linear, borrowed from its arrays.
#[derive(Clone, Copy, Debug)]
pub struct LamqLinear<'a> {
    pub code: LamqCode,
    pub m: usize,
    pub n: usize,
    /// `m * row_bytes(n)` bytes: little-endian u32 words for trellis.
    pub codes: &'a [u8],
    /// Per-row f16 scale bits.
    pub scales: &'a [u16],
    pub signs_in: &'a [i8],
    pub signs_out: Option<&'a [i8]>,
    /// Shared LUT (`table_rows * g`) or the matrix's own levels.
    pub table: &'a [f32],
}

impl LamqLinear<'_> {
    /// Row `r` decoded and scaled.
    pub fn row(&self, r: usize) -> Vec<f32> {
        let rb = self
            .code
            .row_bytes(self.n)
            .expect("row width fits the code");
        let bytes = &self.codes[r * rb..(r + 1) * rb];
        let scale = crate::quant_1bit::f16_to_f32(self.scales[r]);
        let words: Vec<u32>;
        let row = if self.code.is_trellis() {
            words = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            RowCodes::Words(&words)
        } else {
            RowCodes::Bytes(bytes)
        };
        decode_row(self.code, row, self.n, self.table)
            .into_iter()
            .map(|v| v * scale)
            .collect()
    }

    /// `y = W x` for the original (unrotated) `x`, f64 accumulation.
    pub fn forward(&self, x: &[f32]) -> Vec<f64> {
        let rht_in = Rht::new(self.n).expect("input width has a Hadamard");
        let xr = rht_in.apply_in(x, self.signs_in);
        let y: Vec<f64> = (0..self.m)
            .map(|r| {
                self.row(r)
                    .iter()
                    .zip(&xr)
                    .map(|(&w, &v)| f64::from(w) * v)
                    .sum()
            })
            .collect();
        match self.signs_out {
            Some(s) => Rht::new(self.m)
                .expect("output width has a Hadamard")
                .apply_out(&y, s),
            None => y,
        }
    }

    /// Checks array lengths against the shape. Intake calls this before any
    /// decode so the panicking helpers above never see a bad slice.
    pub fn validate(&self) -> Result<(), String> {
        let name = self.code.name();
        let rb = self
            .code
            .row_bytes(self.n)
            .ok_or_else(|| format!("{name}: width {} does not fit the code", self.n))?;
        if self.codes.len() != self.m * rb {
            return Err(format!(
                "{name}: codes {} bytes, want {}",
                self.codes.len(),
                self.m * rb
            ));
        }
        if self.scales.len() != self.m {
            return Err(format!(
                "{name}: {} scales for {} rows",
                self.scales.len(),
                self.m
            ));
        }
        if self.signs_in.len() != self.n || Rht::new(self.n).is_none() {
            return Err(format!(
                "{name}: signs_in {} or width {} has no Hadamard",
                self.signs_in.len(),
                self.n
            ));
        }
        if let Some(s) = self.signs_out {
            if s.len() != self.m || Rht::new(self.m).is_none() {
                return Err(format!(
                    "{name}: signs_out {} or height {} has no Hadamard",
                    s.len(),
                    self.m
                ));
            }
        }
        if self.table.len() != self.code.table_rows() * self.code.g() {
            return Err(format!(
                "{name}: table {} values, want {}",
                self.table.len(),
                self.code.table_rows() * self.code.g()
            ));
        }
        let bad_sign = |s: &[i8]| s.iter().any(|&v| v != 1 && v != -1);
        if bad_sign(self.signs_in) || self.signs_out.is_some_and(bad_sign) {
            return Err(format!("{name}: a sign is not +1/-1"));
        }
        Ok(())
    }
}

/// An embedding row from its bucket: decoded and scaled `q`, then the output
/// rotation with the bucket's signs (`s * H^T q / sqrt(n)`). The RMS norm that
/// follows is the flow's, not this function's.
pub fn embed_row(bucket: &LamqLinear<'_>, r: usize) -> Vec<f64> {
    let q: Vec<f64> = bucket.row(r).into_iter().map(f64::from).collect();
    Rht::new(bucket.n)
        .expect("embed width has a Hadamard")
        .apply_out(&q, bucket.signs_in)
}

#[cfg(test)]
#[path = "lamq_tests.rs"]
mod tests;
