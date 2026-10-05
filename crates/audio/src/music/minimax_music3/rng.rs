//! MLX 0.32.3-compatible random number generation.
//!
//! Reference: `mlx/random.cpp`, `mlx/backend/cpu/{primitives.cpp,
//! threefry.cpp}`, and `mlx/backend/metal/kernels/random.metal` at
//! mlx v0.32.3 (all three agree on the counter layout, so draws are
//! device-independent).
//!
//! The generator is the JAX-style counter-based threefry2x32 hash: a
//! 64-bit key `(hi, lo)` and a 64-bit counter hash to two words per
//! invocation. `bits(words)` fills the output from both ends, so word
//! `i` and word `half + odd + i` come from the same hash call. That
//! two-ended layout is load bearing: reordering it changes every draw.
//!
//! Uniform and categorical are bit-matched to MLX (integer hashing plus
//! the same f32 division, minimum, and comparison order). `normal`
//! shares the exact uniform stream but evaluates `erfinv` through the
//! Wichura AS241 inverse normal CDF in f64, while MLX uses the
//! platform's `erfinv`; draws agree to f32 rounding but are not
//! guaranteed bit-identical to a Metal-side `mx.random.normal`.

/// An MLX PRNG key: `mx.random.key(seed)` is `(seed >> 32, seed)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key(pub u32, pub u32);

impl Key {
    /// `mx.random.key(seed)`.
    pub fn new(seed: u64) -> Key {
        Key((seed >> 32) as u32, seed as u32)
    }
}

/// The threefry2x32 hash with 20 rounds, exactly as MLX implements it.
pub(crate) fn threefry2x32(key: Key, count: (u32, u32)) -> (u32, u32) {
    const ROTATIONS: [[u32; 4]; 2] = [[13, 15, 26, 6], [17, 29, 16, 24]];
    let ks = [key.0, key.1, key.0 ^ key.1 ^ 0x1BD1_1BDA];
    let (mut x, mut y) = (count.0.wrapping_add(ks[0]), count.1.wrapping_add(ks[1]));
    for round in 0..5usize {
        for &r in &ROTATIONS[round % 2] {
            x = x.wrapping_add(y);
            y = y.rotate_left(r);
            y ^= x;
        }
        x = x.wrapping_add(ks[(round + 1) % 3]);
        y = y.wrapping_add(ks[(round + 2) % 3].wrapping_add(round as u32 + 1));
    }
    (x, y)
}

/// `mx.random.bits` for one key: `words` uint32 values with MLX's
/// two-ended counter layout.
pub fn random_bits(key: Key, words: usize) -> Vec<u32> {
    let mut out = vec![0u32; words];
    let odd = words & 1;
    let half = words >> 1;
    for i in 0..half {
        let (a, b) = threefry2x32(key, (i as u32, (half + odd + i) as u32));
        out[i] = a;
        out[half + odd + i] = b;
    }
    if odd == 1 {
        out[half] = threefry2x32(key, (half as u32, 0)).0;
    }
    out
}

/// `mx.random.split(key)`: a `[2, 2]` bits draw split into two keys.
pub fn split(key: Key) -> (Key, Key) {
    let words = random_bits(key, 4);
    (Key(words[0], words[1]), Key(words[2], words[3]))
}

/// Largest f32 strictly below 1.0 (`nextafter(1.0, 0.0)`).
const BELOW_ONE: f32 = 1.0f32 - f32::EPSILON / 2.0;
/// Largest-magnitude negative f32 strictly above -1.0.
const ABOVE_MINUS_ONE: f32 = -BELOW_ONE;

/// `mx.random.uniform(0, 1)` for `n` draws: `bits / float(0xFFFFFFFF)`
/// clamped below 1.0, in f32 exactly as MLX computes it.
pub fn uniform01(key: Key, n: usize) -> Vec<f32> {
    random_bits(key, n)
        .into_iter()
        .map(|bits| (bits as f32 / 4294967295.0f32).min(BELOW_ONE))
        .collect()
}

/// Inverse standard normal CDF, Acklam's rational approximation
/// (relative error < 1.15e-9, below f32 rounding after the cast).
fn ppnd16(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.024_25;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        poly6(&C, q) / (poly4(&D, q) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        poly6(&A, r) * q / (poly5(&B, r) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(poly6(&C, q) / (poly4(&D, q) * q + 1.0))
    }
}

fn poly6(coeffs: &[f64; 6], r: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * r + c)
}

fn poly5(coeffs: &[f64; 5], r: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * r + c)
}

fn poly4(coeffs: &[f64; 4], r: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * r + c)
}

/// `mx.random.normal` for `n` draws from `key`.
///
/// Mirrors MLX's transform (`uniform(low, high)` then `erfinv` scaled
/// by `sqrt(2)`), which reduces to the inverse normal CDF of the
/// `(u + 1) / 2` remap. The erfinv source differs from Metal's by f32
/// rounding only; see the module docs.
pub fn normal(key: Key, n: usize) -> Vec<f32> {
    let range = 1.0f32 - ABOVE_MINUS_ONE;
    uniform01(key, n)
        .into_iter()
        .map(|u| {
            let x = range * u + ABOVE_MINUS_ONE;
            ppnd16(0.5 * (1.0 + x as f64)) as f32
        })
        .collect()
}

/// The k-th largest value counting duplicates (MLX `min(topk(v, k))`).
pub fn kth_largest(values: &[f32], k: usize) -> f32 {
    debug_assert!(k >= 1 && k <= values.len());
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted[k - 1]
}

/// `mx.random.categorical` for a single row of logits.
///
/// MLX's single-row path is inverse-CDF sampling with one uniform draw:
/// exclusive cumsum of `exp(x - max)`, `u = uniform * total`, then the
/// count of CDF entries `<= u` minus one. The comparison runs in f64 so
/// a boundary landing within an ULP of `u` cannot flip on rounding.
pub fn categorical(logits: &[f32], key: Key) -> usize {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = logits
        .iter()
        .map(|&x| (x - max) as f64)
        .map(|v| v.exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let u = uniform01(key, 1)[0] as f64 * total;
    let mut cumulative = 0.0f64;
    let mut count = 0usize;
    for &w in &weights {
        if cumulative > u {
            break;
        }
        count += 1;
        cumulative += w;
    }
    count.saturating_sub(1).min(weights.len() - 1)
}

/// Top-k masked sampling: `mlx_audio/music/models/minimax_music3/
/// sampling.py::sample_top_k` for one row.
///
/// Returns the sampled index and the successor key (`split(key)[1]`),
/// matching the reference's key evolution.
pub fn sample_top_k(logits: &[f32], key: Key, top_k: usize) -> (usize, Key) {
    let values: Vec<f32> = logits
        .iter()
        .map(|&v| if v.is_nan() { -1e9 } else { v })
        .collect();
    let k = top_k.min(values.len());
    let threshold = kth_largest(&values, k);
    let masked: Vec<f32> = values
        .iter()
        .map(|&v| if v < threshold { -1e9 } else { v })
        .collect();
    let next = split(key).1;
    (categorical(&masked, key), next)
}

/// MLX's global `KeySequence`: `seed` sets the key, every draw splits
/// and keeps the first half.
#[derive(Debug, Clone, Copy)]
pub struct KeySequence {
    key: Key,
}

impl KeySequence {
    pub fn new(seed: u64) -> KeySequence {
        KeySequence {
            key: Key::new(seed),
        }
    }

    /// The next unkeyed draw's key: `split` with the first half kept.
    pub fn next(&mut self) -> Key {
        let (first, second) = split(self.key);
        self.key = first;
        second
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_construction_matches_the_seed_split() {
        assert_eq!(Key::new(42), Key(0, 42));
        assert_eq!(Key::new(1u64 << 32 | 7), Key(1, 7));
    }

    #[test]
    fn two_ended_layout_places_word_pairs_from_one_hash() {
        // 5 words: {0,3} share a hash, {1,4} share one, word 2 is the
        // odd tail hashed with counter zero.
        let words = random_bits(Key::new(11), 5);
        let a = threefry2x32(Key::new(11), (0, 3));
        let b = threefry2x32(Key::new(11), (1, 4));
        let c = threefry2x32(Key::new(11), (2, 0));
        assert_eq!(words, vec![a.0, b.0, c.0, a.1, b.1]);
    }

    #[test]
    fn uniform_stays_in_unit_interval_and_is_deterministic() {
        let first = uniform01(Key::new(7), 16);
        let second = uniform01(Key::new(7), 16);
        assert_eq!(first, second);
        assert!(first.iter().all(|&u| (0.0..1.0).contains(&u)));
    }

    #[test]
    fn categorical_picks_overwhelming_logits() {
        let mut logits = vec![0.0f32; 32];
        logits[17] = 100.0;
        assert_eq!(categorical(&logits, Key::new(3)), 17);
    }

    #[test]
    fn top_k_masking_removes_low_mass_entries() {
        let mut logits = vec![0.0f32; 64];
        for (i, slot) in logits.iter_mut().enumerate() {
            *slot = if i % 2 == 0 { 10.0 } else { -10.0 };
        }
        let (drawn, _) = sample_top_k(&logits, Key::new(5), 8);
        assert_eq!(drawn % 2, 0);
    }

    #[test]
    fn normal_draws_are_standard_normal_distributed() {
        let draws = normal(Key::new(9), 4096);
        let mean = draws.iter().sum::<f32>() / draws.len() as f32;
        let var = draws.iter().map(|v| v * v).sum::<f32>() / draws.len() as f32;
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!((var - 1.0).abs() < 0.2, "var {var}");
    }

    #[test]
    fn inverse_normal_matches_reference_quantiles() {
        let cases = [
            (0.5, 0.0),
            (0.975, 1.959_963_984_540_054),
            (0.001, -3.090_232_306_167_813),
            (1e-6, -4.753_424_308_822_899),
            (0.841_344_746, 0.999_999_99),
            (0.02, -2.053_748_910_631_822_5),
            (0.99, 2.326_347_874_040_840_8),
        ];
        for (p, want) in cases {
            let got = ppnd16(p);
            assert!(
                (got - want).abs() < 1e-6 * want.abs().max(1.0),
                "ppnd16({p}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn kth_largest_counts_duplicates() {
        assert_eq!(kth_largest(&[5.0, 1.0, 5.0, 5.0, 2.0], 3), 5.0);
        assert_eq!(kth_largest(&[5.0, 1.0, 5.0, 5.0, 2.0], 4), 2.0);
    }
}
