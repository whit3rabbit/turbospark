//! Radix-2 FFT with a real-input wrapper, plus Bluestein's chirp-z
//! algorithm for the sizes radix-2 cannot reach.
//!
//! Power-of-two sizes run the iterative radix-2 Cooley-Tukey kernel
//! directly. Every other even size runs Bluestein's algorithm, which turns
//! one n-point DFT into a power-of-two circular convolution, so whisper's
//! 400-point mel frames use the reference's own transform size instead of
//! a zero-padded approximation. Twiddle factors are computed in f64 and
//! stored in f32, and butterfly arithmetic runs in f32. The real transform
//! uses the standard packing trick: an n-point real FFT runs as an
//! n/2-point complex FFT plus a split step.
//!
//! Scaling convention: [`RealFftPlan::forward`] is unnormalized (bin 0 holds
//! the plain sum of the input), and [`RealFftPlan::inverse`] divides by
//! n/2, which makes forward-then-inverse the identity. The parity tests
//! against direct DFT evaluation pin both halves to that convention, for
//! radix-2 and Bluestein sizes alike.

use crate::dsp::TableCache;
use crate::error::AudioError;
use std::sync::Arc;

/// A complex sample stored as two f32s.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ComplexF32 {
    pub re: f32,
    pub im: f32,
}

impl ComplexF32 {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }

    fn scale(self, k: f32) -> Self {
        Self::new(self.re * k, self.im * k)
    }

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

/// The internal complex transform the real-FFT packing runs at half size.
///
/// Power-of-two sizes use the radix-2 kernel directly. Every other even
/// size uses Bluestein's chirp-z algorithm, which turns one n-point DFT
/// into a circular convolution of length L, the next power of two at or
/// above 2n-1, so arbitrary sizes (whisper's 400-point mel frames) ride
/// the same tested radix-2 kernel without a second algorithm family.
#[derive(Debug, Clone)]
enum ComplexFft {
    Radix2 {
        bitrev: Vec<usize>,
        /// Per-stage contiguous twiddle slices: stage tables[s][j] ==
        /// twiddle[j * stride] (plus the conjugated set for the inverse).
        /// The gather `twiddle[j * stride]` inside the butterfly loop is
        /// the kernel's worst cache behavior; flattening it per stage
        /// keeps the arithmetic bit-identical while making the hot loop
        /// a contiguous walk.
        stage_twiddles: Vec<Vec<ComplexF32>>,
        stage_twiddles_inv: Vec<Vec<ComplexF32>>,
    },
    Bluestein {
        n: usize,
        /// Chirp `exp(-i pi k^2 / n)` for k in [0, n), stored f32 from f64.
        chirp: Vec<ComplexF32>,
        /// Convolution length L, the next power of two >= 2n - 1.
        l: usize,
        /// Radix-2 tables for the length-L transforms.
        l_bitrev: Vec<usize>,
        /// The forward spectrum of B, the wrapped conjugate chirp, constant
        /// for the plan: B[m] = conj(chirp[m]) folded to L - m. Computed
        /// once at construction; every call would otherwise rebuild B and
        /// transform it again, one extra length-L FFT per invocation.
        b_spectrum: Vec<ComplexF32>,
        /// Per-stage twiddle tables for the length-L transforms, forward
        /// and conjugated, mirroring the radix-2 variant.
        l_stages_fwd: Vec<Vec<ComplexF32>>,
        l_stages_inv: Vec<Vec<ComplexF32>>,
    },
}

impl ComplexFft {
    fn new(n: usize) -> Result<Self, AudioError> {
        if n == 0 {
            return Err(AudioError::InvalidParameter {
                name: "fft size".to_string(),
                value: "0".to_string(),
                why: "transform size must be positive".to_string(),
            });
        }
        if n.is_power_of_two() {
            let twiddle = twiddle_table(n);
            let (stage_twiddles, stage_twiddles_inv) = stage_tables(n, &twiddle);
            return Ok(Self::Radix2 {
                bitrev: bitrev_table(n),
                stage_twiddles,
                stage_twiddles_inv,
            });
        }
        // Bluestein: chirp, then radix-2 tables for the convolution length.
        let l = (2 * n - 1).next_power_of_two();
        let chirp: Vec<ComplexF32> = (0..n)
            .map(|k| {
                // exp(-i pi k^2 / n): k^2 mod 2n keeps the f64 angle exact
                // for every k, because the chirp has period 2n in k^2.
                let kk = ((k as f64) * (k as f64)) % (2.0 * n as f64);
                let angle = -std::f64::consts::PI * kk / n as f64;
                ComplexF32::new(angle.cos() as f32, angle.sin() as f32)
            })
            .collect();
        // B[m] = conj(chirp[m]) = exp(+i pi m^2 / n) for m in [0, n),
        // wrapped to L - m: the chirp is even in its argument, so the
        // negative differences fold back. Its spectrum is plan-constant.
        let bitrev = bitrev_table(l);
        let twiddle = twiddle_table(l);
        let (l_stages_fwd, l_stages_inv) = stage_tables(l, &twiddle);
        let mut b = vec![ComplexF32::default(); l];
        for m in 0..n {
            b[m] = chirp[m].conj();
            if m > 0 {
                b[l - m] = chirp[m].conj();
            }
        }
        complex_fft_staged(&mut b, &bitrev, Staged::Forward(&l_stages_fwd));
        Ok(Self::Bluestein {
            n,
            chirp,
            l,
            l_bitrev: bitrev,
            b_spectrum: b,
            l_stages_fwd,
            l_stages_inv,
        })
    }

    /// Unnormalized transform in place (forward, or inverse with
    /// conjugated twiddles) -- the same contract as the radix-2 kernel.
    fn run(&self, buf: &mut [ComplexF32], inverse: bool) {
        self.run_with(buf, inverse, &mut Vec::new());
    }

    /// [`Self::run`] with a caller-owned convolution buffer, which only the
    /// Bluestein path touches. Reusing it across frames drops one length-L
    /// allocation per transform; the arithmetic is unchanged.
    fn run_with(&self, buf: &mut [ComplexF32], inverse: bool, conv: &mut Vec<ComplexF32>) {
        match self {
            Self::Radix2 {
                bitrev,
                stage_twiddles,
                stage_twiddles_inv,
            } => {
                let staged = if inverse {
                    Staged::Inverse(stage_twiddles_inv)
                } else {
                    Staged::Forward(stage_twiddles)
                };
                complex_fft_staged(buf, bitrev, staged);
            }
            Self::Bluestein {
                n,
                chirp,
                l,
                l_bitrev,
                b_spectrum,
                l_stages_fwd,
                l_stages_inv,
            } => {
                if inverse {
                    // IDFT(x) = conj(DFT(conj(x))): reuse the forward only.
                    for v in buf.iter_mut() {
                        *v = v.conj();
                    }
                }
                // A[j] = x[j] * chirp[j], zero-padded to L.
                let a = conv;
                a.clear();
                a.resize(*l, ComplexF32::default());
                for (j, &x) in buf.iter().enumerate() {
                    a[j] = x.mul(chirp[j]);
                }
                complex_fft_staged(a, l_bitrev, Staged::Forward(l_stages_fwd));
                for (fa, fb) in a.iter_mut().zip(b_spectrum) {
                    *fa = fa.mul(*fb);
                }
                complex_fft_staged(a, l_bitrev, Staged::Inverse(l_stages_inv));
                let scale = 1.0f32 / *l as f32;
                for k in 0..*n {
                    let scaled = a[k].scale(scale);
                    buf[k] = scaled.mul(chirp[k]);
                }
                if inverse {
                    for v in buf.iter_mut() {
                        *v = v.conj();
                    }
                }
            }
        }
    }
}

/// Bit-reversal permutation for a power-of-two length.
fn bitrev_table(n: usize) -> Vec<usize> {
    (0..n)
        .map(|i| {
            let mut rev = 0usize;
            let mut x = i;
            let bits = n.trailing_zeros().max(1);
            for _ in 0..bits {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
            }
            rev
        })
        .collect()
}

/// Twiddle table e^(-2 pi i j / n) for j in [0, n), stored f32 from f64.
fn twiddle_table(n: usize) -> Vec<ComplexF32> {
    (0..n)
        .map(|j| {
            let angle = -2.0 * std::f64::consts::PI * j as f64 / n as f64;
            ComplexF32::new(angle.cos() as f32, angle.sin() as f32)
        })
        .collect()
}

/// A reusable plan for real FFTs of one size.
///
/// Sizes are even (the real-packing split needs a partner bin) or exactly
/// one; power-of-two sizes run the radix-2 kernel and every other even
/// size runs Bluestein's chirp-z over it. Building the plan costs the
/// permutation and twiddle tables once; a frontend should build it once
/// and transform many frames, which is exactly how the STFT and mel
/// modules consume it.
#[derive(Debug, Clone)]
pub struct RealFftPlan {
    /// Transform length in real samples.
    pub size: usize,
    half: usize,
    /// The half-size complex transform the packing runs; `None` only for
    /// size 1, which never reaches the packing.
    half_fft: Option<ComplexFft>,
    /// Split twiddles e^(-2 pi i k / size) for k in [0, half].
    split: Vec<ComplexF32>,
}

impl RealFftPlan {
    /// Builds a plan for an even size (or size 1).
    pub fn new(size: usize) -> Result<Self, AudioError> {
        // Bound plan tables before radix-2/Bluestein size arithmetic and
        // allocation. Audio frontends use hundreds or thousands of taps.
        if size > 1 << 20 {
            return Err(AudioError::BufferTooLarge {
                what: "real FFT plan",
                samples: size,
            });
        }
        if size == 0 {
            return Err(AudioError::InvalidParameter {
                name: "fft size".to_string(),
                value: "0".to_string(),
                why: "transform size must be positive".to_string(),
            });
        }
        if size != 1 && size % 2 != 0 {
            return Err(AudioError::InvalidParameter {
                name: "fft size".to_string(),
                value: size.to_string(),
                why: "real FFT sizes must be even (or exactly 1)".to_string(),
            });
        }
        let half = size / 2;
        let half_fft = if size == 1 {
            None
        } else {
            Some(ComplexFft::new(half)?)
        };
        let split = (0..=half)
            .map(|k| {
                let angle = -2.0 * std::f64::consts::PI * k as f64 / size as f64;
                ComplexF32::new(angle.cos() as f32, angle.sin() as f32)
            })
            .collect();
        Ok(Self {
            size,
            half,
            half_fft,
            split,
        })
    }

    /// Returns a shared plan for `size`, building it on first use.
    ///
    /// A Bluestein plan (whisper's 400-point frames) costs a length-1024
    /// transform plus its tables, so frontends that run once per utterance
    /// would otherwise rebuild it every call. Plans are immutable, so a
    /// cached plan transforms bit-identically to a fresh one. Very large
    /// plans are built per call rather than pinned in the cache.
    pub fn cached(size: usize) -> Result<Arc<Self>, AudioError> {
        static PLANS: TableCache<usize, RealFftPlan> = TableCache::new(16);
        PLANS.get_or_build(size, size <= MAX_CACHED_PLAN_SIZE, || Self::new(size))
    }

    /// Forward transform of `n` real samples into `n/2 + 1` complex bins.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<ComplexF32>, AudioError> {
        let mut out = Vec::new();
        self.forward_into(input, &mut out, &mut RealFftScratch::default())?;
        Ok(out)
    }

    /// [`Self::forward`] into a caller-owned bin buffer (cleared and resized
    /// to `n/2 + 1`) with caller-owned scratch, so a frame loop allocates
    /// nothing after the first frame. Same arithmetic as `forward`.
    pub fn forward_into(
        &self,
        input: &[f32],
        out: &mut Vec<ComplexF32>,
        scratch: &mut RealFftScratch,
    ) -> Result<(), AudioError> {
        if input.len() != self.size {
            return Err(AudioError::ShapeMismatch {
                what: "real FFT input length",
                expected: self.size,
                actual: input.len(),
            });
        }
        if self.size == 1 {
            out.clear();
            out.push(ComplexF32::new(input[0], 0.0));
            return Ok(());
        }
        // Pack x[2k] + i x[2k+1] and transform at half size.
        let z = &mut scratch.z;
        z.clear();
        z.extend((0..self.half).map(|k| ComplexF32::new(input[2 * k], input[2 * k + 1])));
        self.half_fft
            .as_ref()
            .unwrap()
            .run_with(z, false, &mut scratch.conv);

        let m = self.half;
        out.clear();
        out.resize(m + 1, ComplexF32::default());
        // k = 0 packs DC and Nyquist into one real pair.
        let z0 = z[0];
        out[0] = ComplexF32::new(z0.re + z0.im, 0.0);
        out[m] = ComplexF32::new(z0.re - z0.im, 0.0);
        for k in 1..m {
            let zk = z[k];
            let zmk = z[m - k].conj();
            let c = zk.add(zmk).scale(0.5);
            let d = zk.sub(zmk).scale(0.5).mul(self.split[k]);
            // X[k] = c - i d
            out[k] = ComplexF32::new(c.re + d.im, c.im - d.re);
        }
        Ok(())
    }

    /// Inverse transform of `n/2 + 1` conjugate-symmetric bins back to `n`
    /// real samples, normalized so forward-then-inverse is the identity.
    pub fn inverse(&self, spectrum: &[ComplexF32]) -> Result<Vec<f32>, AudioError> {
        let mut out = Vec::new();
        self.inverse_into(spectrum, &mut out, &mut RealFftScratch::default())?;
        Ok(out)
    }

    /// [`Self::inverse`] into a caller-owned sample buffer with caller-owned
    /// scratch. Same arithmetic as `inverse`.
    pub fn inverse_into(
        &self,
        spectrum: &[ComplexF32],
        out: &mut Vec<f32>,
        scratch: &mut RealFftScratch,
    ) -> Result<(), AudioError> {
        if spectrum.len() != self.half + 1 {
            return Err(AudioError::ShapeMismatch {
                what: "real FFT spectrum length",
                expected: self.half + 1,
                actual: spectrum.len(),
            });
        }
        if self.size == 1 {
            out.clear();
            out.push(spectrum[0].re);
            return Ok(());
        }
        let m = self.half;
        let z = &mut scratch.z;
        z.clear();
        for k in 0..m {
            let other = if k == 0 { spectrum[m] } else { spectrum[m - k] };
            // Inverting the forward split: Z[k] = 0.5 * S + 0.5 * i *
            // conj(w_k) * D, with S = X[k] + conj(partner), D = X[k] -
            // conj(partner), w_k = split[k]. i * conj(t) as a complex pair
            // is (t.im, t.re).
            let t = self.split[k];
            let s = spectrum[k].add(other.conj()).scale(0.5);
            let d = spectrum[k].sub(other.conj()).scale(0.5);
            z.push(s.add(d.mul(ComplexF32::new(t.im, t.re))));
        }
        self.half_fft
            .as_ref()
            .unwrap()
            .run_with(z, true, &mut scratch.conv);
        let scale = 1.0f32 / m as f32;
        out.clear();
        for zk in z.iter() {
            out.push(zk.re * scale);
            out.push(zk.im * scale);
        }
        Ok(())
    }
}

/// Reusable work buffers for [`RealFftPlan::forward_into`] and
/// [`RealFftPlan::inverse_into`]. Contents are meaningless between calls;
/// any scratch works with any plan.
#[derive(Debug, Default, Clone)]
pub struct RealFftScratch {
    z: Vec<ComplexF32>,
    conv: Vec<ComplexF32>,
}

/// Largest real-FFT size kept in the process-wide plan cache (the Bluestein
/// plan for 8192 holds a few hundred KiB); larger plans are rebuilt per call.
const MAX_CACHED_PLAN_SIZE: usize = 1 << 13;

/// Per-stage contiguous twiddle tables for a power-of-two length: stage
/// with half = len/2 needs `twiddle[j * (n / len)]` for j in 0..half, plus
/// the conjugated set for the inverse. Total n - 1 entries per direction.
fn stage_tables(n: usize, twiddle: &[ComplexF32]) -> (Vec<Vec<ComplexF32>>, Vec<Vec<ComplexF32>>) {
    let mut forward = Vec::new();
    let mut inverse = Vec::new();
    let mut len = 2usize;
    while len <= n {
        let stride = n / len;
        let half = len / 2;
        let stage: Vec<ComplexF32> = (0..half).map(|j| twiddle[j * stride]).collect();
        inverse.push(stage.iter().map(|w| w.conj()).collect());
        forward.push(stage);
        len *= 2;
    }
    (forward, inverse)
}

/// Which conjugation of the per-stage twiddle tables the kernel walks.
///
/// Callers pick once per transform so the butterfly loop never branches.
enum Staged<'a> {
    Forward(&'a [Vec<ComplexF32>]),
    Inverse(&'a [Vec<ComplexF32>]),
}

/// The radix-2 kernel over the plan's per-stage tables. Bit-identical to
/// the historical gather form: same table values (conjugation is an exact
/// sign flip), same butterfly pairing, same add/sub order.
fn complex_fft_staged(buf: &mut [ComplexF32], bitrev: &[usize], staged: Staged<'_>) {
    let n = buf.len();
    if n < 2 {
        return;
    }
    for (i, &r) in bitrev.iter().enumerate() {
        if r > i {
            buf.swap(i, r);
        }
    }
    let stages: &[Vec<ComplexF32>] = match staged {
        Staged::Forward(tables) => tables,
        Staged::Inverse(tables) => tables,
    };
    let mut len = 2usize;
    let half_len = len / 2;
    let _ = half_len;
    for table in stages {
        let half = len / 2;
        for chunk in buf.chunks_mut(len) {
            for (j, &w) in table.iter().enumerate() {
                let even = chunk[j];
                let odd = chunk[j + half].mul(w);
                chunk[j] = even.add(odd);
                chunk[j + half] = even.sub(odd);
            }
        }
        len *= 2;
    }
}

/// Plan-less forward real FFT. Builds a plan for the one call; reuse
/// [`RealFftPlan`] when transforming many frames of one size.
pub fn real_fft_forward(input: &[f32]) -> Result<Vec<ComplexF32>, AudioError> {
    RealFftPlan::new(input.len())?.forward(input)
}

/// Public complex FFT plan with the numpy conventions: unscaled
/// forward, inverse scaled by 1/n. Unlike [`RealFftPlan::inverse`],
/// which is only defined for conjugate-symmetric spectra, this runs a
/// true complex transform, so it is the right tool when a model emits
/// arbitrary complex spectra that must be inverted real (vocoders).
pub struct ComplexFftPlan {
    plan: ComplexFft,
    n: usize,
}

impl ComplexFftPlan {
    /// Builds a plan for `n` points (any positive size; non powers of
    /// two take the Bluestein path).
    pub fn new(n: usize) -> Result<Self, AudioError> {
        Ok(ComplexFftPlan {
            plan: ComplexFft::new(n)?,
            n,
        })
    }

    /// Unscaled forward transform.
    pub fn forward(&self, input: &[ComplexF32]) -> Result<Vec<ComplexF32>, AudioError> {
        let mut buf = input.to_vec();
        self.plan.run(&mut buf, false);
        Ok(buf)
    }

    /// Inverse transform with the numpy `ifft` 1/n scaling.
    pub fn inverse(&self, input: &[ComplexF32]) -> Result<Vec<ComplexF32>, AudioError> {
        let mut buf = input.to_vec();
        self.plan.run(&mut buf, true);
        let scale = 1.0 / self.n as f32;
        for v in &mut buf {
            *v = v.scale(scale);
        }
        Ok(buf)
    }
}

/// Plan-less inverse real FFT for a power-of-two output size.
pub fn real_fft_inverse(spectrum: &[ComplexF32]) -> Result<Vec<f32>, AudioError> {
    let size = (spectrum.len() - 1) * 2;
    RealFftPlan::new(size)?.inverse(spectrum)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct DFT in f64, the parity reference.
    fn dft(input: &[f32]) -> Vec<ComplexF32> {
        let n = input.len();
        (0..n)
            .map(|k| {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (j, &x) in input.iter().enumerate() {
                    let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
                    re += f64::from(x) * angle.cos();
                    im += f64::from(x) * angle.sin();
                }
                ComplexF32::new(re as f32, im as f32)
            })
            .collect()
    }

    fn assert_close(a: &[ComplexF32], b: &[ComplexF32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let err = ((x.re - y.re).powi(2) + (x.im - y.im).powi(2)).sqrt();
            assert!(err <= tol, "{what}[{i}] differs by {err}: {x:?} vs {y:?}");
        }
    }

    #[test]
    fn forward_matches_direct_dft() {
        for size in [2usize, 8, 64, 512] {
            let input: Vec<f32> = (0..size)
                .map(|i| ((i as f32) * 0.7).sin() + 0.3 * ((i as f32) * 2.1).cos())
                .collect();
            let plan = RealFftPlan::new(size).unwrap();
            let got = plan.forward(&input).unwrap();
            let want = dft(&input);
            // Bins above size/2 are the conjugate tail the packed transform
            // does not emit; compare the emitted half only.
            assert_close(&got, &want[..size / 2 + 1], 2e-4, "fft bin");
        }
    }

    #[test]
    fn forward_then_inverse_is_identity() {
        for size in [2usize, 8, 64, 512] {
            let input: Vec<f32> = (0..size).map(|i| ((i as f32) * 1.3).sin()).collect();
            let plan = RealFftPlan::new(size).unwrap();
            let spectrum = plan.forward(&input).unwrap();
            let back = plan.inverse(&spectrum).unwrap();
            let err = input
                .iter()
                .zip(&back)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(err < 1e-4, "size {size} round trip error {err}");
        }
    }

    #[test]
    fn sine_peaks_in_its_bin() {
        let size = 512usize;
        let bin = 8;
        let input: Vec<f32> = (0..size)
            .map(|i| {
                (2.0 * std::f64::consts::PI * bin as f64 * i as f64 / size as f64).sin() as f32
            })
            .collect();
        let plan = RealFftPlan::new(size).unwrap();
        let spectrum = plan.forward(&input).unwrap();
        let best = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| {
                let am = a.1.re * a.1.re + a.1.im * a.1.im;
                let bm = b.1.re * b.1.re + b.1.im * b.1.im;
                am.partial_cmp(&bm).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(best, bin);
        // Peak magnitude is size/2 for a unit sine.
        let mag = spectrum[bin].re.hypot(spectrum[bin].im);
        assert!(
            (mag - size as f32 / 2.0).abs() < 0.5,
            "peak magnitude {mag}"
        );
    }

    #[test]
    fn size_one_and_rejects() {
        let plan = RealFftPlan::new(1).unwrap();
        assert_eq!(
            plan.forward(&[2.5]).unwrap(),
            vec![ComplexF32::new(2.5, 0.0)]
        );
        assert_eq!(
            plan.inverse(&[ComplexF32::new(2.5, 0.0)]).unwrap(),
            vec![2.5]
        );
        // Odd sizes have no partner bin for the real split; zero is not a
        // transform.
        let err = RealFftPlan::new(7).unwrap_err();
        assert!(err.to_string().contains("must be even"), "{err}");
        let err = RealFftPlan::new(0).unwrap_err();
        assert!(err.to_string().contains("positive"), "{err}");
    }

    #[test]
    fn bluestein_size_matches_direct_dft_and_round_trips() {
        // 400 is whisper's mel frame size: 2n - 1 = 799, so the plan runs
        // Bluestein over a 1024-point radix-2 kernel.
        let size = 400usize;
        let input: Vec<f32> = (0..size)
            .map(|i| ((i as f32) * 0.37).sin() + 0.2 * ((i as f32) * 2.9).cos())
            .collect();
        let plan = RealFftPlan::new(size).unwrap();
        let got = plan.forward(&input).unwrap();
        let want = dft(&input);
        assert_close(&got, &want[..size / 2 + 1], 2e-4, "bluestein bin");
        let back = plan.inverse(&got).unwrap();
        let err = input
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-4, "bluestein round trip error {err}");
    }

    #[test]
    fn bluestein_covers_small_even_and_odd_half_sizes() {
        // The half-size transform Bluestein serves can itself be odd
        // (size 66 -> half 33) and non-power-of-two even (size 44 -> half
        // 22); both must still match the direct DFT.
        for size in [66usize, 44, 12] {
            let input: Vec<f32> = (0..size).map(|i| ((i as f32) * 1.7).sin()).collect();
            let plan = RealFftPlan::new(size).unwrap();
            let got = plan.forward(&input).unwrap();
            let want = dft(&input);
            assert_close(&got, &want[..size / 2 + 1], 2e-4, "bin");
        }
    }

    #[test]
    fn shape_mismatches_name_both_lengths() {
        let plan = RealFftPlan::new(8).unwrap();
        let err = plan.forward(&[0.0; 7]).unwrap_err();
        assert!(err.to_string().contains("expected 8, got 7"), "{err}");
        let err = plan.inverse(&[ComplexF32::default(); 3]).unwrap_err();
        assert!(err.to_string().contains("expected 5, got 3"), "{err}");
    }

    #[test]
    fn planless_helpers_agree_with_plan() {
        let input: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.9).cos()).collect();
        let direct = real_fft_forward(&input).unwrap();
        let planned = RealFftPlan::new(16).unwrap().forward(&input).unwrap();
        assert_eq!(direct, planned);
        let back = real_fft_inverse(&direct).unwrap();
        let err = input
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-5, "planless round trip error {err}");
    }
}

#[cfg(test)]
mod allocation_regression {
    use super::*;
    #[test]
    fn rejects_unbounded_plan_sizes_before_allocating_tables() {
        for size in [usize::MAX - 1, (1usize << 20) + 2] {
            assert!(matches!(
                RealFftPlan::new(size),
                Err(AudioError::BufferTooLarge { .. })
            ));
        }
    }
}

#[cfg(test)]
mod complex_plan_tests {
    use super::*;

    #[test]
    fn complex_plan_forward_matches_dft_256() {
        let n = 256;
        let plan = ComplexFftPlan::new(n).unwrap();
        let input: Vec<ComplexF32> = (0..n)
            .map(|i| ComplexF32::new((i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()))
            .collect();
        let fwd = plan.forward(&input).unwrap();
        let mut worst = 0.0f64;
        for k in 0..n {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (j, x) in input.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
                re += f64::from(x.re) * angle.cos() - f64::from(x.im) * angle.sin();
                im += f64::from(x.re) * angle.sin() + f64::from(x.im) * angle.cos();
            }
            worst = worst.max(
                ((fwd[k].re - re as f32).abs() as f64).max(((fwd[k].im - im as f32).abs()) as f64),
            );
        }
        assert!(worst < 1e-3, "n=256 forward vs DFT worst {worst}");
    }

    #[test]
    fn complex_plan_inverse_matches_idft_256() {
        let n = 256;
        let plan = ComplexFftPlan::new(n).unwrap();
        // Arbitrary (non-Hermitian) complex spectrum.
        let input: Vec<ComplexF32> = (0..n)
            .map(|i| ComplexF32::new((i as f32 * 0.37).sin() * 0.8, (i as f32 * 0.23).cos() * 0.6))
            .collect();
        let out = plan.inverse(&input).unwrap();
        let mut worst = 0.0f64;
        for t in 0..n {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (k, x) in input.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
                re += f64::from(x.re) * angle.cos() - f64::from(x.im) * angle.sin();
                im += f64::from(x.re) * angle.sin() + f64::from(x.im) * angle.cos();
            }
            let scale = 1.0 / n as f64;
            worst = worst
                .max((out[t].re - (re * scale) as f32).abs() as f64)
                .max((out[t].im - (im * scale) as f32).abs() as f64);
        }
        assert!(worst < 1e-4, "n=256 inverse vs IDFT worst {worst}");
    }

    #[test]
    fn complex_plan_roundtrip_and_inverse_match_dft() {
        let n = 16;
        let plan = ComplexFftPlan::new(n).unwrap();
        let input: Vec<ComplexF32> = (0..n)
            .map(|i| ComplexF32::new((i as f32 * 0.7).sin(), (i as f32 * 1.3).cos()))
            .collect();
        // Forward must match a direct DFT.
        let fwd = plan.forward(&input).unwrap();
        for k in 0..n {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (j, x) in input.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
                re += f64::from(x.re) * angle.cos() - f64::from(x.im) * angle.sin();
                im += f64::from(x.re) * angle.sin() + f64::from(x.im) * angle.cos();
            }
            assert!(
                (fwd[k].re - re as f32).abs() < 1e-4,
                "fwd[{k}] re {} vs {re}",
                fwd[k].re
            );
            assert!(
                (fwd[k].im - im as f32).abs() < 1e-4,
                "fwd[{k}] im {} vs {im}",
                fwd[k].im
            );
        }
        // Roundtrip must be the identity.
        let back = plan.inverse(&fwd).unwrap();
        for i in 0..n {
            assert!((back[i].re - input[i].re).abs() < 1e-4, "roundtrip re[{i}]");
            assert!((back[i].im - input[i].im).abs() < 1e-4, "roundtrip im[{i}]");
        }
    }
}

/// Bitwise parity of the buffer-reusing `*_into` paths and the plan cache
/// against the previous allocate-per-call implementations, retained verbatim
/// in `old` below. Every downstream golden fixture was recorded against that
/// arithmetic, so equality must hold in `to_bits()`, not within a tolerance.
#[cfg(test)]
mod parity_tests {
    use super::*;
    use crate::dsp::TestRng;

    mod old {
        use super::super::*;

        /// The pre-scratch `ComplexFft::run`.
        pub(super) fn run(fft: &ComplexFft, buf: &mut [ComplexF32], inverse: bool) {
            match fft {
                ComplexFft::Radix2 {
                    bitrev,
                    stage_twiddles,
                    stage_twiddles_inv,
                } => {
                    let staged = if inverse {
                        Staged::Inverse(stage_twiddles_inv)
                    } else {
                        Staged::Forward(stage_twiddles)
                    };
                    complex_fft_staged(buf, bitrev, staged);
                }
                ComplexFft::Bluestein {
                    n,
                    chirp,
                    l,
                    l_bitrev,
                    b_spectrum,
                    l_stages_fwd,
                    l_stages_inv,
                } => {
                    if inverse {
                        for v in buf.iter_mut() {
                            *v = v.conj();
                        }
                    }
                    let mut a = vec![ComplexF32::default(); *l];
                    for (j, &x) in buf.iter().enumerate() {
                        a[j] = x.mul(chirp[j]);
                    }
                    complex_fft_staged(&mut a, l_bitrev, Staged::Forward(l_stages_fwd));
                    for (fa, fb) in a.iter_mut().zip(b_spectrum) {
                        *fa = fa.mul(*fb);
                    }
                    complex_fft_staged(&mut a, l_bitrev, Staged::Inverse(l_stages_inv));
                    let scale = 1.0f32 / *l as f32;
                    for k in 0..*n {
                        let conv = a[k].scale(scale);
                        buf[k] = conv.mul(chirp[k]);
                    }
                    if inverse {
                        for v in buf.iter_mut() {
                            *v = v.conj();
                        }
                    }
                }
            }
        }

        /// The pre-scratch `RealFftPlan::forward`.
        pub(super) fn forward(plan: &RealFftPlan, input: &[f32]) -> Vec<ComplexF32> {
            if plan.size == 1 {
                return vec![ComplexF32::new(input[0], 0.0)];
            }
            let mut z: Vec<ComplexF32> = (0..plan.half)
                .map(|k| ComplexF32::new(input[2 * k], input[2 * k + 1]))
                .collect();
            run(plan.half_fft.as_ref().unwrap(), &mut z, false);
            let m = plan.half;
            let mut out = vec![ComplexF32::default(); m + 1];
            let z0 = z[0];
            out[0] = ComplexF32::new(z0.re + z0.im, 0.0);
            out[m] = ComplexF32::new(z0.re - z0.im, 0.0);
            for k in 1..m {
                let zk = z[k];
                let zmk = z[m - k].conj();
                let c = zk.add(zmk).scale(0.5);
                let d = zk.sub(zmk).scale(0.5).mul(plan.split[k]);
                out[k] = ComplexF32::new(c.re + d.im, c.im - d.re);
            }
            out
        }

        /// The pre-scratch `RealFftPlan::inverse`.
        pub(super) fn inverse(plan: &RealFftPlan, spectrum: &[ComplexF32]) -> Vec<f32> {
            if plan.size == 1 {
                return vec![spectrum[0].re];
            }
            let m = plan.half;
            let mut z: Vec<ComplexF32> = Vec::with_capacity(m);
            for k in 0..m {
                let other = if k == 0 { spectrum[m] } else { spectrum[m - k] };
                let t = plan.split[k];
                let s = spectrum[k].add(other.conj()).scale(0.5);
                let d = spectrum[k].sub(other.conj()).scale(0.5);
                z.push(s.add(d.mul(ComplexF32::new(t.im, t.re))));
            }
            run(plan.half_fft.as_ref().unwrap(), &mut z, true);
            let scale = 1.0f32 / m as f32;
            let mut out = Vec::with_capacity(plan.size);
            for zk in &z {
                out.push(zk.re * scale);
                out.push(zk.im * scale);
            }
            out
        }
    }

    fn assert_complex_bits(what: &str, got: &[ComplexF32], want: &[ComplexF32]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.re.to_bits(), w.re.to_bits(), "{what}[{i}].re {g:?} {w:?}");
            assert_eq!(g.im.to_bits(), w.im.to_bits(), "{what}[{i}].im {g:?} {w:?}");
        }
    }

    /// Radix-2, Bluestein, and degenerate sizes, in an order that makes the
    /// shared scratch shrink and grow between plans.
    const SIZES: [usize; 12] = [1, 2, 6, 8, 12, 64, 100, 400, 1024, 400, 6, 640];

    #[test]
    fn forward_into_matches_old_forward_bitwise() {
        let mut rng = TestRng(0x9E37_79B9_7F4A_7C15);
        let mut out = Vec::new();
        let mut scratch = RealFftScratch::default();
        for &size in &SIZES {
            let plan = RealFftPlan::new(size).unwrap();
            for _ in 0..3 {
                let input = rng.vec(size);
                plan.forward_into(&input, &mut out, &mut scratch).unwrap();
                assert_complex_bits(
                    &format!("forward_into size {size}"),
                    &out,
                    &old::forward(&plan, &input),
                );
                assert_complex_bits(
                    &format!("forward size {size}"),
                    &plan.forward(&input).unwrap(),
                    &old::forward(&plan, &input),
                );
            }
        }
    }

    #[test]
    fn inverse_into_matches_old_inverse_bitwise() {
        let mut rng = TestRng(0xD1B5_4A32_D192_ED03);
        let mut out = Vec::new();
        let mut scratch = RealFftScratch::default();
        for &size in &SIZES {
            let plan = RealFftPlan::new(size).unwrap();
            for _ in 0..3 {
                // Arbitrary (not necessarily conjugate-symmetric) bins: the
                // arithmetic must agree on whatever it is handed.
                let spectrum: Vec<ComplexF32> = (0..size / 2 + 1)
                    .map(|_| ComplexF32::new(rng.value(), rng.value()))
                    .collect();
                plan.inverse_into(&spectrum, &mut out, &mut scratch)
                    .unwrap();
                let want = old::inverse(&plan, &spectrum);
                assert_eq!(out.len(), want.len());
                for (i, (g, w)) in out.iter().zip(&want).enumerate() {
                    assert_eq!(g.to_bits(), w.to_bits(), "inverse_into size {size}[{i}]");
                }
                let wrapped = plan.inverse(&spectrum).unwrap();
                for (g, w) in wrapped.iter().zip(&want) {
                    assert_eq!(g.to_bits(), w.to_bits(), "inverse size {size}");
                }
            }
        }
    }

    #[test]
    fn cached_plan_transforms_like_a_fresh_one() {
        let mut rng = TestRng(0x2545_F491_4F6C_DD1D);
        for size in [400usize, 512, 6] {
            let cached = RealFftPlan::cached(size).unwrap();
            // Not asserting pointer identity: the bounded cache may be
            // cleared by concurrently running tests between two calls.
            let fresh = RealFftPlan::new(size).unwrap();
            let input = rng.vec(size);
            assert_complex_bits(
                &format!("cached size {size}"),
                &cached.forward(&input).unwrap(),
                &fresh.forward(&input).unwrap(),
            );
        }
        // Errors are returned, not cached.
        assert!(RealFftPlan::cached(7).is_err());
        assert!(RealFftPlan::cached(7).is_err());
        assert!(RealFftPlan::cached(0).is_err());
    }
}
