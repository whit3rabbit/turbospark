//! Small waveform operations and window functions.
//!
//! Port of the pieces of upstream audio.cpp's `waveform_ops.cpp` the other
//! modules share, plus the periodic Hann window the STFT and mel frontend
//! consume. All math runs in f64 internally and stores f32, so cascading
//! these operations does not accumulate f32 rounding faster than necessary.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};

/// A small process-wide cache of immutable DSP tables (FFT plans, mel
/// filterbanks, resampling kernels) keyed by their exact build parameters.
///
/// The values are pure functions of the key, so a hit is bit-identical to a
/// rebuild. The cache is bounded twice over so a long-lived process cannot
/// pin unbounded memory: callers mark oversized tables as not cacheable, and
/// the map is cleared when it reaches `max_entries`.
pub(crate) struct TableCache<K, V> {
    inner: OnceLock<Mutex<HashMap<K, Arc<V>>>>,
    max_entries: usize,
}

impl<K: Eq + Hash, V> TableCache<K, V> {
    pub(crate) const fn new(max_entries: usize) -> Self {
        Self {
            inner: OnceLock::new(),
            max_entries,
        }
    }

    /// Returns the cached table for `key`, building it with `build` on a
    /// miss. `build` runs outside the lock, so two threads racing on one key
    /// may both build; they produce identical tables and one wins.
    pub(crate) fn get_or_build<E>(
        &self,
        key: K,
        cacheable: bool,
        build: impl FnOnce() -> Result<V, E>,
    ) -> Result<Arc<V>, E> {
        let map = self.inner.get_or_init(|| Mutex::new(HashMap::new()));
        if cacheable {
            // A poisoned lock only means another thread panicked while
            // holding it; the map itself is always left consistent.
            let guard = map.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(hit) = guard.get(&key) {
                return Ok(Arc::clone(hit));
            }
        }
        let built = Arc::new(build()?);
        if cacheable {
            let mut guard = map.lock().unwrap_or_else(|e| e.into_inner());
            if guard.len() >= self.max_entries && !guard.contains_key(&key) {
                guard.clear();
            }
            return Ok(Arc::clone(guard.entry(key).or_insert(built)));
        }
        Ok(built)
    }
}

/// Deterministic xorshift stream shared by the bitwise-parity tests; avoids
/// a dev-dependency.
#[cfg(test)]
pub(crate) struct TestRng(pub(crate) u64);

#[cfg(test)]
impl TestRng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Values in about [-8, 8] over several magnitudes, with ~1 in 6 exact
    /// zeros so zero handling shows up in the bits.
    pub(crate) fn value(&mut self) -> f32 {
        let r = self.next();
        if r % 6 == 0 {
            return 0.0;
        }
        let mantissa = ((r >> 8) % 20001) as f32 / 10000.0 - 1.0;
        let scale = [0.001f32, 0.1, 1.0, 8.0][((r >> 40) % 4) as usize];
        mantissa * scale
    }

    pub(crate) fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.value()).collect()
    }
}

/// Periodic Hann window of `size` points: `0.5 * (1 - cos(2 pi i / N))`.
///
/// Periodic (not symmetric): the last point is not the mirror of the first,
/// which is what overlap-add reconstruction needs. N = 1 yields `[1.0]`.
pub fn hann_window(size: usize) -> Vec<f32> {
    match size {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => (0..size)
            .map(|i| {
                (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / size as f64).cos())) as f32
            })
            .collect(),
    }
}

/// The maximum absolute sample; 0.0 for an empty slice.
pub fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()))
}

/// Root mean square; 0.0 for an empty slice. Accumulated in f64.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Multiplies every sample by a linear gain.
pub fn scale_by_gain(samples: &[f32], gain: f32) -> Vec<f32> {
    samples.iter().map(|&s| s * gain).collect()
}

/// Scales the waveform so its peak reaches `target_peak`.
///
/// A silent input (peak 0) is returned unchanged rather than scaled by
/// infinity; the caller decides what silence should become.
pub fn normalize_peak(samples: &[f32], target_peak: f32) -> Vec<f32> {
    let current = peak(samples);
    if current == 0.0 || !current.is_finite() {
        return samples.to_vec();
    }
    scale_by_gain(samples, target_peak / current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_endpoints_and_sum() {
        let w = hann_window(8);
        assert_eq!(w[0], 0.0);
        // Periodic Hann wraps: point 4 is the peak (1.0 at N/2 for even N).
        assert!((w[4] - 1.0).abs() < 1e-6);
        assert_eq!(hann_window(1), vec![1.0]);
        assert!(hann_window(0).is_empty());
    }

    #[test]
    fn peak_rms_and_gain() {
        assert_eq!(peak(&[-0.5, 0.25, 2.0, -3.0]), 3.0);
        assert_eq!(peak(&[]), 0.0);
        // rms of [-1, 1] is 1; of [3, 4] is sqrt(12.5).
        assert!((rms(&[-1.0, 1.0]) - 1.0).abs() < 1e-6);
        assert!((rms(&[3.0, 4.0]) - 3.53553).abs() < 1e-4);
        assert_eq!(scale_by_gain(&[1.0, -2.0], 0.5), vec![0.5, -1.0]);
    }

    #[test]
    fn normalize_peak_scales_to_target() {
        let out = normalize_peak(&[0.25, -0.5, 0.1], 1.0);
        assert!((out[1] + 1.0).abs() < 1e-6);
        assert_eq!(normalize_peak(&[0.0, 0.0], 1.0), vec![0.0, 0.0]);
    }
}
