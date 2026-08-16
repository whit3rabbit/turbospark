//! Env-gated dense-FFN activation census (`MFERENCE_FFN_HIST=/path.json`).
//!
//! Answers whether Muse Glimmer's FFN has exploitable ACTIVATION SPARSITY:
//! if most of `silu(gate) * up` is near zero per token, only the hot rows of
//! gate/up and columns of down need to be in memory, and a
//! PowerInfer-style neuron cache (the dense analog of the expert cache)
//! becomes worth designing. If the mass is spread, that idea is a measured
//! dead end for this checkpoint, like `docs/EXPERT_ROUTING.md`'s. Diagnostic
//! only, never on by default.
//!
//! The capture costs NO extra dispatch and changes NO math: `silu_mul`'s
//! output is redirected into a `layers x inter` capture buffer (one region
//! per layer, so the per-layer vectors survive the token's single command
//! buffer, where `scratch.ffn_act` is overwritten 52 times), and `down_proj`
//! reads from the same region. The kernel's output does not depend on its
//! destination address, so generated text is byte-identical with the flag on.
//!
//! Only non-`skip_head` passes are read back (decode plus the last prompt
//! token): prefill routes differently (the `router_hist` lesson) and a
//! per-prefill-token 2 MB readback would slow a long prompt for data the
//! analysis would then have to exclude anyway. Prefill passes are counted so
//! the capture can say what it skipped.
//!
//! Per-layer accumulators, all computed online so the JSON stays MB-scale:
//! - `mass[neuron]`: cumulative |act|, for coverage curves (n95/n99) and
//!   static-hot-set analysis offline;
//! - log2 magnitude histogram (count and mass per bucket), for the
//!   CATS-style threshold sweep;
//! - top-K overlap between CONSECUTIVE passes at K in `OVERLAP_KS`, the
//!   temporal-locality number that decides whether a neuron cache can work
//!   at all (it is what makes the expert cache hit at ~84%).
//!
//! Sanity check on any capture: per layer,
//! `sum(hist_count) == decode_passes * inter`.

use model_io::{ArchConfig, ModelFamily};

/// Top-K sizes for the consecutive-pass overlap statistic. 4,096 of Muse
/// Glimmer's 19,968 is ~20%; anything a cache could plausibly hold is a
/// prefix of that.
pub(crate) const OVERLAP_KS: [usize; 4] = [512, 1024, 2048, 4096];

/// Log2 magnitude buckets. Bucket 0 is exact zero, 63 is non-finite, and
/// bucket `b` in 1..=62 covers `[2^(b-33), 2^(b-32))`, so bucket 33 is
/// `[1, 2)`. FP16's whole range (subnormal 6e-8 to 65504) maps inside.
pub(crate) const NUM_BUCKETS: usize = 64;
const BUCKET_EXP_OFFSET: i32 = 33;

/// The pure accumulator, separated from the Metal capture buffer so its
/// arithmetic is testable without a GPU device.
pub(crate) struct FfnActAccum {
    layers: usize,
    inter: usize,
    /// Effective overlap Ks (`OVERLAP_KS` clamped to `inter`, which only
    /// matters for test fixtures).
    ks: [usize; OVERLAP_KS.len()],
    /// `mass[layer][neuron]`: cumulative |activation| over decode passes.
    mass: Vec<Vec<f64>>,
    hist_count: Vec<[u64; NUM_BUCKETS]>,
    hist_mass: Vec<[f64; NUM_BUCKETS]>,
    /// `overlap_hits[layer][ki]`: sum over consecutive pass pairs of
    /// `|top-K(this pass) intersect top-K(previous pass)|`.
    overlap_hits: Vec<[u64; OVERLAP_KS.len()]>,
    /// Bitset (`inter` bits) of the previous pass's top-K, per layer per K.
    prev_top: Vec<[Vec<u64>; OVERLAP_KS.len()]>,
    /// Consecutive pass pairs counted into `overlap_hits` (same for every
    /// layer, so kept once).
    overlap_pairs: u64,
    decode_passes: u64,
    prefill_passes: u64,
}

fn bucket_index(a: f32) -> usize {
    if a == 0.0 {
        return 0;
    }
    if !a.is_finite() {
        return NUM_BUCKETS - 1;
    }
    let e = a.log2().floor() as i32 + BUCKET_EXP_OFFSET;
    e.clamp(1, NUM_BUCKETS as i32 - 2) as usize
}

impl FfnActAccum {
    pub(crate) fn new(layers: usize, inter: usize) -> Self {
        let words = inter.div_ceil(64);
        let ks = OVERLAP_KS.map(|k| k.min(inter));
        Self {
            layers,
            inter,
            ks,
            mass: vec![vec![0.0; inter]; layers],
            hist_count: vec![[0; NUM_BUCKETS]; layers],
            hist_mass: vec![[0.0; NUM_BUCKETS]; layers],
            overlap_hits: vec![[0; OVERLAP_KS.len()]; layers],
            prev_top: (0..layers)
                .map(|_| std::array::from_fn(|_| vec![0u64; words]))
                .collect(),
            overlap_pairs: 0,
            decode_passes: 0,
            prefill_passes: 0,
        }
    }

    pub(crate) fn note_prefill_pass(&mut self) {
        self.prefill_passes += 1;
    }

    /// One decode pass: `acts` is the whole capture, `layers * inter` values
    /// in layer-major order.
    pub(crate) fn record_pass(&mut self, acts: &[f32]) {
        assert_eq!(acts.len(), self.layers * self.inter, "capture size");
        let count_pair = self.decode_passes > 0;
        for layer in 0..self.layers {
            let row = &acts[layer * self.inter..(layer + 1) * self.inter];
            self.record_layer(layer, row, count_pair);
        }
        if count_pair {
            self.overlap_pairs += 1;
        }
        self.decode_passes += 1;
    }

    fn record_layer(&mut self, layer: usize, row: &[f32], count_pair: bool) {
        let mass = &mut self.mass[layer];
        let hist_count = &mut self.hist_count[layer];
        let hist_mass = &mut self.hist_mass[layer];
        for (j, &v) in row.iter().enumerate() {
            let a = v.abs();
            mass[j] += a as f64;
            let b = bucket_index(a);
            hist_count[b] += 1;
            hist_mass[b] += a as f64;
        }

        // Top-K ids by |act|, descending: one partial select at the largest
        // K, then every smaller K is a prefix of the sorted top slice.
        let k_max = self.ks[OVERLAP_KS.len() - 1];
        let mut order: Vec<u32> = (0..self.inter as u32).collect();
        let desc = |&a: &u32, &b: &u32| {
            row[b as usize]
                .abs()
                .partial_cmp(&row[a as usize].abs())
                .expect("finite activations")
                // Tie-break on the index so the census is deterministic
                // for a replayed capture even where FP16 values tie.
                .then(a.cmp(&b))
        };
        if k_max < self.inter {
            order.select_nth_unstable_by(k_max - 1, desc);
        }
        order[..k_max].sort_unstable_by(desc);

        for (ki, &k) in self.ks.iter().enumerate() {
            if count_pair {
                let prev = &self.prev_top[layer][ki];
                let hits = order[..k]
                    .iter()
                    .filter(|&&id| prev[id as usize / 64] >> (id % 64) & 1 == 1)
                    .count();
                self.overlap_hits[layer][ki] += hits as u64;
            }
            let bits = &mut self.prev_top[layer][ki];
            bits.iter_mut().for_each(|w| *w = 0);
            for &id in &order[..k] {
                bits[id as usize / 64] |= 1 << (id % 64);
            }
        }
    }

    // Hand-rolled: this crate deliberately carries no serde (the
    // `router_hist` precedent).
    pub(crate) fn to_json(&self) -> String {
        let mut out = format!(
            "{{\"num_layers\":{},\"inter\":{},\"decode_passes\":{},\
             \"prefill_passes\":{},\"bucket_exp_offset\":{BUCKET_EXP_OFFSET},\
             \"overlap_ks\":{:?},\"overlap_pairs\":{},\"overlap_hits\":[",
            self.layers,
            self.inter,
            self.decode_passes,
            self.prefill_passes,
            self.ks,
            self.overlap_pairs
        );
        push_rows(&mut out, self.overlap_hits.iter().map(|r| r.iter()), |v| {
            v.to_string()
        });
        out.push_str("],\"hist_count\":[");
        push_rows(&mut out, self.hist_count.iter().map(|r| r.iter()), |v| {
            v.to_string()
        });
        out.push_str("],\"hist_mass\":[");
        push_rows(&mut out, self.hist_mass.iter().map(|r| r.iter()), |v| {
            format!("{v:.5e}")
        });
        out.push_str("],\"mass\":[");
        push_rows(&mut out, self.mass.iter().map(|r| r.iter()), |v| {
            format!("{v:.5e}")
        });
        out.push_str("]}");
        out
    }
}

/// `[[a,b],[c]]` without the trailing bracket, which the caller closes.
fn push_rows<R, I, T>(out: &mut String, rows: R, fmt: impl Fn(&T) -> String)
where
    R: Iterator<Item = I>,
    I: Iterator<Item = T>,
    T: Copy,
{
    for (i, row) in rows.enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        for (j, value) in row.enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push_str(&fmt(&value));
        }
        out.push(']');
    }
}

/// The env-gated census: the accumulator plus the Metal capture buffer the
/// muse flow redirects `silu_mul` into.
pub(crate) struct FfnActHist {
    path: std::path::PathBuf,
    pub(crate) capture: gpu::MetalBuffer,
    accum: FfnActAccum,
}

impl FfnActHist {
    /// `Some` only when `MFERENCE_FFN_HIST` names an output path AND the
    /// family is the one whose flow feeds the capture. Constructing it for
    /// a family that never redirects would write a file of zeros that reads
    /// as "perfectly sparse", the worst kind of wrong.
    pub(crate) fn from_env(context: &gpu::MetalContext, arch: &ArchConfig) -> Option<Self> {
        let path = std::env::var_os("MFERENCE_FFN_HIST")?;
        if arch.family != ModelFamily::MuseGlimmer {
            eprintln!(
                "[ffn-hist] MFERENCE_FFN_HIST is wired for the museGlimmer flow only; \
                 family {:?} does not feed the capture, ignoring",
                arch.family
            );
            return None;
        }
        let layers = arch.num_layers as usize;
        let inter = arch.intermediate_size as usize;
        let capture = context.new_output_buffer((layers * inter * 2) as u64);
        Some(Self {
            path: path.into(),
            capture,
            accum: FfnActAccum::new(layers, inter),
        })
    }

    pub(crate) fn note_prefill_pass(&mut self) {
        self.accum.note_prefill_pass();
    }

    /// Read the whole capture back and fold it in. Called after the token's
    /// command buffer has been waited on, so every layer's region is final.
    pub(crate) fn record_pass(&mut self) {
        let n = self.accum.layers * self.accum.inter;
        let acts = gpu::read_buffer_f16(&self.capture, 0, n);
        let f32s: Vec<f32> = acts.iter().map(|v| v.to_f32()).collect();
        self.accum.record_pass(&f32s);
    }
}

impl Drop for FfnActHist {
    fn drop(&mut self) {
        match std::fs::write(&self.path, self.accum.to_json()) {
            Ok(()) => eprintln!("[ffn-hist] wrote {}", self.path.display()),
            Err(e) => eprintln!("[ffn-hist] FAILED writing {}: {e}", self.path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_index_maps_zero_edges_and_powers() {
        assert_eq!(bucket_index(0.0), 0);
        assert_eq!(bucket_index(1.0), 33); // [1, 2)
        assert_eq!(bucket_index(0.5), 32); // [0.5, 1)
        assert_eq!(bucket_index(1.999), 33);
        assert_eq!(bucket_index(2.0), 34);
        assert_eq!(bucket_index(65504.0), 48); // fp16 max, in [2^15, 2^16)
        assert_eq!(bucket_index(6.0e-8), 9); // fp16 subnormal floor
        assert_eq!(bucket_index(f32::INFINITY), NUM_BUCKETS - 1);
    }

    #[test]
    fn overlap_counts_intersection_of_consecutive_top_k_sets() {
        // The fixture must DISCRIMINATE (AGENTS.md Gotcha 48's rule): k must
        // be strictly under inter, or every top-K is the whole row and any
        // intersection code reads full overlap. 2048 wide keeps
        // OVERLAP_KS[0] = 512 unclamped.
        let mut acc = FfnActAccum::new(1, 2048);
        let mut x = vec![0.0f32; 2048];
        let mut y = vec![0.0f32; 2048];
        x[..512].fill(5.0); // top-512 of x is ids 0..512
        y[1024..1536].fill(5.0); // top-512 of y is ids 1024..1536, disjoint
        acc.record_pass(&x);
        acc.record_pass(&y);
        acc.record_pass(&y);
        // Pair 1 (x, y): disjoint at k = 512 contributes 0.
        // Pair 2 (y, y): identical contributes 512.
        assert_eq!(acc.overlap_hits[0][0], 512);
        assert_eq!(acc.overlap_pairs, 2);
    }

    #[test]
    fn hist_count_sums_to_passes_times_inter() {
        let mut acc = FfnActAccum::new(2, 32);
        let row: Vec<f32> = (0..64).map(|i| i as f32 * 0.25).collect();
        acc.record_pass(&row);
        acc.record_pass(&row);
        for layer in 0..2 {
            let total: u64 = acc.hist_count[layer].iter().sum();
            assert_eq!(total, 2 * 32);
        }
        assert_eq!(acc.decode_passes, 2);
    }

    #[test]
    fn json_serializes_a_tiny_capture_exactly() {
        let mut acc = FfnActAccum::new(1, 2);
        acc.note_prefill_pass();
        acc.record_pass(&[1.0, 0.0]);
        let json = acc.to_json();
        assert!(
            json.starts_with(
                "{\"num_layers\":1,\"inter\":2,\"decode_passes\":1,\
                 \"prefill_passes\":1,\"bucket_exp_offset\":33,\
                 \"overlap_ks\":[2, 2, 2, 2],\"overlap_pairs\":0,"
            ),
            "json header: {json}"
        );
        assert!(
            json.contains("\"mass\":[[1.00000e0,0.00000e0]]}"),
            "mass tail: {json}"
        );
        // Bucket 33 is [1, 2), bucket 0 is exact zero: one count each,
        // buckets 1..=32 empty between them.
        let expected = format!("\"hist_count\":[[1,{}1,", "0,".repeat(32));
        assert!(json.contains(&expected), "hist_count: {json}");
    }
}
