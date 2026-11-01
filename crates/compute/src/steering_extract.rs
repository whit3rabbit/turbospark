//! Difference-of-means direction extraction and per-layer diagnostics, for
//! trait steering (`turbospark-steer`).
//!
//! This is the Rust twin of the reduction half of
//! `scripts/extract_direction.py`, which until this module existed was the
//! only implementation of the math. The capture half stays where it was
//! (`TURBOSPARK_RESID_CAPTURE`, `crates/runtime/src/resid_capture.rs`), and
//! the SVD method stays script-only: the mean is the standard extraction and
//! needs no judgement about rank, and a Rust SVD would be a new linear
//! algebra dependency for a variant the default never needs.
//!
//! # The four columns, and why there are four
//!
//! Every quantity below divides the direction norm `||d_l||` by something
//! different, and each answers a different question (the full measured story
//! is `docs/OBLITERATION.md`):
//!
//! - `norm`: the raw magnitude. It rises with the residual stream's own
//!   scale with depth, so it is comparable across layers only by accident.
//! - `separation`: `||mean_p - mean_n||` over the pooled within-set spread.
//!   Cohen's d generalized to a vector: where does the CONCEPT live.
//! - `share`: `||d_l|| / mean_||x_l||` over both sets: what fraction of the
//!   stream an ablation at `alpha = 1` would touch, which is the quantity
//!   that predicts the collapse.
//! - `removed`: `mean(|c_hat|) / mean_||x_l||`: the fraction the kernel
//!   ACTUALLY subtracts (`|c_hat| = |d . x| / ||d||`), which through the deep
//!   layers sits a factor of about 2.0 under `share` on natural corpora and
//!   180x UNDER it at layer 0 -- inverting which layer reads cheapest.
//!
//! All four RANK layers. None of them predicts the usable alpha: the derived
//! ceiling was published as validated and then REFUTED on the second
//! direction (measured 0.8 against a derived 0.09, relationship inverted),
//! so [`derived_alpha_cap`] exists only to carry the same printed diagnostic
//! the script prints, with the same warning.
//!
//! Accumulation is in f64 and the written directions are f32, matching the
//! script line for line.

/// One side of an extraction corpus: residual snapshots captured at the last
/// prompt token, one row per prompt.
///
/// `rows[r]` is `layers * hidden` floats, the same flat layout the capture
/// sidecar writes.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureSet {
    pub layers: usize,
    pub hidden: usize,
    pub rows: Vec<Vec<f32>>,
}

impl CaptureSet {
    /// Build from per-snapshot flat rows, validating every row's length.
    pub fn new(layers: usize, hidden: usize, rows: Vec<Vec<f32>>) -> Result<Self, String> {
        if layers == 0 || hidden == 0 {
            return Err(format!("degenerate capture shape {layers} x {hidden}"));
        }
        if rows.is_empty() {
            return Err("capture set holds no snapshots".to_string());
        }
        let want = layers * hidden;
        for (i, row) in rows.iter().enumerate() {
            if row.len() != want {
                return Err(format!(
                    "snapshot {i} holds {} floats, expected {want} ({layers} x {hidden})",
                    row.len()
                ));
            }
        }
        Ok(Self {
            layers,
            hidden,
            rows,
        })
    }
}

/// One direction per layer, `d_l = mean(positive at l) - mean(negative at l)`,
/// accumulated in f64 and written f32.
///
/// The sets need not be the same size. An empty side is an error, because a
/// one-sided difference of means is a centroid and not a direction.
pub fn difference_of_means(
    positive: &CaptureSet,
    negative: &CaptureSet,
) -> Result<Vec<Vec<f32>>, String> {
    if positive.layers != negative.layers || positive.hidden != negative.hidden {
        return Err(format!(
            "capture sets disagree on shape: {} x {} vs {} x {}",
            positive.layers, positive.hidden, negative.layers, negative.hidden
        ));
    }
    let mut dirs = Vec::with_capacity(positive.layers);
    for l in 0..positive.layers {
        let off = l * positive.hidden;
        // Two accumulators: the means are subtracted per SIDE, after each
        // side's own sum is divided by its own count. (One combined
        // accumulator would subtract twice -- caught by the exact-gap test.)
        let mut accp = vec![0.0f64; positive.hidden];
        let mut accn = vec![0.0f64; positive.hidden];
        for row in &positive.rows {
            for (a, v) in accp.iter_mut().zip(&row[off..off + positive.hidden]) {
                *a += f64::from(*v);
            }
        }
        for row in &negative.rows {
            for (a, v) in accn.iter_mut().zip(&row[off..off + positive.hidden]) {
                *a += f64::from(*v);
            }
        }
        let pn = positive.rows.len() as f64;
        let nn = negative.rows.len() as f64;
        dirs.push(
            accp.iter()
                .zip(&accn)
                .map(|(p, n)| (p / pn - n / nn) as f32)
                .collect(),
        );
    }
    Ok(dirs)
}

/// Per-layer ranking diagnostics for one extracted direction set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerStats {
    /// `||d_l||`, raw. Rises with the stream's own scale.
    pub norm: f64,
    /// `||mean_p - mean_n|| / pooled spread`: effect size, comparable across
    /// layers. Zero when either set has no spread.
    pub separation: f64,
    /// `||d_l|| / mean_||x_l||` over both sets.
    pub share: f64,
    /// `mean(|d . x| / ||d||) / mean_||x_l||`: what an ablation at alpha 1
    /// actually subtracts.
    pub removed: f64,
}

/// Compute [`LayerStats`] for every layer of an extracted direction set.
pub fn layer_stats(
    positive: &CaptureSet,
    negative: &CaptureSet,
    directions: &[Vec<f32>],
) -> Result<Vec<LayerStats>, String> {
    if positive.layers != negative.layers || positive.hidden != negative.hidden {
        return Err(format!(
            "capture sets disagree on shape: {} x {} vs {} x {}",
            positive.layers, positive.hidden, negative.layers, negative.hidden
        ));
    }
    if directions.len() != positive.layers {
        return Err(format!(
            "directions hold {} layers, captures describe {}",
            directions.len(),
            positive.layers
        ));
    }
    let hidden = positive.hidden;
    let mut out = Vec::with_capacity(positive.layers);
    for (l, d) in directions.iter().enumerate() {
        if d.len() != hidden {
            return Err(format!(
                "direction {l} holds {} floats, expected {hidden}",
                d.len()
            ));
        }
        let off = l * hidden;
        // Centroids in f64.
        let mut pc = vec![0.0f64; hidden];
        let mut nc = vec![0.0f64; hidden];
        for row in &positive.rows {
            for (a, v) in pc.iter_mut().zip(&row[off..off + hidden]) {
                *a += f64::from(*v);
            }
        }
        for row in &negative.rows {
            for (a, v) in nc.iter_mut().zip(&row[off..off + hidden]) {
                *a += f64::from(*v);
            }
        }
        let pn = positive.rows.len() as f64;
        let nn = negative.rows.len() as f64;
        for v in &mut pc {
            *v /= pn;
        }
        for v in &mut nc {
            *v /= nn;
        }
        // Gap and pooled spread. The gap is the centroid distance; the
        // spreads are mean squared distances from each set's OWN centroid
        // (the script's `vp`/`vn`), pooled with a factor of 1/2.
        let mut gap2 = 0.0f64;
        for dim in 0..hidden {
            let g = pc[dim] - nc[dim];
            gap2 += g * g;
        }
        let gap = gap2.sqrt();
        let mut vp = 0.0f64;
        for row in &positive.rows {
            let mut d2 = 0.0f64;
            for (dim, v) in row[off..off + hidden].iter().enumerate() {
                let c = f64::from(*v) - pc[dim];
                d2 += c * c;
            }
            vp += d2;
        }
        let mut vn = 0.0f64;
        for row in &negative.rows {
            let mut d2 = 0.0f64;
            for (dim, v) in row[off..off + hidden].iter().enumerate() {
                let c = f64::from(*v) - nc[dim];
                d2 += c * c;
            }
            vn += d2;
        }
        let spread = (0.5 * (vp / pn + vn / nn)).sqrt();
        let separation = if spread > 0.0 { gap / spread } else { 0.0 };

        // Direction norm.
        let dn: f64 = d
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();

        // Mean row norm over both sets, and mean |c_hat|.
        let mut stream_sum = 0.0f64;
        let mut chat_sum = 0.0f64;
        let mut count = 0.0f64;
        for rows in [&positive.rows, &negative.rows] {
            for row in rows.iter() {
                let mut x2 = 0.0f64;
                let mut dot = 0.0f64;
                for (dim, v) in row[off..off + hidden].iter().enumerate() {
                    let x = f64::from(*v);
                    x2 += x * x;
                    dot += x * f64::from(d[dim]);
                }
                stream_sum += x2.sqrt();
                chat_sum += dot.abs() / dn.max(1e-300);
                count += 1.0;
            }
        }
        let stream = stream_sum / count.max(1.0);
        let share = if stream > 0.0 { dn / stream } else { 0.0 };
        let removed = if stream > 0.0 && dn > 0.0 {
            (chat_sum / count.max(1.0)) / stream
        } else {
            0.0
        };

        out.push(LayerStats {
            norm: dn,
            separation,
            share,
            removed,
        });
    }
    Ok(out)
}

/// The largest `ablate` alpha that keeps the hottest layer's removal under
/// `budget` of the stream.
///
/// A PRINTED DIAGNOSTIC AND NOT A PREDICTOR. It was published as a validated
/// ceiling and then REFUTED: measured against `crates/bench/tests/
/// steering_sweep.rs` on two directions from one checkpoint it under-called
/// the usable band by 1.1x and 8.9x with the relationship INVERTED (the
/// direction carrying 3.8x the share tolerated 2x MORE alpha). What survives
/// is the layer ordering. Callers print it beside that warning; nobody steers
/// at it.
pub fn derived_alpha_cap(stats: &[LayerStats], budget: f64) -> f64 {
    let worst = stats.iter().map(|s| s.share).fold(0.0f64, f64::max);
    if worst > 0.0 {
        budget / worst
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(layers: usize, hidden: usize, rows: Vec<Vec<f32>>) -> CaptureSet {
        CaptureSet::new(layers, hidden, rows).expect("valid fixture")
    }

    /// Two dimensions, one layer. Positive rows sit at x = +2, negative at
    /// x = -2, dead dimension y carries noise-free equality. The direction
    /// must be exactly [4, 0] (a difference of two exact means).
    #[test]
    fn difference_of_means_is_the_exact_gap_between_centroids() {
        let pos = set(1, 2, vec![vec![2.0, 1.0], vec![2.0, 3.0], vec![2.0, -1.0]]);
        let neg = set(
            1,
            2,
            vec![vec![-2.0, 1.0], vec![-2.0, 3.0], vec![-2.0, -1.0]],
        );
        let dirs = difference_of_means(&pos, &neg).expect("directions");
        assert_eq!(dirs.len(), 1);
        assert!((dirs[0][0] - 4.0).abs() < 1e-5, "dir x = {}", dirs[0][0]);
        assert!(dirs[0][1].abs() < 1e-5, "dir y = {}", dirs[0][1]);
    }

    /// Unequal set sizes weight each side by its own mean, not by a pooled
    /// mean. pos mean x = +2 (3 rows), neg mean x = -1 (2 rows), so d.x = 3.
    #[test]
    fn unequal_set_sizes_use_each_sides_own_mean() {
        let pos = set(1, 1, vec![vec![2.0], vec![2.0], vec![2.0]]);
        let neg = set(1, 1, vec![vec![-1.0], vec![-1.0]]);
        let dirs = difference_of_means(&pos, &neg).expect("directions");
        assert!((dirs[0][0] - 3.0).abs() < 1e-5, "dir = {}", dirs[0][0]);
    }

    /// The classic degenerate-parameter trap this crate records for the
    /// kernel applies here too: swapping the sides must flip the direction's
    /// sign and change NO stat.
    #[test]
    fn swapping_sides_flips_the_sign_and_moves_no_stat() {
        let pos = set(1, 2, vec![vec![3.0, 0.5], vec![2.5, -0.5], vec![3.5, 0.0]]);
        let neg = set(
            1,
            2,
            vec![vec![-3.0, 0.5], vec![-2.5, -0.5], vec![-3.5, 0.0]],
        );
        let dirs = difference_of_means(&pos, &neg).expect("directions");
        let flipped = difference_of_means(&neg, &pos).expect("directions");
        for l in 0..1 {
            for dim in 0..2 {
                assert!(
                    (dirs[l][dim] + flipped[l][dim]).abs() < 1e-5,
                    "flipped side must negate the direction"
                );
            }
        }
        let a = layer_stats(&pos, &neg, &dirs).expect("stats");
        let b = layer_stats(&pos, &neg, &flipped).expect("stats");
        for (x, y) in a.iter().zip(&b) {
            assert!((x.norm - y.norm).abs() < 1e-9);
            assert!((x.separation - y.separation).abs() < 1e-9);
            assert!((x.share - y.share).abs() < 1e-9);
            // `removed` uses |c_hat|, which is sign-free by construction.
            assert!((x.removed - y.removed).abs() < 1e-9);
        }
    }

    /// Separation divides by within-set spread, so scaling EVERY row by a
    /// constant must leave it unchanged while the direction norm scales.
    /// This is the property that makes `sep` comparable across layers and
    /// `norm` not.
    #[test]
    fn separation_is_scale_invariant_and_norm_is_not() {
        let mk = |scale: f32| {
            (
                set(
                    1,
                    2,
                    vec![vec![4.0 * scale, scale], vec![4.0 * scale, -scale]],
                ),
                set(
                    1,
                    2,
                    vec![vec![-4.0 * scale, scale], vec![-4.0 * scale, -scale]],
                ),
            )
        };
        let (p1, n1) = mk(1.0);
        let (p2, n2) = mk(10.0);
        let d1 = difference_of_means(&p1, &n1).expect("dirs");
        let d2 = difference_of_means(&p2, &n2).expect("dirs");
        let s1 = layer_stats(&p1, &n1, &d1).expect("stats");
        let s2 = layer_stats(&p2, &n2, &d2).expect("stats");
        assert!(
            (s1[0].separation - s2[0].separation).abs() < 1e-9,
            "sep must be scale invariant: {} vs {}",
            s1[0].separation,
            s2[0].separation
        );
        assert!(
            (s1[0].share - s2[0].share).abs() < 1e-9,
            "share must be scale invariant"
        );
        assert!(
            (s2[0].norm - s1[0].norm * 10.0).abs() < 1e-6,
            "norm must scale with the rows"
        );
    }

    /// A degenerate set (every positive row identical to every negative row)
    /// yields a zero direction and zero separation without dividing by zero.
    #[test]
    fn identical_sets_are_a_zero_direction_with_zero_separation() {
        let rows = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
        let pos = set(1, 3, rows.clone());
        let neg = set(1, 3, rows.clone());
        let dirs = difference_of_means(&pos, &neg).expect("dirs");
        assert!(dirs[0].iter().all(|v| v.abs() < 1e-6));
        let stats = layer_stats(&pos, &neg, &dirs).expect("stats");
        assert!(stats[0].separation.abs() < 1e-12);
        assert!(stats[0].share.abs() < 1e-12);
        assert_eq!(derived_alpha_cap(&stats, 0.10), f64::INFINITY);
    }

    /// A row exactly equal to the direction has |c_hat| = ||d||, so `removed`
    /// reads 1.0 when one side sits ON the direction and the other at the
    /// origin. This is the arithmetic the kernel subtracts, distinct from
    /// `share`'s ||d||/||x||.
    #[test]
    fn removed_measures_the_projected_component_not_the_direction() {
        // Direction will be [2, 0]. Positive rows sit exactly on it at
        // [2, 0]; negative rows at the origin [0, 0]. Stream mean norm is
        // 1.0; direction norm is 2.0; mean |c_hat| is (2 + 0) / 2 = 1.0.
        let pos = set(1, 2, vec![vec![2.0, 0.0], vec![2.0, 0.0]]);
        let neg = set(1, 2, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        let dirs = difference_of_means(&pos, &neg).expect("dirs");
        let stats = layer_stats(&pos, &neg, &dirs).expect("stats");
        assert!((stats[0].norm - 2.0).abs() < 1e-5);
        assert!(
            (stats[0].share - 2.0).abs() < 1e-5,
            "share = ||d||/||x|| = 2/1"
        );
        assert!(
            (stats[0].removed - 1.0).abs() < 1e-5,
            "removed = mean|c_hat|/||x|| = 1/1, not share's 2.0"
        );
    }

    /// The cap is budget over the worst share, and it is a DIAGNOSTIC: the
    /// test pins the arithmetic, not a claim that the cap predicts anything
    /// (see the module docs for the two-direction refutation).
    #[test]
    fn alpha_cap_is_budget_over_the_worst_share() {
        let stats = vec![
            LayerStats {
                norm: 1.0,
                separation: 0.5,
                share: 0.10,
                removed: 0.05,
            },
            LayerStats {
                norm: 2.0,
                separation: 0.6,
                share: 0.25,
                removed: 0.12,
            },
        ];
        assert!((derived_alpha_cap(&stats, 0.10) - 0.4).abs() < 1e-12);
    }

    /// Shape mismatches must be errors, not silent misalignment: a capture
    /// set from a different model paired with the wrong direction set would
    /// otherwise extract plausible garbage.
    #[test]
    fn mismatched_shapes_are_refused() {
        let a = set(2, 4, vec![vec![0.0; 8]; 3]);
        let b = set(2, 8, vec![vec![0.0; 16]; 3]);
        assert!(difference_of_means(&a, &b).is_err());
        let dirs = difference_of_means(&a, &a).expect("same shapes");
        assert!(layer_stats(&a, &b, &dirs).is_err());
        assert!(layer_stats(&a, &a, &dirs[..1]).is_err());
    }
}
