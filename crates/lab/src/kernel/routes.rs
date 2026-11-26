//! Where `moe_routed` gets its routes (evaluator item LAB-28): a router
//! trace capture, uniform routing (the old model), or a Zipf draw.
//!
//! A capture is the JSON `TURBOSPARK_ROUTER_HIST=<path>` writes with
//! `TURBOSPARK_ROUTER_TRACE=1`: `num_experts`, `top_k` and `trace[layer]`,
//! the flat run of top-k expert ids per forward pass, in pass order and in
//! router rank order. `M` consecutive passes of one layer stand for the `M`
//! tokens one batched pair would serve (single-token forwards; a prefill
//! pass records in token order too, but a capture does not mark batches).

use serde_json::Value;

/// Passes a synthetic Zipf layer holds, enough windows at the widest M.
const SYNTHETIC_PASSES: usize = 4096;

#[derive(Debug, Clone, PartialEq)]
pub enum RouteSpec {
    Uniform,
    Zipf(f64),
    Trace(std::path::PathBuf),
}

impl RouteSpec {
    pub fn parse(s: &str) -> Result<Self, String> {
        if s == "uniform" {
            return Ok(Self::Uniform);
        }
        if let Some(path) = s.strip_prefix("trace:") {
            if path.is_empty() {
                return Err("--routes trace: needs a capture path".into());
            }
            return Ok(Self::Trace(path.into()));
        }
        if let Some(v) = s.strip_prefix("zipf:") {
            let s: f64 = v.parse().map_err(|_| format!("--routes zipf:{v:?}"))?;
            if !s.is_finite() || s < 0.0 {
                return Err(format!("--routes zipf:{v}: the exponent must be >= 0"));
            }
            return Ok(Self::Zipf(s));
        }
        Err(format!(
            "--routes {s:?}: one of uniform, zipf:<s>, trace:<capture.json>"
        ))
    }
}

/// Per layer, one `k`-long expert list per pass, router rank order.
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    pub layers: Vec<Vec<Vec<u32>>>,
}

/// Read a router capture and hold it to the install's shape: a capture of
/// another model would price another union.
pub fn parse_capture(text: &str, experts: usize, k: usize) -> Result<Trace, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("router capture: {e}"))?;
    let field = |name: &str| {
        v[name]
            .as_u64()
            .map(|x| x as usize)
            .ok_or_else(|| format!("router capture has no integer {name:?}"))
    };
    let (ce, ck) = (field("num_experts")?, field("top_k")?);
    if (ce, ck) != (experts, k) {
        return Err(format!(
            "router capture routes top-{ck} of {ce} experts; the install routes top-{k} of {experts}"
        ));
    }
    let rows = v["trace"]
        .as_array()
        .ok_or("router capture has no \"trace\": record it with TURBOSPARK_ROUTER_TRACE=1")?;
    let mut layers = Vec::new();
    for (layer, row) in rows.iter().enumerate() {
        let ids = row
            .as_array()
            .ok_or_else(|| format!("trace[{layer}] is not an array"))?;
        // Dense layers never record.
        if ids.is_empty() {
            continue;
        }
        if ids.len() % k != 0 {
            return Err(format!(
                "trace[{layer}] holds {} ids, not a multiple of top-{k}",
                ids.len()
            ));
        }
        let mut passes = Vec::with_capacity(ids.len() / k);
        for group in ids.chunks_exact(k) {
            let mut p = Vec::with_capacity(k);
            for id in group {
                let e = id
                    .as_u64()
                    .filter(|&e| (e as usize) < experts)
                    .ok_or_else(|| format!("trace[{layer}] has expert id {id} of {experts}"))?
                    as u32;
                if p.contains(&e) {
                    return Err(format!(
                        "trace[{layer}] routes expert {e} twice in one pass"
                    ));
                }
                p.push(e);
            }
            passes.push(p);
        }
        layers.push(passes);
    }
    if layers.is_empty() {
        return Err("router capture has an empty trace".into());
    }
    Ok(Trace { layers })
}

/// One layer of `passes` top-`k` draws without replacement, expert `i`
/// weighted `1 / (i + 1)^s`, ranked by the draw's own key. Deterministic.
pub fn zipf_trace(experts: usize, k: usize, s: f64, passes: usize, seed: u64) -> Trace {
    let weights: Vec<f64> = (0..experts).map(|i| ((i + 1) as f64).powf(-s)).collect();
    let mut state = seed | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // (0, 1): the log below must be finite.
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let layer = (0..passes)
        .map(|_| {
            // Weighted sampling without replacement: the k smallest
            // exponential keys.
            let mut keys: Vec<(f64, u32)> = weights
                .iter()
                .enumerate()
                .map(|(i, w)| (-uniform().ln() / w, i as u32))
                .collect();
            keys.sort_by(|a, b| a.0.total_cmp(&b.0));
            keys.truncate(k);
            keys.into_iter().map(|(_, i)| i).collect()
        })
        .collect();
    Trace {
        layers: vec![layer],
    }
}

pub fn zipf_default(experts: usize, k: usize, s: f64) -> Trace {
    zipf_trace(experts, k, s, SYNTHETIC_PASSES, 0x21BF_0001)
}

/// The routes of one batch of `M` tokens: `slots[t][rank]` is the cache slot
/// (0..union, first appearance order) of token `t`'s rank-`rank` expert.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    pub slots: Vec<Vec<u32>>,
    pub union: usize,
}

pub fn window_of(passes: &[Vec<u32>]) -> Window {
    let mut seen: Vec<u32> = Vec::new();
    let slots = passes
        .iter()
        .map(|p| {
            p.iter()
                .map(|&e| {
                    let at = seen.iter().position(|&s| s == e).unwrap_or_else(|| {
                        seen.push(e);
                        seen.len() - 1
                    });
                    at as u32
                })
                .collect()
        })
        .collect();
    Window {
        slots,
        union: seen.len(),
    }
}

/// Rows (tokens) each bound expert serves: `hist[r]` experts serve `r` rows.
pub fn rows_per_expert(w: &Window, hist: &mut std::collections::BTreeMap<usize, usize>) {
    let mut rows = vec![0usize; w.union];
    for t in &w.slots {
        for &s in t {
            rows[s as usize] += 1;
        }
    }
    for r in rows {
        *hist.entry(r).or_default() += 1;
    }
}

/// What the non-overlapping `m`-pass windows of a trace look like.
#[derive(Debug, Clone, PartialEq)]
pub struct Windows {
    pub count: usize,
    pub union_min: usize,
    pub union_median: usize,
    pub union_max: usize,
    /// Windows whose union is past `cap`.
    pub over_cap: usize,
    pub rows_per_expert: std::collections::BTreeMap<usize, usize>,
    /// The median-union window: the one the GPU arms time.
    pub typical: Window,
}

/// `None` when no layer holds `m` passes.
pub fn windows(trace: &Trace, m: usize, cap: usize) -> Option<Windows> {
    let mut all: Vec<Window> = trace
        .layers
        .iter()
        .flat_map(|layer| layer.chunks_exact(m.max(1)).map(window_of))
        .collect();
    if all.is_empty() {
        return None;
    }
    let mut hist = std::collections::BTreeMap::new();
    for w in &all {
        rows_per_expert(w, &mut hist);
    }
    let over_cap = all.iter().filter(|w| w.union > cap).count();
    // Stable, so equal unions keep capture order and the pick is repeatable.
    all.sort_by_key(|w| w.union);
    Some(Windows {
        count: all.len(),
        union_min: all[0].union,
        union_median: all[all.len() / 2].union,
        union_max: all[all.len() - 1].union,
        over_cap,
        rows_per_expert: hist,
        typical: all.swap_remove(all.len() / 2),
    })
}

/// The route list in `(token, rank)` order, and the same routes sorted by
/// slot (stable), the expert-ordered grid of phase 1. Phase 1 writes each
/// route's activations at its own `(token, rank)`, so the order changes
/// which threadgroups read which blob and not one bit of the result. Phase 2
/// reads the list by position and must take the first.
pub fn route_lists(slots: &[Vec<u32>]) -> (Vec<(u32, u32, u32)>, Vec<(u32, u32, u32)>) {
    let pair: Vec<(u32, u32, u32)> = slots
        .iter()
        .enumerate()
        .flat_map(|(t, ranks)| {
            ranks
                .iter()
                .enumerate()
                .map(move |(r, &slot)| (t as u32, r as u32, slot))
        })
        .collect();
    let mut sorted = pair.clone();
    sorted.sort_by_key(|r| r.2);
    (pair, sorted)
}

/// The largest swept M at which `arm` does not beat `reference`: above it the
/// arm wins at every swept M. `Some(0.0)` when it wins everywhere, `None`
/// when it loses at the widest.
pub fn wins_above(reference: &[(usize, f64)], arm: &[(usize, f64)]) -> Option<f64> {
    let mut above = 0.0;
    let mut last_loses = true;
    for (r, a) in reference.iter().zip(arm) {
        last_loses = a.1 >= r.1;
        if last_loses {
            above = a.0 as f64;
        }
    }
    (!last_loses).then_some(above)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_parse_and_bad_ones_are_named() {
        assert_eq!(RouteSpec::parse("uniform"), Ok(RouteSpec::Uniform));
        assert_eq!(RouteSpec::parse("zipf:1.2"), Ok(RouteSpec::Zipf(1.2)));
        assert_eq!(
            RouteSpec::parse("trace:/a/b.json"),
            Ok(RouteSpec::Trace("/a/b.json".into()))
        );
        for bad in ["zipf:-1", "zipf:x", "zipf:nan", "trace:", "skewed"] {
            assert!(RouteSpec::parse(bad).is_err(), "{bad}");
        }
    }

    /// The shape `router_hist.rs` writes, with a dense layer and a field
    /// this reader does not know.
    const CAPTURE: &str = r#"{"num_layers":3,"num_experts":4,"top_k":2,
        "counts":[[0,0,0,0],[1,1,1,1],[0,2,0,2]],
        "trace":[[],[2,0,2,1],[3,1,1,3]],"weights":[1]}"#;

    #[test]
    fn a_capture_keeps_rank_order_per_layer_and_skips_dense_layers() {
        let t = parse_capture(CAPTURE, 4, 2).unwrap();
        assert_eq!(
            t.layers,
            vec![vec![vec![2, 0], vec![2, 1]], vec![vec![3, 1], vec![1, 3]]]
        );
        let w = windows(&t, 2, 32).unwrap();
        // Layer 1 binds {2,0,1}, layer 2 binds {3,1}.
        assert_eq!((w.count, w.union_min, w.union_max), (2, 2, 3));
        assert_eq!(w.rows_per_expert[&2], 3, "expert 2, then 3 and 1");
        assert_eq!(w.rows_per_expert[&1], 2);
        assert_eq!(windows(&t, 3, 32), None);
        assert_eq!(windows(&t, 2, 2).unwrap().over_cap, 1);
    }

    #[test]
    fn a_capture_of_another_shape_or_a_corrupt_one_is_refused() {
        for (experts, k) in [(5, 2), (4, 3)] {
            assert!(parse_capture(CAPTURE, experts, k).is_err());
        }
        let no_trace = r#"{"num_experts":4,"top_k":2,"counts":[]}"#;
        assert!(parse_capture(no_trace, 4, 2)
            .unwrap_err()
            .contains("TURBOSPARK_ROUTER_TRACE"));
        for bad in ["[[1,1]]", "[[1,4]]", "[[1,2,3]]", "[[]]"] {
            let text = format!(r#"{{"num_experts":4,"top_k":2,"trace":{bad}}}"#);
            assert!(parse_capture(&text, 4, 2).is_err(), "{bad}");
        }
    }

    #[test]
    fn slots_follow_first_appearance_and_keep_rank_order() {
        let w = window_of(&[vec![9, 4], vec![4, 7]]);
        assert_eq!((w.union, w.slots), (3, vec![vec![0, 1], vec![1, 2]]));
    }

    #[test]
    fn sorting_by_expert_keeps_every_route_and_each_token_rank_pair() {
        let (pair, sorted) = route_lists(&[vec![0, 1], vec![1, 2], vec![0, 2]]);
        assert_eq!(pair[1], (0, 1, 1));
        assert!(sorted.windows(2).all(|w| w[0].2 <= w[1].2), "{sorted:?}");
        assert_ne!(pair, sorted);
        let mut back = sorted.clone();
        back.sort_unstable();
        assert_eq!(back, pair);
    }

    #[test]
    fn zipf_is_deterministic_distinct_and_skewed() {
        let a = zipf_trace(64, 4, 1.5, 512, 7);
        assert_eq!(a, zipf_trace(64, 4, 1.5, 512, 7));
        for p in &a.layers[0] {
            let mut s = p.clone();
            s.sort_unstable();
            s.dedup();
            assert_eq!(s.len(), 4);
        }
        let skewed = windows(&a, 8, 64).unwrap().union_median;
        let flat = windows(&zipf_trace(64, 4, 0.0, 512, 7), 8, 64)
            .unwrap()
            .union_median;
        assert!(skewed < flat, "{skewed} against {flat}");
    }

    #[test]
    fn break_even_is_the_last_losing_m() {
        let seq = [(1, 1.0), (2, 2.0), (4, 4.0), (8, 8.0)];
        let b = [(1, 1.5), (2, 1.9), (4, 4.1), (8, 5.0)];
        assert_eq!(wins_above(&seq, &b), Some(4.0));
        let always = [(1, 0.5), (2, 1.0), (4, 2.0), (8, 4.0)];
        assert_eq!(wins_above(&seq, &always), Some(0.0));
        let loses = [(1, 0.5), (2, 1.0), (4, 2.0), (8, 9.0)];
        assert_eq!(wins_above(&seq, &loses), None);
    }
}
