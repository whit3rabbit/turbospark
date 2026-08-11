//! Env-gated router-selection histogram (`MFERENCE_ROUTER_HIST=/path.json`).
//!
//! Counts, per layer, how often each expert appears in the router's top-k
//! across every forward pass (prefill and decode alike, matching the
//! `MFERENCE_PHASES` divisor convention), and writes one JSON file when the
//! runner drops. Diagnostic only, never on by default: it exists to answer
//! whether routing is domain-concentrated enough for a pruned or pre-warmed
//! expert set to be worth building. Per-layer counts must sum to
//! `top_k * moe_forward_passes`, which is the sanity check on any capture.
//!
//! `MFERENCE_ROUTER_TRACE=1` additionally records the top-k ids IN PASS
//! ORDER, which the counts throw away. That ordering is what ROADMAP's
//! speculative-decoding question needs: a batched verify of M consecutive
//! tokens must read the UNION of their routes, so the expert-byte cost of a
//! block is the distinct-expert count over a window of M consecutive
//! passes, and no histogram can answer that. The counts stay in the same
//! file and are recomputable from the trace, which is the capture's own
//! cross-check.

pub(crate) struct RouterHistogram {
    path: std::path::PathBuf,
    /// `counts[layer][expert]`; non-MoE layers never record and stay
    /// all-zero rows, so layer indices line up with the model's.
    counts: Vec<Vec<u64>>,
    /// `trace[layer]` is the flat concatenation of that layer's top-k ids,
    /// one group of `top_k` per forward pass, in pass order. `Some` only
    /// under `MFERENCE_ROUTER_TRACE`.
    ///
    /// ponytail: unbounded, ~1.3 KB per pass at Qwen's 40 layers x top-8,
    /// so a 4,000-pass session costs ~5 MB. Cap it if a capture ever runs
    /// long enough to matter.
    trace: Option<Vec<Vec<u32>>>,
    /// Learned from the first `record`, since `from_env` is not told it.
    top_k: usize,
}

impl RouterHistogram {
    /// `Some` only when `MFERENCE_ROUTER_HIST` names an output path and the
    /// install actually routes (dense installs have nothing to count).
    pub(crate) fn from_env(num_layers: usize, num_experts: usize) -> Option<Self> {
        let path = std::env::var_os("MFERENCE_ROUTER_HIST")?;
        if num_experts == 0 {
            return None;
        }
        let trace = std::env::var_os("MFERENCE_ROUTER_TRACE").map(|_| vec![Vec::new(); num_layers]);
        Some(Self {
            path: path.into(),
            counts: vec![vec![0; num_experts]; num_layers],
            trace,
            top_k: 0,
        })
    }

    pub(crate) fn record(&mut self, layer: usize, selected: &[usize]) {
        for &expert in selected {
            self.counts[layer][expert] += 1;
        }
        if self.top_k == 0 {
            self.top_k = selected.len();
        }
        if let Some(trace) = self.trace.as_mut() {
            trace[layer].extend(selected.iter().map(|&e| e as u32));
        }
    }

    // Hand-rolled: nested integer arrays only, and this crate deliberately
    // carries no serde.
    fn to_json(&self) -> String {
        let num_experts = self.counts.first().map_or(0, Vec::len);
        let mut out = format!(
            "{{\"num_layers\":{},\"num_experts\":{},\"top_k\":{},\"counts\":[",
            self.counts.len(),
            num_experts,
            self.top_k
        );
        push_rows(&mut out, self.counts.iter().map(|row| row.iter()));
        if let Some(trace) = self.trace.as_ref() {
            out.push_str("],\"trace\":[");
            push_rows(&mut out, trace.iter().map(|row| row.iter()));
        }
        out.push_str("]}");
        out
    }
}

/// `[[a,b],[c]]` without the trailing bracket, which the caller closes.
fn push_rows<'a, R, I, T>(out: &mut String, rows: R)
where
    R: Iterator<Item = I>,
    I: Iterator<Item = &'a T>,
    T: std::fmt::Display + 'a,
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
            out.push_str(&value.to_string());
        }
        out.push(']');
    }
}

impl Drop for RouterHistogram {
    fn drop(&mut self) {
        match std::fs::write(&self.path, self.to_json()) {
            Ok(()) => eprintln!("[router-hist] wrote {}", self.path.display()),
            Err(e) => eprintln!("[router-hist] FAILED writing {}: {e}", self.path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(trace: bool) -> RouterHistogram {
        RouterHistogram {
            path: std::path::PathBuf::new(),
            counts: vec![vec![0; 3]; 2],
            trace: trace.then(|| vec![Vec::new(); 2]),
            top_k: 0,
        }
    }

    #[test]
    fn records_and_serializes_layer_expert_counts() {
        let mut hist = fixture(false);
        hist.record(0, &[2, 0]);
        hist.record(0, &[2, 1]);
        hist.record(1, &[1, 1]);
        assert_eq!(
            hist.to_json(),
            "{\"num_layers\":2,\"num_experts\":3,\"top_k\":2,\"counts\":[[1,1,2],[0,2,0]]}"
        );
        // Drop would try to write the (empty) path; forget it so the unit
        // test touches no filesystem.
        std::mem::forget(hist);
    }

    #[test]
    fn trace_keeps_pass_order_that_the_counts_discard() {
        let mut hist = fixture(true);
        hist.record(0, &[2, 0]);
        hist.record(0, &[2, 1]);
        hist.record(1, &[1, 1]);
        // Same counts as above, plus the ordering: layer 0's two passes are
        // (2,0) then (2,1), which is 3 distinct experts over a window of 2
        // against the 4 a union-free cost model would charge.
        assert_eq!(
            hist.to_json(),
            "{\"num_layers\":2,\"num_experts\":3,\"top_k\":2,\
             \"counts\":[[1,1,2],[0,2,0]],\"trace\":[[2,0,2,1],[1,1]]}"
        );
        std::mem::forget(hist);
    }
}
