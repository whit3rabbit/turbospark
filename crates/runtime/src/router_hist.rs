//! Env-gated router-selection histogram (`MFERENCE_ROUTER_HIST=/path.json`).
//!
//! Counts, per layer, how often each expert appears in the router's top-k
//! across every forward pass (prefill and decode alike, matching the
//! `MFERENCE_PHASES` divisor convention), and writes one JSON file when the
//! runner drops. Diagnostic only, never on by default: it exists to answer
//! whether routing is domain-concentrated enough for a pruned or pre-warmed
//! expert set to be worth building. Per-layer counts must sum to
//! `top_k * moe_forward_passes`, which is the sanity check on any capture.

pub(crate) struct RouterHistogram {
    path: std::path::PathBuf,
    /// `counts[layer][expert]`; non-MoE layers never record and stay
    /// all-zero rows, so layer indices line up with the model's.
    counts: Vec<Vec<u64>>,
}

impl RouterHistogram {
    /// `Some` only when `MFERENCE_ROUTER_HIST` names an output path and the
    /// install actually routes (dense installs have nothing to count).
    pub(crate) fn from_env(num_layers: usize, num_experts: usize) -> Option<Self> {
        let path = std::env::var_os("MFERENCE_ROUTER_HIST")?;
        if num_experts == 0 {
            return None;
        }
        Some(Self {
            path: path.into(),
            counts: vec![vec![0; num_experts]; num_layers],
        })
    }

    pub(crate) fn record(&mut self, layer: usize, selected: &[usize]) {
        for &expert in selected {
            self.counts[layer][expert] += 1;
        }
    }

    // Hand-rolled: nested integer arrays only, and this crate deliberately
    // carries no serde.
    fn to_json(&self) -> String {
        let num_experts = self.counts.first().map_or(0, Vec::len);
        let mut out = format!(
            "{{\"num_layers\":{},\"num_experts\":{},\"counts\":[",
            self.counts.len(),
            num_experts
        );
        for (i, layer) in self.counts.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('[');
            for (j, count) in layer.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str(&count.to_string());
            }
            out.push(']');
        }
        out.push_str("]}");
        out
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

    #[test]
    fn records_and_serializes_layer_expert_counts() {
        let mut hist = RouterHistogram {
            path: std::path::PathBuf::new(),
            counts: vec![vec![0; 3]; 2],
        };
        hist.record(0, &[2, 0]);
        hist.record(0, &[2, 1]);
        hist.record(1, &[1, 1]);
        assert_eq!(
            hist.to_json(),
            "{\"num_layers\":2,\"num_experts\":3,\"counts\":[[1,1,2],[0,2,0]]}"
        );
        // Drop would try to write the (empty) path; forget it so the unit
        // test touches no filesystem.
        std::mem::forget(hist);
    }
}
