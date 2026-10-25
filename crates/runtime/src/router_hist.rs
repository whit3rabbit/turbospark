//! Env-gated router-selection histogram (`TURBOSPARK_ROUTER_HIST=/path.json`).
//!
//! Counts, per layer, how often each expert appears in the router's top-k
//! across every forward pass (prefill and decode alike, matching the
//! `TURBOSPARK_PHASES` divisor convention), and writes one JSON file when the
//! runner drops. Diagnostic only, never on by default: it exists to answer
//! whether routing is domain-concentrated enough for a pruned or pre-warmed
//! expert set to be worth building. Per-layer counts must sum to
//! `top_k * moe_forward_passes`, which is the sanity check on any capture.
//!
//! `TURBOSPARK_ROUTER_TRACE=1` additionally records the top-k ids IN PASS
//! ORDER, which the counts throw away. That ordering is what ROADMAP's
//! speculative-decoding question needs: a batched verify of M consecutive
//! tokens must read the UNION of their routes, so the expert-byte cost of a
//! block is the distinct-expert count over a window of M consecutive
//! passes, and no histogram can answer that. The counts stay in the same
//! file and are recomputable from the trace, which is the capture's own
//! cross-check.
//!
//! `TURBOSPARK_ROUTER_LOGITS_LAYER` plus `TURBOSPARK_ROUTER_LOGITS_PASS`
//! capture one full f32 router-logit row into that JSON. Pass is the 0-based
//! routed-forward count for that layer, so it lines up with token position in
//! a one-token-at-a-time prefill probe.

/// Filler for a pass whose guess was never made. Not an expert id, so the
/// analysis (`scripts/pilot_ceiling.py`) drops any group that carries it.
pub(crate) const PILOT_NONE: u32 = u32::MAX;

mod batches;

pub(crate) struct RouterHistogram {
    batches: Option<batches::BatchCapture>,
    path: std::path::PathBuf,
    /// `counts[layer][expert]`; non-MoE layers never record and stay
    /// all-zero rows, so layer indices line up with the model's.
    counts: Vec<Vec<u64>>,
    /// Number of routed forwards recorded for each layer.
    passes: Vec<u64>,
    /// `trace[layer]` is the flat concatenation of that layer's top-k ids,
    /// one group of `top_k` per forward pass, in pass order. `Some` only
    /// under `TURBOSPARK_ROUTER_TRACE`.
    ///
    /// ponytail: unbounded, ~1.3 KB per pass at Qwen's 40 layers x top-8,
    /// so a 4,000-pass session costs ~5 MB. Cap it if a capture ever runs
    /// long enough to matter.
    trace: Option<Vec<Vec<u32>>>,
    /// `pred[layer]` is the ONE-LAYER-AHEAD prediction of that layer's
    /// top-k: what layer `layer`'s own router says when it is run early, on
    /// the previous layer's post-attention residual instead of its own
    /// input. `Some` only under `TURBOSPARK_PILOT_PROBE`.
    ///
    /// Aligned with `trace` by construction: layer L records a prediction
    /// FOR L+1 during pass p, and layer L+1 records its actual selection
    /// later in that same pass, so entry p of `pred[L+1]` and entry p of
    /// `trace[L+1]` describe the same forward pass. Row 0 stays empty (no
    /// layer precedes it), which the analysis skips rather than misreads.
    pred: Option<Vec<Vec<u32>>>,
    /// `TURBOSPARK_PILOT_PROBE=self`: aim the probe at the layer it is already
    /// running in, instead of one ahead.
    ///
    /// The prediction then reproduces the production router exactly, so
    /// recall MUST read 100%. That is the check which separates "the
    /// predictor is weak here" from "the probe is mis-wired", and it exists
    /// because the first wiring of it read 7.7% against a 6.25% random
    /// baseline -- indistinguishable from a real negative result, and
    /// actually an off-by-one in which layer the prediction was filed
    /// against (AGENTS.md Gotcha 57: near-random means UNRELATED, so suspect
    /// the instrument before believing the finding).
    pilot_self_test: bool,
    /// One optional raw router-logit row requested by layer and pass.
    router_logits: Option<RouterLogitsCapture>,
    /// Learned from the first `record`, since `from_env` is not told it.
    top_k: usize,
}

struct RouterLogitsCapture {
    layer: usize,
    pass: usize,
    values: Option<Vec<f32>>,
}

impl RouterHistogram {
    /// `Some` only when `TURBOSPARK_ROUTER_HIST` names an output path and the
    /// install actually routes (dense installs have nothing to count).
    pub(crate) fn from_env(num_layers: usize, num_experts: usize, top_k: usize) -> Option<Self> {
        let path = std::env::var_os("TURBOSPARK_ROUTER_HIST")?;
        if num_experts == 0 {
            return None;
        }
        let trace =
            std::env::var_os("TURBOSPARK_ROUTER_TRACE").map(|_| vec![Vec::new(); num_layers]);
        let probe = std::env::var_os("TURBOSPARK_PILOT_PROBE");
        let pilot_self_test = probe.as_deref().is_some_and(|v| v == "self");
        let pred = probe.map(|_| vec![Vec::new(); num_layers]);
        let router_logits = match (
            std::env::var("TURBOSPARK_ROUTER_LOGITS_LAYER")
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
            std::env::var("TURBOSPARK_ROUTER_LOGITS_PASS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
        ) {
            (Some(layer), Some(pass)) if layer < num_layers => Some(RouterLogitsCapture {
                layer,
                pass,
                values: None,
            }),
            _ => None,
        };
        Some(Self {
            batches: (std::env::var("TURBOSPARK_ROUTER_BATCHES").as_deref() == Ok("1")).then(
                || {
                    let cap = std::env::var("TURBOSPARK_ROUTER_BATCH_LIMIT")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1_000_000);
                    batches::BatchCapture::new(cap)
                },
            ),
            path: path.into(),
            counts: vec![vec![0; num_experts]; num_layers],
            passes: vec![0; num_layers],
            trace,
            pred,
            pilot_self_test,
            router_logits,
            top_k,
        })
    }

    pub(crate) fn begin_batch(&mut self, phase: &'static str, start: usize, count: usize) {
        if let Some(capture) = &mut self.batches {
            capture.begin(phase, start, count);
        }
    }

    pub(crate) fn end_batch(&mut self, completed: bool) {
        if let Some(capture) = &mut self.batches {
            capture.end(completed);
        }
    }

    pub(crate) fn reset_request(&mut self) {
        if let Some(capture) = &mut self.batches {
            capture.reset();
        }
    }

    pub(crate) fn record_io(&mut self, layer: usize, hits: usize, misses: usize, bytes: u64) {
        if let Some(capture) = &mut self.batches {
            capture.io(layer, hits, misses, bytes);
        }
    }

    /// See [`Self::pilot_self_test`]. The offset the probe aims at: 0 in
    /// self-test (reproduce this layer's own router, expect 100% recall),
    /// 1 in the real measurement (predict the next layer).
    pub(crate) fn pilot_offset(&self) -> usize {
        usize::from(!self.pilot_self_test)
    }

    /// Is the one-layer-ahead probe on? Read by the family encoders, which
    /// must not pay for an extra router GEMV when it is off.
    pub(crate) fn pilot_enabled(&self) -> bool {
        self.pred.is_some()
    }

    /// Records the prediction FOR `layer`, made while the previous layer was
    /// running. Never called for layer 0.
    pub(crate) fn record_prediction(&mut self, layer: usize, predicted: &[usize]) {
        if let Some(pred) = self.pred.as_mut() {
            pred[layer].extend(predicted.iter().map(|&e| e as u32));
        }
    }

    /// The layer a probe running in `layer` predicts for: `layer` itself in
    /// self-test, `layer + 1` otherwise, and `None` when that is past the last
    /// layer or the probe is off.
    ///
    /// The encoder (which router to run) and the recorder (which row to file
    /// the guess under) BOTH ask this, so the two cannot disagree. Their
    /// disagreement is the off-by-one that read 7.7% against a 6.25% random
    /// baseline on Gemma (see [`Self::pilot_self_test`]).
    pub(crate) fn pilot_target(&self, layer: usize) -> Option<usize> {
        let target = layer + self.pilot_offset();
        (self.pilot_enabled() && target < self.counts.len()).then_some(target)
    }

    /// Files the guess a probe made in `layer` under the layer it is about
    /// ([`Self::pilot_target`]). The sequential-decode form: unlike
    /// [`Self::record_prediction`], the caller names the layer it is IN and
    /// cannot pick the wrong row.
    ///
    /// Pads the target's row with [`PILOT_NONE`] up to the passes it has
    /// already routed, so entry `p` of `pred` stays pass `p` of `trace` when
    /// some passes ran a path with no probe (chunked prefill). Without it the
    /// first decode guess lands in row 0 and every later pairing is shifted.
    pub(crate) fn record_pilot(&mut self, layer: usize, predicted: &[usize]) {
        let Some(target) = self.pilot_target(layer) else {
            return;
        };
        let routed = self.passes[target] as usize * self.top_k;
        if let Some(pred) = self.pred.as_mut() {
            if pred[target].len() < routed {
                pred[target].resize(routed, PILOT_NONE);
            }
            pred[target].extend(predicted.iter().map(|&e| e as u32));
        }
    }

    /// Capture only the requested row; no serialization or allocation is
    /// paid for when the opt-in layer/pass pair is absent or does not match.
    pub(crate) fn record_router_logits(&mut self, layer: usize, logits: &[f32]) {
        let Some(capture) = self.router_logits.as_mut() else {
            return;
        };
        if capture.layer == layer
            && capture.pass == self.passes[layer] as usize
            && capture.values.is_none()
        {
            capture.values = Some(logits.to_vec());
        }
    }

    pub(crate) fn record(&mut self, layer: usize, selected: &[usize]) {
        if let Some(capture) = &mut self.batches {
            capture.record(
                layer,
                self.passes[layer] as usize * selected.len(),
                selected,
            );
        }
        for &expert in selected {
            self.counts[layer][expert] += 1;
        }
        if self.top_k == 0 {
            self.top_k = selected.len();
        }
        if let Some(trace) = self.trace.as_mut() {
            trace[layer].extend(selected.iter().map(|&e| e as u32));
        }
        self.passes[layer] += 1;
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
        out.push(']');
        if let Some(trace) = self.trace.as_ref() {
            out.push_str(",\"trace\":[");
            push_rows(&mut out, trace.iter().map(|row| row.iter()));
            out.push(']');
        }
        if let Some(pred) = self.pred.as_ref() {
            out.push_str(",\"pred\":[");
            push_rows(&mut out, pred.iter().map(|row| row.iter()));
            out.push(']');
        }
        if let Some(capture) = self.router_logits.as_ref() {
            out.push_str(&format!(
                ",\"router_logits\":{{\"layer\":{},\"pass\":{},\"values\":",
                capture.layer, capture.pass
            ));
            if let Some(values) = capture.values.as_ref() {
                out.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    out.push_str(&value.to_string());
                }
                out.push(']');
            } else {
                out.push_str("null");
            }
            out.push('}');
        }
        if let Some(capture) = &self.batches {
            out.push_str(",\"batches\":");
            out.push_str(&serde_json::to_string(capture).expect("integer capture"));
        }
        out.push('}');
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
        fixture_with(trace, false)
    }

    fn fixture_with(trace: bool, pilot: bool) -> RouterHistogram {
        RouterHistogram {
            batches: None,
            path: std::path::PathBuf::new(),
            counts: vec![vec![0; 3]; 2],
            passes: vec![0; 2],
            trace: trace.then(|| vec![Vec::new(); 2]),
            pred: pilot.then(|| vec![Vec::new(); 2]),
            pilot_self_test: false,
            router_logits: None,
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

    #[test]
    fn a_prediction_is_recorded_against_the_layer_it_is_about() {
        let mut hist = fixture_with(true, true);
        // One pass: layer 0 runs, and while it runs it predicts layer 1.
        // Layer 1 then routes to something the prediction got half right.
        hist.record(0, &[2, 0]);
        hist.record_prediction(1, &[1, 2]);
        hist.record(1, &[1, 0]);
        // `pred` row 0 stays EMPTY -- nothing precedes layer 0, so no
        // prediction is ever made about it, and the analysis must not read
        // row 0 as "predicted nothing correctly".
        assert_eq!(
            hist.to_json(),
            "{\"num_layers\":2,\"num_experts\":3,\"top_k\":2,\
             \"counts\":[[1,0,1],[1,1,0]],\"trace\":[[2,0],[1,0]],\
             \"pred\":[[],[1,2]]}"
        );
        std::mem::forget(hist);
    }

    fn probe_fixture(layers: usize, self_test: bool) -> RouterHistogram {
        let mut hist = fixture_with(true, true);
        hist.counts = vec![vec![0; 8]; layers];
        hist.passes = vec![0; layers];
        hist.trace = Some(vec![Vec::new(); layers]);
        hist.pred = Some(vec![Vec::new(); layers]);
        hist.pilot_self_test = self_test;
        hist.top_k = 2;
        hist
    }

    #[test]
    fn the_probe_targets_the_next_layer_and_self_test_targets_its_own() {
        let real = probe_fixture(3, false);
        assert_eq!(real.pilot_target(0), Some(1));
        assert_eq!(real.pilot_target(1), Some(2));
        // The last layer predicts nothing.
        assert_eq!(real.pilot_target(2), None);
        let own = probe_fixture(3, true);
        assert_eq!(own.pilot_target(0), Some(0));
        assert_eq!(own.pilot_target(2), Some(2));
        let off = fixture_with(true, false);
        assert_eq!(off.pilot_target(0), None);
        std::mem::forget((real, own, off));
    }

    #[test]
    fn a_guess_is_filed_under_the_layer_it_is_about() {
        let mut real = probe_fixture(3, false);
        real.record_pilot(0, &[5, 6]);
        real.record(1, &[5, 7]);
        let pred = real.pred.as_ref().unwrap();
        assert_eq!(pred[0], Vec::<u32>::new());
        assert_eq!(pred[1], vec![5, 6]);
        assert!(pred[2].is_empty());
        // Self-test files the guess under the layer it ran in.
        let mut own = probe_fixture(3, true);
        own.record_pilot(1, &[5, 6]);
        assert_eq!(own.pred.as_ref().unwrap()[1], vec![5, 6]);
        assert!(own.pred.as_ref().unwrap()[2].is_empty());
        std::mem::forget((real, own));
    }

    #[test]
    fn passes_with_no_probe_keep_later_guesses_on_their_own_pass() {
        // Two prefill passes route layer 1 with no probe, then decode guesses.
        let mut hist = probe_fixture(2, false);
        hist.record(0, &[1, 2]);
        hist.record(1, &[1, 2]);
        hist.record(0, &[3, 4]);
        hist.record(1, &[3, 4]);
        hist.record(0, &[5, 6]);
        hist.record_pilot(0, &[5, 7]);
        hist.record(1, &[5, 6]);
        // Pass 2's guess must sit at pass 2, behind two PILOT_NONE groups.
        assert_eq!(
            hist.pred.as_ref().unwrap()[1],
            vec![PILOT_NONE, PILOT_NONE, PILOT_NONE, PILOT_NONE, 5, 7]
        );
        std::mem::forget(hist);
    }

    #[test]
    fn predictions_are_dropped_when_the_probe_is_off() {
        // The probe costs a router GEMV per layer, so the encoders ask
        // `pilot_enabled` before paying for one. If that ever returned true
        // with no storage behind it, `record_prediction` would silently
        // discard and the capture would look like a 0%-recall predictor
        // rather than like a disabled probe.
        let mut hist = fixture_with(true, false);
        assert!(!hist.pilot_enabled());
        hist.record_prediction(1, &[1, 2]);
        hist.record(0, &[2, 0]);
        assert!(!hist.to_json().contains("pred"));
        std::mem::forget(hist);
    }

    #[test]
    fn the_probe_reports_enabled_only_when_storage_exists() {
        let (on, off) = (fixture_with(true, true), fixture_with(true, false));
        assert!(on.pilot_enabled());
        assert!(!off.pilot_enabled());
        // `Drop` writes to `path`, which is empty here; forget both so the
        // unit test touches no filesystem, as its siblings above do.
        std::mem::forget(on);
        std::mem::forget(off);
    }

    #[test]
    fn router_logits_capture_targets_one_layer_and_pass() {
        let mut hist = fixture(false);
        hist.router_logits = Some(RouterLogitsCapture {
            layer: 1,
            pass: 1,
            values: None,
        });

        hist.record_router_logits(0, &[9.0, 8.0, 7.0]);
        hist.record(0, &[0, 1]);
        hist.record_router_logits(1, &[6.0, 5.0, 4.0]);
        hist.record(1, &[0, 1]);
        hist.record_router_logits(1, &[3.5, 2.5, 1.5]);

        assert!(hist
            .to_json()
            .contains("\"router_logits\":{\"layer\":1,\"pass\":1,\"values\":[3.5,2.5,1.5]}"));
        std::mem::forget(hist);
    }
}
