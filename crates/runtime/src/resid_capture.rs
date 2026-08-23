//! Env-gated residual-stream capture (`MFERENCE_RESID_CAPTURE=/path.json`).
//!
//! Lifts the residual stream at the OUTPUT of every layer, at the last prompt
//! token, so a direction can be extracted from it offline
//! (`scripts/extract_direction.py`). That is ROADMAP item 9's stated
//! prerequisite -- "requires activation-capture surface and measurable domain
//! eval" -- and it is what turns a steering vector from something imported
//! into something this engine can derive from its own activations.
//!
//! Diagnostic only, never on by default.
//!
//! # It needs no new kernel
//!
//! `encode_dflash_copy_rows` is already a generic strided FP16 row copy, and
//! the DFlash2 drafter already uses it to lift `scratch.x` at exactly this
//! boundary. So the capture is one small copy per layer into a
//! `layers x hidden` buffer, and ZERO dispatches when the env var is unset.
//! It changes no math: a copy's source is untouched, so generated text is
//! byte-identical with the flag on. (`ffn_hist.rs` gets the same guarantee a
//! different way, by redirecting a kernel's destination rather than adding a
//! copy; that trick is unavailable here because nothing writes the residual
//! stream to a spare buffer, it is accumulated in place.)
//!
//! # Which pass it keeps, and why that is not "the last one"
//!
//! Exactly one snapshot per generation: the FIRST pass with `skip_head`
//! false. That is `produce(prompt[n-1])`, the last prompt token, because
//! `run_raw_completion` runs every earlier prompt token through
//! `produce_prefill` (which sets `skip_head`) and only the last one through
//! `produce`.
//!
//! Keying on that transition rather than on "the final pass of the run" is
//! what makes the capture independent of `--max-new`. The alternative --
//! overwrite on every pass and keep whatever was last -- silently captures a
//! GENERATED token's activation the moment anyone runs with a budget above 1,
//! and a corpus half-captured at the wrong positions produces a direction
//! that is a plausible vector and the wrong one.
//!
//! Re-armed by `reset()`, so a caller that opens once and walks a corpus
//! (the shape `logit_dump.rs` uses) accumulates one snapshot per prompt and
//! pays one model open rather than N.
//!
//! # Output
//!
//! A small JSON header beside a raw `.f32` sidecar, following
//! `logit_dump.rs`'s `meta.json` + `logits.f16` split rather than inlining
//! the numbers: one snapshot of a 64-layer model at hidden 5120 is 327,680
//! floats, which is ~4 MB of JSON text and a slow parse, against 1.3 MB of
//! bytes that numpy reads in one call.

use model_io::{ArchConfig, ModelFamily};

/// One generation's capture: the per-layer residual at the last prompt token.
struct Snapshot {
    position: usize,
    /// `[layers][hidden]`, flattened.
    resid: Vec<f32>,
}

/// The env-gated capture: the Metal buffer each layer copies into, plus the
/// snapshots read back from it.
pub(crate) struct ResidCapture {
    path: std::path::PathBuf,
    pub(crate) capture: gpu::MetalBuffer,
    layers: usize,
    hidden: usize,
    /// Whether this generation's snapshot is still owed. Set at construction
    /// and by [`Self::note_generation_start`], cleared by the first
    /// non-prefill pass.
    armed: bool,
    snapshots: Vec<Snapshot>,
}

impl ResidCapture {
    /// `Some` only when `MFERENCE_RESID_CAPTURE` names an output path AND the
    /// family's flow actually feeds the capture.
    ///
    /// The family guard is `ffn_hist.rs`'s and exists for its reason: a
    /// family whose flow contains no copy would write a file of ZEROS, and a
    /// zero residual reads as a real measurement -- it would extract as a
    /// direction of zero, which `turbospark_compute::steering::inv_norm`
    /// then makes inert, so the whole pipeline would run and steer nothing
    /// with no error anywhere.
    pub(crate) fn from_env(context: &gpu::MetalContext, arch: &ArchConfig) -> Option<Self> {
        let path = std::env::var_os("MFERENCE_RESID_CAPTURE")?;
        // `matches!` over both halves of the shared architecture, never
        // `== QwenGdnMoe`: the dense and MoE halves run ONE flow, and a
        // condition naming only one of them is a latent bug for exactly as
        // long as no checkpoint of the other exists (AGENTS.md Gotcha 61).
        if !matches!(
            arch.family,
            ModelFamily::QwenGdnMoe | ModelFamily::QwenGdnDense
        ) {
            eprintln!(
                "[resid-capture] MFERENCE_RESID_CAPTURE is wired for the qwen flow only; \
                 family {:?} does not feed the capture, ignoring",
                arch.family
            );
            return None;
        }
        let layers = arch.num_layers as usize;
        let hidden = arch.hidden_size as usize;
        let capture = context.new_output_buffer((layers * hidden * 2) as u64);
        Some(Self {
            path: path.into(),
            capture,
            layers,
            hidden,
            armed: true,
            snapshots: Vec::new(),
        })
    }

    /// The byte offset of `layer`'s region in the capture buffer.
    pub(crate) fn layer_offset(&self, layer: usize) -> u64 {
        (layer * self.hidden) as u64 * 2
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    /// Re-arm for a new generation. Called from `reset()`.
    pub(crate) fn note_generation_start(&mut self) {
        self.armed = true;
    }

    /// Read the capture back if this pass is the one worth keeping.
    ///
    /// Called after the token's command buffer has been waited on, so every
    /// layer's region is final.
    pub(crate) fn record_pass(&mut self, position: usize, skip_head: bool) {
        if skip_head || !self.armed {
            return;
        }
        let n = self.layers * self.hidden;
        let halfs = gpu::read_buffer_f16(&self.capture, 0, n);
        self.snapshots.push(Snapshot {
            position,
            resid: halfs.iter().map(|v| v.to_f32()).collect(),
        });
        self.armed = false;
    }

    /// `<path>` with its extension replaced by `f32`, the raw sidecar.
    fn data_path(&self) -> std::path::PathBuf {
        self.path.with_extension("f32")
    }

    fn header_json(&self) -> String {
        let data = self.data_path();
        let data_name = data
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"layers\": {},\n", self.layers));
        out.push_str(&format!("  \"hidden\": {},\n", self.hidden));
        out.push_str(&format!("  \"snapshots\": {},\n", self.snapshots.len()));
        out.push_str("  \"dtype\": \"float32\",\n");
        out.push_str("  \"layout\": \"[snapshot][layer][hidden]\",\n");
        out.push_str(
            "  \"hook\": \"residual stream at the OUTPUT of each layer (post-FFN join)\",\n",
        );
        out.push_str(&format!("  \"data\": \"{data_name}\",\n"));
        out.push_str("  \"positions\": [");
        for (i, s) in self.snapshots.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&s.position.to_string());
        }
        out.push_str("]\n}\n");
        out
    }
}

impl Drop for ResidCapture {
    fn drop(&mut self) {
        if self.snapshots.is_empty() {
            // Nothing was captured, so write NOTHING rather than an empty
            // shell. A zero-snapshot file downstream is indistinguishable
            // from a corpus entry that legitimately had no content, and the
            // extraction script would average it in.
            eprintln!(
                "[resid-capture] no non-prefill pass ran; wrote nothing to {}",
                self.path.display()
            );
            return;
        }
        let mut bytes = Vec::with_capacity(self.snapshots.len() * self.layers * self.hidden * 4);
        for s in &self.snapshots {
            for v in &s.resid {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let data = self.data_path();
        if let Err(e) = std::fs::write(&data, &bytes) {
            eprintln!("[resid-capture] FAILED writing {}: {e}", data.display());
            return;
        }
        match std::fs::write(&self.path, self.header_json()) {
            Ok(()) => eprintln!(
                "[resid-capture] wrote {} ({} snapshot(s)) and {}",
                self.path.display(),
                self.snapshots.len(),
                data.display()
            ),
            Err(e) => eprintln!(
                "[resid-capture] FAILED writing {}: {e}",
                self.path.display()
            ),
        }
    }
}
