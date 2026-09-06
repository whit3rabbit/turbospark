//! Report and share types for the model probe subsystem.

use crate::hf::RepoRef;
use model_io::{ArchConfig, ModelFamily};

/// Expert-cache slot counts the CLI offers, for the working-set table.
pub const SLOT_COUNTS: [u64; 3] = [8, 16, 32];

/// What the probe concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Every gate passed. `pull` may proceed.
    Runnable,
    /// A gate failed, with the reason already phrased for a user.
    Refused(String),
}

impl Verdict {
    pub fn is_runnable(&self) -> bool {
        matches!(self, Verdict::Runnable)
    }
}

/// One ggml block type and how much of the file it accounts for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeShare {
    pub name: String,
    pub tensors: usize,
    /// `None` when this port cannot size the type, which is a louder
    /// statement than zero. See the module header.
    pub bytes: Option<u64>,
    /// Whether this type blocks the install. True for a type with kernels
    /// AND for one that is transcoded away at repack time.
    pub executable: bool,
    /// Whether it is executable because it is TRANSCODED rather than because
    /// it has kernels. Worth distinguishing in the output: printing "has
    /// kernels" beside F32 states something false about this port, and the
    /// false version is the one that invites somebody to go looking for the
    /// F32 GEMV that does not exist.
    pub transcoded: bool,
}

/// The full result of a probe.
#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub repo: RepoRef,
    pub kind: crate::SourceKind,
    /// The weights file, for a GGUF probe.
    pub file: Option<String>,
    /// Total weight bytes on the wire.
    pub download_bytes: Option<u64>,
    /// The architecture string as the file spells it: `general.architecture`
    /// for GGUF, `model_type` for safetensors.
    pub architecture: Option<String>,
    pub family: Option<ModelFamily>,
    pub arch: Option<ArchConfig>,
    /// GGUF only, sorted by descending bytes with unsized types last.
    pub types: Vec<TypeShare>,
    /// MLX affine only: the effective default width and group size.
    pub affine: Option<(u32, u32)>,
    /// One routed expert's bytes, for an MoE. `None` for a dense model.
    pub expert_stride: Option<u64>,
    /// The window the checkpoint was TRAINED at, from the same header this
    /// probe already fetched.
    ///
    /// Deliberately beside `arch` rather than inside it: it is a property of
    /// the CHECKPOINT where `ArchConfig` is per-ARCHITECTURE, and a
    /// YaRN-extended release declares a longer one than the base it was built
    /// from, so putting it there would make every such pair a baseline
    /// mismatch (AGENTS.md Gotcha 55).
    ///
    /// `None` means the file declares none, never "a window of zero".
    pub trained_context: Option<u32>,
    /// Sidecars this port can use, split by whether the repo has them.
    pub sidecars_present: Vec<String>,
    pub sidecars_missing: Vec<String>,
    /// Where a chat template was found, if anywhere.
    pub chat_template: Option<String>,
    pub verdict: Verdict,
    /// Advisory lines that do not gate: expert-cache arithmetic, quantization
    /// surprises, anything a reader should see before committing a download.
    pub warnings: Vec<String>,
}

impl ProbeReport {
    /// Record a refusal, keeping the FIRST one.
    ///
    /// The gates run cheapest-first, which is also most-fundamental-first, so
    /// the earliest refusal is the one a reader needs. Letting a later gate
    /// overwrite an earlier one is not hypothetical: probing
    /// `bartowski/Phi-3.5-mini-instruct-GGUF` refuses at the architecture
    /// (recognized, no decode flow, and the registry says what it would need)
    /// and then refuses again at the sidecars, because a GGUF repo carries no
    /// `tokenizer.json`. The second message is true and useless -- it sends
    /// the reader off to find a sidecar repo for a model that would not run
    /// with one.
    pub fn refuse(&mut self, why: String) {
        if self.verdict.is_runnable() {
            self.verdict = Verdict::Refused(why);
        }
    }

    /// The pinned-slot working set implied by [`Self::expert_stride`], as
    /// `(slots, bytes)` pairs. Empty for a dense model.
    pub fn slot_cache_bytes(&self) -> Vec<(u64, u64)> {
        let (Some(stride), Some(arch)) = (self.expert_stride, self.arch.as_ref()) else {
            return Vec::new();
        };
        SLOT_COUNTS
            .iter()
            .map(|&slots| (slots, slots * arch.num_layers.max(0) as u64 * stride))
            .collect()
    }
}

/// The sidecar files this port can make use of, in the order it looks for
/// them. Only `tokenizer.json` is required; the rest change what the
/// tokenizer knows rather than whether it loads.
pub const KNOWN_SIDECARS: [&str; 6] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
];
