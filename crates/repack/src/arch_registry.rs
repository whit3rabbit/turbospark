//! The architecture registry: which `general.architecture` strings (GGUF) and
//! `model_type` strings (Hugging Face) this port recognizes, and for the ones
//! it cannot run, what they would need (ROADMAP Phase M Stage 1).
//!
//! **This table does not decide what runs. It decides what the error says.**
//! [`ArchSupport::Supported`] is the two families that have a baseline, a name
//! mapping and a decode flow; everything else is [`ArchSupport::Planned`],
//! which is recognition and nothing more. A caller that gets `Planned` must
//! still refuse the checkpoint.
//!
//! **Planned architectures deliberately get NO [`ModelFamily`] variant.**
//! `model_io::known_architecture` is an exhaustive match returning a real
//! baseline, and `arch_validation` compares that baseline against a manifest
//! field by field, so a placeholder variant would validate installs against
//! invented numbers. A variant appears when a baseline and a flow do, not
//! before. That is why this table is keyed by string.
//!
//! **Row admission rule, the same one `gguf_names.rs` states for its name
//! table: every key here was read off a REAL PUBLISHED FILE, never
//! transcribed from llama.cpp's `LLM_ARCH_NAMES`.** Each planned row carries
//! the repository it was read from in [`PlannedArch::witness`], and
//! `tests/arch_registry_network.rs` fetches those headers and asserts the
//! string still matches. A row with no witness cannot be added.
//!
//! Two things the probe settled that are easy to assume wrongly:
//!
//! 1. **Mixtral is not its own architecture.** `TheBloke/Mixtral-8x7B-
//!    Instruct-v0.1-GGUF` reports `llama`, exactly as
//!    `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` does; the MoE is expressed
//!    through `expert_count`, not through the architecture string. So one
//!    `llama` row covers dense Llama 2/3.x AND Mixtral, and the two halves
//!    need very different work (a dense FFN has no routed experts to stream).
//! 2. **A family's HF `model_type` is not its GGUF architecture and not its
//!    `ModelFamily::as_str`.** Qwen 3.6 reports `qwen3_5_moe` in
//!    `config.json`, `qwen35moe` in GGUF and `qwen36` in a manifest. All
//!    three differ, so all three tables are separate.

use model_io::ModelFamily;

/// A recognized architecture that has no decode flow here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedArch {
    /// The missing work, in one clause, so the refusal names a next step
    /// instead of just a fact.
    pub needs: &'static str,
    /// The published file this key was read off, as a fetchable URL.
    /// Consumed by `tests/arch_registry_network.rs`, which re-reads its
    /// header and asserts the string still matches. A row without one
    /// cannot be added.
    pub witness: &'static str,
}

/// What this port can do with an architecture string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchSupport {
    /// Has a baseline, a name mapping and a decode flow.
    Supported(ModelFamily),
    /// Recognized only. Still a refusal at every call site.
    Planned(PlannedArch),
}

/// GGUF `general.architecture` -> family, for the architectures that run.
const SUPPORTED_GGUF: &[(&str, ModelFamily)] = &[
    ("gemma4", ModelFamily::Gemma4),
    // Promoted out of the planned table by ROADMAP Phase M2, and PARTIALLY:
    // this string covers dense Llama 2/3.x and Mistral as well as the Mixtral
    // MoEs, and only the MoE half has a decode flow. The dense half is
    // refused at `RealForwardRunner::open`, by name, rather than here --
    // nothing in a GGUF's architecture string says which half a file is, only
    // its `expert_count` does.
    ("llama", ModelFamily::Llama),
    // llama.cpp named Qwen 3.6's converter after the 3.5 series it shares a
    // graph with; a real Qwen 3.6 GGUF says this, not "qwen36".
    ("qwen35moe", ModelFamily::Qwen36),
    // Promoted out of the planned table after Mixtral showed that the memory
    // ceiling needs FINE-GRAINED MoE rather than merely MoE (AGENTS.md
    // Gotcha 36). Shares the `llama` decode flow; see `ModelFamily::Qwen3Moe`
    // for the two places they differ.
    ("qwen3moe", ModelFamily::Qwen3Moe),
    // Promoted by ROADMAP M5 step 3, and PARTIALLY, on the same terms
    // `llama` was: it has a baseline, a name table and a metadata mapping,
    // and its DECODE FLOW is step 4. Until then `RealForwardRunner::open`
    // refuses it by name and says which four things it needs. Promoting
    // ahead of the flow is what lets `arch_from_gguf` derive an ArchConfig
    // from the real header and be checked against the baseline, which is the
    // cheapest place to catch a wrong shape -- M3 did the same and it is why
    // that bring-up needed no second download.
    ("gpt-oss", ModelFamily::GptOss),
];

/// HF `config.json -> model_type` -> family, for the architectures that run.
///
/// Two rows per family because the multimodal checkpoints wrap the language
/// model in `text_config`, which carries its own suffixed `model_type`. Both
/// were read off the pinned `mlx-community` checkpoints.
const SUPPORTED_HF: &[(&str, ModelFamily)] = &[
    ("gemma4", ModelFamily::Gemma4),
    ("gemma4_text", ModelFamily::Gemma4),
    ("qwen3_5_moe", ModelFamily::Qwen36),
    ("qwen3_5_moe_text", ModelFamily::Qwen36),
];

/// GGUF architectures this port recognizes and cannot run.
///
/// Ordered by value to THIS engine rather than by llama.cpp parity: the
/// engine's memory ceiling comes from streaming routed experts, so an MoE
/// architecture reuses the machinery that ceiling is made of while a dense
/// one makes every byte resident (AGENTS.md Gotcha 19). Hence `llama`'s
/// note naming its two halves separately.
const PLANNED_GGUF: &[(&str, PlannedArch)] = &[
    // ROADMAP M5 Phase 0 REWROTE THIS CLAUSE, and the correction is the
    // point: the original named a layer graph, and the binding obstacle is
    // arithmetic that has nothing to do with one. Scout is 16 experts of
    // 77.8 MiB over 48 layers, so its slot cache wants 58.4 GiB at the
    // default 16 slots -- Mixtral's dead end again and worse, because depth
    // multiplies it (AGENTS.md Gotcha 36). It also carries
    // `rope_freqs.weight`, which M4 made a refusal by name. Two independent
    // gates, neither of which a decode flow would fix.
    (
        "llama4",
        PlannedArch {
            needs: "an expert granularity this engine can stream (16 experts of 77.8 MiB \
                    is 58.4 GiB of slot cache at 16 slots), plus RoPE frequency scaling \
                    and its interleaved chunked-attention layer graph",
            witness: "https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf",
        },
    ),
    (
        "deepseek2",
        PlannedArch {
            needs: "multi-head latent attention kernels (layer mask 3-4, unported)",
            // Shard 1 of 12: a split GGUF puts the whole header in the first
            // shard, so this stays a header read like every other row.
            witness: "https://huggingface.co/unsloth/DeepSeek-V3-GGUF/resolve/main/DeepSeek-V3-Q6_K/DeepSeek-V3-Q6_K-00001-of-00012.gguf",
        },
    ),
    (
        "phi3",
        PlannedArch {
            needs: "dense-FFN GPU path and SuScaled (longrope) RoPE",
            witness: "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q6_K.gguf",
        },
    ),
];

/// Every planned GGUF row, for the network probe that re-reads each witness.
pub fn planned_gguf_architectures() -> impl Iterator<Item = (&'static str, PlannedArch)> {
    PLANNED_GGUF.iter().copied()
}

/// Resolve a GGUF `general.architecture` string. `None` means the string is
/// not recognized at all, which is a different message from `Planned`.
pub fn gguf_arch_support(architecture: &str) -> Option<ArchSupport> {
    if let Some((_, family)) = SUPPORTED_GGUF.iter().find(|(k, _)| *k == architecture) {
        return Some(ArchSupport::Supported(*family));
    }
    PLANNED_GGUF
        .iter()
        .find(|(k, _)| *k == architecture)
        .map(|(_, planned)| ArchSupport::Planned(*planned))
}

/// Resolve an HF `config.json -> model_type` string.
///
/// Deliberately carries only the SUPPORTED rows. Its one caller asks "is this
/// config a different family from the parser it was handed", which an unknown
/// string cannot answer either way, so a planned row here would buy nothing.
pub fn hf_family_for_model_type(model_type: &str) -> Option<ModelFamily> {
    SUPPORTED_HF
        .iter()
        .find(|(k, _)| *k == model_type)
        .map(|(_, family)| *family)
}

/// The family an HF `config.json` claims, or `None` when it claims nothing
/// this port recognizes.
///
/// Reads the root `model_type` and then `text_config.model_type`, because the
/// multimodal checkpoints carry both and disagree in suffix only (`gemma4` vs
/// `gemma4_text`). `architectures` is deliberately not consulted: it holds
/// class names (`Gemma4ForConditionalGeneration`), which is a third naming
/// scheme and would need a third table to buy nothing.
pub fn config_json_family(root: &serde_json::Value) -> Option<ModelFamily> {
    ["model_type"]
        .iter()
        .filter_map(|k| root.get(k).and_then(|v| v.as_str()))
        .chain(
            root.get("text_config")
                .and_then(|tc| tc.get("model_type"))
                .and_then(|v| v.as_str()),
        )
        .find_map(hf_family_for_model_type)
}

/// Guards a family-specific `config.json` parser against being handed
/// another family's file.
///
/// Worth having because the failure it prevents is SILENT: every parser here
/// hardcodes `family:` on the way out, so feeding Qwen's config to
/// `parse_gemma4_config` produced a Gemma-labelled `ArchConfig` built from
/// whatever keys happened to match, with no complaint. A config claiming
/// nothing recognizable is ACCEPTED -- an unknown `model_type` is not
/// evidence of the wrong family, and the trimmed fixtures in `tests/` omit
/// the key entirely.
pub fn refuse_foreign_config(
    root: &serde_json::Value,
    expected: ModelFamily,
) -> Result<(), String> {
    match config_json_family(root) {
        Some(found) if found != expected => Err(format!(
            "model_type says {}, not {}",
            found.as_str(),
            expected.as_str()
        )),
        _ => Ok(()),
    }
}

/// One sentence explaining what this port makes of an architecture string,
/// for an error message. Always names the string and, when there is a next
/// step, where it is written down.
pub fn describe_gguf_architecture(architecture: &str) -> String {
    match gguf_arch_support(architecture) {
        Some(ArchSupport::Supported(family)) => {
            format!("GGUF architecture {architecture:?} is {}", family.as_str())
        }
        Some(ArchSupport::Planned(planned)) => format!(
            "GGUF architecture {architecture:?} is recognized but has no decode flow here; \
             it needs {}. Bring-up checklist: docs/NEW_MODEL.md",
            planned.needs
        ),
        None => format!(
            "GGUF architecture {architecture:?} is not in this port's registry \
             (crates/repack/src/arch_registry.rs). Bring-up checklist: docs/NEW_MODEL.md"
        ),
    }
}
