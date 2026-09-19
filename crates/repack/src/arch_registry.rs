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
    ("qwen35moe", ModelFamily::QwenGdnMoe),
    // The DENSE half of that same architecture. Read off
    // `ornith-ai/Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q4_K_M.gguf`, the first
    // published `qwen35` file: 427 tensors, `block_count 32`, all types
    // already executable (Q4_K / Q6_K / F32).
    //
    // **NOTE HOW CLOSE THIS IS TO THE ROW ABOVE, and that the lookup is exact
    // equality rather than a prefix match** -- the same trap the two HF rows
    // carry, one table down. `qwen35` under a `starts_with` would resolve
    // every dense file to `QwenGdnMoe`: a baseline with 256 experts and a
    // decode flow with a router in it, i.e. fluent wrong output rather than
    // an error. `the_two_qwen_gguf_architectures_do_not_collapse_into_one_family`
    // pins both directions.
    ("qwen35", ModelFamily::QwenGdnDense),
    // Promoted out of the planned table after Mixtral showed that the memory
    // ceiling needs FINE-GRAINED MoE rather than merely MoE (AGENTS.md
    // Gotcha 36). Shares the `llama` decode flow; see `ModelFamily::Qwen3Moe`
    // for the two places they differ.
    ("qwen3moe", ModelFamily::Qwen3Moe),
    // Promoted by ROADMAP M5 step 3 and FULLY supported since step 4: a
    // baseline, a name table, a metadata mapping and its own decode flow
    // (`crates/runtime/src/families/gptoss/`), with both real-model gates
    // frozen against the published 12.1 GB checkpoint.
    //
    // It was promoted BEFORE the flow existed, on the same terms `llama`
    // was, and that ordering is the reusable part: it is what lets
    // `arch_from_gguf` derive an ArchConfig from the real header and check
    // it against the baseline, which is the cheapest place to catch a wrong
    // shape. M3 did the same and it is why that bring-up needed no second
    // download; M5 confirmed all 459 real tensor names mapped before a byte
    // of expert data was fetched.
    ("gpt-oss", ModelFamily::GptOss),
    // Read off `XHToken/Spark-X2.5-4B-GGUF`'s Q4_K_M, header-only 2026-09-08
    // (docs/SPARK_PHASE0.md): 290 tensors, `block_count 36`, fused
    // `attn_qkv` plus a per-head `attn_gate`, per-tensor Q4_K/Q6_K over F32
    // norms, every block type already executable. This is the FIRST family
    // whose GGUF conventions come from an upstream llama.cpp merge (PR 27868,
    // 2026-09-06) rather than a fork, so the name mapping below mirrors code
    // that is merged, tagged and stable.
    ("spark2_5", ModelFamily::Spark25),
    // The TENTH family, and the second to run on `llama`'s decode flow
    // rather than its own. Read off `Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf`
    // (`turbospark-model probe`, header-only, 2026-09-09; `docs/QWEN3_PHASE0.md`)
    // and cross-checked against llama.cpp's own `gguf-py/gguf/constants.py`:
    // `MODEL_ARCH.QWEN3` is a distinct string from `MODEL_ARCH.QWEN3MOE`
    // (`"qwen3"` against `"qwen3moe"`), and its tensor list is `QWEN3MOE`'s
    // minus the four routed-expert rows, plus plain `FFN_GATE`/`FFN_UP`/
    // `FFN_DOWN` -- exactly `Qwen3Moe`'s attention switch plus `Llama`'s
    // dense-FFN switch, both already carried by `RealLlamaState`.
    ("qwen3", ModelFamily::Qwen3Dense),
    // Dense Qwen2/Qwen2.5 uses the shared Llama flow with QKV projection
    // biases and the Qwen2 RMS epsilon. Read off the official single-file
    // Qwen2.5-7B Q3_K_M artifact (the block type remains parse-only until a
    // Q3_K resident kernel exists).
    // `Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q3_k_m.gguf`
    ("qwen2", ModelFamily::Qwen2Dense),
    ("minimax-m2", ModelFamily::MiniMaxM2),
    // Promoted out of the planned table by the `deepseek2` MLA bring-up
    // (ROADMAP Priority 2 item 3). Read off the witness below AND
    // `mradermacher/DeepSeek-V2-Lite-Chat-GGUF`'s Q8_0 (the checkpoint this
    // port installs; the Q4_K_M of the same repo refuses at `open()` on two
    // Q5_K expert layers, since the routed-pair kernel matrix has no Q5_K
    // arm). One string covers DeepSeek V2/V3, Kimi K2.5/K2.6, GLM-4.7-Flash
    // and Mistral-Large-3 -- the roadmap's multi-model unlock. Facts:
    // `docs/DEEPSEEK2_PHASE0.md`.
    ("deepseek2", ModelFamily::Deepseek2),
];

/// HF `config.json -> model_type` -> family, for the architectures that run.
///
/// Two rows per family because the multimodal checkpoints wrap the language
/// model in `text_config`, which carries its own suffixed `model_type`. Both
/// were read off the pinned `mlx-community` checkpoints.
const SUPPORTED_HF: &[(&str, ModelFamily)] = &[
    // Recognition guards other parsers; catalog safetensors intake remains refused.
    ("minimax_m2", ModelFamily::MiniMaxM2),
    ("gemma4", ModelFamily::Gemma4),
    ("gemma4_text", ModelFamily::Gemma4),
    ("qwen3_5_moe", ModelFamily::QwenGdnMoe),
    ("qwen3_5_moe_text", ModelFamily::QwenGdnMoe),
    // ROADMAP's 1-bit entry. Read off `prism-ml/Bonsai-27B-mlx-1bit`
    // @ ef22f239c670078e1507f9769bcaa66657332b96, whose root `model_type`
    // is `qwen3_5` and whose `text_config.model_type` is `qwen3_5_text`.
    //
    // **NOTE HOW CLOSE THESE ARE TO THE TWO ROWS ABOVE, and that the
    // lookup is exact equality rather than a prefix match.** Qwen 3.6
    // reports `qwen3_5_moe`; this reports `qwen3_5`. A `starts_with` here
    // would resolve every Bonsai checkpoint to the MoE family, which is a
    // different baseline and a decode flow with a router in it -- fluent
    // wrong output rather than an error. The two ARE separate families
    // for exactly this reason (`ModelFamily::QwenGdnDense`'s doc).
    ("qwen3_5", ModelFamily::QwenGdnDense),
    ("qwen3_5_text", ModelFamily::QwenGdnDense),
    // The SEVENTH family. Read off `mlx-community/Muse-Glimmer-30B-4bit`
    // @ 3e7677d7a40d348a3daba263a2b1c0aa41910710, whose root `model_type` is
    // `muse_glimmer` and whose `text_config.model_type` is
    // `muse_glimmer_text`. Its `architectures` says
    // `MuseGlimmerForConditionalGeneration`, which is the class-name scheme
    // this table deliberately does not consult.
    ("muse_glimmer", ModelFamily::MuseGlimmer),
    ("muse_glimmer_text", ModelFamily::MuseGlimmer),
    // The EIGHTH family, and the first of the Qwen 4 line. Read off
    // `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`, whose root `model_type`
    // is `qwen4_exp` and whose `text_config.model_type` is `qwen4_exp_text`.
    // `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit` reports the same pair.
    //
    // The prefix-match hazard the two rows above record does NOT apply here:
    // `qwen4_exp` shares no prefix with either `qwen3_5` string, so this pair
    // is separated from them by the first character of the version. What it
    // does share is almost every KEY inside the config, which is why
    // `parse_qwen_family_config` serves all three and why
    // `refuse_foreign_config` matters more for this family than the distance
    // between the strings suggests.
    ("qwen4_exp", ModelFamily::Qwen4Exp),
    ("qwen4_exp_text", ModelFamily::Qwen4Exp),
    // Read off `mlx-community/Qwen2.5-7B-Instruct-4bit` @
    // c8e9187488f846965507bfc2b3957d59fd0d5a27. The root config is the
    // plain `qwen2` form used by this parser.
    ("qwen2", ModelFamily::Qwen2Dense),
    // The FIFTEENTH family. Read off
    // `mlx-community/Qwen3-VL-4B-Instruct-4bit` @
    // 2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b, whose root `model_type` is
    // `qwen3_vl`. The `_text` row follows the `qwen3_5`/`qwen3_5_text`
    // precedent for the trunk-only string the transformers config schema
    // uses; the pinned artifact carries `qwen3_vl` at the root and no
    // `model_type` inside `text_config`.
    // NOTE the prefix hazard these rows sit next to: `qwen3` (dense) and
    // `qwen3_vl` share their first six characters, and the lookup is exact
    // equality, so the two never collapse.
    ("qwen3_vl", ModelFamily::Qwen3Vl),
    ("qwen3_vl_text", ModelFamily::Qwen3Vl),
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
    config_json_resolution(root).map(|(_, family)| family)
}

/// The recognized `model_type` string and family selected from a config.
fn config_json_resolution(root: &serde_json::Value) -> Option<(&str, ModelFamily)> {
    ["model_type"]
        .iter()
        .filter_map(|k| root.get(k).and_then(|v| v.as_str()))
        .chain(
            root.get("text_config")
                .and_then(|tc| tc.get("model_type"))
                .and_then(|v| v.as_str()),
        )
        .find_map(|model_type| {
            hf_family_for_model_type(model_type).map(|family| (model_type, family))
        })
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
    match config_json_resolution(root) {
        Some((model_type, found)) if found != expected => Err(format!(
            // QUOTES THE FILE, then names the two families as families.
            //
            // This used to read "model_type says {found.as_str()}", which
            // printed a string no `config.json` contains: `as_str` is the
            // FAMILY WIRE STRING, a frozen on-disk format constant that
            // deliberately does not match anything upstream spells
            // (`ModelFamily::as_str`'s own doc -- `QwenGdnDense` writes
            // `qwen35` for a file that says `qwen3_5`, and `Qwen4Exp` writes
            // `qwen4exp` for one that says `qwen4_exp`). So the message
            // claimed to quote a key and reported something else, sending a
            // reader to grep a config for a token that is not in it.
            "model_type {} resolves to the {} family, not {}",
            model_type,
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
