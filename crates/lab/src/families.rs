//! What tslab can measure on an install, per family, as data.
//!
//! The runtime decides support in many places (`supports_chunked_prefill`,
//! drafter resolution, each family's scratch). This module collects the
//! answers a measurement needs into one [`Coverage`], so `tslab
//! capabilities` can say up front which benches, arms and workflow steps
//! apply instead of each one failing on its own. The family table is an
//! exhaustive `match`: a new `ModelFamily` variant does not compile until
//! someone says how the lab covers it.
//!
//! Facts here mirror the runtime as of writing; the run itself stays the
//! authority (a refused arm is recorded as a refusal, never measured under
//! another label). `tests/family_coverage.rs` drives every variant.

use std::collections::BTreeMap;

use model_io::{ArchConfig, ModelFamily};
use serde::Serialize;
use serde_json::Value;

use crate::audio_kind::{self, AudioKind};
use crate::gates::GateClass;
use crate::kernel::Subject;

/// Whether a capability applies.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "why")]
pub enum Support {
    Yes,
    /// Depends on how the install is opened; the run decides.
    Conditional(String),
    No(String),
}

impl Support {
    pub fn possible(&self) -> bool {
        !matches!(self, Support::No(_))
    }
}

/// The per-family half of coverage: facts that hold for every install of
/// the family.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FamilyFacts {
    /// The runtime flow (`crates/runtime/src/families/<flow>`).
    pub flow: &'static str,
    pub chunked_prefill: Support,
    /// Drafters the family's flow can open, if the install carries them.
    pub drafters: &'static [&'static str],
    /// Steel INT4 GEMMs in its batched prefill.
    pub steel_prefill: bool,
    /// The decode attention kernel its scratch defaults to.
    pub decode_attention: &'static str,
    /// Which `PhaseCounters` buckets its flow fills.
    pub phase_buckets: &'static str,
    /// The L1 dispatch probe and L4 stand-ins are wired in its decode path.
    pub probe_and_standin: bool,
    pub notes: &'static [&'static str],
}

pub fn facts(f: ModelFamily) -> FamilyFacts {
    use ModelFamily::*;
    let llama_flow = FamilyFacts {
        flow: "llama",
        chunked_prefill: Support::Yes,
        drafters: &[],
        steel_prefill: false,
        decode_attention: "barrier",
        phase_buckets: "moe when the install routes experts, else final_only",
        probe_and_standin: true,
        notes: &[],
    };
    match f {
        QwenGdnDense => FamilyFacts {
            flow: "qwen",
            chunked_prefill: Support::Conditional(
                "refused while an MTP head or DFlash2 drafter is open; the speculative loop owns its prefill".into(),
            ),
            drafters: &["mtp", "dflash"],
            steel_prefill: true,
            decode_attention: "gqa",
            phase_buckets: "final_only",
            probe_and_standin: true,
            notes: &["ANE+Steel prefill split behind the ane-prefill feature"],
        },
        QwenGdnMoe => FamilyFacts {
            flow: "qwen",
            chunked_prefill: Support::Conditional(
                "unless qwen_moe_chunked_refusal names a reason (hadamard, read-overlap seam, open drafter)".into(),
            ),
            drafters: &["mtp"],
            steel_prefill: true,
            decode_attention: "barrier",
            phase_buckets: "moe",
            probe_and_standin: true,
            notes: &["routed experts keep the per-token loop in prefill"],
        },
        Gemma4 => FamilyFacts {
            flow: "gemma4",
            chunked_prefill: Support::Yes,
            drafters: &[],
            steel_prefill: false,
            decode_attention: "barrier",
            phase_buckets: "moe",
            probe_and_standin: true,
            notes: &["sliding and full attention layers with different head shapes"],
        },
        Llama | Qwen3Moe | Qwen3Dense | Qwen2Dense | MiniMaxM2 | Qwen3Vl => llama_flow,
        GptOss => FamilyFacts {
            flow: "gptoss",
            chunked_prefill: Support::Yes,
            drafters: &[],
            steel_prefill: false,
            decode_attention: "barrier",
            phase_buckets: "moe",
            probe_and_standin: false,
            notes: &["learned attention sinks; MXFP4 or GGUF expert layouts the INT4 bench does not synthesize"],
        },
        MuseGlimmer => FamilyFacts {
            flow: "museglimmer",
            chunked_prefill: Support::Yes,
            drafters: &[],
            steel_prefill: false,
            decode_attention: "barrier",
            phase_buckets: "final_only",
            probe_and_standin: false,
            notes: &[],
        },
        Qwen4Exp => FamilyFacts {
            flow: "qwen4",
            chunked_prefill: Support::Yes,
            drafters: &[],
            steel_prefill: false,
            decode_attention: "barrier",
            phase_buckets: "moe",
            probe_and_standin: false,
            notes: &["hyper-connections and the n-gram PLE table"],
        },
        Spark25 => FamilyFacts {
            flow: "spark",
            chunked_prefill: Support::No("the spark flow prefills token by token; use --chunk off".into()),
            drafters: &[],
            steel_prefill: false,
            decode_attention: "barrier",
            phase_buckets: "final_only",
            probe_and_standin: false,
            notes: &["fused q|k|v projection, visible in the resident index"],
        },
        Deepseek2 => FamilyFacts {
            flow: "deepseek2",
            chunked_prefill: Support::No("deliberately sequential; use --chunk off".into()),
            drafters: &[],
            steel_prefill: false,
            decode_attention: "mla",
            phase_buckets: "moe without router or bind buckets",
            probe_and_standin: false,
            notes: &["MLA attention has no kernel bench; its projections come from the resident index"],
        },
        DeepseekV4Flash => FamilyFacts {
            flow: "none",
            chunked_prefill: Support::No("the runtime has no decode flow for this family yet".into()),
            drafters: &[],
            steel_prefill: false,
            decode_attention: "none",
            phase_buckets: "none",
            probe_and_standin: false,
            notes: &["open refuses it: only kernel benches at its shapes can run"],
        },
        Flint => FamilyFacts {
            flow: "flint",
            chunked_prefill: Support::Yes,
            drafters: &[],
            steel_prefill: false,
            decode_attention: "barrier",
            phase_buckets: "moe without router or bind buckets",
            probe_and_standin: false,
            notes: &["learned sinks and sliding windows; attention stays per token in prefill"],
        },
    }
}

/// Everything `tslab capabilities` reports for one install.
#[derive(Debug, Clone, Serialize)]
pub struct Coverage {
    pub install: String,
    /// `text` or `audio`.
    pub kind: &'static str,
    pub family: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub facts: Option<FamilyFacts>,
    /// Architecture traits, text installs only.
    pub traits: BTreeMap<&'static str, bool>,
    /// Drafters this install actually carries (read off its resident index).
    pub drafters_present: Vec<&'static str>,
    pub kl_class: Option<GateClass>,
    /// Resident weight dtypes and how many matmul tensors use each.
    pub weight_dtypes: BTreeMap<String, usize>,
    /// Per bench: the cases it would run, or why it cannot.
    pub benches: BTreeMap<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_task: Option<audio_kind::AudioTaskKind>,
    /// Predicates a workflow step's `when` can name.
    pub tags: Vec<String>,
    pub notes: Vec<String>,
}

impl Coverage {
    pub fn has(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

fn traits(arch: &ArchConfig) -> BTreeMap<&'static str, bool> {
    let has = |m: u8| arch.full_attention_layer_mask.contains(&m);
    BTreeMap::from([
        ("moe", arch.num_experts > 0),
        ("gdn", arch.has_linear_attention_layers()),
        ("mla", arch.has_mla_layers()),
        (
            "compressed_attention",
            arch.has_compressed_attention_layers(),
        ),
        ("full_attention", has(1)),
        ("sliding_attention", has(0) && arch.sliding_window > 0),
        ("sinks", crate::kernel::attn_decode::has_sinks(arch.family)),
        ("vision", arch.vision.is_active()),
    ])
}

/// Coverage for a text install.
pub fn text_coverage(subject: &Subject, arch: &ArchConfig) -> Coverage {
    let f = facts(arch.family);
    let traits = traits(arch);
    let mut notes: Vec<String> = f.notes.iter().map(|s| s.to_string()).collect();
    let drafters_present: Vec<&'static str> =
        match model_io::load_resident_index(&subject.dir.join("model_weights.bin")) {
            Ok(ix) => {
                let mut d = Vec::new();
                if f.drafters.contains(&"mtp") && runtime::install_has_mtp_head(&ix) {
                    d.push("mtp");
                }
                if f.drafters.contains(&"dflash") && runtime::install_has_dflash(&ix) {
                    d.push("dflash");
                }
                d
            }
            Err(e) => {
                notes.push(format!("resident index unreadable, drafters unknown: {e}"));
                Vec::new()
            }
        };
    let mut weight_dtypes = BTreeMap::new();
    match &subject.inventory {
        Ok(inv) => {
            // Matmuls only: a GDN conv1d (10240 x 4) is a tensor, not a GEMV.
            for s in inv.iter().filter(|s| s.cols >= 64) {
                *weight_dtypes.entry(s.dtype_name.clone()).or_insert(0) += s.count;
            }
            let unbenched: Vec<String> = weight_dtypes
                .keys()
                .filter(|d| !matches!(d.as_str(), "int4_affine" | "int8_affine"))
                .cloned()
                .collect();
            if !unbenched.is_empty() {
                notes.push(format!(
                    "no matmul bench for {}: those tensors are not timed",
                    unbenched.join(", ")
                ));
            }
        }
        Err(e) => notes.push(format!("matmul shapes fall back to ArchConfig: {e}")),
    }
    let mut tags = vec![
        "text".to_string(),
        format!("family:{}", arch.family.as_str()),
        format!("flow:{}", f.flow),
        if arch.num_experts > 0 { "moe" } else { "dense" }.to_string(),
    ];
    for (k, v) in &traits {
        if *v && !tags.iter().any(|t| t == k) {
            tags.push(k.to_string());
        }
    }
    if f.chunked_prefill.possible() {
        tags.push("chunked_prefill".into());
    }
    if f.steel_prefill {
        tags.push("steel".into());
    }
    if f.probe_and_standin {
        tags.push("probe".into());
        tags.push("standin".into());
    }
    if f.flow != "none" {
        tags.push("runs".into());
    }
    if !drafters_present.is_empty() {
        tags.push("drafter".into());
    }
    for d in &drafters_present {
        tags.push(d.to_string());
    }
    let benches = bench_coverage(subject, &mut tags);
    Coverage {
        install: subject.dir.display().to_string(),
        kind: "text",
        family: arch.family.as_str().to_string(),
        facts: Some(f),
        traits,
        drafters_present,
        kl_class: Some(GateClass::kl_class_for(arch.num_experts > 0)),
        weight_dtypes,
        benches,
        audio_task: None,
        tags,
        notes,
    }
}

/// Coverage for an audio install.
pub fn audio_coverage(subject: &Subject, cfg: &Value) -> Coverage {
    let kind = audio_kind::kind(cfg);
    let mut tags = vec!["audio".to_string(), format!("family:{}", subject.family)];
    let mut notes = Vec::new();
    match kind {
        Some(k) => {
            tags.push(format!(
                "audio:{}",
                serde_json::to_value(k.task())
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_default()
            ));
            if crate::audio_kind::runs_in_worker(k) {
                tags.push("runs".into());
            } else {
                notes.push(format!(
                    "{} has no worker path in tslab yet; kernel benches only",
                    k.as_str()
                ));
            }
            if k == AudioKind::MegaTts3 {
                notes.push(
                    "MegaTTS3 voice cloning needs consent: true in the audio job (CLAUDE.md)"
                        .into(),
                );
            }
        }
        None => notes.push(format!(
            "unrecognized audio model_type {:?}",
            audio_kind::model_type(cfg)
        )),
    }
    let benches = bench_coverage(subject, &mut tags);
    Coverage {
        install: subject.dir.display().to_string(),
        kind: "audio",
        family: subject.family.clone(),
        facts: None,
        traits: BTreeMap::new(),
        drafters_present: Vec::new(),
        kl_class: None,
        weight_dtypes: BTreeMap::new(),
        benches,
        audio_task: kind.map(|k| k.task()),
        tags,
        notes,
    }
}

fn bench_coverage(subject: &Subject, tags: &mut Vec<String>) -> BTreeMap<String, Value> {
    crate::kernel::coverage(subject)
        .into_iter()
        .map(|(name, r)| {
            let v = match r {
                Ok(n) => {
                    tags.push(format!("bench:{name}"));
                    serde_json::json!({"runs": true, "cases": n})
                }
                Err(why) => serde_json::json!({"runs": false, "why": why}),
            };
            (name, v)
        })
        .collect()
}

pub fn coverage(subject: &Subject) -> Coverage {
    match (&subject.arch, &subject.audio_config) {
        (Some(arch), _) => text_coverage(subject, arch),
        (None, Some(cfg)) => audio_coverage(subject, cfg),
        (None, None) => unreachable!("Subject::open yields a text or an audio install"),
    }
}

/// `tslab capabilities --model M`.
pub fn capabilities_cmd(mut args: crate::args::Args) -> Result<crate::schema::CmdOutput, String> {
    let model = args.required("--model")?;
    args.finish()?;
    let subject = Subject::open(&model)?;
    let c = coverage(&subject);
    crate::say!("{} ({} install, family {})", c.install, c.kind, c.family);
    if let Some(f) = &c.facts {
        crate::say!(
            "  flow {}, decode attention {}, phase buckets {}",
            f.flow,
            f.decode_attention,
            f.phase_buckets
        );
        crate::say!("  chunked prefill: {:?}", f.chunked_prefill);
    }
    if !c.drafters_present.is_empty() {
        crate::say!(
            "  drafters in this install: {}",
            c.drafters_present.join(", ")
        );
    }
    for (name, b) in &c.benches {
        match b["runs"].as_bool() {
            Some(true) => crate::say!("  bench {name}: {} case(s)", b["cases"]),
            _ => crate::say!(
                "  bench {name}: not run ({})",
                b["why"].as_str().unwrap_or("?")
            ),
        }
    }
    crate::say!("  tags: {}", c.tags.join(" "));
    for n in &c.notes {
        crate::say!("  note: {n}");
    }
    let v = serde_json::to_value(&c).map_err(|e| e.to_string())?;
    Ok(crate::schema::CmdOutput::ok(v))
}
