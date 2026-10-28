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
    /// The L1 dispatch probe and L4 stand-ins are wired in its decode loop
    /// (`crates/runtime/src/lab_hooks.rs`): dense FFN layers only, so an
    /// install that routes experts gets neither even in a wired flow.
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
            probe_and_standin: false,
            notes: &["routed experts keep the per-token loop in prefill"],
        },
        Gemma4 => FamilyFacts {
            flow: "gemma4",
            chunked_prefill: Support::Yes,
            drafters: &[],
            steel_prefill: false,
            decode_attention: "barrier",
            phase_buckets: "moe",
            probe_and_standin: false,
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
        Lfm3 => FamilyFacts {
            flow: "lfm3",
            chunked_prefill: Support::No("no chunked driver yet; the sequential path serves prefill, use --chunk off".into()),
            drafters: &[],
            steel_prefill: false,
            decode_attention: "barrier",
            phase_buckets: "final_only",
            probe_and_standin: false,
            notes: &["13 short-conv layers carry a fixed tail and no KV"],
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
    /// The family profile's `recommended` entries whose `when` holds for
    /// this install (`crates/lab/families/`), as `KEY=V` fragments with why.
    pub recommended: Vec<Value>,
    /// The family profile's sweep ids whose `when` holds for this install.
    pub sweeps: Vec<String>,
}

impl Coverage {
    pub fn has(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Fill `recommended` and `sweeps` from the family profile, when one is
    /// embedded for this family, and add the `profile` tag.
    fn attach_profile(&mut self) {
        let set = match crate::family_profile::ProfileSet::embedded() {
            Ok(set) => set,
            Err(e) => {
                self.notes.push(format!("family profiles unreadable: {e}"));
                return;
            }
        };
        let Some(profile) = set.by_family(&self.family) else {
            self.notes
                .push(format!("no family profile for {}", self.family));
            return;
        };
        self.tags.push("profile".into());
        let tags: std::collections::BTreeSet<String> = self.tags.iter().cloned().collect();
        let (rec, sweeps) = profile.applicable(&tags);
        self.recommended = rec
            .iter()
            .map(|r| {
                serde_json::json!({
                    "setting": r.as_arm_fragment(),
                    "why": r.why,
                    "findings": r.findings,
                })
            })
            .collect();
        self.sweeps = sweeps.iter().map(|s| s.id.clone()).collect();
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
    // The ANE+Steel split serves dense qwen only, and only from a build with
    // the feature: the worker refuses an `ane_prefill` arm otherwise.
    if arch.family == ModelFamily::QwenGdnDense {
        if cfg!(feature = "ane-prefill") {
            tags.push("ane".into());
        } else {
            notes.push("built without the ane-prefill feature: the ane sweeps do not apply".into());
        }
    }
    if f.probe_and_standin && arch.num_experts == 0 {
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
    let mut c = Coverage {
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
        recommended: Vec::new(),
        sweeps: Vec::new(),
    };
    c.attach_profile();
    c
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
    let mut c = Coverage {
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
        recommended: Vec::new(),
        sweeps: Vec::new(),
    };
    c.attach_profile();
    c
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
    for r in &c.recommended {
        crate::say!(
            "  recommended {}: {}",
            r["setting"].as_str().unwrap_or("?"),
            r["why"].as_str().unwrap_or("")
        );
    }
    if !c.sweeps.is_empty() {
        crate::say!(
            "  sweeps: {} (tslab sweep --sweep ID, or all)",
            c.sweeps.join(" ")
        );
    }
    for n in &c.notes {
        crate::say!("  note: {n}");
    }
    let v = serde_json::to_value(&c).map_err(|e| e.to_string())?;
    Ok(crate::schema::CmdOutput::ok(v))
}

/// `tslab family (--model M | --family F) [--findings] [--sweeps] [--dir DIR]`:
/// the family profile, filtered to the install when one is given.
pub fn family_cmd(mut args: crate::args::Args) -> Result<crate::schema::CmdOutput, String> {
    use crate::family_profile::ProfileSet;
    let model = args.take("--model")?;
    let family = args.take("--family")?;
    let dir = args.take("--dir")?;
    let want_findings = args.flag("--findings");
    let want_sweeps = args.flag("--sweeps");
    args.finish()?;
    let set = ProfileSet::resolve(dir.as_deref())?;
    let (family, tags, install) = match (&model, &family) {
        (Some(m), None) => {
            let subject = Subject::open(m)?;
            let c = coverage(&subject);
            let tags: std::collections::BTreeSet<String> = c.tags.iter().cloned().collect();
            (c.family.clone(), Some(tags), Some(c.install.clone()))
        }
        (None, Some(f)) => (f.clone(), None, None),
        (None, None) => {
            // No argument: list the profiles.
            for p in set.profiles() {
                crate::say!(
                    "{:<16} {:<5} {:>2} recommended {:>2} sweeps {:>3} findings  {}",
                    p.family,
                    p.kind,
                    p.recommended.len(),
                    p.sweeps.len(),
                    p.findings.len(),
                    p.title
                );
            }
            let v: Vec<Value> = set
                .profiles()
                .iter()
                .map(|p| {
                    serde_json::json!({"family": p.family, "kind": p.kind, "title": p.title,
                        "recommended": p.recommended.len(), "sweeps": p.sweeps.len(),
                        "findings": p.findings.len()})
                })
                .collect();
            return Ok(crate::schema::CmdOutput::ok(
                serde_json::json!({"profiles": v}),
            ));
        }
        (Some(_), Some(_)) => return Err("give --model or --family, not both".into()),
    };
    let Some(profile) = set.by_family(&family) else {
        return Err(format!(
            "no family profile for {family:?}; tslab family lists them"
        ));
    };
    let (rec, sweeps) = match &tags {
        Some(t) => profile.applicable(t),
        None => (
            profile.recommended.iter().collect(),
            profile.sweeps.iter().collect(),
        ),
    };
    crate::say!("{}: {}", profile.family, profile.title);
    if let Some(i) = &install {
        crate::say!("  install: {i}");
    }
    crate::say!(
        "  flow {}, {} finding(s), {} open question(s)",
        profile.flow,
        profile.findings.len(),
        profile.open_questions.len()
    );
    crate::say!(
        "  recommended{}:",
        if tags.is_some() {
            " for this install"
        } else {
            ""
        }
    );
    for r in &rec {
        crate::say!("    {:<28} {}", r.as_arm_fragment(), r.why);
    }
    crate::say!(
        "  sweeps{}:",
        if tags.is_some() {
            " for this install"
        } else {
            ""
        }
    );
    for s in &sweeps {
        crate::say!(
            "    {:<16} {} vs {}  [{:?}, {:?}]  {}",
            s.id,
            s.baseline,
            s.candidates.join(", "),
            s.metric,
            s.exactness,
            s.description
        );
    }
    if want_sweeps {
        for s in &sweeps {
            crate::say!(
                "{}",
                serde_json::to_string_pretty(s).map_err(|e| e.to_string())?
            );
        }
    }
    if want_findings {
        crate::say!("  findings:");
        for f in &profile.findings {
            let same = install
                .as_deref()
                .is_some_and(|i| f.install.split(',').any(|n| i.contains(n.trim())));
            crate::say!(
                "    {} [{}{}] {} on {}: {} ({})",
                f.id,
                f.verdict,
                if same { ", this install" } else { "" },
                f.metric,
                f.install,
                serde_json::to_string(&f.values).unwrap_or_default(),
                match f.provenance.kind.as_str() {
                    "ledger" => format!(
                        "ledger {}:{}",
                        f.provenance.experiment.as_deref().unwrap_or("?"),
                        f.provenance.record.as_deref().unwrap_or("?")
                    ),
                    "doc" => format!(
                        "{} # {}",
                        f.provenance.path.as_deref().unwrap_or("?"),
                        f.provenance.heading.as_deref().unwrap_or("?")
                    ),
                    _ => f.provenance.source.clone().unwrap_or_default(),
                }
            );
            crate::say!("      conditions: {}", f.conditions);
        }
    }
    for q in &profile.open_questions {
        crate::say!("  open: {q}");
    }
    for t in &profile.traps {
        crate::say!("  trap: {t}");
    }
    let findings: Vec<Value> = profile
        .findings
        .iter()
        .map(|f| {
            let mut v = serde_json::to_value(f).unwrap_or(Value::Null);
            if let (Some(i), Some(obj)) = (install.as_deref(), v.as_object_mut()) {
                let same = f.install.split(',').any(|n| i.contains(n.trim()));
                obj.insert("same_install".into(), Value::Bool(same));
            }
            v
        })
        .collect();
    Ok(crate::schema::CmdOutput::ok(serde_json::json!({
        "family": profile.family,
        "kind": profile.kind,
        "title": profile.title,
        "flow": profile.flow,
        "install": install,
        "installs": profile.installs,
        "recommended": rec,
        "sweeps": sweeps,
        "findings": findings,
        "open_questions": profile.open_questions,
        "traps": profile.traps,
        "docs": profile.docs,
    })))
}
