//! `tslab kernel`: microbenches at the install's own shapes.
//!
//! Each bench takes its shapes from the install (the `ArchConfig` in a text
//! install's manifest, the `config.json` of an audio one), so the same
//! command answers "which arm wins at which M" for any family without
//! editing constants. A bench implements [`KernelBench`] and returns a
//! [`BenchResult`]; this module ledgers it as `kernel_curve`, prints it, and
//! puts it in the `--json` envelope, so a new bench needs no reader code.
//!
//! Benches run in this process (no worker): they touch no model weights, and
//! one Metal context serves every arm. The machine lock and host guard still
//! apply, because a GPU microbench is as sensitive to a neighbour as a model
//! run is. Env knobs do not reach a bench: it calls the kernels directly.

pub mod attn_decode;
pub mod gdn_delta;
pub mod int4_matmul;
pub mod int8_matmul;
pub mod moe_routed;
pub mod music3_grid;
pub mod rows_attn;
pub mod support;
pub mod verify_ladder;
pub mod whisper_linear;

pub use crate::fit;

use std::path::PathBuf;

use serde_json::{json, Value};

use crate::args::Args;
use crate::bench::{print_result, BenchResult};
use crate::host::{eligibility, HostMonitor};
use crate::ledger::Ledger;
use crate::lock::MachineLock;
use crate::paths::{experiment_dir, lab_dir};

pub(crate) fn parse_list<T: std::str::FromStr>(s: &str, what: &str) -> Result<Vec<T>, String> {
    s.split(',')
        .map(|v| v.trim().parse().map_err(|_| format!("{what} {v:?}")))
        .collect()
}

/// What a bench measures against: the install it was pointed at.
pub struct Subject {
    pub dir: PathBuf,
    /// `ModelFamily::as_str` for a text install, `config.json` `model_type`
    /// for an audio one.
    pub family: String,
    pub arch: Option<model_io::ArchConfig>,
    /// An audio install's parsed `config.json`.
    pub audio_config: Option<Value>,
    /// The resident index's matmul shapes, or why it could not be read.
    pub inventory: Result<Vec<crate::inventory::Shape>, String>,
}

/// One matmul shape a bench should time: the install's own tensors that
/// share it (`roles`, `count`), or the config-derived fallback.
#[derive(Debug, Clone, PartialEq)]
pub struct MatShape {
    pub rows: usize,
    pub cols: usize,
    pub roles: Vec<String>,
    /// Tensors of this shape one decode token runs (0 when unknown).
    pub count: usize,
}

impl MatShape {
    pub fn case(&self, source: &str) -> Case {
        Case {
            label: format!("{}x{}", self.rows, self.cols),
            params: json!({"rows": self.rows, "cols": self.cols, "roles": self.roles,
                           "count": self.count, "shape_source": source}),
        }
    }
}

/// The install's matmul shapes of one resident dtype, from the resident
/// index when it reads, else from `fallback`. Embedding tables are lookups,
/// not matmuls, unless the head is tied to them. Widths the resident kernels
/// cannot take (not a multiple of 64) are dropped.
pub fn matmul_shapes(
    subject: &Subject,
    dtype: u8,
    fallback: impl FnOnce() -> Vec<(usize, usize)>,
) -> (Vec<MatShape>, &'static str) {
    let tied = subject.arch.as_ref().is_some_and(|a| a.tie_word_embeddings);
    let (mut out, source) = match &subject.inventory {
        Ok(inv) => (
            inv.iter()
                .filter(|s| s.dtype == dtype)
                .filter_map(|s| {
                    let roles: Vec<String> = s
                        .roles
                        .iter()
                        .filter(|r| tied || !r.contains("embed_tokens"))
                        .cloned()
                        .collect();
                    // A shape shared by the embedding and the head keeps the
                    // head: one GEMV per token, not two.
                    let embeds = s.roles.len() - roles.len();
                    (!roles.is_empty()).then(|| MatShape {
                        rows: s.rows,
                        cols: s.cols,
                        count: s.count.saturating_sub(embeds).max(1),
                        roles,
                    })
                })
                .collect::<Vec<_>>(),
            "resident_index",
        ),
        Err(_) => (
            fallback()
                .into_iter()
                .map(|(rows, cols)| MatShape {
                    rows,
                    cols,
                    roles: Vec::new(),
                    count: 0,
                })
                .collect(),
            "arch_config",
        ),
    };
    out.retain(|s| s.rows > 0 && s.cols > 0 && s.cols % 64 == 0);
    out.sort_by_key(|s| (s.rows, s.cols));
    out.dedup_by_key(|s| (s.rows, s.cols));
    (out, source)
}

impl Subject {
    pub fn open(model: &str) -> Result<Self, String> {
        let dir = catalog::resolve_model_arg(model);
        if dir.join("manifest.json").is_file() {
            let arch = repack::peek_manifest_arch(&dir)?;
            return Ok(Self {
                family: arch.family.as_str().to_string(),
                arch: Some(arch),
                audio_config: None,
                inventory: crate::inventory::load(&dir),
                dir,
            });
        }
        let config = dir.join("config.json");
        let text = std::fs::read_to_string(&config).map_err(|e| {
            format!(
                "{}: neither a text install (manifest.json) nor an audio one (config.json): {e}",
                dir.display()
            )
        })?;
        let cfg: Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", config.display()))?;
        Ok(Self {
            family: crate::audio_kind::model_type(&cfg).unwrap_or_else(|| "unknown_audio".into()),
            arch: None,
            audio_config: Some(cfg),
            inventory: Err("an audio install has no resident index".into()),
            dir,
        })
    }

    pub fn arch(&self) -> Result<&model_io::ArchConfig, String> {
        self.arch
            .as_ref()
            .ok_or_else(|| format!("{} is not a text install", self.family))
    }
}

/// One case a bench runs: a label unique within the bench for this install,
/// and the bench's own parameters for it.
#[derive(Debug, Clone)]
pub struct Case {
    pub label: String,
    pub params: Value,
}

#[derive(Debug, Clone)]
pub struct BenchOpts {
    /// The M (row) sweep for row-shaped benches.
    pub rows: Vec<usize>,
    /// The KV-length sweep for attention benches.
    pub contexts: Vec<u32>,
    /// Measured repetitions after the discarded first one.
    pub reps: usize,
    /// Record the FNV-1a of each point's output bits.
    pub hash: bool,
    /// `verify_ladder`: an experiment whose speculative runs supply measured
    /// verify ms per round.
    pub verify_from: Option<String>,
}

impl Default for BenchOpts {
    /// The command line's defaults.
    fn default() -> Self {
        Self {
            rows: vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            contexts: vec![512, 2048, 8192, 16384, 32768],
            reps: 5,
            hash: false,
            verify_from: None,
        }
    }
}

pub trait KernelBench: Sync {
    fn name(&self) -> &'static str;
    /// One line for `tslab kernel --list`.
    fn describe(&self) -> &'static str;
    /// The cases this bench runs on `subject`, or why it cannot run there.
    /// An `Err` is a coverage answer ("no MLA kernel"), not a failure.
    fn cases(&self, subject: &Subject, opts: &BenchOpts) -> Result<Vec<Case>, String>;
    fn run(
        &self,
        context: &mut gpu::MetalContext,
        subject: &Subject,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String>;
}

/// Every bench, in the order `--bench all` runs them.
pub fn registry() -> Vec<&'static dyn KernelBench> {
    vec![
        &int4_matmul::Int4Matmul,
        &int8_matmul::Int8Matmul,
        &attn_decode::AttnDecode,
        &gdn_delta::GdnDelta,
        &rows_attn::RowsAttn,
        &moe_routed::MoeRouted,
        &verify_ladder::VerifyLadder,
        &music3_grid::Music3Grid,
        &whisper_linear::WhisperLinear,
    ]
}

/// The arm labels a registered bench emits, for the benches whose arms are a
/// fixed list. `knobs::LANES` names arms by label, and `tests/knob_registry.rs`
/// holds those names to this list so a renamed arm cannot leave a lane
/// pointing at nothing.
pub fn arm_labels(bench: &str) -> Option<Vec<String>> {
    match bench {
        "int4_matmul" => Some(int4_matmul::ARMS.iter().map(|a| a.label()).collect()),
        "int8_matmul" => Some(int8_matmul::ARMS.iter().map(|a| a.to_string()).collect()),
        "attn_decode" => Some(attn_decode::ARMS.iter().map(|a| a.to_string()).collect()),
        "gdn_delta" => Some(gdn_delta::ARMS.iter().map(|a| a.to_string()).collect()),
        _ => None,
    }
}

pub fn by_name(name: &str) -> Option<&'static dyn KernelBench> {
    registry().into_iter().find(|b| b.name() == name)
}

/// Which benches apply to `subject`, and why the others do not.
pub fn coverage(subject: &Subject) -> Vec<(String, Result<usize, String>)> {
    registry()
        .into_iter()
        .map(|b| {
            (
                b.name().to_string(),
                b.cases(subject, &BenchOpts::default()).map(|c| c.len()),
            )
        })
        .collect()
}

pub fn kernel_cmd(mut args: Args) -> Result<(), String> {
    if args.flag("--list") {
        args.finish()?;
        for b in registry() {
            crate::say!("{:<16} {}", b.name(), b.describe());
        }
        return Ok(());
    }
    let model = args.required("--model")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "kernel".into());
    let bench = args.take("--bench")?.unwrap_or_else(|| "all".into());
    let rows: Vec<usize> = parse_list(
        &args
            .take("--rows")?
            .unwrap_or_else(|| "1,2,4,8,16,32,64,128,256,512".into()),
        "rows",
    )?;
    let contexts: Vec<u32> = parse_list(
        &args
            .take("--contexts")?
            .unwrap_or_else(|| "512,2048,8192,16384,32768".into()),
        "context",
    )?;
    let reps: usize = args.parsed("--reps", 5)?;
    let hash = args.flag("--hash");
    let verify_from = args.take("--verify-from")?;
    // `--shape RxC` predates `--case`; both select one case label.
    let only_case = match (args.take("--case")?, args.take("--shape")?) {
        (Some(c), _) | (None, Some(c)) => Some(c),
        (None, None) => None,
    };
    let rules = crate::cli::host_rules_from(&mut args)?;
    args.finish()?;
    if rows.contains(&0) || reps == 0 {
        return Err("rows and reps must be positive".into());
    }
    let benches: Vec<&dyn KernelBench> = if bench == "all" {
        registry()
    } else {
        bench
            .split(',')
            .map(|n| {
                by_name(n).ok_or_else(|| {
                    let names: Vec<_> = registry().iter().map(|b| b.name()).collect();
                    format!("--bench {n:?}: one of {} or all", names.join(", "))
                })
            })
            .collect::<Result<_, _>>()?
    };
    let opts = BenchOpts {
        rows,
        contexts,
        reps,
        hash,
        verify_from,
    };

    let subject = Subject::open(&model)?;
    let lab = lab_dir();
    let _lock = MachineLock::acquire(&lab, &format!("tslab kernel {experiment}"))?;
    let ledger = Ledger::open(&experiment_dir(&lab, &experiment)?)?;
    let mut context = gpu::MetalContext::new().map_err(|e| e.to_string())?;
    let provenance = crate::provenance::collect();
    let mut monitor = HostMonitor::new();
    eprintln!(
        "tslab kernel: {} ({}), ledger {}",
        subject.dir.display(),
        subject.family,
        ledger.path().display()
    );

    let record = |result: &BenchResult, samples: Vec<crate::host::HostSample>| {
        let (eligible, reasons) = eligibility(&samples, &rules);
        if !eligible {
            crate::say!("  DIRTY: {}", reasons.join("; "));
        }
        ledger.append(
            "kernel_curve",
            json!({"model": subject.dir.display().to_string(), "family": subject.family,
                   "bench": result.bench, "case": result.case, "hashed": opts.hash,
                   "eligible": eligible, "reasons": reasons, "result": result,
                   "exactness_hints": crate::bench::exactness_hints(result),
                   "provenance": provenance}),
        )
    };

    let mut ran = 0;
    for b in benches {
        let cases = match b.cases(&subject, &opts) {
            Ok(c) => c,
            Err(why) => {
                crate::say!("{}: not run on {}: {why}", b.name(), subject.family);
                continue;
            }
        };
        for case in cases {
            if only_case.as_deref().is_some_and(|c| c != case.label) {
                continue;
            }
            let before = monitor.sample(None);
            let result = b.run(&mut context, &subject, &case, &opts)?;
            let after = monitor.sample(None);
            print_result(&result);
            record(&result, vec![before, after])?;
            ran += 1;
        }
    }
    if ran == 0 {
        return Err(format!(
            "no bench case ran on {} (see the reasons above, or check --case)",
            subject.family
        ));
    }
    Ok(())
}
