//! The network arm: which repositories on Hugging Face are worth installing
//! here.
//!
//! **THE FILTER IS THE EXISTING PROBE, AND THAT IS THE WHOLE POINT.** Popular
//! is not the same as runnable: most of the top GGUF repositories are
//! architectures with no decode flow here, or are published only at block
//! types with no kernels. Every candidate goes through
//! [`crate::probe::probe`] unchanged, so a discovered row is refused for the
//! same reason and in the same words `turbospark-model probe` would refuse
//! it, and the accepting path is one this repo already tests.
//!
//! ## Choosing a quantization
//!
//! A popular GGUF repository publishes ten files and the probe refuses to
//! guess between them. Choosing is this module's one real decision, and it is
//! where shoehorn's solve translates into a codebase that installs published
//! artifacts: upstream computes the bits-per-weight a budget affords and
//! synthesizes a mixed-precision file to match, while this picks the largest
//! PUBLISHED file that plausibly fits.
//!
//! The pre-filter is by NAME, and it is deliberately approximate. It exists
//! to avoid reading ten headers, not to decide anything -- the header is what
//! decides, because a file named `Q3_K_M` built with an imatrix contains
//! IQ3_XXS and no Q3_K at all (ROADMAP Phase S measured exactly that). So a
//! name that survives the filter is probed and may still be refused, and a
//! name that does not survive is never probed at all, which is the only cost
//! of being wrong in that direction.
//!
//! ## What it cannot do
//!
//! A GGUF carries llama.cpp's tokenizer representation and this port loads an
//! HF `tokenizer.json`, so a discovered repository needs a SIDECAR repository
//! and nothing in the file list says which one. The card's `base_model` is
//! the answer where it has one; where it does not, the candidate is reported
//! as needing `--sidecar-repo` rather than dropped, because "this would run
//! if you told me where its tokenizer lives" and "this would not run" are
//! different answers and only one of them is a dead end.

use crate::hf::{Client, RepoRef};
use crate::probe::{probe, Verdict};

use super::rank::{rank, suspicious, Evidence};
use super::{fit, Machine, Origin, Recommendation, Shape};

/// How the scan is bounded.
#[derive(Debug, Clone)]
pub struct DiscoverOptions {
    /// How many popular repositories to consider.
    pub scan: usize,
    /// The context window to fit against.
    pub context: u32,
    /// How many repositories to probe at once. Every probe is two or three
    /// small requests and the wall clock is entirely latency, so this is
    /// worth having; it is bounded because the ceiling here is Hugging
    /// Face's patience rather than this machine's.
    pub concurrency: usize,
}

impl Default for DiscoverOptions {
    fn default() -> Self {
        Self {
            scan: 20,
            context: 4096,
            concurrency: 8,
        }
    }
}

/// Quantization suffixes worth probing, best quality first.
///
/// Every entry names a block type with kernels here (`model_io::
/// EXECUTABLE_GGUF_TYPES`). The list is ORDERED, and the order is what makes
/// "the largest that fits" a quality ladder rather than a byte count.
/// `Q4_K` before `IQ4` is deliberate: they are close in size and the K-quant
/// is the better-understood path here.
const QUANT_LADDER: [&str; 7] = [
    "Q8_0", "Q6_K", "Q5_K", "Q4_K", "IQ4_XS", "IQ4_NL", "IQ3_XXS",
];

/// Rank popular Hugging Face repositories for this machine.
///
/// Returns candidates in the same [`Recommendation`] shape the catalog arm
/// produces, so a caller can concatenate the two and rank once. Every row
/// here carries [`Evidence::Discovered`], which sorts below every curated row
/// that fits.
pub fn discover(
    client: &Client,
    machine: &Machine,
    options: &DiscoverOptions,
) -> Result<Vec<Recommendation>, String> {
    let repos = client.popular_gguf_repos(options.scan)?;
    let workers = options.concurrency.clamp(1, 16).min(repos.len().max(1));

    let mut out: Vec<Recommendation> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in repos.chunks(repos.len().div_ceil(workers).max(1)) {
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .filter_map(|repo| consider(client, machine, options, repo))
                    .collect::<Vec<_>>()
            }));
        }
        for handle in handles {
            // A panicking worker would otherwise take the whole scan down
            // with it; one unreachable repository is not a reason to lose
            // nineteen good answers.
            if let Ok(mut rows) = handle.join() {
                out.append(&mut rows);
            }
        }
    });

    rank(&mut out, super::Recommendation::key);
    Ok(out)
}

/// One repository, or `None` when it offers nothing this port could run.
fn consider(
    client: &Client,
    machine: &Machine,
    options: &DiscoverOptions,
    listed: &crate::hf::PopularRepo,
) -> Option<Recommendation> {
    let repo = RepoRef::parse(&listed.id).ok()?;
    let files = client.file_list_with_sizes(&repo).ok()?;
    let file = choose_quantization(&files, machine.physical_bytes)?;

    let sidecar = listed
        .base_model
        .as_deref()
        .and_then(|b| RepoRef::parse(b).ok());
    let report = probe(client, &repo, Some(&file.name), sidecar.as_ref()).ok()?;

    let mut notes = report.warnings.clone();
    if sidecar.is_none() {
        notes.push(
            "its card names no base model, so this port has nowhere to fetch a \
             tokenizer.json from; install it with --sidecar-repo pointing at the \
             checkpoint it was converted from."
                .to_string(),
        );
    }
    if let Verdict::Refused(why) = &report.verdict {
        notes.push(why.clone());
    }

    let install_bytes = report.download_bytes.or(file.size).unwrap_or(0);
    let shape = Shape {
        // A GGUF install is close to its source file: the expert blobs are
        // written verbatim and only the small resident core is transcoded.
        install_bytes,
        expert_stride: report.expert_stride,
        arch: report.arch.clone(),
        measured_counted: None,
    };
    let mut fitted = fit(
        &shape,
        machine.physical_bytes,
        options.context,
        model_io::ExpertCacheSlots::Auto,
    );
    // A refusal upstream of the arithmetic outranks the arithmetic: a model
    // with no decode flow does not "fit" whatever its footprint would be.
    if !report.verdict.is_runnable() {
        fitted.verdict = super::FitVerdict::Refused;
    }

    Some(Recommendation {
        origin: Origin::Discovered {
            repo: listed.id.clone(),
            file: Some(file.name.clone()),
        },
        name: format!("{} ({})", listed.id, quant_label(&file.name)),
        family: report.architecture.clone(),
        fit: fitted,
        evidence: Evidence::Discovered,
        measured: None,
        suspicious: suspicious(&listed.id, install_bytes),
        notes,
    })
}

/// The largest file this machine could hold whose name names a block type
/// with kernels, falling back to the smallest such file when none fits.
///
/// The fallback matters and is not a formality: a streamed MoE whose file is
/// larger than memory RUNS, which is what the engine is for, so refusing to
/// even look at one would drop exactly the class of model this port exists to
/// serve. It is the fit arithmetic downstream that decides, on the real
/// shape, not this.
fn choose_quantization(
    files: &[crate::hf::RepoFile],
    physical: u64,
) -> Option<crate::hf::RepoFile> {
    let mut runnable: Vec<&crate::hf::RepoFile> = files
        .iter()
        .filter(|f| f.name.ends_with(".gguf"))
        // A sharded GGUF (`-00001-of-00005.gguf`) needs a walk this port's
        // installer does not have, and picking one shard installs a fifth of
        // a model that then fails to open.
        .filter(|f| !f.name.contains("-of-0"))
        .filter(|f| ladder_rank(&f.name).is_some())
        .collect();
    if runnable.is_empty() {
        return None;
    }
    // Ties on size are broken by the ladder, so a repository publishing two
    // files of near-identical length picks the better-understood type.
    runnable.sort_by(|a, b| {
        a.size
            .cmp(&b.size)
            .then(ladder_rank(&b.name).cmp(&ladder_rank(&a.name)))
    });
    runnable
        .iter()
        .rfind(|f| f.size.is_some_and(|s| s <= physical))
        .or_else(|| runnable.first())
        .map(|f| (*f).clone())
}

/// Where a filename sits on [`QUANT_LADDER`], or `None` when it names no
/// executable type.
fn ladder_rank(name: &str) -> Option<usize> {
    let upper = name.to_ascii_uppercase();
    QUANT_LADDER
        .iter()
        .position(|q| upper.contains(&format!("-{q}")) || upper.contains(&format!(".{q}")))
}

/// The quantization a filename names, for display.
fn quant_label(name: &str) -> &'static str {
    ladder_rank(name).map(|i| QUANT_LADDER[i]).unwrap_or("?")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::RepoFile;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn file(name: &str, size: u64) -> RepoFile {
        RepoFile {
            name: name.to_string(),
            size: Some(size),
        }
    }

    /// The real file list of `Qwen/Qwen3-30B-A3B-GGUF`, read live
    /// 2026-08-18. Its sizes are the published ones.
    fn qwen3_30b() -> Vec<RepoFile> {
        vec![
            file(".gitattributes", 1823),
            file("Qwen3-30B-A3B-Q4_K_M.gguf", 18_556_685_824),
            file("Qwen3-30B-A3B-Q5_0.gguf", 21_080_509_952),
            file("Qwen3-30B-A3B-Q5_K_M.gguf", 21_725_580_800),
            file("Qwen3-30B-A3B-Q6_K.gguf", 25_092_531_712),
        ]
    }

    /// **Q5_0 has no kernels here and must never be chosen**, however well it
    /// fits. The whole point of the pre-filter is that a size ranking over
    /// every published file would pick it on a large machine.
    #[test]
    fn a_block_type_with_no_kernels_is_not_a_candidate() {
        assert!(ladder_rank("Qwen3-30B-A3B-Q5_0.gguf").is_none());
        assert!(ladder_rank("model-Q4_0.gguf").is_none());
        assert!(ladder_rank("model-Q2_K.gguf").is_none());
        // Q5_K, Q6_K and the IQ set do.
        for name in [
            "m-Q5_K_M.gguf",
            "m-Q6_K.gguf",
            "m-IQ3_XXS.gguf",
            "m.Q8_0.gguf",
        ] {
            assert!(ladder_rank(name).is_some(), "{name}");
        }
    }

    /// The largest that fits, and the fallback when none does. A 64 GB
    /// machine takes the Q6_K; a 20 GB machine cannot hold it and takes the
    /// Q4_K_M; an 8 GB machine holds none of them and still gets the
    /// smallest rather than nothing, because a streamed MoE larger than
    /// memory is the case this engine exists for.
    #[test]
    fn the_largest_fitting_quantization_wins_and_the_smallest_is_the_fallback() {
        let files = qwen3_30b();
        assert_eq!(
            choose_quantization(&files, 64 * GIB).unwrap().name,
            "Qwen3-30B-A3B-Q6_K.gguf"
        );
        assert_eq!(
            choose_quantization(&files, 20 * GIB).unwrap().name,
            "Qwen3-30B-A3B-Q4_K_M.gguf"
        );
        assert_eq!(
            choose_quantization(&files, 8 * GIB).unwrap().name,
            "Qwen3-30B-A3B-Q4_K_M.gguf"
        );
        // A repository with nothing executable in it is not a candidate.
        assert!(choose_quantization(&[file("m-Q2_K.gguf", 1)], 64 * GIB).is_none());
        assert!(choose_quantization(&[file("README.md", 1)], 64 * GIB).is_none());
    }

    /// **A SHARDED GGUF IS NOT A CANDIDATE.** Picking one shard installs a
    /// fifth of a model, and the failure lands at open with a tensor name
    /// rather than here with a reason. The suffix is what the converter
    /// writes and is the only signal in a file list.
    #[test]
    fn a_sharded_gguf_is_skipped_rather_than_partially_installed() {
        let sharded = vec![
            file("big-Q4_K_M-00001-of-00005.gguf", 9_000_000_000),
            file("big-Q4_K_M-00002-of-00005.gguf", 9_000_000_000),
        ];
        assert!(choose_quantization(&sharded, 64 * GIB).is_none());
        // And a single file of the same quantization beside them is chosen.
        let mut mixed = sharded.clone();
        mixed.push(file("small-Q4_K_M.gguf", 4_000_000_000));
        assert_eq!(
            choose_quantization(&mixed, 64 * GIB).unwrap().name,
            "small-Q4_K_M.gguf"
        );
    }

    /// Two files of the same size resolve by quality rather than by
    /// whichever the file list happened to sort first.
    #[test]
    fn a_size_tie_is_broken_by_the_quality_ladder() {
        let tied = vec![
            file("m-IQ4_XS.gguf", 4_000_000_000),
            file("m-Q4_K_M.gguf", 4_000_000_000),
        ];
        assert_eq!(
            choose_quantization(&tied, 64 * GIB).unwrap().name,
            "m-Q4_K_M.gguf"
        );
    }
}
