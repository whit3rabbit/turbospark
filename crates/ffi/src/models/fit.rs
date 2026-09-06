//! Hardware fit recommendations and context ladders.

use catalog::Catalog;
use serde_json::json;

/// `Fit` for a probed repository, or `None` when the header yielded no shape.
///
/// **THE `mapped` TERM HERE IS THE PUBLISHED CHECKPOINT, NOT THE `.gturbo`
/// THIS PORT WOULD WRITE**, and the two genuinely differ -- `CatalogEntry`
/// carries `download_bytes` and `install_bytes` as separate fields for that
/// reason. It is close for a GGUF (`recommend::discover` relies on the same
/// approximation and says why: the expert blobs are written verbatim and only
/// the small resident core is transcoded) and looser for an MLX repository.
/// The JSON says `mappedSource: "download"` so a caller can label it rather
/// than present it as an install size.
///
/// `counted` is unaffected: it is built from the arch and the expert stride,
/// which are properties of the checkpoint rather than of the container.
pub(super) fn probe_fit(
    report: &catalog::ProbeReport,
    context: u32,
    slots: model_io::ExpertCacheSlots,
    guard: model_io::LoadGuard,
    trained: Option<u32>,
) -> Option<serde_json::Value> {
    let physical = runtime::physical_memory();
    if physical == 0 || report.arch.is_none() {
        return None;
    }
    let shape = catalog::Shape {
        install_bytes: report.download_bytes.unwrap_or(0),
        expert_stride: report.expert_stride,
        arch: report.arch.clone(),
        // Nothing is frozen for an arbitrary repository, so this is
        // arithmetic or it is nothing. A curated row's measurement is
        // substituted by `recommend::from_entry`, which is a different
        // question and a different call.
        measured_counted: None,
    };
    let fit = catalog::fit(&shape, physical, context, slots, guard);
    let ladder = catalog::context_ladder(&shape, physical, slots, guard, trained);
    Some(json!({
        "contextLadder": ladder_json(&ladder),
        "trainedContext": trained,
        "verdict": verdict_name(fit.verdict),
        "verdictSummary": fit.verdict.as_str(),
        "runs": fit.verdict.runs(),
        "countedBytes": fit.counted,
        "countedSource": counted_source(fit.counted_source),
        "mappedBytes": fit.mapped,
        "mappedSource": "download",
        "slotCacheSlots": fit.slots,
        "slotCacheBytes": fit.slot_cache_bytes,
        "kvBytes": fit.kv_bytes,
        "residentBytes": fit.resident_bytes,
        "largestContext": fit.largest_context,
        "context": context,
    }))
}

/// A context ladder as JSON.
///
/// **THE CALLER MUST NOT EXTRAPOLATE FROM ONE RUNG.** KV is not linear in the
/// window: a sliding-window layer is a ring capped at `sliding_window + 128`
/// and stops growing past it. Measured 4,096 -> 131,072 on the shipped
/// baselines, Mistral 7B grows 32x while Gemma 4 grows 9x. That is the whole
/// reason this is computed here rather than left to a multiplication.
pub(super) fn ladder_json(rungs: &[catalog::LadderRung]) -> Vec<serde_json::Value> {
    rungs
        .iter()
        .map(|r| {
            json!({
                "context": r.context,
                "kvBytes": r.kv_bytes,
                "counted": r.counted,
                "verdict": verdict_name(r.verdict),
                "runs": r.verdict.runs(),
                "pastTrained": r.past_trained,
                "isTrainedMax": r.is_trained_max,
                "isLargestFitting": r.is_largest_fitting,
            })
        })
        .collect()
}

/// The decode band and the chip it was taken on.
///
/// `measuredOnThisChip` is false when the only frozen row belongs to other
/// silicon. A host must NAME the chip in that case: tok/s does not transfer,
/// and reporting a band as this machine's answer would be a claim nothing
/// measured.
pub(super) fn throughput_json(band: Option<&catalog::ThroughputBand>) -> Option<serde_json::Value> {
    band.map(|b| {
        json!({
            "minTokensPerSecond": b.min,
            "maxTokensPerSecond": b.max,
            "chip": b.chip,
            "measuredOnThisChip": b.this_machine,
        })
    })
}

/// One spelling of the verdict enum, so `ts_recommend_json` and
/// `ts_probe_json` cannot disagree about what `"tight"` is called.
pub(super) fn verdict_name(verdict: catalog::FitVerdict) -> &'static str {
    match verdict {
        catalog::FitVerdict::Resident => "resident",
        catalog::FitVerdict::Streams => "streams",
        catalog::FitVerdict::Tight => "tight",
        catalog::FitVerdict::Refused => "refused",
        catalog::FitVerdict::Unknown => "unknown",
    }
}

/// Where a counted figure came from. `unknown` is not `0`; see
/// `catalog::CountedSource`.
pub(super) fn counted_source(source: catalog::CountedSource) -> &'static str {
    match source {
        catalog::CountedSource::Measured => "measured",
        catalog::CountedSource::Estimated => "estimated",
        catalog::CountedSource::Unknown => "unknown",
    }
}

/// The context ladder for an INSTALLED model, read off its own manifest.
///
/// Answers "what would a longer window cost me", which is a different
/// question from the one `ts_recommend_json` answers and cannot be derived
/// from it: KV is not linear in the window, so a caller multiplying a
/// 4,096 figure is 3.5x high on a sliding-window family.
///
/// Reads the same three things `ts_session_open` reads -- the manifest's
/// `ArchConfig`, its trained window, and what the install commits -- so the
/// ladder and the open cannot disagree about what fits.
pub(crate) fn context_ladder_json(
    path: &str,
    slots: model_io::ExpertCacheSlots,
    guard: model_io::LoadGuard,
) -> Result<String, String> {
    let dir = std::path::Path::new(path);
    let arch = repack::peek_manifest_arch(dir)?;
    let trained = repack::trained_context_meta::peek(dir);
    let physical = runtime::physical_memory();
    if physical == 0 {
        return Err("no physical memory probe available on this platform".to_string());
    }
    let shape = catalog::Shape {
        install_bytes: catalog::directory_bytes(dir),
        expert_stride: expert_stride_of(dir, &arch),
        arch: Some(arch),
        measured_counted: None,
    };
    let rungs = catalog::context_ladder(&shape, physical, slots, guard, trained);
    Ok(json!({
        "path": path,
        "trainedContext": trained,
        "rungs": ladder_json(&rungs),
    })
    .to_string())
}

/// One routed expert's bytes for an install on disk, or `None` when it is
/// dense.
///
/// The layout file is the authority, and a missing one means DENSE rather
/// than an error, because that is exactly what a dense install looks like.
///
/// **THE MEAN PER-LAYER STRIDE, NOT THE MAXIMUM.** `Fit` models the slot
/// cache as `expert_stride * num_layers`, so the stride it wants is whatever
/// makes that product equal the real `sum(per-layer strides)`. A mixed
/// sub-4-bit install has one layer 1.6x its siblings and the model-wide
/// maximum over-states a slot by 35% (`model-io` Gotcha 2), which would walk
/// a fitting model down the verdict ladder for a reason nothing measured.
fn expert_stride_of(dir: &std::path::Path, arch: &model_io::ArchConfig) -> Option<u64> {
    if arch.num_experts <= 0 {
        return None;
    }
    let layout = model_io::load_packed_experts_layout(
        dir,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .ok()?;
    let layers = layout.layers.len() as u64;
    if layers == 0 {
        return None;
    }
    let total: u64 = layout.layers.iter().map(|l| l.expert_stride).sum();
    Some(total / layers)
}

/// Ranks curated models by hardware fit for this machine at `context`.
///
/// **`guard` MUST match what the host will OPEN with.** This ranking and
/// `ts_session_open`'s refusal share one budget by construction, which is
/// what makes a recommendation trustworthy; a hub ranking under `relaxed`
/// while its sessions open under `strict` promises a fit the loader then
/// refuses, in the one place the user cannot see the two disagree.
pub(crate) fn recommend_json(
    context: Option<u32>,
    slots: model_io::ExpertCacheSlots,
    guard: model_io::LoadGuard,
) -> Result<String, String> {
    let physical = runtime::physical_memory();
    if physical == 0 {
        return Err(
            "no physical memory probe available on this platform; cannot recommend models"
                .to_string(),
        );
    }
    let (working_set, chip) = match runtime::recommended_max_working_set() {
        Some((bytes, name)) => (Some(bytes), name),
        None => (None, String::new()),
    };
    let machine = catalog::Machine {
        physical_bytes: physical,
        working_set_bytes: working_set,
        load_guard: guard,
        chip,
    };
    let catalog = Catalog::embedded()?;
    let entries: Vec<&catalog::CatalogEntry> = catalog.entries().collect();
    let context_val = context.unwrap_or(4096);
    let recommendations = catalog::recommend_catalog(&entries, &machine, context_val, slots);
    let rows: Vec<_> = recommendations
        .into_iter()
        .map(|r| {
            let alias = match &r.origin {
                catalog::Origin::Catalog(a) => a.clone(),
                catalog::Origin::Discovered { repo, .. } => repo.clone(),
            };
            json!({
                "alias": alias,
                "name": r.name,
                "family": r.family,
                "verdict": verdict_name(r.fit.verdict),
                "verdictSummary": r.fit.verdict.as_str(),
                "runs": r.fit.verdict.runs(),
                "countedBytes": r.fit.counted,
                // **NEVER RENDER AN ESTIMATE AS A MEASUREMENT.** A row nothing
                // has read reports `unknown` here rather than a zero, which is
                // the distinction `CountedSource` exists for and the one the
                // detail pane presented as "Zero KB" before it had this.
                "countedSource": counted_source(r.fit.counted_source),
                "installBytes": r.fit.mapped,
                "slotCacheSlots": r.fit.slots,
                "largestContext": r.fit.largest_context,
                "notes": r.notes,
                // Chip-matched, unchanged: a caller reading these two got
                // this machine's own measurement and still does.
                "toksPerSecondMin": r.measured.as_ref().map(|m| m.decode_tok_s_min),
                "toksPerSecondMax": r.measured.as_ref().map(|m| m.decode_tok_s_max),
                // May belong to other silicon; carries its chip so it can be
                // labelled rather than passed off as this machine's.
                "throughput": throughput_json(r.throughput.as_ref()),
            })
        })
        .collect();
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}
