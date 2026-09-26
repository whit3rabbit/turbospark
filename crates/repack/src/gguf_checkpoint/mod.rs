//! The GGUF repack walk: a GGUF file in, a `.gturbo` install out.

mod conventions;
mod manifest;
mod ngram;
mod plan;
mod sizing;
mod transcode;
pub use sizing::{minimax_gguf_sizing, MiniMaxSizing};
mod types;

pub use manifest::gguf_manifest_quant;
pub use transcode::qwen4exp_tensor_is_transcoded;
pub use types::{dtype_tag_for_ggml_type, GgufRepackError, GgufRepackOutput, FUSED_GATE_FIRST};

use std::path::Path;

use model_io::{ArchConfig, ModelFamily};

use crate::gguf_config::arch_from_gguf;
use crate::gguf_header::GgufHeader;
use crate::ranged_download::RangeSource;

/// Plan a whole GGUF repack in memory. Fine for a fixture; use
/// [`write_gguf_install_streamed`] for a real multi-GB checkpoint.
pub fn orchestrate_gguf_checkpoint(
    header: &GgufHeader,
    source: &dyn RangeSource,
) -> Result<GgufRepackOutput, GgufRepackError> {
    let arch = arch_from_gguf(header)?;
    if arch.family == ModelFamily::DeepseekV4Flash {
        return Err(GgufRepackError::UnsupportedFamily {
            family: arch.family.as_str(),
        });
    }
    let plan = plan::classify(header, arch.family, arch.num_layers as usize)?;
    let stride = plan::expert_stride(header, &arch, &plan)?;
    let (resident, lossy_narrowing) =
        transcode::resident_entries(header, source, &arch, &plan.resident)?;

    let mut layers = Vec::with_capacity(plan.routed.len());
    for layer in plan.routed.keys().copied() {
        layers.push(plan::plan_one_layer(header, source, &arch, &plan, layer)?.0);
    }

    Ok(GgufRepackOutput {
        arch,
        resident,
        layers,
        expert_stride: stride,
        ignored: plan.ignored,
        lossy_narrowing,
    })
}

/// Streamed install write: the stride comes from the header alone, the
/// resident set is read and written first, then one layer at a time, so a
/// 27 GB checkpoint never has to be materialized locally.
pub fn write_gguf_install_streamed(
    dir: &Path,
    header: &GgufHeader,
    source: &dyn RangeSource,
    model_id: &str,
    mut progress: impl FnMut(&str),
) -> Result<ArchConfig, GgufRepackError> {
    let arch = arch_from_gguf(header)?;
    if arch.family == ModelFamily::DeepseekV4Flash {
        return Err(GgufRepackError::UnsupportedFamily {
            family: arch.family.as_str(),
        });
    }
    let plan = plan::classify(header, arch.family, arch.num_layers as usize)?;
    let stride = plan::expert_stride(header, &arch, &plan)?;
    progress(&format!(
        "classified {} resident tensors, {} routed layers, expert stride {stride}",
        plan.resident.len(),
        plan.routed.len()
    ));
    for note in &plan.ignored {
        progress(&format!("ignored {note}"));
    }

    let (resident, lossy) = transcode::resident_entries(header, source, &arch, &plan.resident)?;
    for (name, count) in &lossy {
        progress(&format!(
            "WARNING {name}: {count} values lost precision converting to BF16"
        ));
    }
    // `resident` is kept alive (not built into a `Vec<u8>` here) so
    // `finish_streaming` below can write its specs' bytes straight to
    // `model_weights.bin` rather than through a second, then a third, full
    // copy of the resident region -- see `resident_writer`'s module doc.
    progress(&format!(
        "{} resident tensors ready to stream to disk",
        resident.len()
    ));

    if let Some(tensor) = plan.ngram {
        ngram::write_gguf_ngram_table(dir, header, source, &arch, tensor, &mut progress)?;
    } else if arch.family == ModelFamily::Qwen4Exp && arch.ple.ngram_size > 0 {
        return Err(GgufRepackError::MissingTensor {
            name: "per_layer_token_embd.weight".to_string(),
        });
    }

    if plan.routed.is_empty() {
        // A DENSE INSTALL STILL NEEDS ITS QUANT BLOCK (ROADMAP M4), which is
        // why this goes through the streaming writer at zero layers rather
        // than through `write_gturbo_install_with_resident_index` -- that one
        // has no way to carry one, and writes `"quant": null`.
        //
        // The two produce an identical `layout.json` (stride 0, no layers),
        // so nothing else changes. What changes is that a dense checkpoint
        // whose (num_layers, hidden_size) happens to match a shipped baseline
        // is held to `is_production_arch`'s rule and needs the block to load
        // at all: Mistral 7B is 32 layers of 4096, which is Mixtral 8x7B's,
        // and it failed at `load_manifest` with `manifest.quant is required`.
        let mut writer = crate::gturbo_writer::StreamingGturboWriter::new(dir, 0, 0)?;
        writer.set_quant(manifest::gguf_manifest_quant(header, &plan));
        writer.finish_streaming(&arch, model_id, |w| {
            crate::resident_writer::write_resident_weights_bin_mixed(&resident, w)
        })?;
        progress("install written (no routed experts)");
        return Ok(arch);
    }

    let mut writer =
        crate::gturbo_writer::StreamingGturboWriter::new(dir, stride, arch.num_experts as usize)?;
    writer.set_quant(manifest::gguf_manifest_quant(header, &plan));
    // RESUME (ROADMAP Phase M2). A 26 GB walk that restarts from layer 0
    // after any transport failure is a real fragility rather than a
    // theoretical one: it cost three 35-minute runs on one checkpoint. A
    // layer file already on disk AT THE SIZE THIS WALK WOULD WRITE is
    // adopted, which needs no network read -- the layout entry is a function
    // of the header, not of the bytes.
    //
    // Deliberately keyed on the expected size and not on mere existence: a
    // truncated leftover, or one from a walk with a different stride, is
    // refused by `adopt_layer` rather than silently believed.
    // THE BLOB FILES ARE NUMBERED BY POSITION, NOT BY CHECKPOINT LAYER.
    // `packed_experts/layout.json`'s entries must appear in order 0..N with
    // no gaps (`PackedExpertsLayout` indexes `layers[layer]` BY POSITION),
    // so the i-th routed layer writes `layer_{i:02}.bin` and carries layout
    // entry `layer: i`. Every family before `deepseek2` routed ALL of its
    // layers, so position and checkpoint index coincided and the
    // distinction was invisible; a dense lead breaks it, because
    // `deepseek2`'s routed layers are `lead..num_layers` and its blob
    // position 0 IS checkpoint layer 1. The runtime maps back with the
    // same `num_dense_leading_layers` the manifest carries.
    let lead = arch.num_dense_leading_layers as usize;
    for (seq, layer) in plan.routed.keys().copied().enumerate() {
        if layer != seq + lead {
            return Err(GgufRepackError::ShapeMismatch {
                tensor: format!("blk.{layer}"),
                detail: format!(
                    "routed layer {layer} is not blob position {seq} (dense lead {lead}): \
                     a layer graph with routed tensors outside [lead, num_layers) needs a \
                     walk that maps positions explicitly"
                ),
            });
        }
        let expected = plan::layer_file_bytes(header, &arch, &plan, layer, stride)?;
        let layer_path = dir
            .join("packed_experts")
            .join(format!("layer_{seq:02}.bin"));
        let on_disk = std::fs::metadata(&layer_path).map(|m| m.len()).ok();
        if on_disk == Some(expected) {
            let (mut blobs, _) = plan::plan_one_layer_shape(header, &arch, &plan, layer)?;
            blobs.layer = seq;
            writer.adopt_layer(&blobs)?;
            progress(&format!(
                "layer {layer} adopted ({expected} bytes already on disk)"
            ));
            continue;
        }
        let (mut blobs, used) = plan::plan_one_layer(header, source, &arch, &plan, layer)?;
        blobs.layer = seq;
        writer.write_layer(&blobs)?;
        progress(&format!(
            "layer {layer} written ({} experts, {used} bytes/expert)",
            blobs.experts.len()
        ));
    }
    writer.finish_streaming(&arch, model_id, |w| {
        crate::resident_writer::write_resident_weights_bin_mixed(&resident, w)
    })?;
    progress("manifest written");
    Ok(arch)
}
