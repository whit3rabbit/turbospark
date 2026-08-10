//! The GGUF repack walk: a GGUF file in, a `.gturbo` install out.

mod manifest;
mod plan;
mod transcode;
mod types;

pub use manifest::gguf_manifest_quant;
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
    let plan = plan::classify(header, arch.family)?;
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
    let plan = plan::classify(header, arch.family)?;
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
    let resident_bytes = crate::resident_writer::build_resident_weights_bin_mixed(&resident);
    drop(resident);
    for (name, count) in &lossy {
        progress(&format!(
            "WARNING {name}: {count} F32 values lost bits narrowing to BF16 \
             (this converter did not upcast from BF16)"
        ));
    }
    progress(&format!(
        "resident region built ({} bytes)",
        resident_bytes.len()
    ));

    if plan.routed.is_empty() {
        crate::gturbo_writer::write_gturbo_install_with_resident_index(
            dir,
            &arch,
            model_id,
            &resident_bytes,
        )?;
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
    for layer in plan.routed.keys().copied() {
        let expected = plan::layer_file_bytes(header, &arch, &plan, layer, stride)?;
        let layer_path = dir
            .join("packed_experts")
            .join(format!("layer_{layer:02}.bin"));
        let on_disk = std::fs::metadata(&layer_path).map(|m| m.len()).ok();
        if on_disk == Some(expected) {
            let (blobs, _) = plan::plan_one_layer_shape(header, &arch, &plan, layer)?;
            writer.adopt_layer(&blobs)?;
            progress(&format!(
                "layer {layer} adopted ({expected} bytes already on disk)"
            ));
            continue;
        }
        let (blobs, used) = plan::plan_one_layer(header, source, &arch, &plan, layer)?;
        writer.write_layer(&blobs)?;
        progress(&format!(
            "layer {layer} written ({} experts, {used} bytes/expert)",
            blobs.experts.len()
        ));
    }
    writer.finish(&arch, model_id, &resident_bytes)?;
    progress("manifest written");
    Ok(arch)
}
