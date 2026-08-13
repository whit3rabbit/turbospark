//! Real Gemma 4 / Qwen 3.6 safetensors checkpoint repack mapping: classifies the
//! `mlx-community/gemma-4-*-4bit` conversion's tensor names, orders the
//! resident LM tensors the way the Swift repacker does
//! (`RepackPlanner.swift`), passes the already-quantized u32 weights and
//! BF16 scales/biases through byte-for-byte (no re-quantization -- the MLX
//! affine int4 packing is exactly this port's packed layout when viewed as
//! little-endian bytes), carries unquantized tensors (norms, `router.scale`,
//! `layer_scalar`) as raw BF16/FP16/FP32 entries, and slices the per-layer
//! `.experts.switch_glu.` routed-expert tensors into per-expert blobs with
//! ONE page-rounded (16 KiB) expert stride for the whole model.

mod config;
mod orchestrate;
mod shards;

pub use config::{
    is_supported_affine_shape, parse_gemma4_config, parse_gemma4_quantization, Gemma4Error,
    Gemma4Quant, AFFINE_1BIT_GROUP_SIZE, AFFINE_GROUP_SIZE,
};
pub use orchestrate::{
    gemma4_manifest_quant, manifest_quant, orchestrate_gemma4_checkpoint,
    orchestrate_gemma4_checkpoint_sharded, Gemma4RepackOutput,
};
pub use shards::{
    classify_for_family, classify_gemma4, pass_through_packed, Gemma4Bucket, Gemma4Shards,
    GTURBO_PAGE_BYTES,
};

use std::path::Path;

use model_io::{ArchConfig, ModelFamily};

use crate::ranged_download::RangeSource;
use crate::safetensors_header::SafetensorsHeader;

/// Streamed install write for real (multi-GB) checkpoints: the expert
/// stride comes from shard headers alone, the resident set is read and
/// written first, then each layer's expert blobs download, hit disk, and
/// drop before the next layer starts -- peak memory is one layer's blobs,
/// not thirty. `progress` gets one call per completed stage.
pub fn write_gemma4_install_streamed(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    shards: &Gemma4Shards<'_>,
    quant: &Gemma4Quant,
    mut progress: impl FnMut(&str),
) -> Result<(), Box<dyn std::error::Error>> {
    let plan = orchestrate::classify_all(shards, arch)?;
    let expert_stride = orchestrate::expert_stride_from_headers(shards, arch, quant, &plan.routed)?;
    progress(&format!(
        "classified {} resident tensors, {} routed layers, expert stride {expert_stride}",
        plan.resident_bases.len(),
        plan.routed.len(),
    ));

    let resident = orchestrate::read_resident_entries(shards, &plan.resident_bases, quant)?;
    let resident_bytes = crate::resident_writer::build_resident_weights_bin_mixed(&resident);
    drop(resident);
    progress(&format!(
        "resident region built ({} bytes)",
        resident_bytes.len()
    ));

    if plan.routed.is_empty() {
        crate::gturbo_writer::write_gturbo_install_with_resident_index(
            dir,
            arch,
            model_id,
            &resident_bytes,
        )?;
        progress("install written (no routed experts)");
        return Ok(());
    }

    let mut writer = crate::gturbo_writer::StreamingGturboWriter::new(
        dir,
        expert_stride,
        arch.num_experts as usize,
    )?;
    writer.set_quant(manifest_quant(quant, arch.family));
    for layer in 0..arch.num_layers as usize {
        let (blobs, used) =
            orchestrate::plan_one_expert_layer(shards, arch, quant, &plan.routed, layer)?;
        writer.write_layer(&blobs)?;
        progress(&format!(
            "layer {layer} written ({} experts, {used} bytes/expert)",
            blobs.experts.len()
        ));
    }
    writer.finish(arch, model_id, &resident_bytes)?;
    progress("manifest written");
    Ok(())
}

/// Convenience: orchestrate + build the resident index + write the full
/// install directory in one call.
pub fn write_gemma4_install(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Box<dyn std::error::Error>> {
    let out = orchestrate_gemma4_checkpoint(header, source, arch, quant)?;
    let resident_bytes = crate::resident_writer::build_resident_weights_bin_mixed(&out.resident);
    if out.layers.is_empty() {
        crate::gturbo_writer::write_gturbo_install_with_resident_index(
            dir,
            arch,
            model_id,
            &resident_bytes,
        )?;
    } else {
        let mut writer = crate::gturbo_writer::StreamingGturboWriter::new(
            dir,
            out.expert_stride,
            arch.num_experts as usize,
        )?;
        writer.set_quant(manifest_quant(quant, arch.family));
        for layer in &out.layers {
            writer.write_layer(layer)?;
        }
        writer.finish(arch, model_id, &resident_bytes)?;
    }
    Ok(out)
}

/// Qwen 3.6 install write. The walk itself is family-agnostic (see
/// [`classify_for_family`] and [`manifest_quant`]), so this is
/// [`write_gemma4_install`] plus the guard that `arch.family` actually says
/// Qwen -- passing a Gemma arch here would silently classify
/// `.mlp.switch_mlp.` tensors as resident.
pub fn write_qwen36_install(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Box<dyn std::error::Error>> {
    if arch.family != ModelFamily::Qwen36 {
        return Err(Box::new(Gemma4Error::Config(format!(
            "write_qwen36_install needs arch.family = qwen36, got {}",
            arch.family.as_str()
        ))));
    }
    write_gemma4_install(dir, arch, model_id, header, source, quant)
}

/// [`write_qwen36_install`] for a real multi-GB checkpoint: the same family
/// guard in front of [`write_gemma4_install_streamed`].
pub fn write_qwen36_install_streamed(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    shards: &Gemma4Shards<'_>,
    quant: &Gemma4Quant,
    progress: impl FnMut(&str),
) -> Result<(), Box<dyn std::error::Error>> {
    if arch.family != ModelFamily::Qwen36 {
        return Err(Box::new(Gemma4Error::Config(format!(
            "write_qwen36_install_streamed needs arch.family = qwen36, got {}",
            arch.family.as_str()
        ))));
    }
    write_gemma4_install_streamed(dir, arch, model_id, shards, quant, progress)
}
