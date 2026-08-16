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

mod classify;
mod config;
mod expert_blobs;
mod manifest_quant;
mod narrow;
mod orchestrate;
mod shards;

pub use classify::{classify_for_family, classify_gemma4, Gemma4Bucket};
pub use config::{
    is_supported_affine_shape, parse_gemma4_config, parse_gemma4_quantization, Gemma4Error,
    Gemma4Quant, AFFINE_1BIT_GROUP_SIZE, AFFINE_2BIT_GROUP_SIZE, AFFINE_GROUP_SIZE,
};
pub use expert_blobs::{expert_stride_from_headers, plan_one_expert_layer};
pub use manifest_quant::{gemma4_manifest_quant, manifest_quant, manifest_quant_for};
pub use narrow::{narrow_raw_to_bf16, pass_through_packed, NarrowedRaw};
pub use orchestrate::{
    orchestrate_gemma4_checkpoint, orchestrate_gemma4_checkpoint_sharded, Gemma4RepackOutput,
};
pub use shards::{Gemma4Shards, GTURBO_PAGE_BYTES};

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
    let expert_stride = expert_stride_from_headers(shards, arch, quant, &plan.routed)?;
    progress(&format!(
        "classified {} resident tensors, {} routed layers, expert stride {expert_stride}",
        plan.resident_bases.len(),
        plan.routed.len(),
    ));

    let resident = orchestrate::read_resident_entries(shards, &plan.resident_bases, quant)?;
    let resident_bytes =
        crate::resident_writer::build_resident_weights_bin_mixed(&resident.entries);
    // Reported rather than merely counted, on the streamed path especially:
    // this is the only place a 25-minute walk says out loud that it narrowed
    // an F16 checkpoint's norms (`narrow_raw_to_bf16`), and a silent lossy
    // step is how a quality question turns into a mystery three phases later.
    let lossy: usize = resident.lossy_narrowing.iter().map(|(_, n)| n).sum();
    if lossy > 0 {
        progress(&format!(
            "narrowed {} unquantized tensors to BF16, {lossy} values lost bits (worst offenders: {})",
            resident.lossy_narrowing.len(),
            resident
                .lossy_narrowing
                .iter()
                .take(3)
                .map(|(n, c)| format!("{n} x{c}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    drop(resident);
    progress(&format!(
        "resident region built ({} bytes)",
        resident_bytes.len()
    ));

    if plan.routed.is_empty() {
        // A DENSE INSTALL STILL NEEDS ITS QUANT BLOCK, which is why this
        // goes through the streaming writer at zero layers rather than
        // through `write_gturbo_install_with_resident_index` -- that one has
        // no way to carry one and writes `"quant": null`. The GGUF walk
        // learned this in ROADMAP M4 (`crates/repack` Gotcha 8); this side
        // kept the old shape only because no DENSE safetensors checkpoint
        // existed until `qwen3_5`.
        //
        // The two produce an identical `layout.json` (stride 0, no layers),
        // so nothing else changes. What changes is that the install now
        // DECLARES its quantization -- which for a 1-bit checkpoint is the
        // only thing that carries `(1, fp16, 128)` to `validate_quant` at
        // all, and without which the whole manifest gate never runs.
        let mut writer = crate::gturbo_writer::StreamingGturboWriter::new(dir, 0, 0)?;
        writer.set_quant(manifest_quant_for(quant, arch.family, false));
        writer.finish(arch, model_id, &resident_bytes)?;
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
        let (blobs, used) = plan_one_expert_layer(shards, arch, quant, &plan.routed, layer)?;
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
        // The streamed walk's reason, verbatim: a dense install still needs
        // its quant block, and `write_gturbo_install_with_resident_index`
        // cannot carry one.
        let mut writer = crate::gturbo_writer::StreamingGturboWriter::new(dir, 0, 0)?;
        writer.set_quant(manifest_quant_for(quant, arch.family, false));
        writer.finish(arch, model_id, &resident_bytes)?;
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
pub fn write_qwen_gdn_moe_install(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Box<dyn std::error::Error>> {
    if arch.family != ModelFamily::QwenGdnMoe {
        return Err(Box::new(Gemma4Error::Config(format!(
            "write_qwen_gdn_moe_install needs arch.family = qwen36, got {}",
            arch.family.as_str()
        ))));
    }
    write_gemma4_install(dir, arch, model_id, header, source, quant)
}

/// [`write_gemma4_install`] behind a `qwen3_5` family guard (ROADMAP's
/// 1-bit entry).
///
/// The same walk again, and the guard is the whole wrapper for the reason
/// [`write_qwen_gdn_moe_install`]'s is: `write_gemma4_install` reads the routed
/// marker and the quant probe names off `arch.family`, so an install written
/// under the wrong tag is well-formed and wrong. That matters more here than
/// for any other pair, because `qwen3_5` and `qwen3_5_moe` are one suffix
/// apart -- a Bonsai checkpoint written as Qwen 3.6 would look for
/// `.mlp.switch_mlp.` experts that do not exist and quietly make every dense
/// FFN tensor resident.
pub fn write_qwen_gdn_dense_install(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Box<dyn std::error::Error>> {
    if arch.family != ModelFamily::QwenGdnDense {
        return Err(Box::new(Gemma4Error::Config(format!(
            "write_qwen_gdn_dense_install needs arch.family = qwen35, got {}",
            arch.family.as_str()
        ))));
    }
    write_gemma4_install(dir, arch, model_id, header, source, quant)
}

/// [`write_qwen_gdn_dense_install`] for the real 4.78 GiB checkpoint.
///
/// The streamed body has nothing to stream on a dense model -- there are no
/// expert layers -- so what this buys over the one-shot writer is the
/// `progress` callback, and on this family that is not cosmetic: it is where
/// the F16-to-BF16 narrowing report comes out (`narrow_raw_to_bf16`), and
/// Bonsai-27B is the first checkpoint whose unquantized tensors are lossy to
/// narrow. A 25-minute walk that does something lossy in silence is how a
/// quality question becomes a mystery three phases later.
pub fn write_qwen_gdn_dense_install_streamed(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    shards: &Gemma4Shards<'_>,
    quant: &Gemma4Quant,
    progress: impl FnMut(&str),
) -> Result<(), Box<dyn std::error::Error>> {
    if arch.family != ModelFamily::QwenGdnDense {
        return Err(Box::new(Gemma4Error::Config(format!(
            "write_qwen_gdn_dense_install_streamed needs arch.family = qwen35, got {}",
            arch.family.as_str()
        ))));
    }
    write_gemma4_install_streamed(dir, arch, model_id, shards, quant, progress)
}

/// The `muse_glimmer` install writer: the same family guard in front of
/// [`write_gemma4_install`].
///
/// Nothing about this walk is family-specific beyond the guard and the
/// routed marker, which is the point -- `muse_glimmer`'s TEN differences
/// from the other families are all in the DECODE FLOW, and none of them is
/// in how its tensors are named, classified or packed. It is dense, so
/// `classify_for_family`'s marker never fires and the install carries zero
/// packed-expert files.
///
/// The multimodal tensors (`vision_tower.`, `vision_adapter.`,
/// `vision_projection.`) are dropped by `classify_for_family`; this port
/// ingests the text tower only.
pub fn write_muse_glimmer_install(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Box<dyn std::error::Error>> {
    if arch.family != ModelFamily::MuseGlimmer {
        return Err(Box::new(Gemma4Error::Config(format!(
            "write_muse_glimmer_install needs arch.family = museGlimmer, got {}",
            arch.family.as_str()
        ))));
    }
    write_gemma4_install(dir, arch, model_id, header, source, quant)
}

/// [`write_muse_glimmer_install`] for the real 19.4 GB checkpoint.
///
/// As with the dense `qwen3_5` sibling there are no expert layers to stream,
/// so what this buys is the `progress` callback. Unlike that one, the
/// narrowing report it carries is expected to be EMPTY: every unquantized
/// tensor in this checkpoint is already BF16 (read off the shard headers,
/// not assumed), so `narrow_raw_to_bf16` has nothing lossy to do. A nonzero
/// count here means the publisher changed the companion dtype, which is
/// worth noticing rather than absorbing.
pub fn write_muse_glimmer_install_streamed(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    shards: &Gemma4Shards<'_>,
    quant: &Gemma4Quant,
    progress: impl FnMut(&str),
) -> Result<(), Box<dyn std::error::Error>> {
    if arch.family != ModelFamily::MuseGlimmer {
        return Err(Box::new(Gemma4Error::Config(format!(
            "write_muse_glimmer_install_streamed needs arch.family = museGlimmer, got {}",
            arch.family.as_str()
        ))));
    }
    write_gemma4_install_streamed(dir, arch, model_id, shards, quant, progress)
}

/// [`write_qwen_gdn_moe_install`] for a real multi-GB checkpoint: the same family
/// guard in front of [`write_gemma4_install_streamed`].
pub fn write_qwen_gdn_moe_install_streamed(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    shards: &Gemma4Shards<'_>,
    quant: &Gemma4Quant,
    progress: impl FnMut(&str),
) -> Result<(), Box<dyn std::error::Error>> {
    if arch.family != ModelFamily::QwenGdnMoe {
        return Err(Box::new(Gemma4Error::Config(format!(
            "write_qwen_gdn_moe_install_streamed needs arch.family = qwen36, got {}",
            arch.family.as_str()
        ))));
    }
    write_gemma4_install_streamed(dir, arch, model_id, shards, quant, progress)
}
