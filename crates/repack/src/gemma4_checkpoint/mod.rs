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
mod dflash;
mod expert_blobs;
mod manifest_quant;
mod mtp;
mod narrow;
mod ngram;
mod orchestrate;
mod shards;
mod vision;

pub use classify::{
    classify_for_family, classify_gemma4, Gemma4Bucket, DFLASH_PREFIX, VISION_INSTALL_PREFIX,
    VISION_PREFIX,
};
pub use config::{
    is_supported_affine_shape, parse_gemma4_config, parse_gemma4_quantization, Gemma4Error,
    Gemma4Quant, AFFINE_1BIT_GROUP_SIZE, AFFINE_2BIT_GROUP_SIZE, AFFINE_GROUP_SIZE,
};
pub use expert_blobs::{expert_stride_from_headers, plan_one_expert_layer};
pub use manifest_quant::{gemma4_manifest_quant, manifest_quant, manifest_quant_for};
pub use narrow::{
    convert_raw_to_fp16, narrow_raw_to_bf16, pass_through_packed, ConvertedFp16, NarrowedRaw,
};
pub use ngram::{write_ngram_table, NgramPlan, NgramTableSpec, NgramTableWriter};
pub use orchestrate::{
    orchestrate_gemma4_checkpoint, orchestrate_gemma4_checkpoint_sharded, Gemma4RepackOutput,
};
pub use shards::{Gemma4Shards, GTURBO_PAGE_BYTES};
pub use vision::{
    read_vision_entries, vision_arch_for_manifest, vision_should_ingest, VisionRead,
    BLOCK_ROLES as VISION_BLOCK_ROLES, RESIDENT_TENSORS as VISION_RESIDENT_TENSORS,
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
    let expert_stride = expert_stride_from_headers(shards, arch, quant, &plan.routed)?;
    progress(&format!(
        "classified {} resident tensors, {} routed layers, expert stride {expert_stride}",
        plan.resident_bases.len(),
        plan.routed.len(),
    ));

    let mut resident = orchestrate::read_resident_entries(shards, &plan.resident_bases, quant)?;
    // THE MTP HEAD, and this arm has to exist HERE as well as in
    // `orchestrate_gemma4_checkpoint_sharded` -- which is the whole reason it
    // is worth a comment. Every REAL install goes through this streamed
    // writer and every fixture went through the other one, so step 1's head
    // ingest was gated by a test that could not reach the code path any real
    // checkpoint takes: the first stream that asked for a head produced a
    // byte-identical HEADLESS install, 851 resident tensors and all, with no
    // error anywhere. `crates/repack` Gotcha 8 says to build the fixture
    // before the download; the lesson this adds is that the fixture has to
    // exercise the WRITER the download will use.
    if !plan.mtp_bases.is_empty() {
        let head = mtp::read_mtp_entries(shards, &plan.mtp_bases)?;
        progress(&format!(
            "ingested a {}-tensor multi-token-prediction head",
            head.entries.len()
        ));
        resident.entries.extend(head.entries);
        resident.lossy_narrowing.extend(head.lossy_narrowing);
    }
    // THE DFLASH2 DRAFTER, and this arm exists HERE for the same reason the
    // head's does one block up: every real install takes this streamed
    // writer and every fixture used to take the other one, and the head's
    // first real stream wrote a headless install because this function
    // classified its names and never read them (`both_writers_carry_the_
    // mtp_head` is the test that caught it). The drafter gets its arm in
    // both writers from day one and the same shape of test.
    if !plan.dflash_bases.is_empty() {
        let drafter = dflash::read_dflash_entries(shards, &plan.dflash_bases)?;
        progress(&format!(
            "ingested a {}-tensor DFlash2 drafter",
            drafter.entries.len()
        ));
        resident.entries.extend(drafter.entries);
        resident.lossy_narrowing.extend(drafter.lossy_narrowing);
    }
    // THE VISION TOWER (ROADMAP M-V3), read HERE rather than beside the
    // `write_packed_vision` call further down, because it is the walk's one
    // arm with TWO destinations: its 27 blocks become packed blobs, but its
    // nine non-block tensors are RESIDENT entries and `resident.entries` is
    // consumed a few lines below. Reading it after that point would write a
    // tower whose merger and patch embedding are simply missing -- an install
    // that validates, opens, and has no way to turn an image into tokens.
    //
    // The same both-writers rule the two arms above carry applies to this one
    // and is stated at the `write_packed_vision` call site.
    let tower = if vision::vision_should_ingest(arch, &plan.vision_bases) {
        let mut tower = vision::read_vision_entries(shards, &plan.vision_bases, &arch.vision)?;
        progress(&format!(
            "ingested a {}-block vision tower ({} resident tensors)",
            tower.blocks.experts.len(),
            tower.entries.len()
        ));
        // MOVED rather than cloned: the merger's two matrices are ~90 MiB
        // between them, and this walk's whole point is that peak memory is one
        // unit of work rather than the model.
        resident.entries.extend(std::mem::take(&mut tower.entries));
        Some(tower)
    } else {
        None
    };
    if let Some(tower) = &tower {
        let lossy: usize = tower.lossy_conversion.iter().map(|(_, n)| n).sum();
        if lossy > 0 {
            progress(&format!(
                "converted {} vision tensors to FP16 with {lossy} values losing bits \
                 (subnormals; the normal range is exact -- see `convert_raw_to_fp16`)",
                tower.lossy_conversion.len()
            ));
        }
    }

    // THE N-GRAM TABLE (`qwen4_exp`), read and written HERE rather than
    // carried through `Gemma4RepackOutput` the way the head's and the
    // tower's bytes are: at 32 GB it cannot be held in memory at all, so the
    // read and the write are one streamed loop rather than two separate
    // steps. See `ngram::write_ngram_table`'s doc for why this arm and
    // `write_gemma4_install`'s are the two places this table is ingested,
    // where every other component's non-streamed arm lives in
    // `orchestrate_gemma4_checkpoint_sharded` instead.
    if !plan.ngram_shards.is_empty() {
        let ngram_plan = NgramPlan::from_classified(&plan.ngram_shards, &plan.ngram_meta);
        write_ngram_table(shards, arch, &ngram_plan, dir, &mut progress)?;
    }

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

    // THE VISION TOWER, and this arm has to exist HERE as well as in
    // `orchestrate_gemma4_checkpoint_sharded` for the reason the MTP head's
    // comment above spells out at length: every REAL install takes this
    // streamed writer and every fixture used to take the other one, so an arm
    // in one writer alone is gated by a test that cannot reach the path a
    // download takes. The head cost a 15-minute stream to discover that; the
    // tower gets its arm in both writers from day one and a
    // `both_writers_carry_the_vision_tower` test that fails if either is
    // removed.
    //
    // It writes BEFORE `writer.finish`, which is not a preference:
    // `build_manifest_json` HASHES every file it lists, so `packed_vision/`
    // has to be on disk by then. Getting the order wrong is an io error
    // naming the missing file, which is the good failure mode and is why the
    // manifest hashes the file rather than the bytes in hand.
    let ingest_vision = vision::vision_should_ingest(arch, &plan.vision_bases);
    if let Some(tower) = &tower {
        crate::gturbo_writer::write_packed_vision(dir, &tower.blocks, tower.block_stride)?;
        progress(&format!(
            "vision tower packed ({} blocks, {} bytes/block)",
            tower.blocks.experts.len(),
            tower.block_stride
        ));
    } else if !plan.vision_bases.is_empty() {
        // Classified and deliberately dropped: the caller passed
        // `VisionConfig::NONE`, which is what every text-only walk does. Said
        // out loud rather than dropped in silence, because "this checkpoint
        // has a tower and this install will not have one" is exactly the kind
        // of fact a reader of a 25-minute log needs and cannot recover later.
        progress(&format!(
            "dropped {} vision-tower tensors (this walk was asked for a text-only install)",
            plan.vision_bases.len()
        ));
    }
    let arch = vision::vision_arch_for_manifest(arch, ingest_vision);
    let arch = arch.as_ref();

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
    // THE VISION TOWER, before either `finish` below, because
    // `build_manifest_json` hashes every file it lists. The streamed writer
    // has the same two lines and the same ordering constraint; see its
    // comment for why both writers carry this rather than one.
    if let Some(tower) = &out.vision {
        crate::gturbo_writer::write_packed_vision(dir, &tower.blocks, tower.block_stride)?;
    }
    // THE N-GRAM TABLE, and this arm exists HERE as well as in
    // `write_gemma4_install_streamed` for that function's reason: the table
    // is 32 GB and cannot be carried in `out.ngram` as bytes, only as names,
    // so the streamed read-and-write happens at each writer entry point
    // rather than once in `orchestrate_gemma4_checkpoint_sharded`.
    // `Gemma4Shards::single` costs nothing beyond this call -- it is a name
    // index over `header`/`source`, not a read.
    if let Some(plan) = &out.ngram {
        let shards = Gemma4Shards::single(header, source);
        write_ngram_table(&shards, arch, plan, dir, |_| {})?;
    }
    let arch = vision::vision_arch_for_manifest(arch, out.vision.is_some());
    let arch = arch.as_ref();
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

/// `qwen4_exp` for a real multi-GB checkpoint: the same family guard in
/// front of [`write_gemma4_install_streamed`], which already carries the
/// n-gram table's own streamed read-then-write arm internally
/// (`ngram.rs`'s `write_ngram_table`, called from both writers) -- so this
/// wrapper needs nothing beyond the guard every other family's has.
pub fn write_qwen4_exp_install_streamed(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    shards: &Gemma4Shards<'_>,
    quant: &Gemma4Quant,
    progress: impl FnMut(&str),
) -> Result<(), Box<dyn std::error::Error>> {
    if arch.family != ModelFamily::Qwen4Exp {
        return Err(Box::new(Gemma4Error::Config(format!(
            "write_qwen4_exp_install_streamed needs arch.family = qwen4exp, got {}",
            arch.family.as_str()
        ))));
    }
    write_gemma4_install_streamed(dir, arch, model_id, shards, quant, progress)
}
