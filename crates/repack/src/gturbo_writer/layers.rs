//! Layer file construction and batch `.gturbo` install writer functions.

use std::collections::BTreeMap;
use std::path::Path;

use model_io::ArchConfig;

use super::manifest::{build_empty_resident_index, build_manifest_json};
use super::types::{io_err, ExpertBlob, LayerBlobs, WriterError};

/// Builds one `layer_NN.bin` file's bytes plus its `layout.json` entry.
///
/// `max_stride` is the model-wide ceiling the manifest declares; the layer is
/// padded to ITS OWN stride, which is the largest blob it actually holds
/// rounded up to a page. On a uniform install the two are equal and nothing
/// changes. On a mixed one they are not, and padding every layer to the
/// maximum is the difference between a 10.3 GB install and a 16.2 GB one
/// (ROADMAP Phase S; see `model_io::LayerLayout::expert_stride`).
///
/// `file_name` is what the layout entry records for the caller to open. It is
/// a parameter rather than the `layer_NN.bin` this used to derive, because
/// ROADMAP M-V3's vision tower packs its blocks through this exact function
/// into a single `blobs.bin` -- the tower is one "layer" of `depth` "experts",
/// and the only thing about it that is not already a routed layer's shape is
/// what the file is called.
pub(crate) fn build_layer_file(
    layer: &LayerBlobs,
    max_stride: u64,
    experts_per_layer: usize,
    file_name: &str,
) -> Result<(Vec<u8>, serde_json::Value), WriterError> {
    if layer.experts.len() != experts_per_layer {
        return Err(WriterError::WrongExpertCount {
            layer: layer.layer,
            expected: experts_per_layer,
            actual: layer.experts.len(),
        });
    }
    let used = |e: &ExpertBlob| e.sub_tensors.iter().map(|s| s.bytes.len() as u64).sum();
    let widest: u64 = layer.experts.iter().map(used).max().unwrap_or(0);
    if widest > max_stride {
        // Reported against the widest expert rather than the first oversized
        // one: the caller's ceiling is what is wrong here, not this expert.
        return Err(WriterError::ExpertOversized {
            layer: layer.layer,
            expert: 0,
            used: widest,
            stride: max_stride,
        });
    }
    // Clamped to the caller's stride rather than only rounded up, because a
    // small install can legitimately declare one BELOW the page constant --
    // the toy fixtures use 4096 -- and the clamped value inherits the
    // manifest's own page-alignment guarantee. Above it, the round-up is what
    // keeps every expert offset page-aligned for the streamer.
    let expert_stride =
        (widest.div_ceil(crate::GTURBO_PAGE_BYTES) * crate::GTURBO_PAGE_BYTES).min(max_stride);
    let mut file_bytes = Vec::with_capacity(layer.experts.len() * expert_stride as usize);
    let mut expert_entries = Vec::with_capacity(layer.experts.len());

    for expert in &layer.experts {
        let expert_offset = file_bytes.len() as u64;
        let mut cursor = 0u64;
        let mut tensor_entries = BTreeMap::new();
        for sub in &expert.sub_tensors {
            let sub_offset = cursor;
            file_bytes.extend_from_slice(&sub.bytes);
            cursor += sub.bytes.len() as u64;
            tensor_entries.insert(
                sub.role.clone(),
                serde_json::json!({
                    "offset": sub_offset,
                    "size": sub.bytes.len() as u64,
                    "dtype": sub.dtype,
                    "shape": sub.shape,
                }),
            );
        }
        debug_assert!(
            cursor <= expert_stride,
            "stride was derived from the widest expert"
        );
        file_bytes.resize(expert_offset as usize + expert_stride as usize, 0u8);
        expert_entries.push(serde_json::json!({
            "expert": expert.expert,
            "offset": expert_offset,
            "size": expert_stride,
            "tensors": tensor_entries,
        }));
    }
    let entry = serde_json::json!({
        "layer": layer.layer,
        "file": file_name,
        "expertStride": expert_stride,
        "experts": expert_entries,
    });
    Ok((file_bytes, entry))
}

/// Writes `packed_vision/{blobs.bin,layout.json}` (ROADMAP M-V3).
///
/// The tower is ONE layer of `blocks.experts.len()` experts, so this is
/// [`build_layer_file`] plus the two-key layout wrapper and nothing else.
/// Written as its own small function rather than as a mode on
/// `write_gturbo_install_impl`, because that one also writes
/// `model_weights.bin` and `manifest.json` -- and the manifest has to be
/// written LAST, after these files exist, since `build_manifest_json` hashes
/// every file it lists.
///
/// **It must therefore be called BEFORE the manifest**, in both writers. A
/// caller that gets the order wrong gets an io error naming
/// `packed_vision/blobs.bin`, which is the good failure mode and is why the
/// hashing reads the file rather than the bytes in hand.
pub fn write_packed_vision(
    dir: &Path,
    blocks: &LayerBlobs,
    block_stride: u64,
) -> Result<(), WriterError> {
    let subdir = dir.join(model_io::PACKED_VISION_DIR);
    std::fs::create_dir_all(&subdir).map_err(|e| io_err(dir, e))?;

    let blocks_per_tower = blocks.experts.len();
    let (file_bytes, entry) =
        build_layer_file(blocks, block_stride, blocks_per_tower, "blobs.bin")?;
    let blobs_path = subdir.join("blobs.bin");
    std::fs::write(&blobs_path, &file_bytes).map_err(|e| io_err(&blobs_path, e))?;

    // The same four top-level keys `packed_experts/layout.json` carries, so
    // `model_io::load_packed_layout_from` decodes this with no second parser.
    // `expertsPerLayer` is the BLOCK COUNT and `numLayers` is 1; the words are
    // the schema's rather than the tower's, which is the cost of the reuse and
    // is cheaper than a parallel format.
    let layout_json = serde_json::json!({
        "expertStride": block_stride,
        "numLayers": 1,
        "expertsPerLayer": blocks_per_tower,
        "layers": [entry],
    });
    let layout_path = subdir.join("layout.json");
    std::fs::write(
        &layout_path,
        serde_json::to_vec_pretty(&layout_json).unwrap(),
    )
    .map_err(|e| io_err(&layout_path, e))?;
    Ok(())
}

/// Writes a full `.gturbo` install to `dir`: `manifest.json`,
/// `packed_experts/layout.json`, one `packed_experts/layer_NN.bin` per
/// entry in `layers`, and a minimal valid `model_weights.bin` wrapping
/// `resident_tensor_bytes` (the raw resident tensor region; an empty slice
/// is a valid, if useless, resident index).
pub fn write_gturbo_install(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    expert_stride: u64,
    experts_per_layer: usize,
    layers: &[LayerBlobs],
    resident_tensor_bytes: &[u8],
) -> Result<(), WriterError> {
    let weights_bytes = build_empty_resident_index(resident_tensor_bytes);
    write_gturbo_install_impl(
        dir,
        arch,
        model_id,
        expert_stride,
        experts_per_layer,
        layers,
        &weights_bytes,
    )
}

/// The general assembly: a caller-supplied complete `model_weights.bin`
/// (real resident index included) PLUS packed-expert layer files. This is
/// what a streamed-MoE install needs: attention/router weights resident,
/// routed experts in `packed_experts/layer_NN.bin` files read at decode
/// time by `turbospark-streaming`'s `PreadExpertStreamer`.
pub fn write_gturbo_install_with_resident_index_and_experts(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    resident_weights_bin: &[u8],
    expert_stride: u64,
    experts_per_layer: usize,
    layers: &[LayerBlobs],
) -> Result<(), WriterError> {
    write_gturbo_install_impl(
        dir,
        arch,
        model_id,
        expert_stride,
        experts_per_layer,
        layers,
        resident_weights_bin,
    )
}

pub(crate) fn write_gturbo_install_impl(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    expert_stride: u64,
    experts_per_layer: usize,
    layers: &[LayerBlobs],
    weights_bytes: &[u8],
) -> Result<(), WriterError> {
    std::fs::create_dir_all(dir.join("packed_experts")).map_err(|e| io_err(dir, e))?;

    let mut layout_layers = Vec::with_capacity(layers.len());
    for layer in layers {
        let file_name = format!("layer_{:02}.bin", layer.layer);
        let (file_bytes, entry) =
            build_layer_file(layer, expert_stride, experts_per_layer, &file_name)?;
        let layer_path = dir.join("packed_experts").join(&file_name);
        std::fs::write(&layer_path, &file_bytes).map_err(|e| io_err(&layer_path, e))?;
        layout_layers.push(entry);
    }

    let layout_json = serde_json::json!({
        "expertStride": expert_stride,
        "numLayers": layers.len(),
        "expertsPerLayer": experts_per_layer,
        "layers": layout_layers,
    });
    let layout_path = dir.join("packed_experts").join("layout.json");
    std::fs::write(
        &layout_path,
        serde_json::to_vec_pretty(&layout_json).unwrap(),
    )
    .map_err(|e| io_err(&layout_path, e))?;

    let weights_path = dir.join("model_weights.bin");
    std::fs::write(&weights_path, weights_bytes).map_err(|e| io_err(&weights_path, e))?;

    let manifest_path = dir.join("manifest.json");
    let manifest_json = build_manifest_json(
        arch,
        model_id,
        expert_stride,
        layers.len(),
        experts_per_layer,
        dir,
    )?;
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest_json).unwrap(),
    )
    .map_err(|e| io_err(&manifest_path, e))?;

    Ok(())
}

/// Writes a `.gturbo` install with a real, named resident-tensor index
/// (`resident_weights_bin`, built by
/// `resident_writer::build_resident_weights_bin`) instead of the empty
/// placeholder index `write_gturbo_install` writes. No packed experts (an
/// empty `packed_experts/layout.json`, zero layers): for a small synthetic
/// model with every weight resident, not streamed.
pub fn write_gturbo_install_with_resident_index(
    dir: &Path,
    arch: &ArchConfig,
    model_id: &str,
    resident_weights_bin: &[u8],
) -> Result<(), WriterError> {
    std::fs::create_dir_all(dir.join("packed_experts")).map_err(|e| io_err(dir, e))?;

    let layout_json = serde_json::json!({
        "expertStride": 0u64,
        "numLayers": 0,
        "expertsPerLayer": 0,
        "layers": [],
    });
    let layout_path = dir.join("packed_experts").join("layout.json");
    std::fs::write(
        &layout_path,
        serde_json::to_vec_pretty(&layout_json).unwrap(),
    )
    .map_err(|e| io_err(&layout_path, e))?;

    let weights_path = dir.join("model_weights.bin");
    std::fs::write(&weights_path, resident_weights_bin).map_err(|e| io_err(&weights_path, e))?;

    let manifest_path = dir.join("manifest.json");
    let manifest_json = build_manifest_json(arch, model_id, 0, 0, 0, dir)?;
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest_json).unwrap(),
    )
    .map_err(|e| io_err(&manifest_path, e))?;

    Ok(())
}
