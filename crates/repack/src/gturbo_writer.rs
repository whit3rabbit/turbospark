//! Assembles a byte-exact `.gturbo` install directory: `packed_experts/
//! layer_NN.bin` blobs (matching `turbospark_model_io::PackedExpertsLayout`'s
//! per-expert sub-tensor byte layout), `packed_experts/layout.json`, a
//! minimal `model_weights.bin` (a valid, empty `ResidentIndex` header
//! followed by a raw tensor-data region), and `manifest.json` with computed
//! per-file SHA-256 checksums. This is the piece Phase 8's "streaming
//! installer" work item was missing: given already-quantized tensor bytes
//! (from `repack::quantize_matrix_int4`/`_int8` or any other source), it
//! writes an install that `turbospark_model_io::load_manifest`/
//! `load_packed_experts_layout`/`load_resident_index` can read back.
//!
//! Building the tensors themselves from a downloaded HF checkpoint (walking
//! `safetensors` files, slicing by role, quantizing every projection) is
//! still the caller's job; this module is the on-disk assembly step that
//! sits after it.

use std::collections::BTreeMap;
use std::path::Path;

use model_io::ArchConfig;

#[derive(Debug, Clone, PartialEq)]
pub enum WriterError {
    Io {
        path: String,
        detail: String,
    },
    ExpertOversized {
        layer: usize,
        expert: usize,
        used: u64,
        stride: u64,
    },
    WrongExpertCount {
        layer: usize,
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for WriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriterError::Io { path, detail } => write!(f, "{path}: {detail}"),
            WriterError::ExpertOversized { layer, expert, used, stride } => write!(
                f,
                "layer {layer} expert {expert} uses {used} bytes, exceeding the {stride}-byte expert stride"
            ),
            WriterError::WrongExpertCount { layer, expected, actual } => {
                write!(f, "layer {layer} has {actual} experts, expected {expected}")
            }
        }
    }
}

impl std::error::Error for WriterError {}

/// One named sub-tensor inside an expert's blob (e.g. `"gate"`,
/// `"gate_scales"`, `"gate_biases"`).
#[derive(Debug, Clone)]
pub struct SubTensor {
    pub role: String,
    pub bytes: Vec<u8>,
    pub dtype: String,
    pub shape: Vec<u64>,
}

/// One expert's full set of sub-tensors, written back to back (zero-padded
/// to `expert_stride`) inside its layer file.
#[derive(Debug, Clone)]
pub struct ExpertBlob {
    pub expert: usize,
    pub sub_tensors: Vec<SubTensor>,
}

/// One `packed_experts/layer_NN.bin` file's worth of experts.
#[derive(Debug, Clone)]
pub struct LayerBlobs {
    pub layer: usize,
    pub experts: Vec<ExpertBlob>,
}

fn io_err(path: &Path, e: std::io::Error) -> WriterError {
    WriterError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    }
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

/// Builds one `layer_NN.bin` file's bytes plus its `layout.json` entry.
///
/// `max_stride` is the model-wide ceiling the manifest declares; the layer is
/// padded to ITS OWN stride, which is the largest blob it actually holds
/// rounded up to a page. On a uniform install the two are equal and nothing
/// changes. On a mixed one they are not, and padding every layer to the
/// maximum is the difference between a 10.3 GB install and a 16.2 GB one
/// (ROADMAP Phase S; see `model_io::LayerLayout::expert_stride`).
fn build_layer_file(
    layer: &LayerBlobs,
    max_stride: u64,
    experts_per_layer: usize,
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
    let file_name = format!("layer_{:02}.bin", layer.layer);
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

/// Incremental install assembly for checkpoints too large to hold every
/// layer's expert blobs in memory at once: create the writer, feed it one
/// [`LayerBlobs`] at a time (each layer file hits disk immediately and its
/// bytes can be dropped), then `finish` with the resident index to write
/// `layout.json`, `model_weights.bin`, and `manifest.json`.
pub struct StreamingGturboWriter {
    dir: std::path::PathBuf,
    expert_stride: u64,
    experts_per_layer: usize,
    layout_layers: Vec<serde_json::Value>,
    quant: Option<serde_json::Value>,
}

impl StreamingGturboWriter {
    pub fn new(
        dir: &Path,
        expert_stride: u64,
        experts_per_layer: usize,
    ) -> Result<Self, WriterError> {
        std::fs::create_dir_all(dir.join("packed_experts")).map_err(|e| io_err(dir, e))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            expert_stride,
            experts_per_layer,
            layout_layers: Vec::new(),
            quant: None,
        })
    }

    /// Quantization metadata for `manifest.json -> quant` (camelCase slot
    /// objects, see `turbospark_model_io::ManifestQuant`). Production-shape
    /// manifests are rejected by the loader without it.
    pub fn set_quant(&mut self, quant: serde_json::Value) {
        self.quant = Some(quant);
    }

    pub fn write_layer(&mut self, layer: &LayerBlobs) -> Result<(), WriterError> {
        let (file_bytes, entry) =
            build_layer_file(layer, self.expert_stride, self.experts_per_layer)?;
        let file_name = format!("layer_{:02}.bin", layer.layer);
        let layer_path = self.dir.join("packed_experts").join(&file_name);
        std::fs::write(&layer_path, &file_bytes).map_err(|e| io_err(&layer_path, e))?;
        self.layout_layers.push(entry);
        Ok(())
    }

    pub fn finish(
        self,
        arch: &ArchConfig,
        model_id: &str,
        resident_weights_bin: &[u8],
    ) -> Result<(), WriterError> {
        let num_layers = self.layout_layers.len();
        let layout_json = serde_json::json!({
            "expertStride": self.expert_stride,
            "numLayers": num_layers,
            "expertsPerLayer": self.experts_per_layer,
            "layers": self.layout_layers,
        });
        let layout_path = self.dir.join("packed_experts").join("layout.json");
        std::fs::write(
            &layout_path,
            serde_json::to_vec_pretty(&layout_json).unwrap(),
        )
        .map_err(|e| io_err(&layout_path, e))?;

        let weights_path = self.dir.join("model_weights.bin");
        std::fs::write(&weights_path, resident_weights_bin)
            .map_err(|e| io_err(&weights_path, e))?;

        let manifest_path = self.dir.join("manifest.json");
        let mut manifest_json = build_manifest_json(
            arch,
            model_id,
            self.expert_stride,
            num_layers,
            self.experts_per_layer,
            &self.dir,
        )?;
        if let Some(quant) = self.quant {
            manifest_json["quant"] = quant;
        }
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest_json).unwrap(),
        )
        .map_err(|e| io_err(&manifest_path, e))?;
        Ok(())
    }
}

fn write_gturbo_install_impl(
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
        let (file_bytes, entry) = build_layer_file(layer, expert_stride, experts_per_layer)?;
        let file_name = format!("layer_{:02}.bin", layer.layer);
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

/// A `model_weights.bin` with a valid 24-byte `ResidentIndexHeader`
/// (`index_size == HEADER_BYTES`, zero entries) followed by the raw
/// resident tensor region.
fn build_empty_resident_index(resident_tensor_bytes: &[u8]) -> Vec<u8> {
    const HEADER_BYTES: u64 = 24;
    let mut bytes = Vec::with_capacity(HEADER_BYTES as usize + resident_tensor_bytes.len());
    bytes.extend_from_slice(&HEADER_BYTES.to_le_bytes()); // index_size
    bytes.extend_from_slice(&(resident_tensor_bytes.len() as u64).to_le_bytes()); // resident_size
    bytes.extend_from_slice(&0u64.to_le_bytes()); // entry_count
    bytes.extend_from_slice(resident_tensor_bytes);
    bytes
}

fn build_manifest_json(
    arch: &ArchConfig,
    model_id: &str,
    expert_stride: u64,
    num_layers: usize,
    experts_per_layer: usize,
    dir: &Path,
) -> Result<serde_json::Value, WriterError> {
    let mut files = serde_json::Map::new();
    for relative in ["model_weights.bin", "packed_experts/layout.json"]
        .into_iter()
        .map(String::from)
        .chain((0..num_layers).map(|l| format!("packed_experts/layer_{l:02}.bin")))
    {
        let path = dir.join(&relative);
        let bytes = std::fs::read(&path).map_err(|e| io_err(&path, e))?;
        files.insert(
            relative,
            serde_json::json!({
                "size": bytes.len() as u64,
                "sha256": model_io::hash_data(&bytes),
            }),
        );
    }

    Ok(serde_json::json!({
        "magic": "GTURBO",
        "versionMajor": 1,
        "versionMinor": 0,
        "flags": {},
        "modelID": model_id,
        "sourceSnapshotHash": null,
        "arch": {
            "hiddenSize": arch.hidden_size,
            "ffnIntermediate": arch.intermediate_size,
            "moeIntermediateSize": arch.moe_intermediate_size,
            "numHeads": arch.num_heads,
            "numKVHeads": arch.num_kv_heads,
            "numFullKVHeads": arch.num_full_kv_heads,
            "headDim": arch.head_dim,
            "fullHeadDim": arch.full_head_dim,
            "vocabSize": arch.vocab_size,
            "slidingWindow": arch.sliding_window,
            "finalLogitSoftcap": arch.final_logit_softcap,
            "ropeTheta": arch.rope_theta,
            "fullRopeTheta": arch.full_rope_theta,
            "partialRotaryFactor": arch.partial_rotary_factor,
            "numLayers": arch.num_layers,
            "numExperts": arch.num_experts,
            "topKExperts": arch.top_k_experts,
            "tieWordEmbeddings": arch.tie_word_embeddings,
            "attentionKEqV": arch.attention_k_eq_v,
            "hiddenActivation": arch.hidden_activation,
            "fullAttentionLayerMask": arch.full_attention_layer_mask,
            // Family-extension fields, written UNCONDITIONALLY. `arch_validation`
            // falls back to the Gemma 4 baseline for anything omitted, so a
            // non-Gemma install that leaves them out can never validate (it was
            // compared against Gemma's values). Gemma installs are unaffected:
            // these are exactly the fallbacks for that family.
            "family": arch.family.as_str(),
            "attnOutputGate": arch.attn_output_gate,
            "attentionScale": arch.attention_scale,
            "embeddingScaledBySqrtHidden": arch.embedding_scaled_by_sqrt_hidden,
            "routerScaled": arch.router_scaled,
            "ffnSandwichNorms": arch.ffn_sandwich_norms,
            "sharedExpertGated": arch.shared_expert_gated,
            "ropeNeoxSubdim": arch.rope_neox_subdim,
            "linearNumKHeads": arch.linear_attention.num_k_heads,
            "linearNumVHeads": arch.linear_attention.num_v_heads,
            "linearKeyHeadDim": arch.linear_attention.key_head_dim,
            "linearValueHeadDim": arch.linear_attention.value_head_dim,
            "linearConvKernelSize": arch.linear_attention.conv_kernel_size,
        },
        "quant": null,
        "files": files,
        "expertsPerLayer": experts_per_layer,
        "numLayers": num_layers,
        "expertStride": expert_stride,
    }))
}
