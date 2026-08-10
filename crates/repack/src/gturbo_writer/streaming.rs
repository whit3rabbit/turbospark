//! Streaming writer implementation for large `.gturbo` install generation.

use std::path::Path;

use model_io::ArchConfig;

use super::layers::build_layer_file;
use super::manifest::build_manifest_json;
use super::types::{io_err, LayerBlobs, WriterError};

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
    /// Creates a new `StreamingGturboWriter` initializing the output directory.
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

    /// Writes one layer of expert blobs directly to disk and records its layout entry.
    pub fn write_layer(&mut self, layer: &LayerBlobs) -> Result<(), WriterError> {
        let (file_bytes, entry) =
            build_layer_file(layer, self.expert_stride, self.experts_per_layer)?;
        let file_name = format!("layer_{:02}.bin", layer.layer);
        let layer_path = self.dir.join("packed_experts").join(&file_name);
        std::fs::write(&layer_path, &file_bytes).map_err(|e| io_err(&layer_path, e))?;
        self.layout_layers.push(entry);
        Ok(())
    }

    /// Records a layer's layout entry for a file ALREADY on disk, without
    /// writing anything (ROADMAP Phase M2's resume path).
    ///
    /// `layer` must have been built by `plan_one_layer_shape`, whose
    /// sub-tensors are correctly sized runs of zeros: the entry is a function
    /// of sizes, dtypes and shapes, so it comes out identical to the one the
    /// real bytes would have produced. The size the writer WOULD have written
    /// is checked against the file that is there, so a truncated or
    /// differently-strided leftover is refused rather than adopted -- which is
    /// the only way this can go quietly wrong.
    pub fn adopt_layer(&mut self, layer: &LayerBlobs) -> Result<(), WriterError> {
        let (file_bytes, entry) =
            build_layer_file(layer, self.expert_stride, self.experts_per_layer)?;
        let file_name = format!("layer_{:02}.bin", layer.layer);
        let layer_path = self.dir.join("packed_experts").join(&file_name);
        let found = std::fs::metadata(&layer_path)
            .map_err(|e| io_err(&layer_path, e))?
            .len();
        if found != file_bytes.len() as u64 {
            return Err(WriterError::Io {
                path: layer_path.display().to_string(),
                detail: format!(
                    "cannot resume: layer file is {found} bytes, this walk would write {}",
                    file_bytes.len()
                ),
            });
        }
        self.layout_layers.push(entry);
        Ok(())
    }

    /// Finalizes the installation by writing `layout.json`, `model_weights.bin`, and `manifest.json`.
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
