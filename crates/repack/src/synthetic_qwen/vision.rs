//! A tiny `qwen3_5` vision tower for the fixtures (ROADMAP M-V3).
//!
//! **EVERY DIMENSION HERE IS DISTINCT ON PURPOSE, AND THAT IS THE WHOLE VALUE
//! OF THE FIXTURE.** The real tower has `hidden = 1152` and `merger_input =
//! 4608 = 4 * 1152`, so several of its shapes are multiples of one another and
//! a walk that confused two of them would still produce correctly-SIZED bytes.
//! `SyntheticGptOssShape` records the same lesson from M5: the published
//! gpt-oss has `hidden == expert width`, so a fixture copying its proportions
//! could not tell a gate bias from a down bias.
//!
//! So the shapes below are chosen to be mutually indivisible where the real
//! ones are not -- `intermediate` is not a multiple of `hidden`, `out_hidden`
//! differs from both, and `num_position_embeddings` is not `hidden`. A
//! transposed read, a wrong stride or a swapped role changes the byte COUNT
//! and is caught by construction rather than by an assertion someone has to
//! remember to write.

use model_io::{ModelFamily, VisionConfig};

use super::dense::HIDDEN;
use super::dense_tensors::f16_vector;
use crate::gemma4_checkpoint::{
    read_vision_entries, write_vision_sidecar, Gemma4Shards, VISION_BLOCK_ROLES, VISION_PREFIX,
};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
use crate::synthetic_real::{assemble_safetensors, Tensor};

/// Blocks in the toy tower. TWENTY-SEVEN, which is `SUPPORTED_VISION_DEPTH`:
/// the real checkpoints all declare 27 and the depth bound refuses anything
/// else at both the config parse and the ingest, so a toy depth would fail
/// every fixture that builds a tower through the real walk. Two rather than
/// one is still the load-bearing property: a single-block tower cannot tell a
/// per-block stride from a whole-file size, and cannot catch a walk that
/// writes every block at block 0's offset.
pub(crate) const V_DEPTH: i64 = 27;
/// The tower's residual width.
pub(crate) const V_HIDDEN: i64 = 64;
/// The per-block MLP width. **NOT a multiple of `V_HIDDEN`**, unlike the real
/// tower's 4304 against 1152 -- see the module header.
pub(crate) const V_INTER: i64 = 96;
/// Attention heads, giving `head_dim = 16`.
pub(crate) const V_HEADS: i64 = 4;
/// Spatial patch edge.
pub(crate) const V_PATCH: i64 = 4;
/// Frames per patch, as in the real tower.
pub(crate) const V_TEMPORAL: i64 = 2;
/// Patch merge edge: 4 patches per output token, as in the real tower.
pub(crate) const V_MERGE: i64 = 2;
/// Position-embedding rows. A 4x4 grid, so the square check has something to
/// pass; distinct from every other constant here.
pub(crate) const V_POS: i64 = 16;

/// The toy tower's config.
///
/// `out_hidden_size` is the TEXT trunk's `HIDDEN`, not an invented number:
/// the merger writes straight into the trunk's residual stream, so on the real
/// checkpoint those two are equal (5120) and a fixture that made them differ
/// would be describing a model that cannot be assembled.
pub fn tiny_vision_config() -> VisionConfig {
    VisionConfig {
        depth: V_DEPTH,
        hidden_size: V_HIDDEN,
        intermediate_size: V_INTER,
        num_heads: V_HEADS,
        patch_size: V_PATCH,
        temporal_patch_size: V_TEMPORAL,
        in_channels: 3,
        spatial_merge_size: V_MERGE,
        num_position_embeddings: V_POS,
        out_hidden_size: HIDDEN as i64,
        // The real checkpoints' triple, which sums to 32. Carried verbatim
        // because it is metadata rather than a shape -- nothing in this
        // fixture's bytes depends on it, and inventing a different one would
        // make the fixture disagree with every published file for no reason.
        mrope_section: [11, 11, 10],
        // The published qwen3_5 checkpoints' value: the key present and
        // empty. A deepstack-bearing synthetic fixture is a qwen3_vl question
        // and this one is not that fixture.
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: 248_053,
        vision_end_token_id: 248_054,
        image_token_id: 248_056,
        video_token_id: 248_057,
    }
}

/// The tower's 2 * 12 + 9 tensors, named exactly as the real checkpoint names
/// them.
///
/// F16 throughout, matching `prism-ml/Bonsai-27B-mlx-1bit`, which is also what
/// the rest of this dense fixture writes. Note the OTHER real checkpoint
/// (`mlx-community/Qwen3.8-27B-4bit`, the one the M-V3 network gate streams)
/// ships the identical tower in BF16, so this fixture exercises the verbatim
/// arm of `convert_raw_to_fp16` and NOT its conversion arm --
/// `a_bf16_tower_converts_to_the_same_fp16_bytes` is what covers the other.
pub(crate) fn vision_tower_tensors() -> Vec<Tensor> {
    vision_tower_tensors_with_out_hidden(HIDDEN)
}

/// [`vision_tower_tensors`] for a trunk of width `out_hidden`: only the
/// merger's `linear_fc2` (`[out_hidden, merger_in]`) and its bias depend on it.
pub(crate) fn vision_tower_tensors_with_out_hidden(out_hidden: usize) -> Vec<Tensor> {
    let vision = tiny_vision_config();
    let hidden = V_HIDDEN as usize;
    let inter = V_INTER as usize;
    let mut ts = Vec::new();

    for block in 0..V_DEPTH as usize {
        let p = format!("{VISION_PREFIX}blocks.{block}");
        let seed = 9000 + 100 * block as u64;
        for (i, (_role, suffix)) in VISION_BLOCK_ROLES.iter().enumerate() {
            let (rows, cols) = block_shape(suffix, hidden, inter);
            let name = format!("{p}.{suffix}");
            ts.push(match cols {
                // A bias or a norm: rank 1.
                None => f16_vector(&name, rows, 1.0, seed + i as u64),
                Some(cols) => f16_matrix(&name, rows, cols, seed + i as u64),
            });
        }
    }

    let merger_in = vision.merger_input_dim() as usize;
    // `patch_embed.proj.weight` is RANK 5 -- `(out, T, P_h, P_w, C)` -- and
    // that rank is the point: `shape4` truncates, and the walk must carry the
    // tensor verbatim regardless. See `vision::RESIDENT_TENSORS` for why no
    // permutation happens here or anywhere.
    let patch_in = (V_TEMPORAL * V_PATCH * V_PATCH * 3) as usize;
    let flat = f16_vector("scratch", hidden * patch_in, 0.0, 8100);
    ts.push(Tensor {
        name: format!("{VISION_PREFIX}patch_embed.proj.weight"),
        dtype: flat.dtype,
        shape: vec![
            hidden as u64,
            V_TEMPORAL as u64,
            V_PATCH as u64,
            V_PATCH as u64,
            3,
        ],
        bytes: flat.bytes,
    });
    ts.push(f16_vector(
        &format!("{VISION_PREFIX}patch_embed.proj.bias"),
        hidden,
        0.0,
        8101,
    ));
    ts.push(f16_matrix(
        &format!("{VISION_PREFIX}pos_embed.weight"),
        V_POS as usize,
        hidden,
        8102,
    ));
    ts.push(f16_vector(
        &format!("{VISION_PREFIX}merger.norm.weight"),
        hidden,
        1.0,
        8103,
    ));
    ts.push(f16_vector(
        &format!("{VISION_PREFIX}merger.norm.bias"),
        hidden,
        0.0,
        8104,
    ));
    ts.push(f16_matrix(
        &format!("{VISION_PREFIX}merger.linear_fc1.weight"),
        merger_in,
        merger_in,
        8105,
    ));
    ts.push(f16_vector(
        &format!("{VISION_PREFIX}merger.linear_fc1.bias"),
        merger_in,
        0.0,
        8106,
    ));
    ts.push(f16_matrix(
        &format!("{VISION_PREFIX}merger.linear_fc2.weight"),
        out_hidden,
        merger_in,
        8107,
    ));
    ts.push(f16_vector(
        &format!("{VISION_PREFIX}merger.linear_fc2.bias"),
        out_hidden,
        0.0,
        8108,
    ));
    ts
}

/// Rewrites every rank-2 tower matrix whose width divides by 64 as an MLX
/// affine 4-bit plane at group 64: a `U32` `.weight` plus BF16 `.scales` and
/// `.biases`, the layout the `qwen4_exp` MLX conversion ships
/// (`docs/QWEN4EXP_VISION_PHASE0.md` item 1).
///
/// The rule is the conversion's own, so the fixture keeps its shape
/// exceptions: the block `linear_fc2` (96 wide here, 4304 on the real tower)
/// stays F16, and so does the rank-5 patch embedding. Rows are quantized by
/// `compute::quantize_int4_affine`, whose nibble order is the MLX one.
pub(crate) fn quantize_tower_int4(ts: Vec<Tensor>) -> Vec<Tensor> {
    const GROUP: usize = compute::quant::GROUP_SIZE;
    let mut out = Vec::with_capacity(ts.len() * 3);
    for t in ts {
        let quantizable = t.dtype == "F16"
            && t.shape.len() == 2
            && t.name.ends_with(".weight")
            && t.shape[1] as usize % GROUP == 0;
        if !quantizable {
            out.push(t);
            continue;
        }
        let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
        let groups = cols / GROUP;
        let (mut packed, mut scales, mut biases) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..rows {
            let row: Vec<f32> = t.bytes[r * cols * 2..(r + 1) * cols * 2]
                .chunks_exact(2)
                .map(|b| compute::f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            let q = compute::quantize_int4_affine(&row);
            packed.extend_from_slice(&q.packed);
            scales.extend(q.scales.iter().flat_map(|v| v.to_le_bytes()));
            biases.extend(q.biases.iter().flat_map(|v| v.to_le_bytes()));
        }
        let base = t.name.strip_suffix(".weight").expect("checked above");
        out.push(Tensor {
            name: t.name.clone(),
            dtype: "U32",
            shape: vec![rows as u64, (cols / 8) as u64],
            bytes: packed,
        });
        for (role, bytes) in [("scales", scales), ("biases", biases)] {
            out.push(Tensor {
                name: format!("{base}.{role}"),
                dtype: "BF16",
                shape: vec![rows as u64, groups as u64],
                bytes,
            });
        }
    }
    out
}

/// `(rows, Some(cols))` for a matrix, `(len, None)` for a vector.
///
/// Driven off `VISION_BLOCK_ROLES`' own suffixes rather than a second list, so
/// a role added there without a shape here fails to compile instead of
/// producing a silently mis-sized fixture.
fn block_shape(suffix: &str, hidden: usize, inter: usize) -> (usize, Option<usize>) {
    match suffix {
        "norm1.weight" | "norm1.bias" | "norm2.bias" | "norm2.weight" => (hidden, None),
        "attn.qkv.weight" => (3 * hidden, Some(hidden)),
        "attn.qkv.bias" => (3 * hidden, None),
        "attn.proj.weight" => (hidden, Some(hidden)),
        "attn.proj.bias" => (hidden, None),
        "mlp.linear_fc1.weight" => (inter, Some(hidden)),
        "mlp.linear_fc1.bias" => (inter, None),
        "mlp.linear_fc2.weight" => (hidden, Some(inter)),
        "mlp.linear_fc2.bias" => (hidden, None),
        other => unreachable!("no shape for vision block role {other}"),
    }
}

fn f16_matrix(name: &str, rows: usize, cols: usize, seed: u64) -> Tensor {
    let flat = f16_vector(name, rows * cols, 0.0, seed);
    Tensor {
        name: flat.name,
        dtype: flat.dtype,
        shape: vec![rows as u64, cols as u64],
        bytes: flat.bytes,
    }
}

/// A SIDECAR-ONLY fixture (vision memory sidecar, part A1): the tower alone,
/// written through [`write_vision_sidecar`] rather than baked into a full
/// dense install.
///
/// **Built from the SAME [`vision_tower_tensors`] the combined-install
/// fixture uses**, which is the load-bearing property of the whole sidecar
/// design and not an implementation convenience: it is what lets a test
/// assert `packed_vision/blobs.bin` and the nine resident tensor byte ranges
/// are byte-IDENTICAL between "tower baked into a full install"
/// (`build_synthetic_qwen_gdn_dense_install_with_vision`) and "tower alone
/// in a sidecar" (this function), by construction rather than by two
/// builders happening to agree.
///
/// Also drops a placeholder `preprocessor_config.json` into `dir`, so
/// `model_io::load_vision_sidecar` can be exercised end to end in tests
/// without a real download -- fetching the real file from the source
/// repository is catalog ingest, a later part of this feature, not this
/// fixture's job.
///
/// Returns `Result` rather than a bare `VisionConfig`: every write in this
/// walk (the packed blocks, the resident index, the manifest, the sidecar
/// record) can fail, and every sibling fixture builder in this module
/// propagates that instead of panicking.
pub fn build_synthetic_vision_sidecar(
    dir: &std::path::Path,
    model_id: &str,
) -> Result<VisionConfig, Box<dyn std::error::Error>> {
    let vision = tiny_vision_config();
    let ts = vision_tower_tensors();
    let vision_bases: Vec<&str> = ts.iter().map(|t| t.name.as_str()).collect();

    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, DEFAULT_MAX_HEADER_BYTES)?;
    let shards = Gemma4Shards::single(&header, &source);

    let read = read_vision_entries(&shards, &vision_bases, &vision)?;

    let family = ModelFamily::QwenGdnDense;
    let record = model_io::SidecarRecord {
        kind: model_io::SIDECAR_KIND.to_string(),
        pairs_with: model_io::PairsWith {
            family: family.as_str().to_string(),
            hidden_size: vision.out_hidden_size,
        },
        source: model_io::SidecarSource {
            repo: "test/vision-sidecar-fixture".to_string(),
            revision: "0".repeat(40),
            prefix: VISION_PREFIX.to_string(),
            file: "model.safetensors".to_string(),
        },
        tower_blocks: vision.depth,
        block_stride: read.block_stride,
    };

    write_vision_sidecar(dir, family, &vision, model_id, &read, record)?;

    // Not a real download (that's catalog ingest, a later part); just enough
    // for `model_io::load_vision_sidecar` to find the file the format
    // requires.
    std::fs::write(dir.join("preprocessor_config.json"), b"{}")?;

    Ok(vision)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma4_checkpoint::{read_vision_entries_quantized, Gemma4Quant};

    fn read(quantized: bool, quant: Option<&Gemma4Quant>) -> Result<crate::VisionRead, String> {
        let mut ts = vision_tower_tensors_with_out_hidden(128);
        if quantized {
            ts = quantize_tower_int4(ts);
        }
        // Companions are skipped by the classifier and read beside their
        // weight, so the bases are the non-companion names only.
        let bases: Vec<String> = ts
            .iter()
            .map(|t| t.name.clone())
            .filter(|n| !n.ends_with(".scales") && !n.ends_with(".biases"))
            .collect();
        let bases: Vec<&str> = bases.iter().map(String::as_str).collect();
        let blob = assemble_safetensors(&ts);
        let source = MemoryRangeSource::new(&blob);
        let header = parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
        let shards = Gemma4Shards::single(&header, &source);
        let mut vision = tiny_vision_config();
        vision.out_hidden_size = 128;
        read_vision_entries_quantized(&shards, &bases, &vision, quant).map_err(|e| e.to_string())
    }

    /// The dequantized tower matches `compute::dequantize_int4_affine` on the
    /// same rows, byte for byte after the FP16 narrowing, and records the
    /// LOGICAL shape rather than the packed one.
    #[test]
    fn an_affine_tower_dequantizes_to_the_reference_bytes() {
        let quant = Gemma4Quant::default();
        let got = read(true, Some(&quant)).expect("a quantized tower ingests");
        let plain = read(false, None).expect("the F16 tower ingests");
        assert_eq!(got.blocks.experts.len(), plain.blocks.experts.len());
        assert_eq!(got.block_stride, plain.block_stride, "same FP16 layout");

        let source = vision_tower_tensors_with_out_hidden(128);
        let qkv = source
            .iter()
            .find(|t| t.name == format!("{VISION_PREFIX}blocks.3.attn.qkv.weight"))
            .unwrap();
        let (rows, cols) = (qkv.shape[0] as usize, qkv.shape[1] as usize);
        let mut want = Vec::new();
        for r in 0..rows {
            let row: Vec<f32> = qkv.bytes[r * cols * 2..(r + 1) * cols * 2]
                .chunks_exact(2)
                .map(|b| compute::f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            let q = compute::quantize_int4_affine(&row);
            for v in compute::dequantize_int4_affine(&q, cols) {
                want.extend_from_slice(&compute::f32_to_f16(v).to_le_bytes());
            }
        }
        let sub = got.blocks.experts[3]
            .sub_tensors
            .iter()
            .find(|s| s.role == "qkv_w")
            .unwrap();
        assert_eq!(sub.shape, vec![rows as u64, cols as u64]);
        assert_eq!(sub.bytes, want);

        // The F16 fc2 passes through untouched, as on the real tower.
        let fc2 = |r: &crate::VisionRead| {
            r.blocks.experts[3]
                .sub_tensors
                .iter()
                .find(|s| s.role == "fc2_w")
                .unwrap()
                .bytes
                .clone()
        };
        assert_eq!(fc2(&got), fc2(&plain));
    }

    /// Without a quant config a packed plane is refused by dtype, never read
    /// as FP16 bytes.
    #[test]
    fn a_packed_tower_without_a_quant_config_is_refused() {
        let err = read(true, None).err().expect("U32 must be refused");
        assert!(err.contains("U32"), "{err}");
    }

    /// A quant config that disagrees with the companions' group count is a
    /// shape refusal, not a silent misread.
    #[test]
    fn a_wrong_group_size_is_refused() {
        let quant = Gemma4Quant {
            group_size: 32,
            ..Default::default()
        };
        let err = read(true, Some(&quant))
            .err()
            .expect("group 32 must be refused");
        assert!(err.contains("expected BF16"), "{err}");
    }
}
