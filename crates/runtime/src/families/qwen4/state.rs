//! `RealQwen4State`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens a `qwen4_exp` install -- the wide (10240)
//! residual buffer and its hyper-connection scratch, the GDN recurrent
//! state, the PLE layer's own buffers and its mmap'd n-gram table, and
//! every MoE scratch buffer `families/qwen/state.rs` already established
//! the shape of.
//!
//! **THE WIDE RESIDUAL IS FAMILY-LOCAL, NOT `DecodeScratch::x`.** That
//! shared field is sized `hidden_size * MAX_PREFILL_BATCH` for every OTHER
//! family; widening it to `hidden_size * hc_count` here would move every
//! other family's footprint and disturb their frozen memory-oracle rows
//! for a buffer they never touch (`docs/QWEN4_PHASE0.md`'s Phase 3 plan,
//! `BatchedScratch`'s own precedent one struct over). `wide_x` is this
//! struct's own field instead; `DecodeScratch`'s hidden-width buffers
//! (`normed`, `q`, `attn_out`, `o`, the FFN quartet, `logits`) ARE reused,
//! because every one of them already operates at `H = 2560` -- the width
//! of a hyper-connection's `mixed` output, never the wide stream itself.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use model_io::{ArchConfig, NgramContext, NgramTableLayout, ResidentBuffer, ResidentIndex};

use crate::families::qwen4::layer_tensor;
use crate::real_forward::RealForwardError;
use crate::real_forward_types::MAX_PREFILL_BATCH;
use crate::real_forward_utils::entry;

/// `(taps - 1) * dilation` for PLE's depthwise conv: `kernel_size=4`,
/// `dilation=ngram_size=3` (`docs/QWEN4_PHASE0.md` item 4).
pub(crate) const PLE_CONV_HISTORY: usize = 9;

/// Opt-in first-GDN-layer input/output capture for comparing the runtime's
/// Q/K normalization against an independent implementation.
pub(crate) struct GdnNormCapture {
    path: PathBuf,
    pub(crate) buffer: gpu::MetalBuffer,
    pub(crate) layer: usize,
    qkv_width: usize,
    key_heads: u32,
    key_dim: u32,
    value_heads: u32,
    value_dim: u32,
    requested_position: Option<usize>,
    claimed: AtomicBool,
    captured_position: AtomicUsize,
    flushed: AtomicBool,
}

impl GdnNormCapture {
    fn from_env(
        context: &gpu::MetalContext,
        arch: &ArchConfig,
        shape: gpu::GdnShape,
    ) -> Option<Self> {
        let path = std::env::var_os("TURBOSPARK_QWEN4_GDN_NORM_CAPTURE")?;
        let layer = arch
            .full_attention_layer_mask
            .iter()
            .position(|&kind| kind == 2)?;
        let requested_position = match std::env::var("TURBOSPARK_QWEN4_GDN_NORM_CAPTURE_POSITION") {
            Ok(value) => match value.parse::<usize>() {
                Ok(position) => Some(position),
                Err(error) => {
                    eprintln!(
                        "[qwen4-gdn-capture] invalid capture position {value:?}: {error}; capture disabled"
                    );
                    return None;
                }
            },
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => {
                eprintln!("[qwen4-gdn-capture] cannot read capture position: {error}");
                return None;
            }
        };
        let qkv_width = shape.qkv_dim() as usize;
        Some(Self {
            path: PathBuf::from(path),
            buffer: context.new_output_buffer((qkv_width * 4) as u64),
            layer,
            qkv_width,
            key_heads: shape.num_k_heads,
            key_dim: shape.key_head_dim,
            value_heads: shape.num_v_heads,
            value_dim: shape.value_head_dim,
            requested_position,
            claimed: AtomicBool::new(false),
            captured_position: AtomicUsize::new(usize::MAX),
            flushed: AtomicBool::new(false),
        })
    }

    pub(crate) fn claim_layer(&self, layer: usize, position: usize) -> bool {
        if layer != self.layer
            || self
                .requested_position
                .is_some_and(|requested| requested != position)
        {
            return false;
        }
        if self
            .claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        self.captured_position.store(position, Ordering::Release);
        true
    }

    fn flush_if_captured(&self) {
        if !self.claimed.load(Ordering::Acquire) || self.flushed.swap(true, Ordering::AcqRel) {
            return;
        }

        let data_path = self.path.with_extension("f16");
        let values = gpu::read_buffer_f16(&self.buffer, 0, self.qkv_width * 2);
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let data_name = data_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let header = format!(
            "{{\n  \"format\": \"qwen4_gdn_norm_capture_v1\",\n  \"layer\": {},\n  \
             \"position\": {},\n  \"key_heads\": {},\n  \"key_dim\": {},\n  \"value_heads\": {},\n  \
             \"value_dim\": {},\n  \"qkv_width\": {},\n  \"dtype\": \"float16\",\n  \
             \"layout\": \"[before:q,k,v][after:q,k,v]\",\n  \
             \"reference_epsilon\": 1e-6,\n  \"mean_space_epsilon\": {:.9e},\n  \
             \"data\": {}\n}}\n",
            self.layer,
            self.captured_position.load(Ordering::Acquire),
            self.key_heads,
            self.key_dim,
            self.value_heads,
            self.value_dim,
            self.qkv_width,
            1e-6 / self.key_dim as f32,
            json_string(&data_name),
        );
        let write_result = (|| -> std::io::Result<()> {
            if let Some(parent) = self
                .path
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&data_path, bytes)?;
            std::fs::write(&self.path, header)
        })();
        match write_result {
            Ok(()) => eprintln!(
                "[qwen4-gdn-capture] wrote {} and {} for layer {}",
                self.path.display(),
                data_path.display(),
                self.layer
            ),
            Err(error) => eprintln!(
                "[qwen4-gdn-capture] failed writing {}: {error}",
                self.path.display()
            ),
        }
    }
}

/// Opt-in residual boundary capture for comparing the Qwen4 layer flow with
/// SlotStream's independent layer reference.
///
/// The capture stores each requested layer's input (after PLE), attention
/// join, and MoE join. The default is the first two layers, matching
/// SlotStream's current_backend_reference.py --kind layers harness.
#[repr(usize)]
pub(crate) enum LayerBoundaryStage {
    InputAfterPle = 0,
    AfterAttentionJoin = 1,
    AfterMoeJoin = 2,
}

pub(crate) struct LayerBoundaryCapture {
    path: PathBuf,
    pub(crate) buffer: gpu::MetalBuffer,
    layers: usize,
    wide_dim: usize,
    requested_position: Option<usize>,
    claimed: AtomicBool,
    captured_position: AtomicUsize,
    flushed: AtomicBool,
}

impl LayerBoundaryCapture {
    fn from_env(context: &gpu::MetalContext, arch: &ArchConfig, wide_dim: usize) -> Option<Self> {
        let path = std::env::var_os("TURBOSPARK_QWEN4_LAYER_CAPTURE")?;
        let total_layers = usize::try_from(arch.num_layers).ok()?;
        let layers = match std::env::var("TURBOSPARK_QWEN4_LAYER_CAPTURE_LAYERS") {
            Ok(value) => match value.parse::<usize>() {
                Ok(layers) if layers > 0 && layers <= total_layers => layers,
                Ok(layers) => {
                    eprintln!(
                        "[qwen4-layer-capture] layer count {layers} is outside 1..={total_layers}; capture disabled"
                    );
                    return None;
                }
                Err(error) => {
                    eprintln!(
                        "[qwen4-layer-capture] invalid layer count {value:?}: {error}; capture disabled"
                    );
                    return None;
                }
            },
            Err(std::env::VarError::NotPresent) => total_layers.min(2),
            Err(error) => {
                eprintln!("[qwen4-layer-capture] cannot read layer count: {error}");
                return None;
            }
        };
        if layers == 0 {
            eprintln!("[qwen4-layer-capture] model has no layers; capture disabled");
            return None;
        }
        let requested_position = match std::env::var("TURBOSPARK_QWEN4_LAYER_CAPTURE_POSITION") {
            Ok(value) => match value.parse::<usize>() {
                Ok(position) => Some(position),
                Err(error) => {
                    eprintln!(
                            "[qwen4-layer-capture] invalid capture position {value:?}: {error}; capture disabled"
                        );
                    return None;
                }
            },
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => {
                eprintln!("[qwen4-layer-capture] cannot read capture position: {error}");
                return None;
            }
        };
        let byte_len = layers
            .checked_mul(3)
            .and_then(|count| count.checked_mul(wide_dim))
            .and_then(|count| count.checked_mul(2));
        let Some(byte_len) = byte_len else {
            eprintln!("[qwen4-layer-capture] capture dimensions overflow; capture disabled");
            return None;
        };
        if wide_dim > u32::MAX as usize {
            eprintln!("[qwen4-layer-capture] residual width exceeds the GPU copy limit");
            return None;
        }
        Some(Self {
            path: PathBuf::from(path),
            buffer: context.new_output_buffer(byte_len as u64),
            layers,
            wide_dim,
            requested_position,
            claimed: AtomicBool::new(false),
            captured_position: AtomicUsize::new(usize::MAX),
            flushed: AtomicBool::new(false),
        })
    }

    pub(crate) fn claim_position(&self, position: usize) -> bool {
        if self
            .requested_position
            .is_some_and(|requested| requested != position)
        {
            return false;
        }
        if self
            .claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        self.captured_position.store(position, Ordering::Release);
        true
    }

    /// Stage order: layer input after PLE, attention join, MoE join.
    pub(crate) fn encode_stage(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        layer: usize,
        stage: LayerBoundaryStage,
        source: (&gpu::MetalBuffer, u64),
    ) -> Result<(), gpu::GpuError> {
        if layer >= self.layers {
            return Ok(());
        }
        let element_offset = (stage as usize)
            .checked_mul(self.layers)
            .and_then(|offset| offset.checked_add(layer))
            .and_then(|offset| offset.checked_mul(self.wide_dim))
            .expect("Qwen4 layer capture offset overflow");
        gpu::encode_dflash_copy_rows(
            context,
            pass,
            source,
            (&self.buffer, (element_offset * 2) as u64),
            1,
            self.wide_dim as u32,
            self.wide_dim as u32,
        )
    }

    fn flush_if_captured(&self) {
        if !self.claimed.load(Ordering::Acquire) || self.flushed.swap(true, Ordering::AcqRel) {
            return;
        }
        let data_path = self.path.with_extension("f16");
        let value_count = self.layers * 3 * self.wide_dim;
        let values = gpu::read_buffer_f16(&self.buffer, 0, value_count);
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let data_name = data_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let header = format!(
            "{{\n  \"format\": \"qwen4_layer_boundary_capture_v1\",\n  \
             \"position\": {},\n  \"layers\": {},\n  \"wide_dim\": {},\n  \
             \"dtype\": \"float16\",\n  \
             \"layout\": \"[stage][layer][wide_dim]\",\n  \
             \"stages\": [\"layer_input_after_ple\", \"after_attention_join\", \"after_moe_join\"],\n  \
             \"reference\": \"SlotStream current_backend_reference.py --kind layers\",\n  \
             \"data\": {}\n}}\n",
            self.captured_position.load(Ordering::Acquire),
            self.layers,
            self.wide_dim,
            json_string(&data_name),
        );
        let write_result = (|| -> std::io::Result<()> {
            if let Some(parent) = self
                .path
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&data_path, bytes)?;
            std::fs::write(&self.path, header)
        })();
        match write_result {
            Ok(()) => eprintln!(
                "[qwen4-layer-capture] wrote {} and {} for {} layer(s)",
                self.path.display(),
                data_path.display(),
                self.layers
            ),
            Err(error) => eprintln!(
                "[qwen4-layer-capture] failed writing {}: {error}",
                self.path.display()
            ),
        }
    }
}

/// Opt-in first-attention-block capture for comparing the Qwen4 attention
/// hyper-connection and branch output against llama.cpp's eval callback.
#[repr(usize)]
pub(crate) enum AttentionIntermediateStage {
    HcNormed = 0,
    HcGate = 1,
    HcMixed = 2,
    BranchOutput = 3,
    InjectGate = 4,
    WideAfterJoin = 5,
}

pub(crate) struct AttentionIntermediateCapture {
    path: PathBuf,
    pub(crate) buffer: gpu::MetalBuffer,
    layer: usize,
    wide_dim: usize,
    hidden_dim: usize,
    hc_count: usize,
    requested_position: Option<usize>,
    claimed: AtomicBool,
    captured_position: AtomicUsize,
    flushed: AtomicBool,
}

impl AttentionIntermediateCapture {
    fn from_env(
        context: &gpu::MetalContext,
        arch: &ArchConfig,
        wide_dim: usize,
        hidden_dim: usize,
        hc_count: usize,
    ) -> Option<Self> {
        let path = std::env::var_os("TURBOSPARK_QWEN4_ATTENTION_CAPTURE")?;
        let total_layers = usize::try_from(arch.num_layers).ok()?;
        let layer = match std::env::var("TURBOSPARK_QWEN4_ATTENTION_CAPTURE_LAYER") {
            Ok(value) => match value.parse::<usize>() {
                Ok(layer) if layer < total_layers => layer,
                Ok(layer) => {
                    eprintln!(
                        "[qwen4-attention-capture] layer {layer} is outside 0..{total_layers}; capture disabled"
                    );
                    return None;
                }
                Err(error) => {
                    eprintln!(
                        "[qwen4-attention-capture] invalid layer {value:?}: {error}; capture disabled"
                    );
                    return None;
                }
            },
            Err(std::env::VarError::NotPresent) => 0,
            Err(error) => {
                eprintln!("[qwen4-attention-capture] cannot read layer: {error}");
                return None;
            }
        };
        let requested_position = match std::env::var("TURBOSPARK_QWEN4_ATTENTION_CAPTURE_POSITION")
        {
            Ok(value) => match value.parse::<usize>() {
                Ok(position) => Some(position),
                Err(error) => {
                    eprintln!(
                        "[qwen4-attention-capture] invalid position {value:?}: {error}; capture disabled"
                    );
                    return None;
                }
            },
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => {
                eprintln!("[qwen4-attention-capture] cannot read position: {error}");
                return None;
            }
        };
        if wide_dim > u32::MAX as usize || hidden_dim > wide_dim || hc_count > wide_dim {
            eprintln!("[qwen4-attention-capture] invalid capture dimensions; capture disabled");
            return None;
        }
        let byte_len = wide_dim
            .checked_mul(6)
            .and_then(|count| count.checked_mul(2))?;
        let buffer = context.new_output_buffer(byte_len as u64);
        gpu::write_buffer_bytes(&buffer, 0, &vec![0u8; byte_len]);
        Some(Self {
            path: PathBuf::from(path),
            buffer,
            layer,
            wide_dim,
            hidden_dim,
            hc_count,
            requested_position,
            claimed: AtomicBool::new(false),
            captured_position: AtomicUsize::new(usize::MAX),
            flushed: AtomicBool::new(false),
        })
    }

    pub(crate) fn claim_position(&self, position: usize) -> bool {
        if self
            .requested_position
            .is_some_and(|requested| requested != position)
        {
            return false;
        }
        if self
            .claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        self.captured_position.store(position, Ordering::Release);
        true
    }

    pub(crate) fn encode_stage(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        layer: usize,
        stage: AttentionIntermediateStage,
        source: (&gpu::MetalBuffer, u64),
    ) -> Result<(), gpu::GpuError> {
        if layer != self.layer || !self.claimed.load(Ordering::Acquire) {
            return Ok(());
        }
        let (stage_index, width) = match stage {
            AttentionIntermediateStage::HcNormed => (0, self.wide_dim),
            AttentionIntermediateStage::HcGate => (1, self.wide_dim),
            AttentionIntermediateStage::HcMixed => (2, self.hidden_dim),
            AttentionIntermediateStage::BranchOutput => (3, self.hidden_dim),
            AttentionIntermediateStage::InjectGate => (4, self.hc_count),
            AttentionIntermediateStage::WideAfterJoin => (5, self.wide_dim),
        };
        gpu::encode_dflash_copy_rows(
            context,
            pass,
            source,
            (&self.buffer, (stage_index * self.wide_dim * 2) as u64),
            1,
            width as u32,
            width as u32,
        )
    }

    fn flush_if_captured(&self) {
        if !self.claimed.load(Ordering::Acquire) || self.flushed.swap(true, Ordering::AcqRel) {
            return;
        }
        let data_path = self.path.with_extension("f16");
        let values = gpu::read_buffer_f16(&self.buffer, 0, self.wide_dim * 6);
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let data_name = data_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let header = format!(
            "{{\n  \"format\": \"qwen4_attention_intermediate_capture_v1\",\n  \
             \"layer\": {},\n  \"position\": {},\n  \"wide_dim\": {},\n  \
             \"hidden_dim\": {},\n  \"hc_count\": {},\n  \"dtype\": \"float16\",\n  \
             \"layout\": \"[stage][wide_slot]\",\n  \
             \"stages\": [\n    {{\"name\": \"hc_normed\", \"width\": {}}},\n    \
             {{\"name\": \"hc_gate\", \"width\": {}}},\n    \
             {{\"name\": \"hc_mixed\", \"width\": {}}},\n    \
             {{\"name\": \"branch_output\", \"width\": {}}},\n    \
             {{\"name\": \"inject_gate\", \"width\": {}}},\n    \
             {{\"name\": \"wide_after_join\", \"width\": {}}}\n  ],\n  \
             \"reference\": \"llama.cpp eval callback and SlotStream layer harness\",\n  \
             \"data\": {}\n}}\n",
            self.layer,
            self.captured_position.load(Ordering::Acquire),
            self.wide_dim,
            self.hidden_dim,
            self.hc_count,
            self.wide_dim,
            self.wide_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.hc_count,
            self.wide_dim,
            json_string(&data_name),
        );
        let write_result = (|| -> std::io::Result<()> {
            if let Some(parent) = self
                .path
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&data_path, bytes)?;
            std::fs::write(&self.path, header)
        })();
        match write_result {
            Ok(()) => eprintln!(
                "[qwen4-attention-capture] wrote {} and {} for layer {}",
                self.path.display(),
                data_path.display(),
                self.layer
            ),
            Err(error) => eprintln!(
                "[qwen4-attention-capture] failed writing {}: {error}",
                self.path.display()
            ),
        }
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if character <= '\u{1f}' => {
                out.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => out.push(character),
        }
    }
    out.push('"');
    out
}

/// Per-open `qwen4_exp` decode state.
pub(crate) struct RealQwen4State {
    pub(crate) shape: gpu::GdnShape,
    pub(crate) gdn: gpu::GdnStateManager,
    pub(crate) rotary_dim: u32,
    pub(crate) hc_count: usize,
    pub(crate) hc_lowrank: usize,
    /// Zero-based index of the one PLE layer (`arch.ple.layer_indices()`'s
    /// single entry, converted from the checkpoint's one-based spelling).
    pub(crate) ple_layer: usize,

    /// The residual stream, `hidden_size * hc_count` wide, holding
    /// [`MAX_PREFILL_BATCH`] rows. See the module doc for why this is not
    /// `DecodeScratch::x` -- it is this struct's own analogue of that field,
    /// widened for the identical reason (the residual crosses layers, so a
    /// chunked prefill driver needs every token of a micro-batch to hold its
    /// own row at once). The sequential decode path uses row 0 only.
    pub(crate) wide_x: gpu::MetalBuffer,
    /// `hc_norm`'s output, wide -- shared by both per-layer hyper-connection
    /// calls and the final mixer, which never overlap within one dispatch
    /// sequence (commit-order execution, `crates/gpu` Gotcha 8). Single-row:
    /// written and read back within one token's own dispatch sequence,
    /// never crossing a command-buffer commit to a different token.
    pub(crate) hc_normed: gpu::MetalBuffer,
    /// `input_mix_weight_down`'s output, `hc_lowrank` wide. Single-row, same
    /// reason as `hc_normed`.
    pub(crate) hc_low: gpu::MetalBuffer,
    /// `input_mix_weight_up`'s output, wide -- this is `w` in `hc_mix`'s
    /// contract. Single-row, same reason as `hc_normed`.
    pub(crate) hc_up: gpu::MetalBuffer,
    /// `block_inject_weight`'s output, `hc_count` wide, holding
    /// [`MAX_PREFILL_BATCH`] rows. **Not single-row like its hyper-connection
    /// siblings above**: `attn_hc`'s inject gate is read by the FIRST
    /// `encode_hc_inject_add` immediately afterward, within the same token's
    /// dispatch sequence (safe unwidened), but `mlp_hc`'s inject gate is read
    /// by the SECOND one only after the layer's `cb1` has committed and a
    /// new `"routed cb"` pass has begun for the per-token routed loop -- one
    /// commit later than the write, in a chunked driver where the layer's
    /// whole `cb1` covers all of a micro-batch's tokens before that commit.
    /// Widened so every token's mlp-side inject gate survives to be read by
    /// its own iteration of the routed loop. The sequential path uses row 0
    /// only for both sublayers.
    pub(crate) hc_inject: gpu::MetalBuffer,

    /// `[2 * num_heads * full_head_dim]`: QSA's packed query/gate rows.
    pub(crate) q_packed: gpu::MetalBuffer,
    pub(crate) attn_gate: gpu::MetalBuffer,

    /// The QSA indexer's persistent state per QSA layer: the raw key
    /// history and the incrementally pooled/normed/roped block cache
    /// (`docs/QWEN4_PHASE0.md` section 5). Positions are the KV cache's,
    /// never its own (its module doc says why).
    pub(crate) qsa: gpu::QsaIndexerCacheManager,
    /// `compressed_attention.index_n_heads` / `index_head_dim` /
    /// `csa_compress_rate`, and `index_top_k` in BLOCKS.
    pub(crate) idx_heads: u32,
    pub(crate) idx_head_dim: u32,
    pub(crate) idx_compress: u32,
    pub(crate) idx_block_topk: usize,
    /// `index_qk_proj`'s output, `[(idx_heads + 1) * idx_head_dim]`: the
    /// query heads first, the one raw key head last.
    pub(crate) idx_qk: gpu::MetalBuffer,
    /// One FP32 score per complete block, `[max_context / idx_compress]`.
    pub(crate) qsa_scores: gpu::MetalBuffer,
    /// The selected positions, `[idx_block_topk * idx_compress + idx_compress]`
    /// `u32`s: the most block selection ever keeps (every chosen block plus
    /// the ragged tail).
    pub(crate) qsa_positions: gpu::MetalBuffer,
    /// DIAGNOSTIC: attend densely above budget as if the indexer selected
    /// everything. `TURBOSPARK_QSA_FORCE_DENSE=1` at open, or
    /// `RealForwardRunner::set_qsa_force_dense`. The only quantitative
    /// instrument for the sparse path on a real install (no reference
    /// engine for this checkpoint fits this machine): the KL between the
    /// two arms past 2,051 tokens.
    pub(crate) qsa_force_dense: bool,
    pub(crate) gdn_qkv_raw: gpu::MetalBuffer,
    pub(crate) gdn_conv_out: gpu::MetalBuffer,
    pub(crate) gdn_norm_capture: Option<GdnNormCapture>,
    pub(crate) layer_boundary_capture: Option<LayerBoundaryCapture>,
    pub(crate) attention_intermediate_capture: Option<AttentionIntermediateCapture>,
    pub(crate) gdn_z: gpu::MetalBuffer,
    pub(crate) gdn_a: gpu::MetalBuffer,
    pub(crate) gdn_b: gpu::MetalBuffer,
    pub(crate) gdn_y: gpu::MetalBuffer,
    pub(crate) gdn_out: gpu::MetalBuffer,

    /// BF16 `[hidden]` of ones -- the router kernel's unused per-element
    /// scale, matching `families/qwen/state.rs`'s `router_ones` exactly
    /// (this checkpoint has no `router.scale` either).
    pub(crate) router_ones: gpu::MetalBuffer,
    pub(crate) per_expert_ones: Vec<f32>,
    /// Holding [`MAX_PREFILL_BATCH`] rows: the host reads back ALL of a
    /// micro-batch's tokens' router logits in one go, after the layer's
    /// `cb1` (which encodes every token's router GEMV) commits and waits
    /// once -- exactly the property `RealGemmaState::router_logits_f32`
    /// widens for. The sequential path writes and reads row 0 only.
    pub(crate) router_logits_f32: gpu::MetalBuffer,
    /// `mlp_hc`'s `mixed` output (`hidden` wide), holding [`MAX_PREFILL_BATCH`]
    /// rows. **Used only by the chunked-prefill driver (`prefill.rs`); the
    /// sequential decode path (`produce.rs`) is unchanged and continues to
    /// write `mlp_hc`'s output into `scratch.normed`**, which is safe there
    /// (only one token is ever in flight) and left alone rather than moved,
    /// per this repo's own rule that a numerics-verified path earns no
    /// drive-by edits.
    ///
    /// The reason the CHUNKED driver cannot reuse `scratch.normed` the way
    /// `attn_hc` does: `encode_moe_layer` reads this value one command-buffer
    /// commit later than it was written (the layer's `cb1` commits, then a
    /// new `"routed cb"` pass begins per token), so a chunked driver that
    /// encodes a whole micro-batch's `cb1` before committing would have every
    /// earlier token's value overwritten by the last token's by the time the
    /// routed loop runs, were this single-row. Never widen `scratch.normed`
    /// itself for this either: it is shared `DecodeScratch` state and
    /// widening it would move every OTHER family's memory-oracle peak for a
    /// buffer they never touch (matching `wide_x`'s own module-doc argument,
    /// one field over).
    pub(crate) moe_x: gpu::MetalBuffer,
    /// The gated shared-expert output. SlotStream's Qwen4 reference adds it
    /// after the routed sum has been rounded to the model dtype. Phase 2
    /// therefore reads a separate zero vector; this value is added to its
    /// output afterward. Single-row: consumed by this token's own routed
    /// iteration.
    pub(crate) h1: gpu::MetalBuffer,
    /// Shared + routed, moe(mixed)'s final output. The caller injects this
    /// into the wide stream after the routed and shared branches have joined.
    /// Single-row, consumed by this token's own hyper-connection call.
    pub(crate) h2: gpu::MetalBuffer,
    /// Zero residual used to make phase 2 round the routed sum before the
    /// separate gated shared-expert addition, matching SlotStream's Qwen4
    /// reference. Immutable after state construction.
    pub(crate) moe_zero: gpu::MetalBuffer,
    pub(crate) shared_gate_logit: gpu::MetalBuffer,

    /// PLE's concatenated n-gram lookup, `[hidden]` -- host dequant, GPU
    /// upload, holding [`MAX_PREFILL_BATCH`] rows. **Widened for
    /// correctness, not throughput, and the sharpest instance of this
    /// pattern in the family**: the upload (`gpu::write_buffer_bytes`) is a
    /// HOST write that happens the instant `encode_ple_layer` runs, not a
    /// GPU dispatch queued for later, so it does not respect command-buffer
    /// commit order at all -- a chunked micro-batch that stayed single-row
    /// here would have every token's `key_proj`/`value_proj` GEMV read
    /// whichever token's embedding was written LAST, since none of those
    /// GEMVs execute until the whole pass commits, by which point every
    /// token's host write has already landed. The sequential path uses row
    /// 0 only.
    pub(crate) ngram_emb: gpu::MetalBuffer,
    /// `norm_key(key_proj(emb))`, wide.
    pub(crate) ple_key: gpu::MetalBuffer,
    /// `value_proj(emb)`, `[hidden]` -- shared across all `hc_count`
    /// streams by `ple_gate`'s own broadcast.
    pub(crate) ple_value: gpu::MetalBuffer,
    /// `norm_query(hidden)`, wide -- reads the WIDE residual directly,
    /// never `mixed`.
    pub(crate) ple_query: gpu::MetalBuffer,
    /// `ple_gate`'s output (`gv.flatten()`), wide.
    pub(crate) ple_gv: gpu::MetalBuffer,
    /// `norm_conv(gv.flatten())`, wide -- the dilated conv's input.
    pub(crate) ple_conv_normed: gpu::MetalBuffer,
    /// `silu(dilated_conv(...))`, wide -- added to `ple_gv` (NOT to
    /// `ple_conv_normed`) to produce PLE's final output, per
    /// `docs/QWEN4_PHASE0.md` item 4: the conv reads the NORMED gated
    /// value and its output joins the UN-normed one.
    pub(crate) ple_conv_out: gpu::MetalBuffer,
    /// The dilated conv's recurrent state: [`PLE_CONV_HISTORY`] rows of
    /// `hidden * hc_count` RAW (pre-conv) values, FP16. Standalone rather
    /// than part of [`RealQwen4State::gdn`]'s conv tails: `GdnStateManager`
    /// sizes its tails from `LinearAttentionConfig`, at the GDN chain's
    /// `qkv_dim` width, and this is a differently-shaped buffer belonging
    /// to exactly one layer.
    pub(crate) ple_conv_tail: gpu::MetalBuffer,
    /// The n-gram table's own addressing (multipliers, per-head vocab
    /// sizes and offsets), read from `ngram_table/header.json`.
    pub(crate) ngram_layout: NgramTableLayout,
    /// The WHOLE table, mmap'd once at open. Demand-paged, so mapping the
    /// full ~32 GB costs no resident memory until a row is actually
    /// touched -- `model_io::ngram_table`'s module doc has the measured
    /// argument for `mmap` over the `pread` streamer at this record size.
    pub(crate) ngram_table: ResidentBuffer,
    /// The decode-time EOS-boundary-aware n-gram context (recurrent state
    /// slot 2 in `docs/QWEN4_PHASE0.md` item 4's table). Reset alongside
    /// the GDN state.
    pub(crate) ngram_context: NgramContext,
    /// `ngram_size - 1`: `NgramContext` does not expose its own length, and
    /// re-deriving it from another stored constant (`PLE_CONV_HISTORY` is
    /// `(taps - 1) * dilation`, an unrelated product that only coincides in
    /// one factor) is exactly the kind of clever-looking arithmetic that
    /// reads correct and is not; a plain stored field has no such trap.
    pub(crate) ngram_context_len: usize,
    /// `arch.ple.eos_token_id`, kept for [`RealQwen4State::reset`] and for
    /// feeding [`model_io::NgramContext::step`] each decode step.
    pub(crate) eos_token_id: i64,
}

impl RealQwen4State {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
        install_dir: &Path,
        max_context: usize,
        expert_cache_slots: usize,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));

        // The QSA indexer's shape (`docs/QWEN4_PHASE0.md` section 5). Below
        // `index_budget` block selection is a no-op and every QSA layer is
        // exactly dense causal attention; above it `attn.rs` scores the
        // pooled blocks and attends over the selected ones. The refusal
        // that used to sit here ("--max-context must not exceed the
        // budget") is gone with the wiring; what replaces it is a check
        // that the indexer is one this flow can run.
        let ca = &arch.compressed_attention;
        if ca.index_budget <= 0
            || ca.index_n_heads <= 0
            || ca.index_head_dim <= 0
            || ca.csa_compress_rate <= 0
            || ca.index_top_k <= 0
        {
            return unsupported(format!(
                "qwen4_exp's QSA indexer needs positive index_budget ({}), index_n_heads ({}), \
                 index_head_dim ({}), csa_compress_rate ({}) and index_top_k ({})",
                ca.index_budget,
                ca.index_n_heads,
                ca.index_head_dim,
                ca.csa_compress_rate,
                ca.index_top_k
            ));
        }
        if ca.index_kv_heads != 1 {
            return unsupported(format!(
                "qwen4_exp's QSA indexer shares ONE pooled key head across its query heads; \
                 index_kv_heads {} is not a shape this flow scores",
                ca.index_kv_heads
            ));
        }
        if ca.index_top_k.checked_mul(ca.csa_compress_rate) != Some(ca.index_budget) {
            return unsupported(format!(
                "qwen4_exp's index_top_k ({}) x csa_compress_rate ({}) must equal index_budget \
                 ({}): top_k counts BLOCKS on this architecture",
                ca.index_top_k, ca.csa_compress_rate, ca.index_budget
            ));
        }

        // These manifest fields cross two narrower execution boundaries:
        // Metal kernels consume u32 dimensions, while cache and scratch
        // extents are host usize byte counts. Refuse values that cannot be
        // represented losslessly or whose complete allocation arithmetic
        // does not fit, rather than letting release-mode arithmetic wrap or
        // `as u32` silently truncate them.
        let idx_heads = u32::try_from(ca.index_n_heads).map_err(|_| {
            RealForwardError::Unsupported(format!(
                "qwen4_exp index_n_heads {} exceeds the u32 kernel limit",
                ca.index_n_heads
            ))
        })?;
        let idx_head_dim = u32::try_from(ca.index_head_dim).map_err(|_| {
            RealForwardError::Unsupported(format!(
                "qwen4_exp index_head_dim {} exceeds the u32 kernel limit",
                ca.index_head_dim
            ))
        })?;
        let idx_compress = u32::try_from(ca.csa_compress_rate).map_err(|_| {
            RealForwardError::Unsupported(format!(
                "qwen4_exp csa_compress_rate {} exceeds the u32 kernel limit",
                ca.csa_compress_rate
            ))
        })?;
        let idx_block_topk = usize::try_from(ca.index_top_k).map_err(|_| {
            RealForwardError::Unsupported(format!(
                "qwen4_exp index_top_k {} exceeds the host index limit",
                ca.index_top_k
            ))
        })?;
        let idx_projection_len_u32 = idx_heads
            .checked_add(1)
            .and_then(|heads| heads.checked_mul(idx_head_dim));
        let idx_projection_len = idx_projection_len_u32.map(|len| len as usize);
        let qsa_raw_bytes = max_context
            .checked_mul(idx_head_dim as usize)
            .and_then(|len| len.checked_mul(2));
        let qsa_pooled_bytes = (max_context / idx_compress as usize)
            .checked_mul(idx_head_dim as usize)
            .and_then(|len| len.checked_mul(2));
        let qsa_scores_bytes = (max_context / idx_compress as usize).max(1).checked_mul(4);
        let qsa_positions_bytes = idx_block_topk
            .checked_add(1)
            .and_then(|len| len.checked_mul(idx_compress as usize))
            .and_then(|len| len.checked_mul(4));
        if idx_projection_len.is_none()
            || qsa_raw_bytes.is_none()
            || qsa_pooled_bytes.is_none()
            || qsa_scores_bytes.is_none()
            || qsa_positions_bytes.is_none()
        {
            return unsupported(
                "qwen4_exp QSA indexer dimensions overflow host buffer arithmetic".to_string(),
            );
        }

        if arch.num_experts <= 0 || arch.top_k_experts <= 0 {
            return unsupported(format!(
                "qwen4_exp is MoE-only in this port: num_experts {}, top_k_experts {} must both \
                 be positive",
                arch.num_experts, arch.top_k_experts
            ));
        }
        if arch.top_k_experts as usize > gpu::MAX_STREAMED_EXPERTS {
            return unsupported(format!(
                "top_k {} exceeds the {}-slot MoE kernels",
                arch.top_k_experts,
                gpu::MAX_STREAMED_EXPERTS
            ));
        }
        // ONE TOKEN'S TOP-K MUST FIT THE CACHE OUTRIGHT, independent of
        // AGENTS.md Gotcha 64's `2 * top_k` pipelining margin: that margin
        // protects a PREVIOUS token's still-in-flight slots from an
        // overlapping plan, and below it `moe_prefill_pipeline::
        // routed_pipeline_banks` already degrades to `banks == 1`
        // (retire-before-plan, an empty `protect` set) rather than relying
        // on an open-time floor -- exactly the mechanism every other
        // chunked-capable family's driver relies on instead of raising this
        // check (none of gemma4, gpt-oss or llama has an equivalent `2 *
        // top_k` floor in their own `state.rs`). So this stays the outright
        // "does one token's routing fit the cache at all" backstop:
        // `top_k_experts` distinct experts must fit `expert_cache_slots`
        // slots at all, below which `ExpertCache::plan` cannot select this
        // layer's routing regardless of pipelining and aborts the process
        // rather than degrading.
        if expert_cache_slots < arch.top_k_experts as usize {
            return unsupported(format!(
                "qwen4_exp routes {} experts per token, which does not fit a \
                 {expert_cache_slots}-slot cache; raise --expert-cache-slots to at least {} \
                 (AGENTS.md Gotcha 64)",
                arch.top_k_experts, arch.top_k_experts
            ));
        }
        if !arch.shared_expert_gated || !arch.attn_output_gate || !arch.rope_neox_subdim {
            return unsupported(
                "the qwen4_exp flow needs sharedExpertGated + attnOutputGate + ropeNeoxSubdim"
                    .to_string(),
            );
        }
        if arch.ffn_sandwich_norms || arch.router_scaled || arch.embedding_scaled_by_sqrt_hidden {
            return unsupported(
                "the qwen4_exp flow has no sandwich norms, no router scale, and no \
                 sqrt(hidden) embedding scale"
                    .to_string(),
            );
        }
        if arch.final_logit_softcap != 0.0 {
            return unsupported("qwen4_exp has no final logit softcap".to_string());
        }
        if arch.tie_word_embeddings {
            return unsupported("qwen4_exp's lm_head is its own tensor, never tied".to_string());
        }
        if arch
            .full_attention_layer_mask
            .iter()
            .any(|&m| m != 1 && m != 2)
        {
            return unsupported(
                "this flow's layers are full attention (1, QSA-below-budget) or gated \
                 DeltaNet (2) only"
                    .to_string(),
            );
        }
        if !arch.has_linear_attention_layers() {
            return unsupported(
                "qwen4_exp with no linear layers is not one this flow can run".to_string(),
            );
        }
        if arch.hyper_connections.mult <= 1 {
            return unsupported(format!(
                "hyper_connections.mult {} must be at least 2; this flow's whole point is \
                 the wide multi-stream residual",
                arch.hyper_connections.mult
            ));
        }
        if arch.hyper_connections.lowrank <= 0 {
            return unsupported(format!(
                "hyper_connections.lowrank {} must be positive",
                arch.hyper_connections.lowrank
            ));
        }
        if !arch.ple.is_active() {
            return unsupported(
                "qwen4_exp with no active PLE table is not one this flow can run".to_string(),
            );
        }
        let ple_layers = arch.ple.layer_indices();
        let [ple_layer_signed] = ple_layers[..] else {
            return unsupported(format!(
                "arch.ple declares {} PLE layers; this flow places exactly one",
                ple_layers.len()
            ));
        };
        if ple_layer_signed < 0 || ple_layer_signed >= arch.num_layers {
            return unsupported(format!(
                "PLE layer index {ple_layer_signed} is outside 0..{}",
                arch.num_layers
            ));
        }
        let ple_layer = ple_layer_signed as usize;
        if arch.layer_is_linear(ple_layer) {
            // Confirmed by the checkpoint (`docs/QWEN4_PHASE0.md` section
            // 0 finding 1: `ple_layer_ids: [2]` resolves to layer index 1,
            // which this baseline's `full_attention_layer_mask` marks
            // linear/GDN). Stated as an invariant the flow relies on
            // rather than assumed silently: PLE's own dataflow runs
            // BEFORE the layer's GDN-or-attention branch either way, so a
            // PLE layer that happened to be a QSA one would still be
            // structurally fine -- this check exists so a future config
            // that moves PLE onto a QSA layer is a loud refusal instead of
            // an unexercised code path.
        }
        if arch.ple.eos_token_id == 0 {
            return unsupported(
                "arch.ple.eos_token_id is unset; PLE's n-gram context resets at EOS \
                 boundaries and needs one (docs/QWEN4_PHASE0.md item 4)"
                    .to_string(),
            );
        }
        if !arch.linear_attention.output_gate_sigmoid {
            return unsupported(
                "qwen4_exp's GDN gated norm is the sigmoid variant \
                 (docs/QWEN4_PHASE0.md item 0 finding 4); outputGateSigmoid must be true"
                    .to_string(),
            );
        }

        let shape = gpu::GdnShape {
            num_k_heads: arch.linear_attention.num_k_heads as u32,
            num_v_heads: arch.linear_attention.num_v_heads as u32,
            key_head_dim: arch.linear_attention.key_head_dim as u32,
            value_head_dim: arch.linear_attention.value_head_dim as u32,
            conv_kernel_size: arch.linear_attention.conv_kernel_size as u32,
        };
        shape.validate().map_err(RealForwardError::Gpu)?;

        let head_dim = arch.full_head_dim;
        let rotary_dim = (head_dim as f64 * arch.partial_rotary_factor).round() as i64;
        if rotary_dim <= 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
            return unsupported(format!(
                "rotary_dim {rotary_dim} (full_head_dim {head_dim} x partial_rotary_factor \
                 {}) must be positive, even, and at most full_head_dim",
                arch.partial_rotary_factor
            ));
        }
        // The indexer rotates with the TRUNK attention's own RoPE object
        // (`crates/compute/src/qsa_indexer.rs`'s module doc: identical
        // Python object in the reference), so the trunk's `rotary_dim` has
        // to fit inside the indexer's head.
        if rotary_dim > ca.index_head_dim {
            return unsupported(format!(
                "rotary_dim {rotary_dim} exceeds the QSA indexer's head dim {}; the indexer \
                 shares the trunk attention's RoPE and cannot rotate more than it holds",
                ca.index_head_dim
            ));
        }

        // Fail at open, not at token 1: the top-level tensors, plus one
        // representative layer of each kind (attn_hyper_connection /
        // mlp_hyper_connection are on EVERY layer, so those are probed for
        // all of them).
        let hidden = arch.hidden_size as usize;
        for name in [
            "language_model.model.embed_tokens.weight".to_string(),
            "language_model.lm_head.weight".to_string(),
            "language_model.model.hyper_connection_mixer.hc_norm.weight".to_string(),
            "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight".to_string(),
            "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight".to_string(),
        ] {
            entry(index, &name)?;
        }
        for layer in 0..arch.num_layers as usize {
            for hc in ["attn_hyper_connection", "mlp_hyper_connection"] {
                for suffix in [
                    "hc_norm.weight",
                    "input_mix_weight_down.weight",
                    "input_mix_weight_up.weight",
                    "block_inject_weight.weight",
                ] {
                    entry(index, &layer_tensor(layer, &format!("{hc}.{suffix}")))?;
                }
            }
            entry(index, &layer_tensor(layer, "mlp.gate.weight"))?;
            entry(index, &layer_tensor(layer, "mlp.shared_expert_gate.weight"))?;
            entry(
                index,
                &layer_tensor(layer, "mlp.shared_expert.gate_proj.weight"),
            )?;
            if arch.layer_is_linear(layer) {
                for suffix in [
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.conv1d.weight",
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                ] {
                    entry(index, &layer_tensor(layer, suffix))?;
                }
            } else {
                for suffix in [
                    "self_attn.q_proj.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.indexer.index_qk_proj.weight",
                    "self_attn.indexer.q_layernorm.weight",
                    "self_attn.indexer.k_layernorm.weight",
                ] {
                    entry(index, &layer_tensor(layer, suffix))?;
                }
            }
            if layer == ple_layer {
                for suffix in [
                    "ple.key_proj.weight",
                    "ple.value_proj.weight",
                    "ple.conv1d.weight",
                    "ple.norm_key.weight",
                    "ple.norm_query.weight",
                    "ple.norm_conv.weight",
                ] {
                    entry(index, &layer_tensor(layer, suffix))?;
                }
            }
        }
        let _ = weights;

        let ngram_layout = model_io::load_ngram_table_layout(install_dir)
            .map_err(RealForwardError::Model)?
            .ok_or_else(|| {
                RealForwardError::Unsupported(
                    "qwen4_exp needs an ngram_table/ store; this install has none".to_string(),
                )
            })?;
        let expected_head_dim = arch.ple.head_dim() as u64;
        if ngram_layout.head_dim != expected_head_dim {
            return unsupported(format!(
                "ngram_table/header.json declares head_dim {}, but arch.ple resolves \
                 {expected_head_dim} ({}head/{}heads)",
                ngram_layout.head_dim,
                arch.ple.ple_embed_dim,
                arch.ple.ngram_heads()
            ));
        }
        let ngram_table = ResidentBuffer::map(
            &install_dir
                .join(model_io::NGRAM_TABLE_DIR)
                .join(model_io::NGRAM_TABLE_BLOB),
            0,
            ngram_layout.blob_bytes().ok_or_else(|| {
                RealForwardError::Unsupported("ngram_table row/record count overflows".to_string())
            })?,
        )
        .map_err(RealForwardError::Model)?;

        let ones: Vec<u8> = (0..hidden)
            .flat_map(|_| 0x3F80u16.to_le_bytes()) // BF16 1.0
            .collect();
        let router_ones = context.new_output_buffer(ones.len() as u64);
        gpu::write_buffer_bytes(&router_ones, 0, &ones);

        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        let moe_zero = halfs(hidden);
        gpu::write_buffer_bytes(&moe_zero, 0, &vec![0u8; hidden * 2]);
        let hc_count = arch.hyper_connections.mult as usize;
        let hc_lowrank = arch.hyper_connections.lowrank as usize;
        let wide_dim = hidden * hc_count;
        let num_experts = arch.num_experts as usize;
        let q_dim = (arch.num_heads * head_dim) as usize;
        let qkv_dim = shape.qkv_dim() as usize;
        let value_dim = shape.value_dim() as usize;
        let v_heads = shape.num_v_heads as usize;

        let ngram_context_len = (arch.ple.ngram_size - 1).max(0) as usize;
        let idx_projection_len = idx_projection_len.expect("QSA projection length checked above");
        let qsa = gpu::QsaIndexerCacheManager::new(context.device(), arch, max_context);
        let qsa_scores =
            context.new_output_buffer(qsa_scores_bytes.expect("QSA score length checked") as u64);
        let qsa_positions = context
            .new_output_buffer(qsa_positions_bytes.expect("QSA positions length checked") as u64);
        let qsa_force_dense = std::env::var("TURBOSPARK_QSA_FORCE_DENSE").as_deref() == Ok("1");
        let ple_conv_tail = halfs(PLE_CONV_HISTORY * wide_dim);
        let gdn_norm_capture = GdnNormCapture::from_env(context, arch, shape);
        let layer_boundary_capture = LayerBoundaryCapture::from_env(context, arch, wide_dim);
        let attention_intermediate_capture =
            AttentionIntermediateCapture::from_env(context, arch, wide_dim, hidden, hc_count);
        gpu::write_buffer_bytes(
            &ple_conv_tail,
            0,
            &vec![0u8; PLE_CONV_HISTORY * wide_dim * 2],
        );

        Ok(Self {
            gdn: gpu::GdnStateManager::new(context.device(), arch),
            shape,
            rotary_dim: rotary_dim as u32,
            hc_count,
            hc_lowrank,
            ple_layer,

            wide_x: halfs(wide_dim * MAX_PREFILL_BATCH),
            hc_normed: halfs(wide_dim),
            hc_low: halfs(hc_lowrank),
            hc_up: halfs(wide_dim),
            hc_inject: halfs(hc_count * MAX_PREFILL_BATCH),

            q_packed: halfs(2 * q_dim),
            attn_gate: halfs(q_dim),
            qsa,
            idx_heads,
            idx_head_dim,
            idx_compress,
            idx_block_topk,
            idx_qk: halfs(idx_projection_len),
            qsa_scores,
            qsa_positions,
            qsa_force_dense,
            gdn_qkv_raw: halfs(qkv_dim),
            gdn_conv_out: halfs(qkv_dim),
            gdn_norm_capture,
            layer_boundary_capture,
            attention_intermediate_capture,
            gdn_z: halfs(value_dim),
            gdn_a: halfs(v_heads),
            gdn_b: halfs(v_heads),
            gdn_y: halfs(value_dim),
            gdn_out: halfs(value_dim),

            router_ones,
            per_expert_ones: vec![1.0; num_experts],
            router_logits_f32: context
                .new_output_buffer((num_experts.max(1) * 4 * MAX_PREFILL_BATCH) as u64),
            moe_x: halfs(hidden * MAX_PREFILL_BATCH),
            h1: halfs(hidden),
            h2: halfs(hidden),
            moe_zero,
            shared_gate_logit: halfs(1),

            ngram_emb: halfs(hidden * MAX_PREFILL_BATCH),
            ple_key: halfs(wide_dim),
            ple_value: halfs(hidden),
            ple_query: halfs(wide_dim),
            ple_gv: halfs(wide_dim),
            ple_conv_normed: halfs(wide_dim),
            ple_conv_out: halfs(wide_dim),
            ple_conv_tail,
            ngram_context: NgramContext::new(ngram_context_len, arch.ple.eos_token_id),
            ngram_context_len,
            eos_token_id: arch.ple.eos_token_id,
            ngram_layout,
            ngram_table,
        })
    }

    /// Rewinds the recurrent state to empty context: the GDN chain's delta
    /// rule and conv tail, the PLE n-gram context, AND the PLE dilated
    /// conv's own tail.
    ///
    /// **THIS WAS THE qwen4_exp QUALITY GATE'S DETERMINISM BUG
    /// (`docs/QWEN4_EXP.md`'s "The quality gate is BLOCKED" section).** The
    /// module doc used to claim the conv tail's staleness across `reset()`
    /// was inert because "today nothing [resets mid-process]" -- that claim
    /// was false the moment `crates/bench`'s quality gate called `reset()`
    /// between two back-to-back warm generations on one open runner, which
    /// is exactly the mid-process reset the old comment said did not exist
    /// yet. Leaving `ple_conv_tail` stale meant generation 2 started PLE's
    /// dilated conv from generation 1's leftover history instead of from
    /// zero, diverging the wide residual from the very first PLE-layer
    /// token onward and cascading into a completely different greedy
    /// digest -- while a fresh process always starts from the zeros
    /// `RealQwen4State::build` writes, which is why cross-process
    /// generation reproduced exactly throughout that investigation.
    pub(crate) fn reset(&mut self) {
        self.gdn.reset();
        // The indexer's raw keys and pooled blocks are addressed by absolute
        // position and cut by the KV cache's own cursor, so this releases
        // pages and rewinds the pooled-block cursor rather than erasing.
        self.qsa.reset();
        self.ngram_context = NgramContext::new(self.ngram_context_len, self.eos_token_id);
        let tail_len = self.ple_conv_tail.length() as usize;
        gpu::write_buffer_bytes(&self.ple_conv_tail, 0, &vec![0u8; tail_len]);
    }

    pub(crate) fn flush_gdn_norm_capture(&self) {
        if let Some(capture) = &self.gdn_norm_capture {
            capture.flush_if_captured();
        }
    }

    pub(crate) fn flush_layer_boundary_capture(&self) {
        if let Some(capture) = &self.layer_boundary_capture {
            capture.flush_if_captured();
        }
    }

    pub(crate) fn flush_attention_intermediate_capture(&self) {
        if let Some(capture) = &self.attention_intermediate_capture {
            capture.flush_if_captured();
        }
    }
}
