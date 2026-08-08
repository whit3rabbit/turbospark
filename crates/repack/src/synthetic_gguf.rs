//! Builds byte-exact GGUF v3 files in memory, for exercising
//! [`crate::gguf_header`] and the GGUF repack walk without a multi-GB
//! download. Sibling of [`crate::synthetic_model`] and
//! [`crate::synthetic_qwen`], and used the same way.
//!
//! This is a WRITER FOR TESTS, not a production encoder: it lays every
//! tensor out at the next aligned offset in the order pushed, and it does
//! not quantize anything. Callers hand it already-packed bytes, which is
//! also exactly how the repack walk treats a real file, so a round trip
//! through here proves the same byte-identity property the walk must
//! preserve.
//!
//! The weights it emits are deterministic but meaningless. The existing
//! rule for the other synthetic builders applies unchanged: a test may
//! assert structure, offsets, and byte identity, never generated text.

use crate::gguf_header::GgufValue;

/// A built GGUF file plus the absolute `[start, end)` byte range of each
/// tensor's data, in push order. The ranges come back alongside the bytes so
/// a test can assert byte identity without re-deriving the very layout it is
/// trying to verify.
pub type GgufFileAndRanges = (Vec<u8>, Vec<(String, (u64, u64))>);

/// Default alignment the spec assumes when `general.alignment` is absent.
/// The builder writes tensor data at multiples of this and does NOT emit
/// the key, so the round trip also covers the absent-key default path.
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone)]
struct PendingTensor {
    name: String,
    ggml_type: u32,
    dims: Vec<u64>,
    data: Vec<u8>,
}

/// Accumulates metadata and tensors, then serializes one GGUF v3 file.
#[derive(Debug, Clone)]
pub struct GgufBuilder {
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<PendingTensor>,
    alignment: u64,
    /// When set, `general.alignment` is written into the metadata. Left
    /// unset, the file relies on the spec's default and the reader's
    /// fallback.
    emit_alignment_key: bool,
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufBuilder {
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
            emit_alignment_key: false,
        }
    }

    /// Override the data-region alignment AND write `general.alignment`.
    /// Panics on a non-power-of-two, matching what the reader rejects:
    /// a fixture that cannot be read back is a broken fixture, not a test
    /// case, and the rejection path has its own hand-built bytes.
    pub fn with_alignment(mut self, alignment: u64) -> Self {
        assert!(
            alignment > 0 && alignment.is_power_of_two(),
            "alignment {alignment} is not a non-zero power of two"
        );
        self.alignment = alignment;
        self.emit_alignment_key = true;
        self
    }

    pub fn metadata(mut self, key: &str, value: GgufValue) -> Self {
        self.metadata.push((key.to_string(), value));
        self
    }

    pub fn metadata_str(self, key: &str, value: &str) -> Self {
        self.metadata(key, GgufValue::String(value.to_string()))
    }

    pub fn metadata_u32(self, key: &str, value: u32) -> Self {
        self.metadata(key, GgufValue::U32(value))
    }

    pub fn metadata_f32(self, key: &str, value: f32) -> Self {
        self.metadata(key, GgufValue::F32(value))
    }

    /// Push a tensor with caller-supplied packed bytes. `dims` are in GGUF's
    /// own fastest-varying-first order, so a logical `[rows, cols]` matrix
    /// is pushed as `[cols, rows]`.
    pub fn tensor(mut self, name: &str, ggml_type: u32, dims: &[u64], data: Vec<u8>) -> Self {
        self.tensors.push(PendingTensor {
            name: name.to_string(),
            ggml_type,
            dims: dims.to_vec(),
            data,
        });
        self
    }

    /// Push a tensor of `blocks` Q8_0 blocks (32 elements, 34 bytes each),
    /// filled with a deterministic pattern derived from `seed` so two
    /// fixtures with different seeds cannot accidentally compare equal.
    pub fn q8_0_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        assert!(
            elements % 32 == 0,
            "{name}: {elements} elements is not a whole number of 32-element Q8_0 blocks"
        );
        let blocks = elements / 32;
        let mut data = Vec::with_capacity((blocks * 34) as usize);
        for b in 0..blocks {
            // f16 scale, then 32 int8 weights. Two properties matter beyond
            // being deterministic, and both are about what a caller does with
            // the fixture rather than about the format:
            //
            // - the scale is a valid small positive f16 (0.0625) rather than
            //   a random bit pattern, so a dequant reference can consume
            //   these bytes unchanged;
            // - the quants span [-8, 7] rather than the whole byte range, so
            //   the resulting weights are near +/-0.5 and a forward pass
            //   through a fixture install stays inside FP16. At full range
            //   the weights reach +/-32, every sublayer multiplies the
            //   residual by ~250, and a two-layer model overflows to
            //   infinity before it reaches the head -- which is a property of
            //   the fixture, not a kernel bug, and cost a debugging round
            //   once already.
            data.extend_from_slice(&0x2C00u16.to_le_bytes());
            for k in 0..32u64 {
                let n = seed
                    .wrapping_add((b as u8).wrapping_mul(31))
                    .wrapping_add(k as u8);
                data.push((((n % 16) as i8) - 8) as u8);
            }
        }
        self.tensor(name, 8, dims, data)
    }

    /// Deterministic weights in `[-0.5, 0.5)` for the K-quant helpers below.
    ///
    /// The range is the point, and it is the same trap the Q8_0 helper's
    /// comment records: at full quantizer range a fixture install's residual
    /// grows by orders of magnitude per sublayer and overflows FP16 before
    /// the head, which reads as a kernel bug and is not one.
    fn small_weights(elements: u64, seed: u8) -> Vec<f32> {
        let mut s = (seed as u32).wrapping_mul(2_654_435_761).wrapping_add(11);
        (0..elements)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Push a Q4_K tensor (256-element superblocks, 144 bytes each). Qwen
    /// 3.6's Q4_K_M puts its routed experts and its embedding table here.
    pub fn q4_k_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        assert!(
            dims[0] % 256 == 0,
            "{name}: rows of {} elements do not tile 256-element Q4_K superblocks",
            dims[0]
        );
        let data = compute::quantize_q4_k(&Self::small_weights(elements, seed));
        self.tensor(name, 12, dims, data)
    }

    /// Push a Q6_K tensor (256-element superblocks, 210 bytes each). The real
    /// Qwen file uses this for exactly one tensor, `output.weight`; a Gemma
    /// fixture has no such tensor because it ties its embeddings, so a
    /// K-quant fixture puts it on the attention projections instead.
    pub fn q6_k_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        assert!(
            dims[0] % 256 == 0,
            "{name}: rows of {} elements do not tile 256-element Q6_K superblocks",
            dims[0]
        );
        let data = compute::quantize_q6_k(&Self::small_weights(elements, seed));
        self.tensor(name, 14, dims, data)
    }

    /// Push an F32 tensor whose every value is EXACTLY representable in
    /// BF16, which is what llama.cpp actually writes for norms: it upcasts
    /// tensors that were BF16 in the original checkpoint, so the low sixteen
    /// mantissa bits are all zero (measured, `gguf_f32_transcode_network.rs`).
    ///
    /// Zeros would satisfy that too, and used to be what this fixture wrote,
    /// but a zero tensor also narrows exactly under a BROKEN transcode. These
    /// values do not.
    pub fn f32_upcast_bf16_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        let mut data = Vec::with_capacity((elements * 4) as usize);
        for i in 0..elements {
            // Sign and exponent fixed around 1.0, mantissa varying, so every
            // value is an ordinary positive number a norm could hold.
            let bf16 = 0x3F00u16 | ((i as u16).wrapping_add(seed as u16) & 0x00FF);
            data.extend_from_slice(&((bf16 as u32) << 16).to_le_bytes());
        }
        self.tensor(name, 0, dims, data)
    }

    /// Push an F32 tensor of ordinary values, low mantissa bits included, so
    /// a quantizing transcode has real rounding to do. What the router is:
    /// GGUF ships it F32 and this port stores it INT8-affine.
    pub fn f32_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        let mut data = Vec::with_capacity((elements * 4) as usize);
        let mut s = (seed as u32).wrapping_mul(2_654_435_761).wrapping_add(17);
        for _ in 0..elements {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0;
            data.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self.tensor(name, 0, dims, data)
    }

    /// Serialize. Returns the file bytes and, alongside them, the absolute
    /// `[start, end)` range of each tensor's data in push order, so a test
    /// can assert byte identity without re-deriving the layout it is trying
    /// to verify.
    pub fn build(&self) -> GgufFileAndRanges {
        let mut out = Vec::new();
        out.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        let kv_count = self.metadata.len() as u64 + u64::from(self.emit_alignment_key);
        out.extend_from_slice(&kv_count.to_le_bytes());

        for (key, value) in &self.metadata {
            write_string(&mut out, key);
            write_value(&mut out, value);
        }
        if self.emit_alignment_key {
            write_string(&mut out, "general.alignment");
            write_value(&mut out, &GgufValue::U32(self.alignment as u32));
        }

        // Data-region offsets have to be known before the table is written,
        // and the table's length depends on the names, so lay the region out
        // first. Offsets are relative to the data region, so this pass does
        // not depend on where the table ends.
        let mut relative = 0u64;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            offsets.push(relative);
            relative = (relative + t.data.len() as u64).div_ceil(self.alignment) * self.alignment;
        }

        for (t, offset) in self.tensors.iter().zip(offsets.iter()) {
            write_string(&mut out, &t.name);
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.ggml_type.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
        }

        let data_region_start = (out.len() as u64).div_ceil(self.alignment) * self.alignment;
        out.resize(data_region_start as usize, 0);

        let mut ranges = Vec::with_capacity(self.tensors.len());
        for (t, offset) in self.tensors.iter().zip(offsets.iter()) {
            let start = data_region_start + offset;
            out.resize(start as usize, 0);
            out.extend_from_slice(&t.data);
            ranges.push((t.name.clone(), (start, start + t.data.len() as u64)));
        }

        (out, ranges)
    }
}

/// Shape knobs for [`build_synthetic_gemma4_gguf`]. Tiny by design, but
/// every dimension is chosen so each tensor's element count is a whole
/// number of 32-element Q8_0 blocks, PER EXPERT as well as in total, which
/// is the constraint a naive shrink of a real config violates first.
#[derive(Debug, Clone, Copy)]
pub struct SyntheticGgufShape {
    pub num_layers: usize,
    pub hidden: u64,
    pub num_heads: u64,
    pub head_dim: u64,
    pub full_head_dim: u64,
    pub num_kv_heads: u64,
    pub num_full_kv_heads: u64,
    pub intermediate: u64,
    pub moe_intermediate: u64,
    pub num_experts: u64,
    pub top_k: u64,
    pub vocab: u64,
    pub sliding_window: u64,
    /// Mix K-quants in the way a real `Q4_K_M` does: routed experts and the
    /// embedding table at Q4_K, the attention projections at Q6_K, everything
    /// else Q8_0. Needs [`SyntheticGgufShape::k_quant`]'s dimensions, since
    /// every K-quant row has to tile 256 elements where Q8_0 needs 32.
    pub k_quants: bool,
}

impl Default for SyntheticGgufShape {
    /// Two layers, one sliding-window and one global, so the per-layer
    /// `head_count_kv` array and the `sliding_window_pattern` inversion are
    /// both exercised by the smallest possible fixture.
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden: 64,
            num_heads: 4,
            head_dim: 16,
            full_head_dim: 32,
            num_kv_heads: 2,
            num_full_kv_heads: 1,
            intermediate: 32,
            moe_intermediate: 16,
            num_experts: 4,
            top_k: 2,
            vocab: 128,
            sliding_window: 8,
            k_quants: false,
        }
    }
}

impl SyntheticGgufShape {
    /// The smallest shape a K-quant fixture can take: every dimension that
    /// becomes a Q4_K or Q6_K ROW is 256, because a superblock is 256
    /// elements and ggml never emits a partial one. That is `hidden` (the
    /// embedding, the experts' gate/up, and every attention projection),
    /// `moe_intermediate` (the experts' down), and `num_heads * head_dim`
    /// (attn_output). Shrinking any of them is what breaks first.
    pub fn k_quant() -> Self {
        Self {
            hidden: 256,
            num_heads: 8,
            head_dim: 32,
            full_head_dim: 32,
            intermediate: 256,
            moe_intermediate: 256,
            vocab: 256,
            k_quants: true,
            ..Self::default()
        }
    }

    /// `true` where the layer slides, matching GGUF's own polarity.
    fn slides(&self, layer: usize) -> bool {
        // Last layer global, the rest sliding: enough to produce both kinds
        // without reproducing Gemma's every-sixth-layer pattern.
        layer + 1 != self.num_layers
    }
}

/// A complete, tiny Gemma 4 GGUF: every tensor name and metadata key the
/// real `gemma-4-26B-A4B-it-Q8_0.gguf` carries, at toy dimensions.
///
/// The name and metadata inventory is copied from the real file (see
/// `gguf_names.rs`), including the parts that are easy to get wrong: gate
/// and up FUSED into `ffn_gate_up_exps`, the router's per-expert scale
/// hiding under `ffn_down_exps.scale`, no `output.weight` because Gemma
/// ties its embeddings, and NO `attn_v` on global layers.
pub fn build_synthetic_gemma4_gguf(shape: SyntheticGgufShape) -> GgufFileAndRanges {
    let s = shape;
    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "gemma4")
        .metadata_u32("gemma4.block_count", s.num_layers as u32)
        .metadata_u32("gemma4.embedding_length", s.hidden as u32)
        .metadata_u32("gemma4.attention.head_count", s.num_heads as u32)
        .metadata_u32("gemma4.expert_count", s.num_experts as u32)
        .metadata_u32("gemma4.expert_used_count", s.top_k as u32)
        .metadata_u32(
            "gemma4.expert_feed_forward_length",
            s.moe_intermediate as u32,
        )
        .metadata_u32("gemma4.feed_forward_length", s.intermediate as u32)
        .metadata_u32("gemma4.attention.key_length", s.full_head_dim as u32)
        .metadata_u32("gemma4.attention.key_length_swa", s.head_dim as u32)
        .metadata_u32("gemma4.attention.sliding_window", s.sliding_window as u32)
        .metadata_f32("gemma4.final_logit_softcapping", 30.0)
        .metadata_f32("gemma4.rope.freq_base", 1_000_000.0)
        .metadata_f32("gemma4.rope.freq_base_swa", 10_000.0)
        .metadata(
            "gemma4.attention.sliding_window_pattern",
            GgufValue::Array(
                (0..s.num_layers)
                    .map(|l| GgufValue::Bool(s.slides(l)))
                    .collect(),
            ),
        )
        .metadata(
            "gemma4.attention.head_count_kv",
            GgufValue::Array(
                (0..s.num_layers)
                    .map(|l| {
                        GgufValue::I32(if s.slides(l) {
                            s.num_kv_heads as i32
                        } else {
                            s.num_full_kv_heads as i32
                        })
                    })
                    .collect(),
            ),
        )
        .f32_upcast_bf16_tensor("output_norm.weight", &[s.hidden], 2);
    b = embed_tensor(b, &s, "token_embd.weight", &[s.hidden, s.vocab], 1);

    for l in 0..s.num_layers {
        let sliding = s.slides(l);
        let hd = if sliding { s.head_dim } else { s.full_head_dim };
        let kv = if sliding {
            s.num_kv_heads
        } else {
            s.num_full_kv_heads
        };
        let q_dim = s.num_heads * hd;
        let seed = (l as u8).wrapping_mul(37).wrapping_add(3);

        b = attn_tensor(
            b,
            &s,
            &format!("blk.{l}.attn_q.weight"),
            &[s.hidden, q_dim],
            seed,
        );
        b = attn_tensor(
            b,
            &s,
            &format!("blk.{l}.attn_k.weight"),
            &[s.hidden, kv * hd],
            seed.wrapping_add(1),
        );
        b = attn_tensor(
            b,
            &s,
            &format!("blk.{l}.attn_output.weight"),
            &[q_dim, s.hidden],
            seed.wrapping_add(2),
        );
        b = b
            .f32_upcast_bf16_tensor(&format!("blk.{l}.attn_q_norm.weight"), &[hd], seed)
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.attn_k_norm.weight"),
                &[hd],
                seed.wrapping_add(1),
            );

        // Global layers carry no V projection at all -- a property of the
        // model, reproduced here so the walk is exercised against it.
        if sliding {
            b = attn_tensor(
                b,
                &s,
                &format!("blk.{l}.attn_v.weight"),
                &[s.hidden, kv * hd],
                seed.wrapping_add(3),
            );
        }

        for (n, norm) in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "pre_ffw_norm_2",
            "post_ffw_norm",
            "post_ffw_norm_1",
            "post_ffw_norm_2",
        ]
        .iter()
        .enumerate()
        {
            b = b.f32_upcast_bf16_tensor(
                &format!("blk.{l}.{norm}.weight"),
                &[s.hidden],
                seed.wrapping_add(10 + n as u8),
            );
        }

        b = b
            .q8_0_tensor(
                &format!("blk.{l}.ffn_gate.weight"),
                &[s.hidden, s.intermediate],
                seed.wrapping_add(4),
            )
            .q8_0_tensor(
                &format!("blk.{l}.ffn_up.weight"),
                &[s.hidden, s.intermediate],
                seed.wrapping_add(5),
            )
            .q8_0_tensor(
                &format!("blk.{l}.ffn_down.weight"),
                &[s.intermediate, s.hidden],
                seed.wrapping_add(6),
            )
            // Router: F32 in GGUF, where an MLX install carries INT8. Real
            // values rather than upcast BF16, because the repack quantizes
            // this one instead of narrowing it.
            .f32_tensor(
                &format!("blk.{l}.ffn_gate_inp.weight"),
                &[s.hidden, s.num_experts],
                seed.wrapping_add(20),
            )
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.ffn_gate_inp.scale"),
                &[s.hidden],
                seed.wrapping_add(21),
            )
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.ffn_down_exps.scale"),
                &[s.num_experts],
                seed.wrapping_add(22),
            )
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.layer_output_scale.weight"),
                &[1],
                seed.wrapping_add(23),
            );

        // Gate and up fused along the output dim.
        b = expert_tensor(
            b,
            &s,
            &format!("blk.{l}.ffn_gate_up_exps.weight"),
            &[s.hidden, 2 * s.moe_intermediate, s.num_experts],
            seed.wrapping_add(7),
        );
        b = expert_tensor(
            b,
            &s,
            &format!("blk.{l}.ffn_down_exps.weight"),
            &[s.moe_intermediate, s.hidden, s.num_experts],
            seed.wrapping_add(8),
        );
    }

    b.build()
}

/// The three roles a K-quant fixture moves off Q8_0, each mirroring where the
/// real `Qwen3.6-35B-A3B-Q4_K_M.gguf` puts that block type. Written as
/// functions rather than a method on the builder because the choice belongs
/// to the fixture's shape, not to GGUF.
fn embed_tensor(
    b: GgufBuilder,
    s: &SyntheticGgufShape,
    name: &str,
    dims: &[u64],
    seed: u8,
) -> GgufBuilder {
    if s.k_quants {
        b.q4_k_tensor(name, dims, seed)
    } else {
        b.q8_0_tensor(name, dims, seed)
    }
}

fn expert_tensor(
    b: GgufBuilder,
    s: &SyntheticGgufShape,
    name: &str,
    dims: &[u64],
    seed: u8,
) -> GgufBuilder {
    if s.k_quants {
        b.q4_k_tensor(name, dims, seed)
    } else {
        b.q8_0_tensor(name, dims, seed)
    }
}

/// Q6_K, where a real file would carry it on `output.weight`. A Gemma
/// fixture ties its embeddings and so has no such tensor, and the attention
/// projections are the next place a resident GEMV reads every token.
fn attn_tensor(
    b: GgufBuilder,
    s: &SyntheticGgufShape,
    name: &str,
    dims: &[u64],
    seed: u8,
) -> GgufBuilder {
    if s.k_quants {
        b.q6_k_tensor(name, dims, seed)
    } else {
        b.q8_0_tensor(name, dims, seed)
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Writes a value's type id followed by its payload, mirroring the reader
/// in `gguf_header`'s `Cursor::value` exactly.
fn write_value(out: &mut Vec<u8>, value: &GgufValue) {
    match value {
        GgufValue::U8(v) => {
            out.extend_from_slice(&0u32.to_le_bytes());
            out.push(*v);
        }
        GgufValue::I8(v) => {
            out.extend_from_slice(&1u32.to_le_bytes());
            out.push(*v as u8);
        }
        GgufValue::U16(v) => {
            out.extend_from_slice(&2u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I16(v) => {
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::U32(v) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I32(v) => {
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::F32(v) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        GgufValue::Bool(v) => {
            out.extend_from_slice(&7u32.to_le_bytes());
            out.push(u8::from(*v));
        }
        GgufValue::String(v) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            write_string(out, v);
        }
        GgufValue::Array(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            // An array writes its ELEMENT type once, then the length, then
            // bare payloads: the elements do not repeat the type id. An
            // empty array has no element to take the type from, so it is
            // written as an empty array of U8 rather than being rejected.
            let elem_type = items.first().map_or(0, value_type_id);
            out.extend_from_slice(&elem_type.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                assert_eq!(
                    value_type_id(item),
                    elem_type,
                    "GGUF arrays are homogeneous; mixed element types cannot be encoded"
                );
                write_payload(out, item);
            }
        }
        GgufValue::U64(v) => {
            out.extend_from_slice(&10u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I64(v) => {
            out.extend_from_slice(&11u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::F64(v) => {
            out.extend_from_slice(&12u32.to_le_bytes());
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
}

fn value_type_id(value: &GgufValue) -> u32 {
    match value {
        GgufValue::U8(_) => 0,
        GgufValue::I8(_) => 1,
        GgufValue::U16(_) => 2,
        GgufValue::I16(_) => 3,
        GgufValue::U32(_) => 4,
        GgufValue::I32(_) => 5,
        GgufValue::F32(_) => 6,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        GgufValue::U64(_) => 10,
        GgufValue::I64(_) => 11,
        GgufValue::F64(_) => 12,
    }
}

/// The payload alone, with no leading type id: what array elements use.
fn write_payload(out: &mut Vec<u8>, value: &GgufValue) {
    match value {
        GgufValue::U8(v) => out.push(*v),
        GgufValue::I8(v) => out.push(*v as u8),
        GgufValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F32(v) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
        GgufValue::Bool(v) => out.push(u8::from(*v)),
        GgufValue::String(v) => write_string(out, v),
        GgufValue::Array(items) => {
            let elem_type = items.first().map_or(0, value_type_id);
            out.extend_from_slice(&elem_type.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_payload(out, item);
            }
        }
        GgufValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F64(v) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
    }
}
