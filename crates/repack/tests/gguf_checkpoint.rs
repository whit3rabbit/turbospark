//! The GGUF repack walk: synthetic GGUF in, `.gturbo` out.
//!
//! The centrepiece is byte identity. Everything else this walk does is
//! bookkeeping that a shape assertion can check, but the one property the
//! whole lossless-repack rule rests on is that the expert bytes written to
//! disk are the SAME BYTES the GGUF held, sliced and never transformed.
//! Two things can silently break it -- the data-region alignment and the
//! per-expert slicing of a rank-3 tensor -- and neither shows up in any
//! structural assertion, so they are checked against the source directly.

use std::sync::atomic::{AtomicU64, Ordering};

use mrefrust_repack::{
    build_synthetic_gemma4_gguf, orchestrate_gguf_checkpoint, parse_gguf_header,
    write_gguf_install_streamed, GgufBuilder, GgufRepackError, GgufValue, MemoryRangeSource,
    ResidentEntrySpec, SyntheticGgufShape, DTYPE_BF16, DTYPE_FP32, DTYPE_GGUF_Q8_0,
    GGUF_DEFAULT_MAX_HEADER_BYTES,
};

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "mrefrust-gguf-repack-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

struct Fixture {
    bytes: Vec<u8>,
    shape: SyntheticGgufShape,
    header: mrefrust_repack::GgufHeader,
}

impl Fixture {
    fn new() -> Self {
        let shape = SyntheticGgufShape::default();
        let (bytes, _) = build_synthetic_gemma4_gguf(shape);
        let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
        Self {
            bytes,
            shape,
            header,
        }
    }

    /// The raw bytes of one source tensor, addressed straight out of the
    /// fixture buffer. Deliberately an independent reading of the same file
    /// rather than anything the walk produced.
    fn tensor(&self, name: &str) -> &[u8] {
        let (start, end) = self
            .header
            .absolute_range(name)
            .expect("known tensor")
            .expect("size");
        &self.bytes[start as usize..end as usize]
    }
}

#[test]
fn resident_tensors_keep_their_bytes_names_and_logical_shapes() {
    let f = Fixture::new();
    let h = &f.header;
    let out = orchestrate_gguf_checkpoint(h, &MemoryRangeSource::new(&f.bytes)).expect("walk");

    let by_name: std::collections::BTreeMap<&str, &ResidentEntrySpec> = out
        .resident
        .iter()
        .map(|e| match e {
            ResidentEntrySpec::Raw(r) => (r.name.as_str(), e),
            ResidentEntrySpec::Int8(t) => (t.name.as_str(), e),
            ResidentEntrySpec::Int4(_) => {
                panic!("nothing in a GGUF becomes INT4; the transcode targets are INT8 only")
            }
        })
        .collect();

    // Canonical names, not GGUF ones.
    assert!(by_name.contains_key("language_model.model.embed_tokens.weight"));
    assert!(by_name.contains_key("language_model.model.norm.weight"));
    assert!(by_name.contains_key("language_model.model.layers.0.self_attn.q_proj.weight"));
    assert!(by_name.contains_key("language_model.model.layers.0.router.proj.weight"));
    // Gemma ties embeddings, so there is no head.
    assert!(!by_name.contains_key("language_model.lm_head.weight"));
    // Global layers have no V projection; layer 1 is the fixture's global one.
    assert!(by_name.contains_key("language_model.model.layers.0.self_attn.v_proj.weight"));
    assert!(!by_name.contains_key("language_model.model.layers.1.self_attn.v_proj.weight"));

    let ResidentEntrySpec::Raw(embed) = by_name["language_model.model.embed_tokens.weight"] else {
        unreachable!()
    };
    assert_eq!(embed.dtype, DTYPE_GGUF_Q8_0);
    // GGUF stores [hidden, vocab]; the index stores logical [vocab, hidden].
    assert_eq!(
        embed.shape,
        (f.shape.vocab as u32, f.shape.hidden as u32, 0, 0)
    );
    assert_eq!(embed.bytes, f.tensor("token_embd.weight"));

    // The router is F32 in GGUF where this port's kernels want INT8, so it
    // is the one resident tensor that is quantized rather than carried.
    // Nothing F32 survives into the install: no F32 kernel exists.
    let ResidentEntrySpec::Int8(router) =
        by_name["language_model.model.layers.0.router.proj.weight"]
    else {
        panic!("the router must arrive INT8-affine, which is the dtype the GEMV reads")
    };
    assert_eq!(router.rows, f.shape.num_experts as u32);
    assert_eq!(router.cols, f.shape.hidden as u32);
    assert!(!out
        .resident
        .iter()
        .any(|e| matches!(e, ResidentEntrySpec::Raw(r) if r.dtype == DTYPE_FP32)));
}

// ---------------------------------------------------------------------------
// The F32 transcode (ROADMAP Phase G Stage 2, item 5)
// ---------------------------------------------------------------------------
//
// GGUF ships norms and the router as F32; this port has kernels for BF16 and
// INT8-affine and none for F32. The decision to transcode at repack time
// rather than build two more kernels was measured, not preferred:
// `gguf_f32_transcode_network.rs` on the real file. What is checked HERE is
// that the walk does what that measurement licensed.
//
// The fixture matters as much as the assertions. Its norms are upcast BF16
// patterns (what llama.cpp really writes) rather than the zeros this file
// used to carry, because zeros narrow exactly under a broken transcode too.

/// Every value of every F32 norm survives the narrowing, bit for bit. The
/// fixture is built from BF16 patterns precisely so this can be an EXACT
/// assertion rather than a tolerance.
#[test]
fn f32_norms_narrow_to_bf16_bit_exactly() {
    let f = Fixture::new();
    let out =
        orchestrate_gguf_checkpoint(&f.header, &MemoryRangeSource::new(&f.bytes)).expect("walk");

    let entry = |name: &str| {
        out.resident
            .iter()
            .find_map(|e| match e {
                ResidentEntrySpec::Raw(r) if r.name == name => Some(r),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no resident entry {name}"))
    };

    let mut checked = 0usize;
    for (gguf, canonical) in [
        ("output_norm.weight", "language_model.model.norm.weight"),
        (
            "blk.0.attn_norm.weight",
            "language_model.model.layers.0.input_layernorm.weight",
        ),
        (
            "blk.0.post_ffw_norm.weight",
            "language_model.model.layers.0.post_feedforward_layernorm.weight",
        ),
        (
            "blk.1.attn_q_norm.weight",
            "language_model.model.layers.1.self_attn.q_norm.weight",
        ),
        // Not a norm, but read by `read_bf16_host` and so on the same path.
        (
            "blk.0.ffn_gate_inp.scale",
            "language_model.model.layers.0.router.scale",
        ),
        (
            "blk.0.layer_output_scale.weight",
            "language_model.model.layers.0.layer_scalar",
        ),
    ] {
        let source: Vec<f32> = f
            .tensor(gguf)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let e = entry(canonical);
        assert_eq!(e.dtype, DTYPE_BF16, "{canonical}");
        assert_eq!(e.bytes.len(), source.len() * 2, "{canonical}");
        for (i, &v) in source.iter().enumerate() {
            let stored = u16::from_le_bytes([e.bytes[i * 2], e.bytes[i * 2 + 1]]);
            assert_eq!(stored, compute::f32_to_bf16(v), "{canonical} element {i}");
            assert_eq!(
                compute::bf16_to_f32(stored),
                v,
                "{canonical} element {i} lost bits"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 6);
    assert!(
        out.lossy_narrowing.is_empty(),
        "nothing in this fixture should lose bits: {:?}",
        out.lossy_narrowing
    );
}

/// The router is the one resident tensor that is genuinely quantized. Shape
/// first (rows and cols are reversed out of GGUF's dim order, and getting
/// that backwards leaves the byte count identical), then a dequantize round
/// trip against the source, which is what actually fails if the row stride
/// or the group stride is wrong.
#[test]
fn the_router_becomes_an_int8_affine_entry_that_dequantizes_back() {
    let f = Fixture::new();
    let out =
        orchestrate_gguf_checkpoint(&f.header, &MemoryRangeSource::new(&f.bytes)).expect("walk");

    let router = out
        .resident
        .iter()
        .find_map(|e| match e {
            ResidentEntrySpec::Int8(t)
                if t.name == "language_model.model.layers.0.router.proj.weight" =>
            {
                Some(t)
            }
            _ => None,
        })
        .expect("an INT8 router entry");

    let experts = f.shape.num_experts as usize;
    let hidden = f.shape.hidden as usize;
    // GGUF stores [hidden, experts]; the logical matrix is [experts, hidden].
    assert_eq!(router.rows as usize, experts);
    assert_eq!(router.cols as usize, hidden);
    assert_eq!(router.packed.len(), experts * hidden);
    assert_eq!(router.scales.len(), experts * hidden / 64);
    assert_eq!(router.biases.len(), experts * hidden / 64);

    let source: Vec<f32> = f
        .tensor("blk.0.ffn_gate_inp.weight")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(source.len(), experts * hidden);

    for r in 0..experts {
        let groups = hidden / 64;
        let row = compute::Int8AffineRow {
            packed: router.packed[r * hidden..(r + 1) * hidden].to_vec(),
            scales: router.scales[r * groups..(r + 1) * groups].to_vec(),
            biases: router.biases[r * groups..(r + 1) * groups].to_vec(),
        };
        let back = compute::dequantize_int8_affine(&row, hidden);
        for g in 0..groups {
            // One INT8 level of that group, which is the most an affine
            // quantizer may move a value. A transposed row or a misaligned
            // group lands orders of magnitude outside this.
            let step = compute::bf16_to_f32(row.scales[g]);
            for k in 0..64 {
                let i = r * hidden + g * 64 + k;
                let err = (back[g * 64 + k] - source[i]).abs();
                assert!(
                    err <= step,
                    "row {r} group {g} element {k}: error {err} exceeds one step {step}"
                );
            }
        }
    }
}

/// A GGUF whose router row is not a whole number of 64-element groups must
/// come back as an error. `quantize_int8_affine` ASSERTS on that shape, and
/// this crate forbids handing a caller's file straight to a panic.
#[test]
fn a_router_row_that_is_not_a_whole_group_is_rejected_rather_than_panicking() {
    // 100 is deliberately not a multiple of 64.
    let (bytes, _) = minimal_gemma_gguf(100)
        .f32_tensor("blk.0.ffn_gate_inp.weight", &[100, 4], 7)
        .build();
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let err = match orchestrate_gguf_checkpoint(&header, &MemoryRangeSource::new(&bytes)) {
        Ok(_) => panic!("a 100-wide router row cannot be INT8-quantized at group 64"),
        Err(e) => e,
    };
    assert!(
        matches!(err, GgufRepackError::ShapeMismatch { .. }),
        "expected a shape error, got {err}"
    );
    assert!(err.to_string().contains("64"), "{err}");
}

/// A converter that did NOT upcast from BF16 still produces an install (the
/// runtime has no F32 kernel, so BF16 is the only destination), but the loss
/// is reported rather than swallowed.
#[test]
fn a_genuinely_f32_norm_is_narrowed_and_counted() {
    let (bytes, _) = minimal_gemma_gguf(64)
        // Ordinary F32 values, low mantissa bits and all.
        .f32_tensor("output_norm.weight", &[64], 9)
        .build();
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let out = orchestrate_gguf_checkpoint(&header, &MemoryRangeSource::new(&bytes)).expect("walk");

    assert_eq!(out.lossy_narrowing.len(), 1);
    assert_eq!(out.lossy_narrowing[0].0, "output_norm.weight");
    assert!(
        out.lossy_narrowing[0].1 > 0,
        "an F32 tensor with real mantissa bits must report some loss"
    );
    // It is still carried, as BF16, because there is nowhere else to put it.
    assert!(out.resident.iter().any(
        |e| matches!(e, ResidentEntrySpec::Raw(r) if r.name == "language_model.model.norm.weight" && r.dtype == DTYPE_BF16)
    ));
}

/// GGML's F32 type id, for the tensors below that carry chosen values rather
/// than the builder's seeded ones.
const GGML_F32: u32 = 0;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The Qwen sibling of [`minimal_gemma_gguf`], sized so the V-head
/// de-interleave has something to say: 4 V heads (so the map is
/// `0 -> 0, 1 -> 2, 2 -> 1, 3 -> 3`, not the identity and not a reversal),
/// 1 K head, 32-wide heads, kernel 2. That makes `conv1d`'s channel run
/// `[q 32 | k 32 | v 128]` and the value stream 128 wide.
///
/// The 32-wide head is not arbitrary: it is exactly one Q8_0 block, which is
/// what lets `out_proj`'s COLUMN permutation stay a byte move. The real model
/// is 128-wide, i.e. four blocks per head. A head narrower than a block is
/// refused rather than shuffled, and the fixture would hide that if its head
/// were 8 wide.
fn minimal_qwen_gguf() -> GgufBuilder {
    GgufBuilder::new()
        .metadata_str("general.architecture", "qwen35moe")
        .metadata_u32("qwen35moe.block_count", 1)
        .metadata_u32("qwen35moe.embedding_length", 64)
        .metadata_u32("qwen35moe.attention.head_count", 4)
        .metadata_u32("qwen35moe.attention.head_count_kv", 2)
        .metadata_u32("qwen35moe.expert_count", 4)
        .metadata_u32("qwen35moe.expert_used_count", 2)
        .metadata_u32("qwen35moe.expert_feed_forward_length", 16)
        .metadata_u32("qwen35moe.full_attention_interval", 4)
        // The gated-DeltaNet dimensions, under the `ssm.` keys GGUF borrows.
        .metadata_u32("qwen35moe.ssm.group_count", 1)
        .metadata_u32("qwen35moe.ssm.time_step_rank", 4)
        .metadata_u32("qwen35moe.ssm.state_size", 32)
        .metadata_u32("qwen35moe.ssm.inner_size", 128)
        .metadata_u32("qwen35moe.ssm.conv_kernel", 2)
        .q8_0_tensor("token_embd.weight", &[64, 128], 1)
}

/// The V-head map the whole convention rests on, as this test reads it:
/// GGUF head `h` holds what MLX head `MAP[h]` holds, so the walk writes
/// source head `h` out at `MAP[h]`.
const V_HEAD_MAP: [usize; 4] = [0, 2, 1, 3];

/// Qwen's gated-DeltaNet parameters arrive under llama.cpp's convention and
/// the walk owes the MLX one. ONE convention over the V-head axis, but a
/// different stride per tensor -- which is the part a single shared helper
/// call gets wrong (see `v_head_axis`). Settled against the real files by
/// `tests/gguf_qwen_core_probe.rs` and `tests/gguf_qwen_quant_probe.rs`; this
/// test pins the arithmetic.
#[test]
fn qwens_gated_deltanet_parameters_are_rewritten_into_the_mlx_convention() {
    // Distinct, exactly BF16-representable, and NEGATIVE: `ssm_a` holds
    // `-exp(A_log)`.
    let ssm_a = [-1.0f32, -2.0, -4.0, -8.0];
    let dt = [1.0f32, 2.0, 4.0, 8.0];
    // Channel `c` is filled with the value `c`, so a moved channel is
    // readable straight off the output.
    let conv: Vec<f32> = (0..192).flat_map(|c| [c as f32, c as f32]).collect();

    let (bytes, _) = minimal_qwen_gguf()
        .tensor("blk.0.ssm_a", GGML_F32, &[4], f32_bytes(&ssm_a))
        .tensor("blk.0.ssm_dt.bias", GGML_F32, &[4], f32_bytes(&dt))
        // Dims are fastest-varying first, so this is logically [192, 2].
        .tensor(
            "blk.0.ssm_conv1d.weight",
            GGML_F32,
            &[2, 192],
            f32_bytes(&conv),
        )
        // The quantized siblings, carried VERBATIM and permuted as bytes.
        // `ssm_beta` is one output row per V head; `ssm_out` takes the V axis
        // on its columns, one Q8_0 block per head.
        .q8_0_tensor("blk.0.ssm_beta.weight", &[64, 4], 3)
        .q8_0_tensor("blk.0.ssm_out.weight", &[128, 64], 5)
        .build();
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let out = orchestrate_gguf_checkpoint(&header, &MemoryRangeSource::new(&bytes)).expect("walk");

    let stored = |name: &str| -> Vec<f32> {
        out.resident
            .iter()
            .find_map(|e| match e {
                ResidentEntrySpec::Raw(r) if r.name == name => Some(r),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no resident entry {name}"))
            .bytes
            .chunks_exact(2)
            .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    };
    let at = |suffix: &str| format!("language_model.model.layers.0.{suffix}");

    // `dt_bias` is the clean case: a pure de-interleave, no value change, so
    // the expectation is the input read in a different order.
    let dt_out = stored(&at("linear_attn.dt_bias"));
    for (h, &to) in V_HEAD_MAP.iter().enumerate() {
        assert_eq!(dt_out[to], dt[h], "dt_bias head {h} belongs at {to}");
    }

    // `A_log` takes the same map AND `A_log = ln(-ssm_a)`. Asserted through
    // the inverse so the test does not simply restate the implementation:
    // exponentiating the stored value must give back the source.
    let a_out = stored(&at("linear_attn.A_log"));
    for (h, &to) in V_HEAD_MAP.iter().enumerate() {
        let round_trip = -a_out[to].exp();
        assert!(
            (round_trip - ssm_a[h]).abs() <= 1e-2 * ssm_a[h].abs(),
            "A_log head {h} at {to}: -exp({}) = {round_trip}, want {}",
            a_out[to],
            ssm_a[h]
        );
    }

    // `conv1d` is the one with a stride: `[q 32 | k 32 | v 128]` channels of
    // 2 elements each. The q and k halves must NOT move, and the v half moves
    // a whole 32-channel head at a time.
    let conv_out = stored(&at("linear_attn.conv1d.weight"));
    assert_eq!(conv_out.len(), 384);
    for c in 0..64 {
        assert_eq!(conv_out[c * 2], c as f32, "q/k channel {c} must not move");
    }
    for (h, &to) in V_HEAD_MAP.iter().enumerate() {
        for i in 0..32 {
            let want = (64 + h * 32 + i) as f32;
            let got = conv_out[(64 + to * 32 + i) * 2];
            assert_eq!(got, want, "v head {h} element {i} belongs at head {to}");
        }
    }

    // The quantized pair, asserted on RAW BYTES. That is the whole point of
    // doing this as a byte move: no dequantization happens, so the output has
    // to be the input's byte ranges in a different order, exactly.
    let raw = |name: &str| -> Vec<u8> {
        out.resident
            .iter()
            .find_map(|e| match e {
                ResidentEntrySpec::Raw(r) if r.name == name => Some(r.bytes.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no resident entry {name}"))
    };
    let source = |gguf: &str| -> Vec<u8> {
        let (s, e) = header.absolute_range(gguf).unwrap().unwrap();
        bytes[s as usize..e as usize].to_vec()
    };

    // `in_proj_b`: 4 rows of 64 columns, so a row is 2 Q8_0 blocks = 68 bytes.
    let (beta_in, beta_out) = (
        source("blk.0.ssm_beta.weight"),
        raw(&at("linear_attn.in_proj_b.weight")),
    );
    assert_eq!(beta_out.len(), beta_in.len());
    for (h, &to) in V_HEAD_MAP.iter().enumerate() {
        assert_eq!(
            beta_out[to * 68..(to + 1) * 68],
            beta_in[h * 68..(h + 1) * 68],
            "in_proj_b row {h} belongs at {to}"
        );
    }

    // `out_proj`: 64 rows of 128 columns, and the V axis is the COLUMNS, so
    // the move happens inside every row over 32-column (one-block, 34-byte)
    // groups.
    let (out_in, out_out) = (
        source("blk.0.ssm_out.weight"),
        raw(&at("linear_attn.out_proj.weight")),
    );
    assert_eq!(out_out.len(), out_in.len());
    let row_bytes = 4 * 34;
    for r in [0usize, 1, 63] {
        for (h, &to) in V_HEAD_MAP.iter().enumerate() {
            let dst = r * row_bytes + to * 34;
            let src = r * row_bytes + h * 34;
            assert_eq!(
                out_out[dst..dst + 34],
                out_in[src..src + 34],
                "out_proj row {r} head {h} belongs at {to}"
            );
        }
    }

    // `A_log` is the one tensor here that reports a lossy narrowing, and it
    // is the TRANSFORM's doing rather than the converter's: `ln(1) = 0`
    // survives BF16 and `ln(2)`, `ln(4)`, `ln(8)` do not. The real Qwen
    // repack reports exactly this for all 30 of its `ssm_a` tensors, which is
    // why that count is accounted for rather than a lead (ROADMAP item 10).
    assert_eq!(out.lossy_narrowing, vec![("blk.0.ssm_a".to_string(), 3)]);
}

/// `ssm_a` is `-exp(A_log)`, so a non-negative value means the tensor is not
/// what this walk thinks it is. Loud beats a NaN reaching the install: a
/// silent `ln` of a negative number would decode as gibberish 19 GB later.
#[test]
fn a_positive_ssm_a_is_refused_rather_than_producing_a_nan() {
    let (bytes, _) = minimal_qwen_gguf()
        .tensor(
            "blk.0.ssm_a",
            GGML_F32,
            &[4],
            f32_bytes(&[-1.0, -2.0, 0.5, -8.0]),
        )
        .build();
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let err = match orchestrate_gguf_checkpoint(&header, &MemoryRangeSource::new(&bytes)) {
        Ok(_) => panic!("a positive ssm_a cannot be -exp(anything)"),
        Err(e) => e,
    };
    assert!(
        matches!(err, GgufRepackError::ShapeMismatch { .. }),
        "expected a shape error, got {err}"
    );
    assert!(err.to_string().contains("negative"), "{err}");
}

/// The smallest GGUF `arch_from_gguf` accepts: one layer, no routed experts,
/// so a test can add exactly the one tensor it wants to say something about.
fn minimal_gemma_gguf(hidden: u32) -> GgufBuilder {
    GgufBuilder::new()
        .metadata_str("general.architecture", "gemma4")
        .metadata_u32("gemma4.block_count", 1)
        .metadata_u32("gemma4.embedding_length", hidden)
        .metadata_u32("gemma4.attention.head_count", 4)
        .metadata_u32("gemma4.attention.head_count_kv", 2)
        .metadata_u32("gemma4.expert_count", 4)
        .metadata_u32("gemma4.expert_used_count", 2)
        .metadata_u32("gemma4.expert_feed_forward_length", 16)
        .metadata(
            "gemma4.attention.sliding_window_pattern",
            GgufValue::Array(vec![GgufValue::Bool(true)]),
        )
        // `arch_from_gguf` reads the vocabulary off the embedding, so even
        // the minimal file carries one.
        .q8_0_tensor("token_embd.weight", &[hidden as u64, 128], 1)
}

/// THE property. Every expert's bytes on the way out must be exactly the
/// bytes that expert occupied in the GGUF, with the fused gate/up tensor's
/// two halves concatenating back to the original slice.
#[test]
fn every_expert_slice_is_byte_identical_to_its_source() {
    let f = Fixture::new();
    let h = &f.header;
    let out = orchestrate_gguf_checkpoint(h, &MemoryRangeSource::new(&f.bytes)).expect("walk");

    let experts = f.shape.num_experts as usize;
    assert_eq!(out.layers.len(), f.shape.num_layers);

    for layer in &out.layers {
        let fused = f.tensor(&format!("blk.{}.ffn_gate_up_exps.weight", layer.layer));
        let down = f.tensor(&format!("blk.{}.ffn_down_exps.weight", layer.layer));
        let fused_per = fused.len() / experts;
        let down_per = down.len() / experts;
        assert_eq!(layer.experts.len(), experts);

        for (e, blob) in layer.experts.iter().enumerate() {
            let roles: Vec<&str> = blob.sub_tensors.iter().map(|s| s.role.as_str()).collect();
            assert_eq!(
                roles,
                vec!["gate", "up", "down"],
                "blob order feeds the phase-2 reduce and is not free to vary"
            );

            let gate = &blob.sub_tensors[0].bytes;
            let up = &blob.sub_tensors[1].bytes;
            let down_bytes = &blob.sub_tensors[2].bytes;

            // The fused halves must reassemble into the source slice.
            let mut rejoined = gate.clone();
            rejoined.extend_from_slice(up);
            assert_eq!(
                rejoined,
                &fused[e * fused_per..(e + 1) * fused_per],
                "layer {} expert {e}: gate+up does not reassemble",
                layer.layer
            );
            assert_eq!(gate.len(), up.len(), "the halves are equal by construction");

            assert_eq!(
                down_bytes.as_slice(),
                &down[e * down_per..(e + 1) * down_per],
                "layer {} expert {e}: down slice differs",
                layer.layer
            );
        }
    }
}

#[test]
fn expert_stride_covers_the_largest_blob_and_is_page_rounded() {
    let f = Fixture::new();
    let h = &f.header;
    let out = orchestrate_gguf_checkpoint(h, &MemoryRangeSource::new(&f.bytes)).expect("walk");

    let largest = out
        .layers
        .iter()
        .flat_map(|l| l.experts.iter())
        .map(|b| b.sub_tensors.iter().map(|s| s.bytes.len() as u64).sum())
        .max()
        .unwrap_or(0);
    assert!(out.expert_stride >= largest);
    assert_eq!(out.expert_stride % mrefrust_repack::GTURBO_PAGE_BYTES, 0);
}

#[test]
fn reports_ignored_tensors_rather_than_dropping_them_silently() {
    let f = Fixture::new();
    let h = &f.header;
    let out = orchestrate_gguf_checkpoint(h, &MemoryRangeSource::new(&f.bytes)).expect("walk");
    // The fixture carries no rope_freqs, so nothing is ignored; the field
    // exists so that when a real file does, the drop is visible.
    assert!(out.ignored.is_empty());

    let (bytes, _) = mrefrust_repack::GgufBuilder::new()
        .metadata_str("general.architecture", "gemma4")
        .metadata_u32("gemma4.block_count", 0)
        .metadata_u32("gemma4.embedding_length", 64)
        .metadata_u32("gemma4.attention.head_count", 4)
        .metadata_u32("gemma4.attention.head_count_kv", 2)
        .metadata_u32("gemma4.expert_count", 4)
        .metadata_u32("gemma4.expert_used_count", 2)
        .metadata_u32("gemma4.expert_feed_forward_length", 16)
        .metadata(
            "gemma4.attention.sliding_window_pattern",
            mrefrust_repack::GgufValue::Array(vec![]),
        )
        .q8_0_tensor("token_embd.weight", &[64, 128], 1)
        .tensor("rope_freqs.weight", 0, &[16], vec![0u8; 64])
        .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let out = orchestrate_gguf_checkpoint(&h, &MemoryRangeSource::new(&bytes)).expect("walk");
    assert_eq!(out.ignored.len(), 1);
    assert!(out.ignored[0].starts_with("rope_freqs.weight ("));
}

/// The written install must be readable as a layout and an index, and its
/// expert bytes on disk must still match the source.
#[test]
fn streamed_install_writes_expert_bytes_unchanged_to_disk() {
    let f = Fixture::new();
    let h = &f.header;
    let dir = tempdir();
    let arch = write_gguf_install_streamed(
        &dir,
        h,
        &MemoryRangeSource::new(&f.bytes),
        "gguf-fixture",
        |_| {},
    )
    .expect("write install");

    let layout = model_io::load_packed_experts_layout(
        &dir,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("layout.json");
    assert_eq!(layout.num_layers, f.shape.num_layers);
    assert_eq!(layout.experts_per_layer, f.shape.num_experts as usize);

    let index = model_io::load_resident_index(&dir.join("model_weights.bin")).expect("index");
    assert!(index
        .entries
        .contains_key("language_model.model.embed_tokens.weight"));
    assert_eq!(arch.num_experts, f.shape.num_experts as i64);

    // Read the layer files back and compare against the source GGUF.
    for layer in 0..f.shape.num_layers {
        let file = std::fs::read(dir.join("packed_experts").join(&layout.layers[layer].file))
            .expect("layer file");
        let fused = f.tensor(&format!("blk.{layer}.ffn_gate_up_exps.weight"));
        let experts = f.shape.num_experts as usize;
        let fused_per = fused.len() / experts;

        for e in 0..experts {
            let entry = layout.expert(layer, e);
            let gate = &entry.sub_tensors["gate"];
            let up = &entry.sub_tensors["up"];
            let read = |t: &model_io::SubTensorEntry| {
                let at = (entry.offset + t.offset) as usize;
                file[at..at + t.size as usize].to_vec()
            };
            let mut rejoined = read(gate);
            rejoined.extend_from_slice(&read(up));
            assert_eq!(
                rejoined,
                &fused[e * fused_per..(e + 1) * fused_per],
                "layer {layer} expert {e} on disk"
            );
        }
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// The manifest describes the bytes, and validation decides what runs. Both
/// halves are asserted here, because the promotion Stage 2 performed was
/// exactly a validation change and NOT a byte change: the walk writes the
/// same install it always did, and `load_manifest` now accepts the Q8_0 one
/// because kernels exist for it while still refusing a block type that has
/// none.
#[test]
fn the_written_manifest_describes_the_bytes_and_gates_on_the_kernels() {
    let f = Fixture::new();
    let h = &f.header;
    let dir = tempdir();
    let arch = write_gguf_install_streamed(
        &dir,
        h,
        &MemoryRangeSource::new(&f.bytes),
        "gguf-fixture",
        |_| {},
    )
    .expect("write install");

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["quant"]["routedExpert"]["scheme"], "gguf");
    assert_eq!(manifest["quant"]["routedExpert"]["ggmlType"], "Q8_0");
    // The router is the one slot that is NOT "gguf", because the transcode
    // really did make it INT8 affine at group 64. Saying "F32" here would
    // describe the source file rather than the bytes on disk. The refusal
    // below therefore has to come from the other four slots.
    assert_eq!(manifest["quant"]["router"]["scheme"], "affine");
    assert_eq!(manifest["quant"]["router"]["weightBits"], 8);
    assert_eq!(manifest["quant"]["router"]["groupSize"], 64);

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("a Q8_0 GGUF install loads: its kernels landed in Stage 2");

    // The same install, claiming a block type with no kernel behind it, is
    // still refused, and the refusal names the type rather than saying
    // "unsupported". Editing the manifest is how this is reached because the
    // fixture is Q8_0 throughout; `crates/runtime` covers the resident-index
    // backstop that catches the reverse forgery.
    let path = dir.join("manifest.json");
    let mut edited: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    // Q4_0, not Q4_K: Q4_K gained its routed pair and embedding lookup with
    // Qwen 3.6's Q4_K_M, so the type this reaches for has to be one the
    // parser knows and no kernel covers.
    edited["quant"]["routedExpert"]["ggmlType"] = serde_json::json!("Q4_0");
    std::fs::write(&path, serde_json::to_vec_pretty(&edited).unwrap()).unwrap();

    let err = model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect_err("a block type with no kernel must not load");
    let text = err.to_string();
    assert!(
        text.contains("Q4_0"),
        "the refusal should name the block type, got: {text}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
