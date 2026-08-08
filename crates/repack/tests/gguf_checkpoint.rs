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
    edited["quant"]["routedExpert"]["ggmlType"] = serde_json::json!("Q4_K");
    std::fs::write(&path, serde_json::to_vec_pretty(&edited).unwrap()).unwrap();

    let err = model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect_err("a block type with no kernel must not load");
    let text = err.to_string();
    assert!(
        text.contains("Q4_K"),
        "the refusal should name the block type, got: {text}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
