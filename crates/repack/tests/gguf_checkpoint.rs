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
    write_gguf_install_streamed, MemoryRangeSource, ResidentEntrySpec, SyntheticGgufShape,
    DTYPE_FP32, DTYPE_GGUF_Q8_0, GGUF_DEFAULT_MAX_HEADER_BYTES,
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
            _ => panic!("GGUF tensors must be carried raw, never re-quantized"),
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

    // The router is F32 in GGUF where an MLX install carries INT8. Carried
    // through as F32 rather than transcoded.
    let ResidentEntrySpec::Raw(router) =
        by_name["language_model.model.layers.0.router.proj.weight"]
    else {
        unreachable!()
    };
    assert_eq!(router.dtype, DTYPE_FP32);
    assert_eq!(router.bytes, f.tensor("blk.0.ffn_gate_inp.weight"));
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

/// The Stage 1 boundary, enforced rather than documented: the manifest
/// declares `scheme: "gguf"`, no kernel reads those blocks, and
/// `load_manifest` refuses it. Promoting the artifact in Stage 2 is a
/// validation change, not a byte change.
#[test]
fn the_written_manifest_is_refused_until_kernels_exist() {
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
    // The router really is F32 in a GGUF, and the manifest says so.
    assert_eq!(manifest["quant"]["router"]["ggmlType"], "F32");

    let err = model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect_err("a GGUF install must not load in Stage 1");
    let text = err.to_string();
    assert!(
        text.contains("quantization"),
        "the refusal should name the quantization, got: {text}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
