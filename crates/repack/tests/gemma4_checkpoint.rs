//! Gemma 4 checkpoint repack mapping: classification, resident ordering,
//! pre-quantized pass-through (INT4 + INT8 override), raw BF16 norms, and
//! per-expert blob slicing — exercised against a tiny in-memory
//! safetensors fixture with the real `mlx-community` tensor naming, then
//! round-tripped through every `turbospark_model_io` loader.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{
    classify_gemma4, is_supported_affine_shape, manifest_quant, orchestrate_gemma4_checkpoint,
    parse_gemma4_config, parse_gemma4_quantization, pass_through_packed, write_gemma4_install,
    Gemma4Bucket, Gemma4Error, Gemma4Shards, MemoryRangeSource, ResidentEntrySpec,
    AFFINE_1BIT_GROUP_SIZE, AFFINE_GROUP_SIZE,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-gemma4-checkpoint-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

const HIDDEN: usize = 64;
const VOCAB: usize = 128;
const LAYERS: usize = 2;
const EXPERTS: usize = 2;
const HEAD_DIM: usize = 32;

fn config_json() -> String {
    serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": HIDDEN,
            "intermediate_size": HIDDEN,
            "moe_intermediate_size": HIDDEN,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 2,
            "head_dim": HEAD_DIM,
            "global_head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "sliding_window": 16,
            "final_logit_softcapping": 30.0,
            "num_hidden_layers": LAYERS,
            "num_experts": EXPERTS,
            "top_k_experts": 1,
            "tie_word_embeddings": true,
            "attention_k_eq_v": true,
            "hidden_activation": "gelu_pytorch_tanh",
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": {"rope_theta": 1_000_000.0, "partial_rotary_factor": 0.25},
                "sliding_attention": {"rope_theta": 10_000.0}
            }
        },
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "language_model.model.layers.0.router.proj": {"group_size": 64, "bits": 8},
            "language_model.model.layers.1.router.proj": {"group_size": 64, "bits": 8}
        }
    })
    .to_string()
}

/// Deterministic filler so pass-through equality is checkable.
fn bytes_for(seed: usize, len: usize) -> Vec<u8> {
    (0..len).map(|i| ((seed * 31 + i) % 251) as u8).collect()
}

struct FixtureTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

fn quantized(name: &str, rows: usize, cols_packed: usize, seed: usize) -> Vec<FixtureTensor> {
    let base = name.strip_suffix(".weight").unwrap();
    let groups_per_row = |factor: usize| (cols_packed * factor) / 64;
    // factor inferred by the reader from the quant spec; the fixture only
    // needs consistent byte sizes, so compute groups from int4 (8) or
    // int8 (4) via the caller-chosen cols_packed.
    let factor = if base.ends_with("router.proj") { 4 } else { 8 };
    vec![
        FixtureTensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![rows as u64, cols_packed as u64],
            bytes: bytes_for(seed, rows * cols_packed * 4),
        },
        FixtureTensor {
            name: format!("{base}.scales"),
            dtype: "BF16",
            shape: vec![rows as u64, groups_per_row(factor) as u64],
            bytes: bytes_for(seed + 1, rows * groups_per_row(factor) * 2),
        },
        FixtureTensor {
            name: format!("{base}.biases"),
            dtype: "BF16",
            shape: vec![rows as u64, groups_per_row(factor) as u64],
            bytes: bytes_for(seed + 2, rows * groups_per_row(factor) * 2),
        },
    ]
}

fn expert_quantized(
    name: &str,
    experts: usize,
    rows: usize,
    cols_packed: usize,
    seed: usize,
) -> Vec<FixtureTensor> {
    let base = name.strip_suffix(".weight").unwrap();
    let groups = cols_packed * 8 / 64;
    vec![
        FixtureTensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![experts as u64, rows as u64, cols_packed as u64],
            bytes: bytes_for(seed, experts * rows * cols_packed * 4),
        },
        FixtureTensor {
            name: format!("{base}.scales"),
            dtype: "BF16",
            shape: vec![experts as u64, rows as u64, groups as u64],
            bytes: bytes_for(seed + 1, experts * rows * groups * 2),
        },
        FixtureTensor {
            name: format!("{base}.biases"),
            dtype: "BF16",
            shape: vec![experts as u64, rows as u64, groups as u64],
            bytes: bytes_for(seed + 2, experts * rows * groups * 2),
        },
    ]
}

fn raw(name: &str, shape: Vec<u64>, seed: usize) -> FixtureTensor {
    let len: u64 = shape.iter().product::<u64>().max(1) * 2;
    FixtureTensor {
        name: name.to_string(),
        dtype: "BF16",
        shape,
        bytes: bytes_for(seed, len as usize),
    }
}

/// Assembles a complete safetensors byte blob from the fixture tensors.
fn assemble(tensors: &[FixtureTensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut cursor = 0u64;
    for t in tensors {
        let end = cursor + t.bytes.len() as u64;
        header.insert(
            t.name.clone(),
            serde_json::json!({
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [cursor, end],
            }),
        );
        cursor = end;
    }
    let header_json = serde_json::Value::Object(header).to_string().into_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    for t in tensors {
        out.extend_from_slice(&t.bytes);
    }
    out
}

fn fixture() -> Vec<FixtureTensor> {
    let mut ts = Vec::new();
    ts.extend(quantized(
        "language_model.model.embed_tokens.weight",
        VOCAB,
        HIDDEN / 8,
        1,
    ));
    // A multimodal tower tensor: classified and excluded, never read.
    ts.push(raw("vision_tower.patch_embed.weight", vec![4], 999));
    for l in 0..LAYERS {
        let p = format!("language_model.model.layers.{l}");
        let seed = 100 * (l + 1);
        for (i, role) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
            ts.extend(quantized(
                &format!("{p}.self_attn.{role}.weight"),
                HIDDEN,
                HIDDEN / 8,
                seed + i * 3,
            ));
        }
        ts.push(raw(
            &format!("{p}.self_attn.q_norm.weight"),
            vec![HEAD_DIM as u64],
            seed + 20,
        ));
        ts.push(raw(
            &format!("{p}.self_attn.k_norm.weight"),
            vec![HEAD_DIM as u64],
            seed + 21,
        ));
        // INT8 router (per the config override): 4 values per u32 word.
        ts.extend(quantized(
            &format!("{p}.router.proj.weight"),
            EXPERTS,
            HIDDEN / 4,
            seed + 22,
        ));
        ts.push(raw(&format!("{p}.router.scale"), vec![1], seed + 25));
        ts.push(raw(
            &format!("{p}.router.per_expert_scale"),
            vec![EXPERTS as u64],
            seed + 26,
        ));
        for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            ts.extend(quantized(
                &format!("{p}.mlp.{role}.weight"),
                HIDDEN,
                HIDDEN / 8,
                seed + 30 + i * 3,
            ));
        }
        for (i, norm) in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
        .iter()
        .enumerate()
        {
            ts.push(raw(
                &format!("{p}.{norm}.weight"),
                vec![HIDDEN as u64],
                seed + 40 + i,
            ));
        }
        for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            ts.extend(expert_quantized(
                &format!("{p}.experts.switch_glu.{role}.weight"),
                EXPERTS,
                HIDDEN,
                HIDDEN / 8,
                seed + 50 + i * 3,
            ));
        }
    }
    ts.push(raw(
        "language_model.model.norm.weight",
        vec![HIDDEN as u64],
        7,
    ));
    ts
}

#[test]
fn config_parses_to_gemma4_arch() {
    let arch = parse_gemma4_config(&config_json()).expect("config parses");
    assert_eq!(arch.hidden_size, HIDDEN as i64);
    assert_eq!(arch.num_layers, LAYERS as i64);
    assert_eq!(arch.full_attention_layer_mask, vec![0u8, 1u8]);
    assert_eq!(arch.full_rope_theta, 1_000_000.0);
    assert_eq!(arch.rope_theta, 10_000.0);
    assert_eq!(arch.partial_rotary_factor, 0.25);
    assert!(arch.tie_word_embeddings);
    assert!(arch.attention_k_eq_v);
    assert!(arch.ffn_sandwich_norms);
    assert_eq!(arch.num_experts, EXPERTS as i64);
}

#[test]
fn quantization_overrides_parse() {
    let quant = parse_gemma4_quantization(&config_json()).expect("quant parses");
    assert_eq!(quant.default_bits, 4);
    assert_eq!(quant.group_size, 64);
    assert_eq!(
        quant
            .bits_overrides
            .get("language_model.model.layers.0.router.proj"),
        Some(&8)
    );
}

// ---------------------------------------------------------------------------
// The 1-bit affine shape (ROADMAP's 1-bit entry, step 3).
//
// These do NOT build a whole install. A 1-bit install is DENSE -- the one
// published checkpoint has no experts at all -- and a Gemma-shaped fixture
// would force a 1-bit routed-expert path that no kernel implements and no
// file asks for. So what is exercised here is the quantization plumbing
// alone: the config spec, and one tensor through `pass_through_packed`.
// The dense install fixture belongs with the family work.
// ---------------------------------------------------------------------------

/// A `quantization` object with an arbitrary global pair and no overrides.
fn quant_config_json(bits: u32, group: u32) -> String {
    serde_json::json!({
        "quantization": {"group_size": group, "bits": bits}
    })
    .to_string()
}

/// One quantized tensor plus its two companions, at an arbitrary bit width
/// and companion dtype so a case can vary exactly one of them.
fn packed_tensor(
    name: &str,
    rows: usize,
    cols: usize,
    bits: u32,
    companions: &str,
) -> Vec<FixtureTensor> {
    let base = name.strip_suffix(".weight").unwrap();
    let group = if bits == 1 { 128 } else { 64 };
    let words = cols * bits as usize / 32;
    let groups = cols / group;
    let companion = |suffix: &str, seed: usize| FixtureTensor {
        name: format!("{base}.{suffix}"),
        dtype: match companions {
            "fp16" => "F16",
            _ => "BF16",
        },
        shape: vec![rows as u64, groups as u64],
        bytes: bytes_for(seed, rows * groups * 2),
    };
    vec![
        FixtureTensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![rows as u64, words as u64],
            bytes: bytes_for(1, rows * words * 4),
        },
        companion("scales", 2),
        companion("biases", 3),
    ]
}

/// Runs one tensor through `pass_through_packed` under a given quant spec.
fn pass_one(bits: u32, group: u32, companions: &str) -> Result<ResidentEntrySpec, Gemma4Error> {
    const ROWS: usize = 4;
    const COLS: usize = 128; // a whole number of groups at 64 and at 128 alike
    let name = "language_model.model.layers.0.self_attn.q_proj.weight";
    let tensors = packed_tensor(name, ROWS, COLS, bits, companions);
    let blob = assemble(&tensors);
    let header = turbospark_repack::parse_header(&blob, 1 << 20).expect("fixture header parses");
    let source = MemoryRangeSource::new(&blob);
    let shards = Gemma4Shards::single(&header, &source);
    let quant = parse_gemma4_quantization(&quant_config_json(bits, group))
        .expect("the fixture's own spec parses");
    pass_through_packed(&shards, name, &quant)
}

/// The published 1-bit checkpoint's spec: `{group_size: 128, bits: 1}`, no
/// per-tensor overrides at all (which is itself a measured fact -- the file's
/// `quantization` object has exactly those two keys).
#[test]
fn a_one_bit_quantization_spec_parses() {
    let quant = parse_gemma4_quantization(&quant_config_json(1, 128)).expect("parses");
    assert_eq!(quant.default_bits, 1);
    assert_eq!(quant.group_size, AFFINE_1BIT_GROUP_SIZE);
    assert!(quant.bits_overrides.is_empty());
}

/// Bit width and group size are ONE shape. The four cross-products have no
/// kernel at either end of the pipeline and are refused at parse.
#[test]
fn the_cross_products_of_bits_and_group_size_are_refused() {
    for (bits, group) in [(1, 64), (4, 128), (8, 128), (2, 64)] {
        assert!(
            !is_supported_affine_shape(bits, group),
            "{bits}-bit at group {group} claims to be supported"
        );
        assert!(
            parse_gemma4_quantization(&quant_config_json(bits, group)).is_err(),
            "{bits}-bit at group {group} parsed"
        );
    }
    assert!(is_supported_affine_shape(4, AFFINE_GROUP_SIZE));
    assert!(is_supported_affine_shape(8, AFFINE_GROUP_SIZE));
    assert!(is_supported_affine_shape(1, AFFINE_1BIT_GROUP_SIZE));
}

/// A per-tensor override cannot straddle the two shapes.
///
/// This is what lets the manifest writer read the companion dtype off the
/// DEFAULT bits: an override may change 4 to 8 within group 64, but it can
/// never make one tensor 1-bit inside a group-64 checkpoint.
#[test]
fn a_per_tensor_override_that_leaves_the_group_size_behind_is_refused() {
    let json = serde_json::json!({
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "language_model.model.layers.0.self_attn.q_proj": {"bits": 1}
        }
    })
    .to_string();
    let err = parse_gemma4_quantization(&json).expect_err("1-bit at group 64 must be refused");
    let text = format!("{err:?}");
    assert!(text.contains("q_proj"), "{text}");
}

/// A 1-bit tensor passes through as `Int1`, 32 elements per packed word.
#[test]
fn a_one_bit_tensor_passes_through_as_int1() {
    match pass_one(1, 128, "fp16").expect("passes through") {
        ResidentEntrySpec::Int1(t) => {
            assert_eq!(t.rows, 4);
            // 4 packed u32 words per row * 32 elements each.
            assert_eq!(t.cols, 128);
            // One group of 128 per row.
            assert_eq!(t.scales.len(), 4);
            assert_eq!(t.biases.len(), 4);
        }
        other => panic!("expected Int1, got {other:?}"),
    }
}

/// **The companion dtype is required per width, and this is the axis that
/// fails silently.** FP16 and BF16 are the same width and share no exponent
/// field, so accepting either would produce an install of exactly the right
/// SIZE whose scales are wrong by orders of magnitude. Both directions are
/// refused, and the message names the dtype.
#[test]
fn the_companion_dtype_is_required_per_bit_width() {
    let err = pass_one(1, 128, "bf16").expect_err("BF16 on a 1-bit tensor must be refused");
    let text = format!("{err:?}");
    assert!(text.contains("BF16"), "{text}");
    assert!(text.contains("F16"), "{text}");

    let err = pass_one(4, 64, "fp16").expect_err("FP16 on a 4-bit tensor must be refused");
    assert!(format!("{err:?}").contains("F16"));
}

/// The INT4 path is unmoved: same variant, same dims, BF16 companions at
/// group 64.
#[test]
fn the_four_bit_pass_through_is_unmoved() {
    match pass_one(4, 64, "bf16").expect("passes through") {
        ResidentEntrySpec::Int4(t) => {
            assert_eq!((t.rows, t.cols), (4, 128));
            // Two groups of 64 per row.
            assert_eq!(t.scales.len(), 8);
        }
        other => panic!("expected Int4, got {other:?}"),
    }
}

/// The manifest the walk writes has to describe the bytes it wrote.
///
/// `manifest_quant` used to emit `bf16`/64 as literals, which was a true
/// statement about every install that existed and became false the moment a
/// 1-bit checkpoint could be walked. `model_io::validate_quant` reads these
/// three fields TOGETHER and accepts only `(4|8, bf16, 64)` and
/// `(1, fp16, 128)`, so a literal here is an install that cannot open.
#[test]
fn the_manifest_quant_block_reports_the_checkpoints_own_companions_and_group() {
    let one_bit = parse_gemma4_quantization(&quant_config_json(1, 128)).expect("parses");
    let json = manifest_quant(&one_bit, model_io::ModelFamily::Gemma4);
    for slot in [
        "embedding",
        "attention",
        "router",
        "sharedExpert",
        "routedExpert",
    ] {
        let s = &json[slot];
        assert_eq!(s["weightBits"], 1, "{slot}");
        assert_eq!(s["scaleType"], "fp16", "{slot}");
        assert_eq!(s["biasType"], "fp16", "{slot}");
        assert_eq!(s["groupSize"], 128, "{slot}");
    }

    // And the INT4 shape is unmoved, router still 8-bit.
    let int4 = parse_gemma4_quantization(&config_json()).expect("parses");
    let json = manifest_quant(&int4, model_io::ModelFamily::Gemma4);
    assert_eq!(json["attention"]["weightBits"], 4);
    assert_eq!(json["router"]["weightBits"], 8);
    assert_eq!(json["attention"]["scaleType"], "bf16");
    assert_eq!(json["attention"]["groupSize"], 64);
}

#[test]
fn classification_buckets() {
    assert_eq!(
        classify_gemma4("language_model.model.embed_tokens.weight", LAYERS),
        Gemma4Bucket::LmResident
    );
    assert_eq!(
        classify_gemma4(
            "language_model.model.layers.1.experts.switch_glu.up_proj.weight",
            LAYERS
        ),
        Gemma4Bucket::RoutedExpert {
            role: "up",
            layer: 1
        }
    );
    assert_eq!(
        classify_gemma4("vision_tower.patch_embed.weight", LAYERS),
        Gemma4Bucket::ExcludedMultimodal
    );
    assert_eq!(
        classify_gemma4("some_other_tower.weight", LAYERS),
        Gemma4Bucket::Unknown
    );
}

#[test]
fn orchestrate_orders_passes_through_and_slices_experts() {
    let tensors = fixture();
    let blob = assemble(&tensors);
    let source = MemoryRangeSource::new(&blob);
    let header = turbospark_repack::fetch_safetensors_header(&source).expect("header");
    let arch = parse_gemma4_config(&config_json()).expect("config");
    let quant = parse_gemma4_quantization(&config_json()).expect("quant");

    let out = orchestrate_gemma4_checkpoint(&header, &source, &arch, &quant).expect("orchestrate");

    // Ordering: embedding first, final norm last, layer groups in between.
    let names: Vec<&str> = out
        .resident
        .iter()
        .map(|e| match e {
            ResidentEntrySpec::Int4(t)
            | ResidentEntrySpec::Int8(t)
            | ResidentEntrySpec::Int1(t) => t.name.as_str(),
            ResidentEntrySpec::Raw(r) => r.name.as_str(),
        })
        .collect();
    assert_eq!(names[0], "language_model.model.embed_tokens.weight");
    assert_eq!(*names.last().unwrap(), "language_model.model.norm.weight");
    let pos = |n: &str| names.iter().position(|&x| x == n).expect(n);
    assert!(
        pos("language_model.model.layers.0.self_attn.q_proj.weight")
            < pos("language_model.model.layers.0.self_attn.k_proj.weight")
    );
    assert!(
        pos("language_model.model.layers.0.post_feedforward_layernorm.weight")
            < pos("language_model.model.layers.1.self_attn.q_proj.weight")
    );

    // Pass-through: embed packed bytes identical to the source bytes.
    let embed_src = &tensors[0];
    match &out.resident[0] {
        ResidentEntrySpec::Int4(t) => {
            assert_eq!(t.packed, embed_src.bytes);
            assert_eq!(t.rows, VOCAB as u32);
            assert_eq!(t.cols, HIDDEN as u32);
        }
        other => panic!("embed should be Int4 pass-through, got {other:?}"),
    }
    // The router carries the INT8 override.
    match &out.resident[pos("language_model.model.layers.0.router.proj.weight")] {
        ResidentEntrySpec::Int8(t) => {
            assert_eq!(t.rows, EXPERTS as u32);
            assert_eq!(t.cols, HIDDEN as u32);
        }
        other => panic!("router should be Int8 pass-through, got {other:?}"),
    }

    // Experts: both layers sliced, one page-rounded stride, per-expert
    // slices match the source layout (expert-major leading dimension).
    assert_eq!(out.layers.len(), LAYERS);
    assert_eq!(out.expert_stride % 16_384, 0);
    let layer0 = &out.layers[0];
    assert_eq!(layer0.experts.len(), EXPERTS);
    let gate_src = tensors
        .iter()
        .find(|t| t.name == "language_model.model.layers.0.experts.switch_glu.gate_proj.weight")
        .unwrap();
    let per_expert = gate_src.bytes.len() / EXPERTS;
    for (e, expert) in layer0.experts.iter().enumerate() {
        assert_eq!(expert.sub_tensors.len(), 9);
        assert_eq!(expert.sub_tensors[0].role, "gate");
        assert_eq!(
            expert.sub_tensors[0].bytes,
            gate_src.bytes[e * per_expert..(e + 1) * per_expert]
        );
        assert_eq!(expert.sub_tensors[6].role, "down");
    }
    assert_eq!(
        out.excluded_multimodal,
        vec!["vision_tower.patch_embed.weight"]
    );
}

#[test]
fn written_install_round_trips_through_model_io() {
    let tensors = fixture();
    let blob = assemble(&tensors);
    let source = MemoryRangeSource::new(&blob);
    let header = turbospark_repack::fetch_safetensors_header(&source).expect("header");
    let arch = parse_gemma4_config(&config_json()).expect("config");
    let quant = parse_gemma4_quantization(&config_json()).expect("quant");

    let dir = temp_dir();
    write_gemma4_install(
        &dir,
        &arch,
        "tiny-gemma4-real-naming",
        &header,
        &source,
        &quant,
    )
    .expect("write install");

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES).expect("manifest validates");
    let index = model_io::load_resident_index(&dir.join("model_weights.bin")).expect("index");
    let embed = &index.entries["language_model.model.embed_tokens.weight"];
    assert_eq!(embed.dtype, 4);
    assert_eq!(embed.shape.0, VOCAB as u32);
    let router = &index.entries["language_model.model.layers.0.router.proj.weight"];
    assert_eq!(router.dtype, 5);
    let q_norm = &index.entries["language_model.model.layers.0.self_attn.q_norm.weight"];
    assert_eq!(q_norm.dtype, 1);
    assert_eq!(q_norm.scale_size, 0);
    // Every entry's data offset is 4-byte aligned (the mixed writer pads).
    for entry in index.entries.values() {
        assert_eq!(entry.file_offset % 4, 0, "{} misaligned", entry.name);
    }

    let layout = model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024)
        .expect("packed experts layout");
    assert_eq!(layout.num_layers, LAYERS);
    assert_eq!(layout.experts_per_layer, EXPERTS);
    assert_eq!(layout.expert_stride % 16_384, 0);
    // Down-projection blob offset is 4-byte aligned (phase-2 uint loads).
    let down = &layout.layers[0].experts[0].sub_tensors["down"];
    assert_eq!(down.offset % 4, 0);
}

#[test]
fn sharded_orchestration_matches_single_source() {
    use turbospark_repack::{orchestrate_gemma4_checkpoint_sharded, Gemma4Shards};

    let tensors = fixture();
    // Split the fixture across two shards: layer 0 (plus top-level) in one,
    // layer 1 in the other, with companions following their weights.
    let (a, b): (Vec<_>, Vec<_>) = tensors.iter().partition(|t| t.name.contains(".layers.1."));
    let to_owned = |v: Vec<&FixtureTensor>| -> Vec<FixtureTensor> {
        v.into_iter()
            .map(|t| FixtureTensor {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                bytes: t.bytes.clone(),
            })
            .collect()
    };
    let shard_b = assemble(&to_owned(a));
    let shard_a = assemble(&to_owned(b));

    let full = assemble(&tensors);
    let arch = parse_gemma4_config(&config_json()).expect("config");
    let quant = parse_gemma4_quantization(&config_json()).expect("quant");

    let src_a = MemoryRangeSource::new(&shard_a);
    let src_b = MemoryRangeSource::new(&shard_b);
    let hdr_a = turbospark_repack::fetch_safetensors_header(&src_a).expect("header a");
    let hdr_b = turbospark_repack::fetch_safetensors_header(&src_b).expect("header b");
    let shards = Gemma4Shards::new(vec![(&hdr_a, &src_a), (&hdr_b, &src_b)]);
    let sharded = orchestrate_gemma4_checkpoint_sharded(&shards, &arch, &quant).expect("sharded");

    let src_full = MemoryRangeSource::new(&full);
    let hdr_full = turbospark_repack::fetch_safetensors_header(&src_full).expect("header full");
    let single =
        orchestrate_gemma4_checkpoint(&hdr_full, &src_full, &arch, &quant).expect("single");

    let names = |out: &turbospark_repack::Gemma4RepackOutput| -> Vec<String> {
        out.resident
            .iter()
            .map(|e| match e {
                ResidentEntrySpec::Int4(t)
                | ResidentEntrySpec::Int8(t)
                | ResidentEntrySpec::Int1(t) => t.name.clone(),
                ResidentEntrySpec::Raw(r) => r.name.clone(),
            })
            .collect()
    };
    assert_eq!(names(&sharded), names(&single));
    assert_eq!(sharded.expert_stride, single.expert_stride);
    assert_eq!(sharded.layers.len(), single.layers.len());
    for (ls, lf) in sharded.layers.iter().zip(single.layers.iter()) {
        for (es, ef) in ls.experts.iter().zip(lf.experts.iter()) {
            for (ss, sf) in es.sub_tensors.iter().zip(ef.sub_tensors.iter()) {
                assert_eq!(ss.bytes, sf.bytes, "layer {} sub {}", ls.layer, ss.role);
            }
        }
    }
}

#[test]
fn streamed_install_matches_in_memory_install() {
    use turbospark_repack::{write_gemma4_install_streamed, Gemma4Shards};

    let tensors = fixture();
    let blob = assemble(&tensors);
    let source = MemoryRangeSource::new(&blob);
    let header = turbospark_repack::fetch_safetensors_header(&source).expect("header");
    let arch = parse_gemma4_config(&config_json()).expect("config");
    let quant = parse_gemma4_quantization(&config_json()).expect("quant");

    let dir_mem = temp_dir();
    write_gemma4_install(&dir_mem, &arch, "streamed-vs-mem", &header, &source, &quant)
        .expect("in-memory install");
    let dir_str = temp_dir();
    let shards = Gemma4Shards::single(&header, &source);
    write_gemma4_install_streamed(&dir_str, &arch, "streamed-vs-mem", &shards, &quant, |_| {})
        .expect("streamed install");

    for file in [
        "model_weights.bin",
        "packed_experts/layout.json",
        "packed_experts/layer_00.bin",
        "packed_experts/layer_01.bin",
        "manifest.json",
    ] {
        let a = std::fs::read(dir_mem.join(file)).expect(file);
        let b = std::fs::read(dir_str.join(file)).expect(file);
        assert_eq!(a, b, "{file} differs between streamed and in-memory paths");
    }
}
