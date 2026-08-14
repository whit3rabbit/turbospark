//! The synthetic DENSE, ONE-BIT `qwen3_5` install, end to end through the
//! real repack walk and back out through every `turbospark_model_io` loader
//! (ROADMAP's 1-bit entry, step 3).
//!
//! **This is the fixture that exists so the 4.78 GiB stream is not the thing
//! that finds the holes** -- `crates/repack` Gotcha 8's rule, which M4's
//! dense `llama` half paid three five-minute re-streams for. Every assertion
//! below is one that a real download would otherwise have made, minutes at a
//! time.
//!
//! What it can and cannot see is worth stating. It CAN see: that the walk
//! writes an install with no packed-expert files, that the manifest it emits
//! is one `load_manifest` accepts, that the resident index tags 1-bit
//! tensors distinctly, that FP16 companions survive as FP16, and that every
//! UNQUANTIZED tensor is narrowed to BF16 (step 4's finding -- this fixture
//! wrote BF16 norms until then, where the real checkpoint writes F16). It
//! CANNOT see anything about the NUMBERS: the weights are untrained, so
//! `crates/runtime/tests/real_forward_qwen35.rs` asserts that a dense 1-bit
//! install decodes and nothing about what it decodes to.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{build_synthetic_qwen35_real_install, tiny_qwen35_arch};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen35-synthetic-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

const VOCAB: i64 = 256;
const LAYERS: i64 = 4;

fn build() -> (PathBuf, model_io::ArchConfig) {
    let dir = temp_dir();
    let arch = build_synthetic_qwen35_real_install(&dir, VOCAB, LAYERS, "qwen35-toy")
        .expect("the 1-bit dense install writes");
    (dir, arch)
}

/// The manifest the walk writes is one `load_manifest` accepts.
///
/// This is the single most valuable assertion in the file, because it is the
/// one M4 discovered by re-streaming: `manifest.quant` has five fixed slots,
/// a dense model has components for only two of them, and a slot that
/// answers wrongly makes a perfectly runnable install fail to open with a
/// message about something it never had.
#[test]
fn the_dense_one_bit_install_writes_a_manifest_that_loads() {
    let (dir, arch) = build();
    let manifest = model_io::load_manifest(&dir, &arch, 4 * 1024 * 1024)
        .expect("the manifest this walk wrote is one the loader accepts");
    // `numLayers` here is the PACKED-EXPERT layer file count, not the
    // model's depth: `load_manifest` uses it to check that a
    // `packed_experts/layer_NN.bin` exists for each. A dense install has
    // none, so 0 is correct and the model's 4 layers live in `arch`.
    assert_eq!(manifest.num_layers, 0);
    assert_eq!(manifest.experts_per_layer, 0);

    // Every slot reports the 1-bit shape, INCLUDING the three the model has
    // no component for. They fall back to the default width rather than to
    // `absent`, which is what makes the manifest loadable at all.
    let quant = manifest.quant.expect("a quant block was written");
    for (name, slot) in [
        ("embedding", &quant.embedding),
        ("attention", &quant.attention),
        ("router", &quant.router),
        ("sharedExpert", &quant.shared_expert),
        ("routedExpert", &quant.routed_expert),
    ] {
        assert_eq!(slot.weight_bits, 1, "{name}");
        assert_eq!(slot.scheme, "affine", "{name}");
        assert_eq!(slot.group_size, 128, "{name}");
        assert_eq!(slot.scale_type, "fp16", "{name}");
        assert_eq!(slot.bias_type, "fp16", "{name}");
    }
}

/// A dense install has ZERO packed-expert layer files, and its layout says
/// so rather than declaring layers that are not there.
#[test]
fn a_dense_install_has_no_packed_expert_files() {
    let (dir, _) = build();
    let layout =
        model_io::load_packed_experts_layout(&dir, 4 * 1024 * 1024).expect("layout.json parses");
    assert_eq!(layout.layers.len(), 0, "a dense model streams nothing");
    for l in 0..LAYERS {
        let path = dir.join(format!("packed_experts/layer_{l:02}.bin"));
        assert!(!path.exists(), "{} should not exist", path.display());
    }
}

/// The resident index tags 1-bit tensors with their OWN dtype, and the
/// companions come through as FP16.
///
/// Tagging them INT4 would be the plausible wrong answer: the entry shape is
/// the same three regions, so nothing structural objects. What would go
/// wrong is at dispatch, four layers of code later.
#[test]
fn one_bit_tensors_carry_their_own_dtype_tag_and_fp16_companions() {
    let (dir, _) = build();
    let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
        .expect("the resident index parses");

    let embed = index
        .entries
        .get("language_model.model.embed_tokens.weight")
        .expect("the embedding table is resident");
    // 15, the 1-bit affine tag: NOT 4 (INT4 affine) and NOT a GGUF block
    // tag, which start at 6.
    assert_eq!(embed.dtype, 15, "the embedding is not tagged 1-bit affine");

    // One bit per element: the packed run is rows * cols / 8 bytes, and
    // there is one FP16 scale and one FP16 bias per 128 elements.
    let (rows, cols) = (VOCAB as u64, 128u64);
    assert_eq!(embed.size_bytes, rows * cols / 8);
    assert_eq!(embed.scale_size, rows * (cols / 128) * 2);
    assert_eq!(embed.bias_size, embed.scale_size);

    // The dense FFN is quantized too, and nothing in the install is tagged
    // as a routed expert.
    let ffn = index
        .entries
        .get("language_model.model.layers.0.mlp.gate_proj.weight")
        .expect("the dense FFN is resident");
    assert_eq!(ffn.dtype, 15);
    assert!(
        !index.entries.keys().any(|k| k.contains("switch_mlp")),
        "a dense install wrote routed-expert tensors"
    );
    assert!(
        !index
            .entries
            .keys()
            .any(|k| k.ends_with(".mlp.gate.weight")),
        "a dense install wrote a router"
    );
}

/// The norms come out BF16 beside FP16 companions, which is the arrangement
/// that makes the dtype axis worth checking at all.
///
/// One install carries both widths for different roles: FP16 for the 1-bit
/// companions, which are read as FP16, and BF16 for the norms and the conv
/// kernel, which are read as BF16. A walk that resolved the companion dtype
/// once per install rather than per tensor would pass every length check here
/// and be wrong on one of the two. Note the two get there differently -- the
/// companions are passed through, the norms are NARROWED from the
/// checkpoint's F16 (see `every_unquantized_tensor_is_narrowed_to_bf16`).
#[test]
fn norms_stay_bf16_beside_the_fp16_companions() {
    let (dir, _) = build();
    let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
        .expect("the resident index parses");

    let norm = index
        .entries
        .get("language_model.model.norm.weight")
        .expect("the final norm is resident");
    // 1 = raw BF16, with no companions at all.
    assert_eq!(norm.dtype, 1);
    assert_eq!(norm.scale_size, 0);
    assert_eq!(norm.bias_size, 0);
    assert_eq!(norm.size_bytes, 128 * 2);

    let conv = index
        .entries
        .get("language_model.model.layers.0.linear_attn.conv1d.weight")
        .expect("layer 0 is linear and carries a conv kernel");
    assert_eq!(conv.dtype, 1);
}

/// EVERY unquantized tensor comes out tagged BF16, whatever the checkpoint
/// wrote -- and this fixture writes F16, like the real Bonsai-27B.
///
/// The walk used to record the SOURCE dtype (`raw_dtype_tag`, now deleted),
/// and nothing in `crates/runtime` reads tag 2 or tag 3: `norm_view`,
/// `read_bf16_host` and every kernel binding a `device const bfloat*`
/// identify an unquantized tensor by BYTE SIZE and decode it as BF16. F16 is
/// the same width, so an install carrying it opens, decodes, and is wrong on
/// every norm by up to 2^112 -- no error anywhere.
///
/// This is the case that would have caught it, and the reason it did not
/// exist before is worth keeping: the fixture was forked from the Qwen 3.6
/// one, whose checkpoint really is BF16, and only the QUANTIZED triple's
/// companions were re-read off the real header. A fixture copies the real
/// file's dtypes or it proves nothing about them.
#[test]
fn every_unquantized_tensor_is_narrowed_to_bf16() {
    let (dir, _) = build();
    let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
        .expect("the resident index parses");

    let raw: Vec<_> = index
        .entries
        .values()
        .filter(|e| e.scale_size == 0 && e.bias_size == 0)
        .collect();
    assert!(
        raw.len() >= 4 * LAYERS as usize,
        "expected the norms and the gated-DeltaNet tensors, got {}",
        raw.len()
    );
    for e in raw {
        assert_eq!(
            e.dtype, 1,
            "{} carries raw dtype {}, which no reader honours",
            e.name, e.dtype
        );
    }
}

/// The narrowing is LOSSY here and the count is reported rather than
/// swallowed, which is the half that matters when a quality number moves.
///
/// It is deliberately measured against a value that cannot survive: BF16 has
/// 7 stored mantissa bits against F16's 10, so a value needing more than 7
/// loses the rest. The real checkpoint's norms lose 19.5% of their values
/// this way and its gated-DeltaNet tensors lose none, because that QAT
/// checkpoint stores those on a grid coarse enough to be exact in both.
#[test]
fn narrowing_f16_to_bf16_counts_what_it_loses() {
    // 1 + 2^-10 needs ten mantissa bits: exact in F16, not in BF16.
    let exact_in_f16 = 1.0f32 + 2f32.powi(-10);
    let bytes = compute::f32_to_f16(exact_in_f16).to_le_bytes().to_vec();
    let narrowed = turbospark_repack::narrow_raw_to_bf16("probe", "F16", bytes).expect("narrows");
    assert_eq!(narrowed.dtype, 1, "narrowed bytes are BF16 bytes");
    assert_eq!(narrowed.bytes.len(), 2);
    assert_eq!(narrowed.lossy, 1, "this value cannot survive the narrowing");

    // A BF16 source is a pass-through, byte for byte, and never counts.
    let bf16_one = compute::f32_to_bf16(1.0).to_le_bytes().to_vec();
    let passed =
        turbospark_repack::narrow_raw_to_bf16("probe", "BF16", bf16_one.clone()).expect("passes");
    assert_eq!(passed.bytes, bf16_one);
    assert_eq!(passed.lossy, 0);

    // And a value that IS representable in both is narrowed without a count,
    // which is what keeps the counter a measurement rather than a dtype flag.
    let half = compute::f32_to_f16(0.5).to_le_bytes().to_vec();
    let clean = turbospark_repack::narrow_raw_to_bf16("probe", "F16", half).expect("narrows");
    assert_eq!(clean.lossy, 0);
}

/// The architecture the fixture declares is the one that comes back out,
/// and it is `qwen35` rather than the MoE sibling.
#[test]
fn the_install_declares_the_dense_family() {
    let (dir, arch) = build();
    assert_eq!(arch, tiny_qwen35_arch(VOCAB, LAYERS));
    assert_eq!(arch.family, model_io::ModelFamily::Qwen35);
    assert_eq!(arch.num_experts, 0);

    let family = model_io::peek_family(&dir, 4 * 1024 * 1024).expect("family peeks");
    assert_eq!(family, model_io::ModelFamily::Qwen35);
}

/// The family guard on the writer refuses an install written under the
/// wrong tag.
///
/// It matters more for this pair than any other in the registry: the walk
/// reads its routed marker and its quant probe names off `arch.family`, and
/// `qwen3_5` and `qwen3_5_moe` are one suffix apart, so the wrong tag
/// produces a well-formed install that looks for experts that do not exist.
#[test]
fn the_writer_refuses_a_mislabelled_arch() {
    let dir = temp_dir();
    let mut arch = tiny_qwen35_arch(VOCAB, LAYERS);
    arch.family = model_io::ModelFamily::Qwen36;

    // An EMPTY safetensors blob suffices: the guard runs before any tensor
    // is read, which is itself the property worth having -- a family check
    // that only fired after a multi-GB walk would be no use at all.
    let header_json = b"{}";
    let mut blob = (header_json.len() as u64).to_le_bytes().to_vec();
    blob.extend_from_slice(header_json);
    let header = turbospark_repack::parse_header(&blob, 1 << 20).expect("an empty header parses");
    let source = turbospark_repack::MemoryRangeSource::new(&blob);
    let quant = turbospark_repack::Gemma4Quant {
        default_bits: 1,
        group_size: 128,
        bits_overrides: Default::default(),
    };

    // `Gemma4RepackOutput` is not `Debug`, so match rather than
    // `expect_err`.
    match turbospark_repack::write_qwen35_install(&dir, &arch, "toy", &header, &source, &quant) {
        Ok(_) => panic!("a qwen36-tagged arch must be refused"),
        Err(err) => assert!(format!("{err}").contains("qwen35"), "{err}"),
    }
}
