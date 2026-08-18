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

use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install, build_synthetic_qwen_gdn_dense_install_at_bits,
    build_synthetic_qwen_gdn_dense_install_with_mtp,
    build_synthetic_qwen_gdn_dense_install_with_mtp_streamed, tiny_qwen_gdn_dense_arch,
};

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
    let arch = build_synthetic_qwen_gdn_dense_install(&dir, VOCAB, LAYERS, "qwen35-toy")
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

/// The same walk at TWO bits (ROADMAP's ternary entry): a dense 2-bit install
/// writes a manifest that loads, tags its tensors 16, and its packed run is
/// twice the 1-bit one's.
///
/// One test rather than a second copy of the three above, because the dense
/// path is not what varies here -- the WIDTH is, and it varies in exactly
/// three observable places: the manifest's `weightBits`, the resident dtype
/// tag, and the packed byte count. Everything else about the install is
/// asserted at one bit and shared.
#[test]
fn the_dense_two_bit_install_loads_and_is_tagged_apart_from_the_one_bit_one() {
    let dir = temp_dir();
    let arch =
        build_synthetic_qwen_gdn_dense_install_at_bits(&dir, VOCAB, LAYERS, "ternary-toy", 2)
            .expect("the 2-bit dense install writes");

    let manifest = model_io::load_manifest(&dir, &arch, 4 * 1024 * 1024)
        .expect("the manifest this walk wrote is one the loader accepts");
    assert_eq!(manifest.num_layers, 0, "a dense install streams nothing");
    let quant = manifest.quant.expect("a quant block was written");
    for (name, slot) in [
        ("embedding", &quant.embedding),
        ("attention", &quant.attention),
        ("router", &quant.router),
        ("sharedExpert", &quant.shared_expert),
        ("routedExpert", &quant.routed_expert),
    ] {
        assert_eq!(slot.weight_bits, 2, "{name}");
        assert_eq!(slot.scheme, "affine", "{name}");
        assert_eq!(slot.group_size, 128, "{name}");
        // FP16 like the 1-bit install and unlike every 4/8-bit one. The
        // manifest writer keys this on the DEFAULT bits, so a checkpoint that
        // moved 2 into the BF16 branch would produce an install of exactly
        // the right size that cannot open.
        assert_eq!(slot.scale_type, "fp16", "{name}");
        assert_eq!(slot.bias_type, "fp16", "{name}");
    }

    let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
        .expect("the resident index parses");
    let embed = index
        .entries
        .get("language_model.model.embed_tokens.weight")
        .expect("the embedding table is resident");
    // 16, the 2-bit affine tag: NOT 15 (1-bit), NOT 4 (INT4), and not a GGUF
    // block tag. All four entry shapes are identical, so nothing structural
    // objects to the wrong one -- the dispatch four layers later would read a
    // row of half or twice the columns and decode it perfectly.
    assert_eq!(embed.dtype, 16, "the embedding is not tagged 2-bit affine");
    let (rows, cols) = (VOCAB as u64, 128u64);
    assert_eq!(embed.size_bytes, rows * cols / 4);
    assert_eq!(embed.scale_size, rows * (cols / 128) * 2);
    assert_eq!(embed.bias_size, embed.scale_size);

    // The discriminating half: the SAME fixture at one bit writes half the
    // bytes for the same tensor, so the width really did reach the walk.
    let one_bit_dir = temp_dir();
    build_synthetic_qwen_gdn_dense_install(&one_bit_dir, VOCAB, LAYERS, "qwen35-toy")
        .expect("the 1-bit dense install writes");
    let one_bit = model_io::load_resident_index(&one_bit_dir.join("model_weights.bin"))
        .expect("the resident index parses");
    let one_bit_embed = &one_bit.entries["language_model.model.embed_tokens.weight"];
    assert_eq!(embed.size_bytes, 2 * one_bit_embed.size_bytes);
    assert_eq!(embed.scale_size, one_bit_embed.scale_size);
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
    assert_eq!(arch, tiny_qwen_gdn_dense_arch(VOCAB, LAYERS));
    assert_eq!(arch.family, model_io::ModelFamily::QwenGdnDense);
    assert_eq!(arch.num_experts, 0);

    let family = model_io::peek_family(&dir, 4 * 1024 * 1024).expect("family peeks");
    assert_eq!(family, model_io::ModelFamily::QwenGdnDense);
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
    let mut arch = tiny_qwen_gdn_dense_arch(VOCAB, LAYERS);
    arch.family = model_io::ModelFamily::QwenGdnMoe;

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
    match turbospark_repack::write_qwen_gdn_dense_install(
        &dir, &arch, "toy", &header, &source, &quant,
    ) {
        Ok(_) => panic!("a qwen36-tagged arch must be refused"),
        Err(err) => assert!(format!("{err}").contains("qwen35"), "{err}"),
    }
}

// -- The multi-token-prediction head (`docs/MTP_SPECULATIVE.md`, step 1) ----
//
// The head ships only in the OFFICIAL `Qwen/Qwen3.8-27B`; the mlx-community
// conversion the trunk install is streamed from drops it. So a real ingest
// reads TWO repositories at two dtypes, and the cheapest place to find out
// what that costs is here rather than twenty minutes into a stream.

fn build_with_mtp() -> (PathBuf, model_io::ArchConfig) {
    let dir = temp_dir();
    let arch = build_synthetic_qwen_gdn_dense_install_with_mtp(&dir, VOCAB, LAYERS, "mtp-toy", 1)
        .expect("the dense install with an MTP head writes");
    (dir, arch)
}

fn resident(dir: &std::path::Path) -> model_io::ResidentIndex {
    model_io::load_resident_index(&dir.join("model_weights.bin"))
        .expect("the resident index parses")
}

/// The walk ingests `mtp.*` and QUANTIZES it, rather than refusing it as an
/// unknown prefix or passing it through raw.
///
/// Both wrong answers are reachable and only one is loud. Refusing is what
/// the unmodified classifier does (`Gemma4Bucket::Unknown`), and it at least
/// fails by name. Passing the matrices through as raw BF16 is the quiet one:
/// the install would be well-formed, four times larger than it needs to be,
/// and would die at the first draft dispatch -- this port dispatches no
/// unquantized GEMV, which is exactly why `mtp_head_network.rs` bothers to
/// assert the head's source dtype.
#[test]
fn the_mtp_head_installs_quantized_beside_the_trunk() {
    let (dir, arch) = build_with_mtp();
    let manifest = model_io::load_manifest(&dir, &arch, 4 * 1024 * 1024)
        .expect("a head does not stop the manifest loading");
    assert_eq!(
        manifest.num_layers, 0,
        "the head is not a packed-expert layer"
    );

    let index = resident(&dir);
    let mtp: Vec<&String> = index
        .entries
        .keys()
        .filter(|k| k.starts_with("mtp."))
        .collect();
    assert_eq!(mtp.len(), 15, "the head's inventory moved: {mtp:?}");

    // The eight MATRICES are INT4-affine (tag 4): not raw BF16 (tag 1), and
    // not the trunk's 1-bit tag (15). Tag 4 regardless of the trunk's width
    // is the point -- the head is quantized BY THIS WALK from BF16, where the
    // trunk is passed through at whatever its publisher chose.
    for name in [
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    ] {
        let e = index
            .entries
            .get(name)
            .unwrap_or_else(|| panic!("{name} is not resident"));
        assert_eq!(e.dtype, 4, "{name} is not tagged INT4 affine");
        assert!(
            e.scale_size > 0 && e.bias_size == e.scale_size,
            "{name} lost a companion plane"
        );
    }

    // The seven NORMS stay unquantized BF16 (tag 1). Quantizing a norm to
    // INT4 is what `hf_checkpoint.rs` does, and its own header calls that out
    // as not how a production repacker would treat them.
    for name in [
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
    ] {
        let e = index
            .entries
            .get(name)
            .unwrap_or_else(|| panic!("{name} is not resident"));
        assert_eq!(e.dtype, 1, "{name} should stay unquantized BF16");
        assert_eq!(e.scale_size, 0, "{name} grew a companion plane");
    }
}

/// `fc` is the one tensor in the model whose INPUT is `2 * hidden`, and the
/// install records that width rather than transposing it.
///
/// The failure this catches is an axis swap. `fc` takes a CONCATENATION of
/// two `[hidden]` vectors, so a walk that recorded `[2 * hidden, hidden]`
/// would produce an entry of exactly the right BYTE COUNT with its axes
/// exchanged, and the draft step would then read the embedding half as the
/// hidden half.
///
/// **The byte count is therefore the wrong instrument, and this fixture
/// proves it rather than leaving it as a caution**: `fc` is `[128, 256]` and
/// `q_proj` is `[256, 128]`, the same 32,768 elements. The first draft of
/// this test asserted a size and its own uniqueness guard caught it. The real
/// head has no such collision (`fc` is 52.4M elements against `q_proj`'s
/// 62.9M), which is exactly how a fixture built to the real one's proportions
/// would have hidden the problem.
#[test]
fn the_mtp_fc_records_its_doubled_input_width() {
    let (dir, _) = build_with_mtp();
    let index = resident(&dir);
    let hidden = 128u32;

    let fc = index.entries.get("mtp.fc.weight").expect("fc is resident");
    let (r, c, _, _) = fc.shape;
    assert_eq!(
        (r, c),
        (hidden, 2 * hidden),
        "fc is not [hidden, 2*hidden]; a transposed read looks identical by size"
    );

    // The guard that makes the assertion above discriminating: at least one
    // other head tensor has fc's byte count, so a size check could not have
    // distinguished them and the SHAPE check is doing real work.
    let same_size: Vec<&String> = index
        .entries
        .iter()
        .filter(|(k, v)| {
            k.starts_with("mtp.") && v.size_bytes == fc.size_bytes && k.as_str() != "mtp.fc.weight"
        })
        .map(|(k, _)| k)
        .collect();
    assert!(
        !same_size.is_empty(),
        "no other head tensor shares fc's byte count, so this fixture cannot \
         demonstrate why the shape check is needed"
    );
}

/// The head's attention block is GATED, and `q_proj` against `o_proj` is
/// where that shows.
///
/// `q_proj` emits `2 * num_heads * head_dim` because half of it is the gate;
/// `o_proj` consumes the unhalved `num_heads * head_dim`. A head read as
/// non-gated would give both the same width, so a fixture with equal widths
/// could not see the difference -- which is why the real probe checks the
/// same pair off the published header.
#[test]
fn the_mtp_block_is_gated_like_the_trunks_full_layers() {
    let (dir, arch) = build_with_mtp();
    assert!(arch.attn_output_gate, "this check assumes a gated family");
    let index = resident(&dir);

    let q = index
        .entries
        .get("mtp.layers.0.self_attn.q_proj.weight")
        .expect("q_proj is resident");
    let o = index
        .entries
        .get("mtp.layers.0.self_attn.o_proj.weight")
        .expect("o_proj is resident");
    // q_proj is [2 * q_out, hidden]; o_proj is [hidden, q_out]. So q_proj is
    // exactly twice o_proj's packed size, and equal sizes would mean the gate
    // half went missing.
    assert_eq!(
        q.size_bytes,
        2 * o.size_bytes,
        "q_proj is not carrying the output gate"
    );
}

/// An install built WITHOUT the head has no `mtp.` tensors at all.
///
/// The guard is against a head that leaks into every install: it would be
/// dead weight in four checkpoints that have no drafter, and the manifest's
/// "absent means no head" rule would never be exercised.
#[test]
fn the_head_is_absent_unless_asked_for() {
    let (dir, _) = build();
    let index = resident(&dir);
    assert!(
        !index.entries.keys().any(|k| k.starts_with("mtp.")),
        "the default dense fixture grew an MTP head"
    );
}

/// **THE STREAMED WRITER CARRIES THE HEAD TOO, and this test exists because
/// its absence shipped a bug.**
///
/// Step 1's ingest landed in `orchestrate_gemma4_checkpoint_sharded`, which
/// is the NON-streamed path. Every fixture above goes through that one; every
/// REAL install goes through `write_gemma4_install_streamed`, which
/// classified `mtp.*` correctly and then never read `plan.mtp_bases`. So the
/// first real stream that asked for a head produced a byte-identical HEADLESS
/// install -- same 851 resident tensors, same 15,132,916,736-byte region --
/// with no error, no warning and nothing in the progress log to say so. It
/// cost a 15-minute stream to find and would have cost another to re-find.
///
/// `crates/repack` Gotcha 8 says build the fixture before the download. The
/// clause this adds: the fixture has to exercise the WRITER the download will
/// use. Two entry points differing only in which writer they call is the
/// cheapest way to say that, and asserting the two agree is what keeps them
/// from drifting again.
#[test]
fn both_writers_carry_the_mtp_head() {
    let streamed_dir = temp_dir();
    build_synthetic_qwen_gdn_dense_install_with_mtp_streamed(
        &streamed_dir,
        VOCAB,
        LAYERS,
        "mtp-toy",
        1,
    )
    .expect("the streamed writer builds a dense install with an MTP head");

    let streamed = model_io::load_resident_index(&streamed_dir.join("model_weights.bin"))
        .expect("resident index");
    let head: Vec<&String> = streamed
        .entries
        .keys()
        .filter(|k| k.starts_with("mtp."))
        .collect();
    assert_eq!(
        head.len(),
        15,
        "the STREAMED writer dropped the head: {head:?}"
    );
    assert!(streamed.entries.contains_key("mtp.fc.weight"));

    // And it agrees with the non-streamed writer tensor for tensor. The two
    // are documented to produce an identical install for a dense model, so a
    // divergence here is a real fork rather than a formatting difference.
    let (plain_dir, _) = build_with_mtp();
    let plain = model_io::load_resident_index(&plain_dir.join("model_weights.bin")).expect("index");
    let names = |i: &model_io::ResidentIndex| -> Vec<String> {
        let mut v: Vec<String> = i.entries.keys().cloned().collect();
        v.sort();
        v
    };
    assert_eq!(
        names(&streamed),
        names(&plain),
        "the two writers disagree about what a dense install contains"
    );
    for name in names(&plain) {
        assert_eq!(
            streamed.entries[&name].dtype, plain.entries[&name].dtype,
            "{name}: the two writers disagree about its dtype"
        );
        assert_eq!(
            streamed.entries[&name].size_bytes, plain.entries[&name].size_bytes,
            "{name}: the two writers disagree about its size"
        );
    }
}
