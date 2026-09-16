//! The `gpt-oss` decode flow (ROADMAP M5 step 4), on a synthetic install.
//!
//! `crates/runtime/CLAUDE.md` Gotcha 11 is the reason this file is shaped the
//! way it is. A synthetic fixture's weights are UNTRAINED, so its output is
//! meaningless by construction and "it decoded" is nearly worthless as
//! evidence: a wrong norm, a dropped bias or an ignored sink all produce
//! meaningless output too. `5279c88` shipped a Qwen whose perplexity read
//! 255,409 with the whole workspace suite green for exactly that reason.
//!
//! So the assertions here are not "it runs". They are **DOES THIS INPUT REACH
//! THE MATH** -- for each of the four things that make `gpt-oss` a fifth flow,
//! perturb that tensor in the install and require the logits to move. A flow
//! that silently drops a bias, passes `None` where sinks belong, or reads an
//! unscaled rope table is INVARIANT to those perturbations, and that is a
//! property a fixture CAN see even though it cannot judge the text.
//!
//! Patching the install in place rather than rebuilding it is AGENTS.md
//! Gotcha 33's trick: resident tensors sit at fixed offsets, every
//! perturbation preserves length, and `open()` runs no receipt or SHA-256
//! check, so a hypothesis costs milliseconds.
//!
//! **WHAT THIS FILE IS BLIND TO, measured rather than assumed.** Mutating
//! `attn.rs` to pass a literal `1.0` where it passes `state.rope_mscale`
//! leaves all six tests below GREEN. YaRN's magnitude scale is a function of
//! the factor alone, so every perturbation that moves the frequency table
//! moves the scale with it and no config difference isolates the two. The
//! value and the argument order are pinned instead by a unit test in
//! `state.rs`; that a fluent-but-wrong model results is a question only the
//! real-model quality gate can answer. Four mutations WERE checked and do
//! redden: dropping the sinks, dropping any one of the four projection
//! biases (the failure names which), dropping the router bias, and ignoring
//! the frequency table.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{
    build_synthetic_gpt_oss_gguf, parse_gguf_header, write_gguf_install_streamed,
    MemoryRangeSource, SyntheticGptOssShape, GGUF_DEFAULT_MAX_HEADER_BYTES,
};
use turbospark_runtime::{LogitProducer, RealForwardRunner};

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("turbospark-gptoss-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn gpt_oss_install() -> (
    std::path::PathBuf,
    model_io::ArchConfig,
    SyntheticGptOssShape,
) {
    let shape = SyntheticGptOssShape::default();
    let (bytes, _) = build_synthetic_gpt_oss_gguf(shape);
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    let dir = tempdir();
    let arch = write_gguf_install_streamed(
        &dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        "gptoss-m5",
        |_| {},
    )
    .expect("write install");
    (dir, arch, shape)
}

/// Decodes `steps` greedy tokens and returns the last step's logits.
///
/// `steps` is deliberately larger than the fixture's 8-token sliding window,
/// so the EVEN layers wrap their ring and the odd ones do not. A window bug
/// that only bites past the window is invisible below it, which is the whole
/// reason `gpt_oss_layer_mask`'s phase is worth refusing rather than
/// tolerating.
fn decode(dir: &std::path::Path, arch: &model_io::ArchConfig, steps: usize) -> Vec<f32> {
    let vocab = arch.vocab_size as usize;
    let mut runner = RealForwardRunner::open(dir, arch.clone()).expect("a gpt-oss install opens");
    runner.reset();
    let mut token = 5i32;
    let mut last = Vec::new();
    for position in 0..steps {
        let mut logits = vec![f16::from_f32(0.0); vocab];
        runner
            .produce(token, position, &mut logits)
            .unwrap_or_else(|e| panic!("produce failed at position {position}: {e}"));
        let bad: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.to_f32().is_finite())
            .map(|(i, _)| i)
            .collect();
        assert!(
            bad.is_empty(),
            "non-finite logit at position {position}: {} of {vocab}, first {:?}",
            bad.len(),
            &bad[..bad.len().min(8)]
        );
        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
        last = logits.iter().map(|v| v.to_f32()).collect();
    }
    last
}

/// Overwrites one resident tensor's bytes in place, leaving its length and
/// every offset around it untouched.
fn patch(dir: &std::path::Path, name: &str, value: u16) {
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("index");
    let entry = index
        .entries
        .get(name)
        .unwrap_or_else(|| panic!("{name} is not in the resident index"));
    let mut bytes = std::fs::read(&path).unwrap();
    let start = entry.file_offset as usize;
    let end = start + entry.size_bytes as usize;
    for chunk in bytes[start..end].chunks_exact_mut(2) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&path, bytes).unwrap();
}

/// Rewrites one `manifest.json -> arch` field in place.
fn patch_manifest(dir: &std::path::Path, field: &str, value: serde_json::Value) {
    let path = dir.join("manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert!(
        m["arch"].get(field).is_some(),
        "manifest has no arch field `{field}`"
    );
    m["arch"][field] = value;
    std::fs::write(&path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
}

/// How far apart two logit vectors are. Used only as "did this move at all",
/// never as a numeric claim: the weights are untrained.
fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

#[test]
fn a_gpt_oss_install_opens_and_decodes_past_its_sliding_window() {
    let (dir, arch, shape) = gpt_oss_install();
    // EVEN LAYERS SLIDE. Asserted here as well as in the repack test because
    // this is the consumer: `KvCacheManager` sizes the ring off this mask and
    // `attn.rs` reads it per layer.
    assert_eq!(arch.full_attention_layer_mask, vec![0u8, 1]);

    let logits = decode(&dir, &arch, 20);
    assert!(
        20 > shape.sliding_window as usize,
        "the walk must cross the window for this test to mean anything"
    );
    let first = logits[0];
    assert!(
        logits.iter().any(|v| *v != first),
        "every logit is {first}: the head produced nothing"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// THE ATTENTION SINKS REACH THE SOFTMAX.
///
/// A sink is one learned logit per query head added to the DENOMINATOR, so
/// driving it far positive drains almost all probability mass away from the
/// value rows and collapses the attention output toward zero. A flow that
/// passed `None`, or bound the buffer without setting `FC_ATTN_HAS_SINKS`,
/// would be exactly invariant to this.
#[test]
fn the_attention_sinks_change_the_output() {
    let (dir, arch, _) = gpt_oss_install();
    let before = decode(&dir, &arch, 6);
    // BF16 8.0. Large against logits scaled by 0.125, so `exp(sink)` dominates
    // the denominator on every head.
    for layer in 0..arch.num_layers as usize {
        patch(
            &dir,
            &format!("language_model.model.layers.{layer}.self_attn.sinks.weight"),
            0x4100,
        );
    }
    let after = decode(&dir, &arch, 6);
    assert!(
        distance(&before, &after) > 0.0,
        "a saturating attention sink left the logits bit-identical, so the sink buffer is \
         not reaching the softmax denominator"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// THE PER-PROJECTION BIASES REACH THE PROJECTIONS.
///
/// Checked one projection at a time rather than all four at once, because
/// "some bias moved the logits" is satisfied by a flow that wires one and
/// drops three -- and a dropped bias is the failure this architecture invites,
/// since no GEMV kernel in this port takes one and all four are separate
/// passes.
#[test]
fn every_projection_bias_changes_the_output() {
    let (dir, arch, _) = gpt_oss_install();
    let before = decode(&dir, &arch, 4);
    for suffix in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        let (dir2, arch2, _) = gpt_oss_install();
        // BF16 1.0, against a fixture whose biases are small random values.
        for layer in 0..arch2.num_layers as usize {
            patch(
                &dir2,
                &format!("language_model.model.layers.{layer}.self_attn.{suffix}.bias"),
                0x3F80,
            );
        }
        let after = decode(&dir2, &arch2, 4);
        assert!(
            distance(&before, &after) > 0.0,
            "rewriting every {suffix}.bias left the logits bit-identical, so that bias is \
             not being added"
        );
        std::fs::remove_dir_all(&dir2).ok();
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// THE ROUTER BIAS REACHES THE TOP-K.
///
/// It is added on the host between the readback and the selection, which is
/// the one place it can go: after the top-k it would select the wrong experts
/// and still produce fluent text, and after the softmax it would be a
/// different distribution over the right ones. Driving one expert's bias far
/// positive pins that expert into every layer's top-4, which no other
/// perturbation of the install can do.
#[test]
fn the_router_bias_changes_the_selected_experts() {
    let (dir, arch, _) = gpt_oss_install();
    let before = decode(&dir, &arch, 4);
    // BF16 64.0 on EVERY expert would be a no-op under a softmax over the
    // selected, so this writes a uniform value and relies on the top-k being
    // taken on RAW logits -- where a constant shift is also a no-op. Patch a
    // single expert instead by writing the whole vector and then zeroing all
    // but one entry.
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("index");
    let mut bytes = std::fs::read(&path).unwrap();
    for layer in 0..arch.num_layers as usize {
        let name = format!("language_model.model.layers.{layer}.mlp.gate.bias");
        let entry = &index.entries[&name];
        let start = entry.file_offset as usize;
        let end = start + entry.size_bytes as usize;
        for (i, chunk) in bytes[start..end].chunks_exact_mut(2).enumerate() {
            // BF16 64.0 on expert 0 alone, 0.0 everywhere else.
            let v: u16 = if i == 0 { 0x4280 } else { 0x0000 };
            chunk.copy_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(&path, bytes).unwrap();

    let after = decode(&dir, &arch, 4);
    assert!(
        distance(&before, &after) > 0.0,
        "pinning expert 0 into every layer's top-k left the logits bit-identical, so the \
         router bias is not being added before the selection"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// YaRN REACHES ROPE.
///
/// The frequency table is built at open from `arch.rope_scaling`, so an
/// install whose declared factor changes must decode differently -- a flow
/// that fell back to a scalar-theta rope kernel, or that built the table from
/// the BASELINE instead of the manifest, would not notice.
///
/// This one edits the MANIFEST rather than a tensor, because YaRN is metadata
/// and has no bytes in `model_weights.bin` to perturb. Both sides have to
/// move together: `arch_validation` compares the passed `ArchConfig` against
/// the manifest field by field, so changing only the struct is refused at
/// load rather than reaching the flow.
#[test]
fn the_yarn_scaling_factor_changes_the_output() {
    let (dir, arch, _) = gpt_oss_install();
    let before = decode(&dir, &arch, 6);

    // A different factor moves both the ramp and the magnitude scale. 8.0 is
    // a binary fraction, so it survives serde_json's ~1-ULP default parser
    // (AGENTS.md Gotcha 24).
    let mut scaled = arch.clone();
    scaled.rope_scaling.factor = 8.0;
    patch_manifest(&dir, "ropeScalingFactor", serde_json::json!(8.0));
    let after = decode(&dir, &scaled, 6);
    assert!(
        distance(&before, &after) > 0.0,
        "changing the YaRN factor left the logits bit-identical, so the frequency table is \
         not reaching the rope kernel"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// `open` returns the runner itself on success, which is not `Debug`, so the
/// error has to be pulled out by hand.
fn refusal(dir: &std::path::Path, arch: model_io::ArchConfig, what: &str) -> String {
    match RealForwardRunner::open(dir, arch) {
        Ok(_) => panic!("{what} must be refused, but it opened"),
        Err(e) => format!("{e}"),
    }
}

/// The refusals `RealGptOssState::build` owes, each of which would otherwise
/// be a model that decodes and is wrong.
///
/// **EVERY ONE OF THEM IS A BACKSTOP, NOT THE FIRST GATE**, and that is why
/// each case here patches `manifest.json` to AGREE with the doctored config
/// first. `arch_validation` compares the passed `ArchConfig` against the
/// manifest field by field, so a struct edited on its own is refused at load
/// with a field-mismatch message and the flow never runs. Making the two
/// agree is exactly the state a hand-edited install would be in, which is the
/// state a backstop exists for -- the same relationship
/// `EXECUTABLE_GGUF_DTYPES` has to `EXECUTABLE_GGUF_TYPES` (AGENTS.md Gotcha
/// 29's twins-and-not-copies).
#[test]
fn a_gpt_oss_install_is_refused_when_its_declared_shape_is_impossible() {
    // MISMATCHED KV DIMENSIONS. The gpt-oss attention flow uses the full pair
    // on every layer, while the cache sizes sliding layers from the sliding
    // pair. Accepting different pairs would make a crafted install panic at
    // the first sliding-layer attention dispatch.
    let (dir, arch, _) = gpt_oss_install();
    let mut mismatched_kv = arch.clone();
    mismatched_kv.num_kv_heads = 1;
    mismatched_kv.head_dim = 1;
    patch_manifest(&dir, "numKVHeads", serde_json::json!(1));
    patch_manifest(&dir, "headDim", serde_json::json!(1));
    let err = refusal(&dir, mismatched_kv, "gpt-oss with mismatched KV dimensions");
    assert!(
        err.contains("identical sliding and full KV dimensions"),
        "the refusal must name the dimensional invariant, got: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();

    // A TIED HEAD. gpt-oss ships `output.weight`; an install claiming
    // otherwise would silently read the embedding table as a head.
    let (dir, arch, _) = gpt_oss_install();
    let mut tied = arch.clone();
    tied.tie_word_embeddings = true;
    patch_manifest(&dir, "tieWordEmbeddings", serde_json::json!(true));
    let err = refusal(&dir, tied, "a tied gpt-oss");
    assert!(
        err.contains("untied"),
        "the refusal must name the tie, got: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();

    // NO YaRN. Unscaled frequencies are fluent and wrong past the original
    // context, which no short smoke reaches.
    let (dir, arch, _) = gpt_oss_install();
    let mut no_yarn = arch.clone();
    no_yarn.rope_scaling.factor = 0.0;
    patch_manifest(&dir, "ropeScalingFactor", serde_json::json!(0.0));
    let err = refusal(&dir, no_yarn, "a gpt-oss with no YaRN");
    assert!(
        err.contains("YaRN"),
        "the refusal must name the scaling, got: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();

    // NO DENSE HALF. Unlike `llama`, this architecture string covers one
    // shape only, so a zero expert count is malformed rather than a second
    // model to serve.
    //
    // The refusal that lands is the open-time expert-count cross-check, not
    // the family gate's own "MoE-only" line: for an install that carries
    // expert blobs, the layout lattice (experts-per-layer agreement, per-layer
    // blob presence, numLayers agreement) closes every route to the family
    // gate before `RealGptOssState::build` runs, no matter how the layout is
    // patched to agree. The family line stays for a resident-expert install;
    // for this streamed fixture the cross-check is the by-name refusal a
    // relabelled install actually hits, and it names the same disagreement.
    let (dir, arch, _) = gpt_oss_install();
    let mut dense = arch.clone();
    dense.num_experts = 0;
    dense.top_k_experts = 0;
    patch_manifest(&dir, "numExperts", serde_json::json!(0));
    patch_manifest(&dir, "topKExperts", serde_json::json!(0));
    let err = refusal(&dir, dense, "a dense gpt-oss");
    assert!(
        err.contains("experts per layer, but the architecture declares"),
        "the refusal must name the layout/architecture expert-count disagreement, got: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
