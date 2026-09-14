//! The tiny `muse_glimmer` install, end to end through the real walk.
//!
//! **THIS RUNS BEFORE THE 19.4 GB STREAM, DELIBERATELY.**
//! `crates/repack/CLAUDE.md` Gotcha 8 records what the alternative costs: the
//! dense `llama` bring-up found three manifest-slot bugs one five-minute
//! re-stream at a time, because every GGUF fixture was MoE and nothing had
//! ever asked what the walk writes when `plan.routed` is empty. The same
//! reasoning applies twice over here -- this family is dense AND multimodal,
//! so it exercises both the empty-routed path and the exclusion list, and a
//! hole in either costs 25 minutes to discover against the real file.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use model_io::ModelFamily;
use turbospark_repack::{build_synthetic_muse_glimmer_install, tiny_muse_glimmer_arch};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-muse-synthetic-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

const VOCAB: i64 = 64;
/// A multiple of 4, so the `[0, 0, 0, 1]` window pattern is whole and the
/// NoPE branch is covered. Eight gives two full periods.
const LAYERS: i64 = 8;

fn build() -> (PathBuf, model_io::ArchConfig) {
    let dir = temp_dir();
    let arch = build_synthetic_muse_glimmer_install(&dir, VOCAB, LAYERS, "muse-glimmer-toy")
        .expect("the install writes");
    (dir, arch)
}

#[test]
fn a_muse_glimmer_install_writes_and_its_manifest_loads() {
    let (dir, arch) = build();

    let manifest = model_io::load_manifest(&dir, &arch, 4 * 1024 * 1024)
        .expect("the manifest loads and validates");
    assert_eq!(manifest.arch.family.as_deref(), Some("museGlimmer"));
    assert_eq!(manifest.arch.num_layers, LAYERS);
    assert_eq!(arch.family, ModelFamily::MuseGlimmer);
}

/// A DENSE install writes no packed-expert files at all.
///
/// Asserted rather than assumed because the failure is silent in the
/// expensive direction: an unrecognized routed marker makes every expert a
/// RESIDENT tensor, which loads and generates fine at many times the
/// intended footprint (AGENTS.md Gotcha 26). Here there are no experts to
/// mis-file, so what this really pins is that the walk's empty-routed path
/// produces a loadable install rather than an empty `layout.json` the loader
/// refuses.
#[test]
fn a_dense_install_has_no_packed_experts() {
    let (dir, _) = build();

    let layers: Vec<_> = std::fs::read_dir(dir.join("packed_experts"))
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.ends_with(".bin"))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        layers.is_empty(),
        "a dense muse_glimmer install must write zero expert blobs, got {layers:?}"
    );
}

/// The resident index carries exactly the tensors the real checkpoint has,
/// and NOT the two it deliberately lacks.
///
/// The absences are the point. `q_norm` / `k_norm` do not exist because this
/// family's q/k norms are no-scale, and a flow that looked them up would
/// fail at open; `self_attn.gate_proj` DOES exist and is the separate
/// attention output gate, which is the tensor a reader coming from Qwen
/// would expect to find packed into `q_proj` instead.
#[test]
fn every_layer_carries_the_expected_tensor_set() {
    let (dir, _) = build();
    let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
        .expect("the resident index reads");

    let suffixes: BTreeSet<String> = index
        .entries
        .keys()
        .filter_map(|k| k.strip_prefix("language_model.model.layers.0."))
        .map(str::to_string)
        .collect();

    let expected: BTreeSet<String> = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.gate_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    assert_eq!(suffixes, expected);

    // Stated separately from the set equality above so a future edit that
    // widened `expected` could not quietly re-admit them.
    for absent in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
        assert!(
            !suffixes.contains(absent),
            "{absent} must not exist: this family's q/k norms are NO-SCALE"
        );
    }
}

/// The multimodal towers are EXCLUDED, and the walk says so rather than
/// silently dropping them.
///
/// The three prefixes were enumerated from the real
/// `model.safetensors.index.json` (806 `vision_tower.`, 6 `vision_adapter.`,
/// 3 `vision_projection.`). An unlisted prefix falls through to
/// `Gemma4Bucket::Unknown`, which the walk refuses -- the right failure mode
/// and an expensive one to hit, since nothing sees it until the stream
/// reaches the shard that tensor lives in.
#[test]
fn the_vision_towers_are_classified_as_excluded() {
    use turbospark_repack::{classify_for_family, Gemma4Bucket};
    for name in [
        "vision_tower.blocks.0.attn.qkv.weight",
        "vision_adapter.mlp.0.weight",
        "vision_projection.weight",
    ] {
        assert_eq!(
            classify_for_family(name, LAYERS as usize, ModelFamily::MuseGlimmer),
            Gemma4Bucket::ExcludedMultimodal,
            "{name} must be excluded, not left Unknown for the walk to refuse"
        );
    }
}

/// A repository-controlled tensor name cannot contradict the family's dense
/// architecture and enter the routed-expert layout path.
#[test]
fn routed_looking_tensors_are_refused_for_muse_glimmer() {
    use turbospark_repack::{classify_for_family, Gemma4Bucket};

    let name = "language_model.model.layers.0.experts.switch_glu.gate_proj.weight";
    assert_eq!(
        classify_for_family(name, LAYERS as usize, ModelFamily::MuseGlimmer),
        Gemma4Bucket::Unknown
    );
}

/// The fixture refuses a layer count that would leave the NoPE branch
/// uncovered.
///
/// A guard on the FIXTURE rather than on the architecture, and it earns its
/// place: with 3 layers the `[0,0,0,1]` pattern contains no full layer at
/// all, so every later test would pass while covering only half the flow.
#[test]
#[should_panic(expected = "window period is 4")]
fn a_layer_count_that_hides_the_nope_branch_is_refused() {
    tiny_muse_glimmer_arch(VOCAB, 3);
}

/// The window mask is the real pattern, and its full layers are where the
/// checkpoint says.
#[test]
fn the_window_pattern_matches_the_real_checkpoint() {
    let arch = tiny_muse_glimmer_arch(VOCAB, LAYERS);
    assert_eq!(arch.full_attention_layer_mask, vec![0, 0, 0, 1, 0, 0, 0, 1]);
    // And the NoPE half of it, which is the same fact read off a different
    // field.
    assert_eq!(arch.full_rope_theta, 0.0);
    assert_eq!(arch.rope_theta, 500_000.0);
}
