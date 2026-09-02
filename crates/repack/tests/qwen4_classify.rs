//! `classify_for_family` against `qwen4_exp`'s REAL tensor-name inventory.
//!
//! **THE FALLBACK IS THE HAZARD, WHICH IS WHY THIS FILE IS PATTERN-COMPLETE
//! RATHER THAN EXAMPLE-DRIVEN.** Every name in this checkpoint sits under
//! `language_model.`, and that branch's last line is `LmResident`. So a name
//! with no arm is not refused, it is silently filed into
//! `model_weights.bin` -- which for the n-gram shards is 32 GB, 30.8% of the
//! checkpoint, into a file whose resident core is 3.8 GB. That is
//! `crates/repack` Gotcha 15's unrecognized-marker failure ("loads and
//! generates fine, just with the whole expert table pinned") at a size where
//! it cannot load at all.
//!
//! The names below are the DISTINCT PATTERNS of all 3,215 tensors in
//! `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit`'s published
//! `model.safetensors.index.json`, with layer and shard numbers collapsed.
//! Read off the artifact rather than off the reference implementation, which
//! matters: the reference reads these names WITHOUT the `language_model.`
//! prefix, because its own `sanitize()` strips it on load. Taking the
//! reference's spelling would have matched nothing in the file (AGENTS.md
//! Gotcha 62 -- read the artifact, not the note about it).

use model_io::ModelFamily;
use turbospark_repack::{classify_for_family, Gemma4Bucket};

const LAYERS: usize = 48;

/// One representative of every distinct name pattern in the published index,
/// with `N` resolved to a real layer and `S` to a real shard.
///
/// Grouped by what each SHOULD classify as, so a reader can check the
/// expectation against the checkpoint rather than against the code.
fn every_pattern() -> Vec<(&'static str, &'static str)> {
    vec![
        // --- the table that must NOT be resident (384 tensors) ---
        (
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
            "shard:weight:0",
        ),
        (
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.scales",
            "shard:scales:0",
        ),
        (
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.biases",
            "shard:biases:0",
        ),
        (
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
            "shard:weight:127",
        ),
        // --- the three int64 buffers, which BF16 narrowing would destroy ---
        (
            "language_model.model.layers.1.ple.ple_embedding.layer_multipliers",
            "meta:layer_multipliers",
        ),
        (
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets",
            "meta:ngram_heads_offsets",
        ),
        (
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
            "meta:ngram_heads_vocab_sizes",
        ),
        // --- the routed experts (48 layers x 3 roles x 3 planes) ---
        (
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight",
            "routed:gate:1",
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.up_proj.weight",
            "routed:up:1",
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight",
            "routed:down:1",
        ),
        // --- everything else in the file is resident ---
        // The rest of the PLE block: ordinary per-layer weights that happen
        // to sit under `ple.`. The container check must NOT swallow them.
        (
            "language_model.model.layers.1.ple.key_proj.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.ple.value_proj.scales",
            "resident",
        ),
        (
            "language_model.model.layers.1.ple.norm_key.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.ple.norm_query.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.ple.norm_conv.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.ple.conv1d.weight",
            "resident",
        ),
        // Hyper-connections, two per layer plus the closing mixer.
        (
            "language_model.model.layers.1.attn_hyper_connection.hc_norm.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.attn_hyper_connection.input_mix_weight_down.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.attn_hyper_connection.input_mix_weight_up.scales",
            "resident",
        ),
        (
            "language_model.model.layers.1.attn_hyper_connection.block_inject_weight.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.mlp_hyper_connection.hc_norm.weight",
            "resident",
        ),
        (
            "language_model.model.hyper_connection_mixer.hc_norm.weight",
            "resident",
        ),
        (
            "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight",
            "resident",
        ),
        // Gated DeltaNet, 36 layers. `A_log` and `dt_bias` carry no `.weight`
        // suffix (Gotcha 15); `in_proj_a` / `in_proj_b` do and are BF16.
        (
            "language_model.model.layers.1.linear_attn.A_log",
            "resident",
        ),
        (
            "language_model.model.layers.1.linear_attn.dt_bias",
            "resident",
        ),
        (
            "language_model.model.layers.1.linear_attn.conv1d.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.linear_attn.in_proj_qkv.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.linear_attn.in_proj_z.scales",
            "resident",
        ),
        (
            "language_model.model.layers.1.linear_attn.in_proj_a.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.linear_attn.in_proj_b.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.linear_attn.norm.weight",
            "resident",
        ),
        (
            "language_model.model.layers.1.linear_attn.out_proj.weight",
            "resident",
        ),
        // Full attention, 12 layers, INCLUDING the indexer this port does not
        // implement. Carried rather than dropped: they are tiny, and an
        // install missing them would need a re-stream the day QSA lands.
        (
            "language_model.model.layers.3.self_attn.q_proj.weight",
            "resident",
        ),
        (
            "language_model.model.layers.3.self_attn.k_norm.weight",
            "resident",
        ),
        (
            "language_model.model.layers.3.self_attn.indexer.index_qk_proj.weight",
            "resident",
        ),
        (
            "language_model.model.layers.3.self_attn.indexer.q_layernorm.weight",
            "resident",
        ),
        (
            "language_model.model.layers.3.self_attn.indexer.k_layernorm.weight",
            "resident",
        ),
        // The shared expert and the two routers.
        (
            "language_model.model.layers.1.mlp.shared_expert.gate_proj.weight",
            "resident",
        ),
        ("language_model.model.layers.1.mlp.gate.weight", "resident"),
        (
            "language_model.model.layers.1.mlp.shared_expert_gate.weight",
            "resident",
        ),
        // Top level.
        ("language_model.model.embed_tokens.weight", "resident"),
        ("language_model.lm_head.weight", "resident"),
    ]
}

fn describe(bucket: &Gemma4Bucket) -> String {
    match bucket {
        Gemma4Bucket::NgramShard { role, shard } => format!("shard:{role}:{shard}"),
        Gemma4Bucket::NgramMeta { field } => format!("meta:{field}"),
        Gemma4Bucket::RoutedExpert { role, layer } => format!("routed:{role}:{layer}"),
        Gemma4Bucket::LmResident => "resident".to_string(),
        Gemma4Bucket::ExcludedMultimodal => "excluded".to_string(),
        Gemma4Bucket::MtpHead => "mtp".to_string(),
        Gemma4Bucket::DflashDrafter => "dflash".to_string(),
        Gemma4Bucket::VisionTower => "vision".to_string(),
        Gemma4Bucket::Unknown => "unknown".to_string(),
    }
}

#[test]
fn every_real_ngram_name_is_classified() {
    for (name, want) in every_pattern() {
        let got = describe(&classify_for_family(name, LAYERS, ModelFamily::Qwen4Exp));
        assert_eq!(got, want, "classifying {name}");
    }
}

/// **THE CONTAINER MUST NOT SWALLOW ITS SIBLINGS** -- `key_proj`,
/// `value_proj`, the three norms and `conv1d` all sit under
/// `...layers.1.ple.`, one segment above the table.
///
/// **THIS CASE IS AN INVARIANT RATHER THAN A GUARD, AND THAT IS RECORDED
/// HERE INSTEAD OF BEING TUNED AWAY.** Mutation-checked 2026-09-01: it
/// reddens under NEITHER of the two mutations it reads as protecting against.
/// Widening `NGRAM_CONTAINER` to `.ple.` leaves it green, and so does making
/// the buffer arm a catch-all. The reason is structural: entry to
/// `classify_qwen4_ngram`'s arms is `split_once(NGRAM_CONTAINER)`, and no
/// `...ple.key_proj.weight` contains `.ple.ple_embedding.` at all, so these
/// six names cannot reach the arms whatever the arms say.
///
/// What DOES catch a shallow container is `every_real_ngram_name_is_classified`
/// (the shards stop resolving, because their tail gains a
/// `ple_embedding.` prefix the shard marker does not expect), and what catches
/// a catch-all buffer arm is
/// `an_unrecognized_name_under_the_container_is_documented_as_falling_through`.
///
/// It is kept because it states the BOUNDARY a reader would otherwise have to
/// derive, and because it becomes a real guard the day the entry condition
/// stops being one `split_once` -- which is exactly the change that would
/// make the risk live. AGENTS.md's mutation rule: a case that reddens
/// everything, or nothing, usually means an invariant is doing the work, and
/// that is a finding.
#[test]
fn the_ple_container_does_not_swallow_the_layers_other_tensors() {
    for leaf in [
        "key_proj.weight",
        "value_proj.weight",
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
        "conv1d.weight",
    ] {
        let name = format!("language_model.model.layers.1.ple.{leaf}");
        assert_eq!(
            classify_for_family(&name, LAYERS, ModelFamily::Qwen4Exp),
            Gemma4Bucket::LmResident,
            "{name} is an ordinary per-layer weight, not part of the table"
        );
    }
}

/// The n-gram arm is FAMILY-GATED, so no other checkpoint can reach it.
///
/// The pairing matters in both directions. `qwen3_5` and `qwen3_5_moe` share
/// this classifier and carry no such tensor, so the arm firing for them would
/// mean a name collision this walk had misread; and `qwen4_exp` NOT firing is
/// the 32 GB failure the file's header describes.
#[test]
fn only_qwen4_exp_reaches_the_ngram_arm() {
    let name = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight";
    assert!(matches!(
        classify_for_family(name, LAYERS, ModelFamily::Qwen4Exp),
        Gemma4Bucket::NgramShard { .. }
    ));
    for other in [
        ModelFamily::QwenGdnMoe,
        ModelFamily::QwenGdnDense,
        ModelFamily::Gemma4,
        ModelFamily::MuseGlimmer,
    ] {
        assert_eq!(
            classify_for_family(name, LAYERS, other),
            Gemma4Bucket::LmResident,
            "{} must not reach the qwen4_exp n-gram arm",
            other.as_str()
        );
    }
}

/// An unrecognized plane or buffer under the container falls to `LmResident`,
/// which is the WRONG direction, so the arms are written to cover the
/// container exhaustively.
///
/// This case does not assert that is safe -- it PINS the fallback so a future
/// reader knows the shape of the risk, and so that adding a fourth plane to
/// the checkpoint reddens something. The published index carries exactly the
/// three planes and three buffers the arms name.
#[test]
fn an_unrecognized_name_under_the_container_is_documented_as_falling_through() {
    let bogus = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.quantiles";
    assert_eq!(
        classify_for_family(bogus, LAYERS, ModelFamily::Qwen4Exp),
        Gemma4Bucket::LmResident,
        "a fourth plane would be filed resident; the arms cover the three that exist"
    );
    let bogus = "language_model.model.layers.1.ple.ple_embedding.some_future_buffer";
    assert_eq!(
        classify_for_family(bogus, LAYERS, ModelFamily::Qwen4Exp),
        Gemma4Bucket::LmResident
    );
}
