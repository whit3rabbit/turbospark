//! Holds the family profiles to the tree: every `ModelFamily` and every
//! `AudioKind` has a profile, every family string a knob row or a profile
//! names is one the runtime writes, and a profile's `when` tags are tags
//! coverage can produce. Offline, synthetic architectures, no installs.
#![cfg(target_os = "macos")]

use std::collections::BTreeSet;

use model_io::ModelFamily;
use turbospark_lab::audio_kind::AudioKind;
use turbospark_lab::family_profile::{ProfileSet, TAGS, TAG_PREFIXES};
use turbospark_lab::knobs::KNOBS;

fn known_families() -> BTreeSet<&'static str> {
    ModelFamily::ALL
        .iter()
        .map(|f| f.as_str())
        .chain(AudioKind::ALL.iter().map(|k| k.as_str()))
        .collect()
}

#[test]
fn every_model_family_and_audio_kind_has_a_profile() {
    let set = ProfileSet::embedded().expect("embedded profiles parse");
    for f in ModelFamily::ALL {
        let p = set
            .by_family(f.as_str())
            .unwrap_or_else(|| panic!("no families/{}.json for {f:?}", f.as_str()));
        assert_eq!(p.kind, "text", "{f:?}");
        assert_eq!(
            p.flow,
            turbospark_lab::families::facts(f).flow,
            "{f:?}: the profile's flow disagrees with families.rs"
        );
    }
    for k in AudioKind::ALL {
        let p = set
            .by_family(k.as_str())
            .unwrap_or_else(|| panic!("no families/{}.json for {k:?}", k.as_str()));
        assert_eq!(p.kind, "audio", "{k:?}");
    }
    assert_eq!(
        set.profiles().len(),
        ModelFamily::ALL.len() + AudioKind::ALL.len(),
        "a profile exists for a family the runtime does not have"
    );
}

/// `knobs.rs` `families` strings are free text; this ties them to the
/// strings the runtime writes, so a renamed family reddens here.
#[test]
fn every_knob_family_is_a_family_the_runtime_writes() {
    let known = known_families();
    for knob in KNOBS {
        for fam in knob.families {
            assert!(
                known.contains(fam),
                "knob {} names family {fam:?}, which is neither a ModelFamily::as_str nor an AudioKind",
                knob.name
            );
        }
    }
}

/// A `when` tag that coverage never produces would silently skip a sweep
/// on every install.
#[test]
fn profile_when_tags_are_tags_coverage_can_produce() {
    use repack as r;
    let set = ProfileSet::embedded().expect("embedded profiles parse");
    // Every plain tag a coverage of some synthetic install produces, plus the
    // prefixes; the profile module's TAGS list must not drift from it.
    let mut produced: BTreeSet<String> = BTreeSet::new();
    for f in ModelFamily::ALL {
        let mut arch = match f {
            ModelFamily::Gemma4 => r::tiny_gemma4_arch(512, 4),
            ModelFamily::QwenGdnMoe => r::tiny_qwen_gdn_moe_arch(512, 4, 8),
            ModelFamily::QwenGdnDense | ModelFamily::DeepseekV4Flash => {
                r::tiny_qwen_gdn_dense_arch(512, 4)
            }
            ModelFamily::Llama | ModelFamily::Qwen3Dense | ModelFamily::Qwen2Dense => {
                r::tiny_dense_llama_arch(512, 4)
            }
            ModelFamily::Qwen3Moe | ModelFamily::MiniMaxM2 | ModelFamily::GptOss => {
                r::tiny_llama_arch(512, 4, 8)
            }
            ModelFamily::Qwen3Vl => r::tiny_qwen3_vl_arch(512, 4),
            ModelFamily::MuseGlimmer => r::tiny_muse_glimmer_arch(512, 4),
            ModelFamily::Qwen4Exp => r::tiny_qwen4_exp_arch(512, 4),
            ModelFamily::Spark25 => r::tiny_spark_arch(512, 4),
            ModelFamily::Deepseek2 => r::tiny_deepseek2_arch(512, 4),
            ModelFamily::Flint => r::tiny_flint_arch(512),
        };
        arch.family = f;
        let s = turbospark_lab::kernel::Subject {
            dir: std::env::temp_dir().join("tslab-family-profiles-no-install"),
            family: f.as_str().to_string(),
            arch: Some(arch),
            audio_config: None,
            inventory: Err("synthetic".into()),
        };
        let c = turbospark_lab::families::coverage(&s);
        assert!(
            c.has("profile"),
            "{f:?}: coverage did not attach the profile"
        );
        produced.extend(c.tags.iter().cloned());
    }
    for t in TAGS {
        let ok = produced.contains(*t)
            // Tags only a real install or an audio install yields.
            || matches!(
                *t,
                "audio" | "drafter" | "mtp" | "dflash" | "nomodel" | "vision" | "compressed_attention"
            )
            // Only a build with the split's feature yields it.
            || (*t == "ane" && !cfg!(feature = "ane-prefill"));
        assert!(ok, "TAGS lists {t:?}, which no synthetic coverage produced");
    }
    for p in set.profiles() {
        let whens = p
            .recommended
            .iter()
            .flat_map(|r| r.when.iter())
            .chain(p.sweeps.iter().flat_map(|s| s.when.iter()));
        for clause in whens {
            for alt in clause.split('|') {
                let t = alt.strip_prefix('!').unwrap_or(alt);
                let ok = TAGS.contains(&t) || TAG_PREFIXES.iter().any(|pre| t.starts_with(pre));
                assert!(ok, "{}: when tag {t:?} is unknown", p.family);
            }
        }
    }
}

/// The install on which a finding was taken should be one the profile lists,
/// or the agent reading it cannot tell which checkpoint the number is about.
#[test]
fn findings_name_installs_and_ledger_experiments_look_like_experiments() {
    let set = ProfileSet::embedded().expect("embedded profiles parse");
    for p in set.profiles() {
        for f in &p.findings {
            if f.provenance.kind == "ledger" {
                let exp = f.provenance.experiment.as_deref().unwrap();
                assert!(
                    turbospark_lab::paths::validate_name(exp).is_ok(),
                    "{}: experiment {exp:?}",
                    f.id
                );
            }
            if f.provenance.kind == "doc" {
                let path = f.provenance.path.as_deref().unwrap();
                let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
                assert!(
                    root.join(path).is_file(),
                    "{}: doc provenance {path} does not exist",
                    f.id
                );
                let heading = f.provenance.heading.as_deref().unwrap();
                let text = std::fs::read_to_string(root.join(path)).unwrap();
                assert!(
                    text.lines()
                        .any(|l| l.starts_with('#') && l.contains(heading)),
                    "{}: {path} has no heading containing {heading:?}",
                    f.id
                );
            }
        }
    }
}
