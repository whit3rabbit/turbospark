//! Holds the knob table to the tree it measures. Offline, not ignored: a
//! renamed env var reddens here on the edit, instead of becoming an arm that
//! measures the default twice.

use std::path::{Path, PathBuf};

use turbospark_lab::knobs::{Compare, KNOBS, LANES};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_sources(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn every_knob_is_read_somewhere_outside_this_crate() {
    let root = repo_root();
    let mut files = Vec::new();
    for crate_dir in std::fs::read_dir(root.join("crates")).unwrap().flatten() {
        if crate_dir.file_name() == "lab" {
            continue;
        }
        rust_sources(&crate_dir.path().join("src"), &mut files);
    }
    let sources: Vec<String> = files
        .iter()
        .filter_map(|f| std::fs::read_to_string(f).ok())
        .collect();
    for knob in KNOBS {
        let literal = format!("\"{}\"", knob.env);
        assert!(
            sources.iter().any(|s| s.contains(&literal)),
            "{} ({}) is not read in crates/*/src any more",
            knob.env,
            knob.name
        );
    }
}

#[test]
fn every_knob_is_documented_where_its_row_says() {
    let root = repo_root();
    for knob in KNOBS {
        let doc = std::fs::read_to_string(root.join(knob.doc))
            .unwrap_or_else(|e| panic!("{} names {}: {e}", knob.name, knob.doc));
        assert!(
            doc.contains(knob.env),
            "{} does not mention {}",
            knob.doc,
            knob.env
        );
    }
}

#[test]
fn every_engagement_line_is_printed_somewhere() {
    let root = repo_root();
    let mut files = Vec::new();
    for crate_dir in std::fs::read_dir(root.join("crates")).unwrap().flatten() {
        if crate_dir.file_name() == "lab" {
            continue;
        }
        rust_sources(&crate_dir.path().join("src"), &mut files);
    }
    let sources: String = files
        .iter()
        .filter_map(|f| std::fs::read_to_string(f).ok())
        .collect();
    let mut lines: Vec<&str> = KNOBS
        .iter()
        .flat_map(|k| k.engagement.iter().map(|e| e.line))
        .collect();
    // The stand-in's lines are per block, checked by name in knobs.rs.
    lines.extend(["standin_attn", "standin_mlp", "standin_gdn"]);
    for line in lines {
        let call = format!("engaged!(\"{line}\")");
        assert!(
            sources.contains(&call),
            "no `{call}` in crates/*/src: the lab would refuse every arm that sets its knob"
        );
    }
}

#[test]
fn every_lane_names_a_knob_and_parity_tests_that_exist() {
    let root = repo_root();
    for lane in LANES {
        assert!(
            KNOBS.iter().any(|k| k.name == lane.knob),
            "lane {} is not a knob row",
            lane.knob
        );
        assert!(
            !lane.parity.is_empty(),
            "lane {} has no parity test: a kernel mode without one cannot claim an exactness class",
            lane.knob
        );
        for path in lane.parity {
            assert!(
                root.join(path).is_file(),
                "lane {} names parity test {path}, which does not exist",
                lane.knob
            );
        }
    }
    let mut names: Vec<&str> = LANES.iter().map(|l| l.knob).collect();
    names.sort_unstable();
    let n = names.len();
    names.dedup();
    assert_eq!(names.len(), n, "a knob has two lanes");
}

/// A lane's bench must be registered and its arm names must be arms the
/// bench emits, or `tslab project` would read a curve that is not there.
#[cfg(target_os = "macos")]
#[test]
fn every_lane_names_a_registered_bench_and_arms_it_emits() {
    use turbospark_lab::kernel::{arm_labels, by_name};
    for lane in LANES {
        assert!(
            by_name(lane.bench).is_some(),
            "lane {} names bench {}, which is not registered",
            lane.knob,
            lane.bench
        );
        let labels = arm_labels(lane.bench).unwrap_or_else(|| {
            panic!(
                "bench {} has no fixed arm list in kernel::arm_labels (lane {})",
                lane.bench, lane.knob
            )
        });
        let named: Vec<&str> = match lane.compare {
            Compare::Arms { baseline, lane } => vec![baseline, lane],
            Compare::Process { arm, baseline_env } => {
                assert!(
                    baseline_env.starts_with("TURBOSPARK_") && baseline_env.contains('='),
                    "{baseline_env}"
                );
                vec![arm]
            }
        };
        for arm in named {
            assert!(
                labels.iter().any(|l| l == arm),
                "lane {}: bench {} emits {labels:?}, not {arm}",
                lane.knob,
                lane.bench
            );
        }
    }
}

/// A Process lane's baseline assignment must be for the lane's own knob.
#[test]
fn a_process_lane_flips_its_own_knob() {
    for lane in LANES {
        if let Compare::Process { baseline_env, .. } = lane.compare {
            let knob = KNOBS.iter().find(|k| k.name == lane.knob).unwrap();
            assert!(
                baseline_env.starts_with(&format!("{}=", knob.env)),
                "lane {}: {baseline_env} is not an assignment of {}",
                lane.knob,
                knob.env
            );
        }
    }
}
