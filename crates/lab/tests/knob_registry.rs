//! Holds the knob table to the tree it measures. Offline, not ignored: a
//! renamed env var reddens here on the edit, instead of becoming an arm that
//! measures the default twice.

use std::path::{Path, PathBuf};

use turbospark_lab::knobs::KNOBS;

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
