//! Checks the Inkling 2 GGUF name table (`inkling2_gguf_to_hf`) against a real
//! GGUF's tensor list and an install made from the HF checkpoint.
//!
//! ```text
//! cargo run --release -p turbospark-repack --example inkling2_gguf_names -- \
//!     <tensor list> <reference install> [core_start, default 6]
//! ```
//!
//! The tensor list is one `name dtype dims` line per tensor (from the GGUF
//! header). Every text tensor must map; the mapped names, with the routed
//! experts and the synthetic per-expert scales set aside and the message
//! queries recombined, must equal the reference install's resident names.

use std::collections::BTreeSet;
use std::path::Path;

use turbospark_repack::{inkling2_gguf_to_hf, Inkling2GgufTransform as T};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        eprintln!(
            "usage: {} <tensor list> <reference install> [core_start]",
            args[0]
        );
        std::process::exit(2);
    }
    let core_start: usize = args.get(3).map_or(6, |s| s.parse().unwrap());
    let list = std::fs::read_to_string(&args[1]).unwrap();
    let index =
        model_io::load_resident_index(&Path::new(&args[2]).join("model_weights.bin")).unwrap();
    let want: BTreeSet<String> = index.entries.keys().cloned().collect();
    let (mut have, mut unmapped, mut experts, mut msg) =
        (BTreeSet::new(), Vec::new(), 0usize, 0usize);
    for line in list.lines() {
        let name = line.split_whitespace().next().unwrap();
        if name.starts_with("v.") || name.starts_with("mm.") {
            continue;
        }
        match inkling2_gguf_to_hf(name, core_start) {
            None => unmapped.push(name.to_string()),
            Some((_, T::ExpertGateUpSplit | T::ExpertDown | T::ExpertScale)) => experts += 1,
            Some((_, T::MessageQueries { .. })) => msg += 1,
            Some((hf, _)) => {
                assert!(have.insert(hf.clone()), "{name}: {hf} mapped twice");
            }
        }
    }
    if msg > 0 {
        // The master convention's per-layer queries recombine into these two.
        have.insert("model.llm.middle_cycle_cross_pass_queries".into());
        have.insert("model.llm.middle_cycle_cross_pass_kv_queries".into());
    }
    let missing: Vec<_> = want.difference(&have).collect();
    let extra: Vec<_> = have.difference(&want).collect();
    eprintln!(
        "{}: {} text tensors mapped, {experts} expert tensors, {msg} message-query tensors; \
         unmapped {unmapped:?}; missing {missing:?}; extra {extra:?}",
        args[1],
        have.len()
    );
    assert!(unmapped.is_empty() && missing.is_empty() && extra.is_empty());
}
