//! The DeepSeek-VL2 trunk on a REAL install: walk a given id sequence token
//! by token and dump what a second implementation can be held to
//! (`docs/DEEPSEEK_VL2_VISION_PHASE0.md`, "The trunk gate").
//!
//! Two files are written next to `TURBOSPARK_DSVL2_OUT`:
//!
//! - `<out>.argmax.txt`: the greedy next-token id at EVERY position, one per
//!   line. Teacher-forced, so one wrong layer shows as the first disagreeing
//!   line rather than as drift.
//! - `<out>.logits.bin`: the LAST position's full-vocabulary logits, little
//!   endian f32.
//!
//! `scripts/deepseek_vl2_trunk_reference.py` computes the same two from the
//! checkpoint's safetensors with numpy alone and compares.
//!
//! ```sh
//! TURBOSPARK_DSVL2_INSTALL_DIR=/Volumes/T7/tshome-dsvl2/models/text/dsvl2-tiny.gturbo \
//! TURBOSPARK_DSVL2_IDS=/Volumes/T7/vision-dumps/dsvl2-ids.txt \
//! TURBOSPARK_DSVL2_OUT=/Volumes/T7/vision-dumps/dsvl2-engine \
//!   cargo test -p turbospark-runtime --test deepseek_vl2_trunk_real --release -- \
//!   --ignored --nocapture
//! ```
//!
//! `TURBOSPARK_DSVL2_IDS` is one id per line.

use half::f16;
use std::path::PathBuf;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key} (see this file's doc)")))
}

#[test]
#[ignore = "needs the real dsvl2-tiny install and an id file; see the module doc"]
fn the_real_trunk_dumps_teacher_forced_argmax_and_last_logits() {
    let dir = env_path("TURBOSPARK_DSVL2_INSTALL_DIR");
    let out = env_path("TURBOSPARK_DSVL2_OUT");
    let ids: Vec<i32> = std::fs::read_to_string(env_path("TURBOSPARK_DSVL2_IDS"))
        .expect("ids file")
        .split_whitespace()
        .map(|t| t.parse().expect("an integer id"))
        .collect();
    assert!(!ids.is_empty(), "no ids");

    let arch = turbospark_repack::peek_manifest_arch(&dir).expect("install manifest");
    assert!(
        !arch.mla.is_active(),
        "this test is for the plain-attention shape"
    );
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open install");
    let vocab = runner.vocab_size();
    let mut head = vec![f16::from_f32(0.0); vocab];
    let mut argmax = String::new();
    for (position, &token) in ids.iter().enumerate() {
        runner
            .produce(token, position, &mut head)
            .expect("produces");
        let (best, _) =
            head.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                    let v = v.to_f32();
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                });
        argmax.push_str(&format!("{best}\n"));
    }
    let last: Vec<u8> = head.iter().flat_map(|v| v.to_f32().to_le_bytes()).collect();
    std::fs::write(out.with_extension("argmax.txt"), argmax).expect("write argmax");
    std::fs::write(out.with_extension("logits.bin"), last).expect("write logits");
    println!("walked {} ids, vocab {vocab}", ids.len());
}
