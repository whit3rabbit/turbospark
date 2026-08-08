#![cfg(target_os = "macos")]
//! Dump this port's full-vocabulary logits for the quality corpus, so a
//! second engine can be handed the SAME TOKEN IDS and the two
//! distributions compared position by position (ROADMAP Phase Q's last
//! deliverable, the token-level KLD against mlx-lm).
//!
//! WHY THIS IS A DIFFERENT QUESTION FROM THE QUALITY GATE. `quality_gate`
//! and `quality_sensitivity` measure this port against ITS OWN past: a
//! perplexity that moves, or a digest that changes, says something here
//! shifted. Neither can say whether the shared starting point was right.
//! A KLD against a second engine reading the SAME quantized bytes
//! separates a kernel bug from the quantization, which is what makes a
//! Phase S (sub-4-bit experts) quality delta attributable.
//!
//! FEED BOTH ENGINES IDS, NEVER A STRING. A tokenizer or chat-template
//! difference would surface as a divergence and be misread as a numerics
//! gap. `meta.json` therefore carries the exact id sequence this port
//! walked, and the other engine is expected to consume that list rather
//! than re-encode the prose.
//!
//! EVERY POSITION IS DUMPED, prompt included. The assistant-slot-only rule
//! that `quality_common` documents at length is a PERPLEXITY constraint:
//! an instruction-tuned checkpoint was never trained to predict prompt
//! tokens, so scoring them measures nothing. A distribution comparison
//! between two engines carries no such requirement -- it is valid wherever
//! both engines ran -- and prompt positions are the cheaper tokens, since
//! they come out of the same single pass. Do not carry the constraint over
//! by reflex; slice with `first_scored_position` if a run ever wants to.
//!
//! CACHE STATE MOVES THE LOW BITS, so this walks the corpus twice and
//! dumps the second walk. Gemma's routed slots are ordered misses-first,
//! which permutes phase 2's reduce, and FP addition is not associative
//! (AGENTS.md Gotcha 27): a cold-cache logit differs from a warm-cache one
//! at a magnitude that is small but is not zero, and a KLD floor built on
//! a cold pass would not reproduce. `reset()` rewinds the KV and the GDN
//! state and deliberately NOT the expert cache, so the second walk is warm.
//! Slots are pinned to `PROTOCOL_EXPERT_CACHE_SLOTS` for the same reason
//! every other frozen number here is.
//!
//! `produce_prefill` must NOT appear in this file. It is allowed to skip
//! the output head, which is the only thing being dumped.
//!
//! Not run by default (needs a real install, and writes a few hundred MB):
//!
//!   TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!   TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/turbospark \
//!     cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture

mod quality_common;

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use foundation::LogitValue;
use runtime::LogitProducer;
use turbospark_bench::protocol::{PROTOCOL_EXPERT_CACHE_SLOTS, PROTOCOL_MAX_CONTEXT};
use turbospark_bench::real_model::open_model_runner;

/// Logits are dumped in the width the runner produced them in.
/// `LogitValue` is IEEE-754 binary16, so writing f16 bits is lossless AND
/// half the bytes of an f32 widening; numpy reads it as `float16` with no
/// conversion. Widening here would only invite the reader to believe there
/// is precision that was never there.
const DUMP_DTYPE: &str = "float16";

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

#[test]
#[ignore = "needs a real .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR) and an output dir (TURBOSPARK_LOGIT_DUMP_DIR)"]
fn dump_reference_logits() {
    let (Some(install), Some(out)) = (
        env_dir("TURBOSPARK_GEMMA4_INSTALL_DIR")
            .or_else(|| env_dir("TURBOSPARK_QWEN36_INSTALL_DIR")),
        env_dir("TURBOSPARK_LOGIT_DUMP_DIR"),
    ) else {
        eprintln!(
            "logit_dump: needs TURBOSPARK_GEMMA4_INSTALL_DIR (or \
             TURBOSPARK_QWEN36_INSTALL_DIR) and TURBOSPARK_LOGIT_DUMP_DIR; skipping."
        );
        return;
    };
    dump(&install, &out);
}

fn dump(install: &Path, out: &Path) {
    std::fs::create_dir_all(out).expect("create the dump directory");
    let (mut runner, tokenizer) =
        open_model_runner(install, PROTOCOL_EXPERT_CACHE_SLOTS).expect("real install should open");

    let prompt_ids = quality_common::user_turn_ids(&tokenizer);
    let answer_ids = tokenizer.encode(quality_common::REFERENCE_ANSWER, false);
    assert!(!answer_ids.is_empty(), "the reference answer must tokenize");
    let mut ids = prompt_ids.to_vec();
    ids.extend(&answer_ids);
    assert!(
        ids.len() <= PROTOCOL_MAX_CONTEXT as usize,
        "prompt plus reference answer is {} tokens, over the \
         {PROTOCOL_MAX_CONTEXT}-token KV the runner was opened with",
        ids.len()
    );

    // Row i holds the next-token logits after consuming ids[i], so the last
    // id is fed to nobody and there is one row fewer than there are ids.
    let rows = ids.len() - 1;
    let vocab = tokenizer.vocab_size;
    eprintln!(
        "logit_dump: {} prompt + {} answer = {} ids -> {rows} rows x {vocab} \
         {DUMP_DTYPE} = {:.1} MiB",
        prompt_ids.len(),
        answer_ids.len(),
        ids.len(),
        (rows * vocab * 2) as f64 / (1024.0 * 1024.0),
    );

    // Walk once and throw it away: this warms the expert cache, which moves
    // the low bits of every logit (see the module doc).
    //
    // `TURBOSPARK_LOGIT_DUMP_COLD=1` skips it, which is how the warm/cold
    // difference gets MEASURED rather than assumed. `quality_gate` takes
    // its perplexity first thing in the process, so its frozen row is a
    // COLD number and a warm dump will not reproduce it; running this
    // target both ways is what tells you the gap is cache state and not a
    // bug in one of them.
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    let cold = std::env::var_os("TURBOSPARK_LOGIT_DUMP_COLD").is_some();
    if !cold {
        walk(&mut runner, &ids, &mut logits, |_, _| {});
    }

    let path = out.join("logits.f16");
    let mut file = BufWriter::new(std::fs::File::create(&path).expect("create the logits file"));
    let mut written = 0usize;
    walk(&mut runner, &ids, &mut logits, |_, logits| {
        let mut bytes = Vec::with_capacity(logits.len() * 2);
        for value in logits {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        file.write_all(&bytes).expect("write a logit row");
        written += 1;
    });
    file.flush().expect("flush the logits file");
    assert_eq!(written, rows, "every position must produce exactly one row");

    let meta = out.join("meta.json");
    std::fs::write(
        &meta,
        meta_json(install, &ids, prompt_ids.len(), vocab, cold),
    )
    .expect("write the sidecar");
    eprintln!(
        "logit_dump: wrote {} and {}",
        path.display(),
        meta.display()
    );
}

/// Teacher-force `ids` through `runner`, handing every position's
/// full-vocabulary logits to `sink`.
///
/// `produce`, never `produce_prefill`: the latter may skip the output head
/// (AGENTS.md Gotcha 2 in `crates/runtime`), which is the only thing this
/// file exists to read.
fn walk(
    runner: &mut runtime::RealForwardRunner,
    ids: &[i32],
    logits: &mut [LogitValue],
    mut sink: impl FnMut(usize, &[LogitValue]),
) {
    runner.reset();
    for (position, &token) in ids.iter().take(ids.len() - 1).enumerate() {
        runner
            .produce(token, position, logits)
            .expect("forward pass over the corpus");
        sink(position, logits);
    }
}

/// The sidecar the second engine reads. Hand-rolled rather than pulling in
/// a JSON serializer: it is one flat object of numbers and one array of
/// integers, and this crate has no serde dependency today.
fn meta_json(install: &Path, ids: &[i32], prompt_len: usize, vocab: usize, cold: bool) -> String {
    let id_list = ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let cache_state = if cold {
        "cold (no warmup walk; matches how quality_gate takes its perplexity)"
    } else {
        "warm (one discarded walk of the same sequence)"
    };
    format!(
        "{{\n  \
         \"engine\": \"turbospark\",\n  \
         \"install\": {:?},\n  \
         \"expert_cache_slots\": {PROTOCOL_EXPERT_CACHE_SLOTS},\n  \
         \"cache_state\": \"{cache_state}\",\n  \
         \"dtype\": \"{DUMP_DTYPE}\",\n  \
         \"layout\": \"row-major [rows][vocab_size], row i = next-token logits after token_ids[i]\",\n  \
         \"normalized\": false,\n  \
         \"head\": \"softcap * tanh(z / softcap) where the family softcaps, otherwise raw; never softmaxed\",\n  \
         \"rows\": {},\n  \
         \"vocab_size\": {vocab},\n  \
         \"prompt_len\": {prompt_len},\n  \
         \"first_scored_position\": {},\n  \
         \"token_ids\": [{id_list}]\n\
         }}\n",
        install.display().to_string(),
        ids.len() - 1,
        prompt_len - 1,
    )
}
