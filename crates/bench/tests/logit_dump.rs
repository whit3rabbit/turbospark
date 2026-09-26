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
use turbospark_bench::real_model::open_model_runner_with_context;

/// Logits are dumped in the width the runner produced them in.
/// `LogitValue` is IEEE-754 binary16, so writing f16 bits is lossless AND
/// half the bytes of an f32 widening; numpy reads it as `float16` with no
/// conversion. Widening here would only invite the reader to believe there
/// is precision that was never there.
const DUMP_DTYPE: &str = "float16";

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

/// Which install the dump walks, plus the family-specific framing the walk
/// needs so its ids are the SAME sequence the family's quality gate scores.
///
/// The assistant prefix is bench crate Gotcha 13's: Harmony's assistant
/// slot is structured, so the reference answer sits behind
/// `<|channel|>final<|message|>` exactly as `reference_perplexity` places
/// it. It is empty for the other families, and that is not an oversight --
/// their templates open an assistant turn and then say words.
///
/// The date pin is Gotcha 14's: Harmony's template reads a clock, so an
/// unpinned dump would encode a different prompt every day and never
/// reproduce. Pinning to the gate's own date is also what lets the COLD
/// dump reproduce the gate's frozen perplexity.
struct DumpTarget {
    install: PathBuf,
    assistant_prefix: &'static str,
    pinned_chat_date: Option<&'static str>,
    max_context: u32,
}

fn resolve_target() -> Option<DumpTarget> {
    let plain = |key: &str| {
        env_dir(key).map(|install| DumpTarget {
            install,
            assistant_prefix: "",
            pinned_chat_date: None,
            max_context: PROTOCOL_MAX_CONTEXT,
        })
    };
    // Qwen4Exp's indexed-kernel budget is 2,048 tokens; keep the diagnostic
    // runner at that window instead of the shared 4,096-token default.
    env_dir("TURBOSPARK_QWEN4EXP_INSTALL_DIR")
        .map(|install| DumpTarget {
            install,
            assistant_prefix: "",
            pinned_chat_date: None,
            max_context: 2048,
        })
        .or_else(|| plain("TURBOSPARK_GEMMA4_INSTALL_DIR"))
        .or_else(|| plain("TURBOSPARK_QWEN2_DENSE_INSTALL_DIR"))
        .or_else(|| plain("TURBOSPARK_QWEN36_INSTALL_DIR"))
        .or_else(|| plain("TURBOSPARK_QWEN38_INSTALL_DIR"))
        .or_else(|| plain("TURBOSPARK_QWEN3MOE_INSTALL_DIR"))
        // ROADMAP's 1-bit entry, step 5. A plain arm like its three
        // neighbours: this family is ChatML, whose template opens an
        // assistant turn and then says words, so there is no structured
        // slot to prefix (Gotcha 13) and no clock in the template to pin
        // (Gotcha 14). It has NO quality gate row, so the perplexity this
        // target prints for it is a diagnostic and not a frozen number --
        // the KL is what it is here for, and that replays IDS rather than
        // prose, so it is valid whatever the framing is.
        .or_else(|| plain("TURBOSPARK_QWEN35_INSTALL_DIR"))
        // ROADMAP's ternary entry. The same family and the same plain arm:
        // one architecture at a third quantization, so nothing about the
        // FRAMING moves and only the install does.
        .or_else(|| plain("TURBOSPARK_TERNARY_INSTALL_DIR"))
        // Ornith-1.5-9B, and a plain arm for the same two reasons as the
        // three above it: ChatML opens an assistant turn and then says words
        // (so no structured slot to prefix, Gotcha 13) and its template reads
        // no clock (so nothing to pin, Gotcha 14).
        //
        // WHAT IT IS FOR IS THE ONE THING THIS FAMILY HAS NEVER HAD. Its four
        // frozen gate rows are all self-referential -- a perplexity and two
        // digests compared against this port's own past -- and
        // `ornith_tensor_probe.rs`, which correlates every installed tensor
        // against the published BF16 checkpoint at 0.998-1.000, is a STATIC
        // check that cannot see how the runtime USES those tensors. A KL
        // against llama.cpp on the identical Q8_0 bytes can, and it is the
        // only instrument that reaches the three things a fixture with
        // untrained weights records itself as blind to: the V-head
        // de-interleave on the DENSE half (`transcode.rs::v_head_axis`, and
        // this is the newest code in the family), the per-head q/k norms'
        // order relative to RoPE, and the RMS epsilon. Each perturbs every
        // layer systematically, which is what makes them visible here.
        .or_else(|| plain("TURBOSPARK_ORNITH9B_INSTALL_DIR"))
        // The same family's MoE half, and a plain arm for the same reasons.
        // Its reference is MLX rather than llama.cpp (`kld_mlx_affine.py`,
        // `ornith-35b-4bit`), because this install is streamed from an MLX
        // affine conversion where the 9B's is streamed from a GGUF -- so the
        // pair covers BOTH of this family's intake formats rather than
        // measuring one of them twice. Being MoE, both of its floors mean
        // something, where the dense 9B's shape floor collapses to nearly
        // nothing (`crates/bench/CLAUDE.md` Gotcha 8).
        .or_else(|| plain("TURBOSPARK_ORNITH35B_INSTALL_DIR"))
        // `deepseek2`, and a plain arm for the same two reasons as its
        // neighbours: the V2 template opens an assistant turn and says words
        // (no structured slot, Gotcha 13) and reads no clock (Gotcha 14).
        // Its KL against llama.cpp is the only instrument that reaches the
        // absorbed-MLA attention's numerics -- the compressed-latency cache
        // and the transposed q absorption have no self-referential gate that
        // can see them.
        .or_else(|| plain("TURBOSPARK_DSV2_INSTALL_DIR"))
        .or_else(|| {
            env_dir("TURBOSPARK_GPTOSS_INSTALL_DIR").map(|install| DumpTarget {
                install,
                assistant_prefix: quality_common::HARMONY_ASSISTANT_PREFIX,
                pinned_chat_date: Some(quality_common::GPTOSS_PINNED_CHAT_DATE),
                max_context: PROTOCOL_MAX_CONTEXT,
            })
        })
}

#[test]
#[ignore = "needs a real .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR or TURBOSPARK_QWEN4EXP_INSTALL_DIR) and an output dir (TURBOSPARK_LOGIT_DUMP_DIR)"]
fn dump_reference_logits() {
    let (Some(target), Some(out)) = (resolve_target(), env_dir("TURBOSPARK_LOGIT_DUMP_DIR")) else {
        eprintln!(
            "logit_dump: needs TURBOSPARK_GEMMA4_INSTALL_DIR (or \
             TURBOSPARK_QWEN2_DENSE_INSTALL_DIR, TURBOSPARK_QWEN36_INSTALL_DIR, \
             TURBOSPARK_QWEN38_INSTALL_DIR, TURBOSPARK_QWEN4EXP_INSTALL_DIR, \
             TURBOSPARK_QWEN3MOE_INSTALL_DIR, \
             TURBOSPARK_QWEN35_INSTALL_DIR, TURBOSPARK_TERNARY_INSTALL_DIR, \
             TURBOSPARK_ORNITH9B_INSTALL_DIR, TURBOSPARK_ORNITH35B_INSTALL_DIR, \
             or \
             TURBOSPARK_GPTOSS_INSTALL_DIR) and \
             TURBOSPARK_LOGIT_DUMP_DIR; skipping."
        );
        return;
    };
    // Before ANY render (one model per process, so this is the whole
    // process's clock).
    if let Some(date) = target.pinned_chat_date {
        std::env::set_var(tokenizer::CHAT_DATE_ENV, date);
        eprintln!("logit_dump: chat template date pinned to {date}");
    }
    dump(
        &target.install,
        &out,
        target.assistant_prefix,
        target.max_context,
    );
}

fn dump(install: &Path, out: &Path, assistant_prefix: &str, max_context: u32) {
    std::fs::create_dir_all(out).expect("create the dump directory");
    let (mut runner, tokenizer) =
        open_model_runner_with_context(install, PROTOCOL_EXPERT_CACHE_SLOTS, max_context)
            .expect("real install should open");

    // The prefix counts as PROMPT, exactly as `reference_perplexity` scores
    // it: `prompt_len` and `first_scored_position` move with it, so the
    // first answer token keeps its meaning in the sidecar.
    let mut prompt_ids = quality_common::user_turn_ids(&tokenizer);
    if !assistant_prefix.is_empty() {
        let prefix_ids = tokenizer.encode(assistant_prefix, false);
        assert!(
            !prefix_ids.is_empty(),
            "a non-empty assistant prefix must tokenize"
        );
        prompt_ids.extend(&prefix_ids);
    }
    let answer_ids = tokenizer.encode(quality_common::REFERENCE_ANSWER, false);
    assert!(!answer_ids.is_empty(), "the reference answer must tokenize");
    let mut ids = prompt_ids.to_vec();
    ids.extend(&answer_ids);
    assert!(
        ids.len() <= max_context as usize,
        "prompt plus reference answer is {} tokens, over the \
         {max_context}-token KV the runner was opened with",
        ids.len()
    );

    // Row i holds the next-token logits after consuming ids[i], so the last
    // id is fed to nobody and there is one row fewer than there are ids.
    let rows = ids.len() - 1;
    let vocab = runner.vocab_size();
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
        meta_json(install, &ids, prompt_ids.len(), vocab, cold, max_context),
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
fn meta_json(
    install: &Path,
    ids: &[i32],
    prompt_len: usize,
    vocab: usize,
    cold: bool,
    max_context: u32,
) -> String {
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
         \"max_context\": {max_context},\n  \
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
