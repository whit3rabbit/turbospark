#![cfg(target_os = "macos")]
//! What is the MTP head actually predicting? A per-position dump.
//!
//! `mtp_accept_length_probe.rs` read **zero** accepted proposals across 7,168
//! offers at four block sizes. A weak drafter still lands common tokens, so
//! exactly zero is structural rather than a verdict about MTP, and the accept
//! length cannot be reported until it is explained.
//!
//! This is the instrument for that. It prints, per decode position, what the
//! trunk went on to emit beside what the head proposed, and tests the head's
//! top-1 against every hypothesis that costs nothing to check:
//!
//! - `t[i+2]`, which is what a depth-1 MTP module is for;
//! - `t[i+1]`, which would mean the position/token pairing is off by one and
//!   the head is duplicating the trunk rather than running ahead of it;
//! - the head's own INPUT token, which would mean it is echoing;
//! - a constant, which would mean the block is not reading its inputs at all.
//!
//! The dump is the deliverable and the classification is a convenience: a
//! degenerate head (one token forever, or a wall of equal logits) is visible
//! in the printout in a way no single ratio would be.
//!
//! ```sh
//! TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
//!   cargo test -p turbospark-bench --test mtp_head_probe --release -- --ignored --nocapture
//! ```

use foundation::LogitValue;
use runtime::LogitProducer;
use tokenizer::{Message, Role};
use turbospark_bench::real_model::open_model_runner_speculative;

/// Draft depth this file asks for. Named here rather than set through
/// `TURBOSPARK_MTP_DRAFT` because the policy is now a PARAMETER: an unset
/// env var means `Auto`, which resolves to a depth too small for the
/// blocks below and would fail deep in the verify rather than at open.
const MTP_DEPTH: usize = 4;

const POSITIONS: usize = 24;

/// Pearson over two logit vectors. `turbospark_compute::pearson` returns 0.0
/// on a constant input by contract (AGENTS.md Gotcha 30); nothing here is
/// constant, but the convention is worth matching.
fn pearson(a: &[LogitValue], b: &[LogitValue]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|v| v.to_f32() as f64).sum::<f64>() / n,
        b.iter().map(|v| v.to_f32() as f64).sum::<f64>() / n,
    );
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (x.to_f32() as f64 - ma, y.to_f32() as f64 - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da == 0.0 || db == 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

fn top_k(logits: &[LogitValue], k: usize) -> Vec<(i32, f32)> {
    let mut v: Vec<(i32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, x)| (i as i32, x.to_f32()))
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(k);
    v
}

#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR"]
fn what_the_mtp_head_predicts() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_MTP_INSTALL_DIR").expect("TURBOSPARK_MTP_INSTALL_DIR"),
    );
    let (mut runner, tokenizer) = open_model_runner_speculative(
        &dir,
        16,
        runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Fixed(MTP_DEPTH)),
    )
    .expect("install opens");
    let vocab = runner.vocab_size();
    assert!(runner.mtp_draft_depth() > 0, "install carries no head");

    let prompt = {
        let rendered = tokenizer
            .apply_chat_template(&[Message::new(
                Role::User,
                "Write a Python function that merges two sorted lists, then explain \
                 its time and space complexity in detail.",
            )])
            .expect("chat template renders");
        tokenizer.encode(&rendered, false)
    };

    // Pass 1: the trunk alone, greedy, recording the true continuation.
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    runner.reset();
    for (i, &t) in prompt.iter().enumerate() {
        runner.produce(t, i, &mut logits).expect("produce");
    }
    let mut truth: Vec<i32> = Vec::new();
    let mut next = top_k(&logits, 1)[0].0;
    for step in 0..POSITIONS + 4 {
        truth.push(next);
        runner
            .produce(next, prompt.len() + step, &mut logits)
            .expect("produce");
        next = top_k(&logits, 1)[0].0;
    }
    // `truth[j]` is the token at absolute position `prompt.len() + j`.

    // Pass 2: replay the same stream, taking ONE draft off each position.
    let mut draft = vec![LogitValue::from_f32(0.0); vocab];
    runner.reset();
    for (i, &t) in prompt.iter().enumerate() {
        runner.produce(t, i, &mut logits).expect("produce");
        if i + 1 < prompt.len() {
            runner.mtp_prime_step(prompt[i + 1], i).expect("prime");
        }
    }

    let (mut hits_next2, mut hits_next1, mut hits_echo) = (0usize, 0usize, 0usize);
    let mut ranks: Vec<usize> = Vec::new();
    let mut corrs: Vec<f64> = Vec::new();
    let mut drafts_kept: Vec<Vec<LogitValue>> = Vec::new();
    let mut proposals: Vec<i32> = Vec::new();
    println!("\n  pos  fed(t[i+1])  head_top1   t[i+1]   t[i+2]   verdict   head_top3 (id:logit)");
    for j in 0..POSITIONS {
        // The trunk has just produced at absolute position `p`, so `scratch.x`
        // holds h(p). The head's pair at `p` is (h(p), t[p+1]).
        let p = prompt.len() - 1 + j;
        let fed = truth[j]; // the token occupying p + 1
        runner
            .mtp_draft_step(fed, p, &mut draft)
            .expect("draft step");
        let top = top_k(&draft, 3);
        let head1 = top[0].0;
        let want_next2 = truth[j + 1]; // the token at p + 2
        let want_next1 = fed;

        if head1 == want_next2 {
            hits_next2 += 1;
        }
        if head1 == want_next1 {
            hits_next1 += 1;
            hits_echo += 1;
        }
        proposals.push(head1);

        // Where does the TRUE next-next token rank in the head's own
        // distribution? This is what tells a BROKEN head from a merely
        // misaligned one: a random direction puts the truth at a uniformly
        // random rank (~vocab/2), while a working head that is merely paired
        // wrong still ranks it near the top.
        let mut better = 0usize;
        let target_logit = draft[want_next2 as usize].to_f32();
        for v in draft.iter() {
            if v.to_f32() > target_logit {
                better += 1;
            }
        }
        ranks.push(better);

        let verdict = if head1 == want_next2 {
            "t[i+2] OK"
        } else if head1 == want_next1 {
            "ECHO    "
        } else {
            "neither "
        };
        println!(
            "{p:>5}  {fed:>11}  {head1:>9}  {want_next1:>7}  {want_next2:>7}   {verdict}  {}",
            top.iter()
                .map(|(i, v)| format!("{i}:{v:.2}"))
                .collect::<Vec<_>>()
                .join(" ")
        );

        // Advance the trunk one position so the next iteration's `scratch.x`
        // is the right hidden state. The head's row for `p` is already
        // written, so its cursor is where the next step needs it.
        runner
            .produce(fed, p + 1, &mut logits)
            .expect("advance the trunk");
        // `logits` now predicts p + 2 -- the SAME position the draft just
        // predicted. Correlating the two localizes the fault: ~+1 means the
        // head agrees with the trunk and the pairing is what is wrong, ~0
        // means its hidden state is unrelated, and ~-1 means something in the
        // head path is sign-inverted.
        corrs.push(pearson(&draft, &logits));
        drafts_kept.push(draft.clone());
    }

    // Which position, if any, IS the head aligned with? Replay the trunk and
    // correlate each kept draft against the trunk's distribution at several
    // offsets. A head that is merely paired wrong shows a clear positive peak
    // at some offset; one that peaks nowhere is not predicting tokens at all.
    {
        let mut trunk_at: Vec<Vec<LogitValue>> = Vec::new();
        runner.reset();
        let mut l = vec![LogitValue::from_f32(0.0); vocab];
        for (i, &t) in prompt.iter().enumerate() {
            runner.produce(t, i, &mut l).expect("produce");
        }
        for (step, &t) in truth.iter().enumerate() {
            runner
                .produce(t, prompt.len() + step, &mut l)
                .expect("produce");
            trunk_at.push(l.clone());
        }
        // `trunk_at[k]` predicts absolute position prompt.len() + k + 1.
        println!("\n  head draft at p vs the trunk's distribution for p+delta:");
        for delta in 1..=4usize {
            let mut acc = Vec::new();
            for (j, d) in drafts_kept.iter().enumerate() {
                // draft j was taken at p = prompt.len() - 1 + j and aims at
                // p + 2; index trunk_at so it predicts p + delta.
                let k = j + delta - 2;
                if delta >= 2 && k < trunk_at.len() {
                    acc.push(pearson(d, &trunk_at[k]));
                }
            }
            if !acc.is_empty() {
                println!(
                    "    delta {delta}: mean {:+.4} over {} pairs",
                    acc.iter().sum::<f64>() / acc.len() as f64,
                    acc.len()
                );
            }
        }
    }

    let mean_corr = corrs.iter().sum::<f64>() / corrs.len() as f64;
    println!(
        "\n  pearson(head draft, trunk logits) over the SAME predicted position:\n    \
           mean {mean_corr:+.4}   min {:+.4}   max {:+.4}",
        corrs.iter().cloned().fold(f64::INFINITY, f64::min),
        corrs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    ranks.sort_unstable();
    let median = ranks[ranks.len() / 2];
    let top100 = ranks.iter().filter(|r| **r < 100).count();
    println!(
        "\n  rank of the true t[i+2] in the head's distribution:\n    \
           median {median} of {vocab}   best {}   worst {}   within top-100: {top100}/{POSITIONS}",
        ranks[0],
        ranks[ranks.len() - 1]
    );
    println!(
        "    a uniformly random direction would put the median near {}",
        vocab / 2
    );

    let distinct: std::collections::BTreeSet<i32> = proposals.iter().copied().collect();
    println!(
        "\n  matches t[i+2] (the MTP target): {hits_next2}/{POSITIONS}\n  \
           matches t[i+1] (off by one / echo): {hits_next1}/{POSITIONS}\n  \
           echoes its own input token:        {hits_echo}/{POSITIONS}\n  \
           distinct proposals:                {}/{POSITIONS}",
        distinct.len()
    );
    if distinct.len() <= 2 {
        println!(
            "  -> DEGENERATE: the head is emitting the same token regardless of input,\n     \
                which points at its weights or its inputs rather than at the pairing."
        );
    }
    println!();
}

/// The head's tensors against a TRUNK full-attention layer's, size for size.
///
/// `docs/MTP_SPECULATIVE.md` claims the head's block is "identical to a trunk
/// full-attention layer, field for field", which is what licenses running it
/// on `attn.rs`'s encoders. That claim was read off the published safetensors
/// header; this checks it against the INSTALLED bytes, which is a different
/// statement and the one the dispatches actually depend on.
#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR"]
fn the_heads_tensors_match_a_trunk_full_attention_layers() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_MTP_INSTALL_DIR").expect("TURBOSPARK_MTP_INSTALL_DIR"),
    );
    let arch = repack::peek_manifest_arch(&dir).expect("manifest peeks");
    let index =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    let full = arch
        .full_attention_layer_mask
        .iter()
        .position(|&m| m == 1)
        .expect("a full-attention layer exists");
    println!(
        "\nhidden={} heads={} kv={} head_dim={} inter={} full layer={full}",
        arch.hidden_size,
        arch.num_heads,
        arch.num_full_kv_heads,
        arch.full_head_dim,
        arch.intermediate_size
    );

    let suffixes = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ];
    println!("\n  tensor                            head(bytes,dtype)   trunk(bytes,dtype)  same");
    let mut mismatches = Vec::new();
    for s in suffixes {
        let h = index.entries.get(&format!("mtp.layers.0.{s}"));
        let t = index
            .entries
            .get(&format!("language_model.model.layers.{full}.{s}"));
        let (hd, td) = (
            h.map(|e| (e.size_bytes, e.dtype)),
            t.map(|e| (e.size_bytes, e.dtype)),
        );
        let same = hd == td;
        if !same {
            mismatches.push(s);
        }
        println!(
            "  {s:<32}  {:>18}  {:>18}  {}",
            hd.map_or("MISSING".to_string(), |(b, d)| format!("{b},{d}")),
            td.map_or("MISSING".to_string(), |(b, d)| format!("{b},{d}")),
            if same { "yes" } else { "NO" }
        );
    }
    for extra in [
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
    ] {
        let e = index.entries.get(extra);
        println!(
            "  {extra:<32}  {:>18}",
            e.map_or("MISSING".to_string(), |e| format!(
                "{},{}",
                e.size_bytes, e.dtype
            ))
        );
    }
    assert!(
        mismatches.is_empty(),
        "the head's block is NOT shaped like a trunk full-attention layer: {mismatches:?}"
    );
}

/// The installed head's BYTES are its own: not zero, not duplicated from a
/// trunk layer, and not all the same tensor repeated.
///
/// Offline and milliseconds. It exists because this head's ingest has already
/// failed once in exactly the way a shape check cannot see: the streamed
/// writer classified `mtp.*` correctly into `plan.mtp_bases` and then never
/// read it, producing a byte-identical HEADLESS install with no error
/// (`docs/MTP_SPECULATIVE.md` step 1). "The tensor is present and the right
/// size" is the assertion that bug survived, so presence is not the question
/// here -- provenance is.
#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR"]
fn the_installed_heads_bytes_are_its_own() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_MTP_INSTALL_DIR").expect("TURBOSPARK_MTP_INSTALL_DIR"),
    );
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("resident index");
    let bytes = std::fs::read(&path).expect("read weights");

    let digest = |name: &str| -> Option<(String, u64)> {
        let e = index.entries.get(name)?;
        let start = e.file_offset as usize;
        let end = start + e.size_bytes as usize;
        Some((model_io::hash_data(&bytes[start..end]), e.size_bytes))
    };

    let mtp: Vec<&String> = index
        .entries
        .keys()
        .filter(|k| k.starts_with("mtp."))
        .collect();
    assert_eq!(mtp.len(), 15, "expected 15 head tensors, got {}", mtp.len());

    // 1. Distinct from each other.
    let mut seen: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for name in &mtp {
        let (d, size) = digest(name).expect("entry");
        println!("  {name:<48} {size:>10}  {}", &d[..16]);
        if let Some(prev) = seen.insert(d.clone(), (*name).clone()) {
            panic!("head tensors {prev} and {name} have IDENTICAL bytes");
        }
    }

    // 2. Not a copy of the trunk layer they are shaped like.
    let arch = repack::peek_manifest_arch(&dir).expect("manifest peeks");
    let full = arch
        .full_attention_layer_mask
        .iter()
        .position(|&m| m == 1)
        .expect("a full-attention layer exists");
    for suffix in [
        "self_attn.q_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "input_layernorm.weight",
    ] {
        let h = digest(&format!("mtp.layers.0.{suffix}")).expect("head tensor");
        let t = digest(&format!("language_model.model.layers.{full}.{suffix}")).expect("trunk");
        assert_ne!(
            h.0, t.0,
            "mtp.layers.0.{suffix} is a byte copy of trunk layer {full}'s"
        );
    }

    // 3. Not zero-filled. A zeroed head is the failure a presence check and a
    //    distinctness check both pass.
    for name in &mtp {
        let e = &index.entries[*name];
        let start = e.file_offset as usize;
        let slice = &bytes[start..start + e.size_bytes as usize];
        assert!(
            slice.iter().any(|&b| b != 0),
            "{name} is entirely zero bytes"
        );
    }
    println!("\n  15 head tensors, all distinct, none zero, none copied from trunk layer {full}");
}

/// Takes ONE draft step at position 0 and lets `TURBOSPARK_MTP_DUMP` capture it.
///
/// Position 0 against an empty head cache is deliberate and is what makes the
/// dump comparable offline: attention over a single key is a softmax over one
/// logit, which is exactly 1.0, so the block's output does not depend on RoPE,
/// on the q/k norms or on any history a script would have to replay. The whole
/// head becomes a function of `concat` alone, which is the form
/// `scripts/mtp_bisect.py` recomputes.
#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR"]
fn dumps_one_draft_step_for_the_bisect() {
    let Some(dir) = std::env::var_os("TURBOSPARK_MTP_DUMP") else {
        println!("\nTURBOSPARK_MTP_DUMP unset; nothing to capture. See scripts/mtp_bisect.py\n");
        return;
    };
    let install = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_MTP_INSTALL_DIR").expect("TURBOSPARK_MTP_INSTALL_DIR"),
    );
    let (mut runner, _) = open_model_runner_speculative(
        &install,
        16,
        runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Fixed(MTP_DEPTH)),
    )
    .expect("install opens");
    let vocab = runner.vocab_size();

    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    let mut draft = vec![LogitValue::from_f32(0.0); vocab];
    runner.reset();
    // One trunk token, so `scratch.x` holds h(0) and the head's cursor is 0.
    runner.produce(9707, 0, &mut logits).expect("produce");
    let next = top_k(&logits, 1)[0].0;
    assert_eq!(runner.mtp_kv_position(), 0, "the head must start empty");
    runner
        .mtp_draft_step(next, 0, &mut draft)
        .expect("draft step");
    println!(
        "\nwrote the step to {}\n  fed token {next} at position 0, head top-1 {}\n",
        std::path::PathBuf::from(&dir).display(),
        top_k(&draft, 1)[0].0
    );
}
