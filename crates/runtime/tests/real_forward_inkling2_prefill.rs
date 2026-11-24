//! Gate for the Inkling 2 chunked prefill's bit-exact arm
//! (`TURBOSPARK_INKLING2_PREFILL_MMA=0`), on a real install.
//!
//! ```text
//! INKLING2_INSTALL=~/models/inkling2-mini-gturbo cargo test --release \
//!     -p turbospark-runtime --test real_forward_inkling2_prefill -- --ignored --nocapture
//! ```
//!
//! Reference: the prompt through `produce`, one token at a time, every
//! position's logits kept, then greedy decode past it. Candidate: a fresh
//! session that prefills the same prompt through `prefill_chunk` in chunks
//! whose sizes are not multiples of the 8-row micro-batch (so short tail
//! batches and chunk edges in the middle of the conv taps are on the path),
//! then decodes the same tokens. The last prompt row and every decode row
//! must equal the reference BIT FOR BIT. The prompt crosses the window ring's
//! wrap, so KV ring addressing is exercised; a wrong KV row, conv tail or
//! block state shows up in the decode rows after the prefill.

use std::path::PathBuf;

use half::f16;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

fn argmax(row: &[f16]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .unwrap()
        .0
}

fn same_bits(a: &[f16], b: &[f16]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
#[ignore = "needs INKLING2_INSTALL (a real Inkling 2 install, resident experts)"]
fn chunked_prefill_matches_sequential_prefill_bit_for_bit() {
    // The bit-exact arm: the default prefill takes the matrix-unit GEMM,
    // which is KL-bounded rather than bit-exact
    // (`real_forward_inkling2_prefill_mma.rs` gates it). This file holds the
    // `TURBOSPARK_INKLING2_PREFILL_MMA=0` path to the bit, so it turns the
    // switch off before anything opens (one test per binary: the switch is
    // read once per process). On an install with no BF16 dense weights
    // (mlx4) the switch changes nothing.
    std::env::set_var("TURBOSPARK_INKLING2_PREFILL_MMA", "0");
    let install = PathBuf::from(std::env::var("INKLING2_INSTALL").expect("INKLING2_INSTALL"));
    let arch = model_io::inkling2_mini();
    let vocab = arch.vocab_size as usize;
    let tok = tokenizer::MfTokenizer::load_from_dir(&install).expect("tokenizer");
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/INKLING2_PHASE0.md"),
    )
    .expect("prose");
    let mut ids = tok.encode(&text, false);
    // Past the window ring's wrap (window + the 128-row panel's slack + the
    // tiled attention's 32 = 1,183 rows), so a panel crosses it.
    let prompt_len = 1400usize;
    assert!(ids.len() >= prompt_len, "fixture prose too short");
    ids.truncate(prompt_len);
    let decode = 32usize;
    let mut runner = RealForwardRunner::open_with_residency(
        &install,
        arch.clone(),
        2048,
        turbospark_runtime::ExpertCacheSlots::Auto,
        turbospark_runtime::DraftPolicies::off(),
        turbospark_runtime::SteeringPolicy::off(),
        1,
        turbospark_runtime::KvQuant::Off,
        turbospark_runtime::ExpertResidency::Resident,
    )
    .expect("open resident");
    assert!(runner.supports_chunked_prefill());

    // Reference: sequential.
    let mut row = vec![f16::ZERO; vocab];
    let t = std::time::Instant::now();
    for (pos, &id) in ids.iter().enumerate() {
        runner.produce(id, pos, &mut row).expect("produce");
    }
    let seq_prefill_s = t.elapsed().as_secs_f64();
    let want_last = row.clone();
    let mut want = Vec::with_capacity(decode);
    let mut generated = Vec::with_capacity(decode);
    for i in 0..decode {
        let next = argmax(&row) as i32;
        generated.push(next);
        runner
            .produce(next, prompt_len + i, &mut row)
            .expect("decode");
        want.push(row.clone());
    }

    // Candidate: chunked, with uneven chunk sizes.
    runner.reset();
    let chunks = [13usize, 512, 300, 575];
    assert_eq!(chunks.iter().sum::<usize>(), prompt_len);
    let t = std::time::Instant::now();
    let mut at = 0usize;
    for &n in &chunks {
        runner
            .prefill_chunk(&ids[at..at + n], at, &mut row)
            .expect("prefill_chunk");
        at += n;
    }
    let chunk_prefill_s = t.elapsed().as_secs_f64();
    assert!(
        same_bits(&row, &want_last),
        "last prompt row differs (top-1 {} vs {})",
        argmax(&row),
        argmax(&want_last)
    );
    let mut exact = 0usize;
    for (i, &next) in generated.iter().enumerate() {
        runner
            .produce(next, prompt_len + i, &mut row)
            .expect("decode");
        if same_bits(&row, &want[i]) {
            exact += 1;
        }
    }
    eprintln!(
        "prefill {prompt_len} tokens: sequential {:.2} s ({:.1} tok/s), chunked {:.2} s \
         ({:.1} tok/s); decode rows after it {exact} of {decode} bit-identical",
        seq_prefill_s,
        prompt_len as f64 / seq_prefill_s,
        chunk_prefill_s,
        prompt_len as f64 / chunk_prefill_s,
    );
    assert_eq!(exact, decode, "decode after chunked prefill drifted");
}
