#![cfg(target_os = "macos")]
//! Capture TurboSpark's greedy token path for the frozen Swift prompt used in
//! the CPU llama.cpp source comparison. Each row is the current next-token
//! ranking before the selected token is fed back into the model.
//!
//! ```sh
//! TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR=/tmp/turbospark-qwen4exp-swift-iq2-xs.gturbo \
//!   cargo test -p turbospark-bench --test qwen4exp_swift_first_token_probe \
//!   --release -- --ignored --nocapture
//! ```

use foundation::LogitValue;
use runtime::{ChunkedPrefillRunner, LogitProducer};
use tokenizer::{Message, Role};
use turbospark_bench::{
    protocol::PROTOCOL_CASES,
    real_model::{open_model_runner_with_context, protocol_parameters},
};

fn top_k(logits: &[LogitValue], k: usize) -> Vec<(i32, f32)> {
    let mut ranked: Vec<_> = logits
        .iter()
        .enumerate()
        .map(|(id, value)| (id as i32, value.to_f32()))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(k);
    ranked
}

#[test]
#[ignore = "needs the pinned Swift IQ2_XS install"]
fn reports_swift_iq2_xs_first_token_logits() {
    let install = std::env::var_os("TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR")
        .expect("set TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR");
    let params = protocol_parameters(model_io::ModelFamily::Qwen4Exp);
    let (mut runner, tokenizer) =
        open_model_runner_with_context(std::path::Path::new(&install), 16, params.max_context)
            .expect("Swift IQ2_XS install opens");
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, PROTOCOL_CASES[0].content)])
        .expect("frozen prompt renders");
    let prompt = tokenizer.encode(&rendered, false);
    assert_eq!(
        prompt.len(),
        62,
        "the frozen source-comparison prompt changed"
    );
    let mut logits = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    runner.reset();
    for (position, &token) in prompt.iter().enumerate() {
        runner
            .produce(token, position, &mut logits)
            .expect("prompt forward pass succeeds");
    }

    eprintln!(
        "prompt_tokens={} prompt_ids={:?} end_of_turn_id={} stop_token_ids={:?}",
        prompt.len(),
        prompt,
        tokenizer.end_of_turn_id,
        tokenizer.stop_token_ids
    );
    for generated in 0..12 {
        assert!(
            logits.iter().all(|value| value.to_f32().is_finite()),
            "the next-token row after {generated} generated tokens contains a non-finite logit"
        );
        let ranked = top_k(&logits, 12);
        assert!(
            ranked[0].1 > ranked[1].1,
            "the next-token row after {generated} generated tokens has no unique top candidate"
        );
        for (rank, (id, logit)) in ranked.iter().copied().enumerate() {
            eprintln!(
                "generated={} rank={} id={} logit={} token={:?}",
                generated,
                rank + 1,
                id,
                logit,
                tokenizer.decode(&[id], false)
            );
        }

        let (token, logit) = ranked[0];
        eprintln!(
            "selected generated={} id={} logit={} token={:?} visible={:?}",
            generated,
            token,
            logit,
            tokenizer.decode(&[token], false),
            tokenizer.decode(&[token], true)
        );
        if tokenizer.stop_token_ids.contains(&token) || generated == 11 {
            break;
        }

        runner
            .produce(token, prompt.len() + generated, &mut logits)
            .expect("generated-token forward pass succeeds");
    }
}

#[test]
#[ignore = "needs the pinned Swift IQ2_XS install"]
fn compares_swift_iq2_xs_chunked_and_sequential_prefill_logits() {
    let install = std::env::var_os("TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR")
        .expect("set TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR");
    let params = protocol_parameters(model_io::ModelFamily::Qwen4Exp);
    let (mut runner, tokenizer) =
        open_model_runner_with_context(std::path::Path::new(&install), 16, params.max_context)
            .expect("Swift IQ2_XS install opens");
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, PROTOCOL_CASES[0].content)])
        .expect("frozen prompt renders");
    let prompt = tokenizer.encode(&rendered, false);
    assert_eq!(
        prompt.len(),
        62,
        "the frozen source-comparison prompt changed"
    );

    let mut sequential = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    runner.reset();
    for (position, &token) in prompt.iter().enumerate() {
        runner
            .produce(token, position, &mut sequential)
            .expect("sequential prompt forward pass succeeds");
    }

    let mut chunked = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    runner.reset();
    runner
        .prefill_chunk_with_status(&prompt, 0, &mut chunked, true)
        .expect("single-chunk prompt forward pass succeeds");

    let sequential_top = top_k(&sequential, 12);
    let chunked_top = top_k(&chunked, 12);
    let max_abs = sequential
        .iter()
        .zip(&chunked)
        .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
        .fold(0.0f32, f32::max);
    let rmse = (sequential
        .iter()
        .zip(&chunked)
        .map(|(left, right)| {
            let delta = left.to_f32() - right.to_f32();
            delta * delta
        })
        .sum::<f32>()
        / sequential.len() as f32)
        .sqrt();
    eprintln!(
        "sequential top={:?}; chunked top={:?}; max_abs={max_abs} rmse={rmse}",
        sequential_top.first(),
        chunked_top.first()
    );
    for (rank, (sequential, chunked)) in sequential_top.iter().zip(&chunked_top).enumerate() {
        eprintln!(
            "rank={} sequential={sequential:?} chunked={chunked:?}",
            rank + 1
        );
    }

    let config = runtime::GenerationConfig {
        shaping: selection::ShapingConfig::new(0.0, 1, Some(0.95), 1.0, None)
            .expect("greedy shaping is valid"),
        max_new_tokens: 12,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: runtime::RateControl::default(),
    };
    let mut decoded_ids = Vec::new();
    let vocab_size = runner.vocab_size();
    runner.reset();
    runtime::run_raw_completion_chunked(
        &mut runner,
        &tokenizer,
        &prompt,
        &config,
        params.max_context,
        vocab_size,
        128,
        |event| {
            if let runtime::RawDecodeProgress::Token { id, .. } = event {
                decoded_ids.push(id);
            }
        },
    )
    .expect("chunked generation loop succeeds");
    eprintln!(
        "chunked generated ids={decoded_ids:?} text={:?}",
        tokenizer.decode(&decoded_ids, true)
    );
    assert_eq!(
        sequential_top.first().map(|row| row.0),
        chunked_top.first().map(|row| row.0),
        "the CLI's default single-chunk prefill must preserve the sequential top token"
    );
    assert_eq!(
        decoded_ids.first().copied(),
        sequential_top.first().map(|row| row.0),
        "the shared greedy decode loop must sample the sequential top token"
    );
}
