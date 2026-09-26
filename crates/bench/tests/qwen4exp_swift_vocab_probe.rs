#![cfg(target_os = "macos")]
//! Real-install probe for token IDs emitted by the Swift Qwen3.8 IQ2_XS tier.
//!
//! The model advertises 248,320 output rows while its HF tokenizer has fewer
//! named tokens. Keep this check separate from the REAP-288 quality baseline.
//!
//! ```sh
//! TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR=/tmp/turbospark-qwen4exp-swift-iq2-xs.gturbo \
//!   cargo test -p turbospark-bench --test qwen4exp_swift_vocab_probe \
//!   --release -- --ignored --nocapture
//! ```

use runtime::{run_raw_completion, GenerationConfig, RateControl, RawDecodeProgress};
use selection::ShapingConfig;
use tokenizer::{Message, Role};
use turbospark_bench::{
    protocol::{
        PROTOCOL_CASES, PROTOCOL_MAX_NEW, PROTOCOL_TEMPERATURE, PROTOCOL_TOP_K, PROTOCOL_TOP_P,
    },
    real_model::{open_model_runner_with_context, protocol_parameters},
};

#[test]
#[ignore = "real model: requires the pinned Swift IQ2_XS install"]
fn swift_iq2_xs_does_not_emit_unmapped_vocab_ids() {
    let dir = std::env::var_os("TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR")
        .expect("set TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR");
    let params = protocol_parameters(model_io::ModelFamily::Qwen4Exp);
    let (mut runner, tokenizer) =
        open_model_runner_with_context(std::path::Path::new(&dir), 16, params.max_context)
            .expect("Swift IQ2_XS install opens");
    let case = PROTOCOL_CASES[1];
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, case.content)])
        .expect("protocol prompt renders");
    let prompt_ids = tokenizer.encode(&rendered, false);
    let shaping = ShapingConfig::new(
        PROTOCOL_TEMPERATURE,
        PROTOCOL_TOP_K,
        Some(PROTOCOL_TOP_P),
        1.0,
        Some(case.seed),
    )
    .expect("protocol shaping is valid");
    let config = GenerationConfig {
        shaping,
        max_new_tokens: PROTOCOL_MAX_NEW,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: RateControl::default(),
    };
    let vocab_size = runner.vocab_size();
    let mut generated = Vec::new();
    let result = run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt_ids,
        &config,
        params.max_context,
        vocab_size,
        |event| {
            if let RawDecodeProgress::Token { id, .. } = event {
                generated.push(id);
            }
        },
    )
    .expect("protocol generation completes");
    let unmapped: Vec<_> = generated
        .iter()
        .copied()
        .filter(|&id| tokenizer.id_to_token(id).is_none())
        .collect();
    eprintln!(
        "{}: {} tokens, {:?}, model_vocab={}, tokenizer_vocab={}, unmapped={:?}",
        case.id, result.new_tokens, result.reason, vocab_size, tokenizer.vocab_size, unmapped
    );
    eprintln!("decoded output: {:?}", tokenizer.decode(&generated, true));
    assert!(
        !generated.is_empty(),
        "the protocol case produced no tokens"
    );
    assert!(
        unmapped.is_empty(),
        "the model sampled IDs absent from the tokenizer: {unmapped:?}"
    );
}
