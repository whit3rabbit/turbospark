//! The vision arm's ROUTING assertion, which no output-level test can make.
//!
//! `tests/real_backend.rs` proves a picture reaches the model over both wire
//! formats; it cannot prove WHICH prefill driver served it, because both
//! drivers produce identical bytes -- the identity proof is the runtime
//! crate's `vision_chunked_synthetic.rs`. What differs at the seam is
//! progress granularity: the sequential loop emits one `Prefill` event per
//! prompt token, the chunked driver one per chunk span. The exec loop drops
//! `Prefill` events, so the count is invisible over HTTP -- this drives
//! `ChatModel::run_completion` DIRECTLY and counts them.
//!
//!   TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/models/qwen38-27b-vision.gturbo \
//!   TURBOSPARK_VISION_PAGE=~/models/vision-probe-qwen38/imgs/page.png \
//!     cargo test -p turbospark-server --lib --release -- --ignored --nocapture

use std::path::PathBuf;
use std::sync::Arc;

use foundation::{prefill_chunk_spans, DEFAULT_CHUNK_SIZE};
use runtime::{
    GenerationConfig, KvQuant, LoadPolicy, RateControl, RawDecodeProgress, Speculation,
    SpeculativeDrafter, SteeringPolicy,
};
use selection::ShapingConfig;
use tokenizer::{ContentPart, Message, ReasoningEffort, Role};

use crate::model::ChatModel;
use crate::vision::{self, RequestImages};
use crate::RealChatModel;

/// The chunked arm in `run_with_images` is the thing under test: deleting it
/// (always calling `run_raw_completion_cancellable`) turns the Prefill count
/// from one per span into one per token and reddens the central assertion.
#[test]
#[ignore = "needs a real vision install (TURBOSPARK_QWEN38_VISION_INSTALL_DIR) and a page \
            (TURBOSPARK_VISION_PAGE)"]
fn an_image_turn_prefills_in_chunk_spans_not_per_token() {
    let (Some(dir), Some(page)) = (
        std::env::var_os("TURBOSPARK_QWEN38_VISION_INSTALL_DIR").map(PathBuf::from),
        std::env::var_os("TURBOSPARK_VISION_PAGE").map(PathBuf::from),
    ) else {
        eprintln!(
            "real_model_tests: TURBOSPARK_QWEN38_VISION_INSTALL_DIR and TURBOSPARK_VISION_PAGE \
             are not both set; skipping."
        );
        return;
    };

    // Pinned exactly as `tests/real_backend.rs`'s image test pins them, and
    // for one more reason: the span count below assumes `reused == 0`, which
    // the pinned-off prefix reuse plus `set_prompt_vision`'s taint both give.
    let model = RealChatModel::open(
        &dir,
        Some(4096),
        Some(16),
        Default::default(),
        Speculation::Off,
        SpeculativeDrafter::Auto,
        crate::GuardrailConfig::OFF,
        SteeringPolicy::off(),
        LoadPolicy::default(),
        ReasoningEffort::Off,
        None,
        false,
        1,
        None,
        KvQuant::Off,
        runtime::ExpertResidency::Auto,
    )
    .expect("the vision install should open");
    let model: Arc<dyn ChatModel> = Arc::new(model);

    // The same render `plan` does: one text part and one image part on one
    // user turn. The template emits a single pad marker for the image; the
    // splice below expands it into the merged-token run the trunk prefills.
    let messages = vec![Message::with_parts(
        Role::User,
        vec![
            ContentPart::Text("Transcribe the text in this image.".to_string()),
            ContentPart::Image,
        ],
    )];
    let prompt = model
        .tokenizer()
        .apply_chat_template_with_reasoning(&messages, ReasoningEffort::Off)
        .expect("the template should render");
    let marker_ids = model.tokenizer().encode(&prompt, false);

    let info = model.vision().expect("the vision install serves images");
    let png = std::fs::read(&page).expect("the test page should be readable");
    let preprocessed = vision::preprocess_all(&[png], &info).expect("the page should preprocess");
    let grids: Vec<_> = preprocessed.iter().map(|p| p.grid).collect();
    let spliced = turbospark_vision_io::splice_and_walk(
        &marker_ids,
        &grids,
        info.specials,
        info.params.merge_size,
    )
    .expect("the image should splice");
    let prompt_ids = spliced.ids;
    // The assertion below is a span COUNT, so the prompt must actually span:
    // at or under one chunk the two drivers emit the same count and the test
    // would prove nothing. A real page merges to well over a thousand.
    assert!(
        prompt_ids.len() > DEFAULT_CHUNK_SIZE as usize,
        "the spliced prompt is {} tokens; the test page should merge to more \
         than one chunk's worth",
        prompt_ids.len()
    );
    let images = RequestImages {
        images: preprocessed,
        positions: spliced.positions,
    };

    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).expect("greedy shaping is valid"),
        max_new_tokens: 8,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: RateControl::default(),
    };

    let mut prefill_events = 0usize;
    let mut last_done = 0usize;
    let result = model
        .run_completion(&prompt_ids, &config, Some(&images), &|| false, &mut |p| {
            if let RawDecodeProgress::Prefill { done, .. } = p {
                prefill_events += 1;
                last_done = done;
            }
        })
        .expect("the image turn should generate");

    // ONE EVENT PER CHUNK SPAN, not one per token. `prefill_chunk_spans` is
    // the walk's own planner queried at `reused == 0`, so the expected count
    // is exact rather than a threshold.
    let expected = prefill_chunk_spans(prompt_ids.len(), 0, DEFAULT_CHUNK_SIZE as usize).len();
    assert_eq!(
        prefill_events,
        expected,
        "the image turn prefilled {} tokens in {} events, expected {} chunk \
         spans; the sequential loop would have emitted one per token",
        prompt_ids.len(),
        prefill_events,
        expected
    );
    assert_eq!(last_done, prompt_ids.len());
    assert_eq!(result.prompt_tokens, prompt_ids.len());
    assert!(result.new_tokens > 0, "the turn should have decoded");
}
