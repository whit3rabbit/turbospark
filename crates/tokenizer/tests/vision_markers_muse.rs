//! Muse-Glimmer's bare image placeholder against the REAL tokenizer and chat
//! template (`docs/MUSEGLIMMER_VISION_PHASE0.md`).
//!
//! The template renders one `<|patch|>` per image and no markers; the
//! processor adds `<|image_start|>` and `<|image_end|>`. The ids for those two
//! are constants of the tower kind (`repack::qwen36_config`) because the
//! config does not name them, so this test is what ties them to the loaded
//! tokenizer: they must be what the tokenizer resolves, not what a
//! `tokenizer.json` read once said.
//!
//! ```sh
//! TURBOSPARK_MUSE_INSTALL_DIR=/Volumes/T7/tshome-muse/models/text/museglimmer-vision.gturbo \
//!   cargo test -p turbospark-tokenizer --test vision_markers_muse -- --ignored
//! ```

use std::path::PathBuf;

use turbospark_tokenizer::MfTokenizer;

const IMAGE_START: i32 = 200080;
const IMAGE_PAD: i32 = 200092;
const IMAGE_END: i32 = 200081;

fn load() -> MfTokenizer {
    let dir = PathBuf::from(
        std::env::var("TURBOSPARK_MUSE_INSTALL_DIR")
            .expect("set TURBOSPARK_MUSE_INSTALL_DIR to a Muse-Glimmer install"),
    );
    MfTokenizer::load_from_dir(&dir).expect("the install's tokenizer loads")
}

#[test]
#[ignore = "needs a Muse-Glimmer install directory"]
fn the_kinds_constants_are_what_the_loaded_tokenizer_resolves() {
    let tok = load();
    assert_eq!(tok.token_to_id("<|image_start|>"), Some(IMAGE_START));
    assert_eq!(tok.token_to_id("<|patch|>"), Some(IMAGE_PAD));
    assert_eq!(tok.token_to_id("<|image_end|>"), Some(IMAGE_END));
}

#[test]
#[ignore = "needs a Muse-Glimmer install directory"]
fn the_bare_placeholder_check_accepts_the_real_template_and_the_qwen_check_refuses_it() {
    let tok = load();
    tok.verify_bare_image_placeholder(IMAGE_START, IMAGE_PAD, IMAGE_END)
        .expect("one <|patch|> and no markers");
    // The marker check that every other tower uses wants a start marker the
    // template itself renders; this template renders none.
    let err = tok
        .verify_image_markers(IMAGE_START, IMAGE_PAD)
        .unwrap_err();
    assert!(err.to_string().contains("0 occurrence"), "{err}");
    // And a bogus end id is refused by name.
    assert!(tok
        .verify_bare_image_placeholder(IMAGE_START, IMAGE_PAD, 999_999_999)
        .is_err());
}
