//! Unit tests for turn speculation gating, wire serialization, and image extraction.

use super::*;
use crate::generate::prompt::collect_image_parts;
use crate::wire::WireMessage;

/// The per-turn half of the speculation decision, in all four states.
///
/// Cheap enough to be exhaustive, and worth being: three of the four
/// cells are "decode sequentially" and the one that is not is the only
/// path in this crate that reaches a batched verify.
#[test]
fn a_turn_speculates_only_when_the_session_can_and_the_turn_is_greedy() {
    assert_eq!(turn_block(Some(2), true), Some(2));
    // Sampled: the session's block is DISCARDED rather than honoured,
    // because acceptance is exact only at temperature 0.
    assert_eq!(turn_block(Some(2), false), None);
    // No drafter: greedy does not conjure one.
    assert_eq!(turn_block(None, true), None);
    assert_eq!(turn_block(None, false), None);
}

fn parse(json: &str) -> Vec<WireMessage> {
    serde_json::from_str(json).expect("wire messages parse")
}

/// **THE REGRESSION GUARD FOR WIDENING `content`.** Every caller that
/// predates images sends a bare string, and `untagged` is what keeps that
/// decoding to the same thing. A parts-only shape would have been an ABI
/// break dressed as a field.
#[test]
fn a_bare_string_content_still_decodes_and_reserializes_as_one() {
    let messages = parse(r#"[{"role":"user","content":"hi"}]"#);
    assert_eq!(messages[0].text(), "hi");
    assert!(messages[0].parts().is_none());
    assert_eq!(messages[0].image_parts().count(), 0);
    // Round-trips as a STRING, not as an object: `WindowFitOutcome`
    // hands `retained` straight back to the caller, so a re-serialized
    // message that changed shape would break every existing consumer.
    let back = serde_json::to_string(&messages).expect("serializes");
    assert!(back.contains(r#""content":"hi""#), "{back}");
}

/// A missing `content` is still the empty string rather than an error,
/// which is what `#[serde(default)]` meant before this widening too.
#[test]
fn an_absent_content_is_the_empty_string() {
    let messages = parse(r#"[{"role":"user"}]"#);
    assert_eq!(messages[0].text(), "");
    assert!(messages[0].parts().is_none());
}

/// Parts keep the order they arrived in, and `text()` sees only the
/// prose. Order is what pairs the nth picture with the nth marker run.
#[test]
fn ordered_parts_decode_in_order_and_text_skips_the_images() {
    let messages = parse(
        r#"[{"role":"user","content":[
             {"type":"image","path":"/a.png"},
             {"type":"text","text":"one"},
             {"type":"image","base64":"QQ=="},
             {"type":"text","text":"two"}]}]"#,
    );
    assert_eq!(messages[0].text(), "onetwo");
    assert_eq!(messages[0].parts().expect("parts").len(), 4);

    let images: Vec<_> = collect_image_parts(&messages).expect("shapes are valid");
    assert_eq!(images.len(), 2);
    // The FIRST image is the path one, because that is the order sent.
    match images[0].image_source().expect("a source") {
        crate::wire::ImageSource::Path(p) => assert_eq!(p, "/a.png"),
        crate::wire::ImageSource::Base64(_) => panic!("images were reordered"),
    }
    match images[1].image_source().expect("a source") {
        crate::wire::ImageSource::Base64(b) => assert_eq!(b, "QQ=="),
        crate::wire::ImageSource::Path(_) => panic!("images were reordered"),
    }
}

/// Images are collected ACROSS messages, still in order: a conversation
/// can carry a picture in an earlier turn as well as the current one.
#[test]
fn image_parts_are_collected_across_messages_in_order() {
    let messages = parse(
        r#"[{"role":"user","content":[{"type":"image","path":"/first.png"}]},
            {"role":"assistant","content":"ok"},
            {"role":"user","content":[{"type":"image","path":"/second.png"}]}]"#,
    );
    let images = collect_image_parts(&messages).expect("shapes are valid");
    let paths: Vec<_> = images
        .iter()
        .map(|i| match i.image_source().expect("a source") {
            crate::wire::ImageSource::Path(p) => p.to_string(),
            crate::wire::ImageSource::Base64(_) => unreachable!(),
        })
        .collect();
    assert_eq!(paths, ["/first.png", "/second.png"]);
}

/// Both spellings, or neither, is REFUSED rather than resolved by
/// precedence -- and refused BEFORE the engine lock, which is what makes
/// it cheap. A caller that sent both meant one of them, and picking
/// silently runs the wrong picture with no error anywhere.
#[test]
fn an_image_part_needs_exactly_one_source() {
    let both =
        parse(r#"[{"role":"user","content":[{"type":"image","path":"/a","base64":"QQ=="}]}]"#);
    let err = collect_image_parts(&both).expect_err("both sources is refused");
    assert!(err.contains("both"), "{err}");

    let neither = parse(r#"[{"role":"user","content":[{"type":"image"}]}]"#);
    let err = collect_image_parts(&neither).expect_err("no source is refused");
    assert!(err.contains("neither"), "{err}");
}

/// A conversation with no image parts collects nothing, which is the
/// condition every text-only turn takes and the one that keeps its path
/// byte-identical.
#[test]
fn a_text_only_conversation_collects_no_images() {
    let messages = parse(
        r#"[{"role":"user","content":"hi"},
            {"role":"user","content":[{"type":"text","text":"still text"}]}]"#,
    );
    assert!(collect_image_parts(&messages)
        .expect("no shapes to get wrong")
        .is_empty());
}
