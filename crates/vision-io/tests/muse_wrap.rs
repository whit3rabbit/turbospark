//! `wrap_bare_image_placeholders`: a template that renders only the image
//! placeholder (Muse-Glimmer's `<|patch|>`) gets `image_start` / `image_end`
//! added around it before the splice, as its processor does.

use turbospark_vision_io::{
    mrope_position_triples, splice_and_walk, wrap_bare_image_placeholders, GridThw,
    VisionSpecialIds,
};

const START: i32 = 80;
const PAD: i32 = 92;
const END: i32 = 81;

#[test]
fn a_bare_placeholder_gains_both_markers() {
    let ids = [1, 2, PAD, 3];
    assert_eq!(
        wrap_bare_image_placeholders(&ids, PAD, START, END),
        vec![1, 2, START, PAD, END, 3]
    );
}

#[test]
fn each_of_several_images_is_wrapped() {
    let ids = [PAD, 5, PAD];
    assert_eq!(
        wrap_bare_image_placeholders(&ids, PAD, START, END),
        vec![START, PAD, END, 5, START, PAD, END]
    );
}

#[test]
fn a_template_that_renders_its_own_start_is_left_alone() {
    // The Qwen shape: start already precedes the pad, and its end follows.
    let ids = [1, START, PAD, END, 2];
    assert_eq!(wrap_bare_image_placeholders(&ids, PAD, START, END), ids);
}

#[test]
fn wrapping_twice_is_the_same_as_once() {
    let once = wrap_bare_image_placeholders(&[1, PAD, 2], PAD, START, END);
    assert_eq!(wrap_bare_image_placeholders(&once, PAD, START, END), once);
}

#[test]
fn the_walk_counts_a_wrapped_image_and_refuses_an_unwrapped_one() {
    let special = VisionSpecialIds {
        vision_start: START,
        image_pad: PAD,
    };
    let grid = GridThw::new(1, 4, 4); // 4 merged tokens at merge 2
    let bare = [1, PAD, 2];
    // Unwrapped: the walk counts start-then-pad pairs, finds none, and the
    // one grid has no image block to describe.
    assert!(splice_and_walk(&bare, &[grid], special, 2).is_err());
    let wrapped = wrap_bare_image_placeholders(&bare, PAD, START, END);
    let spliced = splice_and_walk(&wrapped, &[grid], special, 2).expect("walks");
    assert_eq!(
        spliced.ids,
        vec![1, START, PAD, PAD, PAD, PAD, END, 2],
        "the run sits between the markers"
    );
    assert_eq!(spliced.positions.spans.len(), 1);
    assert_eq!(
        (
            spliced.positions.spans[0].start,
            spliced.positions.spans[0].len
        ),
        (2, 4)
    );
    // And the position table is consistent with the ids it was built for.
    assert_eq!(
        mrope_position_triples(&spliced.ids, &[grid], special, 2)
            .unwrap()
            .triples
            .len(),
        spliced.ids.len()
    );
}
