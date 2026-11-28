//! Real-checkpoint parity for the ACE-Step 1.5 text encoder. The
//! backbone itself is gated by the `lm::qwen3` tests; this checks the
//! Qwen3-Embedding checkpoint and the caption / lyric split upstream
//! uses. Ignored: point TURBOSPARK_ACE_STEP15_TEXT at the checkpoint's
//! `Qwen3-Embedding-0.6B/` directory and TURBOSPARK_ACE_STEP15_TEXT_REF at
//! the output of `tools/gen_ace_step15_text_fixtures.py`.

use std::path::{Path, PathBuf};

use super::tests::{read_npy, rel_err};
use super::text::{TextEncoder, MAX_CAPTION_TOKENS};

pub(super) fn read_i32(path: &Path) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(
        header.contains("'<i4'"),
        "{}: not little-endian i32",
        path.display()
    );
    bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Measured on M4 Max: caption hidden 5.0e-6 relative (69 tokens); lyric rows are
/// table copies and match exactly.
#[test]
#[ignore = "needs TURBOSPARK_ACE_STEP15_TEXT and TURBOSPARK_ACE_STEP15_TEXT_REF"]
fn real_text_encoder_matches_transformers() {
    let (Ok(dir), Ok(ref_dir)) = (
        std::env::var("TURBOSPARK_ACE_STEP15_TEXT"),
        std::env::var("TURBOSPARK_ACE_STEP15_TEXT_REF"),
    ) else {
        panic!("set TURBOSPARK_ACE_STEP15_TEXT and TURBOSPARK_ACE_STEP15_TEXT_REF");
    };
    let ref_dir = PathBuf::from(ref_dir);
    let mut enc = TextEncoder::load(Path::new(&dir)).unwrap();
    let caption = read_i32(&ref_dir.join("caption_ids.npy"));
    let lyric = read_i32(&ref_dir.join("lyric_ids.npy"));
    let hidden = enc.encode_caption(&caption).unwrap();
    let (_, want_hidden) = read_npy(&ref_dir.join("caption_hidden.npy"));
    let embeds = enc.embed_lyrics(&lyric).unwrap();
    let (_, want_embeds) = read_npy(&ref_dir.join("lyric_embeds.npy"));
    let (he, le) = (
        rel_err(&hidden, &want_hidden),
        rel_err(&embeds, &want_embeds),
    );
    eprintln!("text: caption hidden {he:.2e} lyric embeds {le:.2e}");
    assert!(he <= 2e-5, "caption hidden: relative error {he:.3e}");
    assert_eq!(le, 0.0, "lyric rows are table copies");

    assert!(enc.encode_caption(&[]).is_err());
    assert!(enc
        .encode_caption(&vec![0; MAX_CAPTION_TOKENS + 1])
        .is_err());
    assert!(enc.embed_lyrics(&[-1]).is_err());
    assert!(enc.embed_lyrics(&[i32::MAX]).is_err());
}
