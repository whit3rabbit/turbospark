//! Image content parts on both endpoints (ROADMAP M-V8).
//!
//! # One decoder serves both wire formats, and that is the vendored crate's
//! doing rather than this module's
//!
//! `/v1/messages` translates into the OpenAI request the chat route already
//! understands, and `anyllm_translate`'s `message_map/request.rs` maps an
//! Anthropic `ContentBlock::Image` onto `ChatContentPart::ImageUrl` -- a
//! base64 source becomes `data:<media_type>;base64,<data>` and a URL source
//! passes through. So both endpoints arrive here as one shape. Reading the
//! vendored source before implementing a wire shape is this crate's Gotcha 4,
//! and it saved the second implementation.
//!
//! # A REMOTE URL IS REFUSED RATHER THAN FETCHED
//!
//! Fetching one would make this server an HTTP client driven by request
//! content: an SSRF surface, a timeout budget, and a redirect policy, none of
//! which belongs in a local inference server. The refusal names the shape a
//! caller can send instead.
//!
//! # Everything here is PURE, and that is what lets it run before the model
//!
//! Decode and preprocess need no runner, so a malformed image costs a 400
//! rather than a lock on the one runner this process has. Only `encode_image`
//! needs the GPU, and that happens inside the backend's own lock beside the
//! generation it belongs to (`ChatModel::run_completion`) -- see that method
//! for why the two cannot be separate calls.

use anyllm_translate::openai::{ChatContentPart, ChatMessage};
use turbospark_vision_io::{MropePositions, PreprocessParams, PreprocessedImage, VisionSpecialIds};

/// Above this many image parts in one request, `handler::plan` refuses
/// before decoding any of them. A body-size limit alone
/// (`MAX_REQUEST_BODY_BYTES`) bounds total bytes but not COUNT: nothing
/// stops a request from splitting its budget across an unreasonable number
/// of tiny images, each paying its own decode, preprocess, and (once a
/// vision-capable backend is attached) tower-encode cost.
pub(crate) const MAX_IMAGES_PER_REQUEST: usize = 64;

/// How many image parts one message carries, for the whole-request cap in
/// `handler::plan` -- counted separately from [`image_bytes`] so the cap can
/// be checked before any payload is decoded.
pub(crate) fn image_count(message: &ChatMessage) -> usize {
    let Some(anyllm_translate::openai::ChatContent::Parts(parts)) = &message.content else {
        return 0;
    };
    parts
        .iter()
        .filter(|p| matches!(p, ChatContentPart::ImageUrl { .. }))
        .count()
}

/// What a backend needs to know to accept images.
///
/// Read off the INSTALL by the backend rather than assembled here: the pixel
/// budget and the special ids are per-checkpoint, and a constant in this
/// crate would be AGENTS.md Gotcha 38's shape.
#[derive(Debug, Clone)]
pub struct VisionInfo {
    pub params: PreprocessParams,
    pub specials: VisionSpecialIds,
}

/// One request's images, preprocessed, plus the position table for the
/// spliced prompt.
#[derive(Debug, Clone)]
pub struct RequestImages {
    pub images: Vec<PreprocessedImage>,
    pub positions: MropePositions,
}

/// The image parts of one message, in order, as raw encoded bytes.
///
/// Order-preserving: the nth image pairs with the nth marker run the template
/// renders, so a container that reordered would pair each picture with the
/// wrong span.
pub(crate) fn image_bytes(message: &ChatMessage) -> Result<Vec<Vec<u8>>, String> {
    let Some(anyllm_translate::openai::ChatContent::Parts(parts)) = &message.content else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for part in parts {
        if let ChatContentPart::ImageUrl { image_url } = part {
            out.push(decode_data_url(&image_url.url)?);
        }
    }
    Ok(out)
}

/// Check every image part's URL SHAPE without decoding its payload.
///
/// **Validated whether or not this server can serve images**, and that is a
/// consistency decision: a remote URL is a malformed request for this server
/// however it is configured, so refusing it on a vision install and accepting
/// it on a text-only one would leave a client unable to tell which problem it
/// had. The capability drop is reported separately and is a different thing.
///
/// The payload is NOT decoded here, so a text-only backend pays nothing for
/// images it will discard -- which on a multi-megabyte data URL is the whole
/// cost.
pub(crate) fn validate_urls(message: &ChatMessage) -> Result<(), String> {
    let Some(anyllm_translate::openai::ChatContent::Parts(parts)) = &message.content else {
        return Ok(());
    };
    for part in parts {
        if let ChatContentPart::ImageUrl { image_url } = part {
            split_data_url(&image_url.url)?;
        }
    }
    Ok(())
}

/// Whether a message carries any image part at all.
///
/// Separate from [`image_bytes`] so a text-only install can REPORT the drop
/// without paying a base64 decode for images it will not use.
pub(crate) fn has_images(message: &ChatMessage) -> bool {
    matches!(
        &message.content,
        Some(anyllm_translate::openai::ChatContent::Parts(parts))
            if parts
                .iter()
                .any(|p| matches!(p, ChatContentPart::ImageUrl { .. }))
    )
}

/// `data:<media-type>;base64,<payload>` to bytes.
///
/// The media type is parsed but NOT enforced: `turbospark_vision_io` sniffs
/// the real format from the bytes, and a caller whose `media_type` says PNG
/// over a JPEG payload should get the picture rather than a lecture. What is
/// enforced is `;base64`, because a percent-encoded data URL is a different
/// decoding and silently reading it as base64 yields garbage pixels.
/// `pub` because `crates/ffi` decodes the same payloads for its own image
/// parts and a second copy of a base64 decoder is a second thing to get
/// wrong. Both front ends therefore accept exactly the same spelling.
pub fn decode_data_url(url: &str) -> Result<Vec<u8>, String> {
    base64_decode(split_data_url(url)?)
}

/// The payload of a `data:...;base64,` URL, or the reason it is not one.
///
/// Split out from the decode so the SHAPE can be checked without paying for
/// the bytes -- see [`validate_urls`].
fn split_data_url(url: &str) -> Result<&str, String> {
    // Case-INSENSITIVE scheme and `;base64` token match: URI schemes are
    // case-insensitive by RFC 3986, and a client that sends `DATA:` or
    // `;BASE64` (both seen from real HTTP libraries that upper-case headers
    // or URI components) used to be refused as "not a data: URL" for a
    // spelling this server could serve perfectly well. `url.get(..5)`
    // rather than `&url[..5]`: it returns `None` instead of panicking when
    // byte 5 is not a char boundary or the URL is shorter than 5 bytes, so
    // an arbitrary short or multibyte-prefixed string cannot panic here.
    let is_data_scheme = url
        .get(..5)
        .is_some_and(|s| s.eq_ignore_ascii_case("data:"));
    if !is_data_scheme {
        return Err(format!(
            "image_url must be a data: URL; this server does not fetch remote images (got {})",
            elide(url)
        ));
    }
    let rest = &url[5..];
    let Some((meta, payload)) = rest.split_once(',') else {
        return Err(
            "malformed data: URL, expected a comma between the media type and the payload"
                .to_string(),
        );
    };
    if !meta
        .split(';')
        .any(|token| token.eq_ignore_ascii_case("base64"))
    {
        return Err(format!(
            "data: URL is not base64-encoded (media type {meta:?}); send ;base64 data"
        ));
    }
    Ok(payload)
}

/// A URL shortened for an error message.
///
/// A data URL can be megabytes, and echoing one into a 400 body makes the
/// error unreadable and the log unusable.
///
/// Truncates by CHARACTER count, not by byte offset: a request-controlled
/// URL whose byte 60 falls inside a multibyte UTF-8 sequence (a remote-URL
/// refusal is reachable with an arbitrary client-supplied string) would
/// panic on a raw `&url[..60]` slice and drop the connection with no
/// response.
fn elide(url: &str) -> String {
    const MAX: usize = 60;
    let mut chars = url.chars();
    let head: String = chars.by_ref().take(MAX).collect();
    if chars.next().is_none() {
        return head;
    }
    format!("{head}... [{} bytes]", url.len())
}

/// Standard OR URL-safe (RFC 4648 sec. 5, `-`/`_` in place of `+`/`/`)
/// base64, tolerating missing padding and embedded whitespace.
///
/// Both alphabets rather than one: a data URL is JSON-embedded text, not a
/// URL path or query component, so nothing here needs the URL-safe
/// alphabet's actual property (no percent-encoding for `+`/`/`) -- but some
/// client libraries emit it anyway for images, and refusing a byte-for-byte
/// valid encoding because it used the other of two standard alphabets serves
/// no one. The two alphabets never collide (`-`/`_` and `+`/`/` are disjoint
/// symbols for the same six-bit values), so accepting both costs nothing on
/// the ambiguity a caller might otherwise worry about.
///
/// Hand-rolled rather than a dependency: it is twenty-some lines, this is the
/// only caller in the workspace, and the alternative is a new external crate
/// in a server that already declines `anyllm_translate`'s heavier features
/// (Gotcha 4). Whitespace is skipped because a JSON body may carry a wrapped
/// payload, and padding is optional because both endpoints' clients differ on
/// whether they send it.
/// `pub` for the reason [`decode_data_url`] is: `crates/ffi` accepts a bare
/// payload as well as a full data URL, and this is the primitive under both.
pub fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    const INVALID: u8 = 0xFF;
    let value = |c: u8| -> u8 {
        match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => INVALID,
        }
    };
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for (i, c) in input.bytes().enumerate() {
        if c.is_ascii_whitespace() || c == b'=' {
            continue;
        }
        let v = value(c);
        if v == INVALID {
            return Err(format!(
                "invalid base64 character {:?} at offset {i}",
                c as char
            ));
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    if out.is_empty() {
        return Err("data: URL carries an empty payload".to_string());
    }
    Ok(out)
}

/// Decode and preprocess every image, in request order.
pub(crate) fn preprocess_all(
    encoded: &[Vec<u8>],
    info: &VisionInfo,
) -> Result<Vec<PreprocessedImage>, String> {
    encoded
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            let decoded = turbospark_vision_io::decode_image_bytes(bytes)
                .map_err(|e| format!("image {i}: {e}"))?;
            turbospark_vision_io::preprocess(&decoded, &info.params)
                .map_err(|e| format!("image {i}: {e}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_base64_data_url_decodes() {
        // "hello" in standard base64.
        assert_eq!(
            decode_data_url("data:image/png;base64,aGVsbG8=").unwrap(),
            b"hello"
        );
    }

    /// Padding is optional and whitespace is skipped, because clients differ
    /// on both and neither changes the bytes.
    #[test]
    fn padding_and_whitespace_are_tolerated() {
        assert_eq!(
            decode_data_url("data:image/png;base64,aGVsbG8").unwrap(),
            b"hello"
        );
        assert_eq!(
            decode_data_url("data:image/png;base64,aGVs\n bG8=").unwrap(),
            b"hello"
        );
    }

    /// **The refusal that keeps this server from becoming an HTTP client.**
    #[test]
    fn a_remote_url_is_refused_and_names_what_to_send() {
        let err = decode_data_url("https://example.com/cat.png").unwrap_err();
        assert!(err.contains("does not fetch remote"), "{err}");
        assert!(err.contains("data:"), "{err}");
    }

    /// A percent-encoded data URL is a DIFFERENT decoding; reading it as
    /// base64 yields garbage pixels rather than an error, so the encoding is
    /// checked rather than assumed.
    #[test]
    fn a_non_base64_data_url_is_refused() {
        let err = decode_data_url("data:image/png,%89PNG").unwrap_err();
        assert!(err.contains("not base64"), "{err}");
    }

    #[test]
    fn an_invalid_character_is_named_with_its_offset() {
        let err = decode_data_url("data:image/png;base64,aGVs!G8=").unwrap_err();
        assert!(err.contains("invalid base64"), "{err}");
    }

    /// F25: a URI scheme is case-insensitive by RFC 3986, and `;base64` is
    /// an ordinary token on the media-type side of the comma with no
    /// case-sensitivity rule of its own -- but this server used to require
    /// both spelled exactly lowercase.
    #[test]
    fn the_scheme_and_base64_token_are_case_insensitive() {
        assert_eq!(
            decode_data_url("DATA:image/png;BASE64,aGVsbG8=").unwrap(),
            b"hello"
        );
        assert_eq!(
            decode_data_url("Data:image/png;Base64,aGVsbG8=").unwrap(),
            b"hello"
        );
    }

    /// F25: RFC 4648's URL-safe alphabet (`-`/`_` for `+`/`/`) must decode
    /// the same bytes as the standard one -- some client libraries emit it
    /// for embedded images even though nothing here needs the property it
    /// exists for (no percent-encoding of `+`/`/`).
    #[test]
    fn the_url_safe_alphabet_decodes_the_same_bytes_as_standard() {
        // Swapping every `+`/`/` for `-`/`_` in a real standard-base64
        // payload must decode to the identical bytes.
        assert_eq!(
            base64_decode("+/+/").unwrap(),
            base64_decode("-_-_").unwrap()
        );
    }

    /// A megabyte data URL must not end up in a 400 body verbatim.
    #[test]
    fn a_long_url_is_elided_in_the_error() {
        let long = format!("https://example.com/{}", "a".repeat(5000));
        let err = decode_data_url(&long).unwrap_err();
        assert!(err.len() < 200, "error was {} bytes", err.len());
        assert!(err.contains(&format!("{} bytes", long.len())), "{err}");
    }

    /// `elide` used to slice at a raw byte offset (`&url[..60]`), which
    /// panics if byte 60 is not a char boundary. `"h"` plus thirty `"\u{00e9}"`
    /// (2 bytes each) puts the 30th `\u{00e9}` at bytes 59-60, so byte 60 sits
    /// mid-character -- exactly the case a naive slice cannot survive.
    #[test]
    fn a_multibyte_character_straddling_the_truncation_point_does_not_panic() {
        let url = format!("h{}", "\u{00e9}".repeat(30));
        assert!(!url.is_char_boundary(60), "fixture must straddle byte 60");
        let err = decode_data_url(&url).unwrap_err();
        assert!(err.contains("does not fetch remote"), "{err}");
    }
}
