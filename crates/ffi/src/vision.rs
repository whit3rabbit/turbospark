//! Image content parts on `ts_generate`: decode, preprocess, run the tower,
//! and hand the runner the injection map.
//!
//! # The order is forced, not chosen
//!
//! ```text
//! decode + preprocess  -> grid, merged_tokens   (needs no model)
//! render + encode      -> ONE <|image_pad|> per image
//! splice_and_walk      -> expanded ids + spans + position triples
//! encode_image         -> the tower's rows, per image
//! set_prompt_vision    -> the map the trunk reads at prefill
//! ```
//!
//! Preprocessing comes FIRST because the splice needs each image's
//! `merged_tokens`, and the template renders one marker whatever the size.
//! This is `crates/cli/src/generate/vision.rs`'s ordering, and that file is
//! the reference implementation rather than a sibling to keep in sync by
//! hand: everything it calls is in `turbospark-vision-io` or on
//! `RealForwardRunner`, so both front ends drive the same code and only the
//! error wording differs.
//!
//! # Everything is decoded before the tower runs
//!
//! A missing file, an unreadable PNG or a pixel budget the install does not
//! declare should cost a message rather than a GPU encode, so `prepare` is
//! called before any lock is taken and needs no model.
//!
//! # macOS only, and the refusal off it is elsewhere
//!
//! `RealForwardRunner` is macOS-only, so this module is too. A caller sending
//! an image to a SCRIPTED session, or to an install with no tower, is refused
//! by name in `generate` -- which is the portable half and the only place
//! that can answer on either platform.

use std::path::Path;

use turbospark_vision_io::{
    decode_image_bytes, decode_image_file, preprocess, PreprocessParams, PreprocessedImage,
    VisionSpecialIds,
};

use crate::wire::{ImageSource, WirePart};

/// One image, preprocessed and ready for the tower.
pub(crate) struct PreparedImage {
    /// What to name this image in an error. A path, or `image N` for bytes
    /// that never had one.
    pub(crate) label: String,
    pub(crate) image: PreprocessedImage,
}

/// Read the pixel budget and geometry off the INSTALL, never from a constant
/// here.
///
/// `crates/vision-io` Gotcha 6: these checkpoints declare
/// `size: {shortest_edge: 65536, longest_edge: 16777216}` -- pixel COUNTS
/// despite the naming -- against a generic library default of 3,136 ..
/// 1,003,520, a factor of 16 on the ceiling. Falling back to the generic pair
/// would resize every page to a fraction of its intended resolution and
/// produce a correct-looking, lower-quality answer, so an install carrying no
/// `preprocessor_config.json` is REFUSED by name rather than defaulted.
///
/// The geometry is cross-checked against the install's own `arch.vision`,
/// because the two files could disagree and the tower would then refuse at a
/// shape check far from the config that caused it.
///
/// Reads `runner.vision_dir()` rather than the caller's own model directory,
/// so a trunk with an attached vision SIDECAR (vision memory sidecar, Part
/// A3) finds `preprocessor_config.json` beside the sidecar's `manifest.json`
/// rather than the trunk's own install, which for a text-only trunk has no
/// such file at all.
pub(crate) fn preprocess_params(
    runner: &runtime::RealForwardRunner,
) -> Result<PreprocessParams, String> {
    let path = runner.vision_dir().join("preprocessor_config.json");
    let json = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "{}: {e}\n  this install declares no image preprocessing config; re-stream it \
             with its sidecars, or copy the checkpoint's preprocessor_config.json in",
            path.display()
        )
    })?;
    let params = PreprocessParams::from_preprocessor_config_json(&json)
        .map_err(|e| format!("{}: {e}", path.display()))?;

    let vision = runner.vision_config();
    let mismatch = |what: &str, config: usize, arch: i64| {
        format!(
            "{}: {what} is {config} but this install's manifest says {arch}; the two \
             describe different checkpoints",
            path.display()
        )
    };
    if params.patch_size != vision.patch_size as usize {
        return Err(mismatch("patch_size", params.patch_size, vision.patch_size));
    }
    if params.merge_size != vision.spatial_merge_size as usize {
        return Err(mismatch(
            "merge_size",
            params.merge_size,
            vision.spatial_merge_size,
        ));
    }
    if params.temporal_patch_size != vision.temporal_patch_size as usize {
        return Err(mismatch(
            "temporal_patch_size",
            params.temporal_patch_size,
            vision.temporal_patch_size,
        ));
    }
    Ok(params)
}

/// The special ids the splice and the walk key on, read off the install.
pub(crate) fn special_ids(runner: &runtime::RealForwardRunner) -> VisionSpecialIds {
    let vision = runner.vision_config();
    VisionSpecialIds {
        vision_start: vision.vision_start_token_id as i32,
        image_pad: vision.image_token_id as i32,
    }
}

/// Decode and preprocess every image part, in order.
///
/// Order-preserving: the nth prepared image pairs with the nth marker run the
/// template renders, so a reordering silently pairs each picture with the
/// wrong span.
pub(crate) fn prepare(
    parts: &[&WirePart],
    params: &PreprocessParams,
) -> Result<Vec<PreparedImage>, String> {
    parts
        .iter()
        .enumerate()
        .map(|(i, part)| {
            let (label, decoded) = match part.image_source()? {
                ImageSource::Path(p) => (
                    p.to_string(),
                    decode_image_file(Path::new(p)).map_err(|e| format!("image {p}: {e}"))?,
                ),
                // A bare payload OR a full `data:<media>;base64,<data>` URL,
                // both through `turbospark_server::vision`'s own decoder
                // rather than a second copy of one here.
                ImageSource::Base64(b) => {
                    let label = format!("image {}", i + 1);
                    let bytes = if b.starts_with("data:") {
                        turbospark_server::vision::decode_data_url(b)
                    } else {
                        turbospark_server::vision::base64_decode(b)
                    }
                    .map_err(|e| format!("{label}: {e}"))?;
                    (
                        label.clone(),
                        decode_image_bytes(&bytes).map_err(|e| format!("{label}: {e}"))?,
                    )
                }
            };
            let image = preprocess(&decoded, params).map_err(|e| format!("{label}: {e}"))?;
            Ok(PreparedImage { label, image })
        })
        .collect()
}

/// Run the tower over `images` and hand the runner the injection map for
/// `rendered`.
///
/// Returns the SPLICED id sequence, which is what gets prefilled -- the
/// caller's `rendered` carries one placeholder per image and is not what the
/// model sees.
///
/// **The tower runs per image and its scratch is dropped per image**, which
/// is the constant-memory property the whole vision design exists for: peak
/// vision residency is two streamer slots plus ONE page of scratch however
/// many pages a session walks.
pub(crate) fn attach(
    runner: &mut runtime::RealForwardRunner,
    rendered: &[i32],
    images: &[PreparedImage],
    params: &PreprocessParams,
) -> Result<Vec<i32>, String> {
    let grids: Vec<_> = images.iter().map(|p| p.image.grid).collect();
    let spliced = turbospark_vision_io::splice_and_walk(
        rendered,
        &grids,
        special_ids(runner),
        params.merge_size,
    )
    .map_err(|e| {
        format!(
            "cannot place {} image(s) in this prompt: {e}\n  the template renders one \
             <|image_pad|> per image; check the messages actually carry image parts",
            images.len()
        )
    })?;

    let mut embeddings = Vec::with_capacity(images.len());
    for prepared in images {
        let embedding = runner
            .encode_image(&prepared.image, params)
            .map_err(|e| format!("{}: {e}", prepared.label))?;
        embeddings.push(embedding);
    }

    runner
        .set_prompt_vision(&embeddings, &spliced.positions, spliced.ids.len())
        .map_err(|e| format!("cannot inject the encoded images: {e}"))?;
    Ok(spliced.ids)
}
