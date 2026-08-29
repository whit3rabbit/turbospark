//! `--image`: decode, preprocess, run the tower, and hand the runner the
//! injection map (ROADMAP M-V7).
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
//! Preprocessing has to come FIRST because the splice needs each image's
//! `merged_tokens`, and the template renders one marker whatever the size
//! (`docs/VISION_PHASE0.md` item 6).
//!
//! # Everything is decoded before the model opens
//!
//! A missing file, an unreadable PNG or a pixel budget the install does not
//! declare should not cost a multi-gigabyte open first -- the same rule
//! `run_messages_file` already applies to its JSON.

use turbospark_vision_io::{
    decode_image_file, preprocess, PreprocessParams, PreprocessedImage, VisionSpecialIds,
};

use super::Session;

/// One image, preprocessed and ready for the tower.
pub(crate) struct PreparedImage {
    pub(crate) path: String,
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
/// produce a correct-looking, lower-quality answer, so an install that
/// carries no `preprocessor_config.json` is REFUSED by name rather than
/// defaulted.
///
/// The geometry (`patch_size`, `merge_size`, `temporal_patch_size`,
/// `in_channels`) is cross-checked against the install's own `arch.vision`,
/// because the two files could disagree and the tower would then refuse at a
/// shape check far from the config that caused it.
pub(crate) fn preprocess_params(
    session: &Session,
    model_dir: &std::path::Path,
) -> Result<PreprocessParams, String> {
    let path = model_dir.join("preprocessor_config.json");
    let json = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "{}: {e}\n  this install declares no image preprocessing config; re-stream it with \
             its sidecars, or copy the checkpoint's preprocessor_config.json in",
            path.display()
        )
    })?;
    let params = PreprocessParams::from_preprocessor_config_json(&json)
        .map_err(|e| format!("{}: {e}", path.display()))?;

    let vision = session.runner.vision_config();
    let mismatch = |what: &str, config: usize, arch: i64| {
        format!(
            "{}: {what} is {config} but this install's manifest says {arch}; the two describe \
             different checkpoints",
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

/// Decode and preprocess every `--image`, in order.
///
/// Order-preserving: the nth prepared image pairs with the nth marker run the
/// template renders, so a reordering silently pairs each picture with the
/// wrong span.
pub(crate) fn prepare_images(
    paths: &[String],
    params: &PreprocessParams,
) -> Result<Vec<PreparedImage>, String> {
    paths
        .iter()
        .map(|path| {
            let decoded = decode_image_file(std::path::Path::new(path))
                .map_err(|e| format!("--image {path}: {e}"))?;
            let image = preprocess(&decoded, params).map_err(|e| format!("--image {path}: {e}"))?;
            Ok(PreparedImage {
                path: path.clone(),
                image,
            })
        })
        .collect()
}

/// The special ids the splice and the walk key on, read off the install.
pub(crate) fn special_ids(session: &Session) -> VisionSpecialIds {
    let vision = session.runner.vision_config();
    VisionSpecialIds {
        vision_start: vision.vision_start_token_id as i32,
        image_pad: vision.image_token_id as i32,
    }
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
/// many pages a run walks.
pub(crate) fn attach(
    session: &mut Session,
    rendered: &[i32],
    images: &[PreparedImage],
    params: &PreprocessParams,
) -> Result<Vec<i32>, String> {
    let grids: Vec<_> = images.iter().map(|p| p.image.grid).collect();
    let spliced = turbospark_vision_io::splice_and_walk(
        rendered,
        &grids,
        special_ids(session),
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
        let embedding = session
            .runner
            .encode_image(&prepared.image, params)
            .map_err(|e| format!("--image {}: {e}", prepared.path))?;
        embeddings.push(embedding);
    }

    session
        .runner
        .set_prompt_vision(&embeddings, &spliced.positions, spliced.ids.len())
        .map_err(|e| format!("cannot inject the encoded images: {e}"))?;
    Ok(spliced.ids)
}
