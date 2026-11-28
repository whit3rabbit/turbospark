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
///
/// Reads `session.runner.vision_dir()` rather than the session's own model
/// directory, so a trunk with an attached vision SIDECAR (vision memory
/// sidecar, Part A3) finds `preprocessor_config.json` beside the sidecar's
/// `manifest.json` rather than the trunk's own install, which for a
/// text-only trunk has no such file at all.
pub(crate) fn preprocess_params(session: &Session) -> Result<PreprocessParams, String> {
    // The Gemma 4 tower has no `preprocessor_config.json`; it goes through
    // `attach_gemma4`, which callers pick before reaching here.
    if session.runner.unlimited_ocr2_image_ids().is_some() {
        return Err("internal: Unlimited-OCR-2 takes attach_unlimited_ocr2".to_string());
    }
    if session.runner.gemma4_image_ids().is_some() {
        return Err("internal: the Gemma 4 tower takes attach_gemma4".to_string());
    }
    if session.runner.clip_image_ids().is_some() {
        return Err("internal: the CLIP tower takes attach_clip".to_string());
    }
    if session.runner.siglip_image_ids().is_some() {
        return Err("internal: the SigLIP tower takes attach_siglip".to_string());
    }
    if session.runner.inkling2_image_ids().is_some() {
        return Err("internal: the Inkling 2 tower takes attach_inkling2".to_string());
    }
    let path = session.runner.vision_dir().join("preprocessor_config.json");
    let json = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "{}: {e}\n  this install declares no image preprocessing config; re-stream it with \
             its sidecars, or copy the checkpoint's preprocessor_config.json in",
            path.display()
        )
    })?;
    let mut params = PreprocessParams::from_preprocessor_config_json(&json)
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

    // Vision memory sidecar (Part B3): clamp the checkpoint's own declared
    // `max_pixels` ceiling to what THIS session's `--load-guard` tier can
    // actually afford, on top of the KV cache and resident weights it
    // already committed at open -- neither of which `params.max_pixels`
    // alone has any way to see. The SAME tier `--max-context` resolved
    // against, never a second one (`crates/ffi/CLAUDE.md` Gotcha 12's rule).
    let declared_max_pixels = params.max_pixels;
    let budget = session
        .runner
        .resolve_vision_pixel_budget(
            declared_max_pixels,
            params.min_pixels,
            session.load_policy.guard,
            runtime::physical_memory(),
            session.committed_bytes,
            session.kv_bytes,
        )
        .map_err(|e| e.to_string())?;
    if budget.clamped {
        params.max_pixels = budget.resolved_max_pixels;
        eprintln!(
            "vision: max_pixels {declared_max_pixels} -> {} (memory budget, {})",
            budget.resolved_max_pixels,
            session.load_policy.guard.as_str(),
        );
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
    let special = special_ids(session);
    // A template that renders a bare placeholder (Muse-Glimmer's) gets the
    // start and end markers added here, as its processor does.
    let wrapped;
    let rendered = match session.runner.image_wrap_ids() {
        Some((start, end)) => {
            wrapped = turbospark_vision_io::wrap_bare_image_placeholders(
                rendered,
                special.image_pad,
                start,
                end,
            );
            &wrapped[..]
        }
        None => rendered,
    };
    let spliced =
        turbospark_vision_io::splice_and_walk(rendered, &grids, special, params.merge_size)
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

/// The MiniCPM-V tower's whole `--image` path: decode each file and hand the
/// runner the rendered prompt, which preprocesses, expands each rendered
/// `<|image_pad|>` into its marker run with canvas positions, runs the tower
/// and injects the rows. Returns the expanded ids to prefill. Thumbnail only
/// (no slicing). The runner method is the same one the FFI and the server call.
pub(crate) fn attach_minicpm(
    session: &mut Session,
    rendered: &[i32],
    paths: &[String],
) -> Result<Vec<i32>, String> {
    let mut decoded = Vec::with_capacity(paths.len());
    for path in paths {
        decoded.push(
            decode_image_file(std::path::Path::new(path))
                .map_err(|e| format!("--image {path}: {e}"))?,
        );
    }
    let tokenizer = &session.tokenizer;
    let index_ids = |k: usize| tokenizer.encode(&k.to_string(), false);
    let token_text = |id: i32| tokenizer.id_to_token(id);
    session
        .runner
        .attach_minicpm_images(rendered, &decoded, &index_ids, &token_text)
        .map_err(|e| format!("cannot attach the images: {e}"))
}

/// The Gemma 4 tower's whole `--image` path: decode and preprocess each file
/// at the default soft-token budget (the one the install's KV ring is sized
/// for), expand each rendered `<|image|>` to `boi`, one placeholder per soft
/// token, `eoi`, encode, and hand the runner the rows. Returns the expanded
/// ids to prefill. Mirrors `crates/ffi/src/vision.rs::attach_gemma4`; every
/// step it calls lives in `turbospark-vision-io` or on the runner.
pub(crate) fn attach_gemma4(
    session: &mut Session,
    rendered: &[i32],
    paths: &[String],
) -> Result<Vec<i32>, String> {
    let ids = session
        .runner
        .gemma4_image_ids()
        .ok_or("this install's vision tower is not a Gemma 4 one")?;
    let budget = turbospark_vision_io::GEMMA4_DEFAULT_SOFT_TOKENS;
    let mut prepared = Vec::with_capacity(paths.len());
    for path in paths {
        let decoded = decode_image_file(std::path::Path::new(path))
            .map_err(|e| format!("--image {path}: {e}"))?;
        let patches = turbospark_vision_io::gemma4_preprocess(&decoded, budget)
            .map_err(|e| format!("--image {path}: {e}"))?;
        prepared.push((path, patches));
    }
    let soft: Vec<usize> = prepared.iter().map(|(_, p)| p.soft_tokens()).collect();
    let (expanded, spans) = turbospark_vision_io::gemma4_expand_image_placeholders(
        rendered, ids, &soft,
    )
    .map_err(|e| {
        format!(
            "cannot place {} image(s) in this prompt: {e}\n  the template renders one \
                     <|image|> per image; check the messages actually carry image parts",
            paths.len()
        )
    })?;
    let mut rows = Vec::with_capacity(prepared.len());
    for (path, patches) in &prepared {
        rows.push(
            session
                .runner
                .encode_image_gemma4(patches)
                .map_err(|e| format!("--image {path}: {e}"))?,
        );
    }
    session
        .runner
        .set_prompt_vision_gemma4(&rows, &spans, expanded.len())
        .map_err(|e| format!("cannot inject the encoded images: {e}"))?;
    Ok(expanded)
}

/// The CLIP tower's whole `--image` path (LLaVA, `docs/LLAVA_VISION_PHASE0.md`):
/// read the processor config beside the tower, decode and preprocess each
/// file (shortest-edge resize, center crop, normalize), expand each rendered
/// `<image>` to one placeholder per patch, encode, and hand the runner the
/// rows. Returns the expanded ids to prefill.
///
/// The processor config is cross-checked against the install's own
/// `arch.vision` for the two numbers that must agree for the tower to read
/// the rows at all: the patch edge and the position-table size. An install
/// whose config describes a different crop would otherwise fail at a shape
/// check inside the tower, far from the file that caused it.
pub(crate) fn attach_clip(
    session: &mut Session,
    rendered: &[i32],
    paths: &[String],
) -> Result<Vec<i32>, String> {
    let (marker, fill) = session
        .runner
        .clip_image_ids()
        .ok_or("this install's vision tower is not a CLIP one")?;
    let path = session.runner.vision_dir().join("preprocessor_config.json");
    let json = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "{}: {e}\n  this install declares no image preprocessing config; re-stream it with \
             its sidecars, or copy the checkpoint's preprocessor_config.json in",
            path.display()
        )
    })?;
    let params = turbospark_vision_io::ClipPreprocessParams::from_preprocessor_config_json(&json)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let vision = session.runner.vision_config();
    params
        .check_against_tower(vision.patch_size, vision.num_position_embeddings)
        .map_err(|e| format!("{}: {e}", path.display()))?;

    let mut prepared = Vec::with_capacity(paths.len());
    for image_path in paths {
        let decoded = decode_image_file(std::path::Path::new(image_path))
            .map_err(|e| format!("--image {image_path}: {e}"))?;
        let patches = turbospark_vision_io::clip_preprocess(&decoded, &params)
            .map_err(|e| format!("--image {image_path}: {e}"))?;
        prepared.push((image_path, patches));
    }
    let grids: Vec<(usize, usize)> = prepared.iter().map(|(_, p)| (p.grid, p.grid_w)).collect();
    // The llava-hf template emits no `<s>`; the reference's processor adds it.
    let rendered = turbospark_vision_io::clip_prepend_bos(
        rendered,
        session.tokenizer.encode("", true).first().copied(),
    );
    let (expanded, spans) = turbospark_vision_io::flat_expand_image_placeholders(
        &rendered,
        marker,
        fill,
        session.runner.pixtral_break_end_ids(),
        &grids,
    )
    .map_err(|e| {
        format!(
            "cannot place {} image(s) in this prompt: {e}\n  the template renders one \
                     <image> per image; check the messages actually carry image parts",
            paths.len()
        )
    })?;
    let mut rows = Vec::with_capacity(prepared.len());
    for (image_path, patches) in &prepared {
        rows.push(
            session
                .runner
                .encode_image_clip(patches)
                .map_err(|e| format!("--image {image_path}: {e}"))?,
        );
    }
    session
        .runner
        .set_prompt_vision_flat(&rows, &spans, expanded.len())
        .map_err(|e| format!("cannot inject the encoded images: {e}"))?;
    Ok(expanded)
}

/// The Inkling 2 tower's whole `--image` path
/// (`docs/INKLING2_VISION_PHASE0.md`): decode each file the way the reference
/// processor sees it (transparency composited onto its checkerboard), then the
/// runner's shared preprocess, expand, encode and inject. Returns the expanded
/// ids to prefill.
pub(crate) fn attach_inkling2(
    session: &mut Session,
    rendered: &[i32],
    paths: &[String],
) -> Result<Vec<i32>, String> {
    let mut images = Vec::with_capacity(paths.len());
    for image_path in paths {
        let bytes = std::fs::read(image_path).map_err(|e| format!("--image {image_path}: {e}"))?;
        images.push(
            turbospark_vision_io::inkling2_decode_image_bytes(&bytes)
                .map_err(|e| format!("--image {image_path}: {e}"))?,
        );
    }
    session
        .runner
        .attach_inkling2_images(rendered, &images, turbospark_vision_io::INKLING2_MAX_TOKENS)
        .map_err(|e| e.to_string())
}

/// The SigLIP (DeepSeek-VL2) tower's whole `--image` path
/// (`docs/DEEPSEEK_VL2_VISION_PHASE0.md`): read `processor_config.json` beside
/// the tower, pad and tile each file (a global view plus the best-fitting local
/// grid), expand each rendered `<image>` to the full run (projector rows with
/// the newline and separator rows between), encode, and hand the runner the
/// rows. Returns the expanded ids to prefill.
///
/// Tiling is on for at most two images and off for more, as in the reference.
pub(crate) fn attach_siglip(
    session: &mut Session,
    rendered: &[i32],
    paths: &[String],
) -> Result<Vec<i32>, String> {
    let (marker, fill) = session
        .runner
        .siglip_image_ids()
        .ok_or("this install's vision tower is not a SigLIP one")?;
    let path = session.runner.vision_dir().join("processor_config.json");
    let json = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "{}: {e}\n  this install declares no image processor config; re-pull the vision \
             sidecar, or copy the checkpoint's processor_config.json in",
            path.display()
        )
    })?;
    let params = turbospark_vision_io::DeepseekVl2Params::from_processor_config_json(&json)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let vision = session.runner.vision_config();
    let side = params.patches_per_side() as i64;
    if vision.patch_size != turbospark_vision_io::DSVL2_PATCH as i64
        || vision.num_position_embeddings != side * side
        || vision.spatial_merge_size != params.downsample_ratio as i64
    {
        return Err(format!(
            "{}: a {}-pixel tile is {side}x{side} patches with a {}-wide unfold, but this \
             install's tower has patch {}, a {}-row position table and unfold {}; the two \
             describe different checkpoints",
            path.display(),
            params.image_size,
            params.downsample_ratio,
            vision.patch_size,
            vision.num_position_embeddings,
            vision.spatial_merge_size
        ));
    }

    let crop = paths.len() <= 2;
    let mut prepared = Vec::with_capacity(paths.len());
    for image_path in paths {
        let decoded = decode_image_file(std::path::Path::new(image_path))
            .map_err(|e| format!("--image {image_path}: {e}"))?;
        let tiles = turbospark_vision_io::deepseek_vl2_preprocess(&decoded, &params, crop)
            .map_err(|e| format!("--image {image_path}: {e}"))?;
        prepared.push((image_path, tiles));
    }
    let counts: Vec<usize> = prepared.iter().map(|(_, t)| t.image_tokens()).collect();
    let (expanded, spans) =
        turbospark_vision_io::clip_expand_image_placeholders(rendered, marker, fill, &counts)
            .map_err(|e| {
                format!(
                    "cannot place {} image(s) in this prompt: {e}\n  the template renders one \
                     <image> per image; check the messages actually carry image parts",
                    paths.len()
                )
            })?;
    let mut rows = Vec::with_capacity(prepared.len());
    for (image_path, tiles) in &prepared {
        rows.push(
            session
                .runner
                .encode_image_siglip(tiles)
                .map_err(|e| format!("--image {image_path}: {e}"))?,
        );
    }
    session
        .runner
        .set_prompt_vision_flat(&rows, &spans, expanded.len())
        .map_err(|e| format!("cannot inject the encoded images: {e}"))?;
    Ok(expanded)
}

/// The DeepSeek-OCR arm of the image front end
/// (`docs/DEEPSEEK_OCR_VISION_PHASE0.md`): read `processor_config.json` beside
/// the tower, pad each image to the 1024 px base view, encode it (the SAM and
/// CLIP-L tower and the linear projector), and expand every `<image>` marker
/// in the rendered prompt to the 273-position run the encoded rows replace.
pub(crate) fn attach_deepseek_ocr(
    session: &mut Session,
    rendered: &[i32],
    paths: &[String],
) -> Result<Vec<i32>, String> {
    let (marker, fill) = session
        .runner
        .deepseek_ocr_image_ids()
        .ok_or("this install's vision tower is not a DeepSeek-OCR one")?;
    let path = session.runner.vision_dir().join("processor_config.json");
    let json = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "{}: {e}\n  this install declares no image processor config; re-pull the vision \
             sidecar, or copy the checkpoint's processor_config.json in",
            path.display()
        )
    })?;
    let params = turbospark_vision_io::DeepseekOcrParams::from_processor_config_json(&json)
        .map_err(|e| format!("{}: {e}", path.display()))?;

    let mut views = Vec::with_capacity(paths.len());
    for image_path in paths {
        let decoded = decode_image_file(std::path::Path::new(image_path))
            .map_err(|e| format!("--image {image_path}: {e}"))?;
        let view = turbospark_vision_io::deepseek_ocr_preprocess(&decoded, &params)
            .map_err(|e| format!("--image {image_path}: {e}"))?;
        views.push((image_path, view));
    }
    let counts = vec![turbospark_vision_io::OCR_IMAGE_TOKENS; views.len()];
    let (expanded, spans) =
        turbospark_vision_io::clip_expand_image_placeholders(rendered, marker, fill, &counts)
            .map_err(|e| {
                format!(
                    "cannot place {} image(s) in this prompt: {e}\n  the template renders one \
                     <image> per image; check the messages actually carry image parts",
                    paths.len()
                )
            })?;
    let mut rows = Vec::with_capacity(views.len());
    for (image_path, view) in &views {
        rows.push(
            session
                .runner
                .encode_image_deepseek_ocr(view)
                .map_err(|e| format!("--image {image_path}: {e}"))?,
        );
    }
    session
        .runner
        .set_prompt_vision_flat(&rows, &spans, expanded.len())
        .map_err(|e| format!("cannot inject the encoded images: {e}"))?;
    Ok(expanded)
}

/// The Unlimited-OCR-2 tower's attach: pages, expansion, tower, prompt.
pub(crate) fn attach_unlimited_ocr2(
    session: &mut Session,
    rendered: &[i32],
    paths: &[String],
) -> Result<Vec<i32>, String> {
    let ids = session
        .runner
        .unlimited_ocr2_image_ids()
        .ok_or("this install is not unlimited_ocr2")?;
    let mut images = Vec::with_capacity(paths.len());
    for path in paths {
        images.push(
            decode_image_file(std::path::Path::new(path))
                .map_err(|e| format!("--image {path}: {e}"))?,
        );
    }
    let pages = turbospark_vision_io::uocr2_pages(&images);
    let expanded = turbospark_vision_io::uocr2_expand_prompt(rendered, &pages, ids)
        .map_err(|e| format!("cannot place {} image(s) in this prompt: {e}", paths.len()))?;
    session
        .runner
        .attach_unlimited_ocr2_pages(&pages, &expanded)
        .map_err(|e| format!("--image: {e}"))?;
    Ok(expanded)
}

/// The Idefics3 / SmolVLM `--image` path
/// (`docs/IDEFICS3_VISION_PHASE0.md`): resolve the two special tokens, decode
/// the files and let the runner tile, expand, encode and inject.
pub(crate) fn attach_idefics3(
    session: &mut Session,
    rendered: &[i32],
    paths: &[String],
) -> Result<Vec<i32>, String> {
    let image_id = session
        .runner
        .idefics3_image_ids()
        .ok_or("this install's vision tower is not an Idefics3 one")?;
    if session.tokenizer.id_to_token(image_id).as_deref() != Some("<image>") {
        return Err(format!(
            "token id {image_id} is {:?} in this tokenizer but the manifest says it is <image>",
            session.tokenizer.id_to_token(image_id)
        ));
    }
    let fake_id = session
        .tokenizer
        .token_to_id("<fake_token_around_image>")
        .ok_or("this tokenizer has no <fake_token_around_image> token")?;
    let mut images = Vec::with_capacity(paths.len());
    for path in paths {
        let decoded = decode_image_file(std::path::Path::new(path))
            .map_err(|e| format!("--image {path}: {e}"))?;
        images.push((format!("--image {path}"), decoded));
    }
    let tokenizer = &session.tokenizer;
    session
        .runner
        .attach_idefics3(rendered, fake_id, &images, |text| {
            Ok::<_, String>(tokenizer.encode(text, false))
        })
        .map_err(|e| format!("--image: {e}"))
}

/// Flatten an InternViT conversation's image parts into the `<img>\n` marker
/// string the rendered prompt must carry.
///
/// The published `internvl_chat` checkpoints ship a TEXT-ONLY template (the
/// plain Qwen2 one, whose `'<|im_start|>' + message.content` cannot render a
/// content-part list), and the reference formats its question through the
/// conversation module rather than the template. Scoped to the InternViT
/// tower, which means BOTH wrappers: an `internvl` install is flattened too,
/// although its template may handle parts itself (no install of that flavor
/// has run). Every other tower stays on its own path, and a text-only
/// conversation is returned unchanged.
pub(crate) fn flatten_internvl_parts(
    session: &Session,
    messages: &[tokenizer::Message],
) -> Vec<tokenizer::Message> {
    if session.runner.internvl_image_ids().is_none()
        || messages.iter().all(|m| m.image_count() == 0)
    {
        return messages.to_vec();
    }
    messages
        .iter()
        .map(|m| {
            if m.image_count() == 0 {
                return m.clone();
            }
            let mut text = String::new();
            for part in &m.content_parts {
                match part {
                    tokenizer::ContentPart::Image => text.push_str("<img>\n"),
                    tokenizer::ContentPart::Text(t) => text.push_str(t),
                }
            }
            let mut flat = m.clone();
            flat.content = Some(text);
            flat.content_parts = Vec::new();
            flat
        })
        .collect()
}

/// The InternViT tower's whole `--image` path (the InternVL wrappers,
/// `docs/INTERNVL_VISION_PHASE0.md`): read the processor settings beside the
/// tower, decode and preprocess each file into its dynamic tiles (plus
/// thumbnail), expand each rendered image marker into
/// `[<img>] [<IMG_CONTEXT>] * tokens [</img>]`, run the tower, and hand the
/// runner the rows. Returns the expanded ids to prefill.
///
/// The span ids are TOKENIZER-resolved, not manifest fields: `<IMG_CONTEXT>`
/// is checked against the manifest's `image_token_id` (a mismatch means the
/// install and the tokenizer disagree about the vocabulary, the same refusal
/// `attach_minicpm` makes), and `<img>` / `</img>` are looked up by text.
/// Which marker the template rendered decides the expansion's trigger:
/// the `internvl` wrapper's template emits bare `<IMG_CONTEXT>`, the
/// `internvl_chat` one emits `<img>`, and a prompt carrying neither is
/// refused rather than expanded blindly.
pub(crate) fn attach_internvl(
    session: &mut Session,
    rendered: &[i32],
    paths: &[String],
) -> Result<Vec<i32>, String> {
    const IMG_CONTEXT: &str = "<IMG_CONTEXT>";
    const IMG_START: &str = "<img>";
    const IMG_END: &str = "</img>";
    let context_id = session
        .runner
        .internvl_image_ids()
        .ok_or("this install's vision tower is not an InternViT one")?;
    if session.tokenizer.id_to_token(context_id).as_deref() != Some(IMG_CONTEXT) {
        return Err(format!(
            "token id {context_id} is {:?} in this tokenizer but the InternViT prompt layout \
             needs {IMG_CONTEXT:?}",
            session.tokenizer.id_to_token(context_id)
        ));
    }
    let img_start = session
        .tokenizer
        .token_to_id(IMG_START)
        .ok_or("this tokenizer has no <img> token, which the InternViT span needs")?;
    let img_end = session
        .tokenizer
        .token_to_id(IMG_END)
        .ok_or("this tokenizer has no </img> token, which the InternViT span needs")?;

    let path = session.runner.vision_dir().join("preprocessor_config.json");
    let params = match std::fs::read_to_string(&path) {
        Ok(json) => turbospark_vision_io::InternVlParams::from_preprocessor_config_json(&json)
            .map_err(|e| format!("{}: {e}", path.display()))?,
        Err(_) => {
            eprintln!(
                "internvl: {} is unreadable; using the reference processor's defaults",
                path.display()
            );
            turbospark_vision_io::InternVlParams::default()
        }
    };
    let vision = session.runner.vision_config();
    // The tile edge and the tower's position table must describe the same
    // geometry: (tile / patch)^2 = position rows - 1, merge^2 patches per
    // token. A disagreement is a config/preprocessor mismatch, refused here
    // rather than at a shape check inside the tower.
    let side = ((vision.num_position_embeddings - 1) as f64).sqrt().round() as i64;
    let want_tile = side * vision.patch_size;
    if vision.patch_size != 14 || params.tile as i64 != want_tile {
        return Err(format!(
            "{}: a {}-pixel tile against patch {} and a {}-row position table wants tile {}; \
             the two describe different checkpoints",
            path.display(),
            params.tile,
            vision.patch_size,
            vision.num_position_embeddings,
            want_tile
        ));
    }

    let mut prepared = Vec::with_capacity(paths.len());
    for image_path in paths {
        let decoded = decode_image_file(std::path::Path::new(image_path))
            .map_err(|e| format!("--image {image_path}: {e}"))?;
        let tiles = turbospark_vision_io::internvl_preprocess(
            &decoded,
            vision.patch_size as usize,
            &params,
        )
        .map_err(|e| format!("--image {image_path}: {e}"))?;
        prepared.push((image_path, tiles));
    }
    // Tokens per image: every tile contributes (side / merge)^2 projector
    // rows, and the whole run sits in ONE span.
    let side_usize = side as usize;
    let merge = vision.spatial_merge_size as usize;
    let per_tile = (side_usize / merge) * (side_usize / merge);
    let counts: Vec<usize> = prepared.iter().map(|(_, t)| t.tiles * per_tile).collect();
    // The marker is whichever id the template rendered: the context id
    // itself, or the <img> start (the internvl_chat template's form).
    let marker = if rendered.contains(&context_id) {
        context_id
    } else if rendered.contains(&img_start) {
        img_start
    } else {
        return Err(format!(
            "cannot place {} image(s) in this prompt: no image marker the InternViT span \
             expands ({IMG_CONTEXT:?} id {context_id} or {IMG_START:?} id {img_start}); check \
             the messages actually carry image parts",
            paths.len()
        ));
    };
    let (expanded, spans) = turbospark_vision_io::internvl_expand_image_placeholders(
        rendered, marker, img_start, img_end, context_id, &counts,
    )
    .map_err(|e| format!("cannot place {} image(s) in this prompt: {e}", paths.len()))?;
    let mut rows = Vec::with_capacity(prepared.len());
    for (image_path, tiles) in &prepared {
        let encoded = session
            .runner
            .encode_image_internvl(tiles)
            .map_err(|e| format!("--image {image_path}: {e}"))?;
        if encoded.len()
            != counts[rows.len()] * session.runner.vision_config().out_hidden_size as usize
        {
            return Err(format!(
                "--image {image_path}: the tower produced {} row values against the {} tokens \
                 the expansion reserved",
                encoded.len(),
                counts[rows.len()]
            ));
        }
        rows.push(encoded);
    }
    session
        .runner
        .set_prompt_vision_flat(&rows, &spans, expanded.len())
        .map_err(|e| format!("cannot inject the encoded images: {e}"))?;
    Ok(expanded)
}
