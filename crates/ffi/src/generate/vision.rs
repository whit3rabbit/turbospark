//! Vision attachment and lifecycle for FFI generation turns.

use crate::session::{Engine, Session};
use crate::wire::WirePart;

/// Named once so both the scripted refusal and its test read the same words.
pub(crate) const SCRIPTED_HAS_NO_TOWER: &str =
    "this session decodes through a scripted producer, which has no vision tower; \
     open a real install to send images";

/// Encode this turn's images and return the SPLICED prompt.
///
/// **Everything happens under the caller's one lock**, which is the same
/// contract `crates/server/src/model.rs` states for `run_completion`: encode
/// and generate cannot be separate acquisitions, or two turns interleave and
/// each prefills the other's picture, both fluently and with no error.
///
/// With no images this is `rendered` unchanged and touches nothing, so a
/// text-only turn takes the byte-identical path it always did.
#[cfg(target_os = "macos")]
pub(crate) fn attach_images(
    engine: &mut Engine,
    session: &Session,
    rendered: &[i32],
    parts: &[&WirePart],
) -> Result<Vec<i32>, String> {
    if parts.is_empty() {
        return Ok(rendered.to_vec());
    }
    let Engine::Real(runner) = engine else {
        return Err(SCRIPTED_HAS_NO_TOWER.to_string());
    };
    if !runner.has_vision_tower() {
        return Err(format!(
            "this install declares no vision tower, so it cannot accept an image: {}",
            session.info.model_path
        ));
    }
    let params = crate::vision::preprocess_params(runner)?;
    let images = crate::vision::prepare(parts, &params)?;
    crate::vision::attach(runner.as_mut(), rendered, &images, &params)
}

/// The portable half: off macOS there is no runner to encode with.
#[cfg(not(target_os = "macos"))]
pub(crate) fn attach_images(
    _engine: &mut Engine,
    _session: &Session,
    rendered: &[i32],
    parts: &[&WirePart],
) -> Result<Vec<i32>, String> {
    if parts.is_empty() {
        return Ok(rendered.to_vec());
    }
    Err("the engine is macOS-only, so no vision tower can run here".to_string())
}

/// Drops this turn's injection map. Called on BOTH exits from generation.
#[cfg(target_os = "macos")]
pub(crate) fn clear_vision(engine: &mut Engine) {
    if let Engine::Real(runner) = engine {
        runner.clear_prompt_vision();
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn clear_vision(_engine: &mut Engine) {}
