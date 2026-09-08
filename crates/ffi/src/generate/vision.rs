//! Vision attachment and lifecycle for FFI generation turns.

use crate::session::{Engine, Session};
use crate::wire::WirePart;

/// Named once so both the scripted refusal and its test read the same words.
pub(crate) const SCRIPTED_HAS_NO_TOWER: &str =
    "this session decodes through a scripted producer, which has no vision tower; \
     open a real install to send images";

/// Owns the engine's prompt-vision injection map for the rest of a turn, and
/// clears it on every exit -- success, an error return, or a panic unwinding
/// through -- so a caller that fails between attaching the images and
/// running the decode cannot leave the map installed for the NEXT turn on
/// this session to prefill against.
///
/// **This exists because `clear_vision` used to run only after a successful
/// decode.** `clamp_max_new` and `session.shaping` sit between
/// `attach_images` and the decode call and can return early -- a context
/// overflow on an image turn, the LIKELY failure with a page-sized image --
/// which left the map in place with nothing to clear it: the next text-only
/// turn on the session then prefilled against a picture it was never shown,
/// fluently and with no error (AGENTS.md Gotcha 13's exact shape).
pub(crate) struct VisionScope<'a> {
    engine: &'a mut Engine,
    armed: bool,
}

impl<'a> VisionScope<'a> {
    fn disarmed(engine: &'a mut Engine) -> Self {
        Self {
            engine,
            armed: false,
        }
    }

    fn armed(engine: &'a mut Engine) -> Self {
        Self {
            engine,
            armed: true,
        }
    }

    /// The engine this scope is holding the borrow on. `attach_images`
    /// returns a scope that owns the ONLY live mutable borrow of the
    /// session's engine for the rest of the turn (it was built from the same
    /// `&mut Engine` the caller passed in), so the decode dispatch has to
    /// reach the engine through here rather than re-borrowing the original
    /// reference, which the borrow checker would refuse as a second
    /// simultaneous mutable borrow.
    pub(crate) fn engine_mut(&mut self) -> &mut Engine {
        self.engine
    }
}

impl Drop for VisionScope<'_> {
    fn drop(&mut self) {
        if self.armed {
            clear_vision(self.engine);
        }
    }
}

/// Encode this turn's images and return the SPLICED prompt, together with a
/// guard that clears the injection map on drop when an image was attached.
///
/// **Everything happens under the caller's one lock**, which is the same
/// contract `crates/server/src/model.rs` states for `run_completion`: encode
/// and generate cannot be separate acquisitions, or two turns interleave and
/// each prefills the other's picture, both fluently and with no error.
///
/// With no images this is `rendered` unchanged and touches nothing, so a
/// text-only turn takes the byte-identical path it always did; the returned
/// scope is disarmed and its drop is a no-op.
#[cfg(target_os = "macos")]
pub(crate) fn attach_images<'a>(
    engine: &'a mut Engine,
    session: &Session,
    rendered: &[i32],
    parts: &[&WirePart],
) -> Result<(Vec<i32>, VisionScope<'a>), String> {
    if parts.is_empty() {
        return Ok((rendered.to_vec(), VisionScope::disarmed(engine)));
    }
    // Scoped so `runner`'s borrow of `*engine` ends here: `set_prompt_vision`
    // is `crate::vision::attach`'s LAST fallible step (`vision.rs`), so a `?`
    // anywhere in this block means the map was never installed and returning
    // the error directly -- with no scope to arm -- leaves nothing to clean
    // up. Only a path that reaches the end of the block has one to build.
    let spliced = {
        let Engine::Real(runner) = &mut *engine else {
            return Err(SCRIPTED_HAS_NO_TOWER.to_string());
        };
        if !runner.has_vision_tower() {
            return Err(format!(
                "this install declares no vision tower, so it cannot accept an image: {}",
                session.info.model_path
            ));
        }
        let params = crate::vision::preprocess_params(
            runner,
            session.load_policy.guard,
            runtime::physical_memory(),
            session.committed_bytes,
            session.kv_bytes,
        )?;
        let images = crate::vision::prepare(parts, &params)?;
        crate::vision::attach(runner.as_mut(), rendered, &images, &params)?
    };
    Ok((spliced, VisionScope::armed(engine)))
}

/// The portable half: off macOS there is no runner to encode with.
#[cfg(not(target_os = "macos"))]
pub(crate) fn attach_images<'a>(
    engine: &'a mut Engine,
    _session: &Session,
    rendered: &[i32],
    parts: &[&WirePart],
) -> Result<(Vec<i32>, VisionScope<'a>), String> {
    if parts.is_empty() {
        return Ok((rendered.to_vec(), VisionScope::disarmed(engine)));
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

/// Frees the vision tower's open resources on this session (vision memory
/// sidecar, Part C; `runtime::RealForwardRunner::release_vision_tower`).
///
/// Refused the same way [`attach_images`] refuses an image: a SCRIPTED
/// session has no tower to release, and there is no engine at all off
/// macOS. Neither refusal is a lie about capability -- a caller that wants
/// to know whether releasing did anything meaningful already has
/// `has_vision_tower`/`vision_dir` to ask first.
///
/// Idempotent on a session whose tower is already closed, or whose install
/// declares no tower at all: `release_vision_tower` is `self.vision = None`,
/// which is a no-op when it is `None` already.
#[cfg(target_os = "macos")]
pub(crate) fn release_vision(engine: &mut Engine) -> Result<(), String> {
    let Engine::Real(runner) = engine else {
        return Err(SCRIPTED_HAS_NO_TOWER.to_string());
    };
    runner.release_vision_tower();
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn release_vision(_engine: &mut Engine) -> Result<(), String> {
    Err("the engine is macOS-only, so no vision tower can run here".to_string())
}
