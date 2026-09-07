//! Vision subsystem public methods and controls for `RealForwardRunner`.

use std::path::Path;

use crate::real_forward::RealForwardRunner;
use crate::real_forward_types::RealForwardError;

impl RealForwardRunner {
    /// Whether this install declares a vision tower.
    ///
    /// Asks the ARCH rather than whether one has been opened: the tower is
    /// built lazily, so `self.vision.is_none()` means "no image yet" on an
    /// install that has one.
    pub fn has_vision_tower(&self) -> bool {
        self.arch.vision.is_active()
    }

    /// This install's vision configuration, as the manifest declares it.
    ///
    /// The special token ids and the patch geometry a front end needs to
    /// build a prompt, read off the INSTALL rather than recalled: they are
    /// per-checkpoint and a constant here would be AGENTS.md Gotcha 38's
    /// shape. Returns `VisionConfig::NONE` on an install with no tower, whose
    /// `is_active()` is false.
    pub fn vision_config(&self) -> &model_io::VisionConfig {
        &self.arch.vision
    }

    /// Bytes the vision tower's slot cache pins, `VISION_SLOTS x
    /// block_stride`. `None` until the tower has been opened.
    ///
    /// This plus [`Self::vision_last_scratch_bytes`] is the whole of the
    /// tower's residency: the blocks stream, so nothing else about the tower
    /// is held between images.
    pub fn vision_slot_bytes(&self) -> Option<u64> {
        self.vision.as_ref().map(|v| v.slot_bytes)
    }

    /// What the last page's scratch actually allocated.
    pub fn vision_last_scratch_bytes(&self) -> Option<u64> {
        self.vision.as_ref().map(|v| v.last_scratch_bytes)
    }

    /// What a page of `seq` patches WOULD cost in scratch, predicted rather
    /// than measured.
    ///
    /// The sizing a caller budgeting a page would use, and the thing
    /// [`Self::vision_last_scratch_bytes`] is worth checking against -- a
    /// formula compared only against itself asserts nothing.
    pub fn vision_scratch_bytes_for(&self, seq: usize) -> Option<u64> {
        self.vision.as_ref().map(|v| v.scratch_bytes(seq))
    }

    /// Whether the vision tower opened under `TURBOSPARK_VISION_RESIDENCY=
    /// mapped`. `None` until the tower has been opened (lazy, on the first
    /// image); `Some(false)` is the ordinary pread streamer.
    ///
    /// Test-only engagement proof: a byte-identity check between the two
    /// residency arms cannot on its own distinguish "the mapped arm ran and
    /// produced the same output" from "the mapped arm silently fell through
    /// to pread", since both would pass parity trivially in the second case.
    pub fn vision_residency_is_mapped(&self) -> Option<bool> {
        self.vision.as_ref().map(|v| v.is_mapped_residency())
    }

    /// Overrides the vision tower's MLP row tile (Part B1,
    /// `crate::vision::scratch::VISION_MLP_TILE_ROWS` by default) after it
    /// has opened, so a test can force several loop iterations on a page far
    /// smaller than any real one would need tiling for -- e.g. a 16-patch
    /// synthetic fixture at a tile of 4, which is otherwise always a single
    /// iteration under the shipped default.
    ///
    /// No-op if no image has been encoded yet: the tower opens lazily on the
    /// first [`Self::encode_image`] call, so there is nothing here to
    /// override before that. Call `encode_image` once first to open it, then
    /// this, then `encode_image` again to see the tiled arm.
    #[doc(hidden)]
    pub fn set_vision_mlp_tile_rows(&mut self, tile_rows: usize) {
        if let Some(vision) = self.vision.as_mut() {
            vision.mlp_tile_rows = tile_rows;
        }
    }

    /// Run one preprocessed image through the vision tower (ROADMAP M-V4).
    ///
    /// Returns the `[merged_tokens, out_hidden_size]` FP16 rows the trunk's
    /// residual stream wants; M-V5 is what injects them at the image-pad
    /// positions. Nothing in the decode path reads them yet, so calling this
    /// changes no generated token.
    ///
    /// Opens the tower on first use and keeps it for the runner's life --
    /// `VISION_SLOTS x block_stride` of pinned host memory, ~58 MiB on the
    /// real 27B, which a text-only session on the same install never pays.
    /// The per-page scratch is allocated and dropped inside this call.
    ///
    /// An install with no tower is refused BY NAME rather than answering an
    /// empty embedding: a caller that passed an image and silently got no
    /// rows would build a prompt whose image spans are filled with the
    /// placeholder token's own embedding, which reads as a model ignoring the
    /// picture rather than as an install that cannot see one.
    pub fn encode_image(
        &mut self,
        image: &turbospark_vision_io::PreprocessedImage,
        params: &turbospark_vision_io::PreprocessParams,
    ) -> Result<crate::vision::VisionEmbedding, RealForwardError> {
        self.open_vision_tower()?;
        // Two disjoint fields of `self`, which is what lets the tower take
        // the context mutably while it is itself borrowed mutably.
        let tower = self.vision.as_mut().expect("opened just above");
        tower.run(&mut self.context, &self.weights, image, params)
    }

    /// [`Self::encode_image`] plus the residual stream at three intermediate
    /// points, for the cross-engine parity gate.
    ///
    /// Diagnostic, and reached from nothing else. It costs three readbacks of
    /// `[patches, hidden]` and the ordinary entry point pays none of them --
    /// there is no flag that could be left on.
    pub fn encode_image_with_stages(
        &mut self,
        image: &turbospark_vision_io::PreprocessedImage,
        params: &turbospark_vision_io::PreprocessParams,
    ) -> Result<(crate::vision::VisionEmbedding, crate::vision::VisionStages), RealForwardError>
    {
        self.open_vision_tower()?;
        let tower = self.vision.as_mut().expect("opened just above");
        tower.run_with_stages(&mut self.context, &self.weights, image, params)
    }

    /// Hand this runner one prompt's image rows and mRoPE position table, so
    /// the next prefill injects them (ROADMAP M-V5).
    ///
    /// Call it AFTER [`Self::encode_image`] for every image in the prompt and
    /// BEFORE producing the prompt's first token. It is inherent rather than a
    /// trait method on purpose: `LogitProducer` is implemented by a scripted
    /// mock with no notion of an image, and widening it would put a vision
    /// concept in every producer to serve one family.
    ///
    /// Two things about the lifetime. It survives a [`Self::rollback`],
    /// because a speculative rewind stays inside the prompt the map describes.
    /// It does NOT survive `reset()`, which is what makes a bulk-OCR loop safe
    /// -- page N+1's prefill cannot inherit page N's spans.
    pub fn set_prompt_vision(
        &mut self,
        embeddings: &[crate::vision::VisionEmbedding],
        positions: &turbospark_vision_io::MropePositions,
        prompt_len: usize,
    ) -> Result<(), RealForwardError> {
        // Validated against the TRUNK's width rather than the tower's declared
        // `out_hidden_size`, because the row is about to be written into
        // `scratch.x`. A checkpoint whose two disagree is the case worth
        // catching, and reading the config for both sides would not catch it.
        let hidden = self.arch.hidden_size as usize;
        self.prompt_vision = Some(crate::vision::PromptVision::new(
            embeddings, positions, prompt_len, hidden,
        )?);
        // A placeholder span carries the same token ids whatever picture
        // filled it, so a state that consumed one is not described by its
        // ids and must never be reused (`crate::kv_prefix`'s TAINT). Without
        // this, turn two of a two-image chat answers from turn one's pixels
        // -- Gotcha 29's failure mode reached through a different door.
        self.kv_prefix.taint();
        Ok(())
    }

    /// Drop the injection map without resetting the KV cache.
    ///
    /// `reset()` already does this and is what an ordinary generation loop
    /// calls. This is for a caller that wants to continue the SAME context
    /// with the images behind it -- past the last span every position is text
    /// anyway, so the only thing still being read is `rope_position`, and
    /// dropping the map would silently move it back to the raw token index.
    /// Reach for it only when that is what you mean.
    pub fn clear_prompt_vision(&mut self) {
        self.prompt_vision = None;
    }

    /// The injection map currently set, if any.
    pub fn prompt_vision(&self) -> Option<&crate::vision::PromptVision> {
        self.prompt_vision.as_ref()
    }

    fn open_vision_tower(&mut self) -> Result<(), RealForwardError> {
        if !self.arch.vision.is_active() {
            return Err(RealForwardError::Unsupported(
                "this install declares no vision tower; repack the checkpoint with its \
                 vision_tower.* tensors to get one"
                    .to_string(),
            ));
        }
        if self.vision.is_none() {
            self.vision = Some(match &self.vision_sidecar_dir {
                Some(dir) => {
                    crate::vision::VisionTower::open_with_sidecar(dir, &self.context, &self.arch)?
                }
                None => crate::vision::VisionTower::open(
                    &self.install_dir,
                    &self.context,
                    &self.weights,
                    &self.index,
                    &self.arch,
                )?,
            });
        }
        Ok(())
    }

    /// Attach a standalone vision sidecar directory (vision memory sidecar,
    /// Part A2) to an already-open, text-only trunk, so the NEXT image
    /// processed on this session opens its tower from `dir` instead of
    /// refusing for lack of one.
    ///
    /// Call this once, before the first image -- there is no supported way
    /// to detach or replace a sidecar once attached, matching the tower's
    /// own "opens once, lazily" contract. Does NOT open the tower itself:
    /// that still happens lazily on the first [`Self::encode_image`] call,
    /// for the same reason a combined install's tower does (a text-only
    /// session on a sidecar-attached trunk still pays nothing for it until
    /// an image actually arrives).
    ///
    /// Four refusals, checked in this order:
    /// - a sidecar is already attached to this session (double-attach, or
    ///   attach after the tower has already opened from an earlier attach --
    ///   both leave [`Self::vision_dir`]'s backing field `Some`, which is
    ///   what this checks);
    /// - the trunk's OWN install already declares a vision tower (attaching
    ///   a second one would mean two towers for one session);
    /// - `dir` does not validate as a sidecar (`model_io::load_vision_sidecar`
    ///   surfaces the reason: missing record, unknown family, a manifest
    ///   that fails structural validation, or a record/manifest hidden-size
    ///   disagreement internal to the sidecar itself);
    /// - the sidecar's declared pairing (`family`, `hidden_size`) does not
    ///   match this trunk's.
    ///
    /// On success, `self.arch.vision` becomes the sidecar's `VisionConfig`,
    /// so [`Self::has_vision_tower`] and [`Self::vision_config`] read exactly
    /// as they would for a combined install from this point on.
    pub fn attach_vision_sidecar(&mut self, dir: &Path) -> Result<(), RealForwardError> {
        if let Some(existing) = &self.vision_sidecar_dir {
            return Err(RealForwardError::Unsupported(format!(
                "a vision sidecar is already attached at {} for this session; attach happens \
                 once, before the first image is encoded",
                existing.display()
            )));
        }
        if self.arch.vision.is_active() {
            return Err(RealForwardError::Unsupported(format!(
                "this session's own trunk install ({}) already declares a vision tower; \
                 attaching a sidecar at {} would be a second tower for one session, which is \
                 not supported",
                self.install_dir.display(),
                dir.display()
            )));
        }
        let (record, vision) =
            model_io::load_vision_sidecar(dir).map_err(RealForwardError::Model)?;
        let trunk_family = self.arch.family.as_str();
        if record.pairs_with.family != trunk_family {
            return Err(RealForwardError::Unsupported(format!(
                "vision sidecar at {} pairs with family {:?}, but this session's trunk is {:?}",
                dir.display(),
                record.pairs_with.family,
                trunk_family
            )));
        }
        if record.pairs_with.hidden_size != self.arch.hidden_size {
            return Err(RealForwardError::Unsupported(format!(
                "vision sidecar at {} pairs with hidden_size {}, but this session's trunk is {}",
                dir.display(),
                record.pairs_with.hidden_size,
                self.arch.hidden_size
            )));
        }
        self.vision_sidecar_dir = Some(dir.to_path_buf());
        self.arch.vision = vision;
        Ok(())
    }

    /// The directory a first image would open its tower from: the attached
    /// sidecar's directory, or this session's own install directory when
    /// none is attached (a combined install, or a text-only trunk that will
    /// refuse the image outright).
    ///
    /// The ONE place a later caller (CLI/FFI/server) should read to find
    /// `preprocessor_config.json`, so it never has to ask separately whether
    /// a sidecar is attached.
    pub fn vision_dir(&self) -> &Path {
        self.vision_sidecar_dir
            .as_deref()
            .unwrap_or(&self.install_dir)
    }

    /// Whether the tower currently open (if any) is sidecar-backed.
    ///
    /// `None` before any image is processed / no tower open, matching
    /// [`Self::vision_residency_is_mapped`]'s own precedent: a byte-identity
    /// check between a sidecar-backed run and a combined-install run cannot
    /// on its own tell "the sidecar path ran and produced this" apart from
    /// "the sidecar path silently fell through to the trunk's own tower".
    pub fn vision_is_sidecar(&self) -> Option<bool> {
        self.vision.as_ref().map(|v| v.is_sidecar())
    }

    /// Free the vision tower's open resources, without forgetting a sidecar
    /// attachment (vision memory sidecar, Part C).
    ///
    /// Once opened -- lazily, on the first image a session ever processes --
    /// a tower stays open and pinned for the runner's whole life:
    /// [`crate::vision::VISION_SLOTS`] streamer slots (or the mapped-residency
    /// arm's whole-tower mapping), a 10.6 MB host `pos_table`, and, for a
    /// sidecar, a second `ResidentGpuWeights`/mmap pair of its own. A session
    /// that will never see another image -- a GUI where the user attached one
    /// file and moved on, a server whose next hundred requests are all text --
    /// has had no way to give any of that back. This is that way.
    ///
    /// Sets [`Self::vision`] to `None`, which frees everything the tower owns
    /// through ordinary `Drop`: the slot buffers (or the mapped buffer and its
    /// mapping), the position table, and the sidecar's own weights and mmap
    /// when one is attached. Nothing else moves. [`Self::has_vision_tower`]
    /// still reads `arch.vision.is_active()`, which this does not touch, so
    /// the install's declared capability survives release exactly as it
    /// survives never having been opened. [`Self::vision_dir`]'s backing
    /// `vision_sidecar_dir` is untouched too -- releasing forgets the OPEN
    /// RESOURCES, never the ATTACHMENT -- so the next [`Self::encode_image`]
    /// reopens the tower exactly as the first one did, from the attached
    /// sidecar if there is one or from this install if not
    /// (`open_vision_tower` reads only `vision_sidecar_dir` and `arch.vision`,
    /// neither of which this method reaches).
    pub fn release_vision_tower(&mut self) {
        self.vision = None;
    }
}
