//! Vision subsystem public methods and controls for `RealForwardRunner`.

use std::path::Path;

use crate::real_forward::RealForwardRunner;
use crate::real_forward_types::RealForwardError;

/// The FP16 residual-stream rows of one LFM3-VL tower run, as raw bits:
/// `[patches, hidden]` after the patch embedding and after each block.
#[derive(Debug, Default)]
pub struct Lfm3VlStageRows {
    pub patch_embed: Vec<u16>,
    pub blocks: Vec<Vec<u16>>,
}

impl RealForwardRunner {
    /// Whether this install declares a vision tower.
    ///
    /// Asks the ARCH rather than whether one has been opened: the tower is
    /// built lazily, so `self.vision.is_none()` means "no image yet" on an
    /// install that has one. Which tower it is, and so which encode path a
    /// front end takes, is `vision_config().tower`: the qwen tower through
    /// [`Self::encode_image`], the Gemma 4 one through
    /// [`Self::encode_image_gemma4`].
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
        self.vision
            .as_ref()
            .map(|v| v.slot_bytes)
            .or_else(|| self.lfm3_vision.as_ref().map(|v| v.slot_bytes))
    }

    /// What the last page's scratch actually allocated.
    pub fn vision_last_scratch_bytes(&self) -> Option<u64> {
        self.vision
            .as_ref()
            .map(|v| v.last_scratch_bytes)
            .or_else(|| self.qwen25_vision.as_ref().map(|v| v.last_scratch_bytes))
            .or_else(|| self.muse_vision.as_ref().map(|v| v.last_scratch_bytes))
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
    /// A zero override is normalized to one row so scratch allocation and
    /// block dispatch continue to use the same nonempty tile.
    #[doc(hidden)]
    pub fn set_vision_mlp_tile_rows(&mut self, tile_rows: usize) {
        if let Some(vision) = self.vision.as_mut() {
            vision.mlp_tile_rows = tile_rows.max(1);
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
        if self.arch.vision.tower == model_io::VisionTowerKind::Qwen25Vl {
            return self
                .encode_image_qwen25(image, params, None)
                .map(|(e, _)| e);
        }
        if self.arch.vision.tower == model_io::VisionTowerKind::Lfm3Vl {
            return self.encode_image_lfm3(image, params, None);
        }
        if self.arch.vision.tower == model_io::VisionTowerKind::MuseGlimmer {
            return self.encode_image_muse(image, params, None).map(|(e, _)| e);
        }
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
        if self.arch.vision.tower == model_io::VisionTowerKind::Qwen25Vl {
            let (embedding, stages) = self.encode_image_qwen25(
                image,
                params,
                Some(crate::vision::VisionStages::default()),
            )?;
            return Ok((embedding, stages.expect("requested just above")));
        }
        if self.arch.vision.tower == model_io::VisionTowerKind::Lfm3Vl {
            // The LFM3-VL tower reports the patch embedding and every block's
            // output; the shared `VisionStages` carries the first and last.
            let mut stages = crate::vision::lfm3_vl::Lfm3VlStages::default();
            let embedding = self.encode_image_lfm3(image, params, Some(&mut stages))?;
            return Ok((
                embedding,
                crate::vision::lfm3_vl::into_vision_stages(stages),
            ));
        }
        if self.arch.vision.tower == model_io::VisionTowerKind::MuseGlimmer {
            let (embedding, stages) = self.encode_image_muse(
                image,
                params,
                Some(crate::vision::VisionStages::default()),
            )?;
            return Ok((embedding, stages.expect("requested just above")));
        }
        self.open_vision_tower()?;
        let tower = self.vision.as_mut().expect("opened just above");
        tower.run_with_stages(&mut self.context, &self.weights, image, params)
    }

    /// An `lfm3_vl` tower's embedding plus the FP16 residual stream after the
    /// patch embedding and after EVERY block, for the cross-engine probe
    /// (`scripts/lfm3_vl_tower_probe.py`). Diagnostic: the ordinary entry
    /// point pays none of the readbacks.
    ///
    /// Combined installs only: the sidecar format has no LFM3-VL pairing yet,
    /// so a sidecar-attached trunk is refused by name rather than opened
    /// against the wrong weights.
    #[doc(hidden)]
    pub fn encode_image_lfm3_stages(
        &mut self,
        image: &turbospark_vision_io::PreprocessedImage,
        params: &turbospark_vision_io::PreprocessParams,
    ) -> Result<(crate::vision::VisionEmbedding, Lfm3VlStageRows), RealForwardError> {
        let mut stages = crate::vision::lfm3_vl::Lfm3VlStages::default();
        let embedding = self.encode_image_lfm3(image, params, Some(&mut stages))?;
        Ok((
            embedding,
            Lfm3VlStageRows {
                patch_embed: stages.patch_embed,
                blocks: stages.blocks,
            },
        ))
    }

    fn encode_image_lfm3(
        &mut self,
        image: &turbospark_vision_io::PreprocessedImage,
        params: &turbospark_vision_io::PreprocessParams,
        stages: Option<&mut crate::vision::lfm3_vl::Lfm3VlStages>,
    ) -> Result<crate::vision::VisionEmbedding, RealForwardError> {
        if !self.arch.vision.is_active() {
            return Err(RealForwardError::Unsupported(
                "this install declares no vision tower; repack the checkpoint with its \
                 vision_tower.* tensors to get one"
                    .to_string(),
            ));
        }
        if self.vision_sidecar_dir.is_some() {
            return Err(RealForwardError::Unsupported(
                "an lfm3_vl vision tower cannot run from a sidecar yet".to_string(),
            ));
        }
        if self.lfm3_vision.is_none() {
            self.lfm3_vision = Some(crate::vision::lfm3_vl::Lfm3VlVisionTower::open(
                &self.install_dir,
                &self.context,
                &self.index,
                &self.arch,
            )?);
        }
        let tower = self.lfm3_vision.as_mut().expect("opened just above");
        tower.run(&mut self.context, &self.weights, image, params, stages)
    }

    /// The Muse-Glimmer arm of [`Self::encode_image`]: the same
    /// [`crate::vision::VisionEmbedding`] as the other towers, so the prompt
    /// builders and [`Self::set_prompt_vision`] are shared.
    fn encode_image_muse(
        &mut self,
        image: &turbospark_vision_io::PreprocessedImage,
        params: &turbospark_vision_io::PreprocessParams,
        capture: Option<crate::vision::VisionStages>,
    ) -> Result<
        (
            crate::vision::VisionEmbedding,
            Option<crate::vision::VisionStages>,
        ),
        RealForwardError,
    > {
        if !self.arch.vision.is_active() {
            return Err(RealForwardError::Unsupported(
                "this install declares no vision tower; repack the checkpoint with its \
                 vision tensors to get one"
                    .to_string(),
            ));
        }
        if self.muse_vision.is_none() {
            self.muse_vision = Some(match &self.vision_sidecar_dir {
                Some(dir) => crate::vision::muse::MuseVisionTower::open_with_sidecar(
                    dir,
                    &self.context,
                    &self.arch,
                )?,
                None => crate::vision::muse::MuseVisionTower::open(
                    &self.install_dir,
                    &self.context,
                    &self.weights,
                    &self.index,
                    &self.arch,
                )?,
            });
        }
        let tower = self.muse_vision.as_mut().expect("opened just above");
        tower.run(&mut self.context, &self.weights, image, params, capture)
    }

    /// The Qwen2.5-VL arm of [`Self::encode_image`]: the same
    /// [`crate::vision::VisionEmbedding`], so the prompt builders and
    /// [`Self::set_prompt_vision`] are shared with the qwen3 tower.
    fn encode_image_qwen25(
        &mut self,
        image: &turbospark_vision_io::PreprocessedImage,
        params: &turbospark_vision_io::PreprocessParams,
        capture: Option<crate::vision::VisionStages>,
    ) -> Result<
        (
            crate::vision::VisionEmbedding,
            Option<crate::vision::VisionStages>,
        ),
        RealForwardError,
    > {
        if !self.arch.vision.is_active() {
            return Err(RealForwardError::Unsupported(
                "this install declares no vision tower; repack the checkpoint with its \
                 vision_tower.* tensors to get one"
                    .to_string(),
            ));
        }
        if self.qwen25_vision.is_none() {
            self.qwen25_vision = Some(match &self.vision_sidecar_dir {
                Some(dir) => crate::vision::qwen25::Qwen25VisionTower::open_with_sidecar(
                    dir,
                    &self.context,
                    &self.arch,
                )?,
                None => crate::vision::qwen25::Qwen25VisionTower::open(
                    &self.install_dir,
                    &self.context,
                    &self.index,
                    &self.arch,
                )?,
            });
        }
        let tower = self.qwen25_vision.as_mut().expect("opened just above");
        tower.run(&mut self.context, &self.weights, image, params, capture)
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
        // The tower's row sets must match what this arch DECLARES: a tower
        // with deepstack returning none (or the reverse) would make the
        // trunk's per-layer adds silently skip or run short -- competent
        // output from a trunk that never saw the deep features. The count is
        // the config's, so the two sides can only disagree on an install or
        // a tower that is wrong in a way worth naming.
        let declared = self.arch.vision.deepstack_visual_indexes.len();
        let produced = embeddings.first().map(|e| e.deepstack.len()).unwrap_or(0);
        if produced != declared {
            return Err(RealForwardError::Unsupported(format!(
                "the tower produced {} deepstack row set(s) against the {} this install's \
                 config declares",
                produced, declared
            )));
        }
        let mut prompt_vision =
            crate::vision::PromptVision::new(embeddings, positions, prompt_len, hidden)?;
        // The deepstack rows' GPU twins, uploaded once per prompt rather
        // than re-written per micro-batch. A no-op for a tower without
        // deepstack.
        prompt_vision.attach_deepstack_buffers(&mut self.context)?;
        self.prompt_vision = Some(prompt_vision);
        // Refused here as well as at every Qwen4Exp pass, so the caller hears
        // it before prefill rather than at the first token. The map is dropped
        // so a refused prompt leaves no injection behind.
        if let Err(e) = self.refuse_qwen4_vision_past_qsa_budget() {
            self.prompt_vision = None;
            return Err(e);
        }
        // A placeholder span carries the same token ids whatever picture
        // filled it, so a state that consumed one is not described by its
        // ids and must never be reused (`crate::kv_prefix`'s TAINT). Without
        // this, turn two of a two-image chat answers from turn one's pixels
        // -- Gotcha 29's failure mode reached through a different door.
        self.kv_prefix.taint();
        Ok(())
    }

    /// Hand this runner one prompt's Gemma 4 image rows: one
    /// `[span.len, text_hidden]` FP16 row set per placeholder span, in
    /// prompt order, from [`Self::encode_image_gemma4`]. The next
    /// `prefill_chunk` writes them in place of the placeholders' embeddings
    /// and runs each span with Gemma 4's bidirectional sliding mask.
    ///
    /// Positions are the plain token index (Gemma 4 has no mRoPE). The same
    /// lifetime and prefix-taint rules as [`Self::set_prompt_vision`]. Every
    /// refusal a span would hit at prefill time is checked here too, so a
    /// caller learns before encoding the prompt.
    pub fn set_prompt_vision_gemma4(
        &mut self,
        images: &[Vec<u16>],
        spans: &[turbospark_vision_io::ImageSpan],
        prompt_len: usize,
    ) -> Result<(), RealForwardError> {
        if self.arch.vision.tower != model_io::VisionTowerKind::Gemma4
            || self.arch.family != model_io::ModelFamily::Gemma4
            || self.real.is_none()
        {
            return Err(RealForwardError::Unsupported(
                "set_prompt_vision_gemma4 needs a Gemma 4 trunk with a Gemma 4 vision tower"
                    .to_string(),
            ));
        }
        if let Some(longest) = spans.iter().map(|s| s.len).max() {
            if let Some(reason) = self.gemma4_span_refusal(longest) {
                return Err(RealForwardError::Unsupported(reason));
            }
        }
        let hidden = self.arch.hidden_size as usize;
        self.prompt_vision = Some(crate::vision::PromptVision::flat(
            images, spans, prompt_len, hidden,
        )?);
        self.kv_prefix.taint();
        Ok(())
    }

    /// [`Self::set_prompt_vision_gemma4`] without the tower-kind check, for
    /// trunk tests on a synthetic Gemma 4 install that has no tower to encode
    /// with. The placeholder id is the arch's `image_token_id` (0 on an
    /// install that declares no tower). Every span refusal still applies.
    #[doc(hidden)]
    pub fn set_prompt_vision_gemma4_rows_for_tests(
        &mut self,
        images: &[Vec<u16>],
        spans: &[turbospark_vision_io::ImageSpan],
        prompt_len: usize,
    ) -> Result<(), RealForwardError> {
        if self.arch.family != model_io::ModelFamily::Gemma4 || self.real.is_none() {
            return Err(RealForwardError::Unsupported(
                "set_prompt_vision_gemma4_rows_for_tests needs a Gemma 4 trunk".to_string(),
            ));
        }
        if let Some(longest) = spans.iter().map(|s| s.len).max() {
            if let Some(reason) = self.gemma4_span_refusal(longest) {
                return Err(RealForwardError::Unsupported(reason));
            }
        }
        let hidden = self.arch.hidden_size as usize;
        self.prompt_vision = Some(crate::vision::PromptVision::flat(
            images, spans, prompt_len, hidden,
        )?);
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
        qwen_tower_only(&self.arch.vision)?;
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
        self.vision
            .as_ref()
            .map(|v| v.is_sidecar())
            .or_else(|| self.qwen25_vision.as_ref().map(|v| v.is_sidecar()))
            .or_else(|| self.muse_vision.as_ref().map(|v| v.is_sidecar()))
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
        self.gemma4_vision = None;
        self.qwen25_vision = None;
        self.lfm3_vision = None;
        self.clip_vision = None;
        self.muse_vision = None;
    }

    /// `(marker, fill)` for a LLaVA prompt, read off the install, or `None`
    /// when this install's tower is not a CLIP one.
    ///
    /// `marker` is the id the template's bare `<image>` tokenizes to (LLaVA
    /// has no start or end token around an image). `fill` is what the
    /// expanded run carries: the marker itself when the trunk's table has a
    /// row for it, else id 0. The GGUF trunks carry 32,000 rows against the
    /// marker's 32,000, and the trunk range-checks every id before the blit
    /// that replaces its lookup (`docs/LLAVA_VISION_PHASE0.md`).
    pub fn clip_image_ids(&self) -> Option<(i32, i32)> {
        let v = &self.arch.vision;
        if !(v.is_active() && v.tower == model_io::VisionTowerKind::Clip) {
            return None;
        }
        let marker = v.image_token_id as i32;
        let fill = if i64::from(marker) < self.arch.vocab_size {
            marker
        } else {
            0
        };
        Some((marker, fill))
    }

    /// Encode one image through this install's CLIP tower: `[patches,
    /// text_hidden]` FP16 bits, the rows that replace the placeholder run.
    ///
    /// The tower opens lazily on the first image, from the attached sidecar
    /// when there is one and from the install otherwise. Inject the rows with
    /// [`Self::set_prompt_vision_flat`].
    pub fn encode_image_clip(
        &mut self,
        patches: &turbospark_vision_io::ClipPatches,
    ) -> Result<Vec<u16>, RealForwardError> {
        self.open_clip_tower()?;
        let tower = self.clip_vision.as_mut().expect("opened just above");
        tower
            .run(&mut self.context, &self.weights, patches, false)
            .map(|(rows, _)| rows)
    }

    /// [`Self::encode_image_clip`] plus the residual stream at three points,
    /// for the cross-engine parity gate. Diagnostic: the ordinary entry point
    /// pays none of the three readbacks.
    pub fn encode_image_clip_with_stages(
        &mut self,
        patches: &turbospark_vision_io::ClipPatches,
    ) -> Result<(Vec<u16>, crate::vision::ClipStages), RealForwardError> {
        self.open_clip_tower()?;
        let tower = self.clip_vision.as_mut().expect("opened just above");
        let (rows, stages) = tower.run(&mut self.context, &self.weights, patches, true)?;
        Ok((rows, stages.expect("requested just above")))
    }

    fn open_clip_tower(&mut self) -> Result<(), RealForwardError> {
        if self.arch.vision.tower != model_io::VisionTowerKind::Clip
            || !self.arch.vision.is_active()
        {
            return Err(RealForwardError::Unsupported(
                "this install declares no CLIP vision tower".to_string(),
            ));
        }
        if self.clip_vision.is_none() {
            self.clip_vision = Some(match &self.vision_sidecar_dir {
                Some(dir) => crate::vision::clip::ClipVisionTower::open_with_sidecar(
                    dir,
                    &self.context,
                    &self.arch,
                )?,
                None => crate::vision::clip::ClipVisionTower::open(
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

    /// Hand this runner one prompt's CLIP image rows: one `[span.len,
    /// text_hidden]` FP16 row set per placeholder run, in prompt order. The
    /// next prefill writes them in place of the placeholders' embeddings.
    ///
    /// Positions are the plain token index (LLaVA has no mRoPE), so the
    /// trunk's rotary stays on its ordinary kernel; the injection is the
    /// embedding blit alone and attention over the image run is causal like
    /// any other text. The same lifetime and prefix-taint rules as
    /// [`Self::set_prompt_vision`].
    pub fn set_prompt_vision_flat(
        &mut self,
        images: &[Vec<u16>],
        spans: &[turbospark_vision_io::ImageSpan],
        prompt_len: usize,
    ) -> Result<(), RealForwardError> {
        if self.arch.vision.tower != model_io::VisionTowerKind::Clip {
            return Err(RealForwardError::Unsupported(
                "set_prompt_vision_flat needs a CLIP vision tower".to_string(),
            ));
        }
        let hidden = self.arch.hidden_size as usize;
        self.prompt_vision = Some(crate::vision::PromptVision::flat(
            images, spans, prompt_len, hidden,
        )?);
        self.kv_prefix.taint();
        Ok(())
    }

    /// `(image_start, image_end)` for a tower whose chat template renders a
    /// BARE placeholder per image, or `None` when the template renders the
    /// markers itself.
    ///
    /// Muse-Glimmer's template emits only `<|patch|>`; its processor wraps the
    /// expanded run in `<|image_start|>` and `<|image_end|>`, so a front end
    /// has to do that wrap on the rendered ids before the splice
    /// (`turbospark_vision_io::wrap_bare_image_placeholders`).
    pub fn image_wrap_ids(&self) -> Option<(i32, i32)> {
        let v = &self.arch.vision;
        (v.is_active() && v.tower == model_io::VisionTowerKind::MuseGlimmer)
            .then_some((v.vision_start_token_id as i32, v.vision_end_token_id as i32))
    }

    /// The three ids a Gemma 4 image occupies in a prompt, read off the
    /// install, or `None` when its tower is not a Gemma 4 one.
    pub fn gemma4_image_ids(&self) -> Option<turbospark_vision_io::Gemma4ImageIds> {
        let v = &self.arch.vision;
        (v.is_active() && v.tower == model_io::VisionTowerKind::Gemma4).then_some(
            turbospark_vision_io::Gemma4ImageIds {
                boi: v.vision_start_token_id as i32,
                image: v.image_token_id as i32,
                eoi: v.vision_end_token_id as i32,
            },
        )
    }

    /// Encode one image through this install's Gemma 4 tower
    /// (`docs/GEMMA4_VISION_PHASE0.md`): `[soft_tokens, text_hidden]` FP16
    /// bits, the rows that replace the image placeholder's embeddings.
    ///
    /// [`Self::set_prompt_vision_gemma4`] injects the rows and
    /// `prefill_chunk` runs each image span with the sliding layers'
    /// bidirectional mask. Sidecars are refused until the sidecar path
    /// learns this tower's residents.
    pub fn encode_image_gemma4(
        &mut self,
        patches: &turbospark_vision_io::Gemma4Patches,
    ) -> Result<Vec<u16>, RealForwardError> {
        if self.arch.vision.tower != model_io::VisionTowerKind::Gemma4
            || !self.arch.vision.is_active()
        {
            return Err(RealForwardError::Unsupported(
                "this install declares no gemma4 vision tower".to_string(),
            ));
        }
        if self.vision_sidecar_dir.is_some() {
            return Err(RealForwardError::Unsupported(
                "a gemma4 vision tower cannot run from a sidecar yet".to_string(),
            ));
        }
        if self.gemma4_vision.is_none() {
            self.gemma4_vision = Some(crate::vision::gemma4::Gemma4VisionTower::open(
                &self.install_dir,
                &self.context,
                &self.weights,
                &self.index,
                &self.arch,
            )?);
        }
        let tower = self.gemma4_vision.as_mut().expect("opened just above");
        tower.run(&mut self.context, &self.weights, patches)
    }
}

/// Refuses the qwen tower's paths (`encode_image`, `set_prompt_vision`'s
/// tower) on an install whose tower is another kind. `crate::vision`'s
/// weights tables, kernels and injection are that tower's; running a Gemma 4
/// install through them would answer about an image the model never saw.
fn qwen_tower_only(vision: &model_io::VisionConfig) -> Result<(), RealForwardError> {
    match vision.tower {
        model_io::VisionTowerKind::QwenSiglip => Ok(()),
        model_io::VisionTowerKind::MuseGlimmer => Err(RealForwardError::Unsupported(
            "this install's vision tower is muse_glimmer; it opens through encode_image, \
             which routes to the Muse-Glimmer tower"
                .to_string(),
        )),
        model_io::VisionTowerKind::Qwen25Vl => Err(RealForwardError::Unsupported(
            "this install's vision tower is qwen25_vl; it opens through encode_image, which \
             routes to the Qwen2.5-VL tower"
                .to_string(),
        )),
        other => Err(RealForwardError::Unsupported(format!(
            "this install's vision tower is {}; encode its images with \
             encode_image_gemma4 / encode_image_clip and inject them with \
             set_prompt_vision_gemma4 / set_prompt_vision_flat",
            other.as_str()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_qwen_tower_path_refuses_a_gemma4_tower_by_name() {
        let mut v = model_io::VisionConfig::NONE;
        v.depth = 27;
        assert!(qwen_tower_only(&v).is_ok());
        v.tower = model_io::VisionTowerKind::Gemma4;
        let err = qwen_tower_only(&v).unwrap_err().to_string();
        assert!(err.contains("encode_image_gemma4"), "{err}");
    }
}
