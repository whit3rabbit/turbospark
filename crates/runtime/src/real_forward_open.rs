use std::path::Path;

use model_io::{ArchConfig, ExpertCacheSlots, ResidentBuffer};

use crate::real_forward::{RealForwardRunner, MAX_PREFILL_CHUNK_TOKENS, PACKED_LAYOUT_MAX_BYTES};
use crate::real_forward_layout::{
    moe_offsets_from_layout, readable_resident_dtype, routed_layouts_from_layout,
    EXECUTABLE_GGUF_DTYPES, GGUF_BLOCK_DTYPES,
};
use crate::real_forward_types::{DecodeScratch, PhaseCounters, RealForwardError, ROUTED_BANKS};

impl RealForwardRunner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open_inner(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        expert_cache_slots: ExpertCacheSlots,
        fp16_ring_capacity_override: Option<usize>,
        speculation: crate::families::qwen::DraftPolicies,
        steering: crate::steering::SteeringPolicy,
        session_slots: usize,
    ) -> Result<Self, RealForwardError> {
        if expert_cache_slots == ExpertCacheSlots::Fixed(0) {
            return Err(RealForwardError::Unsupported(
                "expert_cache_slots must be positive".to_string(),
            ));
        }
        crate::real_forward_init::validate_arch_config(&expecting)?;

        model_io::load_manifest(dir, &expecting, model_io::DEFAULT_MAX_BYTES)
            .map_err(RealForwardError::Model)?;
        let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
            .map_err(RealForwardError::Model)?;

        if let Some(entry) = index.entries.values().find(|e| {
            GGUF_BLOCK_DTYPES.contains(&e.dtype) && !EXECUTABLE_GGUF_DTYPES.contains(&e.dtype)
        }) {
            // The message names the RESIDENT question, not a global one: a
            // type can have routed kernels and no GEMV (MXFP4 does), so
            // "has no kernel in this port" was about to become false while
            // the refusal stayed correct.
            return Err(RealForwardError::Unsupported(format!(
                "tensor {} carries GGUF block dtype {}, which has no RESIDENT kernel in this \
                 port (ROADMAP Phase G Stage 2; resident-executable tags: {:?})",
                entry.name, entry.dtype, EXECUTABLE_GGUF_DTYPES
            )));
        }
        // The same question for every OTHER tag, and the one that catches the
        // quiet half. The check above only looks at tags the writer calls
        // GGUF blocks; an unquantized tensor written FP16 (tag 2) or FP32
        // (tag 3) passes it and is then decoded as BF16 off its byte size,
        // because that is what every unquantized reader here does. FP16 is
        // the same width, so nothing fails and the values are wrong by up to
        // 2^112 -- fluent garbage from an install that opened cleanly.
        //
        // It takes the NAME as well as the tag since ROADMAP M-V3: the vision
        // tower is FP16 by design and its tensors are read by nothing on the
        // text path, so tag 2 is honoured under `vision.` and refused
        // everywhere else. `readable_resident_dtype`'s own doc has the
        // argument for why the exception is scoped rather than granted.
        if let Some(entry) = index
            .entries
            .values()
            .find(|e| !readable_resident_dtype(&e.name, e.dtype))
        {
            return Err(RealForwardError::Unsupported(format!(
                "tensor {} carries resident dtype {}, which no reader in this crate honours; \
                 unquantized tensors must be narrowed to BF16 (tag 1) at repack time",
                entry.name, entry.dtype
            )));
        }

        let buffer = ResidentBuffer::map(
            &dir.join("model_weights.bin"),
            index.header.index_size,
            index.header.resident_size,
        )
        .map_err(RealForwardError::Model)?;
        let mut context = gpu::MetalContext::new().map_err(RealForwardError::Gpu)?;
        let weights = gpu::ResidentGpuWeights::wrap(context.device(), buffer)
            .map_err(RealForwardError::Gpu)?;

        let kv = gpu::KvCacheManager::new(
            context.device(),
            &expecting,
            max_context,
            true,
            None,
            MAX_PREFILL_CHUNK_TOKENS,
            fp16_ring_capacity_override,
        )
        .map_err(RealForwardError::Gpu)?;
        let scratch = DecodeScratch::new(&context, &expecting);

        let (streamers, slot_buffers, experts_layout, resolved_slots, mapped) =
            crate::real_forward_init::open_expert_streamers(
                dir,
                &expecting,
                expert_cache_slots,
                index.header.resident_size,
                PACKED_LAYOUT_MAX_BYTES,
                &mut context,
            )?;

        let use_silu = expecting.hidden_activation.contains("silu");
        let (moe_offsets, routed_layouts, routed_blobs, routed_blobs_banks) = match &experts_layout
        {
            Some(layout) => {
                let offsets = moe_offsets_from_layout(layout)?;
                let layouts = routed_layouts_from_layout(layout)?;
                let routed = gpu::RoutedBlobsBuffer::new(&mut context, use_silu)
                    .map_err(RealForwardError::Gpu)?;
                let mut banks = Vec::with_capacity(ROUTED_BANKS - 1);
                for _ in 1..ROUTED_BANKS {
                    banks.push(
                        gpu::RoutedBlobsBuffer::new(&mut context, use_silu)
                            .map_err(RealForwardError::Gpu)?,
                    );
                }
                (offsets, layouts, Some(routed), banks)
            }
            None => (Vec::new(), Vec::new(), None, Vec::new()),
        };
        drop(experts_layout);

        let router_hist = crate::router_hist::RouterHistogram::from_env(
            expecting.num_layers as usize,
            expecting.num_experts.max(0) as usize,
        );
        let ffn_hist = crate::ffn_hist::FfnActHist::from_env(&context, &expecting);
        let resid_capture = crate::resid_capture::ResidCapture::from_env(&context, &expecting);
        let mut runner = Self {
            kv_prefix: crate::kv_prefix::KvPrefix::default(),
            prefix_reuse_enabled: false,
            // Built for real AFTER the family-state block below, once it is
            // known whether this install carries GDN recurrent state: a
            // parked slot needs its own `GdnStateManager` only on that
            // family, and building it here would mean re-deriving that
            // question a second time.
            session_pool: crate::session_pool::SessionPool::empty(),
            session_slot_evicted: false,
            context,
            weights,
            index,
            arch: expecting,
            kv,
            scratch,
            slot_buffers,
            expert_cache_slots: resolved_slots,
            streamers,
            mapped,
            moe_offsets,
            routed_blobs,
            routed_blobs_banks,
            real: None,
            real_qwen: None,
            real_mtp: None,
            real_dflash: None,
            real_llama: None,
            real_gpt_oss: None,
            real_muse: None,
            real_qwen4: None,
            phases: PhaseCounters::default(),
            shared_cb_overlap: std::env::var("MFERENCE_SHARED_CB").as_deref() != Ok("0"),
            routed_pipeline: std::env::var("MFERENCE_ROUTED_PIPELINE").as_deref() != Ok("0"),
            routed_batch_prefill: std::env::var("MFERENCE_ROUTED_BATCH").as_deref() == Ok("1"),
            batched_gemv_prefill: std::env::var("MFERENCE_BATCHED_GEMV").as_deref() == Ok("1"),
            routed_layouts,
            router_hist,
            ffn_hist,
            resid_capture,
            steering: None,
            install_dir: dir.to_path_buf(),
            vision: None,
            prompt_vision: None,
            skip_head: false,
        };
        match runner.arch.family {
            model_io::ModelFamily::Gemma4 => {
                if runner
                    .index
                    .entries
                    .contains_key("language_model.model.embed_tokens.weight")
                {
                    runner.real = Some(crate::families::gemma4::RealGemmaState::build(
                        &mut runner.context,
                        &runner.weights,
                        &runner.index,
                        &runner.arch,
                    )?);
                }
            }
            // One flow for both, on the same footing `llama` and `qwen3moe`
            // share `families/llama/`'s: every BEHAVIOURAL field of
            // `qwen_gdn_dense_27b()` equals `qwen_gdn_moe_35b_a3b()`'s and every SHAPE field
            // differs, so `qwen3_5` is the DENSE half of this flow and not a
            // sixth one. `RealQwenState` carries the split, off `num_experts`.
            model_io::ModelFamily::QwenGdnMoe | model_io::ModelFamily::QwenGdnDense => {
                runner.real_qwen = Some(crate::families::qwen::RealQwenState::build(
                    &mut runner.context,
                    &runner.weights,
                    &runner.index,
                    &runner.arch,
                )?);
                // The speculative drafter, and it builds NOTHING unless a
                // depth was asked for -- so an install that has a head is
                // byte-identical and footprint-identical to one that does
                // not until someone turns drafting on
                // (`docs/MTP_SPECULATIVE.md`, step 2).
                // The GDN shape comes from the state built immediately above:
                // step 4's batched scratch needs the recurrent widths, and
                // deriving them a second time here would be a second place
                // for them to be wrong.
                let gdn_shape = runner
                    .real_qwen
                    .as_ref()
                    .expect("real Qwen state built above")
                    .shape;
                // AT MOST ONE DRAFTER IS EVER OPEN, and this is where that
                // is decided rather than assumed. A round drafts with one
                // model: `drafts_block_passes` and the priming/rewind pair
                // answer for the DFlash2 drafter while `produce_batched`
                // takes the MTP head's scratch, so a runner holding both
                // verifies a block of 9 against a scratch sized 3 and dies
                // mid-generation. Two explicit asks is a CALLER error and
                // is named as one; one explicit ask beats a bare `Auto`,
                // which is what the walk's ability to ingest both drafters
                // into a single install makes reachable with no env var
                // set at all (`open_with_slot_policy`, hence the server
                // and the FFI).
                use crate::families::qwen::{DflashDraftPolicy, MtpDraftPolicy};
                let (mtp_policy, dflash_policy) = match (speculation.mtp, speculation.dflash) {
                    (MtpDraftPolicy::Fixed(_), DflashDraftPolicy::Fixed(_)) => {
                        return Err(RealForwardError::Unsupported(
                            "both the multi-token-prediction head and the DFlash2 drafter \
                                 were asked for by name; a speculative round drafts with ONE \
                                 model, so name exactly one (MFERENCE_MTP_DRAFT or \
                                 MFERENCE_DFLASH_DRAFT; --speculative-drafter on the CLI)"
                                .to_string(),
                        ))
                    }
                    (mtp @ MtpDraftPolicy::Fixed(_), _) => (mtp, DflashDraftPolicy::Off),
                    (mtp, dflash) => (mtp, dflash),
                };
                // The SECOND drafter, same doctrine: builds nothing unless
                // asked for at open, so an install that carries one is
                // byte- and footprint-identical to one that does not until
                // a caller turns it on (`docs/DFLASH2.md`). Built FIRST so
                // the head can yield to it when neither was named.
                runner.real_dflash = crate::families::qwen::DflashState::build(
                    &mut runner.context,
                    &runner.index,
                    &runner.arch,
                    max_context,
                    dflash_policy,
                    gdn_shape,
                )?;
                let mtp_policy = if runner.real_dflash.is_some() {
                    MtpDraftPolicy::Off
                } else {
                    mtp_policy
                };
                runner.real_mtp = crate::families::qwen::MtpState::build(
                    &mut runner.context,
                    &runner.index,
                    &runner.arch,
                    max_context,
                    mtp_policy,
                    gdn_shape,
                )?;
            }
            // One flow for both: `qwen3moe` is the same layer graph, and
            // `RealLlamaState` carries the two differences (per-head q/k
            // norms, a different RMS epsilon).
            model_io::ModelFamily::Llama | model_io::ModelFamily::Qwen3Moe => {
                runner.real_llama = Some(crate::families::llama::RealLlamaState::build(
                    &mut runner.context,
                    &runner.weights,
                    &runner.index,
                    &runner.arch,
                )?);
            }
            model_io::ModelFamily::DeepseekV4Flash => {
                return Err(RealForwardError::Unsupported(
                    "the DeepSeek-V4-Flash family has no decode flow yet".to_string(),
                ));
            }
            // Refused BY NAME even though `validate_arch_config` already
            // stops it a few lines up (its mask-2 layers are not on the
            // qwen36/qwen35 allowlist), because that check's message names
            // the layer kind and this one names the FAMILY. When the flow
            // lands, exactly one of the two has to change and a reader
            // should not have to guess which.
            //
            // Its residual stream is `hc_count * hidden_size` wide and its
            // residual adds are gated-residual injects, so no line of
            // `families/qwen/` serves it unchanged. See `ModelFamily::Qwen4Exp`.
            model_io::ModelFamily::Qwen4Exp => {
                runner.real_qwen4 = Some(crate::families::qwen4::RealQwen4State::build(
                    &mut runner.context,
                    &runner.weights,
                    &runner.index,
                    &runner.arch,
                    dir,
                    max_context,
                )?);
            }
            // A FIFTH FLOW, not a sixth family on an existing one: all four
            // of `gpt-oss`'s differences (per-projection biases, attention
            // sinks, YaRN rope scaling, a clamped SwiGLU) are INSIDE the
            // layer, and each produces fluent wrong output rather than an
            // error if a neighbour's flow is used instead.
            model_io::ModelFamily::GptOss => {
                runner.real_gpt_oss = Some(crate::families::gptoss::RealGptOssState::build(
                    &mut runner.context,
                    &runner.weights,
                    &runner.index,
                    &runner.arch,
                )?);
            }
            // A SIXTH FLOW, on the same reasoning `gpt-oss` got the fifth:
            // ten differences, every one inside the layer. See
            // `ModelFamily::MuseGlimmer` for the list.
            model_io::ModelFamily::MuseGlimmer => {
                runner.real_muse = Some(crate::families::museglimmer::RealMuseState::build(
                    &mut runner.context,
                    &runner.index,
                    &runner.arch,
                )?);
            }
        }
        // Pre-allocated once, here rather than lazily on first use: a slot
        // that failed to allocate mid-conversation would be a Metal
        // allocation error with nothing pointing back at `--session-slots`,
        // exactly the failure `docs/LOAD_GUARD.md`'s refusal-before-open
        // discipline exists to avoid (the caller is expected to have
        // already refused an over-committed slot count via
        // `model_io::context_policy::session_pool_bytes`). `session_slots <= 1`
        // allocates nothing at all, which is the whole byte-identity
        // guarantee for the default case.
        let parked_slots = session_slots.saturating_sub(1);
        let mut pool_slots = Vec::with_capacity(parked_slots);
        for _ in 0..parked_slots {
            let kv = gpu::KvCacheManager::new(
                runner.context.device(),
                &runner.arch,
                max_context,
                true,
                None,
                MAX_PREFILL_CHUNK_TOKENS,
                fp16_ring_capacity_override,
            )
            .map_err(RealForwardError::Gpu)?;
            let gdn = runner
                .real_qwen
                .is_some()
                .then(|| gpu::GdnStateManager::new(runner.context.device(), &runner.arch));
            pool_slots.push(crate::session_pool::SessionSlot {
                kv,
                gdn,
                kv_prefix: crate::kv_prefix::KvPrefix::default(),
                last_used: 0,
            });
        }
        runner.session_pool = crate::session_pool::SessionPool::new(pool_slots);
        // LAST, after the family state, because it validates against the
        // resolved `ArchConfig` and because a steering failure should be the
        // last thing an otherwise-good open reports rather than masking one.
        // `SteeringPolicy::off()` returns before touching anything, so this
        // allocates and encodes nothing on every existing caller.
        runner.steering =
            crate::steering::SteeringState::build(&runner.context, &runner.arch, &steering)?;
        if runner.steering.is_some()
            && !crate::steering::family_dispatches_steering(runner.arch.family)
        {
            // Refused BY NAME rather than ignored. A family whose flow does
            // not dispatch the edit would load a direction set, report it on
            // the startup line, and change nothing -- the caller would
            // measure the unsteered engine and report it as the steered one
            // (`MtpState::build`'s argument for an explicitly-requested
            // drafter).
            return Err(RealForwardError::Unsupported(format!(
                "steering is not wired for family {:?}: its flow does not dispatch the edit, \
                 so a direction set here would be a silent no-op. Wired today: the qwen flow \
                 (both halves), the llama flow (Mixtral, Qwen3-MoE, and the dense Llama / \
                 Mistral half), Gemma 4 (sequential decode and chunked prefill, both \
                 batched-routed and per-token), gpt-oss, and museGlimmer. Unwired: \
                 DeepSeek-V4-Flash (no decode flow exists to hook)",
                runner.arch.family
            )));
        }
        // STEERING AND SPECULATION USED TO BE MUTUALLY EXCLUSIVE HERE, and
        // the refusal was not conservatism: the speculative verify runs
        // `produce_batched`, which had no steering hook, so the
        // drafted-and-verified tokens -- the ones COMMITTED -- came from the
        // UNSTEERED model while the sequential fallback tokens came from the
        // steered one. A silent mixture of two models, coherent and wrong.
        //
        // `families/qwen/batched.rs` carries the edit now, at the same
        // boundary and through the same `encode_steering` the per-token path
        // calls, so the two cannot disagree about what the edit IS. What
        // makes lifting the refusal safe is measured rather than argued:
        // `the_batched_forward_steers_every_row_exactly_as_m_produce_calls_do`
        // requires an M-row steered verify to be BIT-IDENTICAL to M steered
        // `produce` calls, which is the same standard the unsteered batched
        // path is already held to.
        //
        // WHAT IS NOT FIXED BY THIS, and is a quality question rather than a
        // correctness one: the DRAFTER is not steered. A direction set covers
        // trunk layers, and neither the MTP head nor the DFlash2 drafter is a
        // trunk layer, so a steered run drafts from the unsteered model and
        // verifies against the steered one. Speculation stays LOSSLESS -- a
        // verify rejects what it does not agree with -- so the cost lands on
        // ACCEPTANCE alone, which is exactly the shape that reads as a verdict
        // about the drafter rather than as a consequence of the edit. It is
        // measured in `docs/OBLITERATION.md` rather than left to be guessed.
        Ok(runner)
    }
}
