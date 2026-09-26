//! Quantization dtypes, GGUF block type tags, and MoE expert blob layout
//! resolution for real forward execution.

use crate::real_forward_types::RealForwardError;

/// Every GGUF block dtype tag the resident index can carry. Mirrors
/// `turbospark_repack::resident_writer`'s list, which is the writer-side home;
/// this crate must not depend on repack, so the two are held equal by
/// `crates/runtime/tests/gguf_install_refused.rs` exercising a real written
/// install rather than by an import.
///
/// IT HAD FALLEN ONE BEHIND THE WRITER AND THAT WEAKENED THE BACKSTOP: Q5_K
/// (13) landed in `resident_writer` in ROADMAP M2 and never arrived here, so
/// a resident tensor tagged 13 was not a "GGUF block dtype" as far as
/// `RealForwardRunner::open` was concerned. It did no harm while Q5_K was
/// executable -- the guard only fires on a type in this list and NOT in
/// [`EXECUTABLE_GGUF_DTYPES`] -- but the whole point of the guard is to
/// believe the bytes over the manifest, and a type missing here is one it
/// cannot see. Add to both lists when the writer gains a tag. It is spelled
/// out of the NAMED constants below rather than as bare literals for the same
/// reason: a literal list is what let one go missing.
pub(crate) const GGUF_BLOCK_DTYPES: [u8; 18] = [
    DTYPE_GGUF_Q8_0,
    DTYPE_GGUF_Q4_K,
    DTYPE_GGUF_Q6_K,
    DTYPE_GGUF_Q4_0,
    DTYPE_GGUF_IQ3_XXS,
    DTYPE_GGUF_IQ4_NL,
    DTYPE_GGUF_IQ4_XS,
    DTYPE_GGUF_Q5_K,
    DTYPE_GGUF_MXFP4,
    DTYPE_GGUF_Q2_K,
    DTYPE_GGUF_IQ2_XXS,
    DTYPE_GGUF_IQ2_XS,
    DTYPE_GGUF_IQ1_S,
    DTYPE_GGUF_IQ3_S,
    DTYPE_GGUF_IQ2_S,
    DTYPE_GGUF_IQ1_M,
    DTYPE_GGUF_Q3_K,
    DTYPE_GGUF_Q2_0,
];
/// GGUF Q8_0: a resident GEMV, an embedding lookup, and a routed-expert
/// decode pair.
pub(crate) const DTYPE_GGUF_Q8_0: u8 = 6;
/// GGUF Q4_K: the same three, landed for Qwen 3.6's Q4_K_M.
pub(crate) const DTYPE_GGUF_Q4_K: u8 = 7;
/// GGUF Q6_K: a resident GEMV and nothing else, which is all any real file
/// asks for -- Qwen's Q4_K_M carries exactly one Q6_K tensor and it is
/// `output.weight`. An install that put Q6_K in an expert or the embedding
/// table would pass this gate and then fail at the dispatch site, by name.
pub(crate) const DTYPE_GGUF_Q6_K: u8 = 8;
/// GGUF Q4_0: parsed and installable, and the ONLY tag here with no kernel of
/// any kind. It is what `gguf_install_refused.rs` forges an install to, so it
/// belongs in [`GGUF_BLOCK_DTYPES`] and never in [`EXECUTABLE_GGUF_DTYPES`].
pub(crate) const DTYPE_GGUF_Q4_0: u8 = 9;
/// GGUF IQ3_XXS, IQ4_NL and IQ4_XS (ROADMAP Phase S). Each has a resident
/// GEMV; on the routed path IQ3_XXS and IQ4_XS have a phase 1 and IQ4_NL a
/// phase 2. IQ4_XS also has an embedding lookup for the Swift Qwen3.8 tier.
pub(crate) const DTYPE_GGUF_IQ3_XXS: u8 = 10;
pub(crate) const DTYPE_GGUF_IQ4_NL: u8 = 11;
pub(crate) const DTYPE_GGUF_IQ4_XS: u8 = 12;
/// GGUF Q5_K: a resident GEMV and nothing else, which is all Mixtral 8x7B's
/// Q4_K_M asks for -- it carries Q5_K on `attn_output` and its experts are
/// Q4_K over Q6_K (ROADMAP Phase M2).
pub(crate) const DTYPE_GGUF_Q5_K: u8 = 13;
/// GGUF MXFP4 (ROADMAP M5, `gpt-oss`). The NARROWEST footing of any type
/// here and the first with a routed pair but no resident GEMV: the only real
/// file carrying it puts MXFP4 in `ffn_{gate,up,down}_exps` and keeps
/// attention, `token_embd` and `output` at Q8_0. A tensor tagged MXFP4 in the
/// RESIDENT index is therefore something no walk has ever written, and it
/// fails at `encode_gemv_any` by name. The tag exists so this list and the
/// writer's stay structurally parallel.
pub(crate) const DTYPE_GGUF_MXFP4: u8 = 14;
pub(crate) const DTYPE_GGUF_Q2_K: u8 = 17;
pub(crate) const DTYPE_GGUF_IQ2_XXS: u8 = 18;
pub(crate) const DTYPE_GGUF_IQ2_XS: u8 = 19;
pub(crate) const DTYPE_GGUF_IQ1_S: u8 = 20;
pub(crate) const DTYPE_GGUF_IQ3_S: u8 = 21;
pub(crate) const DTYPE_GGUF_IQ2_S: u8 = 22;
pub(crate) const DTYPE_GGUF_IQ1_M: u8 = 23;
/// GGUF Q3_K (Dense Qwen2 roadmap item). A resident GEMV and an embedding
/// lookup, no routed pair, on the Q6_K footing: the pinned Qwen2.5 Q3_K_M
/// keeps every attention and FFN projection here, its embedding table at
/// Q4_K and its head at Q6_K. Tag 24 mirrors the writer, and for the same
/// displaced reason -- ggml's own Q3_K id is 11, which IQ4_NL took first.
pub(crate) const DTYPE_GGUF_Q3_K: u8 = 24;
/// GGUF Q2_0, used on all routed matrices by the experimental Swift Q2_0
/// tier and on routed down rows by its mixed-precision IQ2_XS tier.
/// The resident path has no reader, so this tag belongs in the all-block
/// guard above but not in [`EXECUTABLE_GGUF_DTYPES`].
pub(crate) const DTYPE_GGUF_Q2_0: u8 = 25;
/// The RAW (companion-less, unquantized) tag this port can read, and the only
/// one: every consumer of an unquantized resident tensor -- `norm_view`,
/// `read_bf16_host`, every kernel binding a `device const bfloat*` --
/// identifies it by BYTE SIZE and decodes it as BF16.
///
/// The writer also defines FP16 (2) and FP32 (3) tags, and for every TEXT
/// tensor nothing here reads either, which is why [`readable_resident_dtype`]
/// refuses them rather than letting them through to be misread. An F16 norm is
/// the dangerous case: it is the same width as BF16, so every length check
/// passes and the values come out wrong by up to 2^112. The repack side
/// narrows instead (`turbospark_repack`'s `narrow_raw_to_bf16`); this is the
/// backstop for a hand-made install, in the same relationship the GGUF dtype
/// gate has to `model_io::validate_quant`.
pub(crate) const DTYPE_RAW_BF16: u8 = 1;

/// FP16, readable for the VISION TOWER'S resident tensors alone (ROADMAP
/// M-V3). See [`readable_resident_dtype`] for why the exception is scoped by
/// NAME rather than granted outright.
pub(crate) const DTYPE_RAW_FP16: u8 = 2;

/// The prefix the tower's resident tensors carry, and the whole basis of the
/// FP16 exception. Written by `turbospark_repack`'s vision ingest; the two
/// spellings have to agree, and `the_vision_prefix_matches_the_writers`
/// asserts it rather than leaving it to a comment.
pub(crate) const VISION_PREFIX: &str = "vision.";

/// Whether a resident entry's dtype tag has a reader in this crate.
///
/// Listed rather than defaulted, for `encode_gemv_any`'s catch-all's reason:
/// a tag added to the writer and not here is refused at open with its number
/// in the message, where a permissive default would dispatch it as something
/// else.
///
/// **IT TAKES THE NAME BECAUSE THE ANSWER IS NOT A PROPERTY OF THE TAG ALONE**
/// (ROADMAP M-V3). The `qwen3_5` vision tower is FP16 end to end -- its Metal
/// kernels bind `half` where every other kernel in `crates/gpu` binds
/// `bfloat` -- so its resident tensors genuinely are tag 2 and narrowing them
/// to BF16 would cost three mantissa bits on the merger's two large matrices
/// to store a precision no consumer wants. But the hazard the blanket refusal
/// closes is real and unchanged for text: `norm_view` and `read_bf16_host` are
/// dtype-BLIND, resolving an unquantized tensor by byte width, so a tag-2
/// tensor either of them reaches is misread rather than rejected (AGENTS.md
/// Gotcha 45).
///
/// Scoping by prefix is what keeps both halves. Nothing under `vision.` is
/// reachable from any text-path helper -- the tower is read by
/// `crates/runtime/src/vision/` and by nothing else -- so the exception cannot
/// widen into the case it was written to prevent. Granting tag 2 outright
/// would reopen it for every family at once, and for a tensor whose reader is
/// dtype-blind that is not a narrower bug than the one being fixed, it is the
/// same one.
pub(crate) fn readable_resident_dtype(name: &str, dtype: u8) -> bool {
    if dtype == DTYPE_RAW_FP16 {
        return name.starts_with(VISION_PREFIX);
    }
    matches!(
        dtype,
        DTYPE_RAW_BF16 | 4 | 5 | DTYPE_INT1_AFFINE | DTYPE_INT2_AFFINE
    ) || EXECUTABLE_GGUF_DTYPES.contains(&dtype)
}

/// 1-BIT AFFINE (ROADMAP's 1-bit entry), mirroring
/// `turbospark_repack::DTYPE_INT1_AFFINE`. Its number is 15 rather than
/// something beside the 4 and 5 of its affine siblings only because 6..=14
/// were taken by the GGUF tags first.
///
/// **It is NOT a GGUF block type and must never join [`GGUF_BLOCK_DTYPES`] or
/// [`EXECUTABLE_GGUF_DTYPES`].** Those two lists are about self-contained
/// blocks carrying their scale inline; a 1-bit affine tensor has the same
/// three planar regions an INT4 one has, and the guard at `open()` tests
/// membership of the first list before anything else, so a tag listed there
/// by accident would be refused as an unrunnable GGUF type.
pub(crate) const DTYPE_INT1_AFFINE: u8 = 15;

/// 2-BIT AFFINE (ROADMAP's ternary entry), mirroring
/// `turbospark_repack::DTYPE_INT2_AFFINE`. 16 for the tag above's reason: it
/// is the next free number after the GGUF block tags.
///
/// **It is NOT a GGUF block type either**, and the same warning applies. Note
/// what distinguishes it from the 1-bit tag at a dispatch: nothing but the
/// tag. All four affine widths share an entry shape of three planar regions,
/// so a 2-bit tensor read as 1-bit is a row of half the columns -- finite,
/// ordered, and wrong.
pub(crate) const DTYPE_INT2_AFFINE: u8 = 16;

/// The executable subset of [`GGUF_BLOCK_DTYPES`], and the resident-index
/// twin of `model_io::EXECUTABLE_GGUF_TYPES`. Grows only when a kernel plus
/// its parity test land, and the two lists have to move together or
/// `crates/runtime/tests/gguf_install_refused.rs` reddens.
///
/// "TWIN" IS NOT "COPY", AND MXFP4 IS THE FIRST TYPE TO SHOW THE DIFFERENCE.
/// The two lists answer different questions: `model_io`'s reads the
/// manifest's per-SLOT `ggmlType` and asks "does this type have the kernels
/// its slot needs", while this one reads a RESIDENT tensor's dtype tag and
/// asks "can this tensor be dispatched". MXFP4 is executable in the first
/// sense (it has a routed pair) and not in the second (it has no resident
/// GEMV at all, because the one real file carrying it puts it only in
/// `ffn_*_exps`). So `"mxfp4"` joins `EXECUTABLE_GGUF_TYPES` and 14 does NOT
/// join this list, and a hypothetical install with MXFP4 attention passes the
/// manifest gate and is stopped here -- which is the layering working, not a
/// leak.
pub(crate) const EXECUTABLE_GGUF_DTYPES: [u8; 15] = [
    DTYPE_GGUF_Q8_0,
    DTYPE_GGUF_Q4_K,
    DTYPE_GGUF_Q6_K,
    DTYPE_GGUF_IQ3_XXS,
    DTYPE_GGUF_IQ4_NL,
    DTYPE_GGUF_IQ4_XS,
    DTYPE_GGUF_Q5_K,
    DTYPE_GGUF_Q2_K,
    DTYPE_GGUF_IQ2_XXS,
    DTYPE_GGUF_IQ2_XS,
    DTYPE_GGUF_IQ1_S,
    DTYPE_GGUF_IQ3_S,
    DTYPE_GGUF_IQ2_S,
    DTYPE_GGUF_IQ1_M,
    DTYPE_GGUF_Q3_K,
];

/// Which layout one routed sub-tensor uses.
///
/// Forwarders in `real_forward_dispatch.rs` turn this into the right kernel; no
/// call site branches on it directly, so two of them cannot disagree and read
/// one blob two ways.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RoutedBlobLayout {
    /// The INT4-affine planes the vendored `moe.metal` pair reads.
    Affine,
    /// GGUF Q8_0 blocks (Gemma 4's published GGUF is Q8_0 throughout).
    GgufQ8_0,
    /// GGUF Q4_K superblocks (Qwen 3.6's Q4_K_M puts its experts here).
    GgufQ4K,
    /// GGUF IQ3_XXS, the codebook type carrying 29 of the Phase S
    /// candidate's `ffn_gate_up_exps`. Phase 1 only.
    GgufIq3Xxs,
    /// GGUF IQ4_XS, on the candidate's layer 29 `ffn_gate_up_exps`. Phase 1
    /// only.
    GgufIq4Xs,
    /// GGUF IQ4_NL, carrying 29 of the candidate's `ffn_down_exps`. Phase 2
    /// only.
    GgufIq4Nl,
    /// GGUF MXFP4 (ROADMAP M5), which `gpt-oss` puts on ALL THREE routed
    /// sub-tensors. BOTH phases, unlike the four partial rows around it, and
    /// no resident GEMV, unlike every one of them -- that file keeps its
    /// attention, embedding and head at Q8_0.
    GgufMxfp4,
    /// GGUF Q6_K, carrying the `ffn_down_exps` of 16 of Mixtral 8x7B's 32
    /// layers while the other 16 are Q4_K (ROADMAP Phase M2). Phase 2 only,
    /// and the first type whose two layouts differ across LAYERS of one model
    /// rather than across the phases of one expert.
    GgufQ6K,
    /// GGUF Q2_0 routed rows; phase 2 has an exact top-10 reducer.
    GgufQ2_0,
    /// GGUF IQ2_S routed gate/up rows.
    GgufIq2S,
    /// GGUF IQ2_XXS routed gate/up rows.
    GgufIq2Xxs,
    /// GGUF IQ1_M routed gate/up rows.
    GgufIq1M,
}

impl RoutedBlobLayout {
    /// The GGUF layout a `layout.json` sub-tensor dtype names.
    ///
    /// Only reached for a blob already known to be block-quantized, so an
    /// unrecognized name here is an error rather than a fallback to affine.
    /// That split matters: the affine writers spell their packed run `"U32"`
    /// (it is packed into u32 words), not `"int4"`, so a dtype allowlist for
    /// the affine side would have to track a spelling nothing else depends
    /// on. See the caller for the discriminator that is actually used.
    pub(crate) fn from_gguf_dtype(dtype: &str) -> Result<Self, RealForwardError> {
        Ok(match dtype {
            "q8_0" => RoutedBlobLayout::GgufQ8_0,
            "q4_k" => RoutedBlobLayout::GgufQ4K,
            "iq3_xxs" => RoutedBlobLayout::GgufIq3Xxs,
            "iq4_xs" => RoutedBlobLayout::GgufIq4Xs,
            "iq4_nl" => RoutedBlobLayout::GgufIq4Nl,
            "q6_k" => RoutedBlobLayout::GgufQ6K,
            "q2_0" => RoutedBlobLayout::GgufQ2_0,
            "iq2_s" => RoutedBlobLayout::GgufIq2S,
            "iq2_xxs" => RoutedBlobLayout::GgufIq2Xxs,
            "iq1_m" => RoutedBlobLayout::GgufIq1M,
            "mxfp4" => RoutedBlobLayout::GgufMxfp4,
            other => {
                return Err(RealForwardError::Unsupported(format!(
                    "routed expert sub-tensor dtype {other} has no decode kernel in this port"
                )))
            }
        })
    }

    /// The library and a representative function name whose `RoutedBlobs`
    /// argument-buffer layout `RoutedBlobsBuffer` should reflect for this
    /// layout's group (S7). `Affine` is the vendored pair's own library;
    /// every GGUF variant compiles from `moe_gguf`'s ONE concatenated
    /// library (`crates/gpu/CLAUDE.md` Gotcha 4), so any one of its entry
    /// points reflects the same struct layout -- which kernel actually runs
    /// is still resolved per layer by `encode_moe_phase1_any`, this only
    /// picks which reflection the shared argument buffer is built from.
    pub(crate) fn source_function(self) -> (&'static str, &'static str) {
        match self {
            RoutedBlobLayout::Affine => {
                (gpu::moe_decode_source(), "moe_phase1_gate_up_act_u16load")
            }
            RoutedBlobLayout::GgufQ8_0
            | RoutedBlobLayout::GgufQ4K
            | RoutedBlobLayout::GgufIq3Xxs
            | RoutedBlobLayout::GgufIq4Xs
            | RoutedBlobLayout::GgufIq4Nl
            | RoutedBlobLayout::GgufMxfp4
            | RoutedBlobLayout::GgufQ6K => (gpu::moe_gguf_source(), "moe_phase1_gate_up_act_q8_0"),
            RoutedBlobLayout::GgufIq2S
            | RoutedBlobLayout::GgufIq2Xxs
            | RoutedBlobLayout::GgufIq1M => (gpu::moe_gguf_source(), "moe_phase1_gate_up_act_q8_0"),
            RoutedBlobLayout::GgufQ2_0 => (gpu::moe_gguf_source(), "moe_phase1_gate_up_act_q2_0"),
        }
    }
}

/// Which layout each PHASE of one layer's routed experts uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct RoutedLayerLayout {
    /// Read by phase 1, from the `gate` and `up` runs.
    pub(crate) phase1: RoutedBlobLayout,
    /// Read by phase 2, from the `down` run.
    pub(crate) phase2: RoutedBlobLayout,
}

/// Resolves each layer's phase-1 and phase-2 layout from `layout.json`'s
/// per-sub-tensor dtypes (ROADMAP Phase S).
///
/// Off the LAYOUT rather than the manifest, which is where this used to come
/// from, because the manifest has one `ggmlType` for the whole routed slot and
/// a mixed install has no single answer to give it. `gate` and `up` must agree
/// -- one phase-1 kernel reads both, so a blob where they differed could not
/// be dispatched at all -- and that is checked rather than assumed.
///
/// Affine and GGUF are told apart by the presence of the SCALE COMPANIONS,
/// not by the dtype string, which is the same discriminator
/// [`moe_offsets_from_layout`] uses for the same blobs a few lines below. An
/// affine blob has nine sub-tensors and a GGUF blob has three, so the test
/// cannot drift; matching on dtype names instead would mean tracking how each
/// writer spells its packed run (the affine ones say `"U32"`, since that is
/// what the nibbles are packed into).
fn require_ascending_layers(
    layout: &model_io::PackedExpertsLayout,
) -> Result<(), RealForwardError> {
    for pair in layout.layers.windows(2) {
        if pair[0].layer >= pair[1].layer {
            return Err(RealForwardError::Unsupported(format!(
                "packed layout layers are not in strictly ascending order: layer {} followed by {}",
                pair[0].layer, pair[1].layer
            )));
        }
    }
    Ok(())
}

pub(crate) fn routed_layouts_from_layout(
    layout: &model_io::PackedExpertsLayout,
) -> Result<Vec<RoutedLayerLayout>, RealForwardError> {
    require_ascending_layers(layout)?;
    layout
        .layers
        .iter()
        .map(|l| {
            let subs = &l
                .experts
                .first()
                .ok_or_else(|| {
                    RealForwardError::Unsupported(format!("layer {} has no experts", l.layer))
                })?
                .sub_tensors;
            if subs.contains_key("gate_scales") {
                return Ok(RoutedLayerLayout {
                    phase1: RoutedBlobLayout::Affine,
                    phase2: RoutedBlobLayout::Affine,
                });
            }
            let of = |role: &str| -> Result<RoutedBlobLayout, RealForwardError> {
                let entry = subs.get(role).ok_or_else(|| {
                    RealForwardError::MissingTensor(format!("expert blob {role}"))
                })?;
                RoutedBlobLayout::from_gguf_dtype(&entry.dtype)
            };
            let (gate, up) = (of("gate")?, of("up")?);
            if gate != up {
                return Err(RealForwardError::Unsupported(format!(
                    "layer {} packs a {gate:?} gate against a {up:?} up; one phase-1 kernel \
                     reads both, so they cannot differ",
                    l.layer
                )));
            }
            Ok(RoutedLayerLayout {
                phase1: gate,
                phase2: of("down")?,
            })
        })
        .collect()
}

/// Resolves the shader's `ExpertOffsets` for every layer, from expert 0 of
/// each (the writer packs every expert of a layer identically).
///
/// Per layer since ROADMAP Phase S. It was one struct off layer 0, which held
/// while every blob in an install had the same shape; on a mixed install the
/// sub-tensor sizes differ per layer, so the offsets do too.
///
/// Two blob shapes reach this. An INT4-affine blob has all nine sub-tensors,
/// and its phase-2 down projection reads weight bytes with 4-byte loads, so
/// `down`'s offset must be 4-byte aligned. A GGUF blob has only the three
/// weight runs -- its scales live inside the blocks -- so the six companion
/// offsets resolve to zero for every type except MXFP4 and the GGUF kernels
/// never read them there. The GGUF kernels read the block bytes one at a
/// time, so no alignment applies to those.
///
/// **MXFP4 is the exception, and it is not covered by either clause above.**
/// `gpt-oss` writes real `gate_biases`/`up_biases`/`down_biases` sub-tensors
/// beside its three MXFP4 weight runs (the family carries per-expert
/// biases), and `moe_gguf.metal` reads each one through a `device const
/// float*` cast (Gotcha 29's per-type rule: MXFP4 rows are ODD-length, byte
/// for byte, so nothing about the blob layout guarantees 4-byte alignment
/// the way the affine writer's own layout does). The writer packs
/// sub-tensors back to back with no padding, so a bias offset is 4-byte
/// aligned only when the preceding tensor's own byte count happens to be a
/// multiple of 4 -- true on `gpt-oss-20b` (2,880 rows) and not guaranteed on
/// a differently-shaped MXFP4 checkpoint.
pub(crate) fn moe_offsets_from_layout(
    layout: &model_io::PackedExpertsLayout,
) -> Result<Vec<gpu::MoeExpertOffsets>, RealForwardError> {
    require_ascending_layers(layout)?;
    layout
        .layers
        .iter()
        .map(|l| {
            let entry = &l
                .experts
                .first()
                .ok_or_else(|| {
                    RealForwardError::Unsupported(format!("layer {} has no experts", l.layer))
                })?
                .sub_tensors;
            let get = |name: &str| -> Result<u32, RealForwardError> {
                let sub = entry.get(name).ok_or_else(|| {
                    RealForwardError::MissingTensor(format!("expert blob {name}"))
                })?;
                u32::try_from(sub.offset).map_err(|_| {
                    RealForwardError::Unsupported(format!(
                        "layer {} expert blob {name} offset {} exceeds 32-bit address space",
                        l.layer, sub.offset
                    ))
                })
            };
            // Absent companions mean a block-quantized blob, not a broken one.
            let companion = |name: &str| -> Result<u32, RealForwardError> {
                let Some(s) = entry.get(name) else {
                    return Ok(0);
                };
                u32::try_from(s.offset).map_err(|_| {
                    RealForwardError::Unsupported(format!(
                        "layer {} expert blob {name} offset {} exceeds 32-bit address space",
                        l.layer, s.offset
                    ))
                })
            };
            let planar = entry.contains_key("gate_scales");
            let offsets = gpu::MoeExpertOffsets {
                gate_w: get("gate")?,
                gate_s: companion("gate_scales")?,
                gate_b: companion("gate_biases")?,
                up_w: get("up")?,
                up_s: companion("up_scales")?,
                up_b: companion("up_biases")?,
                down_w: get("down")?,
                down_s: companion("down_scales")?,
                down_b: companion("down_biases")?,
            };
            if planar && offsets.down_w % 4 != 0 {
                return Err(RealForwardError::Unsupported(format!(
                    "layer {} down projection offset {} is not 4-byte aligned",
                    l.layer, offsets.down_w
                )));
            }
            if !planar {
                for (role, offset) in [
                    ("gate", offsets.gate_b),
                    ("up", offsets.up_b),
                    ("down", offsets.down_b),
                ] {
                    if offset != 0 && offset % 4 != 0 {
                        return Err(RealForwardError::Unsupported(format!(
                            "layer {} {role} bias offset {offset} is not 4-byte aligned \
                             (MXFP4's `device const float*` bias read requires it)",
                            l.layer
                        )));
                    }
                }
            }
            Ok(offsets)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_io::{ExpertEntry, LayerLayout, PackedExpertsLayout, SubTensorEntry};
    use std::collections::BTreeMap;

    fn make_test_layout(
        layer_indices: &[usize],
        gate_offset: u64,
    ) -> model_io::PackedExpertsLayout {
        let layers = layer_indices
            .iter()
            .map(|&layer| {
                let mut sub_tensors = BTreeMap::new();
                sub_tensors.insert(
                    "gate".to_string(),
                    model_io::SubTensorEntry {
                        offset: gate_offset,
                        size: 64,
                        dtype: "Q4_0".to_string(),
                    },
                );
                sub_tensors.insert(
                    "up".to_string(),
                    model_io::SubTensorEntry {
                        offset: 64,
                        size: 64,
                        dtype: "Q4_0".to_string(),
                    },
                );
                sub_tensors.insert(
                    "down".to_string(),
                    model_io::SubTensorEntry {
                        offset: 128,
                        size: 64,
                        dtype: "Q4_0".to_string(),
                    },
                );
                model_io::LayerLayout {
                    layer,
                    file: format!("layer_{layer}.bin"),
                    expert_stride: 256,
                    experts: vec![model_io::ExpertEntry {
                        expert: 0,
                        offset: 0,
                        size: 256,
                        sub_tensors,
                    }],
                }
            })
            .collect();
        model_io::PackedExpertsLayout {
            expert_stride: 256,
            num_layers: layer_indices.len(),
            experts_per_layer: 1,
            layers,
        }
    }

    fn sub(offset: u64) -> SubTensorEntry {
        SubTensorEntry {
            offset,
            size: 4,
            dtype: "mxfp4".to_string(),
        }
    }

    fn routed_dtype_layout(gate_up: &str, down: &str) -> PackedExpertsLayout {
        let entry = |offset, dtype: &str| SubTensorEntry {
            offset,
            size: 64,
            dtype: dtype.to_string(),
        };
        let mut sub_tensors = BTreeMap::new();
        sub_tensors.insert("gate".to_string(), entry(0, gate_up));
        sub_tensors.insert("up".to_string(), entry(64, gate_up));
        sub_tensors.insert("down".to_string(), entry(128, down));
        PackedExpertsLayout {
            expert_stride: 192,
            num_layers: 1,
            experts_per_layer: 1,
            layers: vec![LayerLayout {
                layer: 0,
                file: "layer_00.bin".to_string(),
                expert_stride: 192,
                experts: vec![ExpertEntry {
                    expert: 0,
                    offset: 0,
                    size: 192,
                    sub_tensors,
                }],
            }],
        }
    }

    #[test]
    fn qwen4_exp_routed_gguf_dtypes_resolve_by_phase() {
        for (gate_up, expected) in [
            ("iq2_s", RoutedBlobLayout::GgufIq2S),
            ("iq2_xxs", RoutedBlobLayout::GgufIq2Xxs),
            ("iq1_m", RoutedBlobLayout::GgufIq1M),
        ] {
            let layout = routed_dtype_layout(gate_up, "q2_0");
            let routed = routed_layouts_from_layout(&layout).expect("routed types resolve");
            assert_eq!(routed[0].phase1, expected, "gate/up dtype {gate_up}");
            assert_eq!(routed[0].phase2, RoutedBlobLayout::GgufQ2_0);
        }
        let q2_0 = routed_dtype_layout("q2_0", "q2_0");
        let routed = routed_layouts_from_layout(&q2_0).expect("Q2_0 routes both phases");
        assert_eq!(routed[0].phase1, RoutedBlobLayout::GgufQ2_0);
        assert_eq!(routed[0].phase2, RoutedBlobLayout::GgufQ2_0);
    }

    /// A GGUF-shaped (non-planar) layer: three weight runs plus, as MXFP4
    /// does, three bias runs. `gate_biases` sits at offset 2 mod 4, which
    /// AGENTS.md/CLAUDE.md B2 says the writer can produce for any MXFP4
    /// checkpoint whose preceding tensor's byte count is not itself a
    /// multiple of 4 (real `gpt-oss-20b` happens to be 4-aligned by luck of
    /// its row count; this fixture is deliberately not).
    fn misaligned_mxfp4_layout() -> PackedExpertsLayout {
        let mut subs = BTreeMap::new();
        subs.insert("gate".to_string(), sub(0));
        subs.insert("up".to_string(), sub(100));
        subs.insert("down".to_string(), sub(200));
        subs.insert("gate_biases".to_string(), sub(302));
        subs.insert("up_biases".to_string(), sub(320));
        subs.insert("down_biases".to_string(), sub(340));
        PackedExpertsLayout {
            expert_stride: 400,
            num_layers: 1,
            experts_per_layer: 1,
            layers: vec![LayerLayout {
                layer: 0,
                file: "layer_00.bin".to_string(),
                expert_stride: 400,
                experts: vec![ExpertEntry {
                    expert: 0,
                    offset: 0,
                    size: 400,
                    sub_tensors: subs,
                }],
            }],
        }
    }

    #[test]
    fn out_of_order_layers_are_refused() {
        let layout = make_test_layout(&[1, 0], 0);
        let err = routed_layouts_from_layout(&layout).expect_err("out of order layers");
        match err {
            RealForwardError::Unsupported(msg) => {
                assert!(
                    msg.contains("not in strictly ascending order"),
                    "msg: {msg}"
                );
            }
            other => panic!("expected Unsupported error, got {other:?}"),
        }

        let err2 = moe_offsets_from_layout(&layout).expect_err("out of order layers");
        match err2 {
            RealForwardError::Unsupported(msg) => {
                assert!(
                    msg.contains("not in strictly ascending order"),
                    "msg: {msg}"
                );
            }
            other => panic!("expected Unsupported error, got {other:?}"),
        }
    }

    #[test]
    fn offset_overflowing_u32_is_refused() {
        let layout = make_test_layout(&[0], (u32::MAX as u64) + 1);
        let err = moe_offsets_from_layout(&layout).expect_err("overflowing offset");
        match err {
            RealForwardError::Unsupported(msg) => {
                assert!(msg.contains("exceeds 32-bit address space"), "msg: {msg}");
            }
            other => panic!("expected Unsupported error, got {other:?}"),
        }
    }

    #[test]
    fn a_misaligned_mxfp4_bias_offset_is_refused() {
        let layout = misaligned_mxfp4_layout();
        let err = moe_offsets_from_layout(&layout).expect_err("2-mod-4 bias offset must refuse");
        let msg = err.to_string();
        assert!(msg.contains("gate"), "{msg}");
        assert!(msg.contains("4-byte aligned"), "{msg}");
    }

    #[test]
    fn a_4_byte_aligned_mxfp4_bias_offset_is_accepted() {
        let mut layout = misaligned_mxfp4_layout();
        for name in ["gate_biases", "up_biases", "down_biases"] {
            layout.layers[0].experts[0]
                .sub_tensors
                .get_mut(name)
                .unwrap()
                .offset &= !3;
        }
        let offsets = moe_offsets_from_layout(&layout).expect("4-byte-aligned offsets are fine");
        assert_eq!(offsets.len(), 1);
    }

    /// A GGUF blob with NO bias planes at all (every GGUF type before MXFP4)
    /// must stay accepted: the zero-offset companion sentinel is not itself
    /// a misalignment.
    #[test]
    fn a_gguf_layer_with_no_bias_planes_is_unaffected() {
        let mut subs = BTreeMap::new();
        subs.insert("gate".to_string(), sub(0));
        subs.insert("up".to_string(), sub(100));
        subs.insert("down".to_string(), sub(200));
        let layout = PackedExpertsLayout {
            expert_stride: 300,
            num_layers: 1,
            experts_per_layer: 1,
            layers: vec![LayerLayout {
                layer: 0,
                file: "layer_00.bin".to_string(),
                expert_stride: 300,
                experts: vec![ExpertEntry {
                    expert: 0,
                    offset: 0,
                    size: 300,
                    sub_tensors: subs,
                }],
            }],
        };
        let offsets = moe_offsets_from_layout(&layout).expect("no bias planes is fine");
        assert_eq!(offsets[0].gate_b, 0);
        assert_eq!(offsets[0].up_b, 0);
        assert_eq!(offsets[0].down_b, 0);
    }
}
