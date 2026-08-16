use super::types::{ManifestQuant, ManifestQuantSlot};
use crate::error::ModelError;

/// Affine-quant group size (matches `turbospark_compute::quant::GROUP_SIZE`;
/// duplicated here rather than adding a compute dependency to this crate).
const QUANT_GROUP_SIZE: i64 = 64;

/// Affine-quant group size at ONE bit (matches
/// `turbospark_compute::quant_1bit::BONSAI_GROUP_SIZE`, duplicated for the
/// reason above).
///
/// A separate constant rather than a widening of [`QUANT_GROUP_SIZE`],
/// because the two are not alternatives a caller picks between: 64 is what
/// every INT4/INT8 GEMV kernel is compiled against, and 128 is what the one
/// published 1-bit checkpoint declares. See [`validate_quant`] for why the
/// bit width, the group size and the companion dtype are checked as one
/// conjunction rather than three independent axes.
const QUANT_1BIT_GROUP_SIZE: i64 = 128;

/// Affine-quant group size at TWO bits (matches
/// `turbospark_compute::quant_2bit::TERNARY_GROUP_SIZE`, duplicated for the
/// reason above).
///
/// Equal to [`QUANT_1BIT_GROUP_SIZE`] and a separate constant for the same
/// reason that one is separate from [`QUANT_GROUP_SIZE`]: the two happen to
/// agree because one publisher chose 128 for both of its checkpoints, not
/// because sub-4-bit implies 128.
const QUANT_2BIT_GROUP_SIZE: i64 = 128;

/// The GGUF block types this port can EXECUTE, as they are spelled in a
/// manifest's `ggmlType` (ROADMAP Phase G Stage 2).
///
/// A GGUF-sourced install is written for every block type the parser knows,
/// which is deliberately more than the set with kernels behind it: the repack
/// walk's job is to carry bytes, and refusing to install would lose the
/// artifact. This list is what decides whether one can be OPENED, and it
/// grows only when a kernel plus its parity test land. `crates/runtime`
/// applies the same rule again to the resident index's dtype tags, which is
/// the backstop for a hand-edited manifest.
/// Q6_K is here on weaker grounds than the other two and the difference is
/// worth knowing: it has a resident GEMV and nothing else, because the only
/// real file that uses it puts it in `output.weight`. An install that carried
/// Q6_K experts would pass this gate and fail at the routed dispatch instead,
/// which is a worse error message but not a wrong answer.
/// The three IQ types (ROADMAP Phase S) join on the same terms as the rest,
/// and one of them is narrower than it looks: IQ3_XXS and IQ4_XS have a
/// routed phase-1 kernel, IQ4_NL a routed phase-2 one, and all three a
/// resident GEMV, but there is no IQ4_NL phase 1 and no IQ3_XXS phase 2
/// because no real file asks for either. That is the same weaker footing
/// Q6_K stands on, and it fails the same way: at the dispatch site, by name.
///
/// MXFP4 (ROADMAP M5, `gpt-oss`) joins on the narrowest footing yet, and it
/// is narrow in a NEW DIRECTION: it has both routed phases and NO resident
/// GEMV, where Q6_K and Q5_K have a resident GEMV and (almost) no routed
/// kernels. That is the real file's shape rather than a choice -- the one
/// checkpoint carrying MXFP4 puts it in `ffn_{gate,up,down}_exps` and keeps
/// attention, `token_embd` and `output` at Q8_0. This list is what the
/// manifest's per-slot `ggmlType` is read against, so MXFP4 belongs in it;
/// `RealForwardRunner`'s `EXECUTABLE_GGUF_DTYPES`, which reads RESIDENT
/// tensors, deliberately omits it, and its doc explains why the two are twins
/// rather than copies.
pub const EXECUTABLE_GGUF_TYPES: [&str; 8] = [
    "q8_0", "q4_k", "q5_k", "q6_k", "iq3_xxs", "iq4_nl", "iq4_xs", "mxfp4",
];

/// Accepts a quant block iff every slot's shape has kernels behind it.
///
/// Four shapes are accepted; the third came with ROADMAP's 1-bit entry and
/// the fourth with its ternary one.
///
/// 1. **INT4/INT8 affine**: BF16 companions at group 64, per-slot bit widths.
/// 2. **GGUF**: every declared block type in [`EXECUTABLE_GGUF_TYPES`].
/// 3. **1-bit affine**: FP16 companions at group 128.
/// 4. **2-bit affine**: FP16 companions at group 128.
///
/// **EACH SUB-4-BIT SHAPE IS CHECKED AS ONE CONJUNCTION, NOT AS WIDENINGS OF
/// THE FIRST, and that is the point of writing them as separate predicates.**
/// It would have been shorter to add `1` and `2` to the affine bit lists,
/// `128` to the group sizes and `fp16` to the companion types, and the result
/// would accept a dozen combinations that no kernel implements -- 1-bit at
/// group 64, 4-bit with FP16 companions, and so on. The real shapes are
/// `(4|8, bf16, 64)`, `(1, fp16, 128)` and `(2, fp16, 128)`, because a
/// checkpoint's bit width, companion dtype and group size travel together,
/// and the FP16-versus-BF16 axis is the dangerous one: the two planes are the
/// same width, so a wrong reading passes every length check and decodes these
/// checkpoints' 0.027 and 0.0137 scales as ~1e-16.
///
/// Note the fourth shape does NOT subsume the `weight_bits == 2` the affine
/// arm already allows on `routedExpert`: that one is BF16 at group 64, for
/// the DeepSeek-V4-Flash dynamic-quant checkpoint, and the two 2-bit shapes
/// share nothing but their width.
///
/// **Both sub-4-bit shapes are accepted on ALL FIVE SLOTS, including
/// `routedExpert`, though neither has a routed-expert kernel.** That is not
/// an oversight and it is not a claim that such an MoE install would run.
/// `manifest.quant` has five fixed slots and no architecture fills all five;
/// both published sub-4-bit checkpoints are DENSE, so their router,
/// shared-expert and routed-expert probes find nothing and fall back to the
/// type the rest of the model uses
/// (`crates/repack` Gotcha 8: refusing a slot for a component the install
/// does not have is how a runnable model fails to open). An install that
/// really did carry 1-bit routed experts passes here and fails at the routed
/// dispatch, by name -- the same weaker footing Q6_K and the IQ types stand
/// on, stated in [`EXECUTABLE_GGUF_TYPES`]'s doc.
pub(crate) fn validate_quant(quant: &ManifestQuant) -> Result<(), ModelError> {
    let slots: [(&str, &ManifestQuantSlot, &[i64]); 5] = [
        ("embedding", &quant.embedding, &[4]),
        ("attention", &quant.attention, &[4]),
        ("router", &quant.router, &[8]),
        ("sharedExpert", &quant.shared_expert, &[4, 8]),
        // Routed experts additionally allow 2-bit: the DeepSeek-V4-Flash
        // dynamic-quant checkpoint ships Q2 experts under a Q4 core.
        ("routedExpert", &quant.routed_expert, &[2, 4]),
    ];
    // A slot that is byte-for-byte the ATTENTION slot is a DEFAULTED
    // statement about a component the model does not have, not a claim about
    // bytes. The repack walks write it that way on purpose
    // (`crates/repack`'s `manifest_quant_for` and Gotcha 8's `or_attention`):
    // `manifest.quant` has five fixed slots and no architecture fills all
    // five, so refusing one is how a runnable dense install fails to open
    // with a message about something it never had -- which cost M4 one
    // five-minute re-stream per slot.
    //
    // The residual is worth naming: this admits an MoE install whose ROUTER
    // genuinely is 4-bit and happens to match its attention slot, which the
    // INT8-only `router_gemv_gemma4_r4` would misread. Nothing writes one --
    // Gemma and Qwen both override the router to 8 bits, and a checkpoint
    // that did not would be a new family's problem -- but it is admitted
    // here rather than refused, unlike the bit lists below.
    let attention = &quant.attention;
    for (name, slot, allowed_bits) in slots {
        if !std::ptr::eq(slot, attention) && slot == attention {
            continue;
        }
        let affine = allowed_bits.contains(&slot.weight_bits)
            && slot.scheme.to_lowercase() == "affine"
            && slot.scale_type.to_lowercase() == "bf16"
            && slot.bias_type.to_lowercase() == "bf16"
            && slot.group_size == QUANT_GROUP_SIZE;
        // The 1-bit shape, whole. See this function's doc for why the three
        // fields are one conjunction and why every slot accepts it.
        let affine_1bit = slot.weight_bits == 1
            && slot.scheme.to_lowercase() == "affine"
            && slot.scale_type.to_lowercase() == "fp16"
            && slot.bias_type.to_lowercase() == "fp16"
            && slot.group_size == QUANT_1BIT_GROUP_SIZE;
        // The 2-bit shape (ROADMAP's ternary entry), a FOURTH conjunction and
        // not a widening of the third: it is a separate `(bits, companions,
        // group)` triple that happens to share two of its three fields with
        // the 1-bit one. Note `routedExpert` already admits `weight_bits == 2`
        // through the affine arm above, at BF16 and group 64 -- a different
        // shape entirely, for the DeepSeek-V4 dynamic-quant checkpoint -- so
        // the two must not be collapsed into one bit list.
        let affine_2bit = slot.weight_bits == 2
            && slot.scheme.to_lowercase() == "affine"
            && slot.scale_type.to_lowercase() == "fp16"
            && slot.bias_type.to_lowercase() == "fp16"
            && slot.group_size == QUANT_2BIT_GROUP_SIZE;
        // A GGUF slot carries no bits, no group size and no companion types:
        // the scale lives inside each block. What it does carry is the block
        // type -- possibly SEVERAL, since ROADMAP Phase S -- and that is the
        // whole question: a Q4_K install and a Q8_0 one are equally
        // well-formed here and only one of them has kernels.
        //
        // Every declared type must be executable, not just the dominant one.
        // Checking only `ggmlType` would let a mixed install through on the
        // strength of its majority and fail at a dispatch thirty layers in.
        let declared = slot.declared_types();
        let gguf = slot.scheme.to_lowercase() == "gguf"
            && !declared.is_empty()
            && declared
                .iter()
                .all(|t| EXECUTABLE_GGUF_TYPES.contains(&t.to_lowercase().as_str()));
        if !(affine || affine_1bit || affine_2bit || gguf) {
            let detail = match slot.scheme.to_lowercase().as_str() {
                // An affine slot has four fields that can each be wrong and a
                // bare "unsupported" names none of them. It matters most on
                // the companion dtype, where the failure is otherwise
                // invisible: FP16 and BF16 are the same width, so nothing
                // downstream notices, and this message is the only place the
                // mismatch is ever spelled out.
                "affine" => format!(
                    "unsupported quantization for {name}: affine slot is \
                     {}-bit with {}/{} companions at group {}, and the shapes \
                     with kernels are {allowed_bits:?}-bit bf16 at group \
                     {QUANT_GROUP_SIZE}, 1-bit fp16 at group \
                     {QUANT_1BIT_GROUP_SIZE} and 2-bit fp16 at group \
                     {QUANT_2BIT_GROUP_SIZE}",
                    slot.weight_bits, slot.scale_type, slot.bias_type, slot.group_size
                ),
                "gguf" => {
                    let offending: Vec<&str> = declared
                        .iter()
                        .copied()
                        .filter(|t| !EXECUTABLE_GGUF_TYPES.contains(&t.to_lowercase().as_str()))
                        .collect();
                    let named = if offending.is_empty() {
                        "unspecified".to_string()
                    } else {
                        offending.join(", ")
                    };
                    format!(
                        "unsupported quantization for {name}: GGUF block type {named} has no \
                        kernel in this port (executable types: {})",
                        EXECUTABLE_GGUF_TYPES.join(", ")
                    )
                }
                _ => format!("unsupported quantization for {name}"),
            };
            return Err(ModelError::IndexCorrupt { detail });
        }
    }
    Ok(())
}
