use model_io::ModelFamily;

use crate::protocol::{PROTOCOL_MAX_CONTEXT, PROTOCOL_MAX_NEW};

/// The KV window and the generation budget the frozen protocol runs a given
/// family at. Both are PER-FAMILY parameters and neither is a knob: see
/// [`open_model_runner_with_context`] and [`run_protocol_case_with_budget`]
/// for why each one had to stop being a shared constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolParameters {
    /// The family these came from, so a caller can NAME it in a header
    /// without peeking `manifest.json` a second time. Both numbers below are
    /// meaningless without it.
    pub family: ModelFamily,
    /// KV window the runner is opened at, and the limit the generation loop
    /// enforces. MUST be the same number in both places.
    pub max_context: u32,
    /// New-token budget per case.
    pub max_new: u32,
}

/// The dense `llama` window (`mistral_memory_oracle.rs`'s). `long-synthesis`
/// is 3,444 tokens under Mistral's 32k vocab and `3444 + PROTOCOL_MAX_NEW`
/// does not fit 4,096, so the case does not run at all at the shared window.
const DENSE_LLAMA_MAX_CONTEXT: u32 = 8192;
/// The `gpt-oss` window (`gptoss_memory_oracle.rs`'s). Follows from the
/// budget below: `long-synthesis` is 2,839 tokens under o200k and
/// `2839 + 3072` does not fit 4,096.
const GPTOSS_MAX_CONTEXT: u32 = 8192;
/// The `gpt-oss` budget (`gptoss_memory_oracle.rs`'s). Harmony puts the
/// model's reasoning in an `analysis` channel BEFORE its answer, so the three
/// cases need 818 / 2,153 / 1,108 sampled tokens to reach `<|return|>`.
const GPTOSS_MAX_NEW: u32 = 3072;
/// The `muse_glimmer` window (`museglimmer_memory_oracle.rs`'s). Follows from
/// the budget below: `long-synthesis` is 2,820 tokens once the template's
/// system preamble is counted, and `2820 + 2048` does not fit 4,096.
const MUSE_GLIMMER_MAX_CONTEXT: u32 = 8192;
/// The `muse_glimmer` budget. The model reasons to a `to=self` message before
/// its `to=user` answer, so the three cases need 1,054 / 1,378 / 1,246
/// sampled tokens to reach `<|eot|>` -- the SHORT one already exceeds the
/// shared 1,024.
const MUSE_GLIMMER_MAX_NEW: u32 = 2048;

/// Resolve the protocol's two per-family parameters from the install's own
/// declared family.
///
/// **The match is exhaustive with NO wildcard arm, and that is the guard
/// rather than a style choice.** A `_ =>` here would let a seventh family
/// silently inherit Gemma's window and budget, which is the shape of bug
/// AGENTS.md Gotchas 24, 37 and 39 are all instances of: a default is a
/// claim about what silence means, and a table keyed by X holding a property
/// of Y stays invisible for exactly as long as the mapping is injective.
/// Adding a family should not compile until someone has answered this.
///
/// The two moved rows are the oracles' own numbers, asserted equal by
/// `mistral_memory_oracle.rs` and `gptoss_memory_oracle.rs` so the binary and
/// the oracles cannot drift apart. Read a peak or a tok/s row WITH these two
/// numbers; a row taken at one window says nothing about another (crate
/// Gotchas 11 and 12).
///
/// `const` so the two oracle targets can assert agreement in a `const`
/// block, which fails the BUILD rather than only firing on the rare
/// occasions those `#[ignore]`d targets run with an install present.
pub const fn protocol_parameters(family: ModelFamily) -> ProtocolParameters {
    match family {
        // The shared protocol, and what every frozen row in
        // `docs/BENCHMARKS.md` was measured at.
        // `qwen3_5` is here on the TOKENIZER's evidence rather than on a
        // measurement, and this comment is the answer the doc above asks
        // for. The window is decided by how many tokens the frozen prose
        // becomes, which belongs to the checkpoint's tokenizer -- and this
        // one declares `vocab_size: 248320`, Qwen 3.6's exactly, under the
        // same ChatML dialect. So the protocol's ~2.8k-token long case fits
        // 4,096 for the same reason it does there, and NOT for the reason
        // it fails on the dense `llama` half below (Mistral's 32k
        // sentencepiece vocab makes the same prose 3,444 tokens).
        // UNVERIFIED until an install exists: the first memory-oracle run
        // is what confirms it, and a `long-synthesis` that stops on
        // maxTokens is what would refute it.
        ModelFamily::Gemma4
        | ModelFamily::QwenGdnMoe
        | ModelFamily::QwenGdnDense
        | ModelFamily::Qwen3Moe
        | ModelFamily::DeepseekV4Flash => ProtocolParameters {
            family,
            max_context: PROTOCOL_MAX_CONTEXT,
            max_new: PROTOCOL_MAX_NEW,
        },
        // ONE FAMILY, BOTH HALVES OF THE ARCHITECTURE STRING. The window is
        // the dense half's requirement, measured on Mistral-7B-Instruct-v0.3
        // (ROADMAP M4). Mixtral shares that checkpoint's 32k sentencepiece
        // tokenizer, so the same window is right for it by the same
        // arithmetic -- not measured there, and it will not be: Mixtral
        // cannot stream on this engine at any useful slot count and runs the
        // protocol at 0.16 tok/s (AGENTS.md Gotcha 36).
        ModelFamily::Llama => ProtocolParameters {
            family,
            max_context: DENSE_LLAMA_MAX_CONTEXT,
            max_new: PROTOCOL_MAX_NEW,
        },
        // THE SECOND FAMILY THAT MOVES BOTH, and it moves them for gpt-oss's
        // reason: it REASONS BEFORE ANSWERING. `muse_glimmer`'s system
        // preamble carries `Reasoning strength: high.` and the model emits a
        // `to=self` message before its `to=user` one, so the shared 1,024
        // budget truncates the SHORT case, never mind the long one.
        //
        // **AN EARLIER DRAFT OF THIS FILE PUT THIS FAMILY IN THE SHARED
        // GROUP, and the mistake is worth keeping written down.** The prompt
        // side was measured properly -- the three frozen prompts encode to
        // 46 / 408 / 2,764 tokens under this checkpoint's own
        // `tokenizer.json`, so `2764 + 1024` fits 4,096 -- and the OUTPUT
        // side was assumed, in a comment that said "the model emits no
        // reasoning channel, so the shared 1,024 budget stands". Measuring
        // the prompt and assuming the completion is exactly the shape of
        // AGENTS.md Gotchas 37 to 39: half the question answered from the
        // file and half from expectation.
        //
        // Measured end to end at 8,192 / 3,072, greedy, on the real install
        // (2026-08-15), all three stopping endOfTurn:
        //   short-explanation   102 prompt + 1,054 new = 1,156
        //   medium-review       464 prompt + 1,378 new = 1,842
        //   long-synthesis    2,820 prompt + 1,246 new = 4,066
        // Note the prompts are LONGER than the raw prose tokenizes to (102
        // against 46) because the rendered template adds ~56 tokens of
        // system preamble, which is the other half of what a raw
        // tokenization misses.
        //
        // 2,048 is 1.49x the largest observed completion, the same margin
        // gpt-oss's 3,072 takes over its 2,153. The WINDOW has to be 8,192
        // because `2,820 + 2,048` does not fit 4,096 -- and note the largest
        // total observed (4,066) fits 4,096 with 30 tokens to spare, which
        // is exactly the kind of margin that turns a frozen row red on a
        // checkpoint revision.
        ModelFamily::MuseGlimmer => ProtocolParameters {
            family,
            max_context: MUSE_GLIMMER_MAX_CONTEXT,
            max_new: MUSE_GLIMMER_MAX_NEW,
        },
        // The only family that moves BOTH.
        ModelFamily::GptOss => ProtocolParameters {
            family,
            max_context: GPTOSS_MAX_CONTEXT,
            max_new: GPTOSS_MAX_NEW,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The families whose published rows were measured at the shared
    /// protocol. If one of these ever moves, every number in
    /// `docs/BENCHMARKS.md` for that family is at a different workload.
    #[test]
    fn four_families_run_the_shared_protocol_parameters() {
        for family in [
            ModelFamily::Gemma4,
            ModelFamily::QwenGdnMoe,
            ModelFamily::Qwen3Moe,
            ModelFamily::DeepseekV4Flash,
        ] {
            let params = protocol_parameters(family);
            assert_eq!(
                params.max_context,
                PROTOCOL_MAX_CONTEXT,
                "{} is measured at the shared window; moving it invalidates its frozen rows",
                family.as_str()
            );
            assert_eq!(
                params.max_new,
                PROTOCOL_MAX_NEW,
                "{} is measured at the shared budget; moving it invalidates its frozen rows",
                family.as_str()
            );
        }
    }

    /// The dense `llama` window, which `mistral_memory_oracle.rs` freezes a
    /// 1,300 MiB ceiling against.
    #[test]
    fn the_llama_family_runs_at_the_dense_window() {
        let params = protocol_parameters(ModelFamily::Llama);
        assert_eq!(params.max_context, 8192, "mistral_memory_oracle.rs's row");
        assert_eq!(params.max_new, PROTOCOL_MAX_NEW);
        assert!(
            params.max_context > PROTOCOL_MAX_CONTEXT,
            "the point of the row is that `long-synthesis` does not fit 4,096 \
             under a 32k-vocab tokenizer"
        );
    }

    /// The one family that moves BOTH parameters, which
    /// `gptoss_memory_oracle.rs` freezes a 5,700 MiB ceiling against.
    #[test]
    fn the_gptoss_family_moves_both_parameters() {
        let params = protocol_parameters(ModelFamily::GptOss);
        assert_eq!(params.max_context, 8192, "gptoss_memory_oracle.rs's row");
        assert_eq!(params.max_new, 3072, "gptoss_memory_oracle.rs's row");
        assert!(
            params.max_new > PROTOCOL_MAX_NEW,
            "Harmony's reasoning channel needs 2,153 sampled tokens on \
             `medium-review`; at the shared budget two of three cases stop on \
             maxTokens"
        );
    }
}
